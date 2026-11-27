import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import time
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from src.api.cache import (
    close_redis_client,
    ensure_semantic_cache_collection,
    get_redis_client,
    lookup_cache,
    store_cache,
)
from src.core.config import settings
from src.core.observability import flush_langfuse, get_langfuse_run_config
from src.core.schemas import QueryRequest, RAGResponse, TraceStep
from src.graph.workflow import rag_graph
from src.retrieval.hybrid_search import (
    close_qdrant_client,
    embed_query_async,
    get_dense_model,
    get_qdrant_client,
    get_sparse_model,
)
from src.retrieval.reranker import close_cohere_client

logger = logging.getLogger(__name__)
_CONCURRENCY_GUARD = asyncio.Semaphore(4)
_INFLIGHT_LOCKS: dict = dict()
_RATE_WINDOW_SECONDS = 60
_DAILY_KEY_TTL_SECONDS = 25 * 3600


def _inflight_lock(query: str, service_filter: str | None) -> asyncio.Lock:
    sig = hashlib.sha256(f"{service_filter or 'all'}::{query.lower()}".encode()).hexdigest()[:16]
    lock = _INFLIGHT_LOCKS.get(sig)
    if lock is None:
        if len(_INFLIGHT_LOCKS) > 512:
            _INFLIGHT_LOCKS.clear()
        lock = asyncio.Lock()
        _INFLIGHT_LOCKS[sig] = lock
    return lock


async def _try_acquire_slot() -> None:
    try:
        await asyncio.wait_for(_CONCURRENCY_GUARD.acquire(), timeout=0.001)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="Server busy; retry shortly.", headers={"Retry-After": "3"}) from exc


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        last = fwd.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


async def _bump_counter(key: str, ttl_seconds: int) -> int:
    pipe = get_redis_client().pipeline(transaction=True)
    pipe.set(key, 0, ex=ttl_seconds, nx=True)
    pipe.incr(key)
    results = await pipe.execute()
    return int(results[-1])


async def _enforce_rate_limit(request: Request) -> None:
    try:
        count = await _bump_counter(f"sysarch:rl:{_client_ip(request)}", _RATE_WINDOW_SECONDS)
    except Exception as exc:
        logger.warning("Rate-limiter skipped: %s", exc)
        return
    if count > settings.RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({settings.RATE_LIMIT_PER_MIN} queries/min); retry shortly.",
            headers={"Retry-After": "15"},
        )


async def _enforce_daily_cap() -> None:
    now = datetime.now(timezone.utc)
    try:
        count = await _bump_counter(f"sysarch:daily:{now:%Y%m%d}", _DAILY_KEY_TTL_SECONDS)
    except Exception as exc:
        logger.warning("Daily cap skipped: %s", exc)
        return
    if count > settings.DAILY_QUERY_CAP:
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        raise HTTPException(
            status_code=429,
            detail=f"Daily query budget ({settings.DAILY_QUERY_CAP}/day) reached; resets at 00:00 UTC.",
            headers={"Retry-After": str(max(1, int((midnight - now).total_seconds())))},
        )


async def _admit_pipeline() -> None:
    await _try_acquire_slot()
    try:
        await _enforce_daily_cap()
    except Exception:
        _CONCURRENCY_GUARD.release()
        raise


def _cors_origins() -> list:
    return list(o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip())


def _summarize_node(node_name: str, out: dict) -> str:
    if node_name == "analyze_query":
        svcs = ", ".join(out.get("services", list())) or "all"
        return f"Routed services: {svcs}"
    if node_name in ("retrieve", "retrieve_and_rerank"):
        n = len(out.get("reranked_chunks", list()))
        return f"Retrieved & reranked {n} chunks"
    if node_name in ("grade", "grade_documents"):
        g = out.get("grade_result")
        if g is not None:
            return f"Sufficient={g.is_sufficient} (conf={g.confidence:.2f})"
    if node_name in ("rewrite", "rewrite_query"):
        if out.get("rewrite_exhausted"):
            return "Rewrite budget exhausted"
        return f"Rewritten query (attempt {out.get('retry_count', 1)})"
    if node_name in ("synthesize", "synthesize_answer"):
        c = len(out.get("citations", list()))
        return f"Synthesized answer with {c} grounded citations"
    return "Completed"


def _response_from_state(state: dict, t0: float, request_id: str, trace: list | None = None) -> RAGResponse:
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return RAGResponse(
        request_id=request_id,
        answer=str(state.get("final_answer", "")),
        citations=state.get("citations", list()),
        services_searched=state.get("services", list()),
        retry_count=int(state.get("retry_count", 0)),
        refused=bool(state.get("refused", False)),
        refusal_reason=state.get("refusal_reason"),
        cache_hit=False,
        cache_tier="live_crag",
        latency_ms=elapsed,
        original_latency_ms=elapsed,
        trace=trace or list(),
    )


def _initial_state(req: QueryRequest) -> dict:
    init_state = {"query": req.query}
    if req.service_filter:
        init_state["services"] = list((req.service_filter,))
    return init_state


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    await asyncio.to_thread(lambda: (get_dense_model(), get_sparse_model()))
    await embed_query_async("warmup")
    await ensure_semantic_cache_collection()
    yield
    await close_cohere_client()
    await close_qdrant_client()
    await close_redis_client()
    flush_langfuse()


_is_prod = settings.ENV.lower() == "production"
app = FastAPI(
    title="SysArch-RAG API",
    description="Self-Correcting Cloud & Distributed Systems Intelligence Engine",
    version="0.1.0",
    docs_url=None if _is_prod else "/docs",
    redoc_url=None if _is_prod else "/redoc",
    openapi_url=None if _is_prod else "/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=list(("GET", "POST")),
    allow_headers=list(("*",)),
    expose_headers=list(("Retry-After",)),
)


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict:
    qdrant_ok = await get_qdrant_client().collection_exists(settings.QDRANT_COLLECTION)
    redis_ok = bool(await get_redis_client().ping())
    if not (qdrant_ok and redis_ok):
        raise HTTPException(status_code=503, detail={"qdrant": qdrant_ok, "redis": redis_ok})
    return {"status": "ready", "collection": settings.QDRANT_COLLECTION, "redis": True}


async def run_rag_pipeline(req: QueryRequest) -> RAGResponse:
    t0 = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    dense_vec, _ = await embed_query_async(req.query)

    async with _inflight_lock(req.query, req.service_filter):
        cached = await lookup_cache(req.query, req.service_filter, dense_vec=dense_vec)
        if cached is not None:
            cached.request_id = req_id
            cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return cached

        await _admit_pipeline()
        try:
            async with asyncio.timeout(45.0):
                run_cfg = get_langfuse_run_config(session_id=req.session_id, tags=list(("api-v1",)))
                run_cfg["recursion_limit"] = 15
                state = await rag_graph.ainvoke(_initial_state(req), config=run_cfg)
                resp = _response_from_state(state, t0, req_id)
                if resp.refusal_reason == "service_error":
                    raise HTTPException(status_code=503, detail=resp.answer)
                await store_cache(req.query, req.service_filter, dense_vec, resp)
                return resp
        finally:
            _CONCURRENCY_GUARD.release()


@app.post("/v1/query", response_model=RAGResponse)
async def query_endpoint(req: QueryRequest, request: Request) -> RAGResponse:
    await _enforce_rate_limit(request)
    try:
        return await run_rag_pipeline(req)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="RAG pipeline timed out.") from exc


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.post("/v1/query/stream")
async def query_stream_endpoint(req: QueryRequest, request: Request) -> StreamingResponse:
    await _enforce_rate_limit(request)
    t0 = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    dense_vec, _ = await embed_query_async(req.query)
    cached = await lookup_cache(req.query, req.service_filter, dense_vec=dense_vec)
    if cached is not None:
        cached.request_id = req_id
        cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        hit = cached

        async def cached_gen() -> AsyncGenerator[str, None]:
            yield _sse({"type": "start", "request_id": req_id, "query": req.query})
            yield _sse({"type": "cache", "hit": True, "tier": hit.cache_tier})
            yield _sse({"type": "trace", "node": "cache_lookup"})
            for step in hit.trace:
                yield _sse({"type": "node_end", "node": step.node, "duration_ms": step.duration_ms, "summary": step.summary})
            yield _sse({"type": "final", "payload": hit.model_dump()})

        return StreamingResponse(cached_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

    await _admit_pipeline()

    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"type": "start", "request_id": req_id, "query": req.query})
            yield _sse({"type": "cache", "hit": False, "tier": "live_crag"})
            yield _sse({"type": "trace", "node": "cache_lookup"})
            async with asyncio.timeout(45.0):
                run_cfg = get_langfuse_run_config(session_id=req.session_id, tags=list(("api-stream-v1",)))
                run_cfg["recursion_limit"] = 15

                final_state = dict()
                trace_steps = list()
                step_t0 = time.perf_counter()
                yield _sse({"type": "node_start", "node": "analyze_query"})

                async for update in rag_graph.astream(_initial_state(req), config=run_cfg, stream_mode="updates"):
                    for node_name, node_out in update.items():
                        dur_ms = round((time.perf_counter() - step_t0) * 1000, 2)
                        summary = _summarize_node(node_name, node_out)
                        step = TraceStep(node=node_name, duration_ms=dur_ms, summary=summary)
                        trace_steps.append(step)
                        final_state.update(node_out)
                        yield _sse({"type": "trace", "node": node_name})
                        yield _sse({"type": "node_end", "node": node_name, "duration_ms": dur_ms, "summary": summary})
                        step_t0 = time.perf_counter()

                resp = _response_from_state(final_state, t0, req_id, trace=trace_steps)
                if resp.refusal_reason == "service_error":
                    yield _sse({"type": "error", "detail": resp.answer})
                    return
                await store_cache(req.query, req.service_filter, dense_vec, resp)
                yield _sse({"type": "final", "payload": resp.model_dump()})
        except TimeoutError:
            yield _sse({"type": "error", "detail": "RAG pipeline timed out after 45s."})
        except Exception as exc:
            logger.exception("SSE stream error: %s", exc)
            yield _sse({"type": "error", "detail": "Internal streaming error."})
        finally:
            _CONCURRENCY_GUARD.release()

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS)
