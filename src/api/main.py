import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

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
_INFLIGHT_LOCKS: dict[str, asyncio.Lock] = {}
_RATE_WINDOW_SECONDS = 60
_DAILY_KEY_TTL_SECONDS = 25 * 3600
_FALLBACK_LOCK = asyncio.Lock()
_LOCAL_WINDOWS: dict[str, deque[float]] = {}
_LOCAL_DAILY: dict[str, int] = {}
_SESSION_ID_RE = re.compile(r"^[\w-]{1,64}$")


def _inflight_lock(
    query: str, service_filter: str | None, detail_level: str
) -> asyncio.Lock:
    sig = hashlib.sha256(
        f"{service_filter or 'all'}::{detail_level}::{query.lower()}".encode()
    ).hexdigest()[:16]
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
        raise HTTPException(
            status_code=429,
            detail="Server busy; retry shortly.",
            headers={"Retry-After": "3"},
        ) from exc


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        hops = [part.strip() for part in fwd.split(",") if part.strip()]
        if hops:
            idx = max(0, len(hops) - max(1, settings.TRUSTED_PROXY_HOPS))
            return hops[idx]
    return request.client.host if request.client else "unknown"


def _user_identity(request: Request, session_id: str | None = None) -> str | None:
    raw = (
        session_id
        or request.headers.get("x-session-id")
        or request.headers.get("x-user-id")
    )
    if raw and _SESSION_ID_RE.match(raw.strip()):
        return raw.strip()
    return None


async def _bump_counter(key: str, ttl_seconds: int) -> int:
    pipe = get_redis_client().pipeline(transaction=True)
    pipe.set(key, 0, ex=ttl_seconds, nx=True)
    pipe.incr(key)
    results = await pipe.execute()
    return int(results[-1])


async def _bump_local_window(bucket: str, window_seconds: int) -> int:
    now = time.monotonic()
    async with _FALLBACK_LOCK:
        if len(_LOCAL_WINDOWS) > 4096:
            _LOCAL_WINDOWS.clear()
        q = _LOCAL_WINDOWS.setdefault(bucket, deque())
        while q and (now - q[0]) >= window_seconds:
            q.popleft()
        q.append(now)
        return len(q)


async def _bump_local_daily(day_key: str) -> int:
    async with _FALLBACK_LOCK:
        for k in list(_LOCAL_DAILY.keys()):
            if k != day_key:
                _LOCAL_DAILY.pop(k, None)
        _LOCAL_DAILY[day_key] = _LOCAL_DAILY.get(day_key, 0) + 1
        return _LOCAL_DAILY[day_key]


async def _enforce_rate_limit(request: Request, session_id: str | None = None) -> None:
    ip = _client_ip(request)
    try:
        ip_count = await _bump_counter(f"sysarch:rl:ip:{ip}", _RATE_WINDOW_SECONDS)
    except Exception as exc:
        logger.warning(
            "Redis rate-limiter degraded; using local sliding window: %s", exc
        )
        ip_count = await _bump_local_window(f"ip:{ip}", _RATE_WINDOW_SECONDS)

    if ip_count > settings.RATE_LIMIT_PER_MIN:
        raise HTTPException(
            status_code=429,
            detail=f"IP rate limit exceeded ({settings.RATE_LIMIT_PER_MIN} queries/min); retry shortly.",
            headers={"Retry-After": "15"},
        )

    uid = _user_identity(request, session_id=session_id)
    if uid:
        try:
            user_count = await _bump_counter(
                f"sysarch:rl:usr:{uid}", _RATE_WINDOW_SECONDS
            )
        except Exception:
            user_count = await _bump_local_window(f"usr:{uid}", _RATE_WINDOW_SECONDS)

        if user_count > settings.USER_RATE_LIMIT_PER_MIN:
            raise HTTPException(
                status_code=429,
                detail=f"User session rate limit exceeded ({settings.USER_RATE_LIMIT_PER_MIN} queries/min); retry shortly.",
                headers={"Retry-After": "15"},
            )


async def _enforce_daily_cap() -> None:
    now = datetime.now(UTC)
    day_tag = f"{now:%Y%m%d}"
    try:
        count = await _bump_counter(f"sysarch:daily:{day_tag}", _DAILY_KEY_TTL_SECONDS)
    except Exception as exc:
        logger.warning("Redis daily cap degraded; enforcing local daily guard: %s", exc)
        count = await _bump_local_daily(day_tag)

    if count > settings.DAILY_QUERY_CAP:
        midnight = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
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


def _cors_origins() -> list[str]:
    return [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]


def _summarize_node(node_name: str, out: dict) -> tuple[str, str]:
    if node_name == "analyze_query":
        svcs = ", ".join(out.get("services", [])) or "all"
        status = "fallback" if out.get("analysis_fallback") else "ok"
        return status, f"Routed services: {svcs}"
    if node_name in ("retrieve", "retrieve_and_rerank"):
        if out.get("error"):
            return "error", "Retrieval backend error"
        n = len(out.get("reranked_chunks", []))
        return "ok", f"Retrieved & reranked {n} chunks"
    if node_name in ("grade", "grade_documents"):
        g = out.get("grade_result")
        if g is not None:
            status = (
                "fallback"
                if "fallback" in g.reasoning.lower()
                else ("ok" if g.is_sufficient else "retry")
            )
            return status, f"Sufficient={g.is_sufficient} (conf={g.confidence:.2f})"
    if node_name in ("rewrite", "rewrite_query"):
        if out.get("rewrite_exhausted"):
            return "fallback", "Rewrite budget exhausted"
        return "retry", f"Rewritten query (attempt {out.get('retry_count', 1)})"
    if node_name in ("synthesize", "synthesize_answer"):
        if out.get("refusal_reason") == "service_error":
            return "error", "Upstream synthesis error"
        c = len(out.get("citations", []))
        return "ok", f"Synthesized answer with {c} grounded citations"
    return "ok", "Completed"


def _response_from_state(
    state: dict, t0: float, request_id: str, trace: list[TraceStep] | None = None
) -> RAGResponse:
    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    return RAGResponse(
        request_id=request_id,
        answer=str(state.get("final_answer", "")),
        citations=state.get("citations", []),
        services_searched=state.get("services", []),
        retry_count=int(state.get("retry_count", 0)),
        refused=bool(state.get("refused", False)),
        refusal_reason=state.get("refusal_reason"),
        cache_hit=False,
        cache_tier="live_crag",
        latency_ms=elapsed,
        original_latency_ms=elapsed,
        trace=trace or [],
    )


def _initial_state(req: QueryRequest) -> dict:
    init_state: dict = {"query": req.query, "detail_level": req.detail_level}
    if req.service_filter:
        init_state["services"] = [req.service_filter]
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
_enable_docs = (not _is_prod) or settings.EXPOSE_DOCS_IN_PROD
app = FastAPI(
    title="SysArch-RAG API",
    description="Self-Correcting Cloud & Distributed Systems Intelligence Engine",
    version="0.1.0",
    docs_url="/docs" if _enable_docs else None,
    redoc_url="/redoc" if _enable_docs else None,
    openapi_url="/openapi.json" if _enable_docs else None,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "X-Session-Id", "X-User-Id"],
    expose_headers=["Retry-After", "X-Request-ID"],
)


class BodySizeLimitMiddleware:
    def __init__(self, app_asgi, max_bytes: int) -> None:
        self.app_asgi = app_asgi
        settings.MAX_REQUEST_BYTES = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("method") in ("POST", "PUT", "PATCH"):
            headers = dict(scope.get("headers") or [])
            raw_cl = headers.get(b"content-length")
            if raw_cl is not None:
                try:
                    if int(raw_cl.decode()) > settings.MAX_REQUEST_BYTES:
                        resp = JSONResponse(
                            status_code=413,
                            content={"detail": "Request payload too large."},
                        )
                        await resp(scope, receive, send)
                        return
                except ValueError:
                    resp = JSONResponse(
                        status_code=400,
                        content={"detail": "Invalid Content-Length header."},
                    )
                    await resp(scope, receive, send)
                    return

            received = 0

            async def limited_receive():
                nonlocal received
                msg = await receive()
                if msg["type"] == "http.request":
                    received += len(msg.get("body", b""))
                    if received > settings.MAX_REQUEST_BYTES:
                        raise HTTPException(
                            status_code=413, detail="Request payload too large."
                        )
                return msg

            try:
                await self.app_asgi(scope, limited_receive, send)
            except HTTPException as exc:
                if exc.status_code == 413:
                    resp = JSONResponse(status_code=413, content={"detail": exc.detail})
                    await resp(scope, receive, send)
                    return
                raise
            return

        await self.app_asgi(scope, receive, send)


app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.MAX_REQUEST_BYTES)


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    logger.exception("Unhandled API exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict:
    try:
        qdrant_ok = await get_qdrant_client().collection_exists(
            settings.QDRANT_COLLECTION
        )
    except Exception:
        qdrant_ok = False
    try:
        redis_ok = bool(await get_redis_client().ping())
    except Exception:
        redis_ok = False
    if not (qdrant_ok and redis_ok):
        raise HTTPException(status_code=503, detail="Service dependencies unavailable.")
    return {"status": "ready"}


async def run_rag_pipeline(req: QueryRequest) -> RAGResponse:
    t0 = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    dense_vec, _ = await embed_query_async(req.query)

    async with _inflight_lock(req.query, req.service_filter, req.detail_level):
        cached = await lookup_cache(
            req.query,
            req.service_filter,
            dense_vec=dense_vec,
            detail_level=req.detail_level,
        )
        if cached is not None:
            cached.request_id = req_id
            cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            return cached

        await _admit_pipeline()
        try:
            async with asyncio.timeout(45.0):
                run_cfg = get_langfuse_run_config(
                    session_id=req.session_id, tags=["api-v1"]
                )
                run_cfg["recursion_limit"] = 15
                state = await rag_graph.ainvoke(_initial_state(req), config=run_cfg)
                resp = _response_from_state(state, t0, req_id)
                if resp.refusal_reason == "service_error":
                    raise HTTPException(
                        status_code=503,
                        detail="Upstream retrieval or inference service unavailable.",
                    )
                await store_cache(
                    req.query,
                    req.service_filter,
                    dense_vec,
                    resp,
                    detail_level=req.detail_level,
                )
                return resp
        finally:
            _CONCURRENCY_GUARD.release()


@app.post("/v1/query", response_model=RAGResponse)
async def query_endpoint(req: QueryRequest, request: Request) -> RAGResponse:
    await _enforce_rate_limit(request, session_id=req.session_id)
    try:
        return await run_rag_pipeline(req)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="RAG pipeline timed out.") from exc


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.post("/v1/query/stream")
async def query_stream_endpoint(
    req: QueryRequest, request: Request
) -> StreamingResponse:
    await _enforce_rate_limit(request, session_id=req.session_id)
    t0 = time.perf_counter()
    req_id = f"req_{uuid.uuid4().hex[:12]}"
    dense_vec, _ = await embed_query_async(req.query)
    cached = await lookup_cache(
        req.query,
        req.service_filter,
        dense_vec=dense_vec,
        detail_level=req.detail_level,
    )
    if cached is not None:
        cached.request_id = req_id
        cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        hit = cached

        async def cached_gen() -> AsyncGenerator[str, None]:
            yield _sse({"type": "start", "request_id": req_id, "query": req.query})
            yield _sse({"type": "cache", "hit": True, "tier": hit.cache_tier})
            yield _sse({"type": "trace", "node": "cache_lookup"})
            for step in hit.trace:
                yield _sse(
                    {
                        "type": "node_end",
                        "node": step.node,
                        "status": step.status,
                        "duration_ms": step.duration_ms,
                        "summary": step.summary,
                    }
                )
            yield _sse({"type": "final", "payload": hit.model_dump()})

        return StreamingResponse(
            cached_gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    await _admit_pipeline()
    slot_owned_by_gen = False

    async def event_gen() -> AsyncGenerator[str, None]:
        nonlocal slot_owned_by_gen
        slot_owned_by_gen = True
        try:
            yield _sse({"type": "start", "request_id": req_id, "query": req.query})
            yield _sse({"type": "cache", "hit": False, "tier": "live_crag"})
            yield _sse({"type": "trace", "node": "cache_lookup"})
            async with asyncio.timeout(45.0):
                run_cfg = get_langfuse_run_config(
                    session_id=req.session_id, tags=["api-stream-v1"]
                )
                run_cfg["recursion_limit"] = 15
                final_state: dict = {}
                trace_steps: list[TraceStep] = []
                step_t0 = time.perf_counter()
                yield _sse({"type": "node_start", "node": "analyze_query"})

                async for update in rag_graph.astream(
                    _initial_state(req), config=run_cfg, stream_mode="updates"
                ):
                    for node_name, node_out in update.items():
                        dur_ms = round((time.perf_counter() - step_t0) * 1000, 2)
                        status, summary = _summarize_node(node_name, node_out)
                        step = TraceStep(
                            node=node_name,
                            status=status,
                            duration_ms=dur_ms,
                            summary=summary,
                        )
                        trace_steps.append(step)
                        final_state.update(node_out)
                        yield _sse({"type": "trace", "node": node_name})
                        yield _sse(
                            {
                                "type": "node_end",
                                "node": node_name,
                                "status": status,
                                "duration_ms": dur_ms,
                                "summary": summary,
                            }
                        )
                        step_t0 = time.perf_counter()

                resp = _response_from_state(final_state, t0, req_id, trace=trace_steps)
                if resp.refusal_reason == "service_error":
                    yield _sse(
                        {
                            "type": "error",
                            "detail": "Upstream retrieval or inference service unavailable.",
                        }
                    )
                    return
                await store_cache(
                    req.query,
                    req.service_filter,
                    dense_vec,
                    resp,
                    detail_level=req.detail_level,
                )
                yield _sse({"type": "final", "payload": resp.model_dump()})
        except TimeoutError:
            yield _sse({"type": "error", "detail": "RAG pipeline timed out after 45s."})
        except Exception:
            logger.exception("SSE stream error")
            yield _sse({"type": "error", "detail": "Internal streaming error."})
        finally:
            _CONCURRENCY_GUARD.release()

    try:
        return StreamingResponse(
            event_gen(), media_type="text/event-stream", headers=_SSE_HEADERS
        )
    except Exception:
        if not slot_owned_by_gen:
            _CONCURRENCY_GUARD.release()
        raise
