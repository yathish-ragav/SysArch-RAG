import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
import time
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
from src.core.schemas import QueryRequest, RAGResponse
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
# Max simultaneous full pipeline runs (each can fan out to Groq + Cohere). Beyond this we answer 429.
_CONCURRENCY_GUARD = asyncio.Semaphore(4)
_RATE_WINDOW_SECONDS = 60
_DAILY_KEY_TTL_SECONDS = 25 * 3600  # outlives the UTC day so the key always expires on its own


def _client_ip(request: Request) -> str:
    """Return the caller IP used as the rate-limit key.

    Each proxy APPENDS the address it saw, so the rightmost entry is the one our own proxy (Render) wrote and
    can be trusted. The leftmost entry is whatever the client chose to send, so using it would let anyone
    dodge the limiter by rotating a fake header. Assumes exactly one trusted proxy hop; adjust if you add a
    CDN in front. Falls back to the socket peer for local dev (no proxy). Never call .strip() on the raw
    split() result: that is a list (the original crash).
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        last = fwd.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


async def _bump_counter(key: str, ttl_seconds: int) -> int:
    """Atomic fixed-window counter: SET NX EX then INCR inside one MULTI/EXEC.

    `SET key 0 EX ttl NX` only creates the key if absent (so the window start is never extended), then INCR
    returns the new count.

    The key is created with a TTL before it is ever incremented, and the transaction prevents the
    key expiring between the two commands, so no counter is ever left without a TTL.
    """
    pipe = get_redis_client().pipeline(transaction=True)
    pipe.set(key, 0, ex=ttl_seconds, nx=True)
    pipe.incr(key)
    results = await pipe.execute()
    return int(results[-1])


async def _enforce_rate_limit(request: Request) -> None:
    # Per-IP burst protection. Fail OPEN if Redis is unreachable: a cache outage should degrade protection,
    # not take the whole API down.
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
    """Global UTC-day budget on full pipeline runs (cache hits cost no Groq/Cohere quota)."""
    # One counter per UTC calendar day. The 25h TTL guarantees the key self-deletes even if the date rolls.
    now = datetime.now(timezone.utc)
    try:
        count = await _bump_counter(f"sysarch:daily:{now:%Y%m%d}", _DAILY_KEY_TTL_SECONDS)
    except Exception as exc:
        logger.warning("Daily cap skipped: %s", exc)
        return
    if count > settings.DAILY_QUERY_CAP:
        # Retry-After = seconds until the next UTC midnight, when the counter key changes.
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        raise HTTPException(
            status_code=429,
            detail=f"Daily query budget ({settings.DAILY_QUERY_CAP}/day) reached; resets at 00:00 UTC.",
            headers={"Retry-After": str(max(1, int((midnight - now).total_seconds())))},
        )


async def _admit_pipeline() -> None:
    """Called only after a cache miss: reject with 429 if the server is busy or the daily cap is spent."""
    # Order matters: check "busy" first so a rejected request does not burn a unit of the daily budget.
    if _CONCURRENCY_GUARD.locked():
        raise HTTPException(status_code=429, detail="Server busy; retry shortly.", headers={"Retry-After": "3"})
    await _enforce_daily_cap()


def _cors_origins() -> list[str]:
    # Parse the comma-separated CORS_ORIGINS setting into the list CORSMiddleware expects.
    return [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]


def _response_from_state(state: dict, t0: float) -> RAGResponse:
    return RAGResponse(
        answer=str(state.get("final_answer", "")),
        citations=state.get("citations", []),
        services_searched=state.get("services", []),
        retry_count=int(state.get("retry_count", 0)),
        refused=bool(state.get("refused", False)),
        refusal_reason=state.get("refusal_reason"),
        cache_hit=False,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


def _initial_state(req: QueryRequest) -> dict:
    init_state: dict = {"query": req.query}
    if req.service_filter:
        init_state["services"] = [req.service_filter]
    return init_state


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    # Warm CPU ONNX models at startup so first request has zero cold-start stall
    await asyncio.to_thread(lambda: (get_dense_model(), get_sparse_model()))
    await embed_query_async("warmup")
    await ensure_semantic_cache_collection()
    yield
    await close_cohere_client()
    await close_qdrant_client()
    await close_redis_client()
    flush_langfuse()


app = FastAPI(
    title="SysArch-RAG API",
    description="Self-Correcting Cloud & Distributed Systems Intelligence Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict[str, object]:
    qdrant_ok = await get_qdrant_client().collection_exists(settings.QDRANT_COLLECTION)
    redis_ok = bool(await get_redis_client().ping())
    if not (qdrant_ok and redis_ok):
        raise HTTPException(status_code=503, detail={"qdrant": qdrant_ok, "redis": redis_ok})
    return {"status": "ready", "collection": settings.QDRANT_COLLECTION, "redis": True}


async def run_rag_pipeline(req: QueryRequest) -> RAGResponse:
    t0 = time.perf_counter()
    dense_vec, _ = await embed_query_async(req.query)

    cached = await lookup_cache(req.query, req.service_filter, dense_vec=dense_vec)
    if cached is not None:
        cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return cached

    await _admit_pipeline()

    async with _CONCURRENCY_GUARD, asyncio.timeout(45.0):
        run_cfg = get_langfuse_run_config(session_id=req.session_id, tags=["api-v1"])
        run_cfg["recursion_limit"] = 15
        state = await rag_graph.ainvoke(_initial_state(req), config=run_cfg)
        resp = _response_from_state(state, t0)
        if resp.refusal_reason == "service_error":
            raise HTTPException(status_code=503, detail=resp.answer)
        await store_cache(req.query, req.service_filter, dense_vec, resp)
        return resp


@app.post("/v1/query", response_model=RAGResponse)
async def query_endpoint(req: QueryRequest, request: Request) -> RAGResponse:
    await _enforce_rate_limit(request)
    try:
        return await run_rag_pipeline(req)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="RAG pipeline timed out.") from exc


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/v1/query/stream")
async def query_stream_endpoint(req: QueryRequest, request: Request) -> StreamingResponse:
    await _enforce_rate_limit(request)
    t0 = time.perf_counter()

    # Cache FIRST, admission control SECOND. A cache hit costs no Groq/Cohere quota and takes ~30ms, so it
    # must never be rejected just because the pipeline is saturated. (Previously the busy check ran first.)
    # This all happens before StreamingResponse is created, because once streaming starts the HTTP status
    # is already 200 and a 429 could only be sent as an in-band error event.
    dense_vec, _ = await embed_query_async(req.query)
    cached = await lookup_cache(req.query, req.service_filter, dense_vec=dense_vec)
    if cached is not None:
        cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        hit = cached

        async def cached_gen() -> AsyncGenerator[str, None]:
            yield _sse({"type": "trace", "node": "cache_lookup"})
            yield _sse({"type": "final", "payload": hit.model_dump()})

        return StreamingResponse(cached_gen(), media_type="text/event-stream")

    # Cache miss: busy / daily-cap rejections are real HTTP 429s (with Retry-After) that the frontend can
    # handle via response.status, raised before any stream bytes are sent.
    await _admit_pipeline()

    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            yield _sse({"type": "trace", "node": "cache_lookup"})
            async with _CONCURRENCY_GUARD, asyncio.timeout(45.0):
                run_cfg = get_langfuse_run_config(session_id=req.session_id, tags=["api-stream-v1"])
                run_cfg["recursion_limit"] = 15

                final_state: dict = {}
                async for update in rag_graph.astream(_initial_state(req), config=run_cfg, stream_mode="updates"):
                    for node_name, node_out in update.items():
                        final_state.update(node_out)
                        yield _sse({"type": "trace", "node": node_name})

                resp = _response_from_state(final_state, t0)
                if resp.refusal_reason == "service_error":
                    # Upstream fault: surface as an error event and skip store_cache so it is never cached.
                    yield _sse({"type": "error", "detail": resp.answer})
                    return
                await store_cache(req.query, req.service_filter, dense_vec, resp)
                yield _sse({"type": "final", "payload": resp.model_dump()})
        except TimeoutError:
            yield _sse({"type": "error", "detail": "RAG pipeline timed out after 45s."})
        except Exception as exc:
            logger.exception("SSE stream error: %s", exc)
            yield _sse({"type": "error", "detail": "Internal streaming error."})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
