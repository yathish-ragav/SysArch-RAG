import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import json
import logging
import time
from fastapi import FastAPI, HTTPException
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
_CONCURRENCY_GUARD = asyncio.Semaphore(4)


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
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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

    if _CONCURRENCY_GUARD.locked():
        raise HTTPException(status_code=429, detail="Server busy; retry shortly.", headers={"Retry-After": "3"})

    async with _CONCURRENCY_GUARD, asyncio.timeout(45.0):
        run_cfg = get_langfuse_run_config(session_id=req.session_id, tags=["api-v1"])
        run_cfg["recursion_limit"] = 15
        init_state = {"query": req.query}
        if req.service_filter:
            init_state["services"] = [req.service_filter]

        state = await rag_graph.ainvoke(init_state, config=run_cfg)
        resp = RAGResponse(
            answer=str(state.get("final_answer", "")),
            citations=state.get("citations", []),
            services_searched=state.get("services", []),
            retry_count=int(state.get("retry_count", 0)),
            refused=bool(state.get("refused", False)),
            cache_hit=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        await store_cache(req.query, req.service_filter, dense_vec, resp)
        return resp


@app.post("/v1/query", response_model=RAGResponse)
async def query_endpoint(req: QueryRequest) -> RAGResponse:
    try:
        return await run_rag_pipeline(req)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="RAG pipeline timed out.") from exc


@app.post("/v1/query/stream")
async def query_stream_endpoint(req: QueryRequest) -> StreamingResponse:
    async def event_gen() -> AsyncGenerator[str, None]:
        t0 = time.perf_counter()
        try:
            yield f"data: {json.dumps({'type': 'trace', 'node': 'cache_lookup'})}\n\n"
            dense_vec, _ = await embed_query_async(req.query)
            cached = await lookup_cache(req.query, req.service_filter, dense_vec=dense_vec)
            if cached is not None:
                cached.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                yield f"data: {json.dumps({'type': 'final', 'payload': cached.model_dump()})}\n\n"
                return

            if _CONCURRENCY_GUARD.locked():
                yield f"data: {json.dumps({'type': 'error', 'detail': 'Server busy; retry shortly.'})}\n\n"
                return

            async with _CONCURRENCY_GUARD, asyncio.timeout(45.0):
                run_cfg = get_langfuse_run_config(session_id=req.session_id, tags=["api-stream-v1"])
                run_cfg["recursion_limit"] = 15
                init_state = {"query": req.query}
                if req.service_filter:
                    init_state["services"] = [req.service_filter]

                final_state: dict = {}
                async for update in rag_graph.astream(init_state, config=run_cfg, stream_mode="updates"):
                    for node_name, node_out in update.items():
                        final_state.update(node_out)
                        yield f"data: {json.dumps({'type': 'trace', 'node': node_name})}\n\n"

                resp = RAGResponse(
                    answer=str(final_state.get("final_answer", "")),
                    citations=final_state.get("citations", []),
                    services_searched=final_state.get("services", []),
                    retry_count=int(final_state.get("retry_count", 0)),
                    refused=bool(final_state.get("refused", False)),
                    cache_hit=False,
                    latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                await store_cache(req.query, req.service_filter, dense_vec, resp)
                yield f"data: {json.dumps({'type': 'final', 'payload': resp.model_dump()})}\n\n"
        except TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'detail': 'RAG pipeline timed out after 45s.'})}\n\n"
        except Exception as exc:
            logger.exception("SSE stream error: %s", exc)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Internal streaming error.'})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
