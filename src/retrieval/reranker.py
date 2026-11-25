import asyncio
import logging
import time
import cohere
from cohere.errors import TooManyRequestsError
import httpx
from src.core.config import settings
from src.core.schemas import ScoredChunk

logger = logging.getLogger(__name__)

_http_client: httpx.AsyncClient | None = None
_cohere_client: cohere.AsyncClientV2 | None = None
_rate_lock = asyncio.Lock()
_last_call_ts: float = 0.0
_MIN_INTERVAL_SEC: float = 6.8  # <= 8.8 calls/min (safe under Cohere Trial 10 req/min limit)
_rerank_cache: dict[tuple[str, tuple[str, ...], int], list[ScoredChunk]] = {}


def get_cohere_client() -> cohere.AsyncClientV2:
    global _http_client, _cohere_client
    if _cohere_client is None:
        _http_client = httpx.AsyncClient(timeout=25.0, limits=httpx.Limits(max_connections=20, max_keepalive_connections=10))
        _cohere_client = cohere.AsyncClientV2(api_key=settings.COHERE_API_KEY, httpx_client=_http_client)
    return _cohere_client


async def close_cohere_client() -> None:
    global _http_client, _cohere_client
    if _http_client is not None:
        await _http_client.aclose()
    _http_client = None
    _cohere_client = None


async def rerank_chunks(
    query: str,
    candidates: list[ScoredChunk],
    top_n: int = 5,
) -> list[ScoredChunk]:
    global _last_call_ts
    if not candidates:
        return []

    effective_top_n = min(top_n, len(candidates))
    cache_key = (query.strip(), tuple(c.chunk.chunk_id for c in candidates), effective_top_n)
    if cache_key in _rerank_cache:
        return _rerank_cache[cache_key]

    response = None
    for attempt in range(4):
        async with _rate_lock:
            now = time.monotonic()
            wait_s = _MIN_INTERVAL_SEC - (now - _last_call_ts)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            _last_call_ts = time.monotonic()

        try:
            response = await get_cohere_client().rerank(
                model=settings.RERANK_MODEL,
                query=query,
                documents=[c.chunk.content for c in candidates],
                top_n=effective_top_n,
            )
            break
        except TooManyRequestsError:
            backoff = 15.0 * (attempt + 1)
            logger.warning("Cohere 429 rate limit hit (attempt %d/4); sleeping %.1fs", attempt + 1, backoff)
            await asyncio.sleep(backoff)
            if attempt == 3:
                raise

    assert response is not None
    reranked: list[ScoredChunk] = []
    for item in response.results:
        original = candidates[item.index]
        reranked.append(
            ScoredChunk(
                chunk=original.chunk,
                hybrid_score=original.hybrid_score,
                rerank_score=float(item.relevance_score),
            )
        )
    _rerank_cache[cache_key] = reranked
    return reranked


async def _smoke_test() -> None:
    from src.retrieval.hybrid_search import close_qdrant_client, hybrid_search

    try:
        test_query = "How does etcd handle Raft leader election timeout and heartbeat interval?"
        candidates = await hybrid_search(test_query, top_k=25, services=["etcd-cncf"])
        top_chunks = await rerank_chunks(test_query, candidates, top_n=3)
        for rank, sc in enumerate(top_chunks, start=1):
            print(f"#{rank} | rerank={sc.rerank_score:.4f} | rrf={sc.hybrid_score:.4f} | {sc.chunk.metadata.file_path} :: {sc.chunk.metadata.section_hierarchy}")
    finally:
        await close_cohere_client()
        await close_qdrant_client()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
