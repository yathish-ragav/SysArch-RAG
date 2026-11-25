import asyncio
import cohere
import httpx
from src.core.config import settings
from src.core.schemas import ScoredChunk

_http_client: httpx.AsyncClient | None = None
_cohere_client: cohere.AsyncClientV2 | None = None


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
    if not candidates:
        return []

    response = await get_cohere_client().rerank(
        model=settings.RERANK_MODEL,
        query=query,
        documents=[c.chunk.content for c in candidates],
        top_n=min(top_n, len(candidates)),
    )

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
