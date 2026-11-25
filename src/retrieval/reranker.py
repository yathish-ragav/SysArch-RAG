import asyncio
import cohere
import httpx
from src.core.config import settings
from src.core.schemas import ScoredChunk
from src.retrieval.hybrid_search import hybrid_search


async def rerank_chunks(
    query: str,
    candidates: list[ScoredChunk],
    top_n: int = 5,
) -> list[ScoredChunk]:
    if not candidates:
        return []

    async with httpx.AsyncClient(timeout=30.0) as http_client:
        co = cohere.AsyncClientV2(
            api_key=settings.COHERE_API_KEY,
            httpx_client=http_client,
        )
        response = await co.rerank(
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
    test_query = "How does etcd handle Raft leader election timeout and heartbeat interval?"
    candidates = await hybrid_search(test_query, top_k=15, service_filter="etcd-cncf")
    top_chunks = await rerank_chunks(test_query, candidates, top_n=3)
    for rank, sc in enumerate(top_chunks, start=1):
        print(f"#{rank} | rerank={sc.rerank_score:.4f} | rrf={sc.hybrid_score:.4f} | {sc.chunk.metadata.file_path} :: {sc.chunk.metadata.section_hierarchy}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
