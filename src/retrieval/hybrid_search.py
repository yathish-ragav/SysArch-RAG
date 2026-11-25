import asyncio
from functools import lru_cache
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models
from src.core.config import settings
from src.core.schemas import DocMetadata, DocumentChunk, ScoredChunk
from src.ingestion.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

_EMBED_SEMAPHORE = asyncio.Semaphore(3)
_qdrant_client: AsyncQdrantClient | None = None


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.DENSE_MODEL, threads=2)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.SPARSE_MODEL, threads=2)


def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=20.0)
    return _qdrant_client


async def close_qdrant_client() -> None:
    global _qdrant_client
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None


def _embed_sync(query: str) -> tuple[list[float], models.SparseVector]:
    dense = next(get_dense_model().embed([query])).tolist()
    sparse_iter = getattr(get_sparse_model(), "query_embed", get_sparse_model().embed)
    sparse_raw = next(sparse_iter([query]))
    sparse = models.SparseVector(indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist())
    return dense, sparse


async def embed_query_async(query: str) -> tuple[list[float], models.SparseVector]:
    async with _EMBED_SEMAPHORE:
        return await asyncio.to_thread(_embed_sync, query)


def _build_filter(services: list[str] | None = None) -> models.Filter | None:
    if not services:
        return None
    if len(services) == 1:
        cond: models.Condition = models.FieldCondition(key="service", match=models.MatchValue(value=services[0]))
    else:
        cond = models.FieldCondition(key="service", match=models.MatchAny(any=services))
    return models.Filter(must=[cond])


async def hybrid_search(
    query: str,
    top_k: int = 25,
    services: list[str] | None = None,
    precomputed: tuple[list[float], models.SparseVector] | None = None,
) -> list[ScoredChunk]:
    dense_vec, sparse_vec = precomputed if precomputed is not None else await embed_query_async(query)
    q_filter = _build_filter(services=services)
    prefetch_limit = max(top_k * 2, 40)

    response = await get_qdrant_client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, filter=q_filter, limit=prefetch_limit),
            models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, filter=q_filter, limit=prefetch_limit),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    results: list[ScoredChunk] = []
    for pt in response.points:
        payload = pt.payload or {}
        if "content" not in payload or "file_path" not in payload:
            continue
        chunk = DocumentChunk(
            chunk_id=str(pt.id),
            content=str(payload["content"]),
            metadata=DocMetadata(
                service=str(payload.get("service", "unknown")),
                version=str(payload.get("version", "unknown")),
                file_path=str(payload["file_path"]),
                section_hierarchy=str(payload.get("section_hierarchy", "")),
                has_code_block=bool(payload.get("has_code_block", False)),
            ),
        )
        results.append(ScoredChunk(chunk=chunk, hybrid_score=float(pt.score)))
    return results
