from functools import lru_cache
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models
from src.core.config import settings
from src.core.schemas import DocMetadata, DocumentChunk, ScoredChunk
from src.ingestion.indexer import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.DENSE_MODEL)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.SPARSE_MODEL)


def _build_filter(service: str | None = None, version: str | None = None) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if service:
        conditions.append(models.FieldCondition(key="service", match=models.MatchValue(value=service)))
    if version:
        conditions.append(models.FieldCondition(key="version", match=models.MatchValue(value=version)))
    return models.Filter(must=conditions) if conditions else None


async def hybrid_search(
    query: str,
    top_k: int = 20,
    service_filter: str | None = None,
    version_filter: str | None = None,
) -> list[ScoredChunk]:
    dense_vec = next(get_dense_model().embed([query])).tolist()
    sparse_raw = next(get_sparse_model().embed([query]))
    sparse_vec = models.SparseVector(indices=sparse_raw.indices.tolist(), values=sparse_raw.values.tolist())
    q_filter = _build_filter(service=service_filter, version=version_filter)

    client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    try:
        response = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, filter=q_filter, limit=top_k),
                models.Prefetch(query=sparse_vec, using=SPARSE_VECTOR_NAME, filter=q_filter, limit=top_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
    finally:
        await client.close()

    results: list[ScoredChunk] = []
    for pt in response.points:
        payload = pt.payload or {}
        chunk = DocumentChunk(
            chunk_id=str(pt.id),
            content=payload.get("content", ""),
            metadata=DocMetadata(
                service=payload.get("service", "kubernetes"),
                version=payload.get("version", "1.31"),
                file_path=payload.get("file_path", ""),
                section_hierarchy=payload.get("section_hierarchy", ""),
                has_code_block=bool(payload.get("has_code_block", False)),
            ),
        )
        results.append(ScoredChunk(chunk=chunk, hybrid_score=float(pt.score)))
    return results
