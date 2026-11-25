import asyncio
from pathlib import Path
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models
from src.core.config import settings
from src.core.schemas import DocumentChunk
from src.ingestion.chunker import load_and_chunk_corpus

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIM = 768


async def ensure_hybrid_collection(client: AsyncQdrantClient, collection_name: str) -> None:
    if await client.collection_exists(collection_name):
        await client.delete_collection(collection_name)
    await client.create_collection(
        collection_name=collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )
    for field_name in ("service", "version", "has_code_block"):
        schema = models.PayloadSchemaType.BOOL if field_name == "has_code_block" else models.PayloadSchemaType.KEYWORD
        await client.create_payload_index(collection_name=collection_name, field_name=field_name, field_schema=schema)


async def index_chunks(chunks: list[DocumentChunk], batch_size: int = 32) -> None:
    print(f"Loading CPU ONNX models ({settings.DENSE_MODEL} & {settings.SPARSE_MODEL})...")
    dense_model = TextEmbedding(model_name=settings.DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)
    client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    try:
        await ensure_hybrid_collection(client, settings.QDRANT_COLLECTION)
        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.content for c in batch]
            dense_vecs = list(dense_model.embed(texts))
            sparse_vecs = list(sparse_model.embed(texts))
            points = [
                models.PointStruct(
                    id=c.chunk_id,
                    vector={
                        DENSE_VECTOR_NAME: d.tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
                    },
                    payload={"content": c.content, **c.metadata.model_dump()},
                )
                for c, d, s in zip(batch, dense_vecs, sparse_vecs)
            ]
            await client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points, wait=True)
            print(f"Indexed {min(start + batch_size, total)}/{total} chunks...")
        info = await client.get_collection(settings.QDRANT_COLLECTION)
        print(f"SUCCESS: Collection '{settings.QDRANT_COLLECTION}' live with {info.points_count} hybrid points.")
    finally:
        await client.close()


if __name__ == "__main__":
    chunks = load_and_chunk_corpus(Path("data/raw_docs"), max_files_per_service=150)
    asyncio.run(index_chunks(chunks, batch_size=32))
