import asyncio
from datetime import datetime, timezone
from pathlib import Path
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import AsyncQdrantClient, models
from src.core.config import settings
from src.core.schemas import DocumentChunk
from src.ingestion.chunker import load_and_chunk_corpus

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


async def create_versioned_collection(client: AsyncQdrantClient, target_name: str, dense_dim: int) -> None:
    if await client.collection_exists(target_name):
        await client.delete_collection(target_name)
    await client.create_collection(
        collection_name=target_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
                modifier=models.Modifier.IDF,
            )
        },
    )
    for field_name in ("service", "version", "file_path", "has_code_block"):
        schema = models.PayloadSchemaType.BOOL if field_name == "has_code_block" else models.PayloadSchemaType.KEYWORD
        await client.create_payload_index(collection_name=target_name, field_name=field_name, field_schema=schema)


async def index_chunks(chunks: list[DocumentChunk], batch_size: int = 64) -> None:
    unique_ids = {c.chunk_id for c in chunks}
    assert len(unique_ids) == len(chunks), f"Chunk ID collision: {len(unique_ids)} != {len(chunks)}"

    dense_model = TextEmbedding(model_name=settings.DENSE_MODEL)
    sparse_model = SparseTextEmbedding(model_name=settings.SPARSE_MODEL)
    dense_dim = len(next(dense_model.embed(["dimension check"])))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    versioned_name = f"{settings.QDRANT_COLLECTION}_{stamp}"

    client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY, timeout=60.0)
    try:
        await create_versioned_collection(client, versioned_name, dense_dim)
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
            await client.upsert(collection_name=versioned_name, points=points, wait=True)
            if (start // batch_size) % 10 == 0 or start + batch_size >= total:
                print(f"Indexed {min(start + batch_size, total)}/{total} into '{versioned_name}'...")

        info = await client.get_collection(versioned_name)
        assert info.points_count == total, f"Point count mismatch: {info.points_count} != {total}"

        if await client.collection_exists(settings.QDRANT_COLLECTION):
            aliases = await client.get_collection_aliases(settings.QDRANT_COLLECTION)
            if not aliases.aliases:
                await client.delete_collection(settings.QDRANT_COLLECTION)

        await client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(collection_name=versioned_name, alias_name=settings.QDRANT_COLLECTION)
                )
            ]
        )
        print(f"SUCCESS: Alias '{settings.QDRANT_COLLECTION}' -> '{versioned_name}' ({info.points_count} hybrid IDF points).")
    finally:
        await client.close()


if __name__ == "__main__":
    chunks = load_and_chunk_corpus(Path("data/raw_docs"))
    asyncio.run(index_chunks(chunks, batch_size=64))
