import hashlib
import logging
import re
import time
import uuid

import redis.asyncio as redis
from qdrant_client import models

from src.core.config import settings
from src.core.schemas import RAGResponse
from src.retrieval.hybrid_search import get_qdrant_client

logger = logging.getLogger(__name__)
SEMANTIC_CACHE_COLLECTION = "sysarch_semantic_cache"
CACHE_TTL_SECONDS = 3600 * 12
NEGATIVE_CACHE_TTL_SECONDS = 600

_redis_client: redis.Redis | None = None
_corpus_version_cache: tuple[str, float] | None = None

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "about",
        "of",
        "from",
        "how",
        "what",
        "when",
        "where",
        "why",
        "which",
        "who",
        "whom",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "if",
        "work",
        "works",
        "working",
        "explain",
        "describe",
        "tell",
        "me",
    }
)


def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=2.0,
            socket_connect_timeout=2.0,
        )
    return _redis_client


async def close_redis_client() -> None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None


async def get_corpus_version() -> str:
    global _corpus_version_cache
    now = time.monotonic()
    if _corpus_version_cache and (now - _corpus_version_cache[1]) < 60.0:
        return _corpus_version_cache[0]
    try:
        aliases = await get_qdrant_client().get_aliases()
        ver = next(
            (
                a.collection_name
                for a in aliases.aliases
                if a.alias_name == settings.QDRANT_COLLECTION
            ),
            settings.QDRANT_COLLECTION,
        )
        _corpus_version_cache = (ver, now)
        return ver
    except Exception as exc:
        logger.warning(
            "Corpus version lookup failed; using '%s' for 60s: %s",
            settings.QDRANT_COLLECTION,
            exc,
        )
        _corpus_version_cache = (settings.QDRANT_COLLECTION, now)
        return settings.QDRANT_COLLECTION


async def ensure_semantic_cache_collection(dense_dim: int = 768) -> None:
    client = get_qdrant_client()
    if not await client.collection_exists(SEMANTIC_CACHE_COLLECTION):
        await client.create_collection(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            vectors_config=models.VectorParams(
                size=dense_dim, distance=models.Distance.COSINE
            ),
        )
        await client.create_payload_index(
            SEMANTIC_CACHE_COLLECTION, "scope_key", models.PayloadSchemaType.KEYWORD
        )


def _normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def _technical_signature(
    q: str,
) -> tuple[frozenset[str], frozenset[str], frozenset[str], frozenset[str]]:
    nums = frozenset(re.findall(r"\bv?\d+(?:\.\d+)*\b", q.lower()))
    flags = frozenset(re.findall(r"--[a-zA-Z0-9_-]+", q.lower()))
    kinds = frozenset(re.findall(r"\b[A-Z][a-zA-Z0-9]+[A-Z][a-zA-Z0-9]+\b", q))
    content_tokens = frozenset(
        w
        for w in re.findall(r"[a-z0-9_-]{3,}", q.lower())
        if w not in _STOPWORDS and not w.isdigit()
    )
    return nums, flags, kinds, content_tokens


def _is_semantically_compatible(query: str, cached_query: str) -> bool:
    nums_a, flags_a, kinds_a, tok_a = _technical_signature(query)
    nums_b, flags_b, kinds_b, tok_b = _technical_signature(cached_query)
    if nums_a != nums_b or flags_a != flags_b or kinds_a != kinds_b:
        return False
    if not tok_a or not tok_b:
        return _normalize_query(query) == _normalize_query(cached_query)
    overlap = len(tok_a & tok_b) / max(len(tok_a | tok_b), 1)
    return overlap >= 0.65


def _build_keys(
    query: str,
    service_filter: str | None,
    corpus_ver: str,
    detail_level: str = "standard",
) -> tuple[str, str]:
    scope = f"{corpus_ver}:{settings.SYNTHESIS_MODEL}:{service_filter or 'all'}:{detail_level}"
    norm = _normalize_query(query)
    exact_hash = hashlib.sha256(f"{scope}:{norm}".encode()).hexdigest()
    return f"sysarch:l1:{exact_hash}", scope


async def lookup_cache(
    query: str,
    service_filter: str | None,
    dense_vec: list[float] | None = None,
    detail_level: str = "standard",
) -> RAGResponse | None:
    corpus_ver = await get_corpus_version()
    l1_key, scope = _build_keys(
        query, service_filter, corpus_ver, detail_level=detail_level
    )
    r = get_redis_client()

    try:
        raw = await r.get(l1_key)
        if raw:
            resp = RAGResponse.model_validate_json(raw)
            resp.cache_hit = True
            resp.cache_tier = "exact_redis"
            return resp
    except Exception as exc:
        logger.warning("Redis L1 lookup skipped: %s", exc)

    if dense_vec is None:
        return None
    try:
        hits = await get_qdrant_client().query_points(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            query=dense_vec,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="scope_key", match=models.MatchValue(value=scope)
                    )
                ]
            ),
            score_threshold=0.97,
            limit=1,
            with_payload=True,
        )
        if hits.points:
            payload = hits.points[0].payload or {}
            cached_q = str(payload.get("query", ""))
            if (
                _is_semantically_compatible(query, cached_q)
                and "response_json" in payload
            ):
                resp = RAGResponse.model_validate_json(str(payload["response_json"]))
                resp.cache_hit = True
                resp.cache_tier = "semantic_qdrant"
                if resp.original_latency_ms is None:
                    resp.original_latency_ms = resp.latency_ms
                return resp
    except Exception as exc:
        logger.warning("Qdrant L2 cache lookup skipped: %s", exc)
    return None


async def store_cache(
    query: str,
    service_filter: str | None,
    dense_vec: list[float],
    response: RAGResponse,
    detail_level: str = "standard",
) -> None:
    if response.refusal_reason == "service_error":
        return
    corpus_ver = await get_corpus_version()
    l1_key, scope = _build_keys(
        query, service_filter, corpus_ver, detail_level=detail_level
    )
    payload_json = response.model_dump_json()

    if response.refused:
        try:
            await get_redis_client().set(
                l1_key, payload_json, ex=NEGATIVE_CACHE_TTL_SECONDS
            )
        except Exception as exc:
            logger.warning("Redis negative L1 write skipped: %s", exc)
        return

    if not response.citations:
        return

    try:
        await get_redis_client().set(l1_key, payload_json, ex=CACHE_TTL_SECONDS)
    except Exception as exc:
        logger.warning("Redis L1 write skipped: %s", exc)

    try:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, l1_key))
        await get_qdrant_client().upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=dense_vec,
                    payload={
                        "scope_key": scope,
                        "query": query,
                        "response_json": payload_json,
                    },
                )
            ],
            wait=False,
        )
    except Exception as exc:
        logger.warning("Qdrant L2 write skipped: %s", exc)
