from qdrant_client import models
from src.api.cache import _technical_signature
from src.core.schemas import DocMetadata, DocumentChunk, ScoredChunk
from src.retrieval.hybrid_search import _build_filter
from src.retrieval.reranker import _rerank_cache, rerank_chunks


def test_build_filter_single_and_multi_service():
    assert _build_filter(None) is None

    single = _build_filter(["kubernetes"])
    assert single is not None
    assert isinstance(single.must[0], models.FieldCondition)
    assert single.must[0].match == models.MatchValue(value="kubernetes")

    multi = _build_filter(["kubernetes", "etcd-cncf"])
    assert multi is not None
    assert isinstance(multi.must[0], models.FieldCondition)
    assert multi.must[0].match == models.MatchAny(any=["kubernetes", "etcd-cncf"])


def test_technical_signature_blocks_mismatched_versions_flags_and_kinds():
    sig_base = _technical_signature("Configure PodDisruptionBudget with --min-available in Kubernetes 1.32")
    sig_diff_ver = _technical_signature("Configure PodDisruptionBudget with --min-available in Kubernetes 1.31")
    sig_diff_flag = _technical_signature("Configure PodDisruptionBudget with --max-unavailable in Kubernetes 1.32")
    sig_diff_kind = _technical_signature("Configure HorizontalPodAutoscaler with --min-available in Kubernetes 1.32")

    assert sig_base != sig_diff_ver
    assert sig_base != sig_diff_flag
    assert sig_base != sig_diff_kind


async def test_rerank_chunks_uses_cache_without_network():
    chunk = DocumentChunk(
        chunk_id="test-chunk-1",
        content="etcd uses Raft consensus.",
        metadata=DocMetadata(service="etcd-cncf", version="v3.5", file_path="etcd/raft.md", section_hierarchy="Raft"),
    )
    candidate = ScoredChunk(chunk=chunk, hybrid_score=0.88)
    expected = [ScoredChunk(chunk=chunk, hybrid_score=0.88, rerank_score=0.95)]
    _rerank_cache[("raft consensus", ("test-chunk-1",), 1)] = expected

    result = await rerank_chunks("raft consensus", [candidate], top_n=1)
    assert result == expected
