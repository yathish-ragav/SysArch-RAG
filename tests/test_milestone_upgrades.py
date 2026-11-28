import pytest

from src.core.schemas import DocMetadata, DocumentChunk, GradeResult, ScoredChunk
from src.graph import nodes
from src.ingestion import indexer
from src.retrieval import reranker


def _scored(
    cid: str, service: str, path: str, text: str, score: float = 0.9
) -> ScoredChunk:
    return ScoredChunk(
        chunk=DocumentChunk(
            chunk_id=cid,
            content=f"[{service} | v1.32 | {path} | Root]\n{text}",
            metadata=DocMetadata(
                service=service,
                version="v1.32",
                file_path=path,
                section_hierarchy="Root",
                has_code_block="```" in text,
            ),
        ),
        hybrid_score=0.8,
        rerank_score=score,
    )


def test_official_doc_urls_and_claim_grounding():
    assert (
        nodes._official_doc_url("kubernetes", "kubernetes/concepts/workloads/pods.md")
        == "https://kubernetes.io/docs/concepts/workloads/pods/"
    )
    assert (
        nodes._official_doc_url("aws-eks", "aws-eks/pod-identities.md")
        == "https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html"
    )
    assert (
        nodes._official_doc_url("etcd-cncf", "etcd-cncf/learning/why.md")
        == "https://etcd.io/docs/latest/learning/why/"
    )

    smap = {
        "S1": _scored(
            "1",
            "etcd-cncf",
            "etcd-cncf/raft.md",
            "Raft elects a single leader using randomized election timeouts.",
        )
    }
    raw = (
        "Raft elects a single leader using randomized election timeouts [S1].\n\n"
        "Completely unrelated hallucinated sentence about quantum blockchain [S1].\n\n"
        "Uncited sentence without source tag."
    )
    cleaned, ids = nodes._verify_claim_grounding(raw, smap)
    assert "randomized election timeouts [S1]" in cleaned
    assert "quantum blockchain" not in cleaned
    assert "Uncited sentence" not in cleaned
    assert ids == ["S1"]


async def test_synthesis_detail_levels_and_full_text(monkeypatch):
    seen_prompts: list[str] = []

    class _StubSynth:
        async def ainvoke(self, messages):
            seen_prompts.append(messages[0].content)
            return type(
                "Msg",
                (),
                {
                    "content": "Pod Topology Spread uses `maxSkew` to balance pods across zones [S1]."
                },
            )()

    monkeypatch.setattr(nodes, "_synthesis_llm", lambda: _StubSynth())
    chunk = _scored(
        "c1",
        "kubernetes",
        "kubernetes/concepts/scheduling.md",
        "Pod Topology Spread uses maxSkew to balance pods across failure domains.",
    )
    out = await nodes.synthesize_answer_node(
        {
            "query": "How does maxSkew work?",
            "detail_level": "detailed",
            "reranked_chunks": [chunk],
            "grade_result": GradeResult(
                is_sufficient=True, confidence=0.95, reasoning="ok"
            ),
        }
    )
    assert out["refused"] is False
    assert "STYLE (DETAILED)" in seen_prompts[0]
    assert out["citations"][0].source_url.startswith("https://kubernetes.io")
    assert "maxSkew" in out["citations"][0].full_text


async def test_cohere_pace_slot_uses_redis_set_nx(monkeypatch):
    calls: list[tuple] = []

    class _FakeRedis:
        async def set(self, key, val, px=None, nx=False):
            calls.append((key, val, px, nx))
            return True

    monkeypatch.setattr(reranker, "get_redis_client", lambda: _FakeRedis())
    await reranker._acquire_cohere_pace_slot()
    assert len(calls) == 1
    assert calls[0][0] == reranker._PACE_KEY
    assert calls[0][3] is True


async def test_indexer_aborts_when_build_lock_held(monkeypatch):
    class _LockedRedis:
        async def set(self, *_a, **_kw):
            return False

        async def aclose(self):
            return None

    monkeypatch.setattr(indexer.redis, "from_url", lambda *_a, **_kw: _LockedRedis())
    chunk = _scored("id1", "kubernetes", "kubernetes/a.md", "hello").chunk
    with pytest.raises(RuntimeError, match="already running"):
        await indexer.index_chunks([chunk])
