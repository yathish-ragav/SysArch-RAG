from langchain_core.messages import AIMessage
from src.core.schemas import DocMetadata, DocumentChunk, GradeResult, QueryAnalysisOut, ScoredChunk
from src.graph import nodes
from src.graph.workflow import _route_after_grading


def test_schema_service_aliases_and_confidence_clamping():
    qa = QueryAnalysisOut(services=["k8s", "EKS", "etcd", "unknown"], sub_queries=["pdb"])
    assert qa.services == ["kubernetes", "aws-eks", "etcd-cncf"]

    gr = GradeResult(is_sufficient=True, confidence=92, reasoning="Strong match")
    assert gr.confidence == 0.92


def test_route_after_grading_cyclic_logic():
    low = GradeResult(is_sufficient=False, confidence=0.2, reasoning="Weak")
    high = GradeResult(is_sufficient=True, confidence=0.95, reasoning="Sufficient")

    assert _route_after_grading({"grade_result": low, "retry_count": 0}) == "rewrite"
    assert _route_after_grading({"grade_result": low, "retry_count": 2}) == "synthesize"
    assert _route_after_grading({"grade_result": high, "retry_count": 0}) == "synthesize"


async def test_synthesize_answer_node_refusal_and_citation_grounding(monkeypatch):
    refused_out = await nodes.synthesize_answer_node({"query": "How to bake a cake?", "reranked_chunks": []})
    assert refused_out["refused"] is True
    assert refused_out["citations"] == []

    class _FakeLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content="PodDisruptionBudgets limit voluntary disruptions [S1].")

    monkeypatch.setattr(nodes, "_synthesis_llm", lambda: _FakeLLM())
    sc = ScoredChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            content="[KUBERNETES | pdb.md | Spec]\nPodDisruptionBudgets limit voluntary disruptions.",
            metadata=DocMetadata(service="kubernetes", version="1.32", file_path="pdb.md", section_hierarchy="Spec"),
        ),
        hybrid_score=0.9,
        rerank_score=0.96,
    )
    ok_out = await nodes.synthesize_answer_node(
        {
            "query": "What do PDBs do?",
            "reranked_chunks": [sc],
            "grade_result": GradeResult(is_sufficient=True, confidence=0.96, reasoning="Matched"),
        }
    )
    assert ok_out["refused"] is False
    assert len(ok_out["citations"]) == 1
    assert ok_out["citations"][0].ref_id == "S1"
