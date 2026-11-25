import asyncio
from langgraph.graph import END, START, StateGraph
from src.core.config import settings
from src.core.observability import flush_langfuse, get_langfuse_run_config
from src.graph.nodes import (
    analyze_query_node,
    grade_documents_node,
    retrieve_and_rerank_node,
    rewrite_query_node,
    synthesize_answer_node,
)
from src.graph.state import GraphState
from src.retrieval.hybrid_search import close_qdrant_client
from src.retrieval.reranker import close_cohere_client


def _route_after_grading(state: GraphState) -> str:
    grade = state.get("grade_result")
    retries = state.get("retry_count", 0)
    if grade and grade.is_sufficient:
        return "synthesize"
    if retries < settings.MAX_RETRIES:
        return "rewrite"
    return "synthesize"


def build_rag_graph():
    builder = StateGraph(GraphState)
    builder.add_node("analyze_query", analyze_query_node)
    builder.add_node("retrieve_and_rerank", retrieve_and_rerank_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("synthesize", synthesize_answer_node)

    builder.add_edge(START, "analyze_query")
    builder.add_edge("analyze_query", "retrieve_and_rerank")
    builder.add_edge("retrieve_and_rerank", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {"synthesize": "synthesize", "rewrite": "rewrite_query"},
    )
    builder.add_edge("rewrite_query", "retrieve_and_rerank")
    builder.add_edge("synthesize", END)
    return builder.compile()


rag_graph = build_rag_graph()


async def _smoke_test() -> None:
    try:
        run_cfg = get_langfuse_run_config(session_id="hardened-smoke-test", tags=["hardened-v1"])
        run_cfg["recursion_limit"] = 15
        result = await rag_graph.ainvoke(
            {"query": "How do PodDisruptionBudgets protect workloads during a voluntary node drain in Kubernetes?"},
            config=run_cfg,
        )
        print(
            f"Services: {result.get('services')} | Fallback: {result.get('analysis_fallback')} | "
            f"Retries: {result.get('retry_count', 0)} | Refused: {result.get('refused')}"
        )
        print("Citations:", [(c.ref_id, c.file_path, c.section) for c in result.get("citations", [])])
        print("\n--- ANSWER ---\n", result.get("final_answer"))
    finally:
        await close_cohere_client()
        await close_qdrant_client()
        flush_langfuse()


if __name__ == "__main__":
    asyncio.run(_smoke_test())
