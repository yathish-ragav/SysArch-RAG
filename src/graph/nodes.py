import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from src.core.config import settings
from src.core.schemas import Citation, GradeResult, QueryAnalysis, ScoredChunk
from src.graph.state import GraphState
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_chunks


def _fast_llm() -> ChatGroq:
    return ChatGroq(model=settings.GRADER_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.0)


def _synthesis_llm() -> ChatGroq:
    return ChatGroq(model=settings.SYNTHESIS_MODEL, api_key=settings.GROQ_API_KEY, temperature=0.1)


async def analyze_query_node(state: GraphState) -> GraphState:
    query = state["query"]
    structured_llm = _fast_llm().with_structured_output(QueryAnalysis, method="json_mode")
    prompt = (
        "Return a JSON object with keys: "
        '"service_filter" ("kubernetes", "aws-eks", "etcd-cncf", or null), '
        '"version_filter" (string or null), and '
        '"sub_queries" (list of 1-2 focused technical search strings).'
    )
    try:
        analysis: QueryAnalysis = await structured_llm.ainvoke(
            [SystemMessage(content=prompt), HumanMessage(content=query)]
        )
        sub_queries = analysis.sub_queries[:2] if analysis.sub_queries else [query]
        svc = analysis.service_filter
        ver = analysis.version_filter
    except Exception:
        sub_queries, svc, ver = [query], None, None

    return {
        "active_query": query,
        "sub_queries": sub_queries,
        "service_filter": svc,
        "version_filter": ver,
        "retry_count": state.get("retry_count", 0),
    }


async def retrieve_and_rerank_node(state: GraphState) -> GraphState:
    queries = state.get("sub_queries") or [state.get("active_query", state["query"])]
    tasks = [
        hybrid_search(q, top_k=15, service_filter=state.get("service_filter"), version_filter=state.get("version_filter"))
        for q in queries
    ]
    batch_results = await asyncio.gather(*tasks)

    deduped: dict[str, ScoredChunk] = {}
    for chunk_list in batch_results:
        for sc in chunk_list:
            existing = deduped.get(sc.chunk.chunk_id)
            if existing is None or sc.hybrid_score > existing.hybrid_score:
                deduped[sc.chunk.chunk_id] = sc

    candidates = sorted(deduped.values(), key=lambda x: x.hybrid_score, reverse=True)[:20]
    reranked = await rerank_chunks(state.get("active_query", state["query"]), candidates, top_n=5)
    return {"reranked_chunks": reranked}


async def grade_documents_node(state: GraphState) -> GraphState:
    chunks = state.get("reranked_chunks", [])
    top_score = chunks[0].rerank_score if chunks and chunks[0].rerank_score is not None else 0.0
    if not chunks or top_score < settings.RERANK_SCORE_THRESHOLD:
        return {
            "grade_result": GradeResult(
                is_sufficient=False,
                confidence=float(top_score),
                reasoning=f"Top rerank score ({top_score:.2f}) below threshold ({settings.RERANK_SCORE_THRESHOLD}).",
            )
        }

    context_preview = "\n\n---\n\n".join(c.chunk.content[:900] for c in chunks[:3])
    grader = _fast_llm().with_structured_output(GradeResult, method="json_mode")
    grade: GradeResult = await grader.ainvoke(
        [
            SystemMessage(content='Return JSON with keys: "is_sufficient" (bool), "confidence" (float 0-1), "reasoning" (string).'),
            HumanMessage(content=f"Query: {state['query']}\n\nRetrieved Docs:\n{context_preview}"),
        ]
    )
    return {"grade_result": grade}


class _RewrittenQuery(BaseModel):
    rewritten_query: str = Field(description="Expanded technical search query")


async def rewrite_query_node(state: GraphState) -> GraphState:
    retry_count = state.get("retry_count", 0) + 1
    grade = state.get("grade_result")
    reason = grade.reasoning if grade else "Low relevance score"
    rewriter = _fast_llm().with_structured_output(_RewrittenQuery, method="json_mode")
    res: _RewrittenQuery = await rewriter.ainvoke(
        [
            SystemMessage(content='Return JSON with key "rewritten_query" expanding technical Kubernetes/AWS EKS/etcd terms.'),
            HumanMessage(content=f"Original query: {state['query']}\nFailure reason: {reason}"),
        ]
    )
    return {
        "active_query": res.rewritten_query,
        "sub_queries": [res.rewritten_query],
        "service_filter": None,
        "version_filter": None,
        "retry_count": retry_count,
    }


async def synthesize_answer_node(state: GraphState) -> GraphState:
    chunks = state.get("reranked_chunks", [])
    grade = state.get("grade_result")
    citations = [
        Citation(file_path=c.chunk.metadata.file_path, section=c.chunk.metadata.section_hierarchy, snippet=c.chunk.content[:220])
        for c in chunks[:4]
    ]
    if not chunks or (grade and not grade.is_sufficient and state.get("retry_count", 0) >= settings.MAX_RETRIES):
        return {
            "final_answer": "Insufficient verified documentation found in the indexed corpus to answer with high confidence.",
            "citations": citations,
        }

    formatted_ctx = "\n\n".join(
        f"[Source {i+1}: {c.chunk.metadata.file_path} | {c.chunk.metadata.section_hierarchy}]\n{c.chunk.content}"
        for i, c in enumerate(chunks[:5])
    )
    sys_msg = (
        "You are SysArch-RAG, a principal cloud infrastructure engineer. Answer strictly from context. "
        "Cite sources inline as `[file_path :: section]`. Preserve exact YAML/CLI flags."
    )
    ai_msg = await _synthesis_llm().ainvoke(
        [SystemMessage(content=sys_msg), HumanMessage(content=f"Question: {state['query']}\n\nContext:\n{formatted_ctx}")]
    )
    return {"final_answer": str(ai_msg.content), "citations": citations}
