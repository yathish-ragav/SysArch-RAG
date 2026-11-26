from typing import TypedDict
from src.core.schemas import Citation, GradeResult, RefusalReason, ScoredChunk


class GraphState(TypedDict, total=False):
    query: str
    active_query: str
    sub_queries: list[str]
    services: list[str]
    service_locked: bool
    tried_queries: list[str]
    reranked_chunks: list[ScoredChunk]
    grade_result: GradeResult
    retry_count: int
    rewrite_exhausted: bool
    analysis_fallback: bool
    refused: bool
    refusal_reason: RefusalReason | None
    error: str | None
    final_answer: str
    citations: list[Citation]
