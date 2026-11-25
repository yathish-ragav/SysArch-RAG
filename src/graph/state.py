from typing import TypedDict
from src.core.schemas import Citation, GradeResult, ScoredChunk


class GraphState(TypedDict, total=False):
    query: str
    active_query: str
    sub_queries: list[str]
    services: list[str]
    tried_queries: list[str]
    reranked_chunks: list[ScoredChunk]
    grade_result: GradeResult
    retry_count: int
    analysis_fallback: bool
    refused: bool
    error: str | None
    final_answer: str
    citations: list[Citation]
