from typing import Literal, TypedDict
from src.core.schemas import Citation, GradeResult, ScoredChunk


class GraphState(TypedDict, total=False):
    query: str
    active_query: str
    sub_queries: list[str]
    service_filter: Literal["kubernetes", "aws-eks", "etcd-cncf"] | None
    version_filter: str | None
    reranked_chunks: list[ScoredChunk]
    grade_result: GradeResult
    retry_count: int
    final_answer: str
    citations: list[Citation]
