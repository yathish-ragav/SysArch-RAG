from typing import Literal
from pydantic import BaseModel, Field


class DocMetadata(BaseModel):
    service: Literal["kubernetes", "aws-eks", "etcd-cncf"]
    version: str
    file_path: str
    section_hierarchy: str
    has_code_block: bool = False


class DocumentChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: DocMetadata


class ScoredChunk(BaseModel):
    chunk: DocumentChunk
    hybrid_score: float
    rerank_score: float | None = None


class QueryAnalysis(BaseModel):
    original_query: str
    sub_queries: list[str] = Field(default_factory=list)
    service_filter: Literal["kubernetes", "aws-eks", "etcd-cncf"] | None = None
    version_filter: str | None = None


class GradeResult(BaseModel):
    is_sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class Citation(BaseModel):
    file_path: str
    section: str
    snippet: str


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retry_count: int = 0
    cache_hit: bool = False
    latency_ms: float = 0.0
