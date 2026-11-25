from typing import Literal
from pydantic import BaseModel, Field, field_validator

ServiceName = Literal["kubernetes", "aws-eks", "etcd-cncf"]

SERVICE_ALIASES: dict[str, ServiceName] = {
    "kubernetes": "kubernetes",
    "k8s": "kubernetes",
    "aws-eks": "aws-eks",
    "eks": "aws-eks",
    "amazon-eks": "aws-eks",
    "etcd-cncf": "etcd-cncf",
    "etcd": "etcd-cncf",
    "cncf": "etcd-cncf",
}


class DocMetadata(BaseModel):
    service: str
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


class QueryAnalysisOut(BaseModel):
    services: list[str] = Field(default_factory=list)
    sub_queries: list[str] = Field(default_factory=list)

    @field_validator("services", mode="before")
    @classmethod
    def normalize_services(cls, v: object) -> list[str]:
        if not isinstance(v, list):
            return []
        out: list[str] = []
        for item in v:
            key = str(item).strip().lower()
            if key in SERVICE_ALIASES and SERVICE_ALIASES[key] not in out:
                out.append(SERVICE_ALIASES[key])
        return out


class GradeResult(BaseModel):
    is_sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        val = float(v) if isinstance(v, (int, float, str)) else 0.0
        if val > 1.0:
            val = val / 100.0 if val <= 100.0 else 1.0
        return max(0.0, min(1.0, val))


class Citation(BaseModel):
    ref_id: str
    file_path: str
    section: str
    snippet: str


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1500)
    service_filter: ServiceName | None = None
    session_id: str | None = None


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    services_searched: list[str] = Field(default_factory=list)
    retry_count: int = 0
    refused: bool = False
    cache_hit: bool = False
    latency_ms: float = 0.0
