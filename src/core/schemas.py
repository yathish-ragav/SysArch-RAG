from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ServiceName = Literal["kubernetes", "aws-eks", "etcd-cncf"]
DetailLevel = Literal["brief", "standard", "detailed"]
RefusalReason = Literal["insufficient_context", "service_error"]

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
    service: str = "unknown"
    version: str = "unknown"
    file_path: str
    section: str = ""
    section_hierarchy: str = ""
    source_url: str = ""
    snippet: str
    full_text: str = ""

    @model_validator(mode="after")
    def _sync_sections(self) -> "Citation":
        if not self.section_hierarchy and self.section:
            self.section_hierarchy = self.section
        if not self.section and self.section_hierarchy:
            self.section = self.section_hierarchy
        if not self.full_text and self.snippet:
            self.full_text = self.snippet
        return self


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1500)
    service_filter: ServiceName | None = None
    detail_level: DetailLevel = "standard"
    session_id: str | None = Field(default=None, pattern=r"^[\w-]{1,64}$")

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v


class TraceStep(BaseModel):
    node: str
    status: Literal["ok", "fallback", "retry", "error"] = "ok"
    duration_ms: float = Field(default=0.0)
    summary: str = Field(default="")


class RAGResponse(BaseModel):
    request_id: str = Field(default="")
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    services_searched: list[str] = Field(default_factory=list)
    retry_count: int = 0
    refused: bool = False
    refusal_reason: RefusalReason | None = None
    cache_hit: bool = False
    cache_tier: Literal["exact_redis", "semantic_qdrant", "live_crag"] = "live_crag"
    latency_ms: float = 0.0
    original_latency_ms: float | None = None
    trace: list[TraceStep] = Field(default_factory=list)
