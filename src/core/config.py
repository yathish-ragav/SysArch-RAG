from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM & Reranking APIs
    GROQ_API_KEY: str
    COHERE_API_KEY: str

    # Local / Cloud Vector DB & Cache
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "sysarch_docs"
    REDIS_URL: str = "redis://localhost:6379"

    # API hardening (all overridable via .env / Render env vars)
    # CORS_ORIGINS: comma-separated browser origins allowed to call the API. In production set this to
    # your Vercel URL, e.g. "https://sysarch-rag.vercel.app". Never use "*" on a public endpoint.
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Per-IP fixed window (60s) limit on /v1/query and /v1/query/stream.
    RATE_LIMIT_PER_MIN: int = 12
    # Global UTC-day budget of full pipeline runs (cache misses) to protect Groq/Cohere free-tier quotas.
    DAILY_QUERY_CAP: int = 400

    # LangSmith Observability
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str = "sysarch-rag"

    # Langfuse Cloud Observability
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://us.cloud.langfuse.com"

    # Model & Retrieval Thresholds
    DENSE_MODEL: str = "BAAI/bge-base-en-v1.5"
    SPARSE_MODEL: str = "Qdrant/bm25"
    RERANK_MODEL: str = "rerank-v3.5"
    SYNTHESIS_MODEL: str = "openai/gpt-oss-120b"
    GRADER_MODEL: str = "openai/gpt-oss-20b"
    RERANK_SCORE_THRESHOLD: float = 0.65
    MAX_RETRIES: int = 2


settings = Settings()
