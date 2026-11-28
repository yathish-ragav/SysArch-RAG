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

    # API hardening
    ENV: str = "development"
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"
    EXPOSE_DOCS_IN_PROD: bool = True
    MAX_REQUEST_BYTES: int = 16384
    TRUSTED_PROXY_HOPS: int = 1
    RATE_LIMIT_PER_MIN: int = 12
    USER_RATE_LIMIT_PER_MIN: int = 15
    DAILY_QUERY_CAP: int = 400

    # LangSmith Observability
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://langchain.com"
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str = "sysarch-rag"

    # Langfuse Cloud Observability
    LANGFUSE_PUBLIC_KEY: str
    LANGFUSE_SECRET_KEY: str
    LANGFUSE_HOST: str = "https://langfuse.com"

    # Model & Retrieval Thresholds
    DENSE_MODEL: str = "BAAI/bge-base-en-v1.5"
    SPARSE_MODEL: str = "Qdrant/bm25"
    RERANK_MODEL: str = "rerank-v3.5"
    SYNTHESIS_MODEL: str = "openai/gpt-oss-120b"
    GRADER_MODEL: str = "openai/gpt-oss-20b"
    RERANK_SCORE_THRESHOLD: float = 0.65
    MAX_RETRIES: int = 2


settings = Settings()
