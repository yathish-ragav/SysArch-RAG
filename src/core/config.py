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
    SYNTHESIS_MODEL: str = "llama-3.3-70b-versatile"
    GRADER_MODEL: str = "llama-3.1-8b-instant"
    RERANK_SCORE_THRESHOLD: float = 0.65
    MAX_RETRIES: int = 2


settings = Settings()
