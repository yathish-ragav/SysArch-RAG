import os
from functools import lru_cache
from typing import Any

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

from src.core.config import settings


def configure_observability_env() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
    os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
    os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse:
    configure_observability_env()
    return Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
    )


def get_langfuse_run_config(
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    get_langfuse_client()
    handler = LangfuseCallbackHandler(public_key=settings.LANGFUSE_PUBLIC_KEY)
    metadata: dict[str, Any] = {"langfuse_tags": tags or ["sysarch-rag"]}
    if session_id:
        metadata["langfuse_session_id"] = session_id
    return {"callbacks": [handler], "metadata": metadata}


def flush_langfuse() -> None:
    get_langfuse_client().flush()
