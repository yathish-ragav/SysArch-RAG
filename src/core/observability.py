import asyncio
import os
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from qdrant_client import AsyncQdrantClient
import redis.asyncio as redis
from src.core.config import settings


def configure_langsmith_env() -> None:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT


def get_observability_callbacks(
    session_id: str | None = None,
    tags: list[str] | None = None,
) -> list[LangfuseCallbackHandler]:
    configure_langsmith_env()
    langfuse_handler = LangfuseCallbackHandler(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        session_id=session_id,
        tags=tags or ["sysarch-rag"],
    )
    return [langfuse_handler]


async def verify_connections() -> dict[str, str]:
    status: dict[str, str] = {}

    # 1. Verify Qdrant
    qdrant = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    try:
        collections = await qdrant.get_collections()
        status["qdrant"] = f"OK ({len(collections.collections)} collections)"
    except Exception as exc:
        status["qdrant"] = f"ERROR: {exc}"
    finally:
        await qdrant.close()

    # 2. Verify Redis
    redis_client = redis.from_url(settings.REDIS_URL)
    try:
        pong = await redis_client.ping()
        status["redis"] = f"OK (ping={pong})"
    except Exception as exc:
        status["redis"] = f"ERROR: {exc}"
    finally:
        await redis_client.aclose()

    # 3. Verify Langfuse Cloud
    try:
        lf = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        auth_ok = lf.auth_check()
        status["langfuse"] = f"OK (auth={auth_ok})"
    except Exception as exc:
        status["langfuse"] = f"ERROR: {exc}"

    # 4. Verify LangSmith Env
    configure_langsmith_env()
    status["langsmith"] = (
        f"OK (project={settings.LANGCHAIN_PROJECT})"
        if settings.LANGCHAIN_API_KEY.startswith("lsv2_")
        else "WARNING: Check LANGCHAIN_API_KEY format"
    )

    return status


if __name__ == "__main__":
    results = asyncio.run(verify_connections())
    for service, state in results.items():
        print(f"{service.upper()}: {state}")
