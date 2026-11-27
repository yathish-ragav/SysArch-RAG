FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.6.6 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
RUN uv sync --frozen --no-dev
ENV FASTEMBED_CACHE_PATH=/opt/fastembed_cache
RUN /app/.venv/bin/python -c "from fastembed import TextEmbedding, SparseTextEmbedding; TextEmbedding('BAAI/bge-base-en-v1.5'); SparseTextEmbedding('Qdrant/bm25')"

FROM python:3.11-slim AS runtime
WORKDIR /app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/fastembed_cache /opt/fastembed_cache
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH" FASTEMBED_CACHE_PATH=/opt/fastembed_cache PYTHONUNBUFFERED=1
RUN chown -R appuser:appgroup /app /opt/fastembed_cache
USER appuser
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
