<div align="center">

# SysArch-RAG

**Cyclic Corrective RAG & Hybrid Retrieval Engine for Systems Architecture**

[![CI - Tests](https://github.com)](https://github.com)

![Python](https://shields.io)
![FastAPI](https://shields.io)
![LangGraph](https://shields.io)
![Qdrant](https://shields.io)
![Next.js](https://shields.io)

[Live demo](https://sys-arch-rag.vercel.app) · [API docs](https://sysarch-rag-production.up.railway.app/docs) · [System architecture](#system-architecture) · [Tech stack](#tech-stack) · [Quickstart](#quickstart)

</div>

> Production-grade agentic RAG system featuring two-tier semantic caching, Qdrant hybrid RRF retrieval (`BAAI/bge-base-en-v1.5` + sparse BM25 IDF), Cohere cross-encoder reranking, and a self-correcting LangGraph CRAG loop with verified `[S1]..[S3]` citation extraction.

---

## System Architecture

```text
User Query
  │
  ▼
[FastAPI (/v1/query & /v1/query/stream SSE)]
  ├── Tier 1 Cache: Redis Exact-Hash L1 (corpus-version & model scoped)
  └── Tier 2 Cache: Qdrant Semantic L2 (cosine >= 0.97 + CLI/version/kind guard)
        │
        ▼ (Cache Miss)
  [LangGraph Cyclic CRAG Engine]
    ├── 1. Query Analyzer: Extracts target service filters & search sub-queries
    ├── 2. Hybrid Retriever: Qdrant RRF Fusion (BAAI/bge-base-en-v1.5 + BM25 IDF) -> Top 25
    ├── 3. Cross-Encoder Reranker: Cohere rerank-v3.5 -> Gap-Pruned Top 3 (threshold >= 0.65)
    ├── 4. Relevance Grader: Score-band fast path + structured LLM sufficiency check
    │      └── [Insufficient & retries < 2] ──▶ 5. Query Rewriter (anti-duplicate expansion) ──┐
    │                                                                                        │
    └── 6. Grounded Synthesizer: XML-delimited context + verified [S1]..[S3] citation extraction
```

---

## Key Engineering Decisions

* **Structure-Aware, Fence-Safe Ingestion (`src/ingestion/chunker.py`)**: Resolves Hugo `code sample` shortcodes and AsciiDoc code blocks prior to splitting on Markdown headers (`#`–`###`). Keeps YAML manifests and CLI blocks intact under a 1,500-character ceiling so embeddings never truncate mid-manifest.
* **Zero-Downtime Atomic Indexing (`src/ingestion/indexer.py`)**: Builds versioned Qdrant collections (`sysarch_docs_<timestamp>`) with `Modifier.IDF` enabled on sparse BM25 vectors, verifies point counts, and atomically swaps the `sysarch_docs_live` production alias.
* **Two-Tier Cache with Metadata Guardrails (`src/api/cache.py`)**: Eliminates redundant LLM latency via an L1 Upstash Redis exact-match hash and an L2 Qdrant semantic cache gated by strict CLI/version/kind metadata matching so queries for different versions never collide.
* **Self-Correcting CRAG Loop (`src/graph/workflow.py`)**: Evaluates retrieved chunks via a fast score-band check and structured LLM sufficiency grader. If evidence is insufficient, an anti-duplicate query rewriter reformulates sub-queries up to 2 retries before grounded synthesis.
* **Strict Citation Grounding (`src/graph/nodes.py`)**: Wraps retrieved passages in XML delimiters and validates every `[S1]..[S3]` inline citation against the active context window to prevent hallucinated references.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Agentic Orchestration** | `LangGraph` (Cyclic CRAG State Machine), `LangChain`, `LangSmith`, `Langfuse` |
| **LLM & Fine-Tuning** | `Groq` (`openai/gpt-oss-120b` synthesis, `openai/gpt-oss-20b` grader), `QLoRA` (`Qwen2.5-3B-Instruct`, `PEFT` / `TRL`) |
| **Vector DB & Retrieval** | `Qdrant Cloud` (Hybrid RRF: `BAAI/bge-base-en-v1.5` 768d + `Qdrant/bm25` IDF), `Cohere` (`rerank-v3.5`) |
| **Caching & Backend API** | `FastAPI` (Typed SSE Streaming), `Upstash Redis` (L1 Exact Cache + Rate Limiter), `Pydantic v2` |
| **Frontend Console** | `Next.js 16`, `React 19`, `TypeScript`, `Tailwind CSS` (Deployed on Vercel) |
| **Evaluation & DevOps** | `RAGAS` (`Faithfulness`, `Context Recall`, `Context Precision`), `Pytest` (25/25), `Docker`, `GitHub Actions CI`, `Railway` |

---

## Project Structure

```text
sysarch-rag/
├── .github/workflows/
│   └── eval-gate.yml          # Automated pytest & retrieval CI gate
├── scripts/
│   └── fetch_docs.py          # Corpus ingestion & documentation crawler
├── src/
│   ├── api/
│   │   ├── cache.py           # L1 Redis & L2 Qdrant semantic cache
│   │   └── main.py            # FastAPI app, /v1/query & /v1/query/stream SSE
│   ├── core/
│   │   ├── config.py          # Pydantic Settings & environment validation
│   │   ├── observability.py   # LangSmith / Langfuse tracing hooks
│   │   └── schemas.py         # Typed request, response & citation models
│   ├── evaluation/
│   │   └── run_ragas.py       # Offline RAGAS faithfulness & recall benchmark
│   ├── graph/
│   │   ├── nodes.py           # Analyzer, retriever, grader, rewriter & synthesizer
│   │   ├── state.py           # LangGraph TypedDict state definition
│   │   └── workflow.py        # Cyclic CRAG state machine & conditional edges
│   ├── ingestion/
│   │   ├── chunker.py         # Fence-safe Markdown & code-block splitter
│   │   └── indexer.py         # Atomic Qdrant collection & alias swapper
│   └── retrieval/
│       ├── hybrid_search.py   # Dense + sparse BM25 RRF fusion query
│       └── reranker.py        # Cohere rerank-v3.5 & gap-pruning filter
├── tests/
│   ├── test_graph.py          # CRAG routing, retry limits & synthesis tests
│   ├── test_ingestion.py      # Chunker fence integrity & metadata tests
│   └── test_retrieval.py      # Hybrid search & reranker threshold tests
├── Dockerfile                 # Multi-stage production container
├── docker-compose.yml         # Local Qdrant + Redis + API stack
└── pyproject.toml             # uv dependency & tool configuration
```

---

## Quickstart

### 1. Prerequisites
* Python `3.11+` and [`uv`](https://astral.sh)
* API keys for **Groq**, **Qdrant Cloud**, **Cohere**, and **Upstash Redis**

### 2. Installation
```bash
git clone https://github.com
cd SysArch-RAG
uv sync --all-extras
cp .env.example .env
```

### 3. Ingest Corpus & Build Index
```bash
uv run python scripts/fetch_docs.py
uv run python -m src.ingestion.indexer
```

### 4. Run the API Server
```bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

---

## API Reference

### `POST /v1/query`
Executes the full CRAG pipeline and returns a structured JSON response with verified citations and latency telemetry.

```bash
curl -X POST "https://railway.app" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does Kubernetes NetworkPolicy isolate pod ingress traffic?",
    "top_k": 3
  }'
```

### `POST /v1/query/stream`
Streams real-time Server-Sent Events (`status`, `sources`, `token`, `done`) for step-by-step UI progress tracking in the Next.js console.

---

## Evaluation & Testing

Run the unit and retrieval test suite:
```bash
uv run pytest -v
```

Run the RAGAS evaluation pipeline against the golden architecture dataset:
```bash
uv run python -m src.evaluation.run_ragas
```

| Metric | Target | Description |
| :--- | :--- | :--- |
| **Faithfulness** | `>= 0.90` | Claims in the generated answer strictly supported by `[S1]..[S3]` context |
| **Context Precision** | `>= 0.85` | Relevant chunks ranked at the top after Cohere `rerank-v3.5` gap pruning |
| **Context Recall** | `>= 0.85` | Ground-truth architectural facts captured by Hybrid RRF retrieval |
| **Unit & Graph Suite** | `25 / 25` | Automated CI gate covering chunking, RRF retrieval, caching, and CRAG edges |

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
