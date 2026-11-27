<div align="center">

# SysArch-RAG: Cyclic Corrective RAG & Hybrid Retrieval Engine for Systems Architecture

*Grounded Cloud & Distributed Systems Intelligence for Kubernetes, AWS EKS, and etcd*

[![CI - Tests](https://github.com)](https://github.com)

![Demo](https://shields.io) ![LangGraph](https://shields.io)
![Qdrant](https://shields.io) ![Cohere](https://shields.io)

[Live demo](https://vercel.app) · [API docs](https://railway.app) · [System architecture](#system-architecture) · [Tech stack](#tech-stack) · [Quickstart](#quickstart)

</div>

Grounded Cloud & Distributed Systems Intelligence for **Kubernetes**, **AWS EKS**, and **etcd**—combining Qdrant Hybrid RRF retrieval, Cohere `rerank-v3.5` cross-encoder pruning, two-tier semantic caching, and a self-correcting LangGraph CRAG engine with verified `[S1]..[S3]` citation grounding.

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
| **Agentic Orchestration** | LangGraph (Cyclic CRAG State Machine), LangChain, LangSmith, Langfuse |
| **LLM & Fine-Tuning** | Groq (`openai/gpt-oss-120b` synthesis, `openai/gpt-oss-20b` grader), QLoRA (`Qwen2.5-3B-Instruct`, `PEFT` / `TRL`) |
| **Vector DB & Retrieval** | Qdrant Cloud (Hybrid RRF: `BAAI/bge-base-en-v1.5` 768d + `Qdrant/bm25` IDF), Cohere `rerank-v3.5` |
| **Caching & Backend API** | FastAPI (Typed SSE Streaming), Upstash Redis (L1 Exact Cache + Rate Limiter), Pydantic v2 |
| **Frontend Console** | Next.js 16, React 19, TypeScript, Tailwind CSS (Deployed on Vercel) |
| **Evaluation & DevOps** | RAGAS (`Faithfulness`, `Context Recall`, `Context Precision`), Pytest (25/25), Docker, GitHub Actions CI, Railway |

---

## Quickstart

```bash
git clone https://github.com
cd SysArch-RAG
uv sync --frozen
cp .env.example .env

# Run unit & retrieval test suite
uv run pytest -v

# Start FastAPI server
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
