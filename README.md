# SysArch-RAG

> **Live Production Console:** [https://sys-arch-rag.vercel.app](https://sys-arch-rag.vercel.app)  
> **Live FastAPI Backend & Swagger Docs:** [https://railway.app](https://railway.app)


**Self-Correcting Corrective RAG (CRAG) Engine for Cloud & Distributed Systems Documentation**

SysArch-RAG is a production-grade retrieval and reasoning engine covering **Kubernetes (`v1.32`)**, **AWS EKS**, and **etcd (`v3.5`)** operator documentation. Unlike single-pass cosine-similarity chatbots, it routes queries across service namespaces, fuses dense semantic vectors with BM25 IDF lexical search, cross-encoder reranks candidates, grades context sufficiency in a cyclic **LangGraph** state machine, and enforces deterministic inline source citations (`[S1]`–`[S3]`).

---

## System Architecture

```text
User Query
   │
   ▼
[FastAPI (/v1/query & /v1/query/stream SSE)]
   ├──► Tier 1 Cache: Redis Exact-Hash L1 (corpus-version & model scoped)
   ├──► Tier 2 Cache: Qdrant Semantic L2 (cosine >= 0.97 + CLI/version/Kind guard)
   │
   ▼ (Cache Miss)
[LangGraph Cyclic CRAG Engine]
   ├──► 1. Query Analyzer: Extracts target service filters & search sub-queries
   ├──► 2. Hybrid Retriever: Qdrant RRF Fusion (BAAI/bge-base-en-v1.5 + BM25 IDF) -> Top 25
   ├──► 3. Cross-Encoder Reranker: Cohere rerank-v3.5 -> Gap-Pruned Top 3 (threshold >= 0.65)
   ├──► 4. Relevance Grader: Score-band fast path + structured LLM sufficiency check
   │       ├── [Insufficient & retries < 2] ──► 5. Query Rewriter (anti-duplicate expansion) ──┐
   │       └──────────────────────────────────────────────────────────────────────────────────┘
   └──► 6. Grounded Synthesizer: XML-delimited context + verified [S1]..[S3] citation extraction
```

---

## Key Engineering Decisions

- **Structure-Aware, Fence-Safe Ingestion (`src/ingestion/chunker.py`):** Resolves Hugo `code_sample` shortcodes and AsciiDoc code blocks prior to splitting on Markdown headers (`#`–`###`). Keeps YAML manifests and CLI blocks intact under a 1,500-character ceiling so embeddings never truncate mid-manifest.
- **Zero-Downtime Atomic Indexing (`src/ingestion/indexer.py`):** Builds versioned Qdrant collections (`sysarch_docs_<timestamp>`) with `Modifier.IDF` enabled on sparse BM25 vectors, verifies point counts, and atomically swaps the `sysarch_docs` collection alias.
- **CPU-Optimized Local Embeddings (`src/retrieval/hybrid_search.py`):** Runs `BAAI/bge-base-en-v1.5` (768-dim) and `Qdrant/bm25` via ONNX Runtime (`fastembed`) off the FastAPI event loop using `asyncio.to_thread` and bounded semaphores—eliminating PyTorch/CUDA bloat on CPU-only hosts.
- **Hallucination-Resistant Citation Grounding (`src/graph/nodes.py`):** Filters navigation boilerplate (`whatsnext`), applies adaptive rerank score-gap pruning (`top_score - 0.30`, capped at 3 chunks), extracts cited handles (`[S1]`–`[S3]`) post-generation, maps them deterministically back to Qdrant metadata, and triggers a structured refusal if zero valid citations survive.
- **Two-Tier Cache with Technical Signature Guard (`src/api/cache.py`):** Combines sub-millisecond Redis exact lookup with Qdrant vector cache (`cosine >= 0.97`) while verifying that version numbers, `--flags`, and `CamelCase` resource kinds match before serving a semantic hit.
- **Domain-Adapted QLoRA Relevance Grader (`fine_tuning/`):** Fine-tuned `Qwen/Qwen2.5-3B-Instruct` with 4-bit NF4 QLoRA (`r=16, alpha=32`) on 360 synthetic positive and hard-negative Kubernetes/EKS/etcd retrieval pairs.

---

## Indexed Corpus & Benchmarks

### 1. Knowledge Base (`11,282` Hybrid IDF Chunks)

| Service Namespace | Source Repository | Markdown Files | Indexed Hybrid Chunks |
| :--- | :--- | :---: | :---: |
| `kubernetes` (`v1.32`) | `kubernetes/website` (`concepts/` + `tasks/` + resolved `examples/`) | 360 | 4,305 |
| `aws-eks` (`latest`) | `awsdocs/amazon-eks-user-guide` (`latest/ug/*.adoc` -> `.md`) | 507 | 6,199 |
| `etcd-cncf` (`v3.5`) | `etcd-io/website` (`content/en/docs/v3.5`) | 91 | 778 |
| **Total** | | **958** | **11,282** |

### 2. Runtime Latency & QLoRA Grader Benchmark

| Metric / Benchmark | Result |
| :--- | :--- |
| **L1 Exact Cache Hit (Redis)** | ~34 ms end-to-end |
| **L2 Semantic Cache Hit (Qdrant + Signature Guard)** | ~28 ms end-to-end |
| **Cold CRAG Pipeline (Embed + RRF + Rerank + Grade + Synthesis)** | ~2.7–5.0 s |
| **Base `Qwen2.5-3B-Instruct` Grader (54 held-out val cases)** | `98.15%` valid JSON \| `64.81%` accuracy |
| **QLoRA Fine-Tuned `Qwen2.5-3B-Instruct` Grader (54 val cases)** | **`100.00%` valid JSON \| `88.89%` accuracy (`+24.08%`)** |

---


### 3. End-to-End RAGAS Quality Gate (`data/eval_report.json` — 15 Golden Cases)

| Evaluation Metric | Score | Notes |
| :--- | :---: | :--- |
| **Refusal Accuracy** | **1.0000 (100%)** | `15/15` cases (`2/2` out-of-corpus blocked, `13/13` valid answered) |
| **Context Recall** | **0.8462** | `11/13` answerable cases achieved `1.00` recall |
| **Faithfulness** | **0.8385** | Evaluated with `openai/gpt-oss-20b` judge |
| **Context Precision** | **0.8141** | Hybrid RRF + Cohere `rerank-v3.5` |

## Project Structure

```text
sysarch-rag/
├── .github/workflows/
│   └── eval-gate.yml          # GitHub Actions CI offline pytest + Ragas quality gate
├── data/
│   └── golden_eval_set.json   # Multi-service & out-of-corpus golden evaluation cases
├── fine_tuning/
│   ├── generate_dataset.py    # Positive & hard-negative grading dataset builder
│   ├── train_qlora.py         # 4-bit NF4 QLoRA fine-tuning & validation script
│   └── benchmark_results.json # Base vs. QLoRA accuracy & JSON validity metrics
├── scripts/
│   └── fetch_docs.py          # Sparse-checkout & AsciiDoc/Hugo normalizer
├── src/
│   ├── core/
│   │   ├── config.py          # Pydantic v2 environment settings
│   │   ├── observability.py   # LangSmith + Langfuse singleton tracing
│   │   └── schemas.py         # Strict request, state, grading & citation models
│   ├── ingestion/
│   │   ├── chunker.py         # Frontmatter parser & fence-safe Markdown splitter
│   │   └── indexer.py         # Dense + BM25 IDF vectorizer & atomic alias swapper
│   ├── retrieval/
│   │   ├── hybrid_search.py   # Async thread-offloaded FastEmbed + Qdrant RRF search
│   │   └── reranker.py        # Rate-paced & cached Cohere AsyncClientV2 reranker
│   ├── graph/
│   │   ├── state.py           # LangGraph TypedDict state definition
│   │   ├── nodes.py           # Analyzer, retriever, grader, rewriter & synthesizer
│   │   └── workflow.py        # Compiled cyclic CRAG state machine
│   ├── evaluation/
│   │   └── run_ragas.py       # Rate-paced Ragas quality gate & --judge-only runner
│   └── api/
│       ├── cache.py           # Redis L1 + Qdrant L2 semantic cache
│       └── main.py            # FastAPI lifespan singletons, /readyz, /v1/query & SSE
├── tests/                     # Offline pytest unit test suite (ingestion, retrieval, graph)
├── docker-compose.yml         # Local Qdrant & Redis infrastructure
├── .env.example               # Required environment variable template
└── pyproject.toml             # Managed with uv
```

---

## Quickstart

### 1. Start Infrastructure & Configure Environment
```bash
docker compose up -d
cp .env.example .env   # Fill in GROQ_API_KEY, COHERE_API_KEY, LANGCHAIN_API_KEY, LANGFUSE_* keys
uv sync
```

### 2. Fetch Documentation & Build Hybrid Index
```bash
uv run python scripts/fetch_docs.py
uv run python -m src.ingestion.indexer
```

### 3. Run Offline Tests & FastAPI Server
```bash
uv run pytest -v
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Query the API
```bash
curl -X POST http://localhost:8000/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do PodDisruptionBudgets protect workloads during a voluntary node drain?"}'
```

---

## Build Status

- [x] **Modules 1–3:** Dual LLMOps (`LangSmith` + `Langfuse`), structure-aware ingestion (`11,282` hybrid IDF chunks), Qdrant RRF + Cohere `rerank-v3.5`
- [x] **Modules 4–5:** Cyclic LangGraph CRAG workflow, grounded `[S1]`–`[S3]` citations, FastAPI SSE + Two-Tier Redis/Qdrant Semantic Cache
- [x] **Module 6 (CI & Golden Eval):** Offline `pytest` suite + GitHub Actions CI gate (`eval-gate.yml`) + 15-case golden evaluation pipeline
- [x] **Module 7 (QLoRA Fine-Tuning):** `Qwen2.5-3B-Instruct` 4-bit QLoRA domain grader (`64.81% -> 88.89%` accuracy, `100%` valid JSON)
- [ ] **Module 8 (Frontend & Cloud Deploy):** Next.js TypeScript + Tailwind UI with live SSE CRAG trace inspector & cloud deployment
