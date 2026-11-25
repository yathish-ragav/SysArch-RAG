# SysArch-RAG — Master Architecture & Execution Blueprint

## 1. Project Identity & Schedule
- **Project Name:** SysArch-RAG: Self-Correcting Cloud & Distributed Systems Intelligence Engine
- **Repository:** `sysarch-rag`
- **Schedule:** 5-Day Intensive Production Sprint (7–8 hours/day, ~35–40 hours total), starting Nov 25, 2026.
- **Role Split:** User is the **Lead Systems Architect** (defines boundaries, verifies traces, audits quality). AI is the **Senior AI Engineer & Code Verifier** (writes modular, type-safe, async Python/TypeScript code and flags edge-case failures).

## 2. Hardware & Environment Constraints (CRITICAL)
- **OS & Machine:** Ubuntu Linux on Asus Vivobook 16 (**CPU-only, no dedicated GPU**).
- **Mandatory Adaptations:**
  - **Embeddings:** Use `fastembed` (lightweight ONNX CPU runtime) for `BAAI/bge-base-en-v1.5` (dense) and `Qdrant/bm25` (sparse). Never load heavy PyTorch GPU models locally.
  - **Reranking:** Use **Cohere Rerank API (`rerank-v3.5`)** via `COHERE_API_KEY` instead of local GPU cross-encoders.
  - **LLM Synthesis:** Use **Groq API (`llama-3.3-70b-versatile`)** via `GROQ_API_KEY` for fast cloud inference.
  - **Fine-Tuning:** Run **Unsloth (QLoRA)** on **Google Colab free T4 GPU** on Day 3 to fine-tune `Qwen-2.5-Coder-7B` for relevance grading & JSON citations.

## 3. Complete Production Stack & Credentials Ready
- **Orchestration:** `LangGraph` (cyclic Corrective RAG state machine) + `LangChain` (Markdown/AST splitters).
- **Vector DB & Cache:** `Qdrant` (local Docker port `6333` for dev -> Qdrant Cloud for prod) + `Redis` (local Docker port `6379` for dev -> Upstash Redis for prod).
- **Dual Observability:**
  - **LangSmith** (`LANGCHAIN_API_KEY`, endpoint `https://langchain.com`) — traces LangGraph node transitions & hosts golden evaluation datasets.
  - **Langfuse Cloud** (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, host `https://langfuse.com`) — tracks p95 latency, token cost, and user feedback (cloud tier used to save ~1.5GB local RAM).
- **Evaluation ("Checking"):** `Ragas` + `DeepEval` enforced in GitHub Actions CI/CD (`Faithfulness >= 0.95`, `Context Recall >= 0.90`).
- **Backend & Frontend:** `FastAPI` (async SSE streaming + Redis semantic cache) + `Next.js 14` (or `Chainlit`) deployed on Render/Railway + Vercel.

## 4. Knowledge Base & Structure-Aware Ingestion
- **Sources (~400–600 official `.md` files via `git sparse-checkout`):**
  1. `kubernetes/website` (`content/en/docs/concepts` & `tasks`) -> `{service: "kubernetes", version: "1.31"}`
  2. `awsdocs/amazon-eks-user-guide` (`doc_source`) -> `{service: "aws-eks", version: "latest"}`
  3. `etcd-io/etcd` (`Documentation`) & `cncf/toc` -> `{service: "etcd-cncf", version: "v3.5"}`
- **Chunking Rules:** Split strictly on Markdown headers (`#`, `##`, `###`). Keep fenced code blocks (` ```yaml `, ` ```bash `, ` ```python `) 100% atomic—never split mid-block. Upsert both dense and BM25 sparse vectors to Qdrant with payload metadata indexes.

## 5. 7-Node LangGraph Corrective RAG (CRAG) Pipeline
1. `analyze_query`: Extracts metadata filters (`service`, `version`) and decomposes multi-part questions (Pydantic v2 structured output).
2. `hybrid_retrieve`: Queries Qdrant using Reciprocal Rank Fusion (RRF) across dense + sparse vectors (returns top 25 chunks).
3. `rerank_documents`: Calls Cohere `rerank-v3.5` to score top 25 chunks down to top 4 (`score >= 0.65`).
4. `grade_documents`: Evaluates if top 4 chunks sufficiently answer the query.
   - **Conditional Edge:** If insufficient AND `retry_count < 2` -> route to `rewrite_query` (broaden terms, relax filters, increment `retry_count`) -> loop back to `hybrid_retrieve`.
   - If insufficient AND `retry_count >= 2` -> route to `fallback_refusal` (deterministic refusal; zero hallucination).
5. `generate_answer`: Synthesizes grounded response with mandatory inline citations `[file_path § section]`.
6. `verify_hallucination`: Audits final output against retrieved chunks before streaming.

## 6. 5-Day Execution Roadmap
- **Day 1:** Folder & GitHub setup + Docker (`Qdrant` + `Redis`) + `fetch_docs.sh` + structure-aware Qdrant hybrid ingestion.
- **Day 2:** Cohere reranker + 7-node cyclic LangGraph pipeline + LangSmith & Langfuse dual tracing.
- **Day 3:** 50-question golden dataset + baseline Ragas metrics + Unsloth QLoRA fine-tuning on Colab + post-tuning Ragas comparison.
- **Day 4:** FastAPI streaming server + Redis semantic cache (`<80ms` repeat hits) + prompt-injection guardrails + GitHub Actions Ragas CI/CD gate.
- **Day 5:** Frontend chat UI with live agent reasoning trace + cloud deployment (Qdrant Cloud, Upstash, Render, Vercel) + README benchmarks.
