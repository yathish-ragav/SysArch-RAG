<div align="center">

# SysArch-RAG: Cyclic Corrective RAG & Hybrid Retrieval Engine for Systems Architecture

*Grounded Cloud & Distributed Systems Intelligence for Kubernetes, AWS EKS, and etcd*

[![CI - Tests](https://img.shields.io/badge/CI_--_Tests-passing-2ea44f?style=flat&logo=github&logoColor=white)](https://://github.com)

![Demo](https://img.shields.io/badge/DEMO-LIVE-2ea44f?style=for-the-badge) ![LangGraph](https://img.shields.io/badge/LANGGRAPH-CYCLIC_CRAG-1d76db?style=for-the-badge&logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/QDRANT-HYBRID_RRF-e5a50a?style=for-the-badge&logo=qdrant&logoColor=white) ![Cohere](https://img.shields.io/badge/COHERE-RERANK_V3.5-ea580c?style=for-the-badge)

[Live demo](https://sys-arch-rag.vercel.app) · [API docs](https://sysarch-rag-production.up.railway.app/docs) · [System architecture](#architecture) · [Tech stack](#technology-stack) · [Quickstart](#getting-started)

</div>


---

## Overview

SysArch-RAG is a documentation-focused question-answering system for
cloud-native and distributed-systems engineering. It combines dense and
sparse retrieval, reciprocal-rank fusion, neural reranking, evidence
grading, bounded query reformulation, citation-aware answer synthesis,
and a final grounding pass in a cyclic **Corrective RAG (CRAG)**
workflow.

The knowledge domain is deliberately scoped to three technical
documentation families:

-   **Kubernetes** --- concepts and operational tasks.
-   **Amazon EKS** --- the Amazon EKS user guide.
-   **etcd** --- v3.5 documentation.

The system exposes a FastAPI backend with both JSON and Server-Sent
Events (SSE) query interfaces, and a Next.js frontend for interactive
queries, source inspection, and pipeline trace visibility. Redis and
Qdrant support caching and retrieval infrastructure. LangSmith and
Langfuse integrations provide execution tracing.

> **Project status and evaluation transparency**
>
> The included evaluation snapshot contains 15 RAGAS cases. Its measured
> Faithfulness (0.8385), Context Recall (0.8462), and Context Precision
> (0.8141) are below the configured thresholds (0.95, 0.90, and 0.85
> respectively); Refusal Accuracy is 1.00 and meets its threshold. These
> are reported measurements, not a claim that every quality gate has
> passed. See [Evaluation](#evaluation-and-measured-results).

## Contents

-   [What it does](#what-it-does)
-   [Architecture](#architecture)
-   [Request lifecycle](#request-lifecycle)
-   [Knowledge corpus and provenance](#knowledge-corpus-and-provenance)
-   [Technology stack](#technology-stack)
-   [Repository layout](#repository-layout)
-   [Getting started](#getting-started)
-   [Environment configuration](#environment-configuration)
-   [Ingesting and indexing
    documentation](#ingesting-and-indexing-documentation)
-   [API reference](#api-reference)
-   [Caching and operational
    behavior](#caching-and-operational-behavior)
-   [Observability](#observability)
-   [Evaluation and measured results](#evaluation-and-measured-results)
-   [Fine-tuning experiment](#fine-tuning-experiment)
-   [Deployment notes](#deployment-notes)
-   [Security and operational
    considerations](#security-and-operational-considerations)
-   [Known limitations](#known-limitations)
-   [Roadmap](#roadmap)
-   [Acknowledgements](#acknowledgements)

## What it does

### Documentation-grounded technical answers

SysArch-RAG is designed to answer questions using an indexed
documentation corpus rather than relying exclusively on model parametric
knowledge. The response schema carries:

-   The generated answer.
-   Citation references and source metadata.
-   Services searched.
-   Retry count and refusal state.
-   Cache hit and cache tier.
-   End-to-end latency.
-   A list of graph trace steps, including node status, duration, and
    summary.

### Service-aware retrieval

The query contract supports an optional service filter:

  Service key    Documentation domain
  -------------- ----------------------
  `kubernetes`   Kubernetes
  `aws-eks`      Amazon EKS
  `etcd-cncf`    etcd / CNCF

The backend also normalizes aliases such as `k8s`, `eks`, `amazon-eks`,
`etcd`, and `cncf` during query analysis.

### Corrective retrieval

A first retrieval pass is not treated as automatically sufficient.
Retrieved evidence is reranked and graded. If the evidence is judged
insufficient, the graph can reformulate the query and retrieve again,
subject to a bounded retry count. If the available evidence remains
inadequate, the answer path can return a deterministic
insufficient-context refusal rather than inventing documentation
support.

### Streaming and trace visibility

The streaming endpoint emits structured SSE events. The frontend
consumes events to show pipeline activity and then renders the final
response and citations. The API also supports a non-streaming JSON
response for scripts and integrations.

## Architecture

``` mermaid
flowchart TD
    U[User / Next.js UI] --> API[FastAPI]
    API --> RL[Request validation and rate controls]
    RL --> CACHE[Redis exact cache + Qdrant semantic cache]
    CACHE -->|Cache miss| GRAPH[LangGraph CRAG workflow]
    CACHE -->|Cache hit| RESP[Cached response]
    GRAPH --> ANALYZE[Analyze query]
    ANALYZE --> EMBED[Dense + sparse query embeddings]
    EMBED --> QDRANT[Qdrant hybrid retrieval]
    QDRANT --> RERANK[Cohere reranking]
    RERANK --> GRADE[Evidence sufficiency grading]
    GRADE -->|Insufficient, retries remain| REWRITE[Rewrite query]
    REWRITE --> QDRANT
    GRADE -->|Sufficient or retry limit reached| SYNTH[Synthesize cited answer]
    SYNTH --> VERIFY[Claim/citation grounding pass]
    VERIFY --> RESP
    RESP --> API
    API --> U
    GRAPH -. traces .-> LS[LangSmith]
    GRAPH -. traces .-> LF[Langfuse]
```

### Main components

  -------------------------------------------------------------------------
  Layer                   Implementation            Responsibility
  ----------------------- ------------------------- -----------------------
  User interface          Next.js 16, React,        Query submission,
                          TypeScript                service/detail
                                                    selection, streaming
                                                    trace, citations and
                                                    response history

  API                     FastAPI, Pydantic v2,     Request validation,
                          Uvicorn                   JSON/SSE endpoints,
                                                    health/readiness, CORS,
                                                    request controls

  Orchestration           LangGraph                 Stateful corrective
                                                    retrieval and answer
                                                    workflow

  Dense embeddings        FastEmbed,                Semantic query/document
                          `BAAI/bge-base-en-v1.5`   representation

  Sparse embeddings       FastEmbed, `Qdrant/bm25`  Lexical term matching

  Vector retrieval        Qdrant                    Dense/sparse retrieval,
                                                    RRF fusion, payload
                                                    filtering, versioned
                                                    index

  Reranking               Cohere `rerank-v3.5`      Relevance scoring and
                                                    candidate refinement

  LLM inference           Groq API                  Query analysis,
                                                    evidence grading,
                                                    rewriting, answer
                                                    synthesis

  Cache                   Redis + Qdrant            Exact response cache
                                                    and semantic response
                                                    cache

  Tracing                 LangSmith + Langfuse      Graph execution and
                                                    application-level
                                                    observability

  Evaluation              RAGAS                     Faithfulness, context
                                                    recall, context
                                                    precision, refusal
                                                    accuracy
  -------------------------------------------------------------------------

### Why hybrid retrieval?

Dense retrieval is useful when a question describes a concept using
language different from the documentation. Sparse retrieval is useful
for exact technical vocabulary, API names, flags, resource kinds, error
strings, and configuration keys. SysArch-RAG retrieves candidates
through both paths and combines their rankings using **Reciprocal Rank
Fusion (RRF)** before neural reranking.

This is intended to reduce dependence on either semantic similarity or
lexical overlap alone.

## Request lifecycle

The graph implementation is composed of five LangGraph nodes. The graph
state and response schema also carry retry, refusal, citation, and trace
information.

1.  **`analyze_query`** --- interprets the question, identifies
    supported service filters, and can decompose a multi-part question
    into sub-queries using structured output.
2.  **`retrieve_and_rerank`** --- creates dense and sparse query
    vectors, performs Qdrant hybrid retrieval with RRF, deduplicates
    candidates, and applies Cohere reranking.
3.  **`grade_documents`** --- evaluates whether the retrieved evidence
    is sufficient to answer the question.
4.  **`rewrite_query`** --- when evidence is insufficient and retries
    remain, reformulates the query and returns to retrieval.
5.  **`synthesize`** --- produces a response from selected evidence,
    attaches source citations, and applies the grounding/citation
    validation pass.

The configured maximum retry count is `MAX_RETRIES=2`. The workflow is
bounded; it does not loop indefinitely.

### Retrieval details

-   Dense model: `BAAI/bge-base-en-v1.5`.
-   Sparse model: `Qdrant/bm25`.
-   Candidate retrieval: dense and sparse Qdrant prefetches are fused
    with RRF.
-   Default hybrid retrieval output: 25 candidates.
-   Prefetch depth: at least 40 candidates per retrieval path, or twice
    the requested top-k.
-   Reranker: Cohere `rerank-v3.5`.
-   Configured rerank score threshold: `0.65`.
-   The retrieval implementation applies service metadata filters when a
    service has been identified.

These are implementation defaults and configuration values, not
universal quality guarantees.

## Knowledge corpus and provenance

The ingestion script fetches official documentation repositories,
transforms source formats into Markdown where required, writes a
manifest, and stages the corpus under `data/raw_docs`.

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Corpus key     Upstream repository                                                                   Indexed source scope             Manifest files Manifest
                                                                                                                                                       revision
  -------------- ------------------------------------------------------------------------------------- ----------------------------- ----------------- -------------
  `kubernetes`   [`kubernetes/website`](https://github.com/kubernetes/website)                         `content/en/docs/concepts`,                 360 `a6b9d38`
                                                                                                       `content/en/docs/tasks`;                        
                                                                                                       ingestion script checks out                     
                                                                                                       `release-1.32`                                  

  `aws-eks`      [`awsdocs/amazon-eks-user-guide`](https://github.com/awsdocs/amazon-eks-user-guide)   `latest/ug` AsciiDoc                        507 `8d43223`
                                                                                                       converted to Markdown                           

  `etcd-cncf`    [`etcd-io/website`](https://github.com/etcd-io/website)                               `content/en/docs/v3.5`                       91 `e60635a`

  **Total**                                                                                                                                    **958** 
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------

The counts and abbreviated revisions above come from the included
`MANIFEST.json` snapshot. They describe that snapshot; they should not
be interpreted as live upstream counts.

### Source normalization

The fetcher performs source-specific transformations:

-   Kubernetes Hugo shortcode handling, including embedding referenced
    code samples where available and normalizing note/caution/warning
    blocks.
-   Amazon EKS AsciiDoc-to-Markdown conversion, including source-code
    fence conversion and heading normalization.
-   etcd Markdown ingestion from the v3.5 documentation tree.

### Chunk metadata

Each indexed chunk carries structured metadata including:

-   `service`
-   `version`
-   `file_path`
-   `section_hierarchy`
-   `has_code_block`

Chunk IDs are deterministic and derived from content and source identity
in the chunker. The configured maximum chunk size is 1,500 characters.
The chunker contains explicit handling for fenced code blocks; see
[Known limitations](#known-limitations) for the implications of
oversized blocks.

### Index structure

The indexer stores named dense and sparse vectors (`dense` and `sparse`)
in Qdrant. It creates versioned collections and uses a collection alias
for the configured active corpus. Indexing validates the number of
indexed points against the number of chunks before completing the index
build.

## Technology stack

### Backend

-   Python 3.11+
-   FastAPI and Uvicorn
-   Pydantic v2 and Pydantic Settings
-   LangChain and LangGraph
-   FastEmbed
-   Qdrant Client
-   Redis asyncio client
-   Cohere Python SDK
-   LangSmith and Langfuse integrations

### Frontend

-   Next.js 16
-   React
-   TypeScript
-   Browser `fetch` streaming with `ReadableStream` and `TextDecoder`

### Infrastructure

-   Docker Compose for local Qdrant and Redis.
-   Multi-stage Python Docker image.
-   Non-root runtime user in the backend image.
-   Embedding models are downloaded into the image build cache for
    runtime use.

## Repository layout

The source export describes the following primary modules (some
repository files may exist beyond the exported source listing):

``` text
sysarch-rag/
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI routes, controls, JSON and SSE query handling
│   │   └── cache.py             # Redis exact cache and Qdrant semantic cache
│   ├── core/
│   │   ├── config.py            # Environment-backed settings
│   │   ├── schemas.py           # Pydantic request, citation and response contracts
│   │   └── observability.py     # LangSmith / Langfuse setup
│   ├── evaluation/
│   │   └── run_ragas.py         # RAGAS evaluation runner
│   ├── graph/
│   │   ├── state.py             # Typed graph state
│   │   ├── nodes.py             # CRAG node implementations
│   │   └── workflow.py          # LangGraph graph and conditional edges
│   ├── ingestion/
│   │   ├── chunker.py           # Markdown parsing and chunk metadata
│   │   └── indexer.py           # Qdrant collection creation and indexing
│   └── retrieval/
│       ├── hybrid_search.py     # Dense + sparse RRF retrieval
│       └── reranker.py          # Cohere reranking and pacing
├── scripts/
│   └── fetch_docs.py            # Official documentation fetch and normalization
├── fine_tuning/
│   ├── generate_dataset.py      # Grader dataset generation
│   └── train_qlora.py           # QLoRA training/evaluation experiment
├── tests/
│   ├── test_ingestion.py
│   ├── test_graph.py
│   ├── test_retrieval.py
│   └── test_milestone_upgrades.py
├── frontend/
│   └── src/
│       ├── app/
│       ├── components/
│       ├── lib/
│       └── types/
├── data/
│   └── raw_docs/                # Fetched documentation corpus (generated)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Getting started

### Prerequisites

-   Python 3.11 or newer.
-   `uv` for Python dependency management.
-   Docker Engine and Docker Compose.
-   Node.js and the package manager used by the frontend project.
-   API credentials for Groq and Cohere.
-   Qdrant and Redis endpoints (local Docker services or hosted
    equivalents).

The embedding stack is CPU-oriented: FastEmbed runs ONNX-based embedding
models without requiring a local CUDA GPU. LLM synthesis and reranking
are API-backed services.

### 1. Clone the repository

``` bash
git clone https://github.com/yathish-ragav/sysarch-rag.git
cd sysarch-rag
```

Replace `yathish-ragav` with the repository owner.

### 2. Start local infrastructure

The included Compose configuration starts Qdrant and Redis with
persistent local volumes. Their ports are bound to loopback.

``` bash
docker compose up -d qdrant redis
docker compose ps
```

Default local endpoints:

-   Qdrant HTTP: `http://localhost:6333`
-   Qdrant gRPC: `http://localhost:6334`
-   Redis: `redis://localhost:6379`

### 3. Configure environment variables

Create a local `.env` file in the repository root. Populate it with your
own credentials; do not commit this file.

``` dotenv
# Required inference / reranking credentials
GROQ_API_KEY=
COHERE_API_KEY=

# Vector database and cache
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=sysarch_docs
REDIS_URL=redis://localhost:6379

# Runtime
ENV=development
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
EXPOSE_DOCS_IN_PROD=false

# Request controls
MAX_REQUEST_BYTES=16384
TRUSTED_PROXY_HOPS=1
RATE_LIMIT_PER_MIN=12
USER_RATE_LIMIT_PER_MIN=15
DAILY_QUERY_CAP=400

# LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=sysarch-rag

# Langfuse
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# Models and retrieval thresholds (optional overrides)
DENSE_MODEL=BAAI/bge-base-en-v1.5
SPARSE_MODEL=Qdrant/bm25
RERANK_MODEL=rerank-v3.5
SYNTHESIS_MODEL=openai/gpt-oss-120b
GRADER_MODEL=openai/gpt-oss-20b
RERANK_SCORE_THRESHOLD=0.65
MAX_RETRIES=2
```

**Configuration note:** The values shown for the LangSmith and Langfuse
hosts above are common hosted endpoints; the application reads these
from environment variables. Confirm the correct host for the
account/region you use. The code's own defaults are
`https://langchain.com` and `https://langfuse.com`, so explicitly
setting the intended deployment endpoints is recommended.

The application settings require `GROQ_API_KEY`, `COHERE_API_KEY`,
`LANGCHAIN_API_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY` at
startup. Qdrant and Redis have local defaults.

### 4. Install backend dependencies

``` bash
uv sync
```

### 5. Fetch the documentation corpus

``` bash
uv run python scripts/fetch_docs.py
```

This fetches upstream docs, normalizes them, writes `MANIFEST.json`, and
stages the result under `data/raw_docs`.

### 6. Build the Qdrant index

The source includes the chunking and indexing implementation. Run the
repository's ingestion/indexing entry point if provided by your
checkout. The source export identifies `load_and_chunk_corpus()` in
`src/ingestion/chunker.py` and `index_chunks()` /
`create_versioned_collection()` in `src/ingestion/indexer.py`; it does
not establish a standalone CLI command for indexing, so this README does
not invent one.

### 7. Start the backend

``` bash
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend base URL: `http://localhost:8000`

-   Liveness: `GET /healthz`
-   Readiness: `GET /readyz`
-   OpenAPI UI: `GET /docs` when docs are enabled by the application
    configuration.

### 8. Start the frontend

Set the API base URL in the frontend environment:

``` dotenv
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Then use the frontend package manager and scripts defined in
`frontend/package.json`:

``` bash
cd frontend
npm install
npm run dev
```

The frontend source uses `NEXT_PUBLIC_API_BASE_URL` and falls back to
`http://localhost:8000` if it is not set.

## Environment configuration

The following table summarizes settings defined in `src/core/config.py`.

  -----------------------------------------------------------------------------------------------
  Variable                                 Required Default                    Purpose
  --------------------------- --------------------- -------------------------- ------------------
  `GROQ_API_KEY`                                Yes ---                        Groq model
                                                                               inference

  `COHERE_API_KEY`                              Yes ---                        Cohere reranking

  `QDRANT_URL`                                   No `http://localhost:6333`    Qdrant endpoint

  `QDRANT_API_KEY`                               No `None`                     Qdrant Cloud
                                                                               authentication

  `QDRANT_COLLECTION`                            No `sysarch_docs`             Active document
                                                                               collection

  `REDIS_URL`                                    No `redis://localhost:6379`   Redis endpoint

  `ENV`                                          No `development`              Runtime
                                                                               environment label

  `CORS_ORIGINS`                                 No localhost origins          Comma-separated
                                                                               CORS allowlist

  `EXPOSE_DOCS_IN_PROD`                          No `true`                     Controls API docs
                                                                               exposure

  `MAX_REQUEST_BYTES`                            No `16384`                    Maximum request
                                                                               body size

  `TRUSTED_PROXY_HOPS`                           No `1`                        Forwarded-IP proxy
                                                                               hop handling

  `RATE_LIMIT_PER_MIN`                           No `12`                       IP-based request
                                                                               limit

  `USER_RATE_LIMIT_PER_MIN`                      No `15`                       Session/user-key
                                                                               request limit

  `DAILY_QUERY_CAP`                              No `400`                      Daily live
                                                                               pipeline cap

  `LANGCHAIN_TRACING_V2`                         No `true`                     LangSmith tracing
                                                                               toggle

  `LANGCHAIN_ENDPOINT`                           No `https://langchain.com`    LangSmith endpoint

  `LANGCHAIN_API_KEY`                           Yes ---                        LangSmith
                                                                               credential

  `LANGCHAIN_PROJECT`                            No `sysarch-rag`              LangSmith project

  `LANGFUSE_PUBLIC_KEY`                         Yes ---                        Langfuse public
                                                                               key

  `LANGFUSE_SECRET_KEY`                         Yes ---                        Langfuse secret
                                                                               key

  `LANGFUSE_HOST`                                No `https://langfuse.com`     Langfuse host

  `DENSE_MODEL`                                  No `BAAI/bge-base-en-v1.5`    Dense embedding
                                                                               model

  `SPARSE_MODEL`                                 No `Qdrant/bm25`              Sparse embedding
                                                                               model

  `RERANK_MODEL`                                 No `rerank-v3.5`              Cohere reranker

  `SYNTHESIS_MODEL`                              No `openai/gpt-oss-120b`      Answer synthesis
                                                                               model

  `GRADER_MODEL`                                 No `openai/gpt-oss-20b`       Evidence grading
                                                                               model

  `RERANK_SCORE_THRESHOLD`                       No `0.65`                     Reranking cutoff

  `MAX_RETRIES`                                  No `2`                        Maximum corrective
                                                                               retries
  -----------------------------------------------------------------------------------------------

## Ingesting and indexing documentation

### Fetch / refresh

``` bash
uv run python scripts/fetch_docs.py
```

The script performs shallow sparse checkouts and builds a staged corpus.
It checks minimum document counts before replacing the existing raw
corpus and records abbreviated upstream commit hashes and file counts in
`MANIFEST.json`.

### Chunking and indexing

The ingestion path is split into two explicit responsibilities:

1.  `load_and_chunk_corpus()` reads the Markdown corpus and emits
    `DocumentChunk` records with deterministic IDs and metadata.
2.  `create_versioned_collection()` and `index_chunks()` create/populate
    the Qdrant index using named dense and sparse vectors.

The indexer verifies the indexed point count against the chunk count and
uses a collection alias to switch the active corpus version. This
supports versioned index builds rather than requiring queries to target
a partially populated collection.

> The flattened source export does not identify a dedicated ingestion
> CLI or CI command. Use the command or orchestration entry point
> present in the actual checkout rather than assuming one.

## API reference

### `POST /v1/query`

Runs a query and returns a complete JSON response.

**Request**

``` json
{
  "query": "How do PodDisruptionBudgets affect a voluntary node drain?",
  "service_filter": "kubernetes",
  "detail_level": "standard",
  "session_id": "demo-session"
}
```

Fields:

  --------------------------------------------------------------------------
  Field              Type                          Required Constraints /
                                                            values
  ------------------ ---------------- --------------------- ----------------
  `query`            string                             Yes 3--1,500
                                                            characters

  `service_filter`   string or null                      No `kubernetes`,
                                                            `aws-eks`,
                                                            `etcd-cncf`

  `detail_level`     string                              No `brief`,
                                                            `standard`,
                                                            `detailed`;
                                                            default
                                                            `standard`

  `session_id`       string or null                      No 1--64 characters
                                                            matching
                                                            letters, digits,
                                                            `_`, `-`
  --------------------------------------------------------------------------

**Example**

``` bash
curl -sS http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "How do PodDisruptionBudgets affect a voluntary node drain?",
    "service_filter": "kubernetes",
    "detail_level": "standard",
    "session_id": "demo-session"
  }'
```

**Response shape**

``` json
{
  "request_id": "req_<id>",
  "answer": "A documentation-grounded answer ... [S1]",
  "citations": [
    {
      "ref_id": "S1",
      "service": "kubernetes",
      "version": "1.32",
      "file_path": "kubernetes/concepts/workloads/pods/disruptions.md",
      "section": "Disruptions > Pod disruption budgets",
      "section_hierarchy": "Disruptions > Pod disruption budgets",
      "source_url": "https://kubernetes.io/docs/...",
      "snippet": "Relevant source excerpt",
      "full_text": "Evidence text used for the citation"
    }
  ],
  "services_searched": ["kubernetes"],
  "retry_count": 0,
  "refused": false,
  "refusal_reason": null,
  "cache_hit": false,
  "cache_tier": "live_crag",
  "latency_ms": 1234.5,
  "original_latency_ms": null,
  "trace": [
    {
      "node": "analyze_query",
      "status": "ok",
      "duration_ms": 12.3,
      "summary": "..."
    }
  ]
}
```

The values above illustrate the response contract; they are not a
captured production response.

### `POST /v1/query/stream`

Runs the same query flow and streams progress as `text/event-stream`.

``` bash
curl -N http://localhost:8000/v1/query/stream \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  -d '{
    "query": "Explain etcd quorum and failure tolerance.",
    "service_filter": "etcd-cncf",
    "detail_level": "detailed",
    "session_id": "demo-session"
  }'
```

The frontend recognizes these event types:

  Event          Meaning
  -------------- ----------------------------------------------------
  `start`        Request has started; includes request ID and query
  `cache`        Cache lookup result and cache tier
  `trace`        A pipeline node/trace marker
  `node_start`   Node execution start marker
  `node_end`     Node status, duration, and summary
  `final`        Final `RAGResponse` payload
  `error`        Stream-level error detail

SSE frames are JSON encoded in `data:` records and separated by a blank
line.

### Health endpoints

  -----------------------------------------------------------------------
  Endpoint                            Purpose
  ----------------------------------- -----------------------------------
  `GET /healthz`                      Process liveness response

  `GET /readyz`                       Checks that the configured Qdrant
                                      collection exists and Redis
                                      responds to `PING`
  -----------------------------------------------------------------------

Readiness does not establish that every external inference provider is
reachable or that a complete end-to-end query will succeed.

## Caching and operational behavior

The cache layer uses two complementary paths:

-   **Exact Redis cache** for normalized query keys and request
    dimensions.
-   **Semantic Qdrant cache** for sufficiently similar queries, guarded
    by technical compatibility checks.

The cache scope includes the corpus version, synthesis model, service
filter, and detail level. The semantic cache also compares technical
signatures so that queries containing materially different versions,
flags, or technical kinds are less likely to be treated as equivalent.

Implementation constants in the source include:

-   Positive cache TTL: 12 hours.
-   Negative/refusal cache TTL: 10 minutes.
-   Semantic cache collection: `sysarch_semantic_cache`.

Cache hits are returned with `cache_hit` and `cache_tier` metadata. The
response contract distinguishes `exact_redis`, `semantic_qdrant`, and
`live_crag`.

The API also implements request-size limits, CORS allowlisting, rate
controls, a bounded concurrent pipeline semaphore, and a 45-second
pipeline timeout. Exact limits are configurable; see [Environment
configuration](#environment-configuration).

## Observability

SysArch-RAG integrates two tracing systems:

### LangSmith

LangSmith tracing is configured through:

-   `LANGCHAIN_TRACING_V2`
-   `LANGCHAIN_ENDPOINT`
-   `LANGCHAIN_API_KEY`
-   `LANGCHAIN_PROJECT`

The LangGraph invocation attaches callbacks and metadata to the run,
including session identifiers where provided.

### Langfuse

Langfuse is configured through:

-   `LANGFUSE_PUBLIC_KEY`
-   `LANGFUSE_SECRET_KEY`
-   `LANGFUSE_HOST`

The application creates a Langfuse callback handler, attaches tags and
optional session metadata, and flushes the client during
shutdown/workflow cleanup.

Use the two systems to inspect graph execution, latency, failures, and
model interactions. Actual trace availability depends on valid
credentials, endpoint configuration, network access, and provider-side
project setup.

## Evaluation and measured results

The repository includes a RAGAS evaluation runner and a checked-in
evaluation snapshot. The supplied `eval_report.json` records:

-   Evaluation timestamp: `2026-11-27T17:57:49.668420+00:00`
-   Judge model: `openai/gpt-oss-20b`
-   Evaluation cases: **15**

### RAGAS snapshot

  Metric                Measured   Configured threshold Result
  ------------------- ---------- ---------------------- -----------------
  Faithfulness            0.8385                   0.95 Below threshold
  Context Recall          0.8462                   0.90 Below threshold
  Context Precision       0.8141                   0.85 Below threshold
  Refusal Accuracy        1.0000                   1.00 Meets threshold

<sub>Source: included `eval_report.json`. Values are the
reported aggregate snapshot, not a live benchmark.</sub>

The evaluation runner defines thresholds of 0.95 Faithfulness, 0.90
Context Recall, 0.85 Context Precision, and 1.00 Refusal Accuracy. The
supplied report shows that the first three metrics do not yet reach
those targets. A small evaluation set is useful for iterative
diagnostics but is not sufficient by itself to establish broad
production reliability.

### Interpreting the metrics

-   **Faithfulness** measures whether generated claims are supported by
    the retrieved context.
-   **Context Recall** measures whether the retrieved context contains
    information needed to answer the reference question.
-   **Context Precision** measures how much of the retrieved context is
    relevant.
-   **Refusal Accuracy** measures whether the system correctly refuses
    cases designated as unanswerable.

Evaluation scores depend on the dataset, judge model, prompt, and
scoring configuration. Re-run evaluation against a versioned dataset
when the corpus, prompts, retrieval configuration, or models change.

### Running evaluation

The source includes `src/evaluation/run_ragas.py`. Its command-line
entry point is:

``` bash
uv run python -m src.evaluation.run_ragas
```

This assumes the evaluation dataset and environment expected by that
module are present in the checkout.

## Fine-tuning experiment

The repository contains a separate grader fine-tuning experiment using:

-   Base model: `Qwen/Qwen2.5-3B-Instruct`
-   QLoRA training script: `fine_tuning/train_qlora.py`
-   Dataset generation: `fine_tuning/generate_dataset.py`

The included `benchmark_results.json` reports the following 54-example
comparison:

  Metric                  Base model    QLoRA
  --------------------- ------------ --------
  Evaluation examples             54       54
  JSON-valid rate             0.9815   1.0000
  Accuracy                    0.6481   0.8889

This is an experiment result for the grader task. It should not be
conflated with the RAGAS end-to-end evaluation, and it does not
establish that the fine-tuned adapter is used by the deployed inference
path. In the supplied runtime configuration, the grader and synthesis
models are configured through the Groq API.

## Deployment notes

The supplied backend Dockerfile uses a multi-stage build:

1.  A Python 3.11 slim builder stage installs the locked dependencies
    using `uv`.
2.  FastEmbed dense and sparse models are downloaded during image build.
3.  A slim runtime stage copies the virtual environment and model cache.
4.  The runtime runs as a non-root `appuser`.
5.  Uvicorn binds to `0.0.0.0` and uses the platform-provided `PORT`
    when available.

The Compose file is intended for local infrastructure. It starts:

-   Qdrant `v1.13.4`, with persistent `qdrant_data`.
-   Redis `7.4.2-alpine`, with persistent `redis_data`, a 256 MB
    max-memory setting, and `allkeys-lru` eviction.
-   Both services bind published ports to `127.0.0.1`.

### Production topology

A typical hosted deployment separates the stateless application from
managed data services:

``` text
Browser
  |
  v
Next.js frontend (Vercel or equivalent)
  |
  v
FastAPI application (container platform)
  |--------------------|
  v                    v
Qdrant Cloud          Managed Redis
  |
  +--> Dense + sparse document index
  +--> Semantic response cache

FastAPI / LangGraph
  |--> Groq inference
  |--> Cohere reranking
  |--> LangSmith tracing
  +--> Langfuse tracing
```

The source blueprint describes Vercel for the frontend and a container
hosting platform for the API, with hosted Qdrant and Redis for
production. Configure the deployed frontend's
`NEXT_PUBLIC_API_BASE_URL`, backend secrets, CORS allowlist, provider
endpoints, and persistent data services for the actual environment.

The Dockerfile command enables proxy headers and trusts forwarded IP
headers broadly (`--forwarded-allow-ips="*"`). In a production
deployment, ensure the application is only reachable through the
intended trusted proxy/load balancer and set `TRUSTED_PROXY_HOPS` to
match the real proxy chain.

## Security and operational considerations

-   Keep API keys in the deployment platform's secret manager or an
    untracked local `.env` file.
-   Never commit `.env`, provider credentials, Qdrant API keys, Redis
    credentials, or trace secrets.
-   Restrict CORS to the deployed frontend origin(s).
-   Use TLS for public API and managed data-service connections.
-   Keep Qdrant and Redis off public unauthenticated interfaces.
-   Configure proxy trust and forwarded-IP handling to match the actual
    ingress topology.
-   Review API documentation exposure (`EXPOSE_DOCS_IN_PROD`) before
    production use.
-   Treat session identifiers as request correlation/rate-limit keys,
    not as proof of user authentication.
-   Monitor provider quotas, rate limits, timeout rates, cache behavior,
    and readiness failures.

SysArch-RAG is a technical documentation assistant, not a substitute for
validating commands and operational changes in a real cluster. Always
verify high-impact infrastructure actions against the relevant upstream
documentation and the target environment.

## Known limitations

This section intentionally distinguishes measured or visible
implementation behavior from guarantees the project does not claim.

1.  **Evaluation thresholds are not all met in the supplied snapshot.**
    Faithfulness, context recall, and context precision are below
    configured targets. See the exact values above.
2.  **Evaluation coverage is limited.** The supplied report contains 15
    cases. It is a useful snapshot, not proof of performance across all
    Kubernetes, EKS, and etcd topics or versions.
3.  **Documentation freshness is tied to the fetched corpus.** The
    manifest records source revisions. New upstream documentation is not
    automatically reflected until the corpus is refreshed and
    re-indexed.
4.  **The EKS source scope is `latest/ug`.** Unlike the Kubernetes
    release branch and etcd v3.5 path, this source is not represented as
    a fixed historical EKS documentation version in the manifest.
5.  **The fine-tuning benchmark is a separate grader experiment.** It is
    not the RAGAS score and does not, by itself, show that the trained
    adapter serves production requests.
6.  **Readiness is dependency-scoped.** `/readyz` checks Qdrant
    collection availability and Redis ping; it does not prove that Groq,
    Cohere, or tracing providers are healthy.
7.  **Provider-backed components require external services.** LLM
    synthesis and Cohere reranking depend on credentials, network
    connectivity, provider availability, and quotas.
8.  **Local fallback controls are not a distributed identity system.**
    Session headers and IDs are caller-supplied correlation inputs.
    Deployments requiring authenticated per-user access should place an
    authentication/authorization layer in front of the API.
9.  **The repository source export is not a substitute for deployment
    verification.** Runtime behavior depends on environment variables,
    hosted service configuration, current corpus/index state, and the
    deployed frontend/backend revisions.

## Roadmap

The project architecture and implementation support continued work in
these areas:

-   Expand and version the golden evaluation set across Kubernetes, EKS,
    and etcd question classes.
-   Track quality and latency regressions across corpus, prompt, model,
    and retrieval changes.
-   Improve evidence sufficiency and claim-level support measurement.
-   Add broader operational telemetry for provider health, cache
    effectiveness, and ingestion outcomes.
-   Keep source manifests, index versions, and evaluation artifacts
    aligned for reproducible releases.
-   Extend documentation coverage while preserving explicit
    source/version metadata.

## Acknowledgements

SysArch-RAG builds on the documentation and open-source ecosystems
maintained by:

-   [Kubernetes](https://github.com/kubernetes/website)
-   [Amazon EKS
    documentation](https://github.com/awsdocs/amazon-eks-user-guide)
-   [etcd](https://github.com/etcd-io/website)
-   [LangChain and LangGraph](https://www.langchain.com/)
-   [Qdrant](https://qdrant.tech/)
-   [FastEmbed](https://github.com/qdrant/fastembed)
-   [Cohere](https://cohere.com/)
-   [Groq](https://groq.com/)
-   [RAGAS](https://docs.ragas.io/)

------------------------------------------------------------------------

<div align="center">
**SysArch-RAG** · Evidence-first answers for cloud-native and
distributed systems engineering.
</div>
