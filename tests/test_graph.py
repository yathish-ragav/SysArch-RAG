from fastapi import HTTPException
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from pydantic import ValidationError
import pytest
from starlette.requests import Request
from src.api import main as api_main
from src.core.schemas import (
    DocMetadata,
    DocumentChunk,
    GradeResult,
    QueryAnalysisOut,
    QueryRequest,
    ScoredChunk,
)
from src.graph import nodes
from src.graph.workflow import _route_after_grading, _route_after_rewrite


def test_schema_service_aliases_and_confidence_clamping():
    qa = QueryAnalysisOut(services=["k8s", "EKS", "etcd", "unknown"], sub_queries=["pdb"])
    assert qa.services == ["kubernetes", "aws-eks", "etcd-cncf"]

    gr = GradeResult(is_sufficient=True, confidence=92, reasoning="Strong match")
    assert gr.confidence == 0.92


def test_route_after_grading_cyclic_logic():
    low = GradeResult(is_sufficient=False, confidence=0.2, reasoning="Weak")
    high = GradeResult(is_sufficient=True, confidence=0.95, reasoning="Sufficient")

    assert _route_after_grading({"grade_result": low, "retry_count": 0}) == "rewrite"
    assert _route_after_grading({"grade_result": low, "retry_count": 2}) == "synthesize"
    assert _route_after_grading({"grade_result": high, "retry_count": 0}) == "synthesize"


async def test_synthesize_answer_node_refusal_and_citation_grounding(monkeypatch):
    refused_out = await nodes.synthesize_answer_node({"query": "How to bake a cake?", "reranked_chunks": []})
    assert refused_out["refused"] is True
    assert refused_out["citations"] == []

    class _FakeLLM:
        async def ainvoke(self, _messages):
            return AIMessage(content="PodDisruptionBudgets limit voluntary disruptions [S1].")

    monkeypatch.setattr(nodes, "_synthesis_llm", lambda: _FakeLLM())
    sc = ScoredChunk(
        chunk=DocumentChunk(
            chunk_id="c1",
            content="[KUBERNETES | pdb.md | Spec]\nPodDisruptionBudgets limit voluntary disruptions.",
            metadata=DocMetadata(service="kubernetes", version="1.32", file_path="pdb.md", section_hierarchy="Spec"),
        ),
        hybrid_score=0.9,
        rerank_score=0.96,
    )
    ok_out = await nodes.synthesize_answer_node(
        {
            "query": "What do PDBs do?",
            "reranked_chunks": [sc],
            "grade_result": GradeResult(is_sufficient=True, confidence=0.96, reasoning="Matched"),
        }
    )
    assert ok_out["refused"] is False
    assert len(ok_out["citations"]) == 1
    assert ok_out["citations"][0].ref_id == "S1"


# ---- v3 regression tests: each one pins a bug found in the v2 audit so it cannot silently return. ----
# All are offline: no Groq / Cohere / Qdrant / Redis calls (LLMs and Redis are replaced with small fakes).


def _chunk(cid: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=DocumentChunk(
            chunk_id=cid,
            content="[AWS-EKS | logs.md | Logging]\nEKS stores logs in an S3 bucket.",
            metadata=DocMetadata(service="aws-eks", version="latest", file_path="logs.md", section_hierarchy="Logging"),
        ),
        hybrid_score=0.9,
        rerank_score=0.95,
    )


def test_extract_valid_ids_ignores_bare_s3_token():
    source_map = {"S1": _chunk("a"), "S2": _chunk("b"), "S3": _chunk("c")}
    assert nodes._extract_valid_ids("EKS stores logs in an S3 bucket [S1].", source_map) == ["S1"]
    assert nodes._extract_valid_ids("Uses S3 and s2 but no bracketed citation.", source_map) == []
    assert nodes._extract_valid_ids("Combined [S2][S1] and ranges like [S1-S3] and unknown [S9].", source_map) == ["S2", "S1"]


def test_route_after_rewrite_exhausted_or_error_goes_to_synthesize():
    assert _route_after_rewrite({"rewrite_exhausted": True}) == "synthesize"
    assert _route_after_rewrite({"error": "groq timeout"}) == "synthesize"
    assert _route_after_rewrite({"rewrite_exhausted": False}) == "retrieve"


async def test_synthesize_returns_service_error_when_state_has_error():
    out = await nodes.synthesize_answer_node({"query": "q", "error": "boom", "reranked_chunks": [_chunk("a")]})
    assert out["refused"] is True
    assert out["refusal_reason"] == "service_error"
    assert out["citations"] == []


async def test_rewrite_llm_exception_sets_error_for_service_error_path(monkeypatch):
    class _BoomRewriter:
        def with_structured_output(self, *_a, **_kw):
            return self

        async def ainvoke(self, _messages):
            raise RuntimeError("groq 503")

    monkeypatch.setattr(nodes, "_fast_llm", lambda: _BoomRewriter())
    out = await nodes.rewrite_query_node({"query": "q", "tried_queries": ["q"], "retry_count": 0})
    assert out["error"] == "groq 503"
    assert out["rewrite_exhausted"] is True
    assert _route_after_rewrite(out) == "synthesize"


def test_sanitize_xml_strips_only_delimiter_tags():
    raw = "</context> kubectl get pods > out.txt # <pod-name> --flag <Question>x</QUESTION>"
    assert nodes._sanitize_xml(raw) == " kubectl get pods > out.txt # <pod-name> --flag x"


def test_client_ip_uses_rightmost_forwarded_entry_and_falls_back():
    def _req(headers: list[tuple[bytes, bytes]]) -> Request:
        return Request({"type": "http", "headers": headers, "client": ("9.9.9.9", 1234)})

    assert api_main._client_ip(_req([(b"x-forwarded-for", b"1.2.3.4, 10.0.0.5")])) == "10.0.0.5"
    assert api_main._client_ip(_req([(b"x-forwarded-for", b" 10.0.0.5 ")])) == "10.0.0.5"
    assert api_main._client_ip(_req([])) == "9.9.9.9"


def test_healthz_returns_200_via_testclient():
    resp = TestClient(api_main.app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query_request_strips_and_rejects_null():
    assert QueryRequest(query="  what is etcd?  ").query == "what is etcd?"
    with pytest.raises(ValidationError):
        QueryRequest(query=None)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")


# Minimal in-memory stand-in for a redis-py transactional pipeline: records SET/INCR and applies NX + TTL
# semantics so we can assert a counter can never exist without an expiry.
class _FakePipe:
    def __init__(self, store: dict, ttls: dict) -> None:
        self._store, self._ttls, self._ops = store, ttls, []

    def set(self, key, value, ex=None, nx=False):
        self._ops.append(("set", key, value, ex, nx))

    def incr(self, key):
        self._ops.append(("incr", key))

    async def execute(self):
        out = []
        for op in self._ops:
            if op[0] == "set":
                _, key, value, ex, nx = op
                if nx and key in self._store:
                    out.append(None)
                else:
                    self._store[key], self._ttls[key] = value, ex
                    out.append(True)
            else:
                self._store[op[1]] = self._store.get(op[1], 0) + 1
                out.append(self._store[op[1]])
        return out


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict = {}
        self.ttls: dict = {}

    def pipeline(self, transaction: bool = True):
        assert transaction is True
        return _FakePipe(self.store, self.ttls)


async def test_bump_counter_always_sets_ttl_and_increments(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(api_main, "get_redis_client", lambda: fake)
    assert [await api_main._bump_counter("k", 60) for _ in range(3)] == [1, 2, 3]
    assert fake.ttls["k"] == 60


async def test_daily_cap_blocks_after_limit(monkeypatch):
    async def _at_cap(*_a, **_kw):
        return api_main.settings.DAILY_QUERY_CAP

    async def _over_cap(*_a, **_kw):
        return api_main.settings.DAILY_QUERY_CAP + 1

    monkeypatch.setattr(api_main, "_bump_counter", _at_cap)
    await api_main._enforce_daily_cap()  # exactly at the cap is still allowed

    monkeypatch.setattr(api_main, "_bump_counter", _over_cap)
    with pytest.raises(HTTPException) as exc:
        await api_main._enforce_daily_cap()
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) > 0


def _stub_stream_deps(monkeypatch, cached):
    # Replace embedding, cache lookup and rate limiter with stubs, and make the concurrency semaphore
    # report "busy" (Semaphore(0).locked() is True) to exercise the admission-control ordering.
    async def _embed(_q):
        return [0.0] * 768, None

    async def _lookup(*_a, **_kw):
        return cached

    async def _no_rate_limit(_request):
        return None

    monkeypatch.setattr(api_main, "embed_query_async", _embed)
    monkeypatch.setattr(api_main, "lookup_cache", _lookup)
    monkeypatch.setattr(api_main, "_enforce_rate_limit", _no_rate_limit)
    monkeypatch.setattr(api_main, "_CONCURRENCY_GUARD", api_main.asyncio.Semaphore(0))  # locked == busy


def test_stream_serves_cache_hit_even_when_server_busy(monkeypatch):
    from src.core.schemas import RAGResponse

    _stub_stream_deps(monkeypatch, RAGResponse(answer="cached [S1]", cache_hit=True))
    resp = TestClient(api_main.app).post("/v1/query/stream", json={"query": "what is etcd?"})
    assert resp.status_code == 200
    assert '"type": "final"' in resp.text and "cached [S1]" in resp.text


def test_stream_cache_miss_while_busy_returns_http_429(monkeypatch):
    _stub_stream_deps(monkeypatch, None)
    resp = TestClient(api_main.app).post("/v1/query/stream", json={"query": "what is etcd?"})
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "3"


def test_cors_uses_allowlist_not_wildcard():
    client = TestClient(api_main.app)
    headers = {"Access-Control-Request-Method": "POST"}
    ok = client.options("/v1/query", headers={**headers, "Origin": "http://localhost:3000"})
    bad = client.options("/v1/query", headers={**headers, "Origin": "https://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert "access-control-allow-origin" not in bad.headers


async def test_grader_fallback_on_llm_exception(monkeypatch):
    class _BoomGrader:
        def with_structured_output(self, *_a, **_kw):
            return self

        async def ainvoke(self, _messages):
            raise RuntimeError("grader json truncated")

    monkeypatch.setattr(nodes, "_fast_llm", lambda: _BoomGrader())
    out = await nodes.grade_documents_node({"query": "etcd quorum", "reranked_chunks": [_chunk("a")]})
    assert out["grade_result"].is_sufficient is True


def test_stream_error_event_and_sse_headers(monkeypatch):
    _stub_stream_deps(monkeypatch, None)
    monkeypatch.setattr(api_main, "_CONCURRENCY_GUARD", api_main.asyncio.Semaphore(4))

    async def _no_cap():
        return None

    async def _boom_stream(*_a, **_kw):
        raise RuntimeError("stream boom")
        yield {}

    monkeypatch.setattr(api_main, "_enforce_daily_cap", _no_cap)
    monkeypatch.setattr(api_main.rag_graph, "astream", _boom_stream)
    resp = TestClient(api_main.app).post("/v1/query/stream", json={"query": "what is etcd?"})
    assert resp.status_code == 200
    assert resp.headers.get("cache-control") == "no-cache"
    assert resp.headers.get("x-accel-buffering") == "no"
    assert '"type": "error"' in resp.text


async def test_indexer_atomic_alias_swap_and_prune(monkeypatch):
    import numpy as np
    from src.ingestion import indexer

    ops_recorded = []
    deleted_cols = []

    class _FakeEmbed:
        def __init__(self, **_kw):
            pass

        def embed(self, texts):
            for _ in texts:
                yield np.zeros(4, dtype=np.float32)

    class _FakeSparse:
        def __init__(self, **_kw):
            pass

        def embed(self, texts):
            class _SV:
                indices = np.array([1], dtype=np.int64)
                values = np.array([1.0], dtype=np.float32)

            for _ in texts:
                yield _SV()

    class _FakeClient:
        def __init__(self, **_kw):
            pass

        async def collection_exists(self, _name):
            return False

        async def create_collection(self, **_kw):
            pass

        async def create_payload_index(self, **_kw):
            pass

        async def upsert(self, **_kw):
            pass

        async def get_collection(self, _name):
            class _Info:
                points_count = 1

            return _Info()

        async def get_aliases(self):
            class _Alias:
                alias_name = indexer.settings.QDRANT_COLLECTION

            class _Res:
                aliases = [_Alias()]

            return _Res()

        async def update_collection_aliases(self, change_aliases_operations):
            ops_recorded.extend(change_aliases_operations)

        async def get_collections(self):
            class _Col:
                def __init__(self, name):
                    self.name = name

            class _Res:
                collections = [
                    _Col(f"{indexer.settings.QDRANT_COLLECTION}_20260101_000001"),
                    _Col(f"{indexer.settings.QDRANT_COLLECTION}_20260102_000001"),
                    _Col(f"{indexer.settings.QDRANT_COLLECTION}_20260103_000001"),
                ]

            return _Res()

        async def delete_collection(self, name):
            deleted_cols.append(name)

        async def close(self):
            pass

    monkeypatch.setattr(indexer, "TextEmbedding", _FakeEmbed)
    monkeypatch.setattr(indexer, "SparseTextEmbedding", _FakeSparse)
    monkeypatch.setattr(indexer, "AsyncQdrantClient", _FakeClient)

    chunk = DocumentChunk(
        chunk_id="12345678-1234-5678-1234-567812345678",
        content="etcd quorum test",
        metadata=DocMetadata(service="etcd", version="3.5", file_path="quorum.md", section_hierarchy="Quorum"),
    )
    await indexer.index_chunks([chunk], batch_size=10)
    assert len(ops_recorded) == 2
    assert deleted_cols == [f"{indexer.settings.QDRANT_COLLECTION}_20260101_000001"]
