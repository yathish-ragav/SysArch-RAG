import argparse
import asyncio
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness

from src.core.config import settings
from src.core.observability import flush_langfuse, get_langfuse_run_config
from src.graph.workflow import rag_graph
from src.retrieval.hybrid_search import close_qdrant_client, get_dense_model
from src.retrieval.reranker import close_cohere_client

THRESHOLDS = {
    "faithfulness": 0.95,
    "context_recall": 0.90,
    "context_precision": 0.85,
    "refusal_accuracy": 1.00,
}

JUDGE_MODEL = "openai/gpt-oss-20b"

_last_ts: float = 0.0


class ThrottledChatGroq(ChatGroq):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        global _last_ts
        wait = 4.5 - (time.monotonic() - _last_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_ts = time.monotonic()
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


class FastEmbedLangchainAdapter(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in get_dense_model().embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(get_dense_model().embed([text])).tolist()


async def _run_single_case(
    item: dict,
    idx: int,
    total: int,
) -> dict:

    q = item["question"]
    t0 = time.perf_counter()

    cfg = get_langfuse_run_config(
        session_id=f"ragas-eval-{item['id']}",
        tags=["ragas-eval"],
    )

    cfg["recursion_limit"] = 15

    state = await rag_graph.ainvoke(
        {"query": q},
        config=cfg,
    )

    latency_ms = round(
        (time.perf_counter() - t0) * 1000,
        1,
    )

    chunks = state.get("reranked_chunks", [])
    refused = bool(state.get("refused", False))
    citations = state.get("citations", [])

    print(
        f"[{idx}/{total}] {item['id']} "
        f"| refused={refused} "
        f"(expected={item['should_refuse']}) "
        f"| retries={state.get('retry_count', 0)} "
        f"| citations={len(citations)} "
        f"| {latency_ms} ms"
    )

    return {
        "id": item["id"],
        "category": item["category"],
        "question": q,
        "ground_truth": item["ground_truth"],
        "should_refuse": bool(item["should_refuse"]),
        "answer": str(state.get("final_answer", "")),
        "contexts": [c.chunk.content[:1200] for c in chunks[:4]],
        "refused": refused,
        "citations_count": len(citations),
        "latency_ms": latency_ms,
    }


def _save_report(
    report_path: Path,
    results: list[dict],
    ragas_scores: dict[str, float],
    refusal_accuracy: float,
) -> dict:

    summary = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "judge_model": JUDGE_MODEL,
        "total_cases": len(results),
        "metrics": {
            **ragas_scores,
            "refusal_accuracy": refusal_accuracy,
        },
        "thresholds": THRESHOLDS,
        "cases": results,
    }

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


async def run_evaluation(
    dataset_path: Path,
    report_path: Path,
    limit: int | None = None,
    enforce_gate: bool = False,
    judge_only: bool = False,
    resume: bool = False,
    ids_filter: set[str] | None = None,
) -> int:

    raw_items: list[dict] = json.loads(dataset_path.read_text(encoding="utf-8"))

    if ids_filter:
        raw_items = [x for x in raw_items if x["id"] in ids_filter]

    if limit is not None:
        raw_items = raw_items[:limit]

    cached_by_id: dict[str, dict] = {}

    if (judge_only or resume) and report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))

        cached_by_id = {c["id"]: c for c in existing.get("cases", [])}

    results: list[dict] = []

    if judge_only:
        results = [cached_by_id[x["id"]] for x in raw_items if x["id"] in cached_by_id]

        print(f"Loaded {len(results)} cached cases from {report_path} (--judge-only).")

    else:
        try:
            for i, item in enumerate(raw_items, start=1):
                if (
                    resume
                    and item["id"] in cached_by_id
                    and (
                        not cached_by_id[item["id"]]["refused"] or item["should_refuse"]
                    )
                ):
                    print(
                        f"[{i}/{len(raw_items)}] {item['id']} | reused from checkpoint"
                    )

                    results.append(cached_by_id[item["id"]])

                    continue

                res = await _run_single_case(
                    item,
                    i,
                    len(raw_items),
                )

                results.append(res)

                ref_acc = round(
                    sum(1 for r in results if r["refused"] == r["should_refuse"])
                    / len(results),
                    4,
                )

                _save_report(
                    report_path,
                    results,
                    {},
                    ref_acc,
                )

                await asyncio.sleep(2.0)

        finally:
            await close_cohere_client()
            await close_qdrant_client()
            flush_langfuse()

    refusal_accuracy = (
        round(
            sum(1 for r in results if r["refused"] == r["should_refuse"])
            / len(results),
            4,
        )
        if results
        else 0.0
    )

    answerable = [r for r in results if not r["should_refuse"]]

    ragas_scores: dict[str, float] = {}

    if answerable:
        hf_ds = Dataset.from_dict(
            {
                "question": [r["question"] for r in answerable],
                "answer": [r["answer"] for r in answerable],
                "contexts": [
                    [c[:1200] for c in r["contexts"][:4]]
                    if r["contexts"]
                    else ["NO_CONTEXT"]
                    for r in answerable
                ],
                "ground_truth": [r["ground_truth"] for r in answerable],
            }
        )

        judge_llm = LangchainLLMWrapper(
            ThrottledChatGroq(
                model=JUDGE_MODEL,
                api_key=settings.GROQ_API_KEY,
                temperature=0.0,
                max_tokens=4096,
                reasoning_effort="low",
                timeout=90.0,
                max_retries=3,
            )
        )

        judge_emb = LangchainEmbeddingsWrapper(FastEmbedLangchainAdapter())

        run_cfg = RunConfig(
            timeout=300,
            max_retries=5,
            max_wait=60,
            max_workers=1,
        )

        eval_out = evaluate(
            dataset=hf_ds,
            metrics=[
                Faithfulness(),
                ContextRecall(),
                ContextPrecision(),
            ],
            llm=judge_llm,
            embeddings=judge_emb,
            run_config=run_cfg,
            raise_exceptions=False,
        )

        df = eval_out.to_pandas()

        for idx_row, case_obj in enumerate(answerable):
            if idx_row < len(df):
                case_obj["ragas"] = {
                    col: (
                        None
                        if math.isnan(float(df.iloc[idx_row][col]))
                        else round(
                            float(df.iloc[idx_row][col]),
                            4,
                        )
                    )
                    for col in (
                        "faithfulness",
                        "context_recall",
                        "context_precision",
                    )
                    if col in df.columns
                }

        for col in (
            "faithfulness",
            "context_recall",
            "context_precision",
        ):
            if col in df.columns:
                valid = df[col].dropna()

                ragas_scores[col] = (
                    round(float(valid.mean()), 4) if len(valid) > 0 else 0.0
                )

    summary = _save_report(
        report_path,
        results,
        ragas_scores,
        refusal_accuracy,
    )

    print("\n" + "=" * 60 + "\nSYSARCH-RAG EVALUATION SUMMARY\n" + "=" * 60)

    for k, v in summary["metrics"].items():
        t = THRESHOLDS.get(k, 0.0)

        ok = not math.isnan(v) and v >= t

        print(f"  {k:<20}: {v:.4f} (target >= {t:.2f}) [{'PASS' if ok else 'FAIL'}]")

    if enforce_gate and any(
        math.isnan(summary["metrics"].get(k, 0.0)) or summary["metrics"].get(k, 0.0) < t
        for k, t in THRESHOLDS.items()
    ):
        return 1

    return 0


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/golden_eval_set.json"),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/eval_report.json"),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--ids",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--enforce-gate",
        action="store_true",
    )

    parser.add_argument(
        "--judge-only",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    ids_filter = (
        {s.strip() for s in args.ids.split(",") if s.strip()} if args.ids else None
    )

    raise SystemExit(
        asyncio.run(
            run_evaluation(
                args.dataset,
                args.report,
                args.limit,
                args.enforce_gate,
                args.judge_only,
                args.resume,
                ids_filter,
            )
        )
    )


if __name__ == "__main__":
    main()
