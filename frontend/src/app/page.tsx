"use client";

import React, { useMemo, useRef, useState } from "react";
import { streamRagQuery } from "@/lib/streamQuery";
import { Citation, RAGResponse, ServiceFilter, TraceStep } from "@/types/rag";

const SERVICE_OPTIONS: { label: string; value: ServiceFilter }[] = [
  { label: "All Services", value: null },
  { label: "Kubernetes", value: "kubernetes" },
  { label: "AWS EKS", value: "aws-eks" },
  { label: "etcd", value: "etcd-cncf" },
];

const SAMPLE_QUERIES: { label: string; query: string; service: ServiceFilter }[] = [
  {
    label: "K8s + EKS PDB Upgrade Drain",
    query: "How do PodDisruptionBudgets affect managed node group upgrades in Amazon EKS?",
    service: null,
  },
  {
    label: "etcd Quorum & Split-Brain",
    query: "What happens to etcd write availability when a 3-node cluster loses 2 members, and how is quorum restored?",
    service: "etcd-cncf",
  },
  {
    label: "EKS IRSA Pod Identity",
    query: "How does IAM Roles for Service Accounts (IRSA) issue temporary AWS credentials to Kubernetes pods?",
    service: "aws-eks",
  },
  {
    label: "Out-of-Scope Refusal Test",
    query: "What is the best sourdough bread hydration ratio for beginners?",
    service: null,
  },
];

function renderAnswerBlocks(answer: string, onSelectCitation: (refId: string) => void) {
  const parts = answer.split(/(```[\s\(\S\)]*?```)/g);
  return parts.map((part, idx) => {
    if (part.startsWith("```") && part.endsWith("```")) {
      const inner = part.slice(3, -3);
      const firstNewline = inner.indexOf("\n");
      const lang = firstNewline > 0 ? inner.slice(0, firstNewline).trim() : "code";
      const code = firstNewline > 0 ? inner.slice(firstNewline + 1) : inner;
      return (
        <div key={idx} className="my-3 overflow-hidden rounded-lg border border-slate-800 bg-slate-950">
          <div className="border-b border-slate-800 px-3 py-1.5 text-xs font-mono text-slate-400">
            {lang || "snippet"}
          </div>
          <pre className="overflow-x-auto p-3 text-xs leading-relaxed text-emerald-300 font-mono">
            <code>{code}</code>
          </pre>
        </div>
      );
    }
    const tokens = part.split(/(\[S\d+\])/g);
    return (
      <p key={idx} className="whitespace-pre-wrap leading-7 text-slate-200">
        {tokens.map((tok, tIdx) => {
          const match = tok.match(/^\[(S\d+)\]$/);
          if (match) {
            return (
              <button
                key={tIdx}
                type="button"
                onClick={() => onSelectCitation(match[1])}
                className="mx-0.5 inline-flex items-center rounded bg-cyan-500/15 px-1.5 py-0.5 font-mono text-xs font-semibold text-cyan-300 ring-1 ring-inset ring-cyan-500/40 hover:bg-cyan-500/25"
              >
                {tok}
              </button>
            );
          }
          return <React.Fragment key={tIdx}>{tok}</React.Fragment>;
        })}
      </p>
    );
  });
}

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [serviceFilter, setServiceFilter] = useState<ServiceFilter>(null);
  const [loading, setLoading] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [cacheStatus, setCacheStatus] = useState<{ hit: boolean; tier: string } | null>(null);
  const [liveSteps, setLiveSteps] = useState<TraceStep[]>([]);
  const [result, setResult] = useState<RAGResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedRef, setSelectedRef] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const sessionId = useMemo(() => "web-console-v1", []);

  async function runQuery(targetQuery: string, targetService: ServiceFilter) {
    const trimmed = targetQuery.trim();
    if (trimmed.length < 3) {
      setError("Please enter at least 3 characters.");
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedRef(null);
    setCacheStatus(null);
    setLiveSteps([]);
    setActiveNode("cache_lookup");

    try {
      await streamRagQuery(
        { query: trimmed, serviceFilter: targetService, sessionId },
        (ev) => {
          if (ev.type === "cache") setCacheStatus({ hit: ev.hit, tier: ev.tier });
          else if (ev.type === "node_start" || ev.type === "trace") setActiveNode(ev.node);
          else if (ev.type === "node_end") {
            setLiveSteps((prev) => [...prev, { node: ev.node, duration_ms: ev.duration_ms, summary: ev.summary }]);
          } else if (ev.type === "final") {
            setResult(ev.payload);
            setActiveNode(null);
            if (ev.payload.citations.length > 0) setSelectedRef(ev.payload.citations[0].ref_id);
          } else if (ev.type === "error") {
            setError(ev.detail);
            setActiveNode(null);
          }
        },
        controller.signal
      );
    } catch (err: unknown) {
      if ((err as { name?: string })?.name !== "AbortError") {
        setError(err instanceof Error ? err.message : "Failed to connect to backend.");
      }
    } finally {
      setLoading(false);
      setActiveNode(null);
    }
  }

  const displayedSteps = result?.trace?.length ? result.trace : liveSteps;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800/80 bg-slate-900/70">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" />
              <h1 className="text-lg font-bold tracking-tight text-white">SysArch-RAG</h1>
              <span className="rounded-md border border-slate-700 bg-slate-800 px-2 py-0.5 font-mono text-xs text-cyan-300">
                Corrective Agentic RAG
              </span>
            </div>
            <p className="mt-0.5 text-xs text-slate-400">
              Grounded Cloud &amp; Distributed Systems Intelligence • Kubernetes • AWS EKS • etcd
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
            <span className="rounded border border-slate-800 bg-slate-900 px-2.5 py-1">Hybrid RRF + Cohere v3.5</span>
            <span className="rounded border border-slate-800 bg-slate-900 px-2.5 py-1">Strict [S#] Grounding</span>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-xl">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">Scope:</span>
            {SERVICE_OPTIONS.map((opt) => (
              <button
                key={opt.label}
                type="button"
                onClick={() => setServiceFilter(opt.value)}
                className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                  serviceFilter === opt.value
                    ? "bg-cyan-500 text-slate-950"
                    : "border border-slate-700 bg-slate-800/80 text-slate-300 hover:border-slate-600"
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <form onSubmit={(e) => { e.preventDefault(); void runQuery(query, serviceFilter); }} className="flex flex-col gap-3 sm:flex-row">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask an infrastructure question (e.g., How does etcd handle leader election and quorum loss?)..."
              className="flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
            />
            <button
              type="submit"
              disabled={loading}
              className="rounded-lg bg-cyan-500 px-6 py-3 text-sm font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:opacity-50"
            >
              {loading ? "Running CRAG..." : "Ask Engine"}
            </button>
          </form>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            <span className="text-xs text-slate-500">Try sample:</span>
            {SAMPLE_QUERIES.map((s) => (
              <button
                key={s.label}
                type="button"
                disabled={loading}
                onClick={() => { setQuery(s.query); setServiceFilter(s.service); void runQuery(s.query, s.service); }}
                className="rounded-md border border-slate-800 bg-slate-950/80 px-2.5 py-1 text-xs text-slate-300 hover:border-cyan-500/50 hover:text-cyan-200 disabled:opacity-50"
              >
                {s.label}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="mt-6 rounded-xl border border-rose-500/40 bg-rose-950/40 p-4 text-sm text-rose-200">
            <strong>Pipeline Notice:</strong> {error}
          </div>
        )}

        {(loading || result || (!error && displayedSteps.length > 0)) && (
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="space-y-6 lg:col-span-8">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
                <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Synthesized Response</h2>
                    {result?.refused && (
                      <span className="rounded bg-amber-500/20 px-2 py-0.5 font-mono text-xs text-amber-300">
                        Refused ({result.refusal_reason})
                      </span>
                    )}
                  </div>
                  {result && (
                    <div className="flex flex-wrap items-center gap-2 font-mono text-xs">
                      <span className={`rounded px-2 py-0.5 ${result.cache_hit ? "bg-emerald-500/15 text-emerald-300" : "bg-indigo-500/15 text-indigo-300"}`}>
                        {result.cache_hit ? `CACHE HIT (${result.cache_tier})` : "LIVE CRAG"}
                      </span>
                      <span className="rounded bg-slate-800 px-2 py-0.5 text-slate-300">
                        {result.latency_ms.toFixed(0)} ms
                        {result.cache_hit && result.original_latency_ms ? ` (saved ${(result.original_latency_ms - result.latency_ms).toFixed(0)} ms)` : ""}
                      </span>
                      {result.retry_count > 0 && (
                        <span className="rounded bg-amber-500/15 px-2 py-0.5 text-amber-300">Retries: {result.retry_count}</span>
                      )}
                    </div>
                  )}
                </div>

                {result ? (
                  <div className="space-y-2 text-sm">{renderAnswerBlocks(result.answer, (refId) => setSelectedRef(refId))}</div>
                ) : (
                  <div className="flex items-center gap-3 py-8 text-sm text-slate-400">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-cyan-400 border-t-transparent" />
                    <span>Executing node: <strong className="font-mono text-cyan-300">{activeNode || "starting"}</strong>...</span>
                  </div>
                )}
              </div>

              {result && result.citations.length > 0 && (
                <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6">
                  <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-300">
                    Grounded Evidence ({result.citations.length})
                  </h3>
                  <div className="space-y-3">
                    {result.citations.map((cit: Citation) => (
                      <div
                        key={cit.ref_id}
                        onClick={() => setSelectedRef(cit.ref_id)}
                        className={`cursor-pointer rounded-lg border p-4 transition ${
                          selectedRef === cit.ref_id ? "border-cyan-500/70 bg-cyan-950/20" : "border-slate-800 bg-slate-950/60 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2">
                            <span className="rounded bg-cyan-500/20 px-2 py-0.5 font-mono text-xs font-bold text-cyan-300">[{cit.ref_id}]</span>
                            <span className="font-mono text-xs font-semibold text-slate-200">{cit.service}</span>
                            <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] text-slate-400">v{cit.version}</span>
                          </div>
                          <span className="font-mono text-xs text-slate-400">{cit.file_path}</span>
                        </div>
                        <p className="mt-1.5 text-xs font-medium text-slate-300">{cit.section_hierarchy}</p>
                        {cit.snippet && <p className="mt-2 border-l-2 border-slate-700 pl-3 text-xs leading-relaxed text-slate-400">{cit.snippet}</p>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <aside className="lg:col-span-4">
              <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
                <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-300">Live LangGraph Trace</h2>
                {result?.request_id && <p className="mt-1 font-mono text-[11px] text-slate-500">ID: {result.request_id}</p>}
                {cacheStatus && (
                  <div className="mt-3 rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 font-mono text-xs">
                    Cache Lookup: <span className={cacheStatus.hit ? "text-emerald-400" : "text-slate-300"}>{cacheStatus.hit ? `HIT (${cacheStatus.tier})` : "MISS -> CRAG"}</span>
                  </div>
                )}
                <ol className="mt-4 space-y-2.5">
                  {displayedSteps.map((step, i) => (
                    <li key={`${step.node}-${i}`} className="rounded-lg border border-slate-800 bg-slate-950/90 p-3 text-xs">
                      <div className="flex items-center justify-between font-mono">
                        <span className="font-semibold text-cyan-300">{step.node}</span>
                        <span className="text-slate-400">{step.duration_ms.toFixed(0)} ms</span>
                      </div>
                      {step.summary && <p className="mt-1 text-slate-400">{step.summary}</p>}
                    </li>
                  ))}
                  {loading && activeNode && (
                    <li className="rounded-lg border border-cyan-500/40 bg-cyan-950/20 p-3 font-mono text-xs text-cyan-200">
                      Running {activeNode}...
                    </li>
                  )}
                </ol>
              </div>
            </aside>
          </div>
        )}
      </main>
    </div>
  );
}
