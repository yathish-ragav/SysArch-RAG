"use client";

import { useEffect, useRef, useState } from "react";
import { AnswerMarkdown } from "@/components/AnswerMarkdown";
import { streamRagQuery } from "@/lib/streamQuery";
import type { DetailLevel, HistoryItem, RAGResponse, ServiceFilter, TraceStep } from "@/types/rag";

const SERVICES: { id: ServiceFilter; label: string }[] = [
  { id: "all", label: "All Docs" },
  { id: "kubernetes", label: "Kubernetes v1.32" },
  { id: "aws-eks", label: "AWS EKS" },
  { id: "etcd-cncf", label: "etcd v3.5" },
];

const DEPTHS: { id: DetailLevel; label: string; desc: string }[] = [
  { id: "brief", label: "Brief", desc: "2-3 sentences" },
  { id: "standard", label: "Standard", desc: "Balanced + snippet" },
  { id: "detailed", label: "Detailed", desc: "Full walkthrough" },
];

const SAMPLE_QUERIES = [
  "How does etcd Raft leader election prevent split-brain during network partitions?",
  "How do I configure Pod Topology Spread Constraints with maxSkew in Kubernetes?",
  "How does Amazon EKS Pod Identity differ from IRSA for IAM role assumption?",
];

const HISTORY_STORAGE_KEY = "sysarch_query_history_v1";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [service, setService] = useState<ServiceFilter>("all");
  const [detailLevel, setDetailLevel] = useState<DetailLevel>("standard");
  const [loading, setLoading] = useState(false);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [steps, setSteps] = useState<TraceStep[]>([]);
  const [cacheMeta, setCacheMeta] = useState<{ hit: boolean; tier: string } | null>(null);
  const [response, setResponse] = useState<RAGResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedAnswer, setCopiedAnswer] = useState(false);
  const [expandedRefs, setExpandedRefs] = useState<Record<string, boolean>>({});
  const [activeRefId, setActiveRefId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      if (raw) setHistory(JSON.parse(raw) as HistoryItem[]);
    } catch {
      // ignore corrupted storage
    }
  }, []);

  const saveHistory = (items: HistoryItem[]) => {
    setHistory(items);
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items.slice(0, 10)));
    } catch {
      // ignore quota errors
    }
  };

  const handleSubmit = async (overrideQuery?: string) => {
    const q = (overrideQuery ?? query).trim();
    if (!q || loading) return;
    if (overrideQuery) setQuery(overrideQuery);

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setLoading(true);
    setError(null);
    setResponse(null);
    setSteps([]);
    setCacheMeta(null);
    setActiveNode("cache_lookup");
    setExpandedRefs({});
    setActiveRefId(null);

    const collectedSteps: TraceStep[] = [];
    try {
      await streamRagQuery(
        q,
        service,
        detailLevel,
        {
          onTrace: (node) => setActiveNode(node),
          onStepEnd: (step) => {
            collectedSteps.push(step);
            setSteps([...collectedSteps]);
          },
          onCache: (hit, tier) => setCacheMeta({ hit, tier }),
          onFinal: (payload) => {
            setResponse(payload);
            const finalSteps = payload.trace && payload.trace.length > 0 ? payload.trace : collectedSteps;
            setSteps(finalSteps);
            setActiveNode(null);
            const item: HistoryItem = {
              id: payload.request_id || String(Date.now()),
              query: q,
              service,
              detailLevel,
              timestamp: new Date().toLocaleTimeString(),
              response: payload,
              steps: finalSteps,
            };
            saveHistory([item, ...history.filter((h) => h.query !== q)].slice(0, 10));
          },
          onError: (msg) => {
            setError(msg);
            setActiveNode(null);
          },
        },
        ctrl.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        setError((err as Error).message || "Network error");
      }
    } finally {
      setLoading(false);
      setActiveNode(null);
    }
  };

  const handleCopyAnswer = async () => {
    if (!response?.answer) return;
    await navigator.clipboard.writeText(response.answer);
    setCopiedAnswer(true);
    setTimeout(() => setCopiedAnswer(false), 1600);
  };

  const handleExportMarkdown = () => {
    if (!response) return;
    const sourcesMd = response.citations
      .map((c) => `- **[${c.ref_id}]** \`${c.file_path}\` (${c.section_hierarchy})${c.source_url ? ` — [Official Docs](${c.source_url})` : ""}`)
      .join("\n");
    const md = `# SysArch-RAG Technical Brief\n\n**Query:** ${query}\n**Service Scope:** ${service} | **Depth:** ${detailLevel} | **Tier:** ${response.cache_tier} | **Latency:** ${response.latency_ms} ms\n\n## Answer\n\n${response.answer}\n\n## Verified Sources\n\n${sourcesMd || "_No citations_"}\n`;
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `sysarch-rag-${response.request_id || "brief"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const focusCitation = (refId: string) => {
    setActiveRefId(refId);
    setExpandedRefs((prev) => ({ ...prev, [refId]: true }));
    document.getElementById(`source-${refId}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  const statusBadge = (status: TraceStep["status"]) => {
    if (status === "retry") return "bg-amber-500/15 text-amber-300 border-amber-500/40";
    if (status === "fallback") return "bg-orange-500/15 text-orange-300 border-orange-500/40";
    if (status === "error") return "bg-rose-500/15 text-rose-300 border-rose-500/40";
    return "bg-emerald-500/15 text-emerald-300 border-emerald-500/40";
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        <header className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800 pb-5">
          <div>
            <div className="flex items-center gap-2.5">
              <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-400" />
              <h1 className="text-xl font-semibold tracking-tight">SysArch-RAG Console</h1>
              <span className="rounded bg-zinc-900 px-2 py-0.5 font-mono text-[11px] text-zinc-400 ring-1 ring-zinc-800">
                Corrective RAG · Grounded
              </span>
            </div>
            <p className="mt-1 text-xs text-zinc-400">
              Self-correcting technical retrieval across Kubernetes, AWS EKS, and etcd official documentation.
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {SERVICES.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={() => setService(s.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                  service === s.id
                    ? "bg-sky-500 text-zinc-950 font-semibold"
                    : "bg-zinc-900 text-zinc-300 hover:bg-zinc-800 ring-1 ring-zinc-800"
                }`}
              >
                {s.label}
              </button>
            ))}
          </div>
        </header>

        <section className="mb-6 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <span className="text-xs font-medium text-zinc-400">Response Depth</span>
            <div className="flex gap-1.5">
              {DEPTHS.map((d) => (
                <button
                  key={d.id}
                  type="button"
                  onClick={() => setDetailLevel(d.id)}
                  className={`rounded-md px-2.5 py-1 text-xs transition ${
                    detailLevel === d.id
                      ? "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/50 font-medium"
                      : "bg-zinc-950 text-zinc-400 hover:text-zinc-200 ring-1 ring-zinc-800"
                  }`}
                >
                  {d.label} <span className="hidden sm:inline text-[10px] opacity-75">({d.desc})</span>
                </button>
              ))}
            </div>
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault();
              void handleSubmit();
            }}
            className="flex flex-col gap-3 sm:flex-row"
          >
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask an architecture, networking, IAM, or Raft consensus question..."
              className="flex-1 rounded-lg border border-zinc-800 bg-zinc-950 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 focus:border-sky-500 focus:outline-none"
            />
            <div className="flex gap-2">
              {loading && (
                <button
                  type="button"
                  onClick={() => abortRef.current?.abort()}
                  className="rounded-lg border border-rose-500/40 bg-rose-500/15 px-3.5 py-2.5 text-xs font-medium text-rose-300 hover:bg-rose-500/25"
                >
                  Stop
                </button>
              )}
              <button
                type="submit"
                disabled={loading || query.trim().length < 3}
                className="rounded-lg bg-sky-500 px-5 py-2.5 text-xs font-semibold text-zinc-950 hover:bg-sky-400 disabled:opacity-40 transition"
              >
                {loading ? "Running CRAG..." : "Ask Docs"}
              </button>
            </div>
          </form>

          <div className="mt-3 flex flex-wrap gap-2">
            {SAMPLE_QUERIES.map((sq) => (
              <button
                key={sq}
                type="button"
                onClick={() => void handleSubmit(sq)}
                className="rounded-full border border-zinc-800 bg-zinc-950/80 px-3 py-1 text-left text-[11px] text-zinc-400 hover:border-zinc-700 hover:text-zinc-200 transition"
              >
                {sq}
              </button>
            ))}
          </div>
        </section>

        {error && (
          <div className="mb-6 rounded-xl border border-rose-500/40 bg-rose-950/30 p-4 text-xs text-rose-200">
            <span className="font-semibold">Request Blocked / Error:</span> {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <div className="space-y-6 lg:col-span-2">
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800/80 pb-3">
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Grounded Answer</h2>
                  {cacheMeta && (
                    <span className="rounded bg-zinc-950 px-2 py-0.5 font-mono text-[11px] text-sky-300 ring-1 ring-zinc-800">
                      {cacheMeta.hit ? `CACHE HIT (${cacheMeta.tier})` : "LIVE CRAG"}
                    </span>
                  )}
                  {response && (
                    <span className="rounded bg-zinc-950 px-2 py-0.5 font-mono text-[11px] text-emerald-300 ring-1 ring-zinc-800">
                      {response.latency_ms} ms
                      {response.cache_hit && response.original_latency_ms
                        ? ` (saved ${Math.max(0, Math.round(response.original_latency_ms - response.latency_ms))} ms)`
                        : ""}
                    </span>
                  )}
                  {response && response.retry_count > 0 && (
                    <span className="rounded bg-amber-500/15 px-2 py-0.5 font-mono text-[11px] text-amber-300 ring-1 ring-amber-500/40">
                      Self-Corrected ({response.retry_count}x)
                    </span>
                  )}
                </div>
                {response && (
                  <div className="flex gap-1.5">
                    <button type="button" onClick={handleCopyAnswer} className="rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800">
                      {copiedAnswer ? "Copied!" : "Copy Answer"}
                    </button>
                    <button type="button" onClick={handleExportMarkdown} className="rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1 text-[11px] text-zinc-300 hover:bg-zinc-800">
                      Export .md
                    </button>
                  </div>
                )}
              </div>
              {loading && !response && (
                <div className="py-8 text-center text-xs text-zinc-400">
                  Executing node: <span className="font-mono text-sky-300">{activeNode || "initializing"}</span>...
                </div>
              )}
              {!loading && !response && !error && (
                <div className="py-8 text-center text-xs text-zinc-500">
                  Select a service filter and submit a query to inspect grounded documentation answers.
                </div>
              )}
              {response && (
                <div className={response.refused ? "rounded-lg border border-amber-500/30 bg-amber-950/20 p-3.5" : ""}>
                  <AnswerMarkdown answer={response.answer} onSelectCitation={focusCitation} />
                </div>
              )}
            </section>

            {response && response.citations.length > 0 && (
              <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
                <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-400">
                  Verified Evidence Sources ({response.citations.length})
                </h2>
                <div className="space-y-3">
                  {response.citations.map((c) => {
                    const isExpanded = !!expandedRefs[c.ref_id];
                    const isHighlighted = activeRefId === c.ref_id;
                    return (
                      <div key={c.ref_id} id={`source-${c.ref_id}`} className={`rounded-lg border p-3.5 transition ${isHighlighted ? "border-sky-500/60 bg-sky-950/20" : "border-zinc-800 bg-zinc-950/80"}`}>
                        <div className="flex flex-wrap items-center justify-between gap-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="rounded bg-sky-500/20 px-2 py-0.5 font-mono text-xs font-semibold text-sky-300">[{c.ref_id}]</span>
                            <span className="rounded bg-zinc-900 px-2 py-0.5 font-mono text-[11px] text-zinc-300">{c.service} · {c.version}</span>
                            <span className="font-mono text-[11px] text-zinc-400">{c.file_path}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {c.source_url && <a href={c.source_url} target="_blank" rel="noreferrer" className="text-[11px] text-sky-400 hover:underline">Official Doc ↗</a>}
                            <button type="button" onClick={() => setExpandedRefs((p) => ({ ...p, [c.ref_id]: !p[c.ref_id] }))} className="rounded border border-zinc-800 px-2 py-0.5 text-[11px] text-zinc-400 hover:text-zinc-200">
                              {isExpanded ? "Collapse" : "Full Excerpt"}
                            </button>
                          </div>
                        </div>
                        <p className="mt-1.5 text-[11px] font-medium text-zinc-400">{c.section_hierarchy}</p>
                        <pre className="mt-2 whitespace-pre-wrap rounded bg-zinc-900/70 p-2.5 font-mono text-xs text-zinc-300">{isExpanded ? (c.full_text || c.snippet) : c.snippet}</pre>
                      </div>
                    );
                  })}
                </div>
              </section>
            )}
          </div>

          <aside className="space-y-6">
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-400">CRAG Execution Trace</h2>
              {steps.length === 0 ? (
                <p className="text-xs text-zinc-500">Node timings and routing steps appear here during execution.</p>
              ) : (
                <ol className="space-y-2">
                  {steps.map((st, idx) => (
                    <li key={`${st.node}-${idx}`} className="rounded-lg border border-zinc-800 bg-zinc-950 p-2.5 text-xs">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono font-medium text-zinc-200">{st.node}</span>
                        <span className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${statusBadge(st.status)}`}>{st.status} · {st.duration_ms}ms</span>
                      </div>
                      <p className="mt-1 text-[11px] text-zinc-400">{st.summary}</p>
                    </li>
                  ))}
                </ol>
              )}
            </section>
            <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">Recent Sessions ({history.length})</h2>
                {history.length > 0 && <button type="button" onClick={() => saveHistory([])} className="text-[11px] text-zinc-500 hover:text-zinc-300">Clear</button>}
              </div>
              {history.length === 0 ? (
                <p className="text-xs text-zinc-500">Your last 10 verified queries are saved locally in this browser.</p>
              ) : (
                <ul className="space-y-2">
                  {history.map((item) => (
                    <li key={item.id}>
                      <button type="button" onClick={() => { setQuery(item.query); setService(item.service); setDetailLevel(item.detailLevel); setResponse(item.response); setSteps(item.steps); setCacheMeta({ hit: item.response.cache_hit, tier: item.response.cache_tier }); setError(null); }} className="w-full rounded-lg border border-zinc-800 bg-zinc-950 p-2.5 text-left hover:border-zinc-700 transition">
                        <p className="line-clamp-2 text-xs text-zinc-200">{item.query}</p>
                        <div className="mt-1 flex items-center justify-between font-mono text-[10px] text-zinc-500">
                          <span>{item.service} · {item.detailLevel}</span>
                          <span>{item.timestamp}</span>
                        </div>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}
