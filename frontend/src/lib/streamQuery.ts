import type { DetailLevel, RAGResponse, ServiceFilter, StreamEvent, TraceStep } from "@/types/rag";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";

export function getOrCreateSessionId(): string {
  if (typeof window === "undefined") return "ssr-session";
  const key = "sysarch_session_id";
  let id = window.localStorage.getItem(key);
  if (!id || !/^[\w-]{1,64}$/.test(id)) {
    id = `sess_${crypto.randomUUID().replace(/-/g, "").slice(0, 16)}`;
    window.localStorage.setItem(key, id);
  }
  return id;
}

export interface StreamHandlers {
  onTrace?: (node: string) => void;
  onStepEnd?: (step: TraceStep) => void;
  onCache?: (hit: boolean, tier: RAGResponse["cache_tier"]) => void;
  onFinal?: (payload: RAGResponse) => void;
  onError?: (message: string) => void;
}

export async function streamRagQuery(
  query: string,
  serviceFilter: ServiceFilter,
  detailLevel: DetailLevel,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const sessionId = getOrCreateSessionId();
  const body: Record<string, string> = {
    query,
    detail_level: detailLevel,
    session_id: sessionId,
  };
  if (serviceFilter !== "all") {
    body.service_filter = serviceFilter;
  }

  const res = await fetch(`${API_BASE}/v1/query/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      "X-Session-Id": sessionId,
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    let msg = `HTTP ${res.status}`;
    const retryAfter = res.headers.get("Retry-After");
    try {
      const errJson = await res.json();
      if (errJson?.detail) msg = String(errJson.detail);
    } catch {
      // ignore non-JSON body
    }
    if (retryAfter && !msg.includes(retryAfter)) {
      msg = `${msg} (retry in ${retryAfter}s)`;
    }
    handlers.onError?.(msg);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const dataLine = frame
        .split("\n")
        .find((line) => line.startsWith("data: "));
      if (!dataLine) continue;

      const evt = JSON.parse(dataLine.slice(6)) as StreamEvent;
      if (evt.type === "trace") handlers.onTrace?.(evt.node);
      if (evt.type === "node_end") {
        handlers.onStepEnd?.({
          node: evt.node,
          status: evt.status,
          duration_ms: evt.duration_ms,
          summary: evt.summary,
        });
      }
      if (evt.type === "cache") handlers.onCache?.(evt.hit, evt.tier);
      if (evt.type === "final") handlers.onFinal?.(evt.payload);
      if (evt.type === "error") handlers.onError?.(evt.detail);
    }
  }
}
