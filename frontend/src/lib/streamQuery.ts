import { ServiceFilter, SSEEvent } from "@/types/rag";

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export async function streamRagQuery(
  params: { query: string; serviceFilter: ServiceFilter; sessionId?: string },
  onEvent: (event: SSEEvent) => void,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/v1/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      query: params.query,
      service_filter: params.serviceFilter,
      session_id: params.sessionId || undefined,
    }),
    signal,
  });

  if (!res.ok) {
    const retryAfter = res.headers.get("Retry-After");
    let detail = `Request failed (${res.status})`;
    try {
      const errJson = await res.json();
      if (errJson?.detail) detail = String(errJson.detail);
    } catch {
      // ignore non-JSON error body
    }
    if (res.status === 429 && retryAfter) {
      detail = `${detail} (Retry in ${retryAfter}s)`;
    }
    throw new Error(detail);
  }

  if (!res.body) {
    throw new Error("Streaming response body missing.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        onEvent(JSON.parse(raw) as SSEEvent);
      }
    }
  }
}
