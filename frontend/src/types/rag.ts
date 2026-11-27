export type ServiceFilter = "kubernetes" | "aws-eks" | "etcd-cncf" | null;

export interface Citation {
  ref_id: string;
  service: string;
  version: string;
  file_path: string;
  section_hierarchy: string;
  source_url: string;
  snippet: string;
}

export interface TraceStep {
  node: string;
  duration_ms: number;
  summary: string;
}

export interface RAGResponse {
  request_id: string;
  answer: string;
  citations: Citation[];
  services_searched: string[];
  retry_count: number;
  refused: boolean;
  refusal_reason: string | null;
  cache_hit: boolean;
  cache_tier: "exact_redis" | "semantic_qdrant" | "live_crag";
  latency_ms: number;
  original_latency_ms: number | null;
  trace: TraceStep[];
}

export type SSEEvent =
  | { type: "start"; request_id: string; query: string }
  | { type: "cache"; hit: boolean; tier: RAGResponse["cache_tier"] }
  | { type: "trace"; node: string }
  | { type: "node_start"; node: string }
  | { type: "node_end"; node: string; duration_ms: number; summary: string }
  | { type: "final"; payload: RAGResponse }
  | { type: "error"; detail: string };
