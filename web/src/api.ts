export type JobState =
  | "QUEUED"
  | "STARTING"
  | "RUNNING"
  | "PREEMPTING"
  | "SUCCEEDED"
  | "FAILED"
  | "CANCELLED";

export interface Job {
  id: string;
  name: string;
  team: string;
  user: string;
  state: JobState;
  priority: "low" | "normal" | "high";
  workers: number;
  gpus_per_worker: number;
  gpus: number;
  submitted_at: number;
  started_at: number | null;
  finished_at: number | null;
  wait_reason: string | null;
  queue_position: number | null;
  nodes: string[];
  borrowed: boolean;
  attempts: number;
  retry_of: string | null;
}

export interface JobDetail extends Job {
  spec: { command: string[]; image: string | null; time_limit: number | null };
  events: { from: string | null; to: JobState; reason: string | null; actor: string; at: number }[];
  checkpoints: { step: number; size: number; created_at: number }[];
  attempt_history: {
    number: number;
    nodes: string[];
    started_at: number;
    ended_at: number | null;
    exit_code: number | null;
    exit_reason: string | null;
  }[];
}

export interface Allocation {
  job_id: string;
  name: string;
  team: string;
  gpus: number;
  borrowed: boolean;
  state: JobState;
  priority: string;
}

export interface Cluster {
  nodes: { name: string; gpus: number; allocations: Allocation[] }[];
  teams: { name: string; quota: number; used: number; borrowed: number }[];
}

export interface Point {
  name: string;
  step: number;
  value: number | null; // null: the job logged NaN or infinity
  at: number;
}

export interface Diagnosis {
  category: string;
  summary: string;
  evidence: string[];
  proposed_patch: Record<string, unknown> | null;
  approved_by: string | null;
  resubmitted_as: string | null;
}

export type StreamEvent =
  | { type: "job.state"; job_id: string; to: JobState }
  | { type: "job.metrics"; job_id: string; points: Point[] }
  | { type: "job.log"; job_id: string; line: string }
  | { type: "job.diagnosis"; job_id: string };

const KEY = "forge.apiKey";

export function getKey(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setKey(key: string | null): void {
  try {
    if (key) localStorage.setItem(KEY, key);
    else localStorage.removeItem(KEY);
  } catch {
    // Storage unavailable: the key lasts for this page load only.
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function api<T>(key: string, path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { ...init?.headers, Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new ApiError(res.status, body.error ?? res.statusText);
  }
  return (await res.json()) as T;
}

/**
 * Server-sent events over fetch. EventSource cannot send an Authorization
 * header, and putting the key in the URL would leak it into logs.
 * Reconnects with a short delay until the returned function is called.
 */
export function stream(key: string, job: string | null, onEvent: (e: StreamEvent) => void): () => void {
  const abort = new AbortController();
  const run = async () => {
    while (!abort.signal.aborted) {
      try {
        const res = await fetch(job ? `/v1/stream?job=${encodeURIComponent(job)}` : "/v1/stream", {
          headers: { Authorization: `Bearer ${key}` },
          signal: abort.signal,
        });
        if (!res.ok || !res.body) throw new Error(`stream: ${res.status}`);
        const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
        let buffer = "";
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += value;
          let end: number;
          while ((end = buffer.indexOf("\n\n")) >= 0) {
            const frame = buffer.slice(0, end);
            buffer = buffer.slice(end + 2);
            const data = frame.split("\n").find((l) => l.startsWith("data: "));
            if (data) onEvent(JSON.parse(data.slice(6)) as StreamEvent);
          }
        }
      } catch {
        // fall through to retry
      }
      await new Promise((r) => setTimeout(r, 2000));
    }
  };
  void run();
  return () => abort.abort();
}

export function ago(ts: number | null, now = Date.now() / 1000): string {
  if (ts == null) return "–";
  return duration(now - ts);
}

export function duration(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  if (s < 90) return `${s}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

export const WAIT_REASONS: Record<string, string> = {
  quota_exceeded: "Team is over its quota and nothing is idle to borrow",
  insufficient_gpus: "Cluster is full",
  gang_unsatisfied: "Enough GPUs are free, but not together on the nodes it needs",
  waiting_for_preemption: "Lower-priority work is checkpointing to make room",
  reserved_for_job: "Capacity is being held for a job ahead in the queue",
  unschedulable: "Asks for more GPUs than this cluster has",
};
