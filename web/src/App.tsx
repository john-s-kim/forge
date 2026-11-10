import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ago,
  api,
  ApiError,
  type Cluster,
  type Diagnosis,
  duration,
  getKey,
  type Job,
  type JobDetail,
  type Point,
  setKey,
  stream,
  WAIT_REASONS,
} from "./api";
import { LineChart } from "./LineChart";

const ACTIVE = ["STARTING", "RUNNING", "PREEMPTING"];

function useHashRoute(): string | null {
  const read = () => window.location.hash.replace(/^#\/?/, "").replace(/^jobs\//, "") || null;
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export function App() {
  const [key, setKeyState] = useState(getKey);
  const jobId = useHashRoute();
  const signOut = useCallback(() => {
    setKey(null);
    setKeyState(null);
  }, []);

  if (!key) {
    return (
      <SignIn
        onKey={(k) => {
          setKey(k);
          setKeyState(k);
        }}
      />
    );
  }
  return (
    <>
      <header>
        <a href="#/" className="brand">
          Forge
        </a>
        <button className="link" onClick={signOut}>
          Sign out
        </button>
      </header>
      <main>
        {jobId ? <Run key={jobId} apiKey={key} id={jobId} /> : <Overview apiKey={key} onDenied={signOut} />}
      </main>
    </>
  );
}

function SignIn({ onKey }: { onKey: (k: string) => void }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api(value.trim(), "/v1/me");
      onKey(value.trim());
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? "That key was not accepted." : "Cannot reach the API.");
    }
  };
  return (
    <form className="signin" onSubmit={submit}>
      <h1>Forge</h1>
      <label htmlFor="key">API key</label>
      <input id="key" type="password" autoFocus value={value} onChange={(e) => setValue(e.target.value)} />
      {error && <p className="error">{error}</p>}
      <button type="submit" disabled={!value.trim()}>
        Sign in
      </button>
    </form>
  );
}

// ---- overview ---------------------------------------------------------------

function Overview({ apiKey, onDenied }: { apiKey: string; onDenied: () => void }) {
  const [cluster, setCluster] = useState<Cluster | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    try {
      const [c, j] = await Promise.all([
        api<Cluster>(apiKey, "/v1/cluster"),
        api<{ jobs: Job[] }>(apiKey, "/v1/jobs"),
      ]);
      setCluster(c);
      setJobs(j.jobs);
      setError("");
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) onDenied();
      else setError("Cannot reach the API. Retrying.");
    }
  }, [apiKey, onDenied]);

  useEffect(() => {
    void refresh();
    // State changes arrive in bursts (a preemption is several); coalesce them.
    let timer: number | undefined;
    const stop = stream(apiKey, null, (e) => {
      if (e.type !== "job.state") return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => void refresh(), 150);
    });
    const poll = window.setInterval(() => void refresh(), 10000);
    return () => {
      stop();
      window.clearInterval(poll);
      window.clearTimeout(timer);
    };
  }, [apiKey, refresh]);

  const teams = useMemo(() => cluster?.teams.map((t) => t.name) ?? [], [cluster]);
  const queued = jobs
    .filter((j) => j.state === "QUEUED")
    .sort((a, b) => (a.queue_position ?? 1e9) - (b.queue_position ?? 1e9));
  const running = jobs.filter((j) => ACTIVE.includes(j.state));
  const finished = jobs.filter((j) => !ACTIVE.includes(j.state) && j.state !== "QUEUED").slice(0, 15);

  return (
    <>
      {error && <p className="error">{error}</p>}
      {cluster && <ClusterView cluster={cluster} teams={teams} />}
      <section>
        <h2>Queue</h2>
        {queued.length === 0 ? (
          <p className="muted">Nothing is waiting.</p>
        ) : (
          <JobTable jobs={queued} teams={teams} columns={["position", "waiting"]} />
        )}
      </section>
      <section>
        <h2>Running</h2>
        {running.length === 0 ? (
          <p className="muted">Nothing is running.</p>
        ) : (
          <JobTable jobs={running} teams={teams} columns={["nodes"]} />
        )}
      </section>
      {finished.length > 0 && (
        <section>
          <h2>Recently finished</h2>
          <JobTable jobs={finished} teams={teams} columns={[]} />
        </section>
      )}
    </>
  );
}

function ClusterView({ cluster, teams }: { cluster: Cluster; teams: string[] }) {
  const total = cluster.nodes.reduce((n, x) => n + x.gpus, 0);
  const used = cluster.teams.reduce((n, t) => n + t.used, 0);
  return (
    <section>
      <h2>
        Cluster{" "}
        <span className="muted">
          {used} of {total} GPUs in use
        </span>
      </h2>
      <div className="cluster">
        <div className="nodes">
          {cluster.nodes.map((node) => {
            const cells = node.allocations.flatMap((a) => Array.from({ length: a.gpus }, () => a));
            return (
              <div className="node" key={node.name}>
                <span className="node-name">{node.name}</span>
                <span className="gpus">
                  {Array.from({ length: node.gpus }, (_, i) => {
                    const a = cells[i];
                    if (!a) return <span key={i} className="gpu free" title="Free" />;
                    return (
                      <a
                        key={i}
                        href={`#/jobs/${a.job_id}`}
                        className={`gpu team-${teams.indexOf(a.team) % 8}${a.borrowed ? " borrowed" : ""}`}
                        title={`${a.name} (${a.team}, ${a.priority}${a.borrowed ? ", borrowed" : ""}): ${a.state.toLowerCase()}`}
                      >
                        {a.team[0]?.toUpperCase()}
                      </a>
                    );
                  })}
                </span>
              </div>
            );
          })}
        </div>
        <table className="teams">
          <thead>
            <tr>
              <th>Team</th>
              <th>Use against quota</th>
              <th className="num">Used</th>
              <th className="num">Quota</th>
            </tr>
          </thead>
          <tbody>
            {cluster.teams.map((t) => {
              const scale = Math.max(t.quota, t.used, 1);
              return (
                <tr key={t.name}>
                  <td>
                    <span className={`swatch team-${teams.indexOf(t.name) % 8}`} /> {t.name}
                  </td>
                  <td>
                    <div className="quota-bar" title={`${t.used - t.borrowed} within quota, ${t.borrowed} borrowed`}>
                      <span
                        className={`fill team-${teams.indexOf(t.name) % 8}`}
                        style={{ width: `${((t.used - t.borrowed) / scale) * 100}%` }}
                      />
                      <span
                        className={`fill borrowed team-${teams.indexOf(t.name) % 8}`}
                        style={{ width: `${(t.borrowed / scale) * 100}%` }}
                      />
                      <span className="quota-mark" style={{ left: `${(t.quota / scale) * 100}%` }} />
                    </div>
                  </td>
                  <td className="num">{t.used}</td>
                  <td className="num">{t.quota}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="legend muted">
        <span className="gpu free" /> free <span className="gpu team-0 borrowed" /> hatched: borrowed beyond the
        team's quota, first to be preempted
      </p>
    </section>
  );
}

type Column = "position" | "waiting" | "nodes";

function JobTable({ jobs, teams, columns }: { jobs: Job[]; teams: string[]; columns: Column[] }) {
  const has = (c: Column) => columns.includes(c);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {has("position") && <th className="num">#</th>}
            <th>Job</th>
            <th>Team</th>
            <th>State</th>
            <th>Priority</th>
            <th className="num">GPUs</th>
            <th className="num">Age</th>
            {has("waiting") && <th>Waiting on</th>}
            {has("nodes") && <th>Nodes</th>}
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.id}>
              {has("position") && <td className="num">{j.queue_position ?? "–"}</td>}
              <td>
                <a href={`#/jobs/${j.id}`}>{j.name}</a> <span className="muted mono">{j.id}</span>
              </td>
              <td>
                <span className={`swatch team-${teams.indexOf(j.team) % 8}`} /> {j.team}
              </td>
              <td>
                <State state={j.state} />
              </td>
              <td>{j.priority}</td>
              <td className="num">
                {j.workers > 1 ? `${j.workers} × ${j.gpus_per_worker}` : j.gpus}
              </td>
              <td className="num">{ago(j.submitted_at)}</td>
              {has("waiting") && <td>{(j.wait_reason && WAIT_REASONS[j.wait_reason]) ?? "–"}</td>}
              {has("nodes") && (
                <td>
                  {[...new Set(j.nodes)].join(", ")}
                  {j.borrowed && <span className="muted"> (borrowed)</span>}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const STATE_ICON: Record<string, string> = {
  SUCCEEDED: "✓",
  FAILED: "✕",
  CANCELLED: "–",
  RUNNING: "▶",
  STARTING: "…",
  PREEMPTING: "⏸",
  QUEUED: "·",
};

function State({ state }: { state: string }) {
  return (
    <span className={`state state-${state.toLowerCase()}`}>
      <span aria-hidden="true">{STATE_ICON[state]}</span> {state.toLowerCase()}
    </span>
  );
}

// ---- one run ----------------------------------------------------------------

function Run({ apiKey, id }: { apiKey: string; id: string }) {
  const [job, setJob] = useState<JobDetail | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [lines, setLines] = useState<string[]>([]);
  const [diagnosis, setDiagnosis] = useState<Diagnosis | null>(null);
  const [error, setError] = useState("");
  const logBox = useRef<HTMLPreElement>(null);

  const loadJob = useCallback(async () => {
    try {
      setJob(await api<JobDetail>(apiKey, `/v1/jobs/${id}`));
    } catch (e) {
      setError(e instanceof ApiError && e.status === 404 ? "No such job." : "Cannot reach the API.");
    }
  }, [apiKey, id]);
  const loadDiagnosis = useCallback(async () => {
    try {
      setDiagnosis(await api<Diagnosis>(apiKey, `/v1/jobs/${id}/diagnosis`));
    } catch {
      // No diagnosis yet.
    }
  }, [apiKey, id]);

  useEffect(() => {
    void loadJob();
    void loadDiagnosis();
    void api<{ points: Point[] }>(apiKey, `/v1/jobs/${id}/metrics?full=1`).then((r) => setPoints(r.points));
    void api<{ lines: string[] }>(apiKey, `/v1/jobs/${id}/logs?tail=500`).then((r) => setLines(r.lines));
    return stream(apiKey, id, (e) => {
      if (e.type === "job.metrics") setPoints((p) => [...p, ...e.points]);
      else if (e.type === "job.log") setLines((l) => [...l.slice(-1999), e.line]);
      else if (e.type === "job.state") void loadJob();
      else if (e.type === "job.diagnosis") void loadDiagnosis();
    });
  }, [apiKey, id, loadJob, loadDiagnosis]);

  useEffect(() => {
    const el = logBox.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [lines]);

  const series = useMemo(() => {
    const by = new Map<string, Map<number, number | null>>();
    for (const p of points) {
      if (!by.has(p.name)) by.set(p.name, new Map());
      by.get(p.name)!.set(p.step, p.value);
    }
    return [...by].map(([name, m]) => ({
      name,
      points: [...m].map(([step, value]) => ({ step, value })).sort((a, b) => a.step - b.step),
    }));
  }, [points]);

  if (error) return <p className="error">{error}</p>;
  if (!job) return <p className="muted">Loading…</p>;

  const act = async (path: string) => {
    try {
      const r = await api<{ id: string }>(apiKey, path, { method: "POST" });
      if (path.endsWith("approve")) window.location.hash = `#/jobs/${r.id}`;
      else void loadJob();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed.");
    }
  };

  return (
    <>
      <section>
        <h1>
          {job.name} <span className="muted mono">{job.id}</span>
        </h1>
        <p>
          <State state={job.state} /> · {job.team} · {job.priority} priority ·{" "}
          {job.workers > 1 ? `${job.workers} workers × ${job.gpus_per_worker} GPUs` : `${job.gpus} GPUs`}
          {job.retry_of && (
            <>
              {" "}
              · retry of <a href={`#/jobs/${job.retry_of}`}>{job.retry_of}</a>
            </>
          )}
        </p>
        {job.state === "QUEUED" && job.wait_reason && (
          <p className="callout">
            Position {job.queue_position} in the queue. {WAIT_REASONS[job.wait_reason]}.
          </p>
        )}
        <p className="mono cmd">{job.spec.command.join(" ")}</p>
        {["QUEUED", ...ACTIVE].includes(job.state) && (
          <button onClick={() => void act(`/v1/jobs/${id}/cancel`)}>Cancel job</button>
        )}
      </section>

      {diagnosis && (
        <section className="diagnosis">
          <h2>
            Why it failed <span className="muted">{diagnosis.category.replace(/_/g, " ")}</span>
          </h2>
          <p>{diagnosis.summary}</p>
          {diagnosis.evidence.length > 0 && (
            <pre className="evidence">{diagnosis.evidence.join("\n")}</pre>
          )}
          {diagnosis.proposed_patch && (
            <>
              <p>Proposed change to the job:</p>
              <pre className="evidence">{JSON.stringify(diagnosis.proposed_patch, null, 2)}</pre>
              {diagnosis.resubmitted_as ? (
                <p>
                  Resubmitted as <a href={`#/jobs/${diagnosis.resubmitted_as}`}>{diagnosis.resubmitted_as}</a> by{" "}
                  {diagnosis.approved_by}.
                </p>
              ) : (
                <button onClick={() => void act(`/v1/jobs/${id}/diagnosis/approve`)}>
                  Resubmit with this change
                </button>
              )}
            </>
          )}
          <p className="muted small">
            Diagnosed by rule matching. Nothing is rerun until someone approves it.
          </p>
        </section>
      )}

      {series.length > 0 && (
        <section>
          <h2>Metrics</h2>
          <div className="charts">
            {series.map((s) => (
              <LineChart key={s.name} title={s.name} points={s.points} />
            ))}
          </div>
        </section>
      )}

      <section>
        <h2>Timeline</h2>
        <ol className="timeline">
          {job.events.map((e, i) => {
            const next = job.events[i + 1];
            return (
              <li key={i}>
                <span className="mono muted">{new Date(e.at * 1000).toLocaleTimeString()}</span>
                <State state={e.to} />
                <span className="muted">
                  {e.reason ?? ""}
                  {next ? ` · ${duration(next.at - e.at)}` : ""}
                </span>
              </li>
            );
          })}
        </ol>
        {job.checkpoints.length > 0 && (
          <p className="muted">
            {job.checkpoints.length} checkpoint{job.checkpoints.length > 1 ? "s" : ""}, latest at step{" "}
            {job.checkpoints[job.checkpoints.length - 1]!.step}.
          </p>
        )}
      </section>

      <section>
        <h2>Logs</h2>
        <pre className="logs" ref={logBox}>
          {lines.length ? lines.join("\n") : "No output yet."}
        </pre>
      </section>
    </>
  );
}
