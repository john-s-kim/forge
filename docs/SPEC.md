# Forge: a miniature GPU training platform

Status: the original design. The README's Status table says what is built and
verified; where the two differ (for example checkpoints on local disk instead of
MinIO, and `forge submit -- <command>` instead of `forge submit train.py`), the
README and the code are current.

## 1. Purpose

Forge is a small, working version of an ML platform: engineers submit training
jobs, a scheduler places them on a GPU cluster, and a dashboard shows what is
running, what is waiting, and why.

The centrepiece is the scheduler. Everything else exists so the scheduler can
be used, observed and measured.

### Goals

- Submit, queue, place, run, preempt and resume training jobs end to end.
- Share a cluster fairly between teams with priorities and quotas.
- Show measurable scheduler quality against a baseline on a real cluster trace.
- Give users live visibility: metrics, logs, queue position, cluster occupancy.
- Diagnose failed runs, and measure how often the diagnosis is right.

### Non-goals

- Real multi-node GPU training performance. No GPUs are required.
- Hyperparameter search, model registry, dataset versioning, notebooks.
- Multi-cluster or multi-region operation, high availability of the control plane.
- Replacing kube-scheduler in general; Forge only places Forge jobs.

## 2. User flow

```bash
forge submit train.py --gpus 4 --workers 2 --priority high --team vision
forge status <job-id>
forge logs <job-id> --follow
forge cancel <job-id>
```

Inside the training script:

```python
import forge

run = forge.init()                       # reads job id and API key from env
start = run.restore(model, optimizer)    # returns step to resume from, or 0

for step in range(start, total_steps):
    loss = train_step()
    run.log(step, loss=loss)
    if run.should_checkpoint(step):      # periodic, or SIGTERM received
        run.checkpoint(step, model, optimizer)
```

## 3. Architecture

```
CLI / SDK ──HTTP──> API (Flask) ──> Postgres   (source of truth)
                     │   │
                     │   └──> Redis            (metric/log tails, queue cache)
                     │
                     └──> NATS  <── scheduler controller ──> Kubernetes API
                           │
                           ├──> dashboard gateway (SSE to React)
                           ├──> alert evaluator
                           └──> triage
Training pods ──> API (metrics, heartbeats) and MinIO (checkpoints)
```

| Component | Tech | Responsibility |
|---|---|---|
| API | Python, Flask, SQLAlchemy | Jobs, runs, teams, quotas, artifacts, auth |
| Scheduler core | Pure Python library | Given cluster state and queue, return placements and preemptions |
| Controller | Python, Kubernetes client | Watches pods and nodes, calls the core, applies decisions |
| Simulator | Python | Drives the same core from a trace with a virtual clock |
| Postgres | | Jobs, attempts, quota ledger, artifact metadata, audit log |
| Redis | | Last N metric points and log lines per run, queue-position cache |
| NATS | | Job state transition events |
| MinIO | | Checkpoints and model files |
| Dashboard | React, TypeScript, Vite | Cluster, queue and run views |
| Monitoring | Prometheus, Grafana | Platform and job health |
| Triage | Python | Failure diagnosis |

Design rule: the scheduler core has no I/O. The controller and the simulator
are two drivers of the same code, so the benchmarked scheduler is the deployed
scheduler.

## 4. Scheduler

### 4.1 Job state machine

```
SUBMITTED -> QUEUED -> ADMITTED -> STARTING -> RUNNING -> SUCCEEDED
                ^                                 |    -> FAILED
                |                                 v    -> CANCELLED
                +--------- PREEMPTING <-----------+
```

Every transition is written to Postgres in one transaction with an audit row,
then published to NATS. Each placement of a job is an *attempt*; a preempted
job keeps its id and gains a new attempt.

### 4.2 Resources and placement

- A job requests `workers × gpus_per_worker`, plus CPU and memory per worker.
- Nodes advertise `nvidia.com/gpu`. Forge tracks free GPUs per node itself.
- Placement is bin-packing: prefer the node that leaves the least free GPUs,
  to keep large contiguous blocks available.
- Forge binds pods by setting `spec.nodeName`, bypassing kube-scheduler for
  Forge pods.
- Gang scheduling: a multi-worker job is placed all-or-nothing.

### 4.3 Ordering

Queue order is decided by, in order:

1. Priority class: `high`, `normal`, `low`.
2. Whether the team is within its guaranteed quota (within-quota first).
3. Effective submit time, with aging so low-priority jobs cannot starve.

### 4.4 Quotas and borrowing

- Each team has a guaranteed GPU quota. The sum of guarantees is at most the
  cluster capacity.
- A team may borrow idle GPUs beyond its guarantee. Borrowed attempts are
  marked as such.
- When a team reclaims its guarantee, borrowed attempts are the first victims.

### 4.5 Preemption

Triggered when the head-of-queue job cannot be placed and is entitled to
capacity (higher priority, or reclaiming quota).

Victim selection, in order:

1. Borrowed attempts before within-quota attempts.
2. Lowest priority class.
3. Least work lost: most recent checkpoint first.

Guards:

- Minimum runtime before an attempt is eligible (default 5 minutes).
- Per-job preemption cap per hour, after which it is pinned.
- Choose the smallest victim set that frees enough GPUs on suitable nodes.

Mechanism: the controller deletes the pod with a grace period. The SDK catches
SIGTERM, writes a checkpoint, and exits. The job returns to QUEUED and the next
attempt receives `FORGE_RESUME_FROM` pointing at the checkpoint.

### 4.6 Backfill (stretch)

When the head-of-queue job is waiting for capacity, reserve its start time and
let smaller jobs run if their declared time limit ends before that reservation.

### 4.7 Baselines for comparison

- FIFO, no preemption.
- Strict priority, no quotas.
- Forge policy (priority, quota with borrowing, preemption), with and without backfill.

## 5. Data model

| Table | Key columns |
|---|---|
| `teams` | id, name, gpu_quota |
| `users` | id, team_id, role (`member`, `admin`), api_key_hash |
| `jobs` | id, team_id, user_id, spec (JSON), priority, state, submitted_at |
| `attempts` | id, job_id, node assignments, borrowed, started_at, ended_at, exit_reason |
| `checkpoints` | id, job_id, attempt_id, step, uri, created_at |
| `artifacts` | id, job_id, name, uri, size, sha256 |
| `events` | id, job_id, from_state, to_state, reason, at |
| `diagnoses` | id, attempt_id, category, summary, proposed_action, approved_by |

Metric time series live in Redis for the live tail and are flushed in batches
to a Postgres `metrics` table (run id, step, name, value) for history.

## 6. API

All endpoints are under `/v1`, take `Authorization: Bearer <key>`, and return JSON.

| Method and path | Purpose |
|---|---|
| `POST /jobs` | Submit a job |
| `GET /jobs`, `GET /jobs/{id}` | List and inspect, including queue position and wait reason |
| `POST /jobs/{id}/cancel` | Cancel |
| `POST /jobs/{id}/metrics` | Batched metric points from the SDK |
| `POST /jobs/{id}/heartbeat` | Liveness from the SDK |
| `POST /jobs/{id}/checkpoints` | Register a checkpoint, returns a signed upload URL |
| `GET /jobs/{id}/logs` | Log tail; `?follow=1` streams over SSE |
| `GET /cluster` | Nodes, GPU occupancy, per-team usage against quota |
| `GET /stream` | SSE feed of state transitions for the dashboard |
| `GET/PUT /teams/{id}/quota` | Admin only |
| `GET /jobs/{id}/diagnosis`, `POST .../approve` | Diagnosis and approval |

Every job answers the question "why am I not running?" with a machine-readable
wait reason: `quota_exceeded`, `insufficient_gpus`, `gang_unsatisfied`,
`reserved_for_job`.

## 7. Dashboard

- **Cluster:** grid of nodes, one cell per GPU, coloured by team; hover shows the job.
  Per-team usage bars against quota, with borrowed capacity marked.
- **Queue:** ordered list with priority, wait time, wait reason, estimated start.
- **Run:** state timeline with attempts and preemptions, live loss curves, live
  logs, checkpoints, artifacts, and the diagnosis if the run failed.
- **Compare (stretch):** overlay metrics from several runs.

Live updates come from one SSE connection. State is held in a normalized store
keyed by job id so events patch rows in place.

## 8. Monitoring

Prometheus metrics exported by the API and controller:

- `forge_queue_wait_seconds` histogram, by team and priority
- `forge_gpus_allocated`, `forge_gpus_total`, by team
- `forge_preemptions_total`, by reason
- `forge_scheduling_cycle_seconds` histogram
- `forge_job_state_transitions_total`

Alerts:

| Alert | Condition |
|---|---|
| Stalled run | RUNNING with no heartbeat or metric for 5 minutes |
| Idle allocation | GPUs allocated but the job has not logged a step for 10 minutes |
| Diverged run | Loss is NaN or infinite |

## 9. Failure triage

Triggered by a `FAILED` transition event.

A rule-based diagnoser reads the exit status, log tail, metrics and cluster
events and stores a category (`oom`, `diverged`, `bad_input_path`,
`image_pull`, `time_limit`, `user_code_error`, `infra`, `unknown`), a short
summary with the evidence lines quoted, and optionally a proposed spec patch.
It cannot submit or cancel jobs; a person approves a patch before anything is rerun.

### Eval harness

- About 20 labelled failures across the categories above.
- Score: category accuracy.
- Runs in CI, results reported in the README.

## 10. Security

- API keys: 32 random bytes, shown once, stored as a salted hash.
- Roles: `member` sees and controls their team's jobs; `admin` sets quotas.
- One namespace per team. Default-deny NetworkPolicy; training pods may reach
  only the API and MinIO.
- Training pods run as non-root with no service account token mounted.
- User code never runs in the control plane.
- Artifact access through short-lived signed URLs.
- Rate limiting on submit and metrics endpoints.
- Audit log of every state change and admin action.

## 11. Running without GPUs

- A local backend that runs jobs as subprocesses on made-up GPU nodes.
- One real workload: a small ResNet on CIFAR-10, to prove SDK logging,
  checkpoint on SIGTERM, and resume.
- Trace replay runs in the simulator with a virtual clock, no cluster needed.

## 12. Evaluation

Replay a public GPU cluster trace (Microsoft Philly or Alibaba's GPU trace;
confirm format and licence before committing to one) through the simulator.

Report, for each policy in 4.7:

- GPU utilization over time
- Queue wait p50, p95, p99, overall and by priority class
- Preemptions per job and GPU-hours of work lost to preemption
- Fairness: each team's usage against its guarantee

## 13. Testing and delivery

- Unit tests for the scheduler core, including property tests: never
  over-allocate a node, never exceed total capacity, gangs are all-or-nothing,
  quota guarantees are honoured once reclaimed.
- API tests against a real Postgres in a container.
- End-to-end test in CI on the local backend: submit, run, preempt, resume, succeed.
- GitHub Actions: lint and type-check (ruff, mypy, eslint, tsc), tests, image
  build, e2e.
- Helm chart for the whole stack; `make dev` brings up a local cluster.

## 14. Repo layout

```
forge/
  api/            Flask app, models, migrations
  scheduler/
    core/         pure policy code
    controller/   Kubernetes driver
    sim/          trace replay driver
  sdk/            Python SDK and CLI
  triage/         failure diagnosis and eval harness
  web/            React dashboard
  deploy/         Helm chart, alert rules
  bench/          trace loaders, result notebooks
  examples/       CIFAR job, seeded failure jobs
```

## 15. Milestones

| Week | Deliverable | Done when |
|---|---|---|
| 1 | Submit, queue, run, metrics | A job submitted from the CLI runs and its loss shows in a bare UI |
| 2 | Priorities, quotas, preemption, resume | e2e test preempts a job and it resumes from its checkpoint |
| 3 | Simulator, trace benchmark, live dashboard, alerts | Results table for all policies; cluster and queue views update live |
| 4 | Triage and eval, Helm, CI, README, demo video | Eval table in README; fresh clone to running demo with one command |

Cut order if time runs short: run comparison, backfill, gang scheduling. Keep
preemption with resume, the trace benchmark and the triage eval.

## 16. Open questions

- Which trace: Philly has job durations and GPU counts with a simpler format;
  Alibaba's is larger and has more resource detail.
- Flask with SSE, or switch the gateway to an async framework if connection
  counts become a problem.
- Whether to add a small C++ or Rust scheduler core later for cycle latency;
  only if the Python core is measurably the bottleneck.
