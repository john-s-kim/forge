# Forge

A miniature GPU training platform: submit training jobs, have a scheduler
place them on a shared cluster, watch them live, and get a diagnosis when one
fails.

The centrepiece is the scheduler. It shares a cluster between teams with
priorities and quotas, lets a team borrow idle capacity, and takes it back by
preempting jobs that then checkpoint and later resume where they left off.

```
forge submit --gpus 4 --priority high -- python train.py --lr 0.1
```

## Try it

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and Node 22. No GPUs and no
cluster: `forge dev` runs jobs as subprocesses on made-up GPU nodes.

```bash
uv venv && uv pip install -e ".[dev]"
(cd web && npm ci && npm run build)

export FORGE_MIN_RUNTIME=5                       # let the demo preempt quickly
export FORGE_NODES=node-0=8,node-1=8,node-2=8,node-3=8
export FORGE_API_KEY=$(.venv/bin/forge admin bootstrap --quota 16)
.venv/bin/forge dev                              # API, scheduler and dashboard on :8080
```

In a second terminal, with the same `FORGE_API_KEY`:

```bash
.venv/bin/python examples/seed_demo.py           # three teams, a queue, a preemption, two failures
.venv/bin/forge cluster
.venv/bin/forge ls
.venv/bin/forge status <job-id>                  # includes why a queued job is not running
.venv/bin/forge diagnosis <job-id>               # for a failed job
```

Open http://127.0.0.1:8080 and sign in with the key.

To run the API and controller as separate processes on Postgres, Redis and NATS instead:

```bash
docker compose up --build -d --wait
docker compose exec api forge admin bootstrap --quota 16     # prints an admin key
```

## How it fits together

```
CLI / SDK ──HTTP──> API (Flask) ──> Postgres     source of truth
                     │   │
                     │   └──> Redis              recent logs and metrics per job
                     │
                     └──> NATS <── controller ──> Kubernetes, or local subprocesses
                           │
                           └──> dashboard (SSE), triage
training process ──> API     metrics, heartbeats, checkpoints (job-scoped token)
```

| Path | What it is |
|---|---|
| `src/forge/scheduler/core.py` | The scheduling policy. Pure: no I/O, no clock. |
| `src/forge/scheduler/sim.py` | Discrete-event simulator that drives the core from a trace. |
| `src/forge/controller/loop.py` | Reconcile loop that drives the same core against real jobs. |
| `src/forge/controller/backend.py`, `kube.py` | Where attempts run: subprocesses, or pods bound with `spec.nodeName`. |
| `src/forge/api/app.py` | REST API, signed checkpoint URLs, SSE feed. |
| `src/forge/sdk.py` | What a training script imports: metrics, checkpoint, resume. |
| `src/forge/triage/` | Failure triage: rule-based diagnosis and an eval harness. |
| `web/` | React and TypeScript dashboard. |
| `deploy/` | Helm chart and alert rules. |

One design rule carries most of the weight: **the scheduler core does no I/O**.
`schedule(cluster, now, policy)` takes a snapshot and returns placements and
preemptions. The simulator and the controller are two drivers of that one
function, so the policy that is benchmarked is the policy that runs.

## Scheduler

- **Placement** is all-or-nothing for multi-worker jobs, best-fit per worker
  (the fullest node that still fits), to keep large blocks free.
- **Order** is priority, then whether the team is within its guaranteed quota,
  then submit time. Waiting jobs gain a priority class every four hours so
  nothing starves.
- **Borrowing.** A team may run beyond its guarantee on idle GPUs. Within a
  team the most recently started attempts are the borrowed ones.
- **Preemption.** A job may evict borrowed attempts if it is within its own
  guarantee or outranks them, and lower-priority attempts of its own team. An
  attempt running within its team's guarantee is never evicted by another team.
  Victims are chosen borrowed first, lowest priority next, least work since the
  last checkpoint last, and the set is pruned to the minimum that frees enough room.
- **Guards.** An attempt is not preempted until it has run for `min_runtime`,
  and a job preempted twice in an hour is pinned.
- **Two-phase.** Victims get SIGTERM and a grace period to checkpoint. Until
  they exit, the scheduler holds their GPUs for the job that asked, and does
  not evict anyone else for it.
- **Backfill** (optional, EASY): the first blocked job gets a reservation from
  running jobs' time limits; later jobs start only if they do not delay it.
- **Why am I not running?** Every queued job carries a reason:
  `quota_exceeded`, `insufficient_gpus`, `gang_unsatisfied`,
  `waiting_for_preemption`, `reserved_for_job` or `unschedulable`.

### Benchmark

```bash
.venv/bin/forge bench --nodes 64 --jobs 5000 --load 0.95
```

5,000 jobs on 512 GPUs (64 nodes of 8), four teams, offered load 95% of
capacity, checkpoints every 10 minutes, 60 s restart cost. Wait is time from
submission to first start.

| policy | utilization | wait p50 | wait p95 | high-priority p95 | low-priority p95 | preemptions | GPU-hours lost |
|---|---|---|---|---|---|---|---|
| FIFO | 86.0% | 3.2h | 6.0h | 6.0h | 6.0h | 0 | 0 |
| FIFO + backfill | 89.6% | 2m | 3.1h | 3.2h | 3.0h | 0 | 0 |
| priority, no quotas | 95.8% | 0s | 4.3h | 3m | 6.2h | 2287 | 2026 |
| Forge | 93.4% | 0s | 4.1h | 3m | 7.2h | 1127 | 1317 |
| Forge + backfill | 93.7% | 0s | 4.1h | 5m | 8.0h | 1097 | 1319 |

What this shows, and what it does not:

- High-priority p95 wait drops from 6 hours to 3 minutes, and utilization
  rises by 7 points over FIFO.
- That is paid for by low-priority jobs, whose p95 wait gets longer, and by
  about 1,300 GPU-hours of repeated work, roughly 5% of the 25,600 submitted.
- Quotas halve the preemptions of plain priority scheduling for the same
  high-priority wait.
- Backfill adds little on top of the Forge policy on this workload, and FIFO
  with backfill has the best overall p95. Which policy is best depends on
  whether the tail of low-priority work matters.
- **The workload is synthetic**, not a replay of a published cluster trace: one
  GPU for most jobs, heavy-tailed durations, a daily arrival cycle
  (`scheduler/trace.py`). The simulator gives backfill exact job durations,
  which real users do not supply. Results move with the seed; `--seed 1` gives
  Forge an overall p95 of 42 minutes against 2.8 hours for FIFO with backfill.
  `--trace file.csv` replays a real trace once converted to the documented format.

An early run of this benchmark exposed a thrashing bug: with a 5 minute
`min_runtime` and 10 minute checkpoints, victims were evicted before saving
anything, re-queued at the front, and evicted again, losing a quarter of all
submitted work. The default `min_runtime` is now 30 minutes for that reason.

## Preemption, end to end

`tests/test_e2e_local.py` runs this with real HTTP, real subprocesses and real
signals: a low-priority job is running; a high-priority job arrives; the
scheduler sends SIGTERM; the SDK saves a checkpoint and exits; the job returns
to the queue; the high-priority job runs; the first job restarts from its
checkpoint. The test asserts that every training step was logged exactly once.

In a training script:

```python
import forge

run = forge.init()
start, state = run.restore()               # (0, None) on a fresh start
for step in range(start, total):
    ...
    run.log(step, loss=loss)
    if run.should_checkpoint():            # periodically, or when asked to stop
        run.checkpoint(step + 1, serialize(model))   # exits here if preempted
```

## Failure triage

When a job fails, the controller gathers its exit status, log tail, metrics
and cluster events and stores a diagnosis: a category, a summary, the lines it
rests on, and sometimes a proposed change to the job (halve `--batch-size`
after an out-of-memory error, cut `--lr` after divergence).

The diagnoser (`triage/evidence.py`) matches patterns over that evidence. It
cannot act: a proposed change is applied only when a person approves it
(`forge diagnosis <id> --approve`, or the button in the dashboard), which
submits a new job.

```bash
.venv/bin/python -m forge.triage.eval
```

On the 21 labelled cases in `triage/cases/cases.json` it scores 17 (81%). The
four it misses need reading rather than matching: a silent SIGKILL with rising
memory in the log, a missing checkpoint surfacing as a storage-client error, a
mistyped flag, and a network fault inside a traceback. The harness scores any
function with the same signature, so a stronger diagnoser can be measured
against the same cases.

## Security

- API keys are random 128-bit values; only an HMAC under the server secret is stored.
- Training processes never receive a user's key. Each gets a token scoped to
  its own job, valid only for reporting metrics, heartbeats and checkpoints.
- Checkpoints move through short-lived signed URLs bound to method and path.
- Members see only their team's jobs; another team's job id returns 404, not 403.
- On Kubernetes: one namespace per team under the `restricted` pod security
  level, pods run as non-root with no service account token and no added
  capabilities, a NetworkPolicy allows training pods to reach only DNS and the
  API, and the controller's pod permissions are granted per team namespace.
- Submissions are validated and rate limited; `FORGE_*` environment names are reserved.

## Monitoring

The API and controller export Prometheus metrics (`/metrics`, and `:9100` for
the controller): queue wait histogram by team and priority, GPUs allocated and
borrowed by team, queue depth, scheduling cycle time, preemptions, state
transitions. `deploy/alerts.yaml` has rules for stalled runs, GPUs held by
jobs that have stopped logging, diverged runs, slow high-priority starts and
a missing controller.

## Status

| Piece | State |
|---|---|
| Scheduler core, simulator, benchmark | Tested (unit tests, plus randomised runs checking that no node is over-allocated and every job finishes) |
| API, auth, tenancy, signed URLs | Tested against a real HTTP server on SQLite |
| Controller with the local backend | Tested end to end, including preempt, checkpoint and resume |
| SDK and CLI | Exercised by the end-to-end tests and the demo |
| Dashboard | Builds under strict TypeScript; checked by hand against the demo |
| Triage rules and eval harness | Tested |
| Postgres, Redis and NATS adapters | Run by hand in the Compose stack (jobs, live logs over NATS, metrics in Redis, a preemption); no automated test yet |
| Dockerfile and `compose.yaml` | Image builds and the stack runs the demo with the API and controller as separate processes |
| Kubernetes backend and Helm chart | Written; **not yet run against a cluster** |
| CI workflow | Written; **not yet run** |

Known gaps: checkpoints are stored on the API's disk rather than in an object
store; the controller is a single instance with no leader election; the rate
limiter and the SSE fan-out are per API process; there is no schema migration
tool yet.

## Development

```bash
.venv/bin/pytest
.venv/bin/ruff format --check src tests examples && .venv/bin/ruff check src tests examples
.venv/bin/mypy
(cd web && npm run build)
```
