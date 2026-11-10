"""Discrete-event simulator that drives the scheduler core from a trace.

Time is virtual. Preemption is modelled with checkpoints: a preempted job
keeps the work up to its last checkpoint, loses the rest, and pays a restart
cost when it resumes.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from forge.scheduler.core import Attempt, Cluster, Job, Policy, Priority, schedule


@dataclass(frozen=True)
class TraceJob:
    id: str
    team: str
    priority: Priority
    workers: int
    gpus_per_worker: int
    submit: float
    duration: float

    @property
    def gpus(self) -> int:
        return self.workers * self.gpus_per_worker


@dataclass(frozen=True)
class SimConfig:
    checkpoint_interval: float = 600.0
    restart_cost: float = 60.0
    check_invariants: bool = False


@dataclass
class JobOutcome:
    job: TraceJob
    first_start: float | None = None
    finish: float | None = None
    preemptions: int = 0
    lost_gpu_seconds: float = 0.0

    @property
    def wait(self) -> float:
        assert self.first_start is not None
        return self.first_start - self.job.submit


@dataclass
class SimResult:
    outcomes: dict[str, JobOutcome]
    utilization: float  # mean fraction of GPUs busy over the arrival window
    gpu_seconds_by_team: dict[str, float] = field(default_factory=dict)
    cycles: int = 0

    def waits(self, priority: Priority | None = None) -> list[float]:
        return sorted(
            o.wait
            for o in self.outcomes.values()
            if o.first_start is not None and (priority is None or o.job.priority == priority)
        )

    @property
    def preemptions(self) -> int:
        return sum(o.preemptions for o in self.outcomes.values())

    @property
    def lost_gpu_hours(self) -> float:
        return sum(o.lost_gpu_seconds for o in self.outcomes.values()) / 3600.0


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    i = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return sorted_values[i]


_ARRIVE, _FINISH = 0, 1


def simulate(
    nodes: dict[str, int],
    quotas: dict[str, int],
    trace: list[TraceJob],
    policy: Policy,
    config: SimConfig | None = None,
) -> SimResult:
    cfg = config or SimConfig()
    cluster = Cluster(nodes=dict(nodes), quotas=dict(quotas))
    outcomes = {tj.id: JobOutcome(tj) for tj in trace}
    remaining = {tj.id: tj.duration for tj in trace}
    preempted_at: dict[str, tuple[float, ...]] = {}
    attempt_no: dict[str, int] = {}
    overhead: dict[str, float] = {}  # restart cost paid by the current attempt
    by_id = {tj.id: tj for tj in trace}

    events: list[tuple[float, int, int, str, int]] = []
    seq = 0
    for tj in trace:
        events.append((tj.submit, seq, _ARRIVE, tj.id, 0))
        seq += 1
    heapq.heapify(events)

    window_start = min((tj.submit for tj in trace), default=0.0)
    window_end = max((tj.submit for tj in trace), default=0.0)
    capacity = sum(nodes.values())
    busy = 0
    busy_area = 0.0
    last_t = window_start
    team_area: dict[str, float] = {}
    team_busy: dict[str, int] = {}
    result = SimResult(outcomes=outcomes, utilization=0.0)

    def advance(t: float) -> None:
        nonlocal busy_area, last_t
        lo, hi = max(last_t, window_start), min(t, window_end)
        if hi > lo:
            busy_area += busy * (hi - lo)
        for team, n in team_busy.items():
            team_area[team] = team_area.get(team, 0.0) + n * (t - last_t)
        last_t = t

    def core_job(tj: TraceJob) -> Job:
        cost = cfg.restart_cost if attempt_no.get(tj.id, 0) else 0.0
        return Job(
            id=tj.id,
            team=tj.team,
            priority=tj.priority,
            workers=tj.workers,
            gpus_per_worker=tj.gpus_per_worker,
            submitted_at=tj.submit,
            time_limit=remaining[tj.id] + cost,
            preempted_at=preempted_at.get(tj.id, ()),
        )

    def start(job_id: str, placed: tuple[str, ...], t: float) -> None:
        nonlocal seq, busy
        job = next(j for j in cluster.queue if j.id == job_id)
        cluster.queue.remove(job)
        n = attempt_no.get(job_id, 0)
        overhead[job_id] = cfg.restart_cost if n else 0.0
        attempt_no[job_id] = n + 1
        cluster.running[job_id] = Attempt(
            job=job, nodes=placed, started_at=t, checkpoint_interval=cfg.checkpoint_interval
        )
        out = outcomes[job_id]
        if out.first_start is None:
            out.first_start = t
        busy += job.gpus
        team_busy[job.team] = team_busy.get(job.team, 0) + job.gpus
        heapq.heappush(
            events, (t + remaining[job_id] + overhead[job_id], seq, _FINISH, job_id, n + 1)
        )
        seq += 1

    def stop(job_id: str) -> Attempt:
        nonlocal busy
        a = cluster.running.pop(job_id)
        busy -= a.job.gpus
        team_busy[a.job.team] -= a.job.gpus
        return a

    def preempt(job_id: str, t: float) -> None:
        a = stop(job_id)
        ran = t - a.started_at
        productive = max(0.0, ran - overhead[job_id])
        kept = math.floor(productive / cfg.checkpoint_interval) * cfg.checkpoint_interval
        remaining[job_id] -= kept
        out = outcomes[job_id]
        out.preemptions += 1
        out.lost_gpu_seconds += (ran - kept) * a.job.gpus
        preempted_at[job_id] = (*preempted_at.get(job_id, ()), t)
        cluster.queue.append(core_job(by_id[job_id]))

    def check() -> None:
        used: dict[str, int] = dict.fromkeys(cluster.nodes, 0)
        for a in cluster.running.values():
            assert len(a.nodes) == a.job.workers, "gang must be placed whole"
            for node in a.nodes:
                used[node] += a.job.gpus_per_worker
        for node, n in used.items():
            assert n <= cluster.nodes[node], f"node {node} over-allocated"
        assert not {j.id for j in cluster.queue} & cluster.running.keys()

    while events:
        t = events[0][0]
        advance(t)
        while events and events[0][0] == t:
            _, _, kind, job_id, n = heapq.heappop(events)
            if kind == _ARRIVE:
                cluster.queue.append(core_job(by_id[job_id]))
            elif job_id in cluster.running and attempt_no[job_id] == n:
                stop(job_id)
                outcomes[job_id].finish = t

        # Preemption is instantaneous here, so re-plan until the plan settles.
        for _ in range(16):
            decision = schedule(cluster, t, policy)
            result.cycles += 1
            for job_id in decision.preemptions:
                preempt(job_id, t)
            for pl in decision.placements:
                start(pl.job_id, pl.nodes, t)
            if cfg.check_invariants:
                check()
            if not decision.preemptions:
                break

    span = window_end - window_start
    result.utilization = busy_area / (capacity * span) if capacity and span > 0 else 0.0
    result.gpu_seconds_by_team = team_area
    return result
