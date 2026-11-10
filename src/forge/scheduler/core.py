"""Scheduling policy.

This module is pure: no I/O, no clock, no Kubernetes. `schedule` takes a
snapshot of the cluster and returns what to start and what to preempt. The
Kubernetes controller and the trace-replay simulator are two drivers of this
same function, so the scheduler that is benchmarked is the one that is deployed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum


class Priority(IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


class WaitReason(StrEnum):
    QUOTA_EXCEEDED = "quota_exceeded"  # over guarantee and no idle capacity to borrow
    INSUFFICIENT_GPUS = "insufficient_gpus"  # within guarantee, cluster is full
    GANG_UNSATISFIED = "gang_unsatisfied"  # enough GPUs in total, but fragmented
    WAITING_FOR_PREEMPTION = "waiting_for_preemption"  # victims are shutting down
    RESERVED_FOR_JOB = "reserved_for_job"  # capacity is held for a job ahead in the queue
    UNSCHEDULABLE = "unschedulable"  # can never fit on this cluster


@dataclass(frozen=True)
class Job:
    id: str
    team: str
    priority: Priority
    workers: int
    gpus_per_worker: int
    submitted_at: float
    time_limit: float | None = None
    preempted_at: tuple[float, ...] = ()

    @property
    def gpus(self) -> int:
        return self.workers * self.gpus_per_worker


@dataclass(frozen=True)
class Attempt:
    """One placement of a job. `nodes` has one entry per worker."""

    job: Job
    nodes: tuple[str, ...]
    started_at: float
    last_checkpoint_at: float | None = None
    # If the driver does not track checkpoints individually, it can declare a
    # fixed cadence instead and the work at risk is derived from it.
    checkpoint_interval: float | None = None
    preempting: bool = False


@dataclass
class Cluster:
    nodes: dict[str, int]  # node name -> GPU capacity
    quotas: dict[str, int]  # team -> guaranteed GPUs
    running: dict[str, Attempt] = field(default_factory=dict)  # by job id
    queue: list[Job] = field(default_factory=list)


@dataclass(frozen=True)
class Policy:
    use_priority: bool = True
    use_quotas: bool = True
    preemption: bool = True
    # "none": a blocked job blocks everything behind it.
    # "greedy": anything that fits may start.
    # "easy": the first blocked job gets a reservation; others may start only
    #         if they do not delay it (EASY backfill).
    backfill: str = "greedy"
    aging_seconds: float | None = 4 * 3600.0  # wait this long to gain one priority class
    # An attempt younger than this is not preempted. Keep it well above the
    # checkpoint interval, or a victim can be evicted before it has saved anything.
    min_runtime: float = 1800.0
    max_preemptions_per_hour: int = 2  # then the job is pinned
    max_preemption_searches: int = 8  # bound on victim searches per cycle


@dataclass(frozen=True)
class Placement:
    job_id: str
    nodes: tuple[str, ...]
    borrowed: bool


@dataclass
class Decision:
    placements: list[Placement] = field(default_factory=list)
    preemptions: list[str] = field(default_factory=list)
    wait_reasons: dict[str, WaitReason] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # queue order used this cycle


class _Free:
    """Free GPUs per node, bucketed by count so best-fit is O(node size)."""

    __slots__ = ("free", "total", "_buckets", "_cap")

    def __init__(self, free: dict[str, int], cap: int) -> None:
        self.free = dict(free)
        self.total = sum(free.values())
        self._cap = cap
        self._buckets: list[dict[str, None]] = [{} for _ in range(cap + 1)]
        for node, n in free.items():
            self._buckets[n][node] = None

    def copy(self) -> _Free:
        new = object.__new__(_Free)
        new.free = dict(self.free)
        new.total = self.total
        new._cap = self._cap
        new._buckets = [dict(b) for b in self._buckets]
        return new

    def set(self, node: str, value: int) -> None:
        old = self.free.get(node)
        if old is None or old == value:
            return
        value = max(0, min(value, self._cap))
        del self._buckets[old][node]
        self._buckets[value][node] = None
        self.free[node] = value
        self.total += value - old

    def take(self, node: str, n: int) -> None:
        if node in self.free:
            self.set(node, self.free[node] - n)

    def give(self, node: str, n: int) -> None:
        if node in self.free:
            self.set(node, self.free[node] + n)

    def give_attempt(self, a: Attempt) -> None:
        for node in a.nodes:
            self.give(node, a.job.gpus_per_worker)

    def take_attempt(self, a: Attempt) -> None:
        for node in a.nodes:
            self.take(node, a.job.gpus_per_worker)

    def fit(self, job: Job, commit: bool) -> tuple[str, ...] | None:
        """All-or-nothing best-fit: each worker goes to the fullest node that
        can still hold it, which keeps large free blocks intact."""
        g = job.gpus_per_worker
        if job.gpus > self.total or g > self._cap:
            return None
        chosen: list[str] = []
        for _ in range(job.workers):
            node = None
            for n in range(g, self._cap + 1):
                bucket = self._buckets[n]
                if bucket:
                    node = next(iter(bucket))
                    break
            if node is None:
                break
            self.take(node, g)
            chosen.append(node)
        ok = len(chosen) == job.workers
        if not ok or not commit:
            for node in chosen:
                self.give(node, g)
        return tuple(chosen) if ok else None


def borrowed_ids(cluster: Cluster, use_quotas: bool = True) -> set[str]:
    """Jobs running on capacity beyond their team's guarantee.

    Within a team, attempts are charged to the guarantee oldest first, so the
    most recently started ones are the borrowed ones.
    """
    if not use_quotas:
        return set()
    out: set[str] = set()
    used: dict[str, int] = {}
    attempts = sorted(cluster.running.values(), key=lambda a: (a.started_at, a.job.id))
    for a in attempts:
        if a.preempting:
            continue
        team = a.job.team
        if used.get(team, 0) + a.job.gpus <= cluster.quotas.get(team, 0):
            used[team] = used.get(team, 0) + a.job.gpus
        else:
            out.add(a.job.id)
    return out


def _work_at_risk(a: Attempt, now: float) -> float:
    if a.last_checkpoint_at is not None:
        return now - a.last_checkpoint_at
    ran = now - a.started_at
    if a.checkpoint_interval:
        return ran % a.checkpoint_interval
    return ran


def _expected_end(a: Attempt, now: float) -> float | None:
    if a.preempting:
        return now
    if a.job.time_limit is None:
        return None
    return a.started_at + a.job.time_limit


def schedule(cluster: Cluster, now: float, policy: Policy | None = None) -> Decision:
    p = policy or Policy()
    d = Decision()
    cap = max(cluster.nodes.values(), default=0)
    capacity = sum(cluster.nodes.values())

    # `free` is what can be used right now. `soon` also counts GPUs held by
    # attempts that are already shutting down; a job that fits in `soon` waits
    # for them rather than causing more preemptions.
    used: dict[str, int] = dict.fromkeys(cluster.nodes, 0)
    used_soon: dict[str, int] = dict.fromkeys(cluster.nodes, 0)
    for a in cluster.running.values():
        for node in a.nodes:
            if node in used:
                used[node] += a.job.gpus_per_worker
                if not a.preempting:
                    used_soon[node] += a.job.gpus_per_worker
    free = _Free({n: max(0, c - used[n]) for n, c in cluster.nodes.items()}, cap)
    soon = _Free({n: max(0, c - used_soon[n]) for n, c in cluster.nodes.items()}, cap)

    borrowed = borrowed_ids(cluster, p.use_quotas)
    guaranteed_use: dict[str, int] = {}
    for a in cluster.running.values():
        if not a.preempting and a.job.id not in borrowed:
            guaranteed_use[a.job.team] = guaranteed_use.get(a.job.team, 0) + a.job.gpus

    def within_quota(job: Job) -> bool:
        if not p.use_quotas:
            return True
        return guaranteed_use.get(job.team, 0) + job.gpus <= cluster.quotas.get(job.team, 0)

    def order_key(job: Job) -> tuple[int, int, float, str]:
        prio = int(job.priority) if p.use_priority else 0
        if p.use_priority and p.aging_seconds:
            prio = min(int(Priority.HIGH), prio + int((now - job.submitted_at) // p.aging_seconds))
        return (-prio, 0 if within_quota(job) else 1, job.submitted_at, job.id)

    def reserve(job: Job, nodes: tuple[str, ...]) -> None:
        """Hold capacity in `free` for a job that was fitted into `soon`."""
        for node in nodes:
            free.take(node, min(free.free.get(node, 0), job.gpus_per_worker))
            if both is not None:
                both.set(node, min(free.free[node], extra.free[node]))  # type: ignore[union-attr]

    victims_taken: set[str] = set()
    started: list[tuple[float | None, Job, tuple[str, ...]]] = []
    searches = 0
    head_blocked = False
    # EASY backfill state: `extra` is what is left at the reservation's start
    # time once the reserved job is placed; `both` is min(free, extra).
    shadow: float | None = None
    extra: _Free | None = None
    both: _Free | None = None
    reservation_tried = False

    for job in sorted(cluster.queue, key=order_key):
        d.order.append(job.id)
        if job.gpus_per_worker > cap or job.gpus > capacity:
            d.wait_reasons[job.id] = WaitReason.UNSCHEDULABLE
            continue
        if head_blocked:
            d.wait_reasons[job.id] = WaitReason.RESERVED_FOR_JOB
            continue

        within = within_quota(job)

        # 1. Start now if it fits (and, under a reservation, does not delay it).
        nodes: tuple[str, ...] | None
        held_back = False
        if both is None:
            nodes = free.fit(job, commit=True)
        else:
            ends_in_time = job.time_limit is not None and now + job.time_limit <= shadow  # type: ignore[operator]
            if ends_in_time:
                nodes = free.fit(job, commit=True)
                if nodes:
                    for node in nodes:
                        both.set(node, min(free.free[node], extra.free[node]))  # type: ignore[union-attr]
            else:
                nodes = both.fit(job, commit=True)
                if nodes:
                    for node in nodes:
                        free.take(node, job.gpus_per_worker)
                        extra.take(node, job.gpus_per_worker)  # type: ignore[union-attr]
                        both.set(node, min(free.free[node], extra.free[node]))  # type: ignore[union-attr]
                else:
                    held_back = free.fit(job, commit=False) is not None
        if nodes:
            for node in nodes:
                soon.take(node, job.gpus_per_worker)
            if within:
                guaranteed_use[job.team] = guaranteed_use.get(job.team, 0) + job.gpus
            d.placements.append(Placement(job.id, nodes, borrowed=not within))
            started.append((now + job.time_limit if job.time_limit else None, job, nodes))
            continue
        if held_back:
            d.wait_reasons[job.id] = WaitReason.RESERVED_FOR_JOB
            continue

        # 2. Wait for attempts that are already being preempted.
        nodes = soon.fit(job, commit=True)
        if nodes:
            reserve(job, nodes)
            d.wait_reasons[job.id] = WaitReason.WAITING_FOR_PREEMPTION
            continue

        # 3. Preempt, if this job is entitled to someone else's capacity.
        if p.preemption and searches < p.max_preemption_searches:
            searches += 1
            victims = _pick_victims(job, within, cluster, borrowed, victims_taken, soon, now, p)
            if victims:
                for v in victims:
                    victims_taken.add(v.job.id)
                    d.preemptions.append(v.job.id)
                    soon.give_attempt(v)
                    if v.job.id not in borrowed:
                        guaranteed_use[v.job.team] -= v.job.gpus
                nodes = soon.fit(job, commit=True)
                assert nodes is not None
                reserve(job, nodes)
                d.wait_reasons[job.id] = WaitReason.WAITING_FOR_PREEMPTION
                continue

        # 4. Blocked.
        if not within:
            d.wait_reasons[job.id] = WaitReason.QUOTA_EXCEEDED
        elif job.gpus <= free.total:
            d.wait_reasons[job.id] = WaitReason.GANG_UNSATISFIED
        else:
            d.wait_reasons[job.id] = WaitReason.INSUFFICIENT_GPUS

        if p.backfill == "none":
            head_blocked = True
        elif p.backfill == "easy" and not reservation_tried:
            reservation_tried = True
            r = _reservation(job, cluster, free, started, now)
            if r is not None:
                shadow, extra = r
                both = _Free({n: min(free.free[n], extra.free[n]) for n in free.free}, cap)

    return d


def _pick_victims(
    job: Job,
    within: bool,
    cluster: Cluster,
    borrowed: set[str],
    taken: set[str],
    soon: _Free,
    now: float,
    p: Policy,
) -> list[Attempt]:
    """Smallest set of attempts whose removal lets `job` fit, or [] if none.

    A team's guarantee is honoured: an attempt running within its team's quota
    is only ever preempted by a higher-priority job of the same team.
    """
    candidates: list[Attempt] = []
    for a in cluster.running.values():
        if a.preempting or a.job.id in taken:
            continue
        if now - a.started_at < p.min_runtime:
            continue
        recent = sum(1 for t in a.job.preempted_at if now - t < 3600.0)
        if recent >= p.max_preemptions_per_hour:
            continue
        outranks = p.use_priority and job.priority > a.job.priority
        if a.job.id in borrowed:
            eligible = within or outranks
        else:
            eligible = outranks and (not p.use_quotas or a.job.team == job.team)
        if eligible:
            candidates.append(a)
    if not candidates:
        return []

    candidates.sort(
        key=lambda a: (a.job.id not in borrowed, int(a.job.priority), _work_at_risk(a, now))
    )
    trial = soon.copy()
    chosen: list[Attempt] = []
    for a in candidates:
        trial.give_attempt(a)
        chosen.append(a)
        if trial.fit(job, commit=False) is not None:
            break
    else:
        return []

    # Drop victims that turned out not to be needed.
    for a in list(chosen[:-1]):
        trial.take_attempt(a)
        if trial.fit(job, commit=False) is not None:
            chosen.remove(a)
        else:
            trial.give_attempt(a)
    return chosen


def _reservation(
    job: Job,
    cluster: Cluster,
    free: _Free,
    started: list[tuple[float | None, Job, tuple[str, ...]]],
    now: float,
) -> tuple[float, _Free] | None:
    """Earliest time `job` is guaranteed to fit, from declared time limits,
    and the capacity left over at that time once it is placed."""
    ends: list[tuple[float, str, int, tuple[str, ...]]] = []
    for a in cluster.running.values():
        end = _expected_end(a, now)
        if end is not None:
            ends.append((end, a.job.id, a.job.gpus_per_worker, a.nodes))
    for end, j, nodes in started:
        if end is not None:
            ends.append((end, j.id, j.gpus_per_worker, nodes))
    ends.sort()

    at = free.copy()
    for end, _, g, nodes in ends:
        for node in nodes:
            at.give(node, g)
        if at.fit(job, commit=True) is not None:
            return max(end, now), at
    return None
