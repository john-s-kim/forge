"""Workloads for the simulator: a synthetic generator and a CSV loader.

The synthetic workload follows the broad shape reported for shared GPU
clusters: most jobs ask for one GPU, a few ask for many, durations are heavy
tailed, and arrivals follow a daily cycle. It is not fitted to any specific
published trace. To replay a real one, convert it to the CSV format below.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from forge.scheduler.core import Priority
from forge.scheduler.sim import TraceJob

# (workers, gpus_per_worker, probability)
_SHAPES = [
    (1, 1, 0.58),
    (1, 2, 0.15),
    (1, 4, 0.12),
    (1, 8, 0.08),
    (2, 8, 0.04),
    (4, 8, 0.03),
]
_PRIORITIES = [(Priority.LOW, 0.30), (Priority.NORMAL, 0.55), (Priority.HIGH, 0.15)]


@dataclass(frozen=True)
class Team:
    name: str
    quota_share: float  # fraction of the cluster guaranteed to this team
    demand_share: float  # fraction of submitted jobs


DEFAULT_TEAMS = [
    Team("vision", 0.40, 0.50),
    Team("planning", 0.30, 0.20),
    Team("robotics", 0.20, 0.25),
    Team("infra", 0.10, 0.05),
]


def quotas_for(teams: list[Team], capacity: int) -> dict[str, int]:
    return {t.name: int(t.quota_share * capacity) for t in teams}


def synthetic(
    n_jobs: int,
    capacity: int,
    load: float = 0.95,
    seed: int = 0,
    teams: list[Team] | None = None,
) -> list[TraceJob]:
    """`load` is offered GPU-seconds divided by cluster GPU-seconds."""
    rng = random.Random(seed)
    teams = teams or DEFAULT_TEAMS
    shapes = rng.choices(_SHAPES, weights=[s[2] for s in _SHAPES], k=n_jobs)
    durations = [
        min(3 * 86400.0, max(60.0, rng.lognormvariate(math.log(1800), 1.5))) for _ in shapes
    ]
    demand = sum(w * g * d for (w, g, _), d in zip(shapes, durations, strict=True))
    span = demand / (capacity * load)
    rate = n_jobs / span

    jobs: list[TraceJob] = []
    t = 0.0
    for i, ((workers, gpus, _), duration) in enumerate(zip(shapes, durations, strict=True)):
        # Thinned Poisson process with a daily cycle (peak is 3x the trough).
        while True:
            t += rng.expovariate(rate * 1.5)
            if rng.random() < (1 + 0.5 * math.sin(2 * math.pi * t / 86400.0)) / 1.5:
                break
        team = rng.choices(teams, weights=[x.demand_share for x in teams])[0]
        priority = rng.choices([p for p, _ in _PRIORITIES], weights=[w for _, w in _PRIORITIES])[0]
        jobs.append(TraceJob(f"j{i:06d}", team.name, priority, workers, gpus, t, duration))
    return jobs


def load_csv(path: str | Path) -> list[TraceJob]:
    """Columns: id, team, priority (low|normal|high), workers, gpus_per_worker,
    submit (seconds), duration (seconds)."""
    jobs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            jobs.append(
                TraceJob(
                    id=row["id"],
                    team=row["team"],
                    priority=Priority[row["priority"].upper()],
                    workers=int(row["workers"]),
                    gpus_per_worker=int(row["gpus_per_worker"]),
                    submit=float(row["submit"]),
                    duration=float(row["duration"]),
                )
            )
    jobs.sort(key=lambda j: j.submit)
    return jobs
