"""Compare scheduling policies on one workload.

python -m forge.scheduler.bench --nodes 64 --jobs 5000 --load 0.95
"""

from __future__ import annotations

import argparse
import time

from forge.scheduler.core import Policy, Priority
from forge.scheduler.sim import SimConfig, SimResult, percentile, simulate
from forge.scheduler.trace import DEFAULT_TEAMS, load_csv, quotas_for, synthetic

POLICIES: dict[str, Policy] = {
    "fifo": Policy(use_priority=False, use_quotas=False, preemption=False, backfill="none"),
    "fifo + backfill": Policy(
        use_priority=False, use_quotas=False, preemption=False, backfill="easy"
    ),
    "priority": Policy(use_quotas=False, backfill="greedy"),
    "forge": Policy(backfill="greedy"),
    "forge + backfill": Policy(backfill="easy"),
}


def _fmt(seconds: float) -> str:
    if seconds != seconds:
        return "-"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def row(name: str, r: SimResult, wall: float) -> str:
    w = r.waits()
    hi = r.waits(Priority.HIGH)
    lo = r.waits(Priority.LOW)
    return (
        f"| {name} | {r.utilization:.1%} | {_fmt(percentile(w, 0.5))} | "
        f"{_fmt(percentile(w, 0.95))} | {_fmt(percentile(hi, 0.95))} | "
        f"{_fmt(percentile(lo, 0.95))} | {r.preemptions} | {r.lost_gpu_hours:.0f} | {wall:.1f}s |"
    )


HEADER = (
    "| policy | utilization | wait p50 | wait p95 | high-prio p95 | low-prio p95 "
    "| preemptions | GPU-hours lost | sim time |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nodes", type=int, default=64)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=5000)
    ap.add_argument("--load", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", help="CSV trace to replay instead of the synthetic workload")
    args = ap.parse_args()

    nodes = {f"node-{i:03d}": args.gpus_per_node for i in range(args.nodes)}
    capacity = sum(nodes.values())
    quotas = quotas_for(DEFAULT_TEAMS, capacity)
    trace = (
        load_csv(args.trace) if args.trace else synthetic(args.jobs, capacity, args.load, args.seed)
    )

    print(f"{len(trace)} jobs, {capacity} GPUs on {len(nodes)} nodes\n")
    print(HEADER)
    for name, policy in POLICIES.items():
        t0 = time.perf_counter()
        r = simulate(nodes, quotas, trace, policy, SimConfig())
        print(row(name, r, time.perf_counter() - t0), flush=True)


if __name__ == "__main__":
    main()
