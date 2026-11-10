import pytest

from forge.scheduler.bench import POLICIES
from forge.scheduler.core import Policy, Priority
from forge.scheduler.sim import SimConfig, TraceJob, percentile, simulate
from forge.scheduler.trace import DEFAULT_TEAMS, quotas_for, synthetic

NODES = {f"n{i}": 8 for i in range(8)}
CAPACITY = 64


@pytest.mark.parametrize("name", POLICIES)
@pytest.mark.parametrize("seed", range(4))
def test_invariants_hold_and_every_job_finishes(name, seed):
    """Randomised workloads: no node is ever over-allocated, gangs are placed
    whole, and nothing starves."""
    trace = synthetic(300, CAPACITY, load=1.1, seed=seed)
    r = simulate(
        NODES,
        quotas_for(DEFAULT_TEAMS, CAPACITY),
        trace,
        POLICIES[name],
        SimConfig(check_invariants=True),
    )
    assert all(o.finish is not None for o in r.outcomes.values())
    assert 0.0 < r.utilization <= 1.0
    for o in r.outcomes.values():
        assert o.finish - o.first_start >= o.job.duration - 1e-6


def test_simulation_is_deterministic():
    trace = synthetic(200, CAPACITY, seed=7)
    q = quotas_for(DEFAULT_TEAMS, CAPACITY)
    a = simulate(NODES, q, trace, POLICIES["forge"])
    b = simulate(NODES, q, trace, POLICIES["forge"])
    assert a.waits() == b.waits() and a.preemptions == b.preemptions


def test_preempted_job_keeps_checkpointed_work():
    trace = [
        TraceJob("low", "a", Priority.LOW, 1, 8, submit=0, duration=3000),
        TraceJob("high", "a", Priority.HIGH, 1, 8, submit=1500, duration=1000),
    ]
    cfg = SimConfig(checkpoint_interval=600, restart_cost=60)
    r = simulate({"n": 8}, {"a": 8}, trace, Policy(min_runtime=300), cfg)
    low, high = r.outcomes["low"], r.outcomes["high"]
    assert high.first_start == 1500 and high.finish == 2500
    assert low.preemptions == 1
    # Ran 1500s, checkpointed at 1200s: 300s on 8 GPUs lost.
    assert low.lost_gpu_seconds == 300 * 8
    # Resumes at 2500 with 1800s left plus the restart cost.
    assert low.finish == 2500 + 1800 + 60


def test_forge_policy_serves_high_priority_faster_than_fifo():
    trace = synthetic(1500, CAPACITY, load=1.0, seed=1)
    q = quotas_for(DEFAULT_TEAMS, CAPACITY)
    fifo = simulate(NODES, q, trace, POLICIES["fifo"])
    forge = simulate(NODES, q, trace, POLICIES["forge"])
    assert percentile(forge.waits(Priority.HIGH), 0.95) < percentile(
        fifo.waits(Priority.HIGH), 0.95
    )
    assert forge.utilization >= fifo.utilization
