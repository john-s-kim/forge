from forge.scheduler import (
    Attempt,
    Cluster,
    Job,
    Policy,
    Priority,
    WaitReason,
    borrowed_ids,
    schedule,
)

NOW = 10_000.0


def job(id, team="a", priority=Priority.NORMAL, workers=1, gpus=1, at=0.0, **kw):
    return Job(id, team, priority, workers, gpus, at, **kw)


def running(j, nodes, started_at=0.0, **kw):
    return Attempt(job=j, nodes=tuple(nodes), started_at=started_at, **kw)


def cluster(nodes=None, quotas=None, run=(), queue=()):
    return Cluster(
        nodes=nodes or {"n1": 8, "n2": 8},
        quotas=quotas or {"a": 8, "b": 8},
        running={a.job.id: a for a in run},
        queue=list(queue),
    )


def test_places_job_that_fits():
    d = schedule(cluster(queue=[job("j1", gpus=4)]), NOW)
    assert [p.job_id for p in d.placements] == ["j1"]
    assert not d.placements[0].borrowed
    assert not d.preemptions


def test_best_fit_keeps_large_blocks_free():
    c = cluster(run=[running(job("r", gpus=6), ["n1"])], queue=[job("j", gpus=2)])
    d = schedule(c, NOW)
    assert d.placements[0].nodes == ("n1",)


def test_gang_is_all_or_nothing():
    c = cluster(run=[running(job("r", gpus=4), ["n1"])], queue=[job("j", workers=2, gpus=8)])
    d = schedule(c, NOW, Policy(preemption=False))
    assert not d.placements
    assert d.wait_reasons["j"] == WaitReason.QUOTA_EXCEEDED


def test_fragmentation_is_reported():
    c = cluster(
        quotas={"a": 16},
        run=[running(job("r1", gpus=4), ["n1"]), running(job("r2", gpus=4), ["n2"])],
        queue=[job("j", gpus=8)],
    )
    d = schedule(c, NOW, Policy(preemption=False))
    assert d.wait_reasons["j"] == WaitReason.GANG_UNSATISFIED


def test_job_larger_than_any_node_is_unschedulable():
    d = schedule(cluster(queue=[job("j", gpus=16)]), NOW)
    assert d.wait_reasons["j"] == WaitReason.UNSCHEDULABLE


def test_priority_orders_the_queue():
    c = cluster(
        nodes={"n1": 4},
        queue=[
            job("low", priority=Priority.LOW, gpus=4),
            job("high", priority=Priority.HIGH, gpus=4, at=5),
        ],
    )
    d = schedule(c, NOW, Policy(aging_seconds=None))
    assert [p.job_id for p in d.placements] == ["high"]


def test_aging_lifts_old_low_priority_jobs():
    c = cluster(
        nodes={"n1": 4},
        queue=[
            job("old-low", priority=Priority.LOW, gpus=4, at=0),
            job("new-high", priority=Priority.HIGH, gpus=4, at=NOW - 1),
        ],
    )
    d = schedule(c, NOW, Policy(aging_seconds=3600))
    assert [p.job_id for p in d.placements] == ["old-low"]


def test_borrowing_idle_capacity_is_allowed_and_marked():
    c = cluster(queue=[job("j1", gpus=8), job("j2", gpus=8, at=1)])
    d = schedule(c, NOW)
    assert {p.job_id: p.borrowed for p in d.placements} == {"j1": False, "j2": True}


def test_owner_reclaims_borrowed_capacity():
    r1, r2 = job("r1", team="a", gpus=8), job("r2", team="a", gpus=8)
    c = cluster(
        run=[running(r1, ["n1"], 0), running(r2, ["n2"], 10)],
        queue=[job("j", team="b", gpus=8)],
    )
    assert borrowed_ids(c) == {"r2"}
    d = schedule(c, NOW)
    assert d.preemptions == ["r2"]
    assert d.wait_reasons["j"] == WaitReason.WAITING_FOR_PREEMPTION
    assert not d.placements


def test_guarantee_is_not_broken_by_other_teams_priority():
    r = job("r", team="a", priority=Priority.LOW, gpus=8)
    c = cluster(
        nodes={"n1": 8},
        quotas={"a": 8, "b": 0},
        run=[running(r, ["n1"])],
        queue=[job("j", team="b", priority=Priority.HIGH, gpus=8)],
    )
    d = schedule(c, NOW)
    assert not d.preemptions
    assert d.wait_reasons["j"] == WaitReason.QUOTA_EXCEEDED


def test_decision_reports_queue_order():
    c = cluster(nodes={"n1": 1}, queue=[job("x", at=2), job("y", at=1)])
    assert schedule(c, NOW).order == ["y", "x"]


def test_higher_priority_preempts_within_team():
    r = job("r", priority=Priority.LOW, gpus=8)
    c = cluster(
        nodes={"n1": 8}, run=[running(r, ["n1"])], queue=[job("j", priority=Priority.HIGH, gpus=8)]
    )
    assert schedule(c, NOW).preemptions == ["r"]


def test_min_runtime_protects_young_attempts():
    r = job("r", priority=Priority.LOW, gpus=8)
    c = cluster(
        nodes={"n1": 8},
        quotas={"a": 16},
        run=[running(r, ["n1"], started_at=NOW - 10)],
        queue=[job("j", priority=Priority.HIGH, gpus=8)],
    )
    d = schedule(c, NOW, Policy(min_runtime=300))
    assert not d.preemptions
    assert d.wait_reasons["j"] == WaitReason.INSUFFICIENT_GPUS


def test_preemption_cap_pins_a_job():
    r = job("r", priority=Priority.LOW, gpus=8, preempted_at=(NOW - 100, NOW - 200, NOW - 300))
    c = cluster(
        nodes={"n1": 8}, run=[running(r, ["n1"])], queue=[job("j", priority=Priority.HIGH, gpus=8)]
    )
    assert not schedule(c, NOW).preemptions


def test_victim_with_least_work_at_risk_goes_first():
    a = running(job("a1", priority=Priority.LOW, gpus=8), ["n1"], last_checkpoint_at=NOW - 500)
    b = running(job("a2", priority=Priority.LOW, gpus=8), ["n2"], last_checkpoint_at=NOW - 5)
    c = cluster(quotas={"a": 16}, run=[a, b], queue=[job("j", priority=Priority.HIGH, gpus=8)])
    assert schedule(c, NOW).preemptions == ["a2"]


def test_victim_set_is_minimal():
    small = running(job("s", priority=Priority.LOW, gpus=2), ["n1"], last_checkpoint_at=NOW - 1)
    big = running(job("b", priority=Priority.LOW, gpus=8), ["n2"], last_checkpoint_at=NOW - 900)
    filler = running(job("f", priority=Priority.HIGH, gpus=6), ["n1"])
    c = cluster(
        quotas={"a": 16}, run=[small, big, filler], queue=[job("j", priority=Priority.HIGH, gpus=8)]
    )
    assert schedule(c, NOW).preemptions == ["b"]


def test_waits_for_victims_instead_of_preempting_more():
    leaving = running(job("r1", priority=Priority.LOW, gpus=8), ["n1"], preempting=True)
    other = running(job("r2", priority=Priority.LOW, gpus=8), ["n2"])
    c = cluster(
        quotas={"a": 16}, run=[leaving, other], queue=[job("j", priority=Priority.HIGH, gpus=8)]
    )
    d = schedule(c, NOW)
    assert not d.preemptions
    assert d.wait_reasons["j"] == WaitReason.WAITING_FOR_PREEMPTION


def test_capacity_freed_for_a_waiter_is_not_given_away():
    # n1 is being vacated for the high-priority job; the small job must not take
    # the GPUs that are already free on it.
    leaving = running(job("r1", priority=Priority.LOW, gpus=4), ["n1"], preempting=True)
    c = cluster(
        nodes={"n1": 8},
        run=[leaving],
        queue=[
            job("big", priority=Priority.HIGH, gpus=8),
            job("small", priority=Priority.LOW, gpus=4, at=1),
        ],
    )
    d = schedule(c, NOW)
    assert not d.placements
    assert d.wait_reasons["big"] == WaitReason.WAITING_FOR_PREEMPTION


def test_fifo_head_blocks_the_queue():
    c = cluster(
        nodes={"n1": 8},
        run=[running(job("r", gpus=4), ["n1"])],
        queue=[job("big", gpus=8), job("small", gpus=2, at=1)],
    )
    d = schedule(
        c, NOW, Policy(use_priority=False, use_quotas=False, preemption=False, backfill="none")
    )
    assert not d.placements
    assert d.wait_reasons["small"] == WaitReason.RESERVED_FOR_JOB


def test_easy_backfill_admits_only_jobs_that_do_not_delay_the_head():
    easy = Policy(use_priority=False, use_quotas=False, preemption=False, backfill="easy")
    r = running(job("r", gpus=4, time_limit=1000), ["n1"], started_at=NOW - 500)  # ends at NOW+500
    c = cluster(
        nodes={"n1": 8},
        run=[r],
        queue=[
            job("big", gpus=8, time_limit=100),
            job("short", gpus=2, at=1, time_limit=400),
            job("long", gpus=2, at=2, time_limit=4000),
        ],
    )
    d = schedule(c, NOW, easy)
    assert [p.job_id for p in d.placements] == ["short"]
    assert d.wait_reasons["long"] == WaitReason.RESERVED_FOR_JOB
