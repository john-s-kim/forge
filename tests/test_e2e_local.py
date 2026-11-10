"""End to end on the local backend: real HTTP, real subprocesses, real signals."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from forge.controller.loop import Controller, make_backend
from forge.triage.diagnose import diagnose

TRAIN = str(Path(__file__).parent.parent / "examples" / "train.py")


@pytest.fixture
def controller(api):
    c = Controller(api.ctx, make_backend(api.ctx), on_failure=lambda j, a: diagnose(api.ctx, j, a))
    stop = threading.Event()
    t = threading.Thread(target=c.run_forever, args=(stop,), daemon=True)
    t.start()
    yield c
    stop.set()
    t.join(timeout=5)


def submit(api, *args, user="alice", **spec):
    r = api.call(
        "POST", "/v1/jobs", user=user, json={"command": [sys.executable, TRAIN, *args], **spec}
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def job(api, job_id):
    return api.call("GET", f"/v1/jobs/{job_id}").json()


def wait_for(predicate, timeout=30.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if value := predicate():
            return value
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def wait_state(api, job_id, *states, timeout=30.0):
    return wait_for(
        lambda: (j := job(api, job_id))["state"] in states and j, timeout, f"{job_id} in {states}"
    )


def test_job_runs_and_reports_metrics(api, controller):
    jid = submit(api, "--steps", "60", "--sleep", "0.01", gpus_per_worker=2)
    done = wait_state(api, jid, "SUCCEEDED", "FAILED")
    assert done["state"] == "SUCCEEDED"
    assert [e["to"] for e in done["events"]] == ["QUEUED", "STARTING", "RUNNING", "SUCCEEDED"]

    points = api.call("GET", f"/v1/jobs/{jid}/metrics", params={"full": 1}).json()["points"]
    losses = [p["value"] for p in points if p["name"] == "loss"]
    assert len(losses) == 60 and losses[-1] < losses[0]
    assert any("done:" in line for line in api.call("GET", f"/v1/jobs/{jid}/logs").json()["lines"])
    assert done["checkpoints"][-1]["step"] == 60


def test_preempted_job_checkpoints_and_resumes(api, controller):
    low = submit(
        api, "--steps", "300", "--sleep", "0.02", "--checkpoint-seconds", "3600",
        gpus_per_worker=4, priority="low",
    )  # fmt: skip
    wait_state(api, low, "RUNNING")
    wait_for(
        lambda: len(api.call("GET", f"/v1/jobs/{low}/metrics").json()["points"]) > 40,
        what="low-priority job to make progress",
    )

    high = submit(api, "--steps", "20", "--sleep", "0.01", gpus_per_worker=4, priority="high")
    assert wait_state(api, high, "SUCCEEDED")["attempts"] == 1
    done = wait_state(api, low, "SUCCEEDED", "FAILED", timeout=60)
    assert done["state"] == "SUCCEEDED"

    states = [e["to"] for e in done["events"]]
    assert states == [
        "QUEUED", "STARTING", "RUNNING", "PREEMPTING", "QUEUED", "STARTING", "RUNNING", "SUCCEEDED",
    ]  # fmt: skip
    first, second = done["attempt_history"]
    assert first["exit_reason"] == "preempted" and second["exit_code"] == 0

    # The only checkpoint before the final one was the one taken on SIGTERM,
    # and the second attempt picked up from it rather than from step 0.
    resumed_at = done["checkpoints"][0]["step"]
    assert 0 < resumed_at < 300
    lines = api.call("GET", f"/v1/jobs/{low}/logs", params={"tail": 2000}).json()["lines"]
    assert f"resumed from step {resumed_at}" in lines

    # No step was trained twice and none was skipped.
    points = api.call("GET", f"/v1/jobs/{low}/metrics", params={"full": 1}).json()["points"]
    steps = [p["step"] for p in points if p["name"] == "loss"]
    assert steps == list(range(300))


def test_failed_job_is_diagnosed_and_fix_can_be_approved(api, controller):
    jid = submit(api, "--fail", "oom", "--batch-size", "256", "--sleep", "0")
    assert wait_state(api, jid, "FAILED")["state"] == "FAILED"
    d = wait_for(
        lambda: (r := api.call("GET", f"/v1/jobs/{jid}/diagnosis")).status_code == 200 and r.json(),
        what="diagnosis",
    )
    assert d["category"] == "oom"
    assert any("out of memory" in line for line in d["evidence"])
    command = d["proposed_patch"]["command"]
    assert command[command.index("--batch-size") + 1] == "128"

    r = api.call("POST", f"/v1/jobs/{jid}/diagnosis/approve")
    assert r.status_code == 201
    retry = r.json()
    assert retry["retry_of"] == jid and "128" in retry["spec"]["command"]
    assert api.call("POST", f"/v1/jobs/{jid}/diagnosis/approve").status_code == 409


def test_cancel_stops_a_running_job(api, controller):
    jid = submit(api, "--steps", "100000", "--sleep", "0.05")
    wait_state(api, jid, "RUNNING")
    assert api.call("POST", f"/v1/jobs/{jid}/cancel").json()["state"] == "CANCELLED"
    wait_for(
        lambda: job(api, jid)["attempt_history"][0]["ended_at"] is not None, what="attempt to end"
    )
    assert api.call("GET", "/v1/cluster").json()["nodes"][0]["allocations"] == []


def test_gang_job_runs_all_workers(api, controller):
    jid = submit(api, "--steps", "10", "--sleep", "0", workers=2, gpus_per_worker=2)
    done = wait_state(api, jid, "SUCCEEDED", "FAILED")
    assert done["state"] == "SUCCEEDED" and done["attempt_history"][0]["nodes"] == ["n0", "n0"]
    lines = api.call("GET", f"/v1/jobs/{jid}/logs").json()["lines"]
    assert any(line.startswith("[w0]") for line in lines) and any(
        line.startswith("[w1]") for line in lines
    )
