"""Fill a fresh `forge dev` with teams, users and a mix of jobs, so there is
something to look at: borrowing, a queue, a preemption and two failures.

    FORGE_API_KEY=<admin key> python examples/seed_demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests

URL = os.environ.get("FORGE_URL", "http://127.0.0.1:8080")
TRAIN = str(Path(__file__).parent / "train.py")


def call(key: str, method: str, path: str, **body: object) -> dict:
    r = requests.request(
        method,
        URL + path,
        json=body or None,
        headers={"Authorization": f"Bearer {key}"},
        timeout=10,
    )
    if r.status_code == 409:
        return {}
    r.raise_for_status()
    return r.json()


def main() -> None:
    admin = os.environ["FORGE_API_KEY"]
    keys = {}
    for team, quota, user in [("planning", 8, "priya"), ("robotics", 8, "ravi")]:
        call(admin, "POST", "/v1/teams", name=team, gpu_quota=quota)
        keys[team] = call(admin, "POST", "/v1/users", name=user, team=team).get("api_key")
    if not all(keys.values()):
        sys.exit("demo users already exist; start from an empty data directory")
    keys["vision"] = admin

    def submit(team: str, name: str, *args: str, **spec: object) -> str:
        command = [sys.executable, TRAIN, "--checkpoint-seconds", "5", *args]
        return call(keys[team], "POST", "/v1/jobs", name=name, command=command, **spec)["id"]

    # Vision fills its own quota and borrows the rest of the idle cluster.
    submit("vision", "resnet50-baseline", "--steps", "4000", "--sleep", "0.05", gpus_per_worker=8)
    submit("vision", "vit-sweep-lr3e-4", "--steps", "4000", "--sleep", "0.05", gpus_per_worker=8)
    submit("vision", "vit-sweep-lr1e-3", "--steps", "4000", "--sleep", "0.05", gpus_per_worker=8,
           priority="low")  # fmt: skip
    submit("robotics", "grasp-policy", "--steps", "3000", "--sleep", "0.05", gpus_per_worker=4)
    submit("vision", "oom-example", "--fail", "oom", "--batch-size", "512", "--sleep", "0.05")
    submit("vision", "diverged-example", "--fail", "nan", "--lr", "3.0", "--sleep", "0.05")
    time.sleep(8)
    # Planning arrives later and reclaims its guarantee: a borrowed vision job is preempted.
    submit("planning", "planner-distill", "--steps", "2000", "--sleep", "0.05", gpus_per_worker=8,
           priority="high")  # fmt: skip
    submit(
        "vision",
        "detr-finetune",
        "--steps",
        "2000",
        "--sleep",
        "0.05",
        workers=2,
        gpus_per_worker=8,
    )
    submit("robotics", "sim2real-eval", "--steps", "1000", "--sleep", "0.05", gpus_per_worker=2,
           priority="low")  # fmt: skip
    print("seeded")


if __name__ == "__main__":
    main()
