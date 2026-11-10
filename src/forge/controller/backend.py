"""Where attempts actually run.

The controller only speaks this interface. `LocalBackend` runs each worker as
a subprocess on this machine against a made-up set of nodes, which is enough
to exercise the whole lifecycle (including SIGTERM, checkpoint and resume)
without a cluster. `KubeBackend` creates pods.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

LogSink = Callable[[str, str, str], None]  # job id, team, line


@dataclass(frozen=True)
class Launch:
    job_id: str
    team: str
    attempt_id: int
    number: int
    image: str | None
    command: list[str]
    env: dict[str, str]
    nodes: list[str]  # one per worker
    gpus_per_worker: int
    grace_seconds: float


@dataclass(frozen=True)
class Status:
    phase: str  # starting | running | succeeded | failed | gone
    exit_code: int | None = None
    reason: str | None = None
    events: list[str] = field(default_factory=list)


class Backend(Protocol):
    def nodes(self) -> dict[str, int]: ...
    def start(self, launch: Launch) -> None: ...
    def adopt(self, launch: Launch) -> None: ...  # re-attach after a controller restart
    def status(self, attempt_id: int) -> Status: ...
    def stop(self, attempt_id: int) -> None: ...  # ask politely (SIGTERM)
    def kill(self, attempt_id: int) -> None: ...
    def forget(self, attempt_id: int) -> None: ...


class LocalBackend:
    def __init__(self, nodes: dict[str, int], log_dir: Path, sink: LogSink) -> None:
        self._nodes = dict(nodes)
        self._log_dir = log_dir
        self._sink = sink
        self._procs: dict[int, list[subprocess.Popen[str]]] = {}
        self._errors: dict[int, str] = {}

    def nodes(self) -> dict[str, int]:
        return dict(self._nodes)

    def start(self, launch: Launch) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        procs: list[subprocess.Popen[str]] = []
        for i, node in enumerate(launch.nodes):
            env = {
                **os.environ,
                **launch.env,
                "FORGE_WORKER_INDEX": str(i),
                "FORGE_NODE": node,
                "PYTHONUNBUFFERED": "1",
            }
            try:
                p = subprocess.Popen(
                    launch.command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                )
            except OSError as e:
                for started in procs:
                    started.kill()
                self._procs[launch.attempt_id] = []
                self._errors[launch.attempt_id] = str(e)
                self._sink(launch.job_id, launch.team, f"forge: could not start worker {i}: {e}")
                return
            procs.append(p)
            prefix = f"[w{i}] " if len(launch.nodes) > 1 else ""
            threading.Thread(target=self._pump, args=(launch, p, prefix), daemon=True).start()
        self._procs[launch.attempt_id] = procs

    def adopt(self, launch: Launch) -> None:
        """Subprocesses do not survive a controller restart, so there is
        nothing to adopt: the attempt reports as gone and the job is requeued."""

    def _pump(self, launch: Launch, p: subprocess.Popen[str], prefix: str) -> None:
        log = self._log_dir / f"{launch.job_id}-{launch.number}.log"
        assert p.stdout is not None
        with open(log, "a") as f:
            for line in p.stdout:
                line = prefix + line.rstrip("\n")
                f.write(line + "\n")
                self._sink(launch.job_id, launch.team, line)

    def status(self, attempt_id: int) -> Status:
        procs = self._procs.get(attempt_id)
        if procs is None:
            return Status("gone", reason="attempt lost")
        if not procs:
            return Status(
                "failed",
                exit_code=127,
                reason="could not start",
                events=[self._errors.get(attempt_id, "")],
            )
        codes = [p.poll() for p in procs]
        failed = [c for c in codes if c not in (None, 0)]
        if failed:
            for p in procs:  # a gang fails together
                if p.poll() is None:
                    p.kill()
            code = failed[0]
            reason = f"killed by signal {-code}" if code < 0 else f"exit code {code}"
            return Status("failed", exit_code=code, reason=reason)
        if all(c == 0 for c in codes):
            return Status("succeeded", exit_code=0)
        return Status("running")

    def _signal(self, attempt_id: int, sig: int) -> None:
        for p in self._procs.get(attempt_id, []):
            if p.poll() is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(p.pid, sig)

    def stop(self, attempt_id: int) -> None:
        self._signal(attempt_id, signal.SIGTERM)

    def kill(self, attempt_id: int) -> None:
        self._signal(attempt_id, signal.SIGKILL)

    def forget(self, attempt_id: int) -> None:
        self._procs.pop(attempt_id, None)
        self._errors.pop(attempt_id, None)
