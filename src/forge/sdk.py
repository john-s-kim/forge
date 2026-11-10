"""Client library used inside a training script.

    import forge

    run = forge.init()
    start, state = run.restore()            # (0, None) on a fresh start
    for step in range(start, total):
        ...
        run.log(step, loss=loss)
        if run.should_checkpoint():
            run.checkpoint(step + 1, serialize(model))   # exits here if preempted

Outside Forge (no FORGE_JOB_ID in the environment) every call is a no-op that
prints, so the same script runs on a laptop.
"""

from __future__ import annotations

import atexit
import math
import os
import signal
import sys
import threading
import time
from typing import Any

import requests

PREEMPTED_EXIT_CODE = 143  # 128 + SIGTERM


class Run:
    def __init__(
        self,
        api_url: str | None = None,
        job_id: str | None = None,
        token: str | None = None,
        checkpoint_seconds: float = 600.0,
        flush_seconds: float = 1.0,
        heartbeat_seconds: float = 10.0,
    ) -> None:
        self.api_url = (api_url or os.environ.get("FORGE_API_URL", "")).rstrip("/")
        self.job_id = job_id or os.environ.get("FORGE_JOB_ID")
        self.attempt = int(os.environ.get("FORGE_ATTEMPT", "1"))
        self.worker = int(os.environ.get("FORGE_WORKER_INDEX", "0"))
        self.online = bool(self.api_url and self.job_id)
        self.checkpoint_seconds = checkpoint_seconds
        self._flush_seconds = flush_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._last_checkpoint = time.monotonic()
        self._preempted = threading.Event()
        self._closed = threading.Event()
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._http = requests.Session()
        token = token or os.environ.get("FORGE_JOB_TOKEN")
        if token:
            self._http.headers["Authorization"] = f"Bearer {token}"

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda *_: self._preempted.set())
        if self.online:
            threading.Thread(target=self._background, daemon=True, name="forge-sdk").start()
            atexit.register(self.close)

    # ---- metrics -----------------------------------------------------------

    def log(self, step: int, **metrics: float) -> None:
        """Record metric values for a step. Buffered; never blocks training on the network."""
        if not self.online:
            print(f"step {step}: " + ", ".join(f"{k}={v:.5g}" for k, v in metrics.items()))
            return
        if self.worker != 0:
            return  # one worker reports for the gang
        with self._lock:
            for name, value in metrics.items():
                v = float(value)
                # JSON has no NaN; the server records null as a diverged point.
                self._buffer.append(
                    {"name": name, "step": int(step), "value": v if math.isfinite(v) else None}
                )

    def flush(self) -> None:
        with self._lock:
            points, self._buffer = self._buffer, []
        if not points:
            return
        try:
            self._post("metrics", {"points": points})
        except requests.RequestException:
            with self._lock:  # keep them for the next attempt, bounded
                self._buffer = (points + self._buffer)[-10000:]

    # ---- checkpoints -------------------------------------------------------

    @property
    def preempted(self) -> bool:
        """True once the scheduler has asked this attempt to stop."""
        return self._preempted.is_set()

    def should_checkpoint(self) -> bool:
        return self.preempted or time.monotonic() - self._last_checkpoint >= self.checkpoint_seconds

    def checkpoint(self, step: int, data: bytes) -> None:
        """Save state from which training can resume at `step`. If the
        scheduler has asked this attempt to stop, exits the process afterwards."""
        if self.online and self.worker == 0:
            r = self._post("checkpoints", {"step": int(step)})
            up = self._http.put(self.api_url + r["upload_url"], data=data, timeout=300)
            up.raise_for_status()
        self._last_checkpoint = time.monotonic()
        if self.preempted:
            print(f"forge: preempted, checkpoint saved at step {step}", flush=True)
            self.close()
            sys.exit(PREEMPTED_EXIT_CODE)

    def restore(self) -> tuple[int, bytes | None]:
        """The latest checkpoint as (step, data), or (0, None) if there is none."""
        if not self.online:
            return 0, None
        r = self._http.get(f"{self.api_url}/v1/jobs/{self.job_id}/checkpoints/latest", timeout=30)
        if r.status_code == 404:
            return 0, None
        r.raise_for_status()
        meta = r.json()
        blob = self._http.get(self.api_url + meta["download_url"], timeout=300)
        blob.raise_for_status()
        return int(meta["step"]), blob.content

    # ---- plumbing ----------------------------------------------------------

    def _post(self, what: str, body: dict[str, Any]) -> dict[str, Any]:
        r = self._http.post(f"{self.api_url}/v1/jobs/{self.job_id}/{what}", json=body, timeout=30)
        r.raise_for_status()
        return r.json()

    def _background(self) -> None:
        last_beat = 0.0
        while not self._closed.wait(self._flush_seconds):
            self.flush()
            if time.monotonic() - last_beat >= self._heartbeat_seconds:
                last_beat = time.monotonic()
                try:
                    # Also learns about preemption if the signal did not reach us.
                    if self._post("heartbeat", {}).get("preempting"):
                        self._preempted.set()
                except requests.RequestException:
                    pass

    def close(self) -> None:
        if not self._closed.is_set():
            self._closed.set()
            if self.online:
                self.flush()


def init(**kwargs: Any) -> Run:
    return Run(**kwargs)
