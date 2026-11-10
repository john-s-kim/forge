"""Event bus and live store.

The bus carries job state transitions and metric points to the dashboard, the
alerting path and failure triage. The live store holds the recent tail of
logs and metrics per job so a page load does not hit the database.

Each has an in-process implementation, used by `forge dev` and the tests, and
a networked one (NATS, Redis) for a deployment where the API and controller
are separate processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import threading
from collections import defaultdict, deque
from collections.abc import Iterator
from typing import Any, Protocol

SUBJECT = "forge.events"
LOG_TAIL = 2000
METRIC_TAIL = 2000


class Bus(Protocol):
    def publish(self, event: dict[str, Any]) -> None: ...
    def subscribe(self) -> Subscription: ...


class Subscription:
    def __init__(self, on_close: Any = None, maxsize: int = 1000) -> None:
        self._q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=maxsize)
        self._on_close = on_close

    def put(self, event: dict[str, Any]) -> None:
        # A slow consumer loses its oldest events rather than blocking publishers.
        while True:
            try:
                self._q.put_nowait(event)
                return
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    self._q.get_nowait()

    def get(self, timeout: float | None = None) -> dict[str, Any] | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self._on_close:
            self._on_close(self)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            yield self._q.get()


class MemoryBus:
    def __init__(self) -> None:
        self._subs: list[Subscription] = []
        self._lock = threading.Lock()

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            s.put(event)

    def subscribe(self) -> Subscription:
        sub = Subscription(on_close=self._remove)
        with self._lock:
            self._subs.append(sub)
        return sub

    def _remove(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._subs:
                self._subs.remove(sub)


class NatsBus:
    """NATS core pub/sub. nats-py is asyncio-only, so it runs on its own loop
    in a background thread and this class bridges to synchronous callers."""

    def __init__(self, url: str) -> None:
        import nats  # type: ignore[import-not-found]

        self._local = MemoryBus()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="nats").start()

        async def connect() -> Any:
            nc = await nats.connect(url, max_reconnect_attempts=-1)

            async def on_message(msg: Any) -> None:
                self._local.publish(json.loads(msg.data))

            await nc.subscribe(SUBJECT, cb=on_message)
            return nc

        self._nc = asyncio.run_coroutine_threadsafe(connect(), self._loop).result(timeout=10)

    def publish(self, event: dict[str, Any]) -> None:
        data = json.dumps(event).encode()
        asyncio.run_coroutine_threadsafe(self._nc.publish(SUBJECT, data), self._loop)

    def subscribe(self) -> Subscription:
        return self._local.subscribe()


class LiveStore(Protocol):
    def append_log(self, job_id: str, line: str) -> None: ...
    def logs(self, job_id: str, tail: int = 200) -> list[str]: ...
    def append_metrics(self, job_id: str, points: list[dict[str, Any]]) -> None: ...
    def metrics(self, job_id: str) -> list[dict[str, Any]]: ...


class MemoryLive:
    def __init__(self) -> None:
        self._logs: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=LOG_TAIL))
        self._metrics: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=METRIC_TAIL)
        )
        self._lock = threading.Lock()

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            self._logs[job_id].append(line)

    def logs(self, job_id: str, tail: int = 200) -> list[str]:
        with self._lock:
            return list(self._logs[job_id])[-tail:]

    def append_metrics(self, job_id: str, points: list[dict[str, Any]]) -> None:
        with self._lock:
            self._metrics[job_id].extend(points)

    def metrics(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._metrics[job_id])


class RedisLive:
    """Capped Redis lists, one per job, expiring a day after the last write."""

    TTL = 86400

    def __init__(self, url: str) -> None:
        import redis  # type: ignore[import-not-found]

        self._r = redis.Redis.from_url(url, decode_responses=True)

    def _push(self, key: str, values: list[str], cap: int) -> None:
        pipe = self._r.pipeline()
        pipe.rpush(key, *values)
        pipe.ltrim(key, -cap, -1)
        pipe.expire(key, self.TTL)
        pipe.execute()

    def append_log(self, job_id: str, line: str) -> None:
        self._push(f"forge:logs:{job_id}", [line], LOG_TAIL)

    def logs(self, job_id: str, tail: int = 200) -> list[str]:
        return self._r.lrange(f"forge:logs:{job_id}", -tail, -1)

    def append_metrics(self, job_id: str, points: list[dict[str, Any]]) -> None:
        if points:
            self._push(f"forge:metrics:{job_id}", [json.dumps(p) for p in points], METRIC_TAIL)

    def metrics(self, job_id: str) -> list[dict[str, Any]]:
        return [json.loads(p) for p in self._r.lrange(f"forge:metrics:{job_id}", 0, -1)]


def make_bus(nats_url: str | None) -> Bus:
    return NatsBus(nats_url) if nats_url else MemoryBus()


def make_live(redis_url: str | None) -> LiveStore:
    return RedisLive(redis_url) if redis_url else MemoryLive()
