"""What is known about a failed attempt, and a rule-based classifier over it.

The rules are deliberately simple: patterns over the exit status, the log tail,
the metrics and the cluster events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CATEGORIES = [
    "oom",  # ran out of GPU or host memory
    "diverged",  # loss became NaN or infinite
    "bad_input_path",  # dataset, config or checkpoint path does not exist
    "image_pull",  # container image could not be pulled
    "time_limit",  # exceeded its declared time limit
    "user_code_error",  # any other bug in the training code
    "infra",  # node, network or platform fault; not the job's doing
    "unknown",
]


@dataclass
class Evidence:
    spec: dict[str, Any]
    exit_code: int | None = None
    exit_reason: str | None = None
    attempts: int = 1
    logs: list[str] = field(default_factory=list)
    metrics: dict[str, list[tuple[int, float | None]]] = field(default_factory=dict)
    backend_events: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Evidence:
        metrics = {k: [(int(s), v) for s, v in pts] for k, pts in d.get("metrics", {}).items()}
        return cls(
            spec=d["spec"],
            exit_code=d.get("exit_code"),
            exit_reason=d.get("exit_reason"),
            attempts=d.get("attempts", 1),
            logs=d.get("logs", []),
            metrics=metrics,
            backend_events=d.get("backend_events", []),
        )


@dataclass
class Finding:
    category: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    proposed_patch: dict[str, Any] | None = None


_OOM = re.compile(r"out of memory|OOMKilled|MemoryError|Cannot allocate memory", re.I)
_NAN = re.compile(r"\bloss\b.*\b(nan|inf)\b|\b(nan|inf)\b.*\bloss\b", re.I)
_PATH = re.compile(r"FileNotFoundError|No such file or directory|does not exist", re.I)
_PULL = re.compile(r"ErrImagePull|ImagePullBackOff|InvalidImageName|image_pull", re.I)
_INFRA = re.compile(
    r"attempt lost|pod deleted|node .*not ready|Evicted|connection (reset|refused)", re.I
)


def _matching(lines: list[str], pattern: re.Pattern[str], limit: int = 3) -> list[str]:
    return [line for line in lines if pattern.search(line)][-limit:]


def _scale_flag(
    command: list[str], flags: tuple[str, ...], factor: float, as_int: bool
) -> list[str] | None:
    """Return a copy of `command` with a numeric flag scaled, e.g. halve --batch-size."""
    out = list(command)
    for i, arg in enumerate(out):
        name, eq, value = arg.partition("=")
        if name not in flags:
            continue
        if not eq:
            if i + 1 >= len(out):
                return None
            value = out[i + 1]
        try:
            scaled = float(value) * factor
        except ValueError:
            return None
        text = str(max(1, int(scaled))) if as_int else f"{scaled:g}"
        if eq:
            out[i] = f"{name}={text}"
        else:
            out[i + 1] = text
        return out
    return None


def classify(ev: Evidence) -> Finding:
    status = [ev.exit_reason or "", *ev.backend_events]
    command = ev.spec.get("command", [])

    if hits := _matching(status, _PULL):
        return Finding(
            "image_pull", f"The image {ev.spec.get('image')!r} could not be pulled.", hits
        )
    if ev.exit_reason == "time_limit":
        limit = ev.spec.get("time_limit")
        patch = {"time_limit": limit * 2} if limit else None
        return Finding(
            "time_limit",
            f"The job ran past its {limit}s time limit and was stopped.",
            [ev.exit_reason],
            patch,
        )
    if (hits := _matching(ev.logs + status, _OOM)) or ev.exit_code == 137:
        cmd = _scale_flag(command, ("--batch-size", "--batch_size", "-b"), 0.5, as_int=True)
        return Finding(
            "oom",
            "The job ran out of memory.",
            hits or [f"exit code {ev.exit_code}"],
            {"command": cmd} if cmd else None,
        )
    bad = [
        f"{name} at step {s} was not finite"
        for name, pts in ev.metrics.items()
        for s, v in pts
        if v is None
    ][:3]
    if bad or (hits := _matching(ev.logs, _NAN)):
        cmd = _scale_flag(
            command, ("--lr", "--learning-rate", "--learning_rate"), 0.1, as_int=False
        )
        return Finding(
            "diverged",
            "Training diverged: a metric became NaN or infinite.",
            bad or hits,
            {"command": cmd} if cmd else None,
        )
    if hits := _matching(ev.logs, _PATH):
        return Finding("bad_input_path", "A file or directory the job needs does not exist.", hits)
    if hits := _matching(status, _INFRA):
        return Finding(
            "infra", "The attempt was lost to a platform fault, not a bug in the job.", hits
        )
    if any("Traceback (most recent call last)" in line for line in ev.logs):
        tail = [line for line in ev.logs if line.strip()][-2:]
        return Finding("user_code_error", "The training code raised an exception.", tail)
    return Finding(
        "unknown", "No known failure signature was found.", status[:1] if status[0] else []
    )
