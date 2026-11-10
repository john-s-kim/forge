"""Operations shared by the API and the controller."""

from __future__ import annotations

import math
import secrets
import time
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from forge import auth, db, telemetry
from forge.config import Settings
from forge.events import Bus, LiveStore, make_bus, make_live

PATCHABLE = {"command", "env", "workers", "gpus_per_worker", "time_limit", "priority"}


class Invalid(ValueError):
    pass


@dataclass
class Context:
    settings: Settings
    sessions: sessionmaker
    bus: Bus
    live: LiveStore

    @classmethod
    def create(cls, settings: Settings | None = None) -> Context:
        s = settings or Settings.from_env()
        return cls(
            s, db.make_sessionmaker(s.database_url), make_bus(s.nats_url), make_live(s.redis_url)
        )


def transition(
    s: Session,
    job: db.Job,
    to: str,
    reason: str | None = None,
    actor: str = "scheduler",
    now: float | None = None,
) -> None:
    """Record a state change. Call `commit` to persist and announce it."""
    at = now if now is not None else time.time()
    s.add(
        db.Event(
            job_id=job.id, from_state=job.state, to_state=to, reason=reason, actor=actor, at=at
        )
    )
    s.info.setdefault("pending", []).append(
        {
            "type": "job.state",
            "job_id": job.id,
            "team": job.team.name,
            "from": job.state,
            "to": to,
            "reason": reason,
            "at": at,
        }
    )
    job.state = to
    if to in db.TERMINAL:
        job.finished_at = at
        job.wait_reason = None
        job.queue_position = None


def commit(s: Session, bus: Bus) -> None:
    """Commit, then publish: subscribers never see an event that was rolled back."""
    pending = s.info.pop("pending", [])
    s.commit()
    for event in pending:
        bus.publish(event)


def create_team(s: Session, name: str, gpu_quota: int) -> db.Team:
    team = db.Team(name=name, gpu_quota=gpu_quota)
    s.add(team)
    s.flush()
    return team


def create_user(
    s: Session, secret: str, name: str, team: db.Team, role: str = "member"
) -> tuple[db.User, str]:
    """Returns the user and the API key. The key is not recoverable afterwards."""
    key = auth.new_api_key()
    user = db.User(name=name, team_id=team.id, role=role, key_hash=auth.hash_key(secret, key))
    s.add(user)
    s.flush()
    return user, key


def validate_spec(spec: dict[str, Any], max_node_gpus: int | None) -> dict[str, Any]:
    def integer(name: str, default: int, lo: int, hi: int) -> int:
        v = spec.get(name, default)
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise Invalid(f"{name} must be an integer between {lo} and {hi}")
        return v

    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise Invalid("command must be a non-empty list of strings")
    env = spec.get("env") or {}
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise Invalid("env must map strings to strings")
    if any(k.startswith("FORGE_") for k in env):
        raise Invalid("env names starting with FORGE_ are reserved")
    priority = spec.get("priority", "normal")
    if priority not in db.PRIORITIES:
        raise Invalid("priority must be low, normal or high")
    time_limit = spec.get("time_limit")
    if time_limit is not None and (
        isinstance(time_limit, bool) or not isinstance(time_limit, int | float) or time_limit <= 0
    ):
        raise Invalid("time_limit must be a positive number of seconds")
    name = spec.get("name") or command[0]
    image = spec.get("image")
    if (
        not isinstance(name, str)
        or len(name) > 128
        or (image is not None and not isinstance(image, str))
    ):
        raise Invalid("name and image must be strings")
    gpus = integer("gpus_per_worker", 1, 1, max_node_gpus or 64)
    return {
        "name": name,
        "image": image,
        "command": command,
        "env": env,
        "workers": integer("workers", 1, 1, 64),
        "gpus_per_worker": gpus,
        "priority": db.PRIORITIES[priority],
        "time_limit": time_limit,
    }


def submit_job(
    s: Session, user: db.User, spec: dict[str, Any], retry_of: str | None = None
) -> db.Job:
    max_node = s.scalar(select(func.max(db.Node.gpus)))
    fields = validate_spec(spec, max_node)
    job = db.Job(
        id=secrets.token_hex(6),
        team_id=user.team_id,
        user_id=user.id,
        state=db.QUEUED,
        submitted_at=time.time(),
        retry_of=retry_of,
        **fields,
    )
    job.team = user.team
    s.add(job)
    s.flush()  # the event row references the job
    s.add(
        db.Event(
            job_id=job.id, from_state=None, to_state=db.QUEUED, actor=user.name, at=job.submitted_at
        )
    )
    s.info.setdefault("pending", []).append(
        {
            "type": "job.state",
            "job_id": job.id,
            "team": user.team.name,
            "from": None,
            "to": db.QUEUED,
            "reason": None,
            "at": job.submitted_at,
        }
    )
    return job


def spec_of(job: db.Job) -> dict[str, Any]:
    return {
        "name": job.name,
        "image": job.image,
        "command": job.command,
        "env": job.env,
        "workers": job.workers,
        "gpus_per_worker": job.gpus_per_worker,
        "priority": db.PRIORITY_NAMES[job.priority],
        "time_limit": job.time_limit,
    }


def apply_patch(spec: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply a proposed fix to a job spec. `env` entries are merged."""
    unknown = set(patch) - PATCHABLE
    if unknown:
        raise Invalid(f"patch may not change: {', '.join(sorted(unknown))}")
    out = dict(spec)
    for k, v in patch.items():
        out[k] = {**spec.get("env", {}), **v} if k == "env" and isinstance(v, dict) else v
    return out


def record_metrics(
    ctx: Context, s: Session, job: db.Job, points: list[dict[str, Any]], now: float | None = None
) -> int:
    at = now if now is not None else time.time()
    clean: list[dict[str, Any]] = []
    for p in points[:5000]:
        name, step, value = p.get("name"), p.get("step"), p.get("value")
        if not isinstance(name, str) or len(name) > 64 or not isinstance(step, int):
            raise Invalid("each point needs a string name and an integer step")
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            v: float | None = float(value)
        else:
            v = None  # NaN, inf and anything unparseable: recorded as a diverged point
        clean.append({"name": name, "step": step, "value": v, "at": at})
    s.add_all(db.Metric(job_id=job.id, **p) for p in clean)
    if diverged := sum(1 for p in clean if p["value"] is None):
        telemetry.DIVERGED.labels(job.team.name).inc(diverged)
    attempt = active_attempt(job)
    if attempt and clean:
        attempt.last_heartbeat_at = at
        attempt.last_step_at = at
    ctx.live.append_metrics(job.id, clean)
    if clean:
        ctx.bus.publish(
            {"type": "job.metrics", "job_id": job.id, "team": job.team.name, "points": clean}
        )
    return len(clean)


def active_attempt(job: db.Job) -> db.Attempt | None:
    return next((a for a in reversed(job.attempts) if a.ended_at is None), None)


def job_view(job: db.Job, detail: bool = False) -> dict[str, Any]:
    attempt = active_attempt(job)
    out: dict[str, Any] = {
        "id": job.id,
        "name": job.name,
        "team": job.team.name,
        "user": job.user.name,
        "state": job.state,
        "priority": db.PRIORITY_NAMES[job.priority],
        "workers": job.workers,
        "gpus_per_worker": job.gpus_per_worker,
        "gpus": job.workers * job.gpus_per_worker,
        "submitted_at": job.submitted_at,
        "finished_at": job.finished_at,
        "wait_reason": job.wait_reason,
        "queue_position": job.queue_position,
        "nodes": attempt.nodes if attempt else [],
        "borrowed": bool(attempt and attempt.borrowed),
        "attempts": len(job.attempts),
        "retry_of": job.retry_of,
        "started_at": job.attempts[0].started_at if job.attempts else None,
    }
    if detail:
        out["spec"] = spec_of(job)
        out["attempt_history"] = [
            {
                "number": a.number,
                "nodes": a.nodes,
                "borrowed": a.borrowed,
                "started_at": a.started_at,
                "ended_at": a.ended_at,
                "exit_code": a.exit_code,
                "exit_reason": a.exit_reason,
            }
            for a in job.attempts
        ]
    return out
