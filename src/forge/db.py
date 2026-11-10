from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import JSON, ForeignKey, Index, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

QUEUED = "QUEUED"
STARTING = "STARTING"
RUNNING = "RUNNING"
PREEMPTING = "PREEMPTING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
CANCELLED = "CANCELLED"
ACTIVE = (STARTING, RUNNING, PREEMPTING)
TERMINAL = (SUCCEEDED, FAILED, CANCELLED)

PRIORITIES = {"low": 0, "normal": 1, "high": 2}
PRIORITY_NAMES = {v: k for k, v in PRIORITIES.items()}


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(63), unique=True)
    gpu_quota: Mapped[int] = mapped_column(default=0)


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(63), unique=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    role: Mapped[str] = mapped_column(String(16), default="member")  # member | admin
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    team: Mapped[Team] = relationship()


class Node(Base):
    """Inventory written by the controller so the API can show the cluster."""

    __tablename__ = "nodes"
    name: Mapped[str] = mapped_column(String(253), primary_key=True)
    gpus: Mapped[int]


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(128))
    image: Mapped[str | None] = mapped_column(String(255))
    command: Mapped[list[str]] = mapped_column(JSON)
    env: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    workers: Mapped[int] = mapped_column(default=1)
    gpus_per_worker: Mapped[int] = mapped_column(default=1)
    priority: Mapped[int] = mapped_column(default=1)
    time_limit: Mapped[float | None]
    state: Mapped[str] = mapped_column(String(16), default=QUEUED, index=True)
    wait_reason: Mapped[str | None] = mapped_column(String(32))
    queue_position: Mapped[int | None]
    submitted_at: Mapped[float]
    finished_at: Mapped[float | None]
    retry_of: Mapped[str | None] = mapped_column(String(16))
    team: Mapped[Team] = relationship()
    user: Mapped[User] = relationship()
    attempts: Mapped[list[Attempt]] = relationship(back_populates="job", order_by="Attempt.number")


class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    number: Mapped[int]
    nodes: Mapped[list[str]] = mapped_column(JSON)
    borrowed: Mapped[bool] = mapped_column(default=False)
    started_at: Mapped[float]
    ended_at: Mapped[float | None] = mapped_column(index=True)
    preempt_requested_at: Mapped[float | None]
    last_heartbeat_at: Mapped[float | None]
    last_step_at: Mapped[float | None]
    exit_code: Mapped[int | None]
    exit_reason: Mapped[str | None] = mapped_column(String(64))
    detail: Mapped[list[Any]] = mapped_column(JSON, default=list)  # backend events at exit
    log_tail: Mapped[str | None] = mapped_column(Text)
    job: Mapped[Job] = relationship(back_populates="attempts")


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"))
    step: Mapped[int]
    path: Mapped[str] = mapped_column(String(512))
    size: Mapped[int | None]  # null until the upload completes
    created_at: Mapped[float]


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    from_state: Mapped[str | None] = mapped_column(String(16))
    to_state: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str | None] = mapped_column(String(255))
    actor: Mapped[str] = mapped_column(String(63), default="scheduler")
    at: Mapped[float]


class Metric(Base):
    __tablename__ = "metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"))
    name: Mapped[str] = mapped_column(String(64))
    step: Mapped[int]
    value: Mapped[float | None]  # null encodes NaN/inf, which JSON and SQL cannot carry
    at: Mapped[float]
    __table_args__ = (Index("ix_metrics_job_name_step", "job_id", "name", "step"),)


class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    attempt_id: Mapped[int] = mapped_column(ForeignKey("attempts.id"))
    category: Mapped[str] = mapped_column(String(32))
    summary: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    proposed_patch: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    approved_by: Mapped[str | None] = mapped_column(String(63))
    resubmitted_as: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[float]


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///") and ":memory:" not in url:
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _pragmas(conn: Any, _: Any) -> None:
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return engine


def make_sessionmaker(url: str) -> sessionmaker:
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
