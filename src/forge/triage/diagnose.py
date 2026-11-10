"""Failure triage.

When a job fails, gather the evidence, classify the failure and store a
diagnosis. A diagnosis can carry a proposed change to the job spec, but
nothing is resubmitted until a person approves it through the API.
"""

from __future__ import annotations

import logging
import time

from sqlalchemy import select

from forge import db, service
from forge.service import Context
from forge.triage.evidence import Evidence, classify

log = logging.getLogger("forge.triage")


def gather(ctx: Context, job: db.Job, attempt: db.Attempt) -> Evidence:
    with ctx.sessions() as s:
        rows = s.scalars(select(db.Metric).where(db.Metric.job_id == job.id).order_by(db.Metric.id))
        metrics: dict[str, list[tuple[int, float | None]]] = {}
        for m in rows:
            metrics.setdefault(m.name, []).append((m.step, m.value))
    return Evidence(
        spec=service.spec_of(job),
        exit_code=attempt.exit_code,
        exit_reason=attempt.exit_reason,
        attempts=len(job.attempts),
        logs=(attempt.log_tail or "").splitlines(),
        metrics={k: v[-50:] for k, v in metrics.items()},
        backend_events=list(attempt.detail or []),
    )


def diagnose(ctx: Context, job_id: str, attempt_id: int) -> None:
    """Diagnose a failed attempt and store the result. Never raises: triage
    must not take the controller down."""
    try:
        with ctx.sessions() as s:
            job, attempt = s.get(db.Job, job_id), s.get(db.Attempt, attempt_id)
            if job is None or attempt is None:
                return
            finding = classify(gather(ctx, job, attempt))
            team = job.team.name
            s.add(
                db.Diagnosis(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    category=finding.category,
                    summary=finding.summary,
                    evidence=finding.evidence,
                    proposed_patch=finding.proposed_patch,
                    created_at=time.time(),
                )
            )
            s.commit()
        ctx.bus.publish(
            {"type": "job.diagnosis", "job_id": job_id, "team": team, "category": finding.category}
        )
    except Exception:
        log.exception("triage failed for job %s", job_id)
