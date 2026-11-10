"""The reconcile loop.

Each cycle: learn what happened to running attempts, snapshot the cluster,
ask the scheduler core what to do, and apply it. All durable state is in the
database, so the controller can be restarted at any point.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from forge import auth, db, service, telemetry
from forge.controller.backend import Backend, Launch, LocalBackend, Status
from forge.scheduler import core
from forge.service import Context

log = logging.getLogger("forge.controller")

STALL_SECONDS = 300.0
IDLE_SECONDS = 600.0


def make_backend(ctx: Context) -> Backend:
    def sink(job_id: str, team: str, line: str) -> None:
        ctx.live.append_log(job_id, line)
        ctx.bus.publish({"type": "job.log", "job_id": job_id, "team": team, "line": line})

    if ctx.settings.backend == "kube":
        from forge.controller.kube import KubeBackend

        return KubeBackend(ctx.settings.gpu_resource, sink)
    return LocalBackend(ctx.settings.local_nodes, ctx.settings.data_dir / "logs", sink)


class Controller:
    def __init__(
        self,
        ctx: Context,
        backend: Backend,
        policy: core.Policy | None = None,
        on_failure: Callable[[str, int], None] | None = None,
    ) -> None:
        self.ctx = ctx
        self.backend = backend
        self.policy = policy or core.Policy(min_runtime=ctx.settings.min_runtime)
        self.on_failure = on_failure
        self._adopted = False

    # ---- public ------------------------------------------------------------

    def run_forever(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.reconcile()
            except Exception:
                log.exception("reconcile failed")
            stop.wait(self.ctx.settings.cycle_seconds)

    def reconcile(self, now: float | None = None) -> core.Decision:
        now = now if now is not None else time.time()
        t0 = time.perf_counter()
        failed: list[tuple[str, int]] = []
        with self.ctx.sessions() as s:
            nodes = self.backend.nodes()
            self._sync_nodes(s, nodes)
            active = list(s.scalars(select(db.Attempt).where(db.Attempt.ended_at.is_(None))))
            if not self._adopted:
                for a in active:
                    self.backend.adopt(self._launch(a.job, a))
                self._adopted = True

            still_running = [a for a in active if not self._observe(s, a, now, failed)]
            cluster = self._snapshot(s, nodes, still_running, now)
            decision = core.schedule(cluster, now, self.policy)
            self._apply(s, decision, cluster, still_running, now)
            self._export(s, cluster, still_running, nodes, now)
            service.commit(s, self.ctx.bus)
        telemetry.CYCLE.observe(time.perf_counter() - t0)
        if self.on_failure:
            for job_id, attempt_id in failed:
                self.on_failure(job_id, attempt_id)
        return decision

    # ---- steps -------------------------------------------------------------

    def _sync_nodes(self, s: Session, nodes: dict[str, int]) -> None:
        known = {n.name: n for n in s.scalars(select(db.Node))}
        for name, gpus in nodes.items():
            if name not in known:
                s.add(db.Node(name=name, gpus=gpus))
            elif known[name].gpus != gpus:
                known[name].gpus = gpus
        for name, node in known.items():
            if name not in nodes:
                s.delete(node)

    def _launch(self, job: db.Job, a: db.Attempt) -> Launch:
        st = self.ctx.settings
        env = {
            **job.env,
            "FORGE_API_URL": st.api_url,
            "FORGE_JOB_ID": job.id,
            "FORGE_JOB_TOKEN": auth.job_token(st.secret, job.id),
            "FORGE_ATTEMPT": str(a.number),
            "FORGE_NUM_WORKERS": str(job.workers),
            "FORGE_GPUS": str(job.gpus_per_worker),
        }
        return Launch(
            job_id=job.id,
            team=job.team.name,
            attempt_id=a.id,
            number=a.number,
            image=job.image,
            command=job.command,
            env=env,
            nodes=list(a.nodes),
            gpus_per_worker=job.gpus_per_worker,
            grace_seconds=st.preemption_grace_seconds,
        )

    def _observe(
        self, s: Session, a: db.Attempt, now: float, failed: list[tuple[str, int]]
    ) -> bool:
        """Bring one attempt up to date. Returns True if it has ended."""
        job = a.job
        st: Status = self.backend.status(a.id)

        if st.phase in ("starting", "running"):
            if job.state == db.STARTING and st.phase == "running":
                self._transition(s, job, db.RUNNING, now=now)
            if job.state == db.CANCELLED:
                self.backend.kill(a.id)
            elif job.state == db.PREEMPTING and a.preempt_requested_at is not None:
                if now - a.preempt_requested_at > self.ctx.settings.preemption_grace_seconds:
                    self.backend.kill(a.id)  # had its chance to checkpoint
            elif job.time_limit and now - a.started_at > job.time_limit and not a.exit_reason:
                a.exit_reason = "time_limit"
                self.backend.kill(a.id)
            return False

        a.ended_at = now
        a.exit_code = st.exit_code
        a.detail = st.events
        a.log_tail = "\n".join(self.ctx.live.logs(job.id, 200))
        self.backend.forget(a.id)

        if job.state == db.CANCELLED:
            a.exit_reason = "cancelled"
        elif st.phase == "succeeded":
            self._transition(s, job, db.SUCCEEDED, now=now)
        elif job.state == db.PREEMPTING:
            a.exit_reason = "preempted"
            self._transition(s, job, db.QUEUED, "preempted", now=now)
        elif st.phase == "gone":
            # Lost through no fault of the job (node loss, controller restart): retry.
            a.exit_reason = st.reason
            self._transition(s, job, db.QUEUED, st.reason, now=now)
        else:
            a.exit_reason = a.exit_reason or st.reason
            self._transition(s, job, db.FAILED, a.exit_reason, now=now)
            failed.append((job.id, a.id))
        return True

    def _snapshot(
        self, s: Session, nodes: dict[str, int], running: list[db.Attempt], now: float
    ) -> core.Cluster:
        preempted: dict[str, list[float]] = {}
        for job_id, at in s.execute(
            select(db.Event.job_id, db.Event.at).where(
                db.Event.to_state == db.PREEMPTING, db.Event.at > now - 3600
            )
        ):
            preempted.setdefault(job_id, []).append(at)
        last_ckpt = dict(
            s.execute(
                select(db.Checkpoint.attempt_id, func.max(db.Checkpoint.created_at))
                .where(db.Checkpoint.size.is_not(None))
                .group_by(db.Checkpoint.attempt_id)
            ).all()
        )

        def to_core(job: db.Job) -> core.Job:
            return core.Job(
                id=job.id,
                team=job.team.name,
                priority=core.Priority(job.priority),
                workers=job.workers,
                gpus_per_worker=job.gpus_per_worker,
                submitted_at=job.submitted_at,
                time_limit=job.time_limit,
                preempted_at=tuple(preempted.get(job.id, ())),
            )

        cluster = core.Cluster(
            nodes=nodes, quotas={t.name: t.gpu_quota for t in s.scalars(select(db.Team))}
        )
        for a in running:
            cluster.running[a.job_id] = core.Attempt(
                job=to_core(a.job),
                nodes=tuple(a.nodes),
                started_at=a.started_at,
                last_checkpoint_at=last_ckpt.get(a.id),
                # Anything not plainly running is on its way out.
                preempting=a.job.state not in (db.STARTING, db.RUNNING),
            )
        cluster.queue = [
            to_core(j) for j in s.scalars(select(db.Job).where(db.Job.state == db.QUEUED))
        ]
        return cluster

    def _apply(
        self,
        s: Session,
        d: core.Decision,
        cluster: core.Cluster,
        running: list[db.Attempt],
        now: float,
    ) -> None:
        by_job = {a.job_id: a for a in running}
        for job_id in d.preemptions:
            a = by_job[job_id]
            a.preempt_requested_at = now
            self._transition(s, a.job, db.PREEMPTING, "capacity reclaimed", now=now)
            telemetry.PREEMPTIONS.inc()
            self.backend.stop(a.id)

        positions = {job_id: i + 1 for i, job_id in enumerate(d.order)}
        placed = {p.job_id: p for p in d.placements}
        for cj in cluster.queue:
            job = s.get(db.Job, cj.id)
            assert job is not None
            if cj.id in placed:
                p = placed[cj.id]
                a = db.Attempt(
                    job_id=job.id,
                    number=len(job.attempts) + 1,
                    nodes=list(p.nodes),
                    borrowed=p.borrowed,
                    started_at=now,
                    last_heartbeat_at=now,
                    last_step_at=now,
                )
                job.attempts.append(a)
                s.flush()
                if a.number == 1:
                    telemetry.QUEUE_WAIT.labels(
                        job.team.name, db.PRIORITY_NAMES[job.priority]
                    ).observe(now - job.submitted_at)
                job.wait_reason, job.queue_position = None, None
                self._transition(s, job, db.STARTING, now=now)
                self.backend.start(self._launch(job, a))
            else:
                reason = d.wait_reasons.get(cj.id)
                job.wait_reason = str(reason) if reason else None
                job.queue_position = positions.get(cj.id)

    def _transition(
        self, s: Session, job: db.Job, to: str, reason: str | None = None, *, now: float
    ) -> None:
        service.transition(s, job, to, reason, now=now)
        telemetry.TRANSITIONS.labels(to).inc()

    def _export(
        self,
        s: Session,
        cluster: core.Cluster,
        running: list[db.Attempt],
        nodes: dict[str, int],
        now: float,
    ) -> None:
        telemetry.GPUS_TOTAL.set(sum(nodes.values()))
        borrowed = core.borrowed_ids(cluster, self.policy.use_quotas)
        used = dict.fromkeys(cluster.quotas, 0)
        lent = dict.fromkeys(cluster.quotas, 0)
        stalled = idle = 0
        for a in running:
            gpus = a.job.workers * a.job.gpus_per_worker
            team = a.job.team.name
            used[team] = used.get(team, 0) + gpus
            if a.job_id in borrowed:
                lent[team] = lent.get(team, 0) + gpus
            # Borrowed status can change as teammates finish; keep the row current.
            a.borrowed = a.job_id in borrowed
            if a.job.state == db.RUNNING:
                if now - (a.last_heartbeat_at or a.started_at) > STALL_SECONDS:
                    stalled += 1
                if now - (a.last_step_at or a.started_at) > IDLE_SECONDS:
                    idle += gpus
        for team, n in used.items():
            telemetry.GPUS_ALLOCATED.labels(team).set(n)
            telemetry.GPUS_BORROWED.labels(team).set(lent.get(team, 0))
        telemetry.STALLED.set(stalled)
        telemetry.IDLE_GPUS.set(idle)
        depth = dict.fromkeys(db.PRIORITIES, 0)
        for j in cluster.queue:
            depth[db.PRIORITY_NAMES[int(j.priority)]] += 1
        for name, n in depth.items():
            telemetry.QUEUE_DEPTH.labels(name).set(n)
