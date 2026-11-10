from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from functools import wraps
from pathlib import Path
from typing import Any

from flask import Flask, Response, g, jsonify, request, send_from_directory
from prometheus_client import CONTENT_TYPE_LATEST, Counter, generate_latest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from forge import auth, db, service
from forge.service import Context, Invalid

REQUESTS = Counter("forge_api_requests_total", "API requests", ["method", "endpoint", "status"])


class RateLimiter:
    """Token bucket per caller. In-process: with several API replicas the
    effective limit is multiplied by the replica count."""

    def __init__(self, rate: float, burst: float) -> None:
        self.rate, self.burst = rate, burst
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, who: str, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        with self._lock:
            tokens, last = self._state.get(who, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1.0
            self._state[who] = (tokens - 1.0 if ok else tokens, now)
            return ok


def create_app(ctx: Context, web_dir: str | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 1 << 30  # checkpoint uploads
    secret = ctx.settings.secret
    submit_limit = RateLimiter(rate=2.0, burst=20.0)
    ckpt_dir = ctx.settings.data_dir / "checkpoints"

    # ---- plumbing ----------------------------------------------------------

    @app.before_request
    def open_session() -> None:
        g.s = ctx.sessions()
        g.user = None
        g.job_scope = None
        header = request.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            token = header.removeprefix("Bearer ").strip()
            if token.startswith(auth.JOB_TOKEN_PREFIX):
                g.job_scope = auth.verify_job_token(secret, token)
            else:
                g.user = g.s.scalar(
                    select(db.User).where(db.User.key_hash == auth.hash_key(secret, token))
                )

    @app.teardown_request
    def close_session(_: BaseException | None) -> None:
        s = g.pop("s", None)
        if s is not None:
            s.close()

    @app.after_request
    def count(resp: Response) -> Response:
        REQUESTS.labels(request.method, request.endpoint or "unknown", resp.status_code).inc()
        return resp

    def error(status: int, message: str) -> tuple[Response, int]:
        return jsonify({"error": message}), status

    @app.errorhandler(Invalid)
    def invalid(e: Invalid) -> tuple[Response, int]:
        return error(400, str(e))

    def user_required(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            if g.user is None:
                return error(401, "missing or invalid API key")
            return fn(*a, **kw)

        return wrapper

    def admin_required(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            if g.user is None:
                return error(401, "missing or invalid API key")
            if g.user.role != "admin":
                return error(403, "admin only")
            return fn(*a, **kw)

        return wrapper

    def load_job(job_id: str, allow_job_token: bool = False) -> db.Job | tuple[Response, int]:
        """The job, if the caller may act on it. A job in another team is
        reported as missing rather than forbidden, so ids cannot be probed."""
        if g.user is None and not (allow_job_token and g.job_scope):
            return error(401, "missing or invalid credentials")
        job = g.s.get(db.Job, job_id)
        if job is None:
            return error(404, "no such job")
        if g.user is not None:
            if g.user.role != "admin" and job.team_id != g.user.team_id:
                return error(404, "no such job")
        elif g.job_scope != job_id:
            return error(404, "no such job")
        return job

    def body() -> dict[str, Any]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise Invalid("request body must be a JSON object")
        return data

    # ---- jobs --------------------------------------------------------------

    @app.post("/v1/jobs")
    @user_required
    def submit() -> Any:
        if not submit_limit.allow(str(g.user.id)):
            return error(429, "too many submissions, slow down")
        job = service.submit_job(g.s, g.user, body())
        service.commit(g.s, ctx.bus)
        return jsonify(service.job_view(job, detail=True)), 201

    @app.get("/v1/jobs")
    @user_required
    def list_jobs() -> Any:
        q = select(db.Job).order_by(db.Job.submitted_at.desc())
        if g.user.role != "admin" or request.args.get("mine"):
            q = q.where(db.Job.team_id == g.user.team_id)
        if state := request.args.get("state"):
            q = q.where(db.Job.state.in_(state.upper().split(",")))
        limit = min(int(request.args.get("limit", 200)), 1000)
        return jsonify({"jobs": [service.job_view(j) for j in g.s.scalars(q.limit(limit))]})

    @app.get("/v1/jobs/<job_id>")
    def get_job(job_id: str) -> Any:
        job = load_job(job_id)
        if not isinstance(job, db.Job):
            return job
        view = service.job_view(job, detail=True)
        view["events"] = [
            {
                "from": e.from_state,
                "to": e.to_state,
                "reason": e.reason,
                "actor": e.actor,
                "at": e.at,
            }
            for e in g.s.scalars(
                select(db.Event).where(db.Event.job_id == job.id).order_by(db.Event.id)
            )
        ]
        view["checkpoints"] = [
            {"step": c.step, "size": c.size, "created_at": c.created_at}
            for c in g.s.scalars(
                select(db.Checkpoint)
                .where(db.Checkpoint.job_id == job.id, db.Checkpoint.size.is_not(None))
                .order_by(db.Checkpoint.step)
            )
        ]
        return jsonify(view)

    @app.post("/v1/jobs/<job_id>/cancel")
    def cancel(job_id: str) -> Any:
        job = load_job(job_id)
        if not isinstance(job, db.Job):
            return job
        if job.state in db.TERMINAL:
            return error(409, f"job is already {job.state}")
        # The controller stops any running attempt on its next cycle.
        service.transition(g.s, job, db.CANCELLED, "cancelled by user", actor=g.user.name)
        service.commit(g.s, ctx.bus)
        return jsonify(service.job_view(job))

    @app.post("/v1/jobs/<job_id>/metrics")
    def post_metrics(job_id: str) -> Any:
        job = load_job(job_id, allow_job_token=True)
        if not isinstance(job, db.Job):
            return job
        points = body().get("points")
        if not isinstance(points, list):
            raise Invalid("points must be a list")
        n = service.record_metrics(ctx, g.s, job, points)
        g.s.commit()
        return jsonify({"accepted": n})

    @app.get("/v1/jobs/<job_id>/metrics")
    def get_metrics(job_id: str) -> Any:
        job = load_job(job_id)
        if not isinstance(job, db.Job):
            return job
        points = ctx.live.metrics(job.id)
        if not points or request.args.get("full"):
            rows = g.s.scalars(
                select(db.Metric)
                .where(db.Metric.job_id == job.id)
                .order_by(db.Metric.id)
                .limit(20000)
            )
            points = [{"name": m.name, "step": m.step, "value": m.value, "at": m.at} for m in rows]
        return jsonify({"points": points})

    @app.post("/v1/jobs/<job_id>/heartbeat")
    def heartbeat(job_id: str) -> Any:
        job = load_job(job_id, allow_job_token=True)
        if not isinstance(job, db.Job):
            return job
        attempt = service.active_attempt(job)
        if attempt:
            attempt.last_heartbeat_at = time.time()
            g.s.commit()
        # Tells the SDK whether the scheduler wants this attempt to stop.
        return jsonify({"state": job.state, "preempting": job.state == db.PREEMPTING})

    @app.get("/v1/jobs/<job_id>/logs")
    def logs(job_id: str) -> Any:
        job = load_job(job_id)
        if not isinstance(job, db.Job):
            return job
        tail = min(int(request.args.get("tail", 200)), 2000)
        lines = ctx.live.logs(job.id, tail)
        if not lines and job.attempts and job.attempts[-1].log_tail:
            lines = job.attempts[-1].log_tail.splitlines()[-tail:]
        return jsonify({"lines": lines})

    # ---- checkpoints -------------------------------------------------------

    def blob_path(job_id: str, step: int) -> str:
        return f"/v1/blobs/{job_id}/{step}"

    @app.post("/v1/jobs/<job_id>/checkpoints")
    def register_checkpoint(job_id: str) -> Any:
        job = load_job(job_id, allow_job_token=True)
        if not isinstance(job, db.Job):
            return job
        step = body().get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise Invalid("step must be a non-negative integer")
        attempt = service.active_attempt(job)
        if attempt is None:
            return error(409, "job has no running attempt")
        path = blob_path(job.id, step)
        g.s.add(
            db.Checkpoint(
                job_id=job.id, attempt_id=attempt.id, step=step, path=path, created_at=time.time()
            )
        )
        g.s.commit()
        return jsonify({"upload_url": f"{path}?sig={auth.sign(secret, 'PUT', path)}"}), 201

    @app.get("/v1/jobs/<job_id>/checkpoints/latest")
    def latest_checkpoint(job_id: str) -> Any:
        job = load_job(job_id, allow_job_token=True)
        if not isinstance(job, db.Job):
            return job
        c = g.s.scalar(
            select(db.Checkpoint)
            .where(db.Checkpoint.job_id == job.id, db.Checkpoint.size.is_not(None))
            .order_by(db.Checkpoint.step.desc(), db.Checkpoint.id.desc())
        )
        if c is None:
            return error(404, "no checkpoint")
        return jsonify(
            {
                "step": c.step,
                "size": c.size,
                "download_url": f"{c.path}?sig={auth.sign(secret, 'GET', c.path)}",
            }
        )

    @app.route("/v1/blobs/<job_id>/<int:step>", methods=["GET", "PUT"])
    def blob(job_id: str, step: int) -> Any:
        """Signed-URL target. The signature is the only credential, which is
        the same contract an S3 presigned URL would give the SDK."""
        path = blob_path(job_id, step)
        if not auth.verify_signature(secret, request.method, path, request.args.get("sig", "")):
            return error(403, "bad or expired signature")
        if not job_id.isalnum():
            return error(404, "no such blob")
        file = ckpt_dir / job_id / f"{step}.ckpt"
        if request.method == "GET":
            if not file.exists():
                return error(404, "no such blob")
            return send_from_directory(
                file.parent.resolve(), file.name, mimetype="application/octet-stream"
            )
        c = g.s.scalar(
            select(db.Checkpoint)
            .where(db.Checkpoint.job_id == job_id, db.Checkpoint.step == step)
            .order_by(db.Checkpoint.id.desc())
        )
        if c is None:
            return error(404, "checkpoint was not registered")
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(".tmp")
        size = 0
        with open(tmp, "wb") as f:
            while chunk := request.stream.read(1 << 20):
                f.write(chunk)
                size += len(chunk)
        os.replace(tmp, file)  # readers never see a partial checkpoint
        c.size = size
        c.created_at = time.time()
        g.s.commit()
        return jsonify({"size": size})

    # ---- diagnosis ---------------------------------------------------------

    def diagnosis_view(d: db.Diagnosis) -> dict[str, Any]:
        return {
            "id": d.id,
            "category": d.category,
            "summary": d.summary,
            "evidence": d.evidence,
            "proposed_patch": d.proposed_patch,
            "approved_by": d.approved_by,
            "resubmitted_as": d.resubmitted_as,
            "created_at": d.created_at,
        }

    def latest_diagnosis(job: db.Job) -> db.Diagnosis | None:
        return g.s.scalar(
            select(db.Diagnosis)
            .where(db.Diagnosis.job_id == job.id)
            .order_by(db.Diagnosis.id.desc())
        )

    @app.get("/v1/jobs/<job_id>/diagnosis")
    def get_diagnosis(job_id: str) -> Any:
        job = load_job(job_id)
        if not isinstance(job, db.Job):
            return job
        d = latest_diagnosis(job)
        return jsonify(diagnosis_view(d)) if d else error(404, "no diagnosis")

    @app.post("/v1/jobs/<job_id>/diagnosis/approve")
    def approve_diagnosis(job_id: str) -> Any:
        """A person accepts the proposed fix; only then is anything resubmitted."""
        job = load_job(job_id)
        if not isinstance(job, db.Job):
            return job
        d = latest_diagnosis(job)
        if d is None or not d.proposed_patch:
            return error(409, "nothing to approve")
        if d.resubmitted_as:
            return error(409, f"already resubmitted as {d.resubmitted_as}")
        spec = service.apply_patch(service.spec_of(job), d.proposed_patch)
        new = service.submit_job(g.s, g.user, spec, retry_of=job.id)
        d.approved_by, d.resubmitted_as = g.user.name, new.id
        service.commit(g.s, ctx.bus)
        return jsonify(service.job_view(new, detail=True)), 201

    # ---- cluster and admin -------------------------------------------------

    @app.get("/v1/cluster")
    @user_required
    def cluster() -> Any:
        nodes: dict[str, dict[str, Any]] = {
            n.name: {"name": n.name, "gpus": n.gpus, "allocations": []}
            for n in g.s.scalars(select(db.Node).order_by(db.Node.name))
        }
        used: dict[str, dict[str, int]] = {}
        attempts = g.s.scalars(select(db.Attempt).where(db.Attempt.ended_at.is_(None)))
        for a in attempts:
            job = a.job
            if job.state not in db.ACTIVE:
                continue
            u = used.setdefault(job.team.name, {"used": 0, "borrowed": 0})
            u["used"] += job.workers * job.gpus_per_worker
            if a.borrowed:
                u["borrowed"] += job.workers * job.gpus_per_worker
            for node in a.nodes:
                if node in nodes:
                    nodes[node]["allocations"].append(
                        {
                            "job_id": job.id,
                            "name": job.name,
                            "team": job.team.name,
                            "gpus": job.gpus_per_worker,
                            "borrowed": a.borrowed,
                            "state": job.state,
                            "priority": db.PRIORITY_NAMES[job.priority],
                        }
                    )
        teams = [
            {"name": t.name, "quota": t.gpu_quota, **used.get(t.name, {"used": 0, "borrowed": 0})}
            for t in g.s.scalars(select(db.Team).order_by(db.Team.name))
        ]
        return jsonify({"nodes": list(nodes.values()), "teams": teams})

    @app.get("/v1/me")
    @user_required
    def me() -> Any:
        return jsonify({"name": g.user.name, "team": g.user.team.name, "role": g.user.role})

    @app.post("/v1/teams")
    @admin_required
    def create_team() -> Any:
        b = body()
        name, quota = b.get("name"), b.get("gpu_quota", 0)
        if not isinstance(name, str) or not name.replace("-", "").isalnum() or len(name) > 40:
            raise Invalid("name must be alphanumeric (dashes allowed), at most 40 characters")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
            raise Invalid("gpu_quota must be a non-negative integer")
        try:
            team = service.create_team(g.s, name, quota)
            g.s.commit()
        except IntegrityError:
            return error(409, "team already exists")
        return jsonify({"name": team.name, "gpu_quota": team.gpu_quota}), 201

    @app.put("/v1/teams/<name>/quota")
    @admin_required
    def set_quota(name: str) -> Any:
        team = g.s.scalar(select(db.Team).where(db.Team.name == name))
        if team is None:
            return error(404, "no such team")
        quota = body().get("gpu_quota")
        if isinstance(quota, bool) or not isinstance(quota, int) or quota < 0:
            raise Invalid("gpu_quota must be a non-negative integer")
        team.gpu_quota = quota
        g.s.commit()
        return jsonify({"name": team.name, "gpu_quota": team.gpu_quota})

    @app.post("/v1/users")
    @admin_required
    def create_user() -> Any:
        b = body()
        team = g.s.scalar(select(db.Team).where(db.Team.name == b.get("team")))
        if team is None:
            return error(404, "no such team")
        name, role = b.get("name"), b.get("role", "member")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 63
            or role not in ("member", "admin")
        ):
            raise Invalid("name is required and role must be member or admin")
        try:
            user, key = service.create_user(g.s, secret, name, team, role)
            g.s.commit()
        except IntegrityError:
            return error(409, "user already exists")
        return jsonify({"name": user.name, "team": team.name, "role": role, "api_key": key}), 201

    # ---- live feed ---------------------------------------------------------

    @app.get("/v1/stream")
    @user_required
    def stream() -> Any:
        team = None if g.user.role == "admin" else g.user.team.name
        only_job = request.args.get("job")
        sub = ctx.bus.subscribe()

        def gen() -> Iterator[str]:
            try:
                yield ": connected\n\n"
                while True:
                    ev = sub.get(timeout=15.0)
                    if ev is None:
                        yield ": keepalive\n\n"
                    elif (team is None or ev.get("team") == team) and (
                        only_job is None or ev.get("job_id") == only_job
                    ):
                        if ev["type"] == "job.log" and only_job is None:
                            continue  # logs are only sent to a per-job stream
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            finally:
                sub.close()

        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- ops ---------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"ok": True})

    @app.get("/metrics")
    def prometheus() -> Any:
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

    if web_dir and Path(web_dir).is_dir():
        root = str(Path(web_dir).resolve())

        @app.get("/")
        @app.get("/<path:path>")
        def web(path: str = "index.html") -> Any:
            if not (Path(root) / path).is_file():
                path = "index.html"  # client-side routes
            return send_from_directory(root, path)

    return app
