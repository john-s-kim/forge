from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any

import click
import requests

# ---- client helpers --------------------------------------------------------


class Client:
    def __init__(self) -> None:
        self.url = os.environ.get("FORGE_URL", "http://127.0.0.1:8080").rstrip("/")
        key = os.environ.get("FORGE_API_KEY")
        if not key:
            raise click.ClickException("set FORGE_API_KEY (see `forge admin bootstrap`)")
        self.http = requests.Session()
        self.http.headers["Authorization"] = f"Bearer {key}"

    def call(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self.http.request(method, self.url + path, timeout=30, **kw)
        except requests.ConnectionError as e:
            raise click.ClickException(f"cannot reach {self.url}: is the API running?") from e
        if r.status_code >= 400:
            try:
                message = r.json().get("error", r.text)
            except ValueError:
                message = r.text
            raise click.ClickException(f"{r.status_code}: {message}")
        return r.json()


def _age(ts: float | None) -> str:
    if ts is None:
        return "-"
    s = int(time.time() - ts)
    return f"{s}s" if s < 90 else f"{s // 60}m" if s < 5400 else f"{s // 3600}h"


def _table(rows: list[list[str]], header: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    for r in [header, *rows]:
        click.echo("  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)).rstrip())


@click.group()
def main() -> None:
    """Forge: submit and manage GPU training jobs."""


# ---- user commands ---------------------------------------------------------


@main.command(context_settings={"ignore_unknown_options": True})
@click.option("--gpus", default=1, show_default=True, help="GPUs per worker.")
@click.option("--workers", default=1, show_default=True, help="Workers, placed all-or-nothing.")
@click.option("--priority", type=click.Choice(["low", "normal", "high"]), default="normal")
@click.option("--time-limit", type=float, help="Seconds. Enables backfill and is enforced.")
@click.option("--image", help="Container image (Kubernetes backend).")
@click.option("--name")
@click.option("-e", "--env", multiple=True, metavar="KEY=VALUE")
@click.argument("command", nargs=-1, required=True, type=click.UNPROCESSED)
def submit(
    gpus: int,
    workers: int,
    priority: str,
    time_limit: float | None,
    image: str | None,
    name: str | None,
    env: tuple[str, ...],
    command: tuple[str, ...],
) -> None:
    """Submit a job:  forge submit --gpus 4 -- python train.py --lr 0.1"""
    spec = {
        "command": list(command),
        "gpus_per_worker": gpus,
        "workers": workers,
        "priority": priority,
        "time_limit": time_limit,
        "image": image,
        "name": name,
        "env": dict(e.split("=", 1) for e in env),
    }
    job = Client().call("POST", "/v1/jobs", json=spec)
    click.echo(job["id"])


@main.command("ls")
@click.option("--state", help="Comma-separated states, e.g. queued,running.")
def list_jobs(state: str | None) -> None:
    """List jobs."""
    jobs = Client().call("GET", "/v1/jobs", params={"state": state} if state else None)["jobs"]
    _table(
        [
            [
                j["id"],
                j["name"][:24],
                j["team"],
                j["state"],
                j["priority"],
                str(j["gpus"]),
                _age(j["submitted_at"]),
                j["wait_reason"] or "",
            ]
            for j in jobs
        ],
        ["ID", "NAME", "TEAM", "STATE", "PRIO", "GPUS", "AGE", "WAITING ON"],
    )


@main.command()
@click.argument("job_id")
def status(job_id: str) -> None:
    """Show one job, including why it is not running."""
    j = Client().call("GET", f"/v1/jobs/{job_id}")
    click.echo(f"{j['id']}  {j['name']}  {j['state']}")
    click.echo(
        f"  team {j['team']}, priority {j['priority']}, {j['workers']} x {j['gpus_per_worker']} GPU"
    )
    if j["state"] == "QUEUED":
        click.echo(f"  position {j['queue_position']} in queue, waiting on: {j['wait_reason']}")
    if j["nodes"]:
        click.echo(
            f"  on {', '.join(j['nodes'])}" + (" (borrowed capacity)" if j["borrowed"] else "")
        )
    for e in j["events"]:
        when = time.strftime("%H:%M:%S", time.localtime(e["at"]))
        click.echo(f"  {when}  {e['from'] or '-':>10} -> {e['to']:<10} {e['reason'] or ''}")
    for c in j["checkpoints"][-3:]:
        click.echo(f"  checkpoint at step {c['step']} ({c['size']} bytes)")


@main.command()
@click.argument("job_id")
@click.option("-f", "--follow", is_flag=True)
@click.option("--tail", default=200, show_default=True)
def logs(job_id: str, follow: bool, tail: int) -> None:
    """Print a job's output."""
    c = Client()
    for line in c.call("GET", f"/v1/jobs/{job_id}/logs", params={"tail": tail})["lines"]:
        click.echo(line)
    if not follow:
        return
    with c.http.get(
        f"{c.url}/v1/stream", params={"job": job_id}, stream=True, timeout=(10, None)
    ) as r:
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            ev = json.loads(raw.removeprefix("data: "))
            if ev["type"] == "job.log":
                click.echo(ev["line"])
            elif ev["type"] == "job.state" and ev["to"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
                click.echo(f"-- {ev['to']} {ev.get('reason') or ''}".rstrip(), err=True)
                return


@main.command()
@click.argument("job_id")
def cancel(job_id: str) -> None:
    """Cancel a job."""
    click.echo(Client().call("POST", f"/v1/jobs/{job_id}/cancel")["state"])


@main.command()
def cluster() -> None:
    """Show GPU occupancy by node and team."""
    c = Client().call("GET", "/v1/cluster")
    for n in c["nodes"]:
        cells = "".join(a["team"][0].upper() * a["gpus"] for a in n["allocations"])
        click.echo(f"{n['name']:<12} [{cells.ljust(n['gpus'], '.')}]")
    click.echo()
    _table(
        [[t["name"], str(t["quota"]), str(t["used"]), str(t["borrowed"])] for t in c["teams"]],
        ["TEAM", "QUOTA", "USED", "BORROWED"],
    )


@main.command()
@click.argument("job_id")
@click.option("--approve", is_flag=True, help="Accept the proposed fix and resubmit.")
def diagnosis(job_id: str, approve: bool) -> None:
    """Show why a job failed, and optionally resubmit it with the proposed fix."""
    c = Client()
    if approve:
        click.echo(c.call("POST", f"/v1/jobs/{job_id}/diagnosis/approve")["id"])
        return
    d = c.call("GET", f"/v1/jobs/{job_id}/diagnosis")
    click.echo(f"{d['category']}\n{d['summary']}")
    for line in d["evidence"]:
        click.echo(f"  > {line}")
    if d["proposed_patch"]:
        click.echo(f"proposed fix: {json.dumps(d['proposed_patch'])}")
        click.echo(f"apply it with: forge diagnosis {job_id} --approve")


# ---- servers ---------------------------------------------------------------


def _context() -> Any:
    from forge.service import Context

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    return Context.create()


def _controller(ctx: Any) -> Any:
    from forge.controller.loop import Controller, make_backend
    from forge.triage.diagnose import diagnose

    def on_failure(job_id: str, attempt_id: int) -> None:
        threading.Thread(target=diagnose, args=(ctx, job_id, attempt_id), daemon=True).start()

    return Controller(ctx, make_backend(ctx), on_failure=on_failure)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8080, show_default=True)
@click.option("--web-dir", default="web/dist", show_default=True)
def dev(host: str, port: int, web_dir: str) -> None:
    """Run the API and the controller in one process, on fake local nodes."""
    from forge.api.app import create_app

    ctx = _context()
    stop = threading.Event()
    threading.Thread(target=_controller(ctx).run_forever, args=(stop,), daemon=True).start()
    try:
        create_app(ctx, web_dir).run(host=host, port=port, threaded=True)
    finally:
        stop.set()


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True)
@click.option("--web-dir", default="web/dist", show_default=True)
def api(host: str, port: int, web_dir: str) -> None:
    """Run the API only (development server; use gunicorn in production)."""
    from forge.api.app import create_app

    create_app(_context(), web_dir).run(host=host, port=port, threaded=True)


@main.command()
@click.option("--metrics-port", default=9100, show_default=True)
def controller(metrics_port: int) -> None:
    """Run the scheduler controller only."""
    from prometheus_client import start_http_server

    start_http_server(metrics_port)
    _controller(_context()).run_forever(threading.Event())


@main.group()
def admin() -> None:
    """Operations that talk to the database directly."""


@admin.command()
@click.option("--team", default="vision", show_default=True)
@click.option("--quota", default=8, show_default=True)
@click.option("--user", default="admin", show_default=True)
def bootstrap(team: str, quota: int, user: str) -> None:
    """Create a first team and admin user. Prints the API key once."""
    from sqlalchemy import select

    from forge import db, service

    ctx = _context()
    with ctx.sessions() as s:
        t = s.scalar(select(db.Team).where(db.Team.name == team)) or service.create_team(
            s, team, quota
        )
        if s.scalar(select(db.User).where(db.User.name == user)):
            raise click.ClickException(f"user {user!r} already exists")
        _, key = service.create_user(s, ctx.settings.secret, user, t, role="admin")
        s.commit()
    click.echo(key)


@main.command(context_settings={"ignore_unknown_options": True})
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def bench(args: tuple[str, ...]) -> None:
    """Compare scheduling policies in the simulator."""
    from forge.scheduler import bench as b

    sys.argv = ["forge bench", *args]
    b.main()


if __name__ == "__main__":
    main()
