from __future__ import annotations

import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
import requests
from werkzeug.serving import make_server

from forge import service
from forge.api.app import create_app
from forge.config import Settings
from forge.service import Context


@pytest.fixture
def ctx(tmp_path) -> Context:
    return Context.create(
        Settings(
            database_url=f"sqlite:///{tmp_path}/forge.db",
            data_dir=tmp_path,
            secret="test-secret",
            local_nodes={"n0": 4},
            cycle_seconds=0.05,
            preemption_grace_seconds=10.0,
            min_runtime=0.5,
        )
    )


@dataclass
class Api:
    url: str
    ctx: Context
    keys: dict[str, str]

    def call(self, method: str, path: str, user: str = "alice", **kw: Any) -> requests.Response:
        headers = {"Authorization": f"Bearer {self.keys[user]}"} if user else {}
        return requests.request(method, self.url + path, headers=headers, timeout=10, **kw)


@pytest.fixture
def api(ctx: Context) -> Iterator[Api]:
    """A real HTTP server, with two teams: alice (vision, admin) and bob (planning)."""
    keys = {}
    with ctx.sessions() as s:
        vision = service.create_team(s, "vision", 4)
        planning = service.create_team(s, "planning", 0)
        _, keys["alice"] = service.create_user(s, ctx.settings.secret, "alice", vision, "admin")
        _, keys["carol"] = service.create_user(s, ctx.settings.secret, "carol", vision)
        _, keys["bob"] = service.create_user(s, ctx.settings.secret, "bob", planning)
        s.commit()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    object.__setattr__(ctx.settings, "api_url", url)
    server = make_server("127.0.0.1", port, create_app(ctx), threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield Api(url, ctx, keys)
    finally:
        server.shutdown()
