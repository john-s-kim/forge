from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _nodes(spec: str) -> dict[str, int]:
    out = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, gpus = part.partition("=")
        out[name] = int(gpus)
    return out


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///.forge/forge.db"
    redis_url: str | None = None  # unset: in-process live store (single-process dev only)
    nats_url: str | None = None  # unset: in-process event bus (single-process dev only)
    secret: str = "dev-secret-change-me"
    data_dir: Path = Path(".forge")
    api_url: str = "http://127.0.0.1:8080"  # how training processes reach the API
    backend: str = "local"  # "local" (subprocesses) or "kube"
    local_nodes: dict[str, int] = field(default_factory=lambda: {"node-0": 8, "node-1": 8})
    gpu_resource: str = "nvidia.com/gpu"
    cycle_seconds: float = 1.0
    preemption_grace_seconds: float = 30.0
    min_runtime: float = 1800.0  # attempts younger than this are not preempted

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ
        d = cls()
        return cls(
            database_url=e.get("FORGE_DATABASE_URL", d.database_url),
            redis_url=e.get("FORGE_REDIS_URL") or None,
            nats_url=e.get("FORGE_NATS_URL") or None,
            secret=e.get("FORGE_SECRET", d.secret),
            data_dir=Path(e.get("FORGE_DATA_DIR", str(d.data_dir))),
            api_url=e.get("FORGE_API_URL", d.api_url),
            backend=e.get("FORGE_BACKEND", d.backend),
            local_nodes=_nodes(e["FORGE_NODES"]) if e.get("FORGE_NODES") else d.local_nodes,
            gpu_resource=e.get("FORGE_GPU_RESOURCE", d.gpu_resource),
            cycle_seconds=float(e.get("FORGE_CYCLE_SECONDS", d.cycle_seconds)),
            preemption_grace_seconds=float(
                e.get("FORGE_PREEMPTION_GRACE", d.preemption_grace_seconds)
            ),
            min_runtime=float(e.get("FORGE_MIN_RUNTIME", d.min_runtime)),
        )
