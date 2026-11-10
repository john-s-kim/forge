"""Kubernetes backend: one pod per worker, bound directly to the node the
scheduler chose (`spec.nodeName`), so kube-scheduler is not involved.

Not yet exercised against a live cluster; see the README status table.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any

from forge.controller.backend import Launch, LogSink, Status

LABEL_ATTEMPT = "forge.dev/attempt"
LABEL_JOB = "forge.dev/job"
PULL_ERRORS = {"ErrImagePull", "ImagePullBackOff", "InvalidImageName"}


def namespace_for(team: str) -> str:
    return f"forge-{team}"


class KubeBackend:
    def __init__(self, gpu_resource: str, sink: LogSink) -> None:
        from kubernetes import client, config  # type: ignore[import-not-found]

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self._client = client
        self._core = client.CoreV1Api()
        self._gpu = gpu_resource
        self._sink = sink
        self._launches: dict[int, Launch] = {}
        self._following: set[str] = set()

    def nodes(self) -> dict[str, int]:
        out = {}
        for n in self._core.list_node().items:
            gpus = int((n.status.allocatable or {}).get(self._gpu, 0))
            ready = any(c.type == "Ready" and c.status == "True" for c in n.status.conditions or [])
            if gpus > 0 and ready and not n.spec.unschedulable:
                out[n.metadata.name] = gpus
        return out

    def start(self, launch: Launch) -> None:
        c = self._client
        self._launches[launch.attempt_id] = launch
        for i, node in enumerate(launch.nodes):
            env = {**launch.env, "FORGE_WORKER_INDEX": str(i), "FORGE_NODE": node}
            pod = c.V1Pod(
                metadata=c.V1ObjectMeta(
                    name=f"forge-{launch.job_id}-{launch.number}-{i}",
                    labels={LABEL_ATTEMPT: str(launch.attempt_id), LABEL_JOB: launch.job_id},
                ),
                spec=c.V1PodSpec(
                    node_name=node,
                    restart_policy="Never",
                    automount_service_account_token=False,
                    termination_grace_period_seconds=int(launch.grace_seconds),
                    security_context=c.V1PodSecurityContext(
                        run_as_non_root=True,
                        run_as_user=1000,
                        seccomp_profile=c.V1SeccompProfile(type="RuntimeDefault"),
                    ),
                    # GPU nodes are usually tainted; the scheduler has already chosen the node.
                    tolerations=[c.V1Toleration(operator="Exists")],
                    containers=[
                        c.V1Container(
                            name="train",
                            image=launch.image,
                            command=launch.command,
                            env=[c.V1EnvVar(name=k, value=v) for k, v in env.items()],
                            resources=c.V1ResourceRequirements(
                                limits={self._gpu: str(launch.gpus_per_worker)}
                            ),
                            security_context=c.V1SecurityContext(
                                allow_privilege_escalation=False,
                                capabilities=c.V1Capabilities(drop=["ALL"]),
                            ),
                        )
                    ],
                ),
            )
            self._core.create_namespaced_pod(namespace_for(launch.team), pod)

    def adopt(self, launch: Launch) -> None:
        self._launches[launch.attempt_id] = launch

    def _pods(self, launch: Launch) -> list[Any]:
        return self._core.list_namespaced_pod(
            namespace_for(launch.team), label_selector=f"{LABEL_ATTEMPT}={launch.attempt_id}"
        ).items

    def status(self, attempt_id: int) -> Status:
        launch = self._launches.get(attempt_id)
        if launch is None:
            return Status("gone", reason="attempt lost")
        pods = self._pods(launch)
        if len(pods) < len(launch.nodes):
            return Status("gone", reason="pod deleted")
        phases = [p.status.phase for p in pods]
        events: list[str] = []
        for p in pods:
            for cs in p.status.container_statuses or []:
                waiting = cs.state.waiting
                if waiting and waiting.reason in PULL_ERRORS:
                    events.append(f"{p.metadata.name}: {waiting.reason}: {waiting.message}")
                    return Status("failed", reason="image_pull", events=events)
                term = cs.state.terminated
                if term and term.exit_code != 0:
                    events.append(f"{p.metadata.name}: {term.reason} (exit {term.exit_code})")
                    return Status(
                        "failed", exit_code=term.exit_code, reason=term.reason, events=events
                    )
            if p.status.phase == "Running" and p.metadata.name not in self._following:
                self._following.add(p.metadata.name)
                threading.Thread(
                    target=self._follow, args=(launch, p.metadata.name), daemon=True
                ).start()
        if any(ph == "Failed" for ph in phases):
            reasons = [p.status.reason or "Failed" for p in pods if p.status.phase == "Failed"]
            return Status("failed", reason=reasons[0], events=reasons)
        if all(ph == "Succeeded" for ph in phases):
            return Status("succeeded", exit_code=0)
        if all(ph in ("Running", "Succeeded") for ph in phases):
            return Status("running")
        return Status("starting")

    def _follow(self, launch: Launch, pod: str) -> None:
        with contextlib.suppress(Exception):
            resp = self._core.read_namespaced_pod_log(
                pod, namespace_for(launch.team), follow=True, _preload_content=False
            )
            for raw in resp.stream():
                for line in raw.decode(errors="replace").splitlines():
                    self._sink(launch.job_id, launch.team, line)

    def _delete(self, attempt_id: int, grace: int | None) -> None:
        launch = self._launches.get(attempt_id)
        if launch is None:
            return
        for p in self._pods(launch):
            with contextlib.suppress(self._client.ApiException):
                self._core.delete_namespaced_pod(
                    p.metadata.name, namespace_for(launch.team), grace_period_seconds=grace
                )

    def stop(self, attempt_id: int) -> None:
        self._delete(attempt_id, None)  # the pod's own termination grace period applies

    def kill(self, attempt_id: int) -> None:
        self._delete(attempt_id, 0)

    def forget(self, attempt_id: int) -> None:
        self._delete(attempt_id, 0)
        self._launches.pop(attempt_id, None)
