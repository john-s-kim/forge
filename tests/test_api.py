from __future__ import annotations

import time

import requests

from forge import auth
from forge.api.app import RateLimiter

SPEC = {"command": ["true"]}


def test_requests_without_a_valid_key_are_rejected(api):
    assert api.call("GET", "/v1/jobs", user=None).status_code == 401
    r = requests.get(api.url + "/v1/jobs", headers={"Authorization": "Bearer fk_nope"}, timeout=5)
    assert r.status_code == 401


def test_keys_are_not_stored(api):
    from forge import db

    with api.ctx.sessions() as s:
        hashes = [u.key_hash for u in s.query(db.User)]
    assert all(k not in hashes for k in api.keys.values())


def test_jobs_are_invisible_to_other_teams(api):
    jid = api.call("POST", "/v1/jobs", user="carol", json=SPEC).json()["id"]
    assert api.call("GET", f"/v1/jobs/{jid}", user="bob").status_code == 404
    assert api.call("POST", f"/v1/jobs/{jid}/cancel", user="bob").status_code == 404
    assert api.call("GET", "/v1/jobs", user="bob").json()["jobs"] == []
    # Teammates and admins can see it.
    assert api.call("GET", f"/v1/jobs/{jid}", user="alice").status_code == 200


def test_only_admins_manage_teams_and_quotas(api):
    body = {"gpu_quota": 99}
    assert api.call("PUT", "/v1/teams/planning/quota", user="bob", json=body).status_code == 403
    assert api.call("PUT", "/v1/teams/planning/quota", user="alice", json=body).status_code == 200
    r = api.call("POST", "/v1/users", user="alice", json={"name": "dave", "team": "planning"})
    assert r.status_code == 201 and r.json()["api_key"].startswith("fk_")


def test_submit_validates_the_spec(api):
    bad = [
        {},
        {"command": "rm -rf /"},
        {"command": ["x"], "workers": 0},
        {"command": ["x"], "gpus_per_worker": True},
        {"command": ["x"], "priority": "urgent"},
        {"command": ["x"], "time_limit": -1},
        {"command": ["x"], "env": {"FORGE_JOB_TOKEN": "stolen"}},
    ]
    for spec in bad:
        assert api.call("POST", "/v1/jobs", json=spec).status_code == 400, spec


def test_job_token_is_scoped_to_one_job(api):
    a = api.call("POST", "/v1/jobs", json=SPEC).json()["id"]
    b = api.call("POST", "/v1/jobs", json=SPEC).json()["id"]
    token = {"Authorization": f"Bearer {auth.job_token(api.ctx.settings.secret, a)}"}
    points = {"points": [{"name": "loss", "step": 0, "value": 1.0}]}

    assert requests.post(f"{api.url}/v1/jobs/{a}/metrics", json=points, headers=token, timeout=5).ok
    other = requests.post(f"{api.url}/v1/jobs/{b}/metrics", json=points, headers=token, timeout=5)
    assert other.status_code == 404
    # A job token cannot do what a user can.
    assert requests.get(f"{api.url}/v1/jobs", headers=token, timeout=5).status_code == 401
    assert (
        requests.post(f"{api.url}/v1/jobs/{a}/cancel", headers=token, timeout=5).status_code == 401
    )


def test_non_finite_metrics_are_recorded_as_null(api):
    jid = api.call("POST", "/v1/jobs", json=SPEC).json()["id"]
    points = [{"name": "loss", "step": 0, "value": None}, {"name": "loss", "step": 1, "value": 0.5}]
    assert api.call("POST", f"/v1/jobs/{jid}/metrics", json={"points": points}).json() == {
        "accepted": 2
    }
    got = api.call("GET", f"/v1/jobs/{jid}/metrics").json()["points"]
    assert [p["value"] for p in got] == [None, 0.5]


def test_signed_urls_expire_and_cannot_be_reused_for_another_path():
    sig = auth.sign("s", "PUT", "/v1/blobs/a/1", ttl=60, now=1000)
    assert auth.verify_signature("s", "PUT", "/v1/blobs/a/1", sig, now=1030)
    assert not auth.verify_signature("s", "PUT", "/v1/blobs/a/1", sig, now=1061)
    assert not auth.verify_signature("s", "PUT", "/v1/blobs/b/1", sig, now=1030)
    assert not auth.verify_signature("s", "GET", "/v1/blobs/a/1", sig, now=1030)
    assert not auth.verify_signature("other", "PUT", "/v1/blobs/a/1", sig, now=1030)


def test_blob_endpoint_requires_a_signature(api):
    assert requests.put(f"{api.url}/v1/blobs/abc/1", data=b"x", timeout=5).status_code == 403
    sig = auth.sign(api.ctx.settings.secret, "PUT", "/v1/blobs/abc/1", now=time.time() - 3600)
    r = requests.put(f"{api.url}/v1/blobs/abc/1", params={"sig": sig}, data=b"x", timeout=5)
    assert r.status_code == 403


def test_rate_limiter_refills_over_time():
    rl = RateLimiter(rate=1.0, burst=2.0)
    assert rl.allow("u", now=0) and rl.allow("u", now=0)
    assert not rl.allow("u", now=0.5)
    assert rl.allow("u", now=1.6)
    assert rl.allow("someone-else", now=0.5)


def test_cancel_is_rejected_once_a_job_is_finished(api):
    jid = api.call("POST", "/v1/jobs", json=SPEC).json()["id"]
    assert api.call("POST", f"/v1/jobs/{jid}/cancel").status_code == 200
    assert api.call("POST", f"/v1/jobs/{jid}/cancel").status_code == 409
