"""Credentials.

Three kinds, all derived from or checked against the server secret:

- API keys identify a user. Only an HMAC of the key is stored.
- Job tokens are handed to training processes. A token is scoped to one job
  and can only report metrics, heartbeats and checkpoints for it, so a leaked
  pod environment does not expose a user's key.
- Signed URLs grant one upload or download for a short time.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time

KEY_PREFIX = "fk_"
JOB_TOKEN_PREFIX = "jt_"


def _mac(secret: str, message: str) -> str:
    return hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()


def new_api_key() -> str:
    return KEY_PREFIX + secrets.token_hex(16)


def hash_key(secret: str, key: str) -> str:
    return _mac(secret, "key:" + key)


def job_token(secret: str, job_id: str) -> str:
    return f"{JOB_TOKEN_PREFIX}{job_id}.{_mac(secret, 'job:' + job_id)}"


def verify_job_token(secret: str, token: str) -> str | None:
    """Returns the job id the token is scoped to, or None."""
    if not token.startswith(JOB_TOKEN_PREFIX):
        return None
    job_id, _, mac = token.removeprefix(JOB_TOKEN_PREFIX).partition(".")
    if job_id and hmac.compare_digest(mac, _mac(secret, "job:" + job_id)):
        return job_id
    return None


def sign(secret: str, method: str, path: str, ttl: float = 900.0, now: float | None = None) -> str:
    expires = int((now if now is not None else time.time()) + ttl)
    return f"{expires}.{_mac(secret, f'{method}:{path}:{expires}')}"


def verify_signature(
    secret: str, method: str, path: str, sig: str, now: float | None = None
) -> bool:
    expires, _, mac = sig.partition(".")
    if not expires.isdigit() or int(expires) < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(mac, _mac(secret, f"{method}:{path}:{expires}"))
