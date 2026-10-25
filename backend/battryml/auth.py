from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass

from battryml.config import Settings

SESSION_COOKIE = "battryml_session"


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


@dataclass(frozen=True)
class AuthManager:
    enabled: bool
    username: str
    password: str
    secret: bytes
    session_seconds: int

    @classmethod
    def from_settings(cls, settings: Settings) -> AuthManager:
        configured_secret = settings.auth_secret.strip()
        secret = configured_secret or hashlib.sha256(
            f"{settings.project_root}:{settings.auth_username}:{settings.auth_password}".encode()
        ).hexdigest()
        return cls(
            enabled=settings.auth_enabled,
            username=settings.auth_username,
            password=settings.auth_password,
            secret=secret.encode("utf-8"),
            session_seconds=settings.auth_session_hours * 3600,
        )

    def authenticate(self, username: str, password: str) -> bool:
        if not self.enabled:
            return True
        return hmac.compare_digest(username, self.username) and hmac.compare_digest(
            password,
            self.password,
        )

    def issue(self, username: str) -> str:
        payload = json.dumps(
            {"sub": username, "exp": int(time.time()) + self.session_seconds},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        encoded_payload = _encode(payload)
        signature = hmac.new(self.secret, encoded_payload.encode("ascii"), hashlib.sha256)
        return f"{encoded_payload}.{_encode(signature.digest())}"

    def verify(self, token: str | None) -> str | None:
        if not self.enabled:
            return self.username
        if not token or "." not in token:
            return None
        encoded_payload, encoded_signature = token.split(".", 1)
        expected = hmac.new(self.secret, encoded_payload.encode("ascii"), hashlib.sha256).digest()
        try:
            supplied = _decode(encoded_signature)
            payload = json.loads(_decode(encoded_payload))
        except (ValueError, json.JSONDecodeError):
            return None
        if not hmac.compare_digest(expected, supplied):
            return None
        if int(payload.get("exp", 0)) <= int(time.time()):
            return None
        username = str(payload.get("sub", "")).strip()
        return username or None
