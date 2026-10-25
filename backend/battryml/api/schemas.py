from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    pipeline_key: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class LoginRequest(BaseModel):
    username: str
    password: str


class DeviceDisplayNameUpdate(BaseModel):
    display_name: str | None = Field(default=None, max_length=80)


class ScanResult(BaseModel):
    source_root: str
    runs_discovered: int
    failures: list[dict[str, str]]
    scanned_at: Any
