from __future__ import annotations

import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _resolve_path(raw_value: str | None, default: Path) -> Path:
    path = Path(raw_value).expanduser() if raw_value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _source_python(source_root: Path, configured: str | None) -> Path:
    if configured:
        return _resolve_path(configured, source_root / "venv" / "Scripts" / "python.exe")
    windows_python = source_root / "venv" / "Scripts" / "python.exe"
    posix_python = source_root / "venv" / "bin" / "python"
    if windows_python.exists():
        return windows_python
    if posix_python.exists():
        return posix_python
    return Path(sys.executable)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    source_root: Path
    source_python: Path
    catalog_path: Path
    job_root: Path
    api_host: str
    api_port: int
    allowed_origins: tuple[str, ...]
    worker_threads: int
    ml_worker_threads: int = 1
    auth_enabled: bool = False
    auth_username: str = "admin"
    auth_password: str = "battryml"
    auth_secret: str = ""
    auth_session_hours: int = 12

    @classmethod
    def from_env(cls, env_file: str | Path | None = None) -> Settings:
        candidate = Path(env_file) if env_file else PROJECT_ROOT / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)

        source_root = _resolve_path(
            os.getenv("BATTRYML_SOURCE_ROOT"),
            PROJECT_ROOT / "brain",
        )
        origins = tuple(
            value.strip()
            for value in os.getenv(
                "BATTRYML_ALLOWED_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173",
            ).split(",")
            if value.strip()
        )
        return cls(
            project_root=PROJECT_ROOT,
            source_root=source_root,
            source_python=_source_python(source_root, os.getenv("BATTRYML_SOURCE_PYTHON")),
            catalog_path=_resolve_path(
                os.getenv("BATTRYML_CATALOG_PATH"),
                PROJECT_ROOT / "var" / "battryml.duckdb",
            ),
            job_root=_resolve_path(
                os.getenv("BATTRYML_JOB_ROOT"),
                PROJECT_ROOT / "var" / "jobs",
            ),
            api_host=os.getenv("BATTRYML_API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("BATTRYML_API_PORT", "8000")),
            allowed_origins=origins,
            worker_threads=max(1, int(os.getenv("BATTRYML_WORKER_THREADS", "2"))),
            ml_worker_threads=max(
                1,
                int(os.getenv("BATTRYML_ML_WORKER_THREADS", "1")),
            ),
            auth_enabled=_env_bool("BATTRYML_AUTH_ENABLED", False),
            auth_username=os.getenv("BATTRYML_AUTH_USERNAME", "admin").strip() or "admin",
            auth_password=os.getenv("BATTRYML_AUTH_PASSWORD", "battryml"),
            auth_secret=os.getenv("BATTRYML_AUTH_SECRET", ""),
            auth_session_hours=max(
                1,
                int(os.getenv("BATTRYML_AUTH_SESSION_HOURS", "12")),
            ),
        )

    def ensure_runtime_dirs(self) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        self.job_root.mkdir(parents=True, exist_ok=True)

    def validate_source(self) -> list[str]:
        problems: list[str] = []
        if not self.source_root.exists():
            problems.append(f"Source project does not exist: {self.source_root}")
        if not self.source_python.exists():
            problems.append(f"Source Python does not exist: {self.source_python}")
        if not (self.source_root / "src" / "capacity_ml.py").exists():
            problems.append("Source project does not expose src.capacity_ml")
        return problems

    def source_database_status(self, *, timeout_seconds: float = 0.75) -> dict[str, object]:
        values = dotenv_values(self.source_root / ".env")
        host = str(values.get("DB_HOST") or "").strip()
        raw_port = str(values.get("DB_PORT") or "5432").strip()
        try:
            port = int(raw_port)
        except ValueError:
            return {
                "configured": False,
                "reachable": False,
                "host": host or None,
                "port": None,
                "message": f"Invalid DB_PORT: {raw_port}",
            }
        if not host:
            return {
                "configured": False,
                "reachable": False,
                "host": None,
                "port": port,
                "message": "DB_HOST is not configured in the source project .env file.",
            }
        try:
            with socket.create_connection((host, port), timeout=timeout_seconds):
                pass
        except OSError:
            return {
                "configured": True,
                "reachable": False,
                "host": host,
                "port": port,
                "message": f"Database is unreachable at {host}:{port}.",
            }
        return {
            "configured": True,
            "reachable": True,
            "host": host,
            "port": port,
            "message": f"Database is reachable at {host}:{port}.",
        }
