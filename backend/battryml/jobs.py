from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from battryml.catalog.repository import CatalogRepository
from battryml.catalog.scanner import LegacyRunScanner
from battryml.config import Settings
from battryml.data.devices import DeviceLibraryService
from battryml.plugins.base import PipelinePlugin
from battryml.plugins.registry import PluginRegistry

RESULT_FILE_NAME = "result.json"
WORKER_POLL_SECONDS = 0.5
RESULT_GRACE_SECONDS = 5.0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _process_is_alive(process_id: int | None) -> bool:
    if process_id is None or process_id <= 0:
        return False
    if os.name == "nt":
        return _windows_process_is_alive(process_id)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _windows_process_is_alive(process_id: int) -> bool:
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    error_access_denied = 5
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, process_id)
    if not handle:
        return ctypes.get_last_error() == error_access_denied
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


class PipelineJobRunner:
    def __init__(
        self,
        settings: Settings,
        repository: CatalogRepository,
        scanner: LegacyRunScanner,
        registry: PluginRegistry,
        devices: DeviceLibraryService,
    ):
        self.settings = settings
        self.repository = repository
        self.scanner = scanner
        self.registry = registry
        self.devices = devices
        self._stop_event = threading.Event()
        self._submit_lock = threading.Lock()
        self.extraction_executor = ThreadPoolExecutor(
            max_workers=settings.worker_threads,
            thread_name_prefix="battryml-extraction",
        )
        self.ml_executor = ThreadPoolExecutor(
            max_workers=settings.ml_worker_threads,
            thread_name_prefix="battryml-ml",
        )

    def submit(self, pipeline_key: str, parameters: dict[str, Any]) -> dict[str, Any]:
        plugin = self.registry.get(pipeline_key)
        normalized = plugin.validate(parameters)
        serial = str(normalized["serial"])
        with self._submit_lock:
            active_job = next(
                (
                    item
                    for item in self.repository.list_unfinished_jobs()
                    if str(item["serial"]) == serial
                ),
                None,
            )
            if active_job is not None:
                raise ValueError(
                    f"Battery {serial} already has an active pipeline task "
                    f"({active_job['pipeline_key']}). Wait for it to finish before "
                    "starting the next stage."
                )

            if plugin.needs_database(normalized):
                database = self.settings.source_database_status()
                if not database["reachable"]:
                    raise ValueError(str(database["message"]))

            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "pipeline_key": pipeline_key,
                "serial": serial,
                "status": "queued",
                "request_json": json.dumps(normalized, sort_keys=True),
                "created_at": _utc_now(),
            }
            self.repository.create_job(job)
        self._executor_for(plugin).submit(
            self._launch_and_monitor,
            job_id,
            plugin,
            normalized,
        )
        return self.repository.get_job(job_id) or job

    def recover_incomplete_jobs(self) -> None:
        for job in self.repository.list_unfinished_jobs():
            try:
                plugin = self.registry.get(str(job["pipeline_key"]))
                parameters = json.loads(str(job["request_json"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self._mark_abandoned(
                    str(job["job_id"]),
                    f"Could not recover job metadata after API restart: {exc}",
                )
                continue

            if job["status"] == "queued":
                self._executor_for(plugin).submit(
                    self._launch_and_monitor,
                    str(job["job_id"]),
                    plugin,
                    parameters,
                )
                continue

            result_path = self._result_path(job)
            if result_path.exists():
                self._executor_for(plugin).submit(
                    self._finalize_from_result,
                    str(job["job_id"]),
                    plugin,
                    parameters,
                    result_path,
                )
                continue

            worker_pid = self._worker_pid(job)
            if _process_is_alive(worker_pid):
                self._executor_for(plugin).submit(
                    self._monitor_recovered_worker,
                    str(job["job_id"]),
                    plugin,
                    parameters,
                    worker_pid,
                    result_path,
                )
            else:
                self._mark_abandoned(
                    str(job["job_id"]),
                    "The job worker stopped during an API restart without writing a result.",
                )

    def _launch_and_monitor(
        self,
        job_id: str,
        plugin: PipelinePlugin,
        parameters: dict[str, Any],
    ) -> None:
        if self._stop_event.is_set():
            return

        job_dir = self.settings.job_root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job_dir / "stdout.log"
        stderr_path = job_dir / "stderr.log"
        result_path = job_dir / RESULT_FILE_NAME

        try:
            command = plugin.build_command(self.settings, parameters)
            self.repository.update_job(
                job_id,
                status="running",
                started_at=_utc_now(),
                completed_at=None,
                return_code=None,
                command_json=json.dumps(command),
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                result_path=str(result_path),
                output_run_id=None,
                error_message=None,
            )
            worker = subprocess.Popen(
                self._worker_command(command, stdout_path, stderr_path, result_path),
                cwd=self.settings.source_root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **self._detached_process_options(),
            )
            self.repository.update_job(job_id, worker_pid=worker.pid)
        except Exception as exc:
            stderr_path.write_text(str(exc), encoding="utf-8")
            self.repository.update_job(
                job_id,
                status="failed",
                completed_at=_utc_now(),
                return_code=-1,
                error_message=str(exc),
            )
            return

        while not self._stop_event.wait(WORKER_POLL_SECONDS):
            if worker.poll() is not None:
                self._finalize_from_result(
                    job_id,
                    plugin,
                    parameters,
                    result_path,
                    worker_return_code=worker.returncode,
                )
                return

    def _executor_for(self, plugin: PipelinePlugin) -> ThreadPoolExecutor:
        return (
            self.extraction_executor
            if plugin.workflow_stage == "extraction"
            else self.ml_executor
        )

    def _monitor_recovered_worker(
        self,
        job_id: str,
        plugin: PipelinePlugin,
        parameters: dict[str, Any],
        worker_pid: int,
        result_path: Path,
    ) -> None:
        while not self._stop_event.is_set():
            if result_path.exists():
                self._finalize_from_result(job_id, plugin, parameters, result_path)
                return
            if not _process_is_alive(worker_pid):
                break
            self._stop_event.wait(WORKER_POLL_SECONDS)

        if self._stop_event.is_set():
            return

        elapsed = 0.0
        while elapsed < RESULT_GRACE_SECONDS and not result_path.exists():
            if self._stop_event.wait(WORKER_POLL_SECONDS):
                return
            elapsed += WORKER_POLL_SECONDS

        if result_path.exists():
            self._finalize_from_result(job_id, plugin, parameters, result_path)
        else:
            self._mark_abandoned(
                job_id,
                "The recovered job worker exited without writing a result.",
            )

    def _finalize_from_result(
        self,
        job_id: str,
        plugin: PipelinePlugin,
        parameters: dict[str, Any],
        result_path: Path,
        *,
        worker_return_code: int | None = None,
    ) -> None:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            return_code = int(result["return_code"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.repository.update_job(
                job_id,
                status="failed",
                completed_at=_utc_now(),
                return_code=worker_return_code if worker_return_code is not None else -1,
                error_message=f"Invalid worker result: {exc}",
            )
            return

        output_run_id = None
        if return_code == 0:
            serial = str(parameters["serial"])
            self.devices.refresh_serial(serial)
            scan_families = plugin.scan_output_families or (
                (plugin.output_family,) if plugin.produces_analysis_run else ()
            )
            for model_family in scan_families:
                self.scanner.scan(serial=serial, model_family=model_family)
            if scan_families:
                latest = self.repository.list_runs(
                    serial=serial,
                    model_family=scan_families[-1],
                    limit=1,
                )
                output_run_id = latest[0]["run_id"] if latest else None

        error_message = None
        if return_code != 0:
            error_message = str(result.get("error_message") or "").strip() or self._log_tail(
                job_id
            )
            if not error_message:
                error_message = f"Command exited with code {return_code}."

        self.repository.update_job(
            job_id,
            status="completed" if return_code == 0 else "failed",
            completed_at=self._completed_at(result),
            return_code=return_code,
            output_run_id=output_run_id,
            error_message=error_message,
        )

    def _worker_command(
        self,
        command: tuple[str, ...],
        stdout_path: Path,
        stderr_path: Path,
        result_path: Path,
    ) -> tuple[str, ...]:
        worker_script = Path(__file__).with_name("job_worker.py")
        return (
            sys.executable,
            str(worker_script),
            "--result-path",
            str(result_path),
            "--stdout-path",
            str(stdout_path),
            "--stderr-path",
            str(stderr_path),
            "--cwd",
            str(self.settings.source_root),
            "--",
            *command,
        )

    @staticmethod
    def _detached_process_options() -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _result_path(self, job: dict[str, Any]) -> Path:
        configured = job.get("result_path")
        return (
            Path(str(configured))
            if configured
            else self.settings.job_root / str(job["job_id"]) / RESULT_FILE_NAME
        )

    @staticmethod
    def _worker_pid(job: dict[str, Any]) -> int | None:
        value = job.get("worker_pid")
        return int(value) if value is not None else None

    @staticmethod
    def _completed_at(result: dict[str, Any]) -> datetime:
        value = result.get("completed_at")
        if value:
            try:
                parsed = datetime.fromisoformat(str(value))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
            except ValueError:
                pass
        return _utc_now()

    def _log_tail(self, job_id: str) -> str:
        job = self.repository.get_job(job_id) or {}
        for field in ("stderr_path", "stdout_path"):
            path_value = job.get(field)
            if not path_value:
                continue
            path = Path(str(path_value))
            if not path.exists():
                continue
            message = path.read_text(encoding="utf-8", errors="replace")[-2000:].strip()
            if message:
                return message
        return ""

    def _mark_abandoned(self, job_id: str, message: str) -> None:
        self.repository.update_job(
            job_id,
            status="abandoned",
            completed_at=_utc_now(),
            error_message=message,
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        self.extraction_executor.shutdown(wait=False, cancel_futures=True)
        self.ml_executor.shutdown(wait=False, cancel_futures=True)
