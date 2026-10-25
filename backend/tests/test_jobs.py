from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from battryml.catalog.database import CatalogDatabase
from battryml.catalog.repository import CatalogRepository
from battryml.catalog.scanner import LegacyRunScanner
from battryml.config import Settings
from battryml.data.devices import DeviceLibraryService
from battryml.jobs import PipelineJobRunner, _process_is_alive
from battryml.plugins.base import ParameterDefinition, PipelinePlugin
from battryml.plugins.registry import PluginRegistry


class SlowTestPlugin(PipelinePlugin):
    key = "test.slow.v1"
    name = "Slow test"
    description = "Exercise durable worker recovery."
    output_family = "test"
    produces_analysis_run = False
    parameters = (ParameterDefinition("serial", "Serial", "text", required=True),)

    def build_command(self, settings: Settings, values: dict) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "import time; time.sleep(2); print('worker completed', flush=True)",
        )


class SlowExtractionPlugin(SlowTestPlugin):
    key = "test.extraction.v1"
    workflow_stage = "extraction"

    def build_command(self, settings: Settings, values: dict) -> tuple[str, ...]:
        return (sys.executable, "-c", "import time; time.sleep(4)")


class FastMlPlugin(SlowTestPlugin):
    key = "test.ml.v1"
    workflow_stage = "events"

    def build_command(self, settings: Settings, values: dict) -> tuple[str, ...]:
        return (sys.executable, "-c", "print('ml completed', flush=True)")


class FailingStdoutPlugin(SlowTestPlugin):
    key = "test.failing-stdout.v1"

    def build_command(self, settings: Settings, values: dict) -> tuple[str, ...]:
        return (
            sys.executable,
            "-c",
            "print('database connection timed out', flush=True); raise SystemExit(1)",
        )


class DatabasePlugin(SlowTestPlugin):
    key = "test.database.v1"
    requires_database = True


def _runner(
    settings: Settings,
    repository: CatalogRepository,
    *plugins: PipelinePlugin,
) -> PipelineJobRunner:
    return PipelineJobRunner(
        settings,
        repository,
        LegacyRunScanner(settings, repository),
        PluginRegistry(list(plugins)),
        DeviceLibraryService(settings, repository),
    )


def _wait_for_status(
    repository: CatalogRepository,
    job_id: str,
    expected: str,
    timeout: float = 10.0,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = repository.get_job(job_id)
        if job and job["status"] == expected:
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not reach {expected}: {repository.get_job(job_id)}")


def _wait_for_worker(repository: CatalogRepository, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = repository.get_job(job_id)
        if job and job["status"] == "running" and job["worker_pid"] is not None:
            return job
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not launch a worker: {repository.get_job(job_id)}")


def test_running_worker_is_recovered_after_runner_restart(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    plugin = SlowTestPlugin()
    first_runner = _runner(platform_settings, repository, plugin)
    second_runner = None
    try:
        submitted = first_runner.submit(plugin.key, {"serial": "TEST001"})
        running = _wait_for_worker(repository, submitted["job_id"])
        worker_pid = int(running["worker_pid"])
        assert _process_is_alive(worker_pid)

        first_runner.shutdown()
        second_runner = _runner(platform_settings, repository, plugin)
        second_runner.recover_incomplete_jobs()

        completed = _wait_for_status(repository, submitted["job_id"], "completed")
        assert completed["return_code"] == 0
        assert completed["result_path"]
        assert "worker completed" in Path(completed["stdout_path"]).read_text(encoding="utf-8")
    finally:
        first_runner.shutdown()
        if second_runner is not None:
            second_runner.shutdown()


def test_running_job_without_worker_or_result_is_abandoned(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    plugin = SlowTestPlugin()
    repository.create_job(
        {
            "job_id": "missing-worker",
            "pipeline_key": plugin.key,
            "serial": "TEST001",
            "status": "running",
            "request_json": '{"serial": "TEST001"}',
            "created_at": datetime.now(UTC),
        }
    )

    runner = _runner(platform_settings, repository, plugin)
    try:
        runner.recover_incomplete_jobs()
        job = repository.get_job("missing-worker")
        assert job is not None
        assert job["status"] == "abandoned"
        assert "without writing a result" in job["error_message"]
    finally:
        runner.shutdown()


def test_ml_job_runs_while_extraction_pool_is_busy(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    extraction = SlowExtractionPlugin()
    ml = FastMlPlugin()
    runner = _runner(platform_settings, repository, extraction, ml)
    try:
        extraction_job = runner.submit(extraction.key, {"serial": "TEST001"})
        _wait_for_worker(repository, extraction_job["job_id"])

        ml_job = runner.submit(ml.key, {"serial": "TEST002"})
        completed = _wait_for_status(
            repository,
            ml_job["job_id"],
            "completed",
            timeout=2.5,
        )

        assert completed["return_code"] == 0
        assert repository.get_job(extraction_job["job_id"])["status"] == "running"
    finally:
        runner.shutdown()


def test_failed_job_uses_stdout_when_stderr_is_empty(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    plugin = FailingStdoutPlugin()
    runner = _runner(platform_settings, repository, plugin)
    try:
        submitted = runner.submit(plugin.key, {"serial": "TEST001"})
        failed = _wait_for_status(repository, submitted["job_id"], "failed")

        assert "database connection timed out" in failed["error_message"]
    finally:
        runner.shutdown()


def test_database_job_is_not_queued_when_source_database_is_unconfigured(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    plugin = DatabasePlugin()
    runner = _runner(platform_settings, repository, plugin)
    try:
        try:
            runner.submit(plugin.key, {"serial": "TEST001"})
        except ValueError as exc:
            assert "DB_HOST is not configured" in str(exc)
        else:
            raise AssertionError("Database-backed job was queued without a database")

        assert repository.list_unfinished_jobs() == []
    finally:
        runner.shutdown()
