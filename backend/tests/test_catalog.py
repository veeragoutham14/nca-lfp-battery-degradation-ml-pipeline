from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from battryml.battery_profiles import battery_profile
from battryml.catalog.database import CatalogDatabase
from battryml.catalog.repository import CatalogRepository
from battryml.catalog.scanner import LegacyRunScanner
from battryml.config import Settings

FIELD_NOMINAL_AH = battery_profile("100000001").nominal_capacity_ah


def test_scanner_registers_run_datasets_and_battery(platform_settings: Settings) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    result = LegacyRunScanner(platform_settings, repository).scan()

    assert result["runs_discovered"] == 1
    assert result["failures"] == []
    assert repository.counts() == {
        "batteries": 1,
        "runs": 1,
        "artifacts": 4,
        "datasets": 3,
        "jobs": 0,
    }
    battery = repository.list_batteries()[0]
    assert battery["serial"] == "TEST001"
    run = repository.list_runs(serial="TEST001")[0]
    assert run["loss_slope_per_1000ah"] == 0.08
    assert run["validation_mae_soh_pct"] == 0.4


def test_scan_is_idempotent(platform_settings: Settings) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    scanner = LegacyRunScanner(platform_settings, repository)
    scanner.scan()
    second_scan = scanner.scan()
    assert repository.counts()["runs"] == 1
    assert second_scan["runs_discovered"] == 0
    assert second_scan["runs_examined"] == 1


def test_scanner_registers_bms_soh_recalibration_evidence_run(
    platform_settings: Settings,
) -> None:
    run_dir = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "bms_soh_recalibration"
        / "serial=TEST_RECAL"
        / "recalibration_run=20260807T120000Z"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "model_family": "bms_soh_recalibration_evidence",
                "serial": "TEST_RECAL",
                "nominal_capacity_ah": FIELD_NOMINAL_AH,
                "observed_start": "2026-08-01T00:00:00+00:00",
                "observed_end": "2026-08-02T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    timestamp = datetime(2026, 8, 1, tzinfo=UTC)
    pq.write_table(
        pa.table({"day_utc": [timestamp], "measured_soh_pct": [98.0]}),
        run_dir / "soh_daily_timeline.parquet",
    )
    pq.write_table(
        pa.table({"event_time_utc": [timestamp], "event_type": ["step_drop"]}),
        run_dir / "recalibration_events.parquet",
    )
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)

    result = LegacyRunScanner(platform_settings, repository).scan(serial="TEST_RECAL")

    assert result["runs_discovered"] == 1
    assert result["failures"] == []
    run = repository.list_runs(serial="TEST_RECAL")[0]
    assert run["model_family"] == "bms_soh_recalibration_evidence"
    assert {dataset["name"] for dataset in repository.list_datasets(run["run_id"])} == {
        "recalibration_events.parquet",
        "soh_daily_timeline.parquet",
    }


def test_database_migrates_existing_jobs_for_durable_workers(tmp_path: Path) -> None:
    catalog_path = tmp_path / "legacy.duckdb"
    with duckdb.connect(str(catalog_path)) as connection:
        connection.execute(
            """
            CREATE TABLE pipeline_jobs (
                job_id VARCHAR PRIMARY KEY,
                pipeline_key VARCHAR NOT NULL,
                serial VARCHAR NOT NULL,
                status VARCHAR NOT NULL,
                request_json VARCHAR NOT NULL,
                command_json VARCHAR,
                created_at TIMESTAMPTZ NOT NULL,
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                return_code INTEGER,
                stdout_path VARCHAR,
                stderr_path VARCHAR,
                output_run_id VARCHAR,
                error_message VARCHAR
            )
            """
        )

    CatalogDatabase(catalog_path).initialize()

    with duckdb.connect(str(catalog_path)) as connection:
        rows = connection.execute("PRAGMA table_info('pipeline_jobs')").fetchall()
        columns = {row[1] for row in rows}
    assert {"worker_pid", "result_path", "archived_at"} <= columns


def test_device_inventory_is_persisted(platform_settings: Settings) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    repository.upsert_device_inventory(
        {
            "serial": "TEST001",
            "raw_file_count": 2,
            "raw_row_count": 120,
            "observed_start_at": None,
            "observed_end_at": None,
            "raw_total_bytes": 4096,
            "raw_latest_mtime_ns": 123,
            "has_classified": True,
            "has_events": False,
            "has_capacity_trend": False,
            "scanned_at": "2026-07-20T00:00:00+00:00",
        }
    )

    item = repository.get_device_inventory("TEST001")
    assert item is not None
    assert item["raw_row_count"] == 120
    assert item["has_classified"] is True

    repository.upsert_device_inventory(
        {
            "serial": "TEST001",
            "raw_file_count": 3,
            "raw_row_count": 240,
            "observed_start_at": None,
            "observed_end_at": None,
            "raw_total_bytes": 8192,
            "raw_latest_mtime_ns": 456,
            "has_classified": True,
            "has_events": True,
            "has_capacity_trend": False,
            "scanned_at": "2026-07-21T00:00:00+00:00",
        }
    )
    updated = repository.get_device_inventory("TEST001")
    assert updated is not None
    assert updated["raw_row_count"] == 240
    assert updated["has_events"] is True
