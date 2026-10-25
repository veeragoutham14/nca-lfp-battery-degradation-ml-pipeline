from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batteries (
    serial VARCHAR PRIMARY KEY,
    source_root VARCHAR NOT NULL,
    first_observed_at TIMESTAMPTZ,
    last_observed_at TIMESTAMPTZ,
    latest_run_id VARCHAR,
    run_count BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id VARCHAR PRIMARY KEY,
    source_run_id VARCHAR NOT NULL,
    serial VARCHAR NOT NULL,
    pipeline_key VARCHAR NOT NULL,
    model_family VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    source_path VARCHAR NOT NULL,
    summary_path VARCHAR NOT NULL,
    nominal_capacity_ah DOUBLE,
    training_rows BIGINT,
    validation_rows BIGINT,
    validation_mae_soh_pct DOUBLE,
    validation_max_error_soh_pct DOUBLE,
    loss_slope_per_1000ah DOUBLE,
    forecast_rows BIGINT,
    observed_start_at TIMESTAMPTZ,
    observed_end_at TIMESTAMPTZ,
    run_created_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL,
    fingerprint VARCHAR NOT NULL,
    summary_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    kind VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    media_type VARCHAR NOT NULL,
    size_bytes BIGINT NOT NULL,
    modified_at TIMESTAMPTZ,
    fingerprint VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id VARCHAR PRIMARY KEY,
    run_id VARCHAR NOT NULL,
    serial VARCHAR NOT NULL,
    stage VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    path VARCHAR NOT NULL,
    format VARCHAR NOT NULL,
    row_count BIGINT,
    schema_json VARCHAR,
    observed_start_at TIMESTAMPTZ,
    observed_end_at TIMESTAMPTZ,
    fingerprint VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_jobs (
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
    worker_pid BIGINT,
    result_path VARCHAR,
    output_run_id VARCHAR,
    error_message VARCHAR,
    archived_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS device_inventory (
    serial VARCHAR PRIMARY KEY,
    raw_file_count BIGINT NOT NULL DEFAULT 0,
    raw_row_count BIGINT NOT NULL DEFAULT 0,
    observed_start_at TIMESTAMPTZ,
    observed_end_at TIMESTAMPTZ,
    raw_total_bytes BIGINT NOT NULL DEFAULT 0,
    raw_latest_mtime_ns BIGINT NOT NULL DEFAULT 0,
    has_cell_signal_audit BOOLEAN NOT NULL DEFAULT FALSE,
    has_cell_health BOOLEAN NOT NULL DEFAULT FALSE,
    has_electrothermal_source_audit BOOLEAN NOT NULL DEFAULT FALSE,
    has_hierarchical_features BOOLEAN NOT NULL DEFAULT FALSE,
    has_soh_signal_audit BOOLEAN NOT NULL DEFAULT FALSE,
    has_soh_recalibration BOOLEAN NOT NULL DEFAULT FALSE,
    has_seasonal_charge_profile BOOLEAN NOT NULL DEFAULT FALSE,
    has_classified BOOLEAN NOT NULL DEFAULT FALSE,
    has_events BOOLEAN NOT NULL DEFAULT FALSE,
    has_capacity_trend BOOLEAN NOT NULL DEFAULT FALSE,
    scanned_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS device_name_overrides (
    serial VARCHAR PRIMARY KEY,
    display_name VARCHAR NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

ALTER TABLE pipeline_jobs ADD COLUMN IF NOT EXISTS worker_pid BIGINT;
ALTER TABLE pipeline_jobs ADD COLUMN IF NOT EXISTS result_path VARCHAR;
ALTER TABLE pipeline_jobs ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_soh_signal_audit BOOLEAN
    DEFAULT FALSE;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_cell_signal_audit BOOLEAN
    DEFAULT FALSE;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_cell_health BOOLEAN
    DEFAULT FALSE;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_electrothermal_source_audit BOOLEAN
    DEFAULT FALSE;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_hierarchical_features BOOLEAN
    DEFAULT FALSE;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_soh_recalibration BOOLEAN
    DEFAULT FALSE;
ALTER TABLE device_inventory ADD COLUMN IF NOT EXISTS has_seasonal_charge_profile BOOLEAN
    DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_runs_serial ON analysis_runs(serial);
CREATE INDEX IF NOT EXISTS idx_runs_created ON analysis_runs(run_created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS idx_datasets_run ON datasets(run_id);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON pipeline_jobs(created_at);
DROP INDEX IF EXISTS idx_inventory_scanned;
"""


class CatalogDatabase:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute(SCHEMA_SQL)

    @contextmanager
    def connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            connection = duckdb.connect(str(self.path))
            try:
                yield connection
            finally:
                connection.close()

    def rows(self, sql: str, parameters: Sequence[Any] | None = None) -> list[dict[str, Any]]:
        with self.connection() as connection:
            cursor = connection.execute(sql, parameters or [])
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def row(self, sql: str, parameters: Sequence[Any] | None = None) -> dict[str, Any] | None:
        rows = self.rows(sql, parameters)
        return rows[0] if rows else None
