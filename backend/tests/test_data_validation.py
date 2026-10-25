from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from battryml.analytics.data_validation import (
    build_data_validation,
    build_electrothermal_validation,
)


def test_data_validation_reader_uses_latest_chemistry_partitioned_run(tmp_path: Path) -> None:
    run = (
        tmp_path
        / "data"
        / "processed"
        / "data_validation"
        / "chemistry=NCA"
        / "serial=TEST001"
        / "validation_run=20260101T000000Z"
    )
    run.mkdir(parents=True)
    (run / "validation_summary.json").write_text(
        json.dumps({"serial": "TEST001", "chemistry": "NCA"}), encoding="utf-8"
    )
    pq.write_table(
        pa.Table.from_pylist([{"signal_key": "soc", "available": True}]),
        run / "signal_inventory.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"month": "2026-01-01T00:00:00+00:00"}]),
        run / "coverage_by_month.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"channel_key": "m01_p01_cv00"}]),
        run / "topology_inventory.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"finding_key": "duplicates"}]),
        run / "quality_findings.parquet",
    )

    result = build_data_validation(
        source_root=tmp_path, serial="TEST001", chemistry="NCA"
    )

    assert result is not None
    assert result["run_id"] == "validation_run=20260101T000000Z"
    assert result["signal_inventory"][0]["signal_key"] == "soc"
    assert build_data_validation(
        source_root=tmp_path, serial="MISSING", chemistry="NCA"
    ) is None


def test_electrothermal_validation_reader_reports_build_readiness_and_staleness(
    tmp_path: Path,
) -> None:
    serial = "TEST001"
    for group in ("temperature", "cell_voltage"):
        source = (
            tmp_path
            / "data"
            / "electrothermal_telemetry"
            / group
            / "5m"
            / "chemistry=NCA"
            / f"serial={serial}"
        )
        source.mkdir(parents=True)
        pq.write_table(pa.table({"time": [1]}), source / "part-00000.parquet")
    run = (
        tmp_path
        / "data"
        / "processed"
        / "electrothermal_import_validation"
        / "chemistry=NCA"
        / f"serial={serial}"
        / "validation_run=20260101T000000Z"
    )
    run.mkdir(parents=True)
    (run / "model_summary.json").write_text(
        json.dumps(
            {
                "ready": True,
                "module_count": 2,
                "validated_at": "2026-01-01T00:00:00+00:00",
                "groups": {
                    key: {
                        "ready": True,
                        "expected_channels": 16 if key == "temperature" else 56,
                        "minimum_channel_coverage": 0.98,
                        "start_utc": "2025-01-01T00:00:00+00:00",
                        "end_utc": "2025-12-31T23:55:00+00:00",
                        "duplicate_timestamps": 0,
                        "misaligned_five_minute_rows": 0,
                    }
                    for key in ("temperature", "cell_voltage")
                },
            }
        ),
        encoding="utf-8",
    )

    result = build_electrothermal_validation(
        source_root=tmp_path, serial=serial, chemistry="NCA"
    )

    assert result is not None
    assert result["build_ready"] is True
    assert result["groups"]["temperature"]["expected_channels"] == 16
    assert result["groups"]["cell_voltage"]["minimum_channel_coverage_pct"] == 98.0

    source_file = next(
        (tmp_path / "data" / "electrothermal_telemetry").rglob("*.parquet")
    )
    summary_mtime_ns = (run / "model_summary.json").stat().st_mtime_ns
    os.utime(source_file, ns=(summary_mtime_ns + 1_000_000_000,) * 2)
    stale = build_electrothermal_validation(
        source_root=tmp_path, serial=serial, chemistry="NCA"
    )
    assert stale is not None
    assert stale["status"] == "stale"
    assert stale["build_ready"] is False
