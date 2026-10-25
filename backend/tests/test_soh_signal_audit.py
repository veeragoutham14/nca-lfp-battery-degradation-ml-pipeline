from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from battryml.analytics.soh_signal_audit import run_audit
from battryml.battery_profiles import battery_profile

FIELD_NOMINAL_AH = battery_profile("100000001").nominal_capacity_ah


def test_soh_signal_audit_detects_drops_recoveries_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    serial = "100000002"
    raw_dir = tmp_path / "raw" / f"serial={serial}"
    raw_dir.mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps: list[datetime] = []
    soh: list[float] = []
    for day_index, value in enumerate((100.0, 98.0, 99.0)):
        for sample_index in range(10):
            timestamps.append(start + timedelta(days=day_index, minutes=sample_index))
            soh.append(value)
    table = pa.table(
        {
            "time": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "serial": pa.array([int(serial)] * len(timestamps), type=pa.int64()),
            "batt_soh_pct": soh,
            "batt_soc_pct": [60.0] * len(timestamps),
            "batt_current_a": [-10.0] * len(timestamps),
            "batt_voltage_v": [320.0] * len(timestamps),
        }
    )
    pq.write_table(table, raw_dir / "telemetry.parquet")

    run_dir = run_audit(
        serial=serial,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        expected_sample_interval_seconds=60.0,
        max_integration_gap_seconds=60.0,
        gap_threshold_seconds=120.0,
        run_id="20260723T120000Z",
    )

    summary = json.loads((run_dir / "model_summary.json").read_text())
    metrics = summary["audit_metrics"]
    assert metrics["step_drop_count"] == 1
    assert metrics["step_recovery_count"] == 1
    assert metrics["largest_step_drop_pct"] == -2.0
    assert metrics["label_status"] == "weak_label_only"
    assert metrics["signal_quality_status"] == "unstable_bms_estimate"
    assert metrics["ground_truth_status"] == "missing_independent_capacity_label"
    assert summary["training_rows"] == 3
    assert (run_dir / "daily_health.parquet").exists()
    assert pq.read_table(run_dir / "soh_change_events.parquet").num_rows == 2
    assert "Reversible SOH movement" in (
        run_dir / "soh_signal_audit_report.html"
    ).read_text(encoding="utf-8")


def test_flat_bms_soh_remains_a_weak_label(tmp_path: Path) -> None:
    serial = "100000031"
    raw_dir = tmp_path / "raw" / f"serial={serial}"
    raw_dir.mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [
        start + timedelta(days=day_index, seconds=sample_index)
        for day_index in range(2)
        for sample_index in range(10)
    ]
    pq.write_table(
        pa.table(
            {
                "time": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
                "serial": pa.array([int(serial)] * len(timestamps), type=pa.int64()),
                "batt_soh_pct": [100.0] * len(timestamps),
                "batt_soc_pct": [60.0] * len(timestamps),
                "batt_current_a": [-10.0] * len(timestamps),
                "batt_voltage_v": [320.0] * len(timestamps),
            }
        ),
        raw_dir / "telemetry.parquet",
    )

    run_dir = run_audit(
        serial=serial,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "processed",
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        expected_sample_interval_seconds=8640.0,
        max_integration_gap_seconds=8640.0,
        gap_threshold_seconds=17280.0,
        run_id="20260723T130000Z",
    )

    summary = json.loads((run_dir / "model_summary.json").read_text())
    metrics = summary["audit_metrics"]
    assert metrics["signal_quality_status"] == "stable_observed_bms_estimate"
    assert metrics["label_status"] == "weak_label_only"
    assert summary["label_policy"]["ground_truth_required"] is True
    assert any("flat BMS estimate" in item for item in summary["findings"])
