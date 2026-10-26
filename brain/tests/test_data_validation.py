from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.data_validation import validate_device
from src.electrothermal_contract import expected_hierarchical_signals


def test_validation_builds_scientific_readiness_and_reuses_unchanged_inputs(
    tmp_path: Path,
) -> None:
    serial = "TEST001"
    data_root = tmp_path / "data"
    raw_dir = data_root / "raw_parquet" / f"serial={serial}"
    processed = data_root / "processed"
    audit_run = (
        processed
        / "bms_electrothermal_source_quality"
        / f"serial={serial}"
        / "source_audit_run=20260101T010000Z"
    )
    feature_run = (
        processed
        / "bms_hierarchical_features"
        / f"serial={serial}"
        / "feature_run=20260101T020000Z"
    )
    month_dir = feature_run / "year=2026" / "month=01"
    raw_dir.mkdir(parents=True)
    audit_run.mkdir(parents=True)
    month_dir.mkdir(parents=True)

    times = pd.date_range(datetime(2026, 1, 1, tzinfo=UTC), periods=900, freq="s")
    pd.DataFrame(
        {
            "time": times,
            "serial": serial,
            "batt_soc_pct": 40.0 + pd.Series(range(900)) / 300.0,
            "batt_soh_pct": 98.0,
            "batt_current_a": 5.0,
            "batt_voltage_v": 100.0,
        }
    ).to_parquet(raw_dir / "part-00000.parquet", index=False)
    (audit_run / "model_summary.json").write_text(
        json.dumps({"source_quality_ready": True}), encoding="utf-8"
    )
    (feature_run / "model_summary.json").write_text(
        json.dumps(
            {
                "start": times[0].isoformat(),
                "end": (times[-1] + pd.Timedelta(seconds=1)).isoformat(),
                "feature_rows": 3,
                "aggregation_minutes": 5,
            }
        ),
        encoding="utf-8",
    )
    channel_map = pd.DataFrame(
        [signal.to_dict() for signal in expected_hierarchical_signals(1)]
    )
    channel_map.to_parquet(feature_run / "channel_map.parquet", index=False)
    pd.DataFrame(
        {
            "window_start_utc": times[::300],
            "system_temperature_min_c": [20.0, 21.0, 22.0],
            "system_temperature_max_c": [24.0, 25.0, 26.0],
        }
    ).to_parquet(month_dir / "system_features_5min.parquet", index=False)

    first = validate_device(
        serial=serial,
        chemistry="NCA",
        module_count=1,
        source_root=tmp_path,
        raw_root=data_root / "raw_parquet",
        processed_root=processed,
        output_root=processed / "data_validation",
    )
    second = validate_device(
        serial=serial,
        chemistry="NCA",
        module_count=1,
        source_root=tmp_path,
        raw_root=data_root / "raw_parquet",
        processed_root=processed,
        output_root=processed / "data_validation",
    )

    summary = json.loads(
        (first / "validation_summary.json").read_text(encoding="utf-8")
    )
    assert first == second
    assert summary["raw_row_count"] == 900
    assert summary["readiness"]["capacity_bms"]["status"] == "ready"
    assert summary["readiness"]["monthly_exposure"]["status"] == "ready"
    assert summary["readiness"]["voltage_group_health"]["status"] == "ready"
    assert summary["feature_evidence"]["available_voltage_groups"] == 28
    assert summary["feature_evidence"]["available_temperature_sensors"] == 8
    assert summary["suspicious_temperature_windows"] == 0
    assert (first / "coverage_by_month.parquet").exists()
    assert (first / "input_manifest.json").exists()


def test_findings_frame_writes_mixed_values_as_text(tmp_path):
    from src.data_validation import _findings_frame
    from src.io_utils import write_parquet_chunk

    frame = _findings_frame(
        [
            {"finding_key": "duplicates", "value": 0},
            {"finding_key": "freshness", "value": None},
            {"finding_key": "lifecycle_counter", "value": "sys_cycles"},
        ]
    )
    write_parquet_chunk(frame, tmp_path / "quality_findings.parquet", compression="snappy")

    assert list(frame["value"]) == ["0", None, "sys_cycles"]
    assert _findings_frame([{"finding_key": "duplicates", "value": 3}])["value"].iloc[0] == 3
