from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from battryml.analytics.cell_signal_overview import build_cell_signal_overview


def test_cell_signal_overview_uses_latest_audit_and_feature_outputs(
    tmp_path: Path,
) -> None:
    audit_run = (
        tmp_path
        / "data"
        / "processed"
        / "cell_signal_audit"
        / "serial=100000001"
        / "cell_audit_run=20260702T000000Z"
    )
    feature_run = (
        tmp_path
        / "data"
        / "processed"
        / "cell_health"
        / "serial=100000001"
        / "feature_run=20260703T000000Z"
    )
    stale_audit_run = (
        tmp_path
        / "data"
        / "processed"
        / "cell_signal_audit"
        / "serial=100000002"
        / "cell_audit_run=20260702T000000Z"
    )
    audit_run.mkdir(parents=True)
    feature_run.mkdir(parents=True)
    stale_audit_run.mkdir(parents=True)
    (audit_run / "model_summary.json").write_text(
        json.dumps(
            {
                "sample_start": "2026-07-01T00:00:00+00:00",
                "sample_end": datetime.now(UTC).isoformat(),
                "module_count": 2,
                "expected_cell_channels": 56,
                "present_cell_channels": 56,
                "ready_cell_channels": 56,
                "total_rows": 72_884,
                "nominal_1hz_coverage_pct": 84.36,
                "minimum_cell_channel_valid_coverage_pct": 100.0,
                "global_cell_voltage_signals_ready": True,
                "cell_temperature_signals_ready": True,
                "cell_voltage_ready": True,
            }
        ),
        encoding="utf-8",
    )
    (stale_audit_run / "model_summary.json").write_text(
        json.dumps(
            {
                "sample_start": "2024-12-30T00:00:00+00:00",
                "sample_end": "2024-12-31T00:00:00+00:00",
                "module_count": 3,
                "expected_cell_channels": 84,
                "present_cell_channels": 84,
                "ready_cell_channels": 84,
                "total_rows": 86_400,
                "nominal_1hz_coverage_pct": 100.0,
                "minimum_cell_channel_valid_coverage_pct": 100.0,
                "global_cell_voltage_signals_ready": True,
                "cell_temperature_signals_ready": True,
                "cell_voltage_ready": True,
            }
        ),
        encoding="utf-8",
    )
    (feature_run / "model_summary.json").write_text("{}", encoding="utf-8")
    devices = [
        {
            "serial": "100000001",
            "variant_key": "nca-home",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
            "configured_start": "2023-08-01T00:00:00+00:00",
            "observed_start_at": "2023-08-01T10:25:15+02:00",
            "observed_end_at": "2026-07-02T00:00:00+00:00",
            "readiness": {"cell_health": False},
        },
        {
            "serial": "100000002",
            "variant_key": "nca-home",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 3,
            "module_count_status": "verified",
            "configured_start": "2023-08-01T00:00:00+00:00",
            "observed_end_at": "2024-12-31T00:00:00+00:00",
            "readiness": {"cell_health": False},
        },
        {
            "serial": "100000017",
            "variant_key": "nca-reference",
            "chemistry": "NCA",
            "device_role": "full_cycle_test",
            "module_count": None,
            "module_count_status": "missing",
            "configured_start": None,
            "observed_start_at": "2022-05-01T00:00:00+00:00",
            "observed_end_at": "2026-05-21T00:00:00+00:00",
            "readiness": {"cell_health": False},
        },
        {
            "serial": "LFP001",
            "chemistry": "LFP",
            "device_role": "field_tester",
        },
    ]

    overview = build_cell_signal_overview(source_root=tmp_path, devices=devices)

    assert overview["metrics"]["inventory_device_count"] == 3
    assert overview["metrics"]["configured_device_count"] == 3
    assert overview["metrics"]["field_device_count"] == 2
    assert overview["metrics"]["all_field_device_count"] == 2
    assert overview["metrics"]["deferred_field_device_count"] == 0
    assert overview["metrics"]["reference_device_count"] == 1
    assert overview["metrics"]["audited_device_count"] == 0
    assert overview["metrics"]["ready_device_count"] == 0
    assert overview["metrics"]["fresh_ready_device_count"] == 0
    assert overview["metrics"]["stale_device_count"] == 0
    assert overview["metrics"]["pending_device_count"] == 3
    assert overview["metrics"]["total_expected_active_cell_channels"] == 196
    assert overview["metrics"]["total_ready_active_cell_channels"] == 0
    assert overview["devices"][0]["cell_health_built"] is False
    assert overview["devices"][0]["feature_contract"] == "missing"
    assert (
        overview["devices"][0]["recommended_feature_start"]
        == "2023-08-01T08:25:15+00:00"
    )
    assert overview["devices"][0]["telemetry_fresh"] is False
    assert overview["devices"][0]["minimum_cell_channel_valid_coverage_pct"] is None
    assert overview["devices"][1]["serial"] == "100000017"
    assert overview["devices"][1]["cohort_role"] == "reference_control"
    assert overview["devices"][1]["module_count"] == 2
    assert overview["devices"][1]["observed_start"] == "2022-05-01T00:00:00+00:00"
    assert overview["devices"][2]["serial"] == "100000002"
    assert overview["devices"][2]["cohort_status"] == "included"
    assert overview["devices"][2]["in_active_cohort"] is True
    assert "3-module" in overview["devices"][2]["cohort_reason"]


def test_cell_signal_overview_prefers_hierarchical_quality_contract(tmp_path: Path) -> None:
    audit_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_electrothermal_source_quality"
        / "serial=100000001"
        / "source_audit_run=20260728T000000Z"
    )
    feature_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_hierarchical_features"
        / "serial=100000001"
        / "feature_run=20260728T010000Z"
    )
    audit_run.mkdir(parents=True)
    feature_run.mkdir(parents=True)
    (audit_run / "model_summary.json").write_text(
        json.dumps(
            {
                "sample_start": "2026-07-27T00:00:00+00:00",
                "sample_end": datetime.now(UTC).isoformat(),
                "source_quality_ready": True,
                "quality_gate_status": "ready_with_warnings",
                "blocking_reasons": [],
                "quality_warnings": [
                    "inactive_temperature_channels_excluded_from_temperature_context"
                ],
                "expected_voltage_channels": 56,
                "ready_voltage_channels": 56,
                "expected_temperature_sensors": 16,
                "active_temperature_sensors": 8,
                "inactive_temperature_sensors": 8,
                "ready_temperature_sensors": 8,
                "minimum_voltage_channel_valid_coverage_pct": 99.9,
                "minimum_temperature_sensor_valid_coverage_pct": 0.0,
                "minimum_active_temperature_sensor_valid_coverage_pct": 99.8,
                "nominal_1hz_coverage_pct": 99.7,
                "voltage_extrema_reconciled": True,
                "temperature_extrema_reconciled": True,
            }
        ),
        encoding="utf-8",
    )
    (feature_run / "model_summary.json").write_text(
        json.dumps(
            {
                "start": "2023-08-01T00:00:00+00:00",
                "end": "2026-07-28T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    overview = build_cell_signal_overview(
        source_root=tmp_path,
        devices=[
            {
                "serial": "100000001",
                "chemistry": "NCA",
                "device_role": "field_tester",
                "module_count": 2,
                "module_count_status": "verified",
            }
        ],
    )
    device = overview["devices"][0]
    assert device["source_contract"] == "hierarchical_electrothermal_v1"
    assert device["cell_voltage_ready"] is True
    assert device["quality_gate_status"] == "ready_with_warnings"
    assert device["blocking_reasons"] == []
    assert device["quality_warnings"] == [
        "inactive_temperature_channels_excluded_from_temperature_context"
    ]
    assert device["active_temperature_sensors"] == 8
    assert device["inactive_temperature_sensors"] == 8
    assert device["ready_temperature_sensors"] == 8
    assert device["minimum_active_temperature_sensor_valid_coverage_pct"] == 99.8
    assert device["feature_contract"] == "hierarchical_electrothermal_v1"
    assert device["feature_end"] == "2026-07-28T00:00:00+00:00"
    assert device["recommended_feature_start"] == "2026-07-28T00:00:00+00:00"
    assert device["feature_build_mode"] == "incremental"


def test_cell_signal_overview_uses_chronological_hierarchical_feature_coverage(
    tmp_path: Path,
) -> None:
    feature_root = (
        tmp_path
        / "data"
        / "processed"
        / "bms_hierarchical_features"
        / "serial=100000001"
    )
    newest_run = feature_root / "feature_run=20260728T020000Z"
    furthest_run = feature_root / "feature_run=20260728T010000Z"
    newest_run.mkdir(parents=True)
    furthest_run.mkdir(parents=True)
    (newest_run / "model_summary.json").write_text(
        json.dumps(
            {
                "start": "2023-08-01T00:00:00+00:00",
                "end": "2024-07-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (furthest_run / "model_summary.json").write_text(
        json.dumps(
            {
                "start": "2026-07-01T00:00:00+00:00",
                "end": "2026-07-28T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    overview = build_cell_signal_overview(
        source_root=tmp_path,
        devices=[
            {
                "serial": "100000001",
                "chemistry": "NCA",
                "device_role": "field_tester",
                "module_count": 2,
                "module_count_status": "verified",
            }
        ],
    )

    device = overview["devices"][0]
    assert device["feature_start"] == "2023-08-01T00:00:00+00:00"
    assert device["feature_end"] == "2026-07-28T00:00:00+00:00"
    assert device["feature_run_id"] == "feature_run=20260728T010000Z"


def test_cell_signal_overview_uses_exclusive_end_and_five_minute_append_boundary(
    tmp_path: Path,
) -> None:
    audit_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_electrothermal_source_quality"
        / "serial=100000005"
        / "source_audit_run=20260728T000000Z"
    )
    feature_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_hierarchical_features"
        / "serial=100000005"
        / "feature_run=20260728T010000Z"
    )
    audit_run.mkdir(parents=True)
    feature_run.mkdir(parents=True)
    audit_summary = audit_run / "model_summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "sample_start": "2026-07-19T00:00:00+00:00",
                "sample_end": "2026-07-20T00:00:00+00:00",
                "source_quality_ready": True,
                "expected_voltage_channels": 56,
                "ready_voltage_channels": 56,
            }
        ),
        encoding="utf-8",
    )
    # The audit ran BEFORE the feature build; its mtime must not force a rebuild.
    audit_mtime = datetime(2026, 7, 20, 10, 0, 0, tzinfo=UTC).timestamp()
    import os

    os.utime(audit_summary, (audit_mtime, audit_mtime))
    summary_path = feature_run / "model_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "start": "2023-08-01T22:00:00+00:00",
                "end": "2026-07-20T11:39:00+00:00",
                "aggregation_minutes": 5,
            }
        ),
        encoding="utf-8",
    )
    devices = [
        {
            "serial": "100000005",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
            "observed_start_at": "2023-08-01T22:00:00+00:00",
            "observed_end_at": "2026-07-20T11:39:59+00:00",
        }
    ]

    overview = build_cell_signal_overview(source_root=tmp_path, devices=devices)
    device = overview["devices"][0]
    assert device["feature_build_mode"] == "incremental"
    assert device["recommended_feature_start"] == "2026-07-20T11:35:00+00:00"
    assert device["recommended_feature_end"] == "2026-07-20T11:40:00+00:00"
    assert device["feature_lag_seconds"] == 60.0

    summary_path.write_text(
        json.dumps(
            {
                "start": "2023-08-01T22:00:00+00:00",
                "end": "2026-07-20T11:40:00+00:00",
                "aggregation_minutes": 5,
            }
        ),
        encoding="utf-8",
    )
    current = build_cell_signal_overview(source_root=tmp_path, devices=devices)["devices"][0]
    assert current["feature_build_mode"] == "up_to_date"
    assert current["recommended_feature_start"] is None
    assert current["recommended_feature_end"] == "2026-07-20T11:40:00+00:00"
    assert current["feature_lag_seconds"] == 0.0


def test_cell_signal_overview_marks_features_stale_when_audit_is_newer(
    tmp_path: Path,
) -> None:
    """Regression: a re-audit after the feature build must force an incremental rebuild.

    The telemetry end may be unchanged, but a newer source audit means the
    five-minute history was built from older source evidence and must be rebuilt.
    """
    audit_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_electrothermal_source_quality"
        / "serial=100000001"
        / "source_audit_run=20260801T000000Z"
    )
    feature_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_hierarchical_features"
        / "serial=100000001"
        / "feature_run=20260728T010000Z"
    )
    audit_run.mkdir(parents=True)
    feature_run.mkdir(parents=True)
    audit_summary = audit_run / "model_summary.json"
    audit_summary.write_text(
        json.dumps(
            {
                "sample_start": "2026-07-27T00:00:00+00:00",
                "sample_end": "2026-07-28T00:00:00+00:00",
                "source_quality_ready": True,
                "expected_voltage_channels": 56,
                "ready_voltage_channels": 56,
            }
        ),
        encoding="utf-8",
    )
    (feature_run / "model_summary.json").write_text(
        json.dumps(
            {
                "start": "2023-08-01T00:00:00+00:00",
                "end": "2026-07-28T00:00:00+00:00",
                "aggregation_minutes": 5,
            }
        ),
        encoding="utf-8",
    )
    # Simulate the audit being written AFTER the feature build completed.
    # The telemetry sample_end (2026-07-28) equals the feature end, so the old
    # telemetry-only comparison would say "up_to_date"; the audit mtime must win.
    audit_mtime = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC).timestamp()
    import os

    os.utime(audit_summary, (audit_mtime, audit_mtime))

    devices = [
        {
            "serial": "100000001",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
            "observed_start_at": "2023-08-01T00:00:00+00:00",
            "observed_end_at": "2026-07-28T00:00:00+00:00",
        }
    ]

    overview = build_cell_signal_overview(source_root=tmp_path, devices=devices)
    device = overview["devices"][0]
    assert device["feature_build_mode"] == "incremental"
    assert device["recommended_feature_start"] == "2026-07-28T00:00:00+00:00"
    assert device["recommended_feature_end"] == "2026-07-28T00:01:00+00:00"