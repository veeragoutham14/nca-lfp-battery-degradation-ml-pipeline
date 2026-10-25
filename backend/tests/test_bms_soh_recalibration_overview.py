from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from battryml.analytics.bms_soh_recalibration_overview import (
    build_bms_soh_recalibration_overview,
    build_bms_soh_recalibration_series,
)
from battryml.battery_profiles import battery_profile

FIELD_NOMINAL_AH = battery_profile("100000001").nominal_capacity_ah


def test_bms_soh_recalibration_overview_and_series_are_source_backed(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "data"
        / "processed"
        / "bms_soh_recalibration"
        / "serial=100000004"
        / "recalibration_run=20260807T120000Z"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "model_family": "bms_soh_recalibration_evidence",
                "detector_version": "v4",
                "serial": "100000004",
                "chemistry": "NCA",
                "module_count": 2,
                "nominal_capacity_ah": FIELD_NOMINAL_AH,
                "observed_start": "2026-06-01T00:00:00+00:00",
                "observed_end": "2026-06-04T00:00:00+00:00",
                "aggregation_minutes": 5,
                "event_detection_method": (
                    "quality_gated_daily_bms_soh_change_events_from_soh_signal_audit"
                ),
                "event_time_semantics": "daily first observation",
                "event_rows": 2,
                "step_drop_rows": 1,
                "step_recovery_rows": 0,
                "change_after_gap_rows": 1,
                "high_confidence_rows": 1,
                "strong_context_match_rows": 1,
                "limited_context_rows": 1,
                "exact_transition_rows": 1,
                "interval_censored_transition_rows": 1,
                "daily_fallback_transition_rows": 0,
                "exact_transition_rate": 0.5,
                "matched_control_rows": 4,
                "one_second_capacity_validation": {
                    "available": True,
                    "cycle_rows": 1,
                    "validation_grade_rows": 1,
                    "median_discharged_ah": 47.0,
                    "median_capacity_soh_pct": 95.14,
                },
                "findings": ["One persistent transition was resolved."],
                "classification_counts": {
                    "near_full_charge_and_rest_anchor": 1,
                    "data_gap_or_estimator_reset_risk": 1,
                },
                "source_soh_audit_run_id": "audit_run=20260807T110000Z",
                "source_soh_audit_label_status": "weak_label_only",
                "configuration": {"context_lookback_hours": 72},
                "claim_limit": "Temporal evidence only.",
                "phase2": {
                    "available": True,
                    "model_family": "capacity_bms_validation",
                    "phase2_version": "v1",
                    "discharge_event_rows": 1,
                    "capacity_claim_eligible_rows": 1,
                    "strict_full_discharge_rows": 1,
                    "near_full_discharge_rows": 0,
                    "operational_full_discharge_rows": 0,
                    "context_only_discharge_rows": 0,
                    "high_quality_capacity_rows": 1,
                    "soh_transition_rows": 1,
                    "transitions_with_preceding_discharge_rows": 1,
                    "qualifying_discharges_without_soh_update_rows": 0,
                    "claim_limit": "BMS-derived evidence only.",
                },
            }
        ),
        encoding="utf-8",
    )
    first_day = datetime(2026, 6, 2, tzinfo=UTC)
    pq.write_table(
        pa.table(
            {
                "day_utc": [first_day, first_day + timedelta(days=1)],
                "measured_soh_pct": [98.0, 96.0],
                "coverage_fraction": [1.0, 0.95],
                "recalibration_event_count": [0, 2],
                "step_drop_count": [0, 1],
                "step_recovery_count": [0, 0],
            }
        ),
        run_dir / "soh_daily_timeline.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "event_id": ["recalibration_00001", "recalibration_00002"],
                "event_time_utc": [first_day, first_day + timedelta(days=1)],
                "event_type": ["step_drop", "change_after_gap"],
                "soh_delta_pp": [-1.0, -2.0],
                "raw_discharge_detection_status": ["resolved", "not_observed"],
                "raw_discharge_candidate_count": [2, 0],
                "raw_discharge_ambiguous": [True, False],
                "raw_start_time_utc": [first_day - timedelta(hours=4), None],
                "raw_end_time_utc": [first_day - timedelta(hours=1), None],
                "first_low_band_time_utc": [first_day - timedelta(hours=2), None],
                "minimum_soc_time_utc": [first_day - timedelta(hours=1), None],
                "terminal_low_soc_tail_minutes": [60.0, None],
                "evidence_classification": [
                    "near_full_charge_and_rest_anchor",
                    "data_gap_or_estimator_reset_risk",
                ],
                "classification_confidence": ["high", "limited"],
            }
        ),
        run_dir / "recalibration_events.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "feature_key": ["near_full_within_12h"],
                "feature_label": ["Reached near-full SOC within 12 hours"],
                "event_count": [2],
                "event_total": [2],
                "event_rate": [1.0],
                "control_count": [1],
                "control_total": [4],
                "control_rate": [0.25],
                "risk_difference": [0.75],
                "prevalence_lift": [4.0],
            }
        ),
        run_dir / "evidence_comparison.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "discharge_episode_id": ["discharge_000001"],
                "raw_start_time_utc": [first_day],
                "raw_end_time_utc": [first_day + timedelta(hours=4)],
                "capacity_validation_status": ["validation_grade"],
                "cycle_continuity_class": ["continuous"],
                "discharged_ah_1s": [47.0],
                "direct_capacity_soh_pct": [95.14],
                "raw_discharge_candidate_count": [2],
                "raw_discharge_ambiguous": [True],
                "first_low_band_time_utc": [first_day + timedelta(hours=3)],
                "minimum_soc_time_utc": [first_day + timedelta(hours=4)],
                "terminal_low_soc_tail_minutes": [60.0],
                "terminal_low_soc_rest_tail_minutes": [15.0],
            }
        ),
        run_dir / "one_second_capacity_validation.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "discharge_event_id": ["phase2_discharge_00001"],
                "end_utc": [first_day - timedelta(hours=1)],
                "evidence_class": ["strict_full_discharge"],
                "discharged_ah": [47.0],
                "measurement_quality": ["high"],
                "reference_comparability": ["directly_comparable"],
            }
        ),
        run_dir / "capacity_discharge_events.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "transition_id": ["recalibration_00001"],
                "transition_time_utc": [first_day],
                "delta_soh_pp": [-1.0],
                "timestamp_quality": ["exact"],
            }
        ),
        run_dir / "bms_soh_transitions.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "transition_id": ["recalibration_00001"],
                "primary_preceding_discharge_event_id": ["phase2_discharge_00001"],
                "nearest_qualifying_discharge_event_id": ["phase2_discharge_00001"],
            }
        ),
        run_dir / "soh_discharge_matches.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "transition_id": ["recalibration_00001"],
                "discharge_event_id": ["phase2_discharge_00001"],
                "evidence_rank": [1],
                "proximity_rank": [1],
            }
        ),
        run_dir / "event_match_candidates.parquet",
    )
    capacity_run = (
        tmp_path
        / "data"
        / "processed"
        / "discharge_capacity_evidence"
        / "serial=100000004"
        / "capacity_evidence_run=20260807T115000Z"
    )
    capacity_run.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "discharge_episode_id": ["canonical_000001"],
                "raw_start_time_utc": [first_day - timedelta(hours=8)],
                "raw_end_time_utc": [first_day - timedelta(hours=2)],
                "raw_soc_drop_pct": [91.0],
                "capacity_validation_status": ["exploratory_soc_boundaries"],
                "cycle_continuity_class": ["interrupted"],
                "discharged_ah_1s": [44.4],
                "soc_normalized_capacity_ah": [48.79],
                "soc_normalized_capacity_soh_pct": [98.77],
                "bms_soh_at_cycle_completion_pct": [98.0],
                "bms_completion_minus_normalized_soh_pp": [-0.77],
                "rest_minutes_inside_cycle": [20.0],
                "longest_rest_minutes": [10.0],
                "rms_discharge_current_a": [8.2],
                "pack_temperature_mean_c": [24.0],
                "cell_voltage_spread_p95_mv": [18.0],
            }
        ),
        capacity_run / "qualified_capacity_points.parquet",
    )
    validation_run = (
        tmp_path
        / "data"
        / "processed"
        / "data_validation"
        / "chemistry=NCA"
        / "serial=100000004"
        / "validation_run=20260807T114500Z"
    )
    validation_run.mkdir(parents=True)
    (validation_run / "model_summary.json").write_text(
        json.dumps({"readiness": {"capacity_bms": {"status": "ready"}}}),
        encoding="utf-8",
    )
    devices = [
        {
            "serial": "100000004",
            "display_name": "NCA Field 03",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
            "nominal_capacity_ah": FIELD_NOMINAL_AH,
            "readiness": {
                "hierarchical_features": True,
                "soh_signal_audit": True,
            },
        },
        {
            "serial": "LFP001",
            "display_name": "LFP Field 01",
            "chemistry": "LFP",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
            "nominal_capacity_ah": 72.0,
            "readiness": {"hierarchical_features": False, "soh_signal_audit": False},
        },
    ]

    overview = build_bms_soh_recalibration_overview(
        source_root=tmp_path,
        devices=devices,
        chemistry="NCA",
    )
    series = build_bms_soh_recalibration_series(
        source_root=tmp_path,
        serial="100000004",
    )

    assert overview["metrics"]["configured_device_count"] == 1
    assert overview["metrics"]["analyzed_device_count"] == 1
    assert overview["metrics"]["total_event_count"] == 2
    assert overview["metrics"]["high_confidence_event_count"] == 1
    assert overview["metrics"]["exact_transition_count"] == 1
    assert overview["devices"][0]["prerequisites_ready"] is True
    assert series is not None
    assert series["summary"]["source_soh_audit_label_status"] == "weak_label_only"
    assert series["points"][1]["time"] == "2026-06-03T00:00:00+00:00"
    assert series["events"][0]["event_time"] == "2026-06-02T00:00:00+00:00"
    assert series["events"][0]["raw_discharge_candidate_count"] == 2
    assert series["events"][0]["minimum_soc_time_utc"] == "2026-06-01T23:00:00+00:00"
    assert series["comparison"][0]["prevalence_lift"] == 4.0
    assert series["capacity_validation"][0]["discharged_ah_1s"] == 44.4
    assert series["capacity_validation"][0]["linked_soh_event_id"] == "recalibration_00001"
    assert series["summary"]["source_capacity_evidence_run_id"] == capacity_run.name
    assert series["summary"]["phase2"]["strict_full_discharge_rows"] == 1
    assert series["capacity_events"][0]["measurement_quality"] == "high"
    assert series["soh_transitions"][0]["timestamp_quality"] == "exact"
    assert series["discharge_matches"][0]["primary_preceding_discharge_event_id"] == "phase2_discharge_00001"
    assert series["match_candidates"][0]["evidence_rank"] == 1
    assert series["summary"]["one_second_capacity_validation"]["operational_window_rows"] == 1
    assert series["reference_comparison"]["available"] is False
