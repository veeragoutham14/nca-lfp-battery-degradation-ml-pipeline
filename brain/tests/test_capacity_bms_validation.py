from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.capacity_bms_validation import (
    _build_matches,
    _canonical_discharge_events,
    _canonical_transitions,
    _complete_episode_inventory,
    _reference_comparability,
    finalize_phase2,
    find_cached_phase2_run,
)
from src.device_specs import FIELD_NOMINAL_AH


def _episodes() -> pd.DataFrame:
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    return pd.DataFrame(
        {
            "discharge_episode_id": ["strict", "near", "operational", "partial"],
            "raw_start_time_utc": [start, start + pd.Timedelta(days=1), start + pd.Timedelta(days=2), start + pd.Timedelta(days=3)],
            "raw_end_time_utc": [start + pd.Timedelta(hours=5), start + pd.Timedelta(days=1, hours=5), start + pd.Timedelta(days=2, hours=5), start + pd.Timedelta(days=3, hours=2)],
            "raw_start_soc_pct": [100.0, 99.0, 95.0, 80.0],
            "raw_end_soc_pct": [0.0, 1.0, 5.0, 50.0],
            "raw_soc_drop_pct": [100.0, 98.0, 90.0, 30.0],
            "discharged_ah_1s": [47.0, 46.0, 42.0, 12.0],
            "discharged_wh_1s": [9300.0, 9100.0, 8300.0, 2400.0],
            "charge_intrusion_ah_1s": [0.0, 0.01, 0.05, 0.0],
            "charge_intrusion_wh_1s": [0.0, 2.0, 10.0, 0.0],
            "active_discharge_minutes": [300.0, 300.0, 300.0, 120.0],
            "raw_coverage_fraction": [1.0, 0.99, 0.95, 1.0],
            "maximum_raw_gap_seconds": [1.0, 2.0, 5.0, 1.0],
            "unobserved_duration_minutes": [0.0, 0.1, 1.0, 0.0],
            "rms_discharge_current_a": [9.4, 9.2, 8.4, 6.0],
            "p95_discharge_current_a": [10.0, 10.0, 9.0, 7.0],
            "p99_discharge_current_a": [10.5, 10.5, 9.5, 7.5],
            "peak_discharge_current_a": [11.0, 11.0, 10.0, 8.0],
            "rms_discharge_c_rate": [0.19, 0.186, 0.17, 0.12],
            "rest_minutes_inside_cycle": [0.0, 2.0, 10.0, 0.0],
            "longest_rest_minutes": [0.0, 2.0, 5.0, 0.0],
            "rest_interruption_count": [0, 1, 2, 0],
            "discharge_dominance_fraction_1s": [1.0, 0.99, 0.95, 1.0],
            "pack_temperature_mean_c": [23.0, 24.0, 25.0, 23.0],
            "pack_temperature_median_c": [23.0, 24.0, 25.0, 23.0],
            "pack_temperature_min_c": [22.0, 23.0, 24.0, 22.0],
            "pack_temperature_max_c": [24.0, 25.0, 26.0, 24.0],
            "pack_temperature_context_coverage_fraction": [1.0, 1.0, 1.0, 1.0],
            "capacity_validation_status": ["validation_grade", "validation_grade", "exploratory_soc_boundaries", "exploratory_soc_boundaries"],
        }
    )


def test_canonical_capacity_contract_preserves_evidence_hierarchy_and_formula() -> None:
    result = _canonical_discharge_events(
        _episodes(),
        serial="100000001",
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        reference_capacity_source="device_configuration",
    )

    assert result["evidence_class"].tolist() == [
        "strict_full_discharge",
        "near_full_discharge",
        "operational_full_discharge",
        "partial_discharge",
    ]
    strict = result.iloc[0]
    assert strict["discharge_derived_soh_proxy_pct"] == 47.0 / FIELD_NOMINAL_AH * 100.0
    assert strict["measurement_quality"] == "high"
    assert pd.isna(result.iloc[1]["discharge_derived_soh_proxy_pct"])
    assert result.iloc[1]["soc_span_normalized_soh_estimate_pct"] == (
        (46.0 - 0.01) / 0.98 / FIELD_NOMINAL_AH * 100.0
    )
    assert result.iloc[2]["measurement_quality"] == "medium"


def test_matching_retains_evidence_first_and_nearest_candidates() -> None:
    discharges = _canonical_discharge_events(
        _episodes().iloc[:3],
        serial="100000001",
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        reference_capacity_source="device_configuration",
    )
    discharges = _reference_comparability(
        discharges,
        serial="100000001",
        chemistry="NCA",
        module_count=2,
        reference_events=discharges.assign(device="100000017"),
    )
    transitions = _canonical_transitions(
        pd.DataFrame(
            {
                "event_id": ["event_1"],
                "event_time_utc": [pd.Timestamp("2026-06-03T07:00:00Z")],
                "previous_soh_pct": [98.0],
                "new_soh_pct": [97.0],
                "soh_delta_pp": [-1.0],
                "transition_detection_status": ["exact"],
            }
        ),
        "100000001",
    )

    matches, candidates = _build_matches(transitions, discharges, lookback_hours=72.0)

    assert len(candidates) == 3
    assert matches.iloc[0]["primary_preceding_discharge_event_id"] == "strict"
    assert matches.iloc[0]["nearest_qualifying_discharge_event_id"] == "operational"
    assert matches.iloc[0]["interpretation"].startswith(
        "The deepest meaningful discharge"
    )
    assert candidates["evidence_rank"].nunique() == 3
    assert candidates["proximity_rank"].nunique() == 3


def test_reference_comparability_accepts_legacy_capacity_claim_schema() -> None:
    discharges = _canonical_discharge_events(
        _episodes().iloc[:1],
        serial="100000001",
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        reference_capacity_source="device_configuration",
    )
    reference = discharges.assign(device="100000017").drop(
        columns="capacity_estimate_eligible"
    )

    result = _reference_comparability(
        discharges,
        serial="100000001",
        chemistry="NCA",
        module_count=2,
        reference_events=reference,
    )

    assert result.iloc[0]["comparison_reference_event_id"] == "strict"
    assert result.iloc[0]["reference_comparability"] == "directly_comparable"


def test_complete_inventory_retains_compact_context_without_capacity_claim() -> None:
    compact = pd.DataFrame(
        {
            "discharge_episode_id": ["strict", "compact_partial"],
            "start_time_utc": pd.to_datetime(
                ["2026-06-01T00:00:00Z", "2026-06-05T00:00:00Z"]
            ),
            "completion_time_utc": pd.to_datetime(
                ["2026-06-01T05:00:00Z", "2026-06-05T02:00:00Z"]
            ),
            "start_soc_pct": [100.0, 70.0],
            "end_soc_pct": [0.0, 40.0],
            "soc_drop_pct": [100.0, 30.0],
            "discharged_ah": [47.0, 12.0],
            "episode_coverage_fraction": [1.0, 0.98],
        }
    )
    inventory = _complete_episode_inventory(compact, _episodes().iloc[:1])
    result = _canonical_discharge_events(
        inventory,
        serial="100000001",
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        reference_capacity_source="device_configuration",
    )

    assert len(result) == 2
    compact_row = result.loc[result["discharge_event_id"].eq("compact_partial")].iloc[0]
    assert compact_row["evidence_class"] == "partial_discharge"
    assert compact_row["measurement_resolution"] == "five_minute_candidate"
    assert compact_row["measurement_quality"] == "indeterminate"
    assert not bool(compact_row["capacity_claim_eligible"])
    assert pd.isna(compact_row["soc_span_normalized_soh_estimate_pct"])


def test_phase2_finalizer_persists_contract_and_reuses_matching_cache(
    tmp_path: Path,
) -> None:
    serial = "100000017"
    validation_root = tmp_path / "data_validation"
    recalibration_root = tmp_path / "bms_soh_recalibration"
    physical_catalog_root = tmp_path / "physical_discharge_catalog"
    validation_run = (
        validation_root
        / "chemistry=NCA"
        / f"serial={serial}"
        / "validation_run=20260821T100000Z"
    )
    catalog_run = (
        physical_catalog_root / f"serial={serial}" / "catalog_run=20260821T101000Z"
    )
    recalibration_run = (
        recalibration_root / f"serial={serial}" / "recalibration_run=20260821T102000Z"
    )
    validation_run.mkdir(parents=True)
    catalog_run.mkdir(parents=True)
    recalibration_run.mkdir(parents=True)
    (validation_run / "model_summary.json").write_text(
        json.dumps({"readiness": {"capacity_bms": {"status": "ready"}}}),
        encoding="utf-8",
    )
    (validation_run / "input_manifest.json").write_text(
        json.dumps({"fingerprint_sha256": "stage1-fingerprint"}), encoding="utf-8"
    )
    _canonical_discharge_events(
        _episodes(),
        serial=serial,
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        reference_capacity_source="measured_first_full_cycle",
    ).to_parquet(
        catalog_run / "physical_discharge_cycles.parquet", index=False
    )
    (recalibration_run / "model_summary.json").write_text(
        json.dumps({"detector_version": "v4", "event_rows": 1}), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "event_id": ["event_1"],
            "event_time_utc": [pd.Timestamp("2026-06-03T07:00:00Z")],
            "previous_soh_pct": [98.0],
            "new_soh_pct": [97.0],
            "soh_delta_pp": [-1.0],
            "transition_detection_status": ["exact"],
        }
    ).to_parquet(recalibration_run / "recalibration_events.parquet", index=False)

    result = finalize_phase2(
        serial=serial,
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        reference_capacity_source="device_configuration",
        validation_root=validation_root,
        recalibration_root=recalibration_root,
        physical_catalog_root=physical_catalog_root,
    )

    assert result == recalibration_run
    assert (result / "capacity_discharge_events.parquet").exists()
    assert (result / "bms_soh_transitions.parquet").exists()
    assert (result / "soh_discharge_matches.parquet").exists()
    assert (result / "event_match_candidates.parquet").exists()
    assert (result / "phase2_manifest.json").exists()
    assert not (result / "raw_discharge_trajectories.parquet").exists()
    assert not (result / "downward_transition_trajectory_matches.parquet").exists()
    assert find_cached_phase2_run(
        serial=serial,
        chemistry="NCA",
        module_count=2,
        reference_capacity_ah=FIELD_NOMINAL_AH,
        validation_root=validation_root,
        recalibration_root=recalibration_root,
        physical_catalog_root=physical_catalog_root,
    ) == recalibration_run
