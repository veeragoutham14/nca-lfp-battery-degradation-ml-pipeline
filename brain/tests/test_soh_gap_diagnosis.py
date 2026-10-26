from __future__ import annotations

import pandas as pd

from src.device_specs import FIELD_NOMINAL_AH
from src.soh_gap_diagnosis import (
    Phase4Config,
    _condition_distance,
    build_reference_curve,
    reconcile_candidates,
)


def test_reference_curve_uses_measured_capacity_and_observed_origin() -> None:
    frame = pd.DataFrame(
        {
            "capacity_measurement_duration_valid": [True] * 5,
            "capacity_ah": [47.3, 47.2, 47.1, 47.0, 46.9],
            "cumulative_all_discharge_ah": [FIELD_NOMINAL_AH, 1064.0, 2064.0, 3064.0, 4064.0],
            "end_anchor_time": pd.date_range("2026-01-01", periods=5, tz="UTC"),
        }
    )
    curve = build_reference_curve(frame, reference_nameplate_ah=FIELD_NOMINAL_AH, minimum_cycles=5)
    assert curve["source_is_bms_soh"] is False
    assert curve["loss_slope_ah_per_1000ah"] > 0
    assert curve["first_measured_capacity_ah"] == 47.3
    assert "pre-observation throughput is unknown" in curve["origin_semantics"]


def test_reconciliation_preserves_legacy_and_canonical_counts() -> None:
    canonical = pd.DataFrame(
        {
            "physical_cycle_id": ["p1", "p2"],
            "capacity_claim_eligible": [True, True],
            "start_utc": pd.to_datetime(["2026-01-01T00:00Z", "2026-01-02T00:00Z"]),
            "end_utc": pd.to_datetime(["2026-01-01T12:00Z", "2026-01-02T12:00Z"]),
        }
    )
    legacy = pd.DataFrame(
        {
            "discharge_event_id": ["l1", "l2", "l3"],
            "start_utc": pd.to_datetime(
                ["2026-01-01T01:00Z", "2026-01-01T03:00Z", "2026-01-03T00:00Z"]
            ),
            "end_utc": pd.to_datetime(
                ["2026-01-01T10:00Z", "2026-01-01T08:00Z", "2026-01-03T01:00Z"]
            ),
        }
    )
    result = reconcile_candidates(canonical, legacy)
    assert result["legacy_candidate_id"].notna().sum() == 3
    assert (result["reconciliation_status"] == "multiple_legacy_candidates_one_physical_cycle").sum() == 2
    assert (result["reconciliation_status"] == "canonical_cycle_without_legacy_candidate").sum() == 1


def test_reference_matching_ignores_voltage_and_weakest_group() -> None:
    base = {
        "soc_start_pct": 95.0,
        "soc_end_pct": 5.0,
        "temperature_mean_c": 20.0,
        "rms_c_rate": 0.2,
        "rest_duration_s": 3600.0,
        "charge_intrusion_ah": 0.0,
        "data_coverage_fraction": 0.99,
    }
    field = pd.Series(base | {"voltage_group_spread_p95_mv": 200.0, "recurring_lowest_voltage_group": "g1"})
    reference = pd.Series(base | {"voltage_group_spread_p95_mv": 1.0, "recurring_lowest_voltage_group": "g9"})
    _, level, reasons = _condition_distance(field, reference, Phase4Config())
    assert level == "high"
    assert reasons == []


def test_cli_passes_the_reference_gate_to_phase4(monkeypatch, tmp_path) -> None:
    import pytest

    from src import soh_gap_diagnosis as module

    captured = {}

    def fake_run_phase4(**kwargs):
        captured.update(kwargs)
        return tmp_path

    monkeypatch.setattr(module, "run_phase4", fake_run_phase4)
    base = [
        "--serial", "100000099", "--chemistry", "NCA", "--module-count", "2",
        "--field-nameplate-capacity-ah", "64", "--reference-nameplate-capacity-ah", "56",
        "--processed-root", str(tmp_path),
    ]
    assert module.main(base) == 0
    assert captured["config"].minimum_reference_strict_cycles == 500
    assert module.main([*base, "--minimum-reference-strict-cycles", "6"]) == 0
    assert captured["config"].minimum_reference_strict_cycles == 6
    with pytest.raises(SystemExit):
        module.main([*base, "--minimum-reference-strict-cycles", "1"])
