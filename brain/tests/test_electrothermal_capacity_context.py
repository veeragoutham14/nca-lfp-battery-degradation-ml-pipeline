from __future__ import annotations

import numpy as np
import pandas as pd

from src.device_specs import FIELD_MODULE_AH, FIELD_NOMINAL_AH
from src.electrothermal_capacity_context import (
    _reference_capacity_events,
    select_phase3_events,
    summarize_event_frame,
)
from src.electrothermal_contract import expected_hierarchical_signals


def _event() -> pd.Series:
    return pd.Series(
        {
            "discharge_event_id": "discharge_1",
            "evidence_class": "operational_full_discharge",
            "measurement_quality": "high",
            "start_utc": pd.Timestamp("2026-01-01T00:00:00Z"),
            "end_utc": pd.Timestamp("2026-01-01T00:02:00Z"),
        }
    )


def test_phase3_measured_context_preserves_hierarchy_and_temperature_dwell() -> None:
    time = pd.date_range("2026-01-01", periods=121, freq="s", tz="UTC")
    frame = pd.DataFrame(
        {
            "time": time,
            "batt_soc_pct": np.linspace(95, 5, len(time)),
            "batt_soh_pct": 90.0,
            "batt_current_a": -FIELD_MODULE_AH,
            "batt_voltage_v": np.linspace(112, 84, len(time)),
        }
    )
    for signal in expected_hierarchical_signals(2):
        if signal.measurement_type == "cell_voltage_channel":
            frame[signal.source_signal_name] = 3700.0
        else:
            frame[signal.source_signal_name] = 8.0
    frame["batt_batterymodules_mod1_pack1_cellvoltage_mv_0"] = np.linspace(
        3700, 3600, len(frame)
    )
    result, points = summarize_event_frame(
        frame,
        event=_event(),
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )
    assert result["phase3_status"] == "measured_context_ready"
    assert result["minutes_below_15c"] == 2.0
    assert result["recurring_lowest_voltage_group"] == "m01_p01_cv00"
    assert result["rms_c_rate"] == FIELD_MODULE_AH / FIELD_NOMINAL_AH
    assert "cold_discharge_below_10c" in result["phase3_reasons"]
    assert result["soc_20_sample_rows"] > 0
    assert result["soc_10_voltage_spread_p95_mv"] > 0
    assert result["m01_p01_temperature_mean_c"] == 8.0
    assert "m01_p01_derived_voltage_v" in points
    assert "m02_p02_temperature_mean_c" in points
    assert points["event_start_utc"].eq(_event()["start_utc"]).all()
    assert not points.empty


def test_reference_selection_canonicalizes_cycles_and_uses_operational_only_without_stronger_month() -> None:
    frame = pd.DataFrame(
        {
            "discharge_event_id": [
                "full_soc_candidate_000001",
                "discharge_000001",
                "discharge_000002",
                "discharge_000003",
            ],
            "capacity_claim_eligible": True,
            "measurement_resolution": "one_second",
            "evidence_class": [
                "operational_full_discharge",
                "operational_full_discharge",
                "operational_full_discharge",
                "strict_full_discharge",
            ],
            "start_utc": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-02-01"], utc=True
            ),
            "end_utc": pd.to_datetime(
                ["2026-01-01 01:00", "2026-01-01 01:00", "2026-01-02 01:00", "2026-02-01 01:00"], utc=True
            ),
            "measurement_quality": ["high", "high", "medium", "high"],
            "data_coverage_fraction": [1.0, 1.0, 0.95, 1.0],
            "maximum_gap_s": [1.0, 1.0, 2.0, 1.0],
        }
    )
    selected = select_phase3_events(frame, serial="100000017")
    assert selected["discharge_event_id"].tolist() == [
        "full_soc_candidate_000001",
        "discharge_000002",
        "discharge_000003",
    ]
    assert selected["physical_cycle_id"].nunique() == 3


def test_reference_capacity_catalog_preserves_rest_interrupted_exact_cycles(tmp_path) -> None:
    pd.DataFrame(
        {
            "event_discharge_id": [1, 2],
            "start_anchor_time": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
            "end_anchor_time": pd.to_datetime(
                ["2026-01-01 04:00", "2026-01-02 06:00"], utc=True
            ),
            "capacity_ah": [47.3, 47.1],
            "capacity_wh": [9460.0, 9420.0],
            "capacity_measurement_quality": [
                "clean_anchor_100_to_0",
                "long_rest_interrupted",
            ],
            "rest_duration_seconds": [7200.0, 14400.0],
            "num_discharge_events": [1, 2],
        }
    ).to_parquet(tmp_path / "capacity_measurements.parquet", index=False)

    events = _reference_capacity_events(tmp_path)

    assert len(events) == 2
    assert events["evidence_class"].eq("strict_full_discharge").all()
    assert events["soc_start_pct"].eq(100.0).all()
    assert events["soc_end_pct"].eq(0.0).all()
    assert events["measurement_quality"].tolist() == ["high", "medium"]
