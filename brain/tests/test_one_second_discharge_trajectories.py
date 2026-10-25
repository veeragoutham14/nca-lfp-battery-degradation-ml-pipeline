from __future__ import annotations

import numpy as np
import pandas as pd

from src.device_specs import FIELD_NOMINAL_AH
from src.one_second_discharge_trajectories import (
    TrajectoryConfig,
    detect_discharge_trajectories,
    rank_transition_trajectories,
    reconcile_compact_low_soc_endpoints,
)


def _raw(soc: list[float], current: list[float], *, seconds: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=len(soc), freq=f"{seconds}s", tz="UTC"),
            "bms_soc_pct": soc,
            "bms_current_a": current,
            "bms_soh_pct": 95.0,
            "bms_voltage_v": 100.0,
        }
    )


def test_direct_one_second_trajectory_reaches_zero_and_integrates_ah() -> None:
    raw = _raw([100.0, 75.0, 50.0, 25.0, 0.0], [-10.0] * 5, seconds=3600)
    result = detect_discharge_trajectories(
        raw,
        anchor_time=raw.iloc[-1]["time"] + pd.Timedelta(hours=1),
        transition_id="down_1",
        nominal_capacity_ah=40.0,
        config=TrajectoryConfig(maximum_integrated_gap_seconds=3600),
    )

    row = result.iloc[0]
    assert row["evidence_class"] == "strict_100_to_0"
    assert row["trajectory_end_reason"] == "reached_empty_soc"
    assert bool(row["observed_100_to_0"])
    assert bool(row["observed_99_to_1"])
    assert bool(row["observed_95_to_5"])
    assert row["discharged_ah"] == 40.0
    assert row["capacity_soh_pct"] == 100.0


def test_rest_and_small_charge_intrusion_keep_trajectory_connected() -> None:
    raw = _raw(
        [100.0, 80.0, 80.0, 79.5, 50.0, 5.0, 5.0, 0.0],
        [-10.0, -10.0, 0.0, 0.2, -10.0, -10.0, 0.0, -10.0],
        seconds=60,
    )
    result = detect_discharge_trajectories(
        raw,
        anchor_time=raw.iloc[-1]["time"] + pd.Timedelta(hours=1),
        transition_id="down_2",
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        config=TrajectoryConfig(maximum_integrated_gap_seconds=60),
    )

    continuity = result.iloc[0]["continuity"]
    assert "rest_continuation" in continuity
    assert result.iloc[0]["trajectory_end_reason"] == "reached_empty_soc"
    assert result.iloc[0]["rest_interruption_count"] >= 1


def test_meaningful_recharge_terminates_before_later_low_soc() -> None:
    raw = _raw(
        [100.0, 80.0, 82.0, 84.0, 5.0],
        [-10.0, -10.0, 20.0, 20.0, -10.0],
        seconds=3600,
    )
    result = detect_discharge_trajectories(
        raw,
        anchor_time=raw.iloc[-1]["time"] + pd.Timedelta(hours=1),
        transition_id="down_3",
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        config=TrajectoryConfig(maximum_integrated_gap_seconds=3600),
    )

    assert result.iloc[0]["trajectory_end_reason"] == "meaningful_recharge"
    assert result.iloc[0]["soc_end_pct"] > 5.0


def test_ranker_reports_deepest_nearest_and_delay_buckets() -> None:
    transitions = pd.DataFrame(
        {
            "transition_id": ["down"],
            "delta_soh_pp": [-1.0],
            "transition_time_utc": [pd.Timestamp("2026-01-04T00:00:00Z")],
        }
    )
    trajectories = pd.DataFrame(
        {
            "transition_id": ["down", "down"],
            "trajectory_id": ["deep", "near"],
            "soc_span_pp": [97.0, 90.0],
            "delay_to_transition_hours": [30.0, 8.0],
            "evidence_class": ["supporting_99_to_1", "supporting_95_to_5"],
            "soc_start_pct": [99.0, 95.0],
            "soc_end_pct": [2.0, 5.0],
            "trajectory_end_reason": ["reached_low_soc", "reached_low_soc"],
            "continuity": ["continuous_discharge", "rest_continuation"],
            "within_12h": [False, True],
            "within_24h": [False, True],
            "within_48h": [True, True],
            "within_72h": [True, True],
        }
    )

    matches, candidates = rank_transition_trajectories(transitions, trajectories)

    assert matches.iloc[0]["deepest_trajectory_id"] == "deep"
    assert matches.iloc[0]["nearest_trajectory_id"] == "near"
    assert bool(matches.iloc[0]["evidence_within_12h"])
    assert len(candidates) == 2


def test_compact_zero_endpoint_is_checked_against_raw_not_trusted() -> None:
    compact = pd.DataFrame(
        {
            "discharge_episode_id": ["compact_zero"],
            "completion_time_utc": [pd.Timestamp("2026-01-01T00:00:04Z")],
            "start_soc_pct": [10.0],
            "end_soc_pct": [0.0],
        }
    )
    raw = _raw([10.0, 8.0, 6.0, 4.0, 3.0], [-5.0] * 5)

    result = reconcile_compact_low_soc_endpoints(compact, raw, pd.DataFrame())

    assert result.iloc[0]["reconciliation_class"] == "compact_raw_disagreement"
    assert result.iloc[0]["raw_min_soc_pct_near_endpoint"] == 3.0


def test_detector_tolerates_soc_noise_below_recovery_limit() -> None:
    raw = _raw(
        [100.0, 70.0, 71.0, 40.0, 5.0],
        [-10.0] * 5,
        seconds=60,
    )
    result = detect_discharge_trajectories(
        raw,
        anchor_time=raw.iloc[-1]["time"] + pd.Timedelta(hours=1),
        transition_id="noise",
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        config=TrajectoryConfig(maximum_integrated_gap_seconds=60),
    )

    assert result.iloc[0]["trajectory_end_reason"] == "dataset_end"
    assert bool(result.iloc[0]["observed_95_to_5"])
    assert not bool(result.iloc[0]["observed_100_to_0"])
    assert np.isclose(result.iloc[0]["soc_span_pp"], 95.0)


def test_transient_all_zero_row_is_not_a_physical_zero_soc_endpoint() -> None:
    raw = _raw([10.0, 10.0, 0.0, 10.0], [0.0, 0.0, 0.0, 0.0])
    raw.loc[2, ["bms_soh_pct", "bms_voltage_v"]] = 0.0
    compact = pd.DataFrame(
        {
            "discharge_episode_id": ["false_zero"],
            "completion_time_utc": [raw.iloc[2]["time"]],
            "start_soc_pct": [10.0],
            "end_soc_pct": [0.0],
        }
    )

    result = reconcile_compact_low_soc_endpoints(compact, raw, pd.DataFrame())

    assert result.iloc[0]["reconciliation_class"] == "transient_all_zero_source_row"
    assert result.iloc[0]["raw_min_soc_pct_near_endpoint"] == 10.0
    assert result.iloc[0]["invalid_all_zero_rows_near_endpoint"] == 1
