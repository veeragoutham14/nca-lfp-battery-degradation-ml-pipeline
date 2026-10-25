from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.device_specs import FIELD_NOMINAL_AH
from src.seasonal_charge_analysis import (
    build_charge_episodes,
    build_daily_stress,
    build_monthly_profiles,
    build_monthly_one_second_current_stress,
    build_transition_context,
    prepare_windows,
)


def _compact_windows(days: int = 31) -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-01", periods=days * 288, freq="5min", tz="UTC")
    position = pd.Series(range(len(timestamps))) % 288
    charging = position.between(6, 11)
    discharging = position.between(120, 125)
    current = pd.Series(0.0, index=range(len(timestamps)))
    current.loc[charging] = 10.0
    current.loc[discharging] = -8.0
    soc = pd.Series(50.0, index=range(len(timestamps)))
    for day in range(days):
        start = day * 288
        soc.iloc[start + 6 : start + 12] = [20, 30, 40, 50, 60, 70]
    return pd.DataFrame(
        {
            "window_start_utc": timestamps,
            "serial": 100000001,
            "bms_soc_mean_pct": soc,
            "bms_soc_min_pct": soc - 1,
            "bms_soc_max_pct": soc + 1,
            "bms_soh_mean_pct": 98.0,
            "bms_current_mean_a": current,
            "bms_current_min_a": current,
            "bms_current_max_a": current,
            "bms_current_max_abs_a": current.abs(),
            "bms_stack_voltage_mean_v": 220.0,
            "bms_power_mean_w": current * 220.0,
            "coverage_fraction": 1.0,
            "system_voltage_spread_mv": 12.0,
            "system_temperature_spread_c": 2.0,
            "system_temperature_mean_c": 25.0,
            "system_temperature_min_c": 24.0,
            "system_temperature_max_c": 27.0,
            "stable_pack_temperature_mean_c": 25.0,
            "stable_pack_temperature_max_c": 27.0,
            "stable_temperature_sensor_count": 8,
        }
    )


def test_charge_episode_uses_positive_current_and_observed_time() -> None:
    windows = prepare_windows(
        _compact_windows(days=1),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )

    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=5,
        episode_gap_minutes=15,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_soc_gain_pct=20.0,
    )

    assert len(episodes) == 1
    episode = episodes.iloc[0]
    assert episode["charged_ah"] == pytest.approx(5.0)
    assert episode["soc_gain_pct"] == pytest.approx(50.0)
    assert episode["partial_charge_capacity_estimate_ah"] == pytest.approx(10.0)
    assert episode["partial_capacity_estimate_is_weak_label"]
    assert episode["start_time_utc"].tzname() == "UTC"


def test_charge_episode_gap_means_noncharging_interruption_duration() -> None:
    windows = prepare_windows(
        _compact_windows(days=1),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )
    windows["bms_current_mean_a"] = 0.0
    windows["operating_state"] = "rest"
    windows["charge_ah"] = 0.0
    charging_indices = [0, 4, 9]
    windows.loc[charging_indices, "bms_current_mean_a"] = 10.0
    windows.loc[charging_indices, "operating_state"] = "charging"
    windows.loc[charging_indices, "charge_ah"] = 10.0 * 5.0 / 60.0

    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=5,
        episode_gap_minutes=15,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_soc_gain_pct=20.0,
    )

    assert len(episodes) == 2
    assert episodes.iloc[0]["charging_window_count"] == 2
    assert episodes.iloc[1]["charging_window_count"] == 1


def test_daily_and_monthly_profiles_are_coverage_gated_and_seasonal() -> None:
    windows = prepare_windows(
        _compact_windows(),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )
    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=5,
        episode_gap_minutes=15,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_soc_gain_pct=20.0,
    )
    daily = build_daily_stress(
        windows,
        episodes,
        pd.DataFrame(),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        high_current_c_rate_threshold=0.5,
        minimum_daily_coverage=0.8,
    )
    monthly = build_monthly_profiles(
        daily,
        episodes,
        minimum_observed_days=14,
        minimum_month_coverage=0.8,
    )

    assert len(daily) == 31
    assert daily["daily_quality_valid"].all()
    assert daily.iloc[0]["charge_ah"] == pytest.approx(5.0)
    assert daily.iloc[0]["discharge_ah"] == pytest.approx(4.0)
    assert list(monthly["season"]) == ["summer", "summer"]
    assert monthly.iloc[0]["seasonal_profile_valid"]
    assert not monthly.iloc[1]["seasonal_profile_valid"]
    assert monthly.iloc[0]["charge_episode_count"] == 30
    full_soc_hours = monthly.iloc[0][
        [
            "soc_0_5_hours",
            "soc_5_20_hours",
            "soc_20_40_hours",
            "soc_40_60_hours",
            "soc_60_80_hours",
            "soc_80_95_hours",
            "soc_95_100_hours",
            "soc_unknown_hours",
        ]
    ].sum()
    assert full_soc_hours == pytest.approx(30 * 24)
    assert monthly.iloc[0]["soc_unknown_hours"] == pytest.approx(0.0)
    assert monthly.iloc[0]["soc_known_coverage_fraction"] == pytest.approx(1.0)
    assert monthly.iloc[0]["c1_hours"] > 0
    assert monthly.iloc[0]["charging_above_95_soc_hours"] == pytest.approx(0.0)


def test_full_soc_profile_counts_missing_window_coverage_as_unknown() -> None:
    source = _compact_windows(days=1)
    source.loc[0, "coverage_fraction"] = 0.5
    windows = prepare_windows(
        source,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )
    daily = build_daily_stress(
        windows,
        pd.DataFrame(),
        pd.DataFrame(),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        high_current_c_rate_threshold=0.5,
        minimum_daily_coverage=0.8,
    )
    monthly = build_monthly_profiles(
        daily,
        pd.DataFrame(),
        minimum_observed_days=1,
        minimum_month_coverage=0.8,
    )

    assert monthly.iloc[0]["soc_unknown_hours"] == pytest.approx(2.5 / 60.0)
    assert monthly.iloc[0]["soc_known_hours"] == pytest.approx(24.0 - 2.5 / 60.0)


def test_soh_transition_context_keeps_pre_and_post_evidence_separate() -> None:
    windows = prepare_windows(
        _compact_windows(days=20),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )
    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=5,
        episode_gap_minutes=15,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_soc_gain_pct=20.0,
    )
    event_time = pd.Timestamp("2026-06-11T00:00:00Z")
    events = pd.DataFrame(
        {
            "event_time": [event_time],
            "event_type": ["step_drop"],
            "previous_soh_pct": [98.0],
            "new_soh_pct": [96.0],
            "soh_delta_pct": [-2.0],
            "coverage_fraction": [1.0],
            "cumulative_all_discharge_ah": [10_000.0],
            "max_gap_seconds": [2.0],
        }
    )
    daily = build_daily_stress(
        windows,
        episodes,
        events,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        high_current_c_rate_threshold=0.5,
        minimum_daily_coverage=0.8,
    )

    context = build_transition_context(
        daily,
        episodes,
        events,
        window_days=7,
        minimum_daily_coverage=0.8,
    )

    assert len(context) == 1
    row = context.iloc[0]
    assert row["event_time_utc"] == event_time
    assert row["pre_observed_days"] == 7
    assert row["post_observed_days"] == 7
    assert row["context_quality_ready"]
    assert row["interpretation"] == "temporal_association_only_not_causal"


def test_transition_context_uses_authoritative_discharge_evidence() -> None:
    windows = prepare_windows(
        _compact_windows(days=20),
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )
    event_time = pd.Timestamp("2026-06-11T00:00:00Z")
    audit_events = pd.DataFrame(
        {
            "event_time": [event_time],
            "event_type": ["step_drop"],
            "previous_soh_pct": [98.0],
            "new_soh_pct": [96.0],
            "soh_delta_pct": [-2.0],
        }
    )
    daily = build_daily_stress(
        windows,
        pd.DataFrame(),
        audit_events,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        high_current_c_rate_threshold=0.5,
        minimum_daily_coverage=0.8,
    )
    recalibration = pd.DataFrame(
        {
            "event_id": ["recalibration_00001"],
            "event_time_utc": [event_time],
            "daily_event_anchor_utc": [event_time],
            "previous_soh_pct": [98.0],
            "new_soh_pct": [96.0],
            "complete_discharge_within_lookback": [True],
            "hours_since_complete_discharge": [2.5],
            "raw_start_soc_pct": [100.0],
            "raw_end_soc_pct": [2.0],
            "discharged_ah_1s": [46.8],
            "raw_coverage_fraction": [0.99],
        }
    )

    context = build_transition_context(
        daily,
        pd.DataFrame(),
        audit_events,
        window_days=7,
        minimum_daily_coverage=0.8,
        recalibration_events=recalibration,
    )

    row = context.iloc[0]
    assert row["authoritative_recalibration_match"]
    assert row["recalibration_complete_discharge_within_lookback"]
    assert row["recalibration_discharged_ah_1s"] == pytest.approx(46.8)


def test_raw_current_stress_is_monthly_and_sign_aware(tmp_path) -> None:
    raw_dir = tmp_path / "raw" / "serial=100000001"
    raw_dir.mkdir(parents=True)
    times = pd.date_range("2026-06-01", periods=121, freq="1s", tz="UTC")
    high_current = FIELD_NOMINAL_AH * 0.6
    current = [0.0] * 20 + [-high_current] * 40 + [high_current] * 40 + [0.0] * 21
    pq.write_table(
        pa.table({"time": times, "batt_current_a": current}),
        raw_dir / "part-00000.parquet",
    )

    monthly = build_monthly_one_second_current_stress(
        tmp_path / "raw",
        serial="100000001",
        chemistry="NCA",
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
        high_current_c_rate_threshold=0.5,
    )

    row = monthly.iloc[0]
    assert row["month_start_utc"] == pd.Timestamp("2026-06-01T00:00:00Z")
    assert row["raw_charge_ah"] > 0
    assert row["raw_discharge_ah"] > 0
    assert row["raw_current_reversal_count"] == 1
    assert row["raw_high_current_episode_count"] == 2
    assert row["raw_c3_hours"] > 0
