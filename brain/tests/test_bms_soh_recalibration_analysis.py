from __future__ import annotations

import pandas as pd

from src.bms_soh_recalibration_analysis import (
    _persistent_transition,
    build_calibration_discharge_episodes,
    build_evidence_comparison,
    build_full_soc_capacity_candidates,
    build_interpretable_model,
    build_one_second_capacity_validation,
    build_recalibration_events,
    detect_raw_discharge_evidence,
)
from src.device_specs import FIELD_MODULE_AH, FIELD_NOMINAL_AH
from src.seasonal_charge_analysis import prepare_windows


def test_raw_detector_uses_full_anchor_window_retains_low_soc_tail_and_tied_minimum() -> None:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=11, freq="1min")
    raw = pd.DataFrame({
        "time": times,
        "bms_soc_pct": [100, 98, 90, 60, 5, 5, 5, 2, 0, 0, 0],
        "bms_current_a": [-10, -10, -10, -10, -10, 0, 0, 0, -10, 0, 0],
        "bms_soh_pct": 98.0,
    })
    anchors = pd.DataFrame({"event_id": ["event_1"], "event_time_utc": [times[-1]]})

    result = detect_raw_discharge_evidence(
        raw, anchors, anchor_column="event_time_utc", lookback_hours=72,
        start_soc_pct=95.0, end_soc_pct=5.0, maximum_integrated_gap_seconds=5,
        rest_current_threshold_a=0.5,
    )

    row = result.iloc[0]
    assert row["raw_discharge_detection_status"] == "resolved"
    assert row["first_low_band_time_utc"] == times[4]
    assert row["minimum_soc_time_utc"] == times[8]
    assert row["raw_end_time_utc"] == times[8]
    assert row["terminal_low_soc_tail_minutes"] == 6.0
    assert row["post_minimum_low_soc_tail_minutes"] == 2.0
    assert row["terminal_low_soc_rest_tail_minutes"] == 0.0
    assert row["terminal_low_soc_potential_rest_tail_minutes"] == 1.0
    assert row["rest_minutes_inside_cycle"] == 0.0
    assert row["potential_gap_spanning_rest_minutes"] == 2.0
    assert not bool(row["raw_discharge_qualified"])


def test_raw_detector_selects_latest_candidate_and_marks_ambiguity_for_any_anchor_type() -> None:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=12, freq="1min")
    raw = pd.DataFrame({
        "time": times,
        "bms_soc_pct": [100, 50, 0, 0, 20, 100, 80, 5, 0, 0, 0, 0],
        "bms_current_a": [-10.0] * len(times),
        "bms_soh_pct": 98.0,
    })
    anchors = pd.DataFrame({"control_id": ["control_1"], "anchor_time_utc": [times[-1]]})

    result = detect_raw_discharge_evidence(
        raw, anchors, anchor_column="anchor_time_utc", lookback_hours=72,
        start_soc_pct=95.0, end_soc_pct=5.0, maximum_integrated_gap_seconds=120,
        rest_current_threshold_a=0.5,
    )

    row = result.iloc[0]
    assert row["raw_discharge_candidate_count"] == 2
    assert bool(row["raw_discharge_ambiguous"])
    assert row["raw_discharge_selection"] == "latest_completed_pre_anchor"
    assert row["raw_start_time_utc"] == times[5]
    assert row["raw_end_time_utc"] == times[8]
    assert bool(row["raw_discharge_qualified"])
    assert row["raw_discharge_dominance_fraction"] == 1.0


def test_raw_detector_keeps_gap_uncertainty_and_charge_intrusion_separate() -> None:
    times = pd.to_datetime(
        [
            "2026-06-01T00:00:00Z",
            "2026-06-01T00:00:01Z",
            "2026-06-01T00:01:01Z",
            "2026-06-01T00:01:02Z",
        ]
    )
    raw = pd.DataFrame({
        "time": times, "bms_soc_pct": [100, 50, 5, 0],
        "bms_current_a": [-10.0, 5.0, -10.0, -10.0], "bms_soh_pct": 98.0,
    })
    anchors = pd.DataFrame({"event_time_utc": [times[-1]]})
    row = detect_raw_discharge_evidence(
        raw, anchors, anchor_column="event_time_utc", lookback_hours=72,
        start_soc_pct=95.0, end_soc_pct=5.0, maximum_integrated_gap_seconds=5,
        rest_current_threshold_a=0.5,
    ).iloc[0]
    assert row["unobserved_discharge_ah_estimate"] > 0
    assert row["missing_ah_assumption_current_a"] == 10.0
    assert row["missing_ah_estimation_method"] == (
        "unobserved_duration_at_maximum_observed_discharge_current"
    )
    assert row["charge_intrusion_ah_1s"] > 0


def test_raw_detector_rejects_positive_only_and_reverse_soc_windows() -> None:
    times = pd.date_range("2026-06-01T00:00:00Z", periods=4, freq="1min")
    positive_only = pd.DataFrame(
        {
            "time": times,
            "bms_soc_pct": [100, 50, 5, 0],
            "bms_current_a": [10.0] * 4,
            "bms_soh_pct": 98.0,
        }
    )
    reversed_soc = positive_only.assign(bms_soc_pct=[0, 5, 50, 100])
    anchors = pd.DataFrame({"event_time_utc": [times[-1]]})
    kwargs = {
        "anchor_column": "event_time_utc",
        "lookback_hours": 72,
        "start_soc_pct": 95.0,
        "end_soc_pct": 5.0,
        "maximum_integrated_gap_seconds": 120,
        "rest_current_threshold_a": 0.5,
    }

    positive = detect_raw_discharge_evidence(positive_only, anchors, **kwargs).iloc[0]
    reverse = detect_raw_discharge_evidence(reversed_soc, anchors, **kwargs).iloc[0]

    assert positive["raw_discharge_detection_status"] == "soc_traversal_without_negative_discharge"
    assert positive["raw_discharge_candidate_count"] == 0
    assert reverse["raw_discharge_detection_status"] == "not_observed"


def test_raw_detector_splits_mixed_sign_interval_without_losing_charge_intrusion() -> None:
    times = pd.to_datetime(["2026-06-01T00:00:00Z", "2026-06-01T00:01:00Z"])
    raw = pd.DataFrame(
        {
            "time": times,
            "bms_soc_pct": [100.0, 0.0],
            "bms_current_a": [-10.0, 10.0],
            "bms_soh_pct": 98.0,
        }
    )
    row = detect_raw_discharge_evidence(
        raw,
        pd.DataFrame({"event_time_utc": [times[-1]]}),
        anchor_column="event_time_utc",
        lookback_hours=72,
        start_soc_pct=95.0,
        end_soc_pct=5.0,
        maximum_integrated_gap_seconds=120,
        rest_current_threshold_a=0.5,
    ).iloc[0]

    assert row["discharged_ah_1s"] > 0
    assert row["charge_intrusion_ah_1s"] > 0


def test_raw_detector_labels_gap_spanning_rest_as_potential() -> None:
    times = pd.to_datetime(
        ["2026-06-01T00:00:00Z", "2026-06-01T00:10:00Z", "2026-06-01T00:20:00Z"]
    )
    raw = pd.DataFrame(
        {
            "time": times,
            "bms_soc_pct": [100.0, 5.0, 0.0],
            "bms_current_a": [-10.0, 0.0, 0.0],
            "bms_soh_pct": 98.0,
        }
    )
    row = detect_raw_discharge_evidence(
        raw,
        pd.DataFrame({"event_time_utc": [times[-1]]}),
        anchor_column="event_time_utc",
        lookback_hours=72,
        start_soc_pct=95.0,
        end_soc_pct=5.0,
        maximum_integrated_gap_seconds=5,
        rest_current_threshold_a=0.5,
    ).iloc[0]

    assert row["raw_coverage_fraction"] == 0.0
    assert row["rest_minutes_inside_cycle"] == 0.0
    assert row["potential_gap_spanning_rest_minutes"] == 10.0


def _windows() -> pd.DataFrame:
    timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=288 * 4, freq="5min")
    frame = pd.DataFrame(
        {
            "window_start_utc": timestamps,
            "serial": 100000004,
            "bms_soc_mean_pct": 60.0,
            "bms_soc_min_pct": 59.0,
            "bms_soc_max_pct": 61.0,
            "bms_soh_mean_pct": 98.0,
            "bms_current_mean_a": 0.0,
            "bms_current_min_a": 0.0,
            "bms_current_max_a": 0.0,
            "bms_current_max_abs_a": 0.0,
            "bms_stack_voltage_mean_v": 220.0,
            "bms_power_mean_w": 0.0,
            "coverage_fraction": 1.0,
            "system_voltage_spread_mv": 16.0,
            "stable_pack_temperature_mean_c": 27.0,
            "stable_pack_temperature_max_c": 29.0,
        }
    )
    # A bounded charge ending at 100% followed by a continuous rest period.
    charge = frame.index[(frame["window_start_utc"] >= pd.Timestamp("2026-06-02T16:00:00Z")) & (frame["window_start_utc"] < pd.Timestamp("2026-06-02T18:00:00Z"))]
    frame.loc[charge, "bms_current_mean_a"] = 10.0
    frame.loc[charge, "bms_current_min_a"] = 10.0
    frame.loc[charge, "bms_current_max_a"] = 10.0
    frame.loc[charge, "bms_current_max_abs_a"] = 10.0
    frame.loc[charge, "bms_power_mean_w"] = 2200.0
    frame.loc[charge, "bms_soc_mean_pct"] = pd.Series(range(len(charge)), index=charge) * (40 / max(1, len(charge) - 1)) + 60
    frame.loc[charge, "bms_soc_min_pct"] = frame.loc[charge, "bms_soc_mean_pct"] - 1
    frame.loc[charge, "bms_soc_max_pct"] = frame.loc[charge, "bms_soc_mean_pct"] + 1
    return prepare_windows(
        frame,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )


def test_recalibration_event_classifies_near_full_charge_and_rest() -> None:
    events = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2026-06-02T22:00:00Z")],
            "event_type": ["step_drop"],
            "previous_soh_pct": [98.0],
            "new_soh_pct": [96.0],
            "soh_delta_pct": [-2.0],
            "coverage_fraction": [1.0],
            "max_gap_seconds": [5.0],
            "cumulative_all_discharge_ah": [1200.0],
            "transition_detection_status": ["exact"],
            "event_time_resolution": ["persistent_raw_change_point"],
        }
    )

    result = build_recalibration_events(
        _windows(),
        events,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_window_coverage=0.8,
        minimum_context_coverage=0.6,
        context_lookback_hours=72,
        near_full_soc_pct=99.0,
        low_soc_pct=10.0,
        minimum_rest_minutes=120,
        data_gap_risk_minutes=60,
        recurrence_window_days=30,
    )

    row = result.iloc[0]
    assert row["event_time_utc"].tzname() == "UTC"
    assert row["evidence_classification"] == "meaningful_charge_completion_and_rest_context"
    assert row["context_match_strength"] == "strong"
    assert row["classification_confidence"] == "high"
    assert row["near_full_charge_within_lookback"]
    assert row["rest_before_event_minutes"] == 240.0


def test_pack_current_sign_uses_positive_charge_and_negative_discharge() -> None:
    frame = _windows().iloc[:2].copy()
    frame["bms_current_mean_a"] = [10.0, -8.0]

    result = prepare_windows(
        frame,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )

    assert result["operating_state"].tolist() == ["charging", "discharging"]
    assert result.iloc[0]["charge_ah"] > 0
    assert result.iloc[0]["discharge_ah"] == 0
    assert result.iloc[1]["charge_ah"] == 0
    assert result.iloc[1]["discharge_ah"] > 0


def test_calibration_discharge_requires_ordered_high_to_low_negative_current() -> None:
    timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=21, freq="5min")
    soc = pd.Series([100.0 - 5.0 * index for index in range(len(timestamps))])
    frame = pd.DataFrame(
        {
            "window_start_utc": timestamps,
            "bms_current_mean_a": -FIELD_MODULE_AH,
            "bms_stack_voltage_mean_v": 220.0,
            "bms_power_mean_w": -FIELD_MODULE_AH * 220.0,
            "coverage_fraction": 1.0,
            "bms_soc_mean_pct": soc,
            "bms_soc_min_pct": soc,
            "bms_soc_max_pct": soc,
        }
    )
    windows = prepare_windows(
        frame,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )

    episodes = build_calibration_discharge_episodes(
        windows,
        aggregation_minutes=5,
        episode_gap_minutes=30,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_window_coverage=0.8,
        high_soc_start_pct=95.0,
        low_soc_end_pct=5.0,
        minimum_soc_drop_pct=90.0,
        minimum_discharge_dominance_fraction=0.8,
    )

    assert len(episodes) == 1
    assert bool(episodes.iloc[0]["complete_high_to_low_discharge"])
    assert episodes.iloc[0]["start_soc_pct"] == 100.0
    assert episodes.iloc[0]["end_soc_pct"] == 0.0
    assert episodes.iloc[0]["discharge_dominance_fraction"] == 1.0


def test_full_soc_capacity_candidate_can_span_rest_without_charging() -> None:
    timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=25, freq="5min")
    soc = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 50, 50, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1, 0], dtype=float)
    current = pd.Series([-FIELD_MODULE_AH] * len(timestamps))
    current.iloc[11:14] = 0.0
    frame = pd.DataFrame(
        {
            "window_start_utc": timestamps,
            "bms_current_mean_a": current,
            "bms_stack_voltage_mean_v": 220.0,
            "bms_power_mean_w": current * 220.0,
            "coverage_fraction": 1.0,
            "bms_soc_mean_pct": soc,
            "bms_soc_min_pct": soc,
            "bms_soc_max_pct": soc,
        }
    )
    windows = prepare_windows(
        frame,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )

    candidates = build_full_soc_capacity_candidates(
        windows,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        start_soc_pct=99.0,
        end_soc_pct=1.0,
        maximum_elapsed_hours=72,
        minimum_window_coverage=0.8,
    )

    assert len(candidates) == 1
    assert candidates.iloc[0]["candidate_source"] == "compact_99_to_1_with_rest"
    assert candidates.iloc[0]["start_soc_pct"] == 100.0
    assert candidates.iloc[0]["end_soc_pct"] <= 1.0


def test_one_second_capacity_validation_integrates_current_and_links_event() -> None:
    timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=3601, freq="1s")
    raw = pd.DataFrame(
        {
            "time": timestamps,
            "bms_soc_pct": pd.Series(range(3600, -1, -1), dtype=float) / 36.0,
            "bms_soh_pct": 98.0,
            "bms_current_a": -FIELD_NOMINAL_AH,
            "bms_voltage_v": 200.0,
        }
    )
    episodes = pd.DataFrame(
        {
            "discharge_episode_id": ["discharge_000001"],
            "start_time_utc": [timestamps[0]],
            "completion_time_utc": [timestamps[-1]],
            "end_time_utc": [timestamps[-1] + pd.Timedelta(minutes=5)],
            "complete_high_to_low_discharge": [True],
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["recalibration_00001"],
            "event_time_utc": [timestamps[-1] + pd.Timedelta(minutes=1)],
            "complete_discharge_within_lookback": [True],
            "complete_discharge_end_time_utc": [timestamps[-1]],
            "new_soh_pct": [99.0],
            "soh_delta_pp": [-1.0],
        }
    )

    result = build_one_second_capacity_validation(
        raw,
        episodes,
        events,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        aggregation_minutes=5,
        candidate_start_soc_pct=95.0,
        candidate_end_soc_pct=5.0,
        strict_start_soc_pct=99.0,
        strict_end_soc_pct=1.0,
        strict_minimum_soc_drop_pct=98.0,
        minimum_coverage_fraction=0.98,
        maximum_unobserved_ah=0.25,
        maximum_charge_intrusion_ah=0.1,
        maximum_integrated_gap_seconds=5,
        rest_current_threshold_a=0.5,
        continuous_maximum_rest_minutes=5.0,
    )

    row = result.iloc[0]
    assert row["capacity_validation_status"] == "validation_grade"
    assert row["cycle_continuity_class"] == "continuous"
    assert abs(row["discharged_ah_1s"] - FIELD_NOMINAL_AH) < 1e-9
    assert abs(row["discharged_wh_1s"] - FIELD_NOMINAL_AH * 200.0) < 1e-9
    assert abs(row["rms_discharge_current_a"] - FIELD_NOMINAL_AH) < 1e-9
    assert abs(row["peak_discharge_current_a"] - FIELD_NOMINAL_AH) < 1e-9
    assert abs(row["direct_capacity_soh_pct"] - 100.0) < 1e-9
    assert row["linked_soh_event_id"] == "recalibration_00001"
    assert abs(row["bms_completion_minus_capacity_soh_pp"] + 2.0) < 1e-9
    assert abs(row["bms_after_event_minus_capacity_soh_pp"] + 1.0) < 1e-9


def test_one_second_capacity_validation_distinguishes_interrupted_cycle() -> None:
    timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=4201, freq="1s")
    active = pd.Series(True, index=range(len(timestamps)))
    active.iloc[1801:2401] = False
    active_progress = active.astype(int).cumsum() - 1
    raw = pd.DataFrame(
        {
            "time": timestamps,
            "bms_soc_pct": 100.0 - active_progress * (100.0 / 3600.0),
            "bms_soh_pct": 98.0,
            "bms_current_a": active.map({True: -FIELD_NOMINAL_AH, False: 0.0}),
        }
    )
    episodes = pd.DataFrame(
        {
            "discharge_episode_id": ["discharge_000001"],
            "start_time_utc": [timestamps[0]],
            "completion_time_utc": [timestamps[-1]],
            "end_time_utc": [timestamps[-1] + pd.Timedelta(minutes=5)],
            "complete_high_to_low_discharge": [True],
        }
    )

    result = build_one_second_capacity_validation(
        raw,
        episodes,
        pd.DataFrame(),
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        aggregation_minutes=5,
        candidate_start_soc_pct=95.0,
        candidate_end_soc_pct=5.0,
        strict_start_soc_pct=99.0,
        strict_end_soc_pct=1.0,
        strict_minimum_soc_drop_pct=98.0,
        minimum_coverage_fraction=0.98,
        maximum_unobserved_ah=0.25,
        maximum_charge_intrusion_ah=0.1,
        maximum_integrated_gap_seconds=5,
        rest_current_threshold_a=0.5,
        continuous_maximum_rest_minutes=5.0,
    )

    row = result.iloc[0]
    assert row["capacity_validation_status"] == "validation_grade"
    assert row["cycle_continuity_class"] == "interrupted"
    assert row["rest_minutes_inside_cycle"] > 9.0
    assert row["longest_rest_minutes"] > 9.0


def test_recalibration_event_prioritizes_complete_discharge_hypothesis() -> None:
    timestamps = pd.date_range("2026-06-01T00:00:00Z", periods=21, freq="5min")
    soc = pd.Series([100.0 - 5.0 * index for index in range(len(timestamps))])
    frame = pd.DataFrame(
        {
            "window_start_utc": timestamps,
            "bms_current_mean_a": -FIELD_MODULE_AH,
            "bms_stack_voltage_mean_v": 220.0,
            "bms_power_mean_w": -FIELD_MODULE_AH * 220.0,
            "coverage_fraction": 1.0,
            "bms_soc_mean_pct": soc,
            "bms_soc_min_pct": soc,
            "bms_soc_max_pct": soc,
            "system_voltage_spread_mv": 16.0,
            "stable_pack_temperature_mean_c": 27.0,
            "stable_pack_temperature_max_c": 29.0,
        }
    )
    windows = prepare_windows(
        frame,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        rest_current_threshold_a=0.5,
    )
    event_time = timestamps[-1] + pd.Timedelta(minutes=10)
    events = pd.DataFrame(
        {
            "event_time": [event_time],
            "event_type": ["step_drop"],
            "previous_soh_pct": [98.0],
            "new_soh_pct": [97.0],
            "soh_delta_pct": [-1.0],
            "coverage_fraction": [1.0],
            "max_gap_seconds": [1.0],
            "cumulative_all_discharge_ah": [1200.0],
            "transition_detection_status": ["exact"],
            "event_time_resolution": ["persistent_raw_change_point"],
        }
    )

    result = build_recalibration_events(
        windows,
        events,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_window_coverage=0.8,
        minimum_context_coverage=0.01,
        context_lookback_hours=72,
        near_full_soc_pct=99.0,
        low_soc_pct=10.0,
        minimum_rest_minutes=120,
        data_gap_risk_minutes=60,
        recurrence_window_days=30,
    )

    row = result.iloc[0]
    assert row["evidence_classification"] == "complete_high_to_low_discharge_context"
    assert bool(row["complete_discharge_within_1h"])
    assert row["complete_discharge_start_soc_pct"] == 100.0
    assert row["complete_discharge_end_soc_pct"] == 0.0
    assert "complete_high_to_low_discharge_before_update" in row["evidence_tags"]


def test_recalibration_event_does_not_hide_a_gap() -> None:
    events = pd.DataFrame(
        {
            "event_time": [pd.Timestamp("2026-06-03T00:00:00Z")],
            "event_type": ["change_after_gap"],
            "previous_soh_pct": [96.0],
            "new_soh_pct": [93.0],
            "soh_delta_pct": [-3.0],
            "coverage_fraction": [1.0],
            "max_gap_seconds": [7200.0],
            "cumulative_all_discharge_ah": [1300.0],
        }
    )

    result = build_recalibration_events(
        _windows(),
        events,
        aggregation_minutes=5,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        minimum_window_coverage=0.8,
        minimum_context_coverage=0.8,
        context_lookback_hours=72,
        near_full_soc_pct=99.0,
        low_soc_pct=10.0,
        minimum_rest_minutes=120,
        data_gap_risk_minutes=60,
        recurrence_window_days=30,
    )

    assert result.iloc[0]["evidence_classification"] == "data_gap_or_estimator_reset_risk"


def test_persistent_transition_resolves_exact_raw_change_point() -> None:
    old_times = pd.date_range("2026-06-02T10:00:00Z", periods=3601, freq="1s")
    new_times = pd.date_range("2026-06-02T11:00:01Z", periods=6 * 3600 + 1, freq="1s")
    raw = pd.DataFrame(
        {
            "time": old_times.append(new_times),
            "bms_soh_pct": [98.0] * len(old_times) + [97.0] * len(new_times),
        }
    )

    result = _persistent_transition(
        raw,
        daily_anchor=pd.Timestamp("2026-06-02T00:00:00Z"),
        previous_soh_pct=98.0,
        new_soh_pct=97.0,
        search_before_hours=6,
        search_after_hours=30,
        persistence_hours=6,
        minimum_new_level_fraction=0.95,
        previous_support_minutes=30,
        exact_gap_seconds=60,
    )

    assert result["transition_detection_status"] == "exact"
    assert result["event_time"] == pd.Timestamp("2026-06-02T11:00:01Z")
    assert result["event_time_uncertainty_seconds"] == 1.0


def test_persistent_transition_search_covers_daily_anchor_uncertainty() -> None:
    old_times = pd.date_range("2026-03-05T15:30:00Z", periods=5193, freq="1s")
    new_times = pd.date_range("2026-03-05T16:56:33Z", periods=6 * 3600 + 1, freq="1s")
    raw = pd.DataFrame(
        {
            "time": old_times.append(new_times),
            "bms_soh_pct": [96.0] * len(old_times) + [97.0] * len(new_times),
        }
    )

    result = _persistent_transition(
        raw,
        daily_anchor=pd.Timestamp("2026-03-05T23:00:00Z"),
        previous_soh_pct=96.0,
        new_soh_pct=97.0,
        search_before_hours=30,
        search_after_hours=30,
        persistence_hours=6,
        minimum_new_level_fraction=0.95,
        previous_support_minutes=30,
        exact_gap_seconds=60,
    )

    assert result["transition_detection_status"] == "exact"
    assert result["event_time"] == pd.Timestamp("2026-03-05T16:56:33Z")
    assert result["event_time_uncertainty_seconds"] == 1.0


def test_evidence_comparison_reports_event_control_prevalence() -> None:
    events = pd.DataFrame(
        {
            "complete_discharge_within_1h": [True, False],
            "complete_discharge_within_3h": [True, False],
            "complete_discharge_within_6h": [True, False],
            "complete_discharge_within_12h": [True, False],
            "complete_discharge_within_24h": [True, False],
            "complete_discharge_within_48h": [True, False],
            "complete_discharge_within_72h": [True, False],
            "near_full_within_12h": [True, True],
            "rest_before_event_minutes": [180.0, 0.0],
            "meaningful_charge_before_update": [True, True],
            "low_before_near_full": [True, False],
            "reached_low_soc": [True, False],
            "maximum_temperature_c": [30.0, 31.0],
            "maximum_voltage_spread_mv": [20.0, 22.0],
            "maximum_absolute_c_rate": [0.2, 0.3],
        }
    )
    controls = pd.DataFrame(
        {
            "complete_discharge_within_1h": [False] * 4,
            "complete_discharge_within_3h": [False] * 4,
            "complete_discharge_within_6h": [False] * 4,
            "complete_discharge_within_12h": [False] * 4,
            "complete_discharge_within_24h": [False] * 4,
            "complete_discharge_within_48h": [False] * 4,
            "complete_discharge_within_72h": [False] * 4,
            "near_full_within_12h": [True, False, False, False],
            "prolonged_rest": [False] * 4,
            "meaningful_charge_before_update": [True, False, False, False],
            "low_before_near_full": [False] * 4,
            "reached_low_soc": [False] * 4,
            "maximum_temperature_c": [28.0] * 4,
            "maximum_voltage_spread_mv": [18.0] * 4,
            "maximum_absolute_c_rate": [0.2] * 4,
        }
    )

    comparison = build_evidence_comparison(events, controls, minimum_rest_minutes=120)
    primary = comparison.iloc[0]
    row = comparison.loc[comparison["feature_key"].eq("near_full_within_12h")].iloc[0]

    assert primary["feature_key"] == "complete_discharge_within_1h"
    assert primary["event_rate"] == 0.5
    assert primary["control_rate"] == 0.0
    assert row["event_rate"] == 1.0
    assert row["control_rate"] == 0.25
    assert row["prevalence_lift"] == 4.0


def test_interpretable_model_uses_grouped_validation() -> None:
    event_rows = []
    control_rows = []
    for index in range(6):
        event_id = f"event_{index}"
        event_rows.append(
            {
                "event_id": event_id,
                "near_full_within_6h": True,
                "near_full_within_12h": True,
                "meaningful_charge_before_update": True,
                "low_before_near_full": index % 2 == 0,
                "reached_low_soc": True,
                "rest_before_event_minutes": 0.0,
                "maximum_temperature_c": 30.0,
                "maximum_voltage_spread_mv": 20.0,
                "maximum_absolute_c_rate": 0.2,
                "context_coverage_fraction": 0.98,
            }
        )
        for control_index in range(3):
            control_rows.append(
                {
                    "matched_event_id": event_id,
                    "near_full_within_6h": False,
                    "near_full_within_12h": control_index == 0,
                    "meaningful_charge_before_update": False,
                    "low_before_near_full": False,
                    "reached_low_soc": control_index == 0,
                    "prolonged_rest": False,
                    "maximum_temperature_c": 28.0,
                    "maximum_voltage_spread_mv": 18.0,
                    "maximum_absolute_c_rate": 0.15,
                    "context_coverage_fraction": 0.97,
                }
            )

    model = build_interpretable_model(
        pd.DataFrame(event_rows),
        pd.DataFrame(control_rows),
        minimum_rest_minutes=120,
    )

    assert model["available"] is True
    assert model["validation_method"] == "leave_one_event_group_out"
    assert model["event_groups"] == 6
    assert model["cross_validated_roc_auc"] is not None
