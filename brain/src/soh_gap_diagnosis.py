from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import StorageConfig
from src.io_utils import write_parquet_chunk

PHASE4_VERSION = "phase4a_v1_7"
REFERENCE_SERIAL = "100000017"


@dataclass(frozen=True)
class Phase4Config:
    minimum_reference_strict_cycles: int = 500
    minimum_reference_matches: int = 3
    maximum_reference_matches: int = 10
    minimum_signal_coverage: float = 0.80
    significant_gap_floor_pp: float = 2.0
    uncertainty_multiplier: float = 1.96
    transition_lookback_hours: float = 72.0
    medium_temperature_delta_c: float = 10.0
    medium_rms_c_rate_delta: float = 0.25
    medium_soc_endpoint_delta_pp: float = 2.0
    medium_rest_delta_hours: float = 8.0
    medium_charge_intrusion_delta_ah: float = 1.0
    high_temperature_delta_c: float = 5.0
    high_rms_c_rate_delta: float = 0.10
    high_soc_endpoint_delta_pp: float = 1.0
    high_rest_delta_hours: float = 4.0
    high_charge_intrusion_delta_ah: float = 0.25
    minimum_physical_decline_observations: int = 2
    maximum_plausible_capacity_fraction_of_nameplate: float = 1.05
    minimum_transition_support_fraction: float = 0.80
    data_gap_risk_minutes: float = 60.0


def _latest_run(root: Path, serial: str, prefix: str, *, chemistry: str | None = None) -> Path:
    base = root
    if chemistry:
        base = base / f"chemistry={chemistry}"
    base = base / f"serial={serial}"
    runs = sorted(path for path in base.glob(f"{prefix}=*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No {prefix} run found for {serial} under {base}")
    return runs[-1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_utc(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result:
            result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    return result


def _overlap_seconds(
    left_start: pd.Timestamp, left_end: pd.Timestamp, right_start: pd.Timestamp, right_end: pd.Timestamp
) -> float:
    return max(0.0, (min(left_end, right_end) - max(left_start, right_start)).total_seconds())


def reconcile_candidates(
    canonical: pd.DataFrame, legacy: pd.DataFrame
) -> pd.DataFrame:
    eligible = canonical.loc[canonical["capacity_claim_eligible"].fillna(False)].copy()
    eligible = _as_utc(eligible, ["start_utc", "end_utc"])
    legacy = _as_utc(legacy, ["start_utc", "end_utc"])
    eligible_starts = eligible["start_utc"].map(lambda value: value.value).to_numpy(dtype=np.int64)
    eligible_ends = eligible["end_utc"].map(lambda value: value.value).to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    mapped_counts: dict[str, int] = {}
    for candidate in legacy.itertuples(index=False):
        start_ns = int(candidate.start_utc.value)
        end_ns = int(candidate.end_utc.value)
        overlap_ns = np.maximum(
            0, np.minimum(end_ns, eligible_ends) - np.maximum(start_ns, eligible_starts)
        )
        if not np.any(overlap_ns > 0):
            rows.append(
                {
                    "legacy_candidate_id": candidate.discharge_event_id,
                    "canonical_physical_cycle_id": None,
                    "reconciliation_status": "legacy_candidate_not_canonical",
                    "reconciliation_reason": "no_overlap_with_capacity_eligible_physical_cycle",
                    "overlap_seconds": 0.0,
                    "legacy_start_utc": candidate.start_utc,
                    "legacy_end_utc": candidate.end_utc,
                }
            )
            continue
        best_index = int(np.argmax(overlap_ns))
        seconds = float(overlap_ns[best_index] / 1_000_000_000)
        physical = eligible.iloc[best_index]
        physical_id = str(physical["physical_cycle_id"])
        mapped_counts[physical_id] = mapped_counts.get(physical_id, 0) + 1
        rows.append(
            {
                "legacy_candidate_id": candidate.discharge_event_id,
                "canonical_physical_cycle_id": physical_id,
                "reconciliation_status": "mapped",
                "reconciliation_reason": "maximum_time_overlap",
                "overlap_seconds": seconds,
                "legacy_start_utc": candidate.start_utc,
                "legacy_end_utc": candidate.end_utc,
            }
        )
    mapped_ids = {row["canonical_physical_cycle_id"] for row in rows if row["canonical_physical_cycle_id"]}
    for physical in eligible.itertuples(index=False):
        physical_id = str(physical.physical_cycle_id)
        if physical_id not in mapped_ids:
            rows.append(
                {
                    "legacy_candidate_id": None,
                    "canonical_physical_cycle_id": physical_id,
                    "reconciliation_status": "canonical_cycle_without_legacy_candidate",
                    "reconciliation_reason": "new_rest_aware_boundary_logic",
                    "overlap_seconds": 0.0,
                    "legacy_start_utc": pd.NaT,
                    "legacy_end_utc": pd.NaT,
                }
            )
    for row in rows:
        physical_id = row["canonical_physical_cycle_id"]
        if physical_id and mapped_counts.get(physical_id, 0) > 1:
            row["reconciliation_status"] = "multiple_legacy_candidates_one_physical_cycle"
            row["reconciliation_reason"] = "legacy_overlapping_windows_deduplicated"
    return pd.DataFrame.from_records(rows)


def _attach_phase3(cycles: pd.DataFrame, phase3: pd.DataFrame) -> pd.DataFrame:
    cycles = _as_utc(cycles, ["start_utc", "end_utc"])
    phase3 = _as_utc(phase3, ["start_utc", "end_utc"])
    phase3_starts = phase3["start_utc"].map(lambda value: value.value).to_numpy(dtype=np.int64)
    phase3_ends = phase3["end_utc"].map(lambda value: value.value).to_numpy(dtype=np.int64)
    diagnostic_columns = [
        "phase3_status",
        "native_coverage_fraction",
        "temperature_mean_c",
        "temperature_min_c",
        "temperature_max_c",
        "temperature_sensor_spread_p95_c",
        "rms_discharge_current_a",
        "rms_c_rate",
        "voltage_group_spread_p95_mv",
        "low_soc_voltage_group_spread_p95_mv",
        "pack_voltage_sum_delta_p95_v",
        "recurring_lowest_voltage_group",
        "recurring_lowest_voltage_group_fraction",
    ]
    records: list[dict[str, Any]] = []
    for cycle in cycles.itertuples(index=False):
        overlap_ns = np.maximum(
            0,
            np.minimum(int(cycle.end_utc.value), phase3_ends)
            - np.maximum(int(cycle.start_utc.value), phase3_starts),
        )
        best_index = int(np.argmax(overlap_ns)) if len(overlap_ns) else -1
        best_overlap = float(overlap_ns[best_index] / 1_000_000_000) if best_index >= 0 else 0.0
        best = phase3.iloc[best_index] if best_index >= 0 and best_overlap > 0 else None
        record = cycle._asdict()
        record["phase3_source_event_id"] = None if best is None else best.get("discharge_event_id")
        record["phase3_overlap_seconds"] = best_overlap
        for column in diagnostic_columns:
            record[column] = None if best is None else best.get(column)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_reference_curve(
    measurements: pd.DataFrame,
    *,
    reference_nameplate_ah: float,
    minimum_cycles: int,
) -> dict[str, Any]:
    frame = _as_utc(measurements, ["end_anchor_time"])
    frame = frame.loc[
        frame["capacity_measurement_duration_valid"].fillna(False)
        & pd.to_numeric(frame["capacity_ah"], errors="coerce").notna()
        & pd.to_numeric(frame["cumulative_all_discharge_ah"], errors="coerce").notna()
    ].sort_values("cumulative_all_discharge_ah")
    strict_inventory_rows = len(frame)
    if strict_inventory_rows < minimum_cycles:
        raise ValueError(
            f"Reference gate failed: {len(frame)} valid strict cycles; require at least {minimum_cycles}"
        )
    if "capacity_measurement_quality" in frame:
        clean = frame.loc[frame["capacity_measurement_quality"].eq("clean_anchor_100_to_0")]
        if len(clean) >= minimum_cycles:
            frame = clean
    first = frame.iloc[0]
    x = frame["cumulative_all_discharge_ah"].to_numpy(dtype=float) / 1000.0
    y = frame["capacity_ah"].to_numpy(dtype=float)
    slope, fitted_intercept = np.polyfit(x, y, 1)
    loss_slope = max(0.0, -float(slope))
    observed_origin_ah = float(first["cumulative_all_discharge_ah"])
    first_capacity = float(first["capacity_ah"])
    intercept_at_zero = float(fitted_intercept)
    predicted = intercept_at_zero - loss_slope * x
    residual = y - predicted
    return {
        "strict_cycle_rows": int(len(measurements)),
        "strict_inventory_valid_rows": int(strict_inventory_rows),
        "valid_fit_rows": int(len(frame)),
        "fit_quality_filter": "clean_anchor_100_to_0",
        "intercept_capacity_ah_at_observed_zero": intercept_at_zero,
        "first_measured_capacity_ah": first_capacity,
        "first_measurement_cumulative_ah": observed_origin_ah,
        "loss_slope_ah_per_1000ah": loss_slope,
        "residual_std_ah": float(np.std(residual, ddof=1)),
        "reference_nameplate_capacity_ah": reference_nameplate_ah,
        "origin_semantics": "zero means start of observed reference telemetry; pre-observation throughput is unknown",
        "intercept_method": "ordinary least squares intercept and slope fitted only to clean measured strict capacities",
        "source_is_bms_soh": False,
    }


def _reference_expected_capacity(cumulative_ah: float, curve: dict[str, Any]) -> float:
    return max(
        0.0,
        float(curve["intercept_capacity_ah_at_observed_zero"])
        - float(curve["loss_slope_ah_per_1000ah"]) * cumulative_ah / 1000.0,
    )


def _condition_distance(field: pd.Series, reference: pd.Series, config: Phase4Config) -> tuple[float, str, list[str]]:
    reasons: list[str] = []
    values: list[tuple[float, float, float]] = []
    specs = [
        ("soc_start_pct", config.high_soc_endpoint_delta_pp, config.medium_soc_endpoint_delta_pp),
        ("soc_end_pct", config.high_soc_endpoint_delta_pp, config.medium_soc_endpoint_delta_pp),
        ("temperature_mean_c", config.high_temperature_delta_c, config.medium_temperature_delta_c),
        ("rms_c_rate", config.high_rms_c_rate_delta, config.medium_rms_c_rate_delta),
        ("rest_duration_s", config.high_rest_delta_hours * 3600, config.medium_rest_delta_hours * 3600),
        ("charge_intrusion_ah", config.high_charge_intrusion_delta_ah, config.medium_charge_intrusion_delta_ah),
    ]
    medium = False
    for column, high_limit, medium_limit in specs:
        left = pd.to_numeric(pd.Series([field.get(column)]), errors="coerce").iloc[0]
        right = pd.to_numeric(pd.Series([reference.get(column)]), errors="coerce").iloc[0]
        if pd.isna(left) or pd.isna(right):
            reasons.append(f"missing_{column}")
            continue
        delta = abs(float(left) - float(right))
        values.append((delta / max(medium_limit, 1e-9), high_limit, medium_limit))
        if delta > medium_limit:
            reasons.append(f"{column}_outside_medium:{delta:.3f}")
            return sum(value[0] for value in values), "low", reasons
        if delta > high_limit:
            medium = True
            reasons.append(f"{column}_outside_high:{delta:.3f}")
    field_coverage = float(field.get("data_coverage_fraction") or 0.0)
    reference_coverage = float(reference.get("data_coverage_fraction") or 0.0)
    if min(field_coverage, reference_coverage) < config.minimum_signal_coverage:
        reasons.append("coverage_below_gate")
        return sum(value[0] for value in values), "indeterminate", reasons
    return sum(value[0] for value in values), "medium" if medium else "high", reasons


def normalize_capacity_observations(
    field_cycles: pd.DataFrame,
    reference_cycles: pd.DataFrame,
    transitions: pd.DataFrame,
    curve: dict[str, Any],
    *,
    field_nameplate_ah: float,
    config: Phase4Config,
) -> pd.DataFrame:
    field = field_cycles.sort_values("end_utc").copy()
    calculated_cumulative = field["discharged_ah"].fillna(0.0).clip(lower=0.0).cumsum()
    if "cumulative_all_discharge_ah" in field:
        field["cumulative_all_discharge_ah"] = pd.to_numeric(
            field["cumulative_all_discharge_ah"], errors="coerce"
        ).fillna(calculated_cumulative)
    else:
        field["cumulative_all_discharge_ah"] = calculated_cumulative
    field = field.loc[field["capacity_claim_eligible"].fillna(False)].copy()
    transitions = _as_utc(transitions, ["transition_time_utc"]).sort_values("transition_time_utc")
    initial_soh = float(transitions.iloc[0]["previous_soh_pct"]) if not transitions.empty else np.nan
    rows: list[dict[str, Any]] = []
    for _, event in field.iterrows():
        previous = transitions.loc[transitions["transition_time_utc"].le(event["end_utc"])]
        bms_soh = initial_soh if previous.empty else float(previous.iloc[-1]["new_soh_pct"])
        cumulative = float(event["cumulative_all_discharge_ah"])
        expected_capacity = _reference_expected_capacity(cumulative, curve)
        expected_soh_nameplate = expected_capacity / field_nameplate_ah * 100.0
        expected_soh_baseline = expected_capacity / float(curve["first_measured_capacity_ah"]) * 100.0
        capacity_soh_nameplate = np.nan
        capacity_soh_baseline = np.nan
        capacity_uncertainty_pp = np.nan
        capacity_method = "unavailable"
        capacity_status_reason = "no_sufficient_same_class_reference_match"
        comparability = "indeterminate"
        match_ids: list[str] = []
        if event["evidence_class"] == "strict_full_discharge":
            capacity_soh_nameplate = float(event["discharged_ah"]) / field_nameplate_ah * 100.0
            capacity_soh_baseline = float(event["discharged_ah"]) / float(curve["first_measured_capacity_ah"]) * 100.0
            capacity_uncertainty_pp = float(curve["residual_std_ah"]) / field_nameplate_ah * 100.0
            capacity_method = "direct_strict_100_to_0"
            capacity_status_reason = "direct_strict_measurement"
            measurement_quality = str(event.get("measurement_quality") or "")
            comparability = (
                "high"
                if measurement_quality == "high"
                else "medium"
                if measurement_quality == "medium"
                else "low"
                if measurement_quality == "low"
                else "indeterminate"
            )
        else:
            same_class = reference_cycles.loc[
                reference_cycles["evidence_class"].eq(event["evidence_class"])
            ]
            matches = [
                (*_condition_distance(event, reference, config), reference)
                for _, reference in same_class.iterrows()
            ]
            matches.sort(key=lambda item: item[0])
            selected = matches[: config.maximum_reference_matches]
            usable = [item for item in selected if item[1] in {"high", "medium"}]
        if event["evidence_class"] != "strict_full_discharge" and len(usable) >= config.minimum_reference_matches:
            ratios = []
            for _, _level, _, reference in usable:
                reference_cumulative = float(reference.get("cumulative_all_discharge_ah") or 0.0)
                reference_expected = _reference_expected_capacity(reference_cumulative, curve)
                if reference_expected > 0:
                    ratios.append(float(reference["discharged_ah"]) / reference_expected)
                    match_ids.append(str(reference["physical_cycle_id"]))
            if len(ratios) >= config.minimum_reference_matches and np.median(ratios) > 0:
                estimated_full_capacity = float(event["discharged_ah"]) / float(np.median(ratios))
                capacity_soh_nameplate = estimated_full_capacity / field_nameplate_ah * 100.0
                capacity_soh_baseline = estimated_full_capacity / float(curve["first_measured_capacity_ah"]) * 100.0
                mad = float(np.median(np.abs(np.asarray(ratios) - np.median(ratios))))
                capacity_uncertainty_pp = max(
                    float(curve["residual_std_ah"]) / field_nameplate_ah * 100.0,
                    1.4826 * mad / max(float(np.median(ratios)), 1e-9) * 100.0,
                )
                capacity_method = "same_class_reference_calibration"
                capacity_status_reason = "same_class_operating_condition_match"
                comparability = "high" if all(item[1] == "high" for item in usable) else "medium"
                if estimated_full_capacity > field_nameplate_ah * config.maximum_plausible_capacity_fraction_of_nameplate:
                    capacity_soh_nameplate = np.nan
                    capacity_soh_baseline = np.nan
                    capacity_uncertainty_pp = np.nan
                    capacity_method = "unavailable"
                    capacity_status_reason = "calibrated_capacity_exceeds_physical_plausibility_gate"
                    comparability = "indeterminate"
                    match_ids = []
        reference_uncertainty_pp = float(curve["residual_std_ah"]) / field_nameplate_ah * 100.0
        combined_uncertainty_pp = float(
            np.sqrt(reference_uncertainty_pp**2 + (capacity_uncertainty_pp if np.isfinite(capacity_uncertainty_pp) else 0.0) ** 2)
        )
        bms_reference_threshold = max(
            config.significant_gap_floor_pp,
            config.uncertainty_multiplier * reference_uncertainty_pp,
        )
        capacity_reference_threshold = max(
            config.significant_gap_floor_pp,
            config.uncertainty_multiplier * combined_uncertainty_pp,
        )
        capacity_gap = capacity_soh_nameplate - expected_soh_nameplate if np.isfinite(capacity_soh_nameplate) else np.nan
        rows.append(
            {
                "device": event["device"],
                "physical_cycle_id": event["physical_cycle_id"],
                "evidence_class": event["evidence_class"],
                "start_utc": event["start_utc"],
                "end_utc": event["end_utc"],
                "cumulative_all_discharge_ah": cumulative,
                "bms_soh_pct": bms_soh,
                "capacity_based_soh_nameplate_pct": capacity_soh_nameplate,
                "capacity_based_soh_reference_baseline_pct": capacity_soh_baseline,
                "capacity_method": capacity_method,
                "capacity_status_reason": capacity_status_reason,
                "reference_expected_capacity_ah": expected_capacity,
                "reference_expected_soh_nameplate_pct": expected_soh_nameplate,
                "reference_expected_soh_reference_baseline_pct": expected_soh_baseline,
                "gap_bms_reference_pp": bms_soh - expected_soh_nameplate,
                "gap_capacity_reference_pp": capacity_gap,
                "gap_bms_capacity_pp": bms_soh - capacity_soh_nameplate if np.isfinite(capacity_soh_nameplate) else np.nan,
                "reference_uncertainty_pp": reference_uncertainty_pp,
                "capacity_uncertainty_pp": capacity_uncertainty_pp,
                "combined_uncertainty_pp": combined_uncertainty_pp,
                "bms_reference_significant_gap_threshold_pp": bms_reference_threshold,
                "capacity_reference_significant_gap_threshold_pp": capacity_reference_threshold,
                "bms_reference_gap_significant": abs(bms_soh - expected_soh_nameplate) >= bms_reference_threshold,
                "capacity_reference_gap_significant": bool(
                    np.isfinite(capacity_gap) and abs(capacity_gap) >= capacity_reference_threshold
                ),
                "reference_comparability": comparability,
                "matched_reference_count": len(match_ids),
                "matched_reference_cycle_ids": match_ids,
                "temperature_mean_c": event.get("temperature_mean_c"),
                "rms_c_rate": event.get("rms_c_rate"),
                "low_soc_voltage_group_spread_p95_mv": event.get("low_soc_voltage_group_spread_p95_mv"),
                "recurring_lowest_voltage_group": event.get("recurring_lowest_voltage_group"),
                "pack_voltage_sum_delta_p95_v": event.get("pack_voltage_sum_delta_p95_v"),
            }
        )
    return pd.DataFrame.from_records(rows)


def _aggregate_daily_interval(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    frame = daily.loc[daily["day_utc"].ge(start.floor("D")) & daily["day_utc"].le(end.floor("D"))].copy()
    sums = [column for column in frame.columns if column.endswith(("_ah", "_wh", "_minutes", "_count"))]
    result: dict[str, Any] = {column: float(pd.to_numeric(frame[column], errors="coerce").sum()) for column in sums}
    weights = pd.to_numeric(frame.get("observed_minutes"), errors="coerce").fillna(0.0)
    for column in ["temperature_mean_c", "temperature_spread_p95_c", "voltage_spread_p95_mv"]:
        if column in frame and weights.sum() > 0:
            values = pd.to_numeric(frame[column], errors="coerce")
            valid = values.notna() & weights.gt(0)
            result[column] = float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan
    expected_minutes = max(0.0, (end - start).total_seconds() / 60.0)
    observed_minutes = min(expected_minutes, float(pd.to_numeric(frame.get("observed_minutes"), errors="coerce").sum()))
    result.update(
        {
            "interval_start_utc": start,
            "interval_end_utc": end,
            "calendar_days": (end - start).total_seconds() / 86400.0,
            "observed_minutes": observed_minutes,
            "expected_minutes": expected_minutes,
            "coverage_fraction": observed_minutes / expected_minutes if expected_minutes > 0 else np.nan,
            "aggregation_resolution": "time_weighted_daily_source; boundary days may include out-of-interval exposure",
        }
    )
    return result


def build_stress_intervals(
    daily: pd.DataFrame,
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    physical_cycles: pd.DataFrame,
) -> pd.DataFrame:
    daily = _as_utc(daily, ["day_utc"])
    transitions = _as_utc(transitions, ["transition_time_utc"])
    observations = _as_utc(observations, ["end_utc"])
    records: list[dict[str, Any]] = []

    def cycle_values(frame: pd.DataFrame, column: str) -> pd.Series:
        return pd.to_numeric(
            frame[column] if column in frame else pd.Series(np.nan, index=frame.index),
            errors="coerce",
        )
    for interval_type, points, id_column, time_column in [
        ("between_soh_transitions", transitions, "transition_id", "transition_time_utc"),
        ("between_capacity_observations", observations, "physical_cycle_id", "end_utc"),
    ]:
        ordered = points.sort_values(time_column)
        for previous, current in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False), strict=False):
            start = pd.Timestamp(getattr(previous, time_column))
            end = pd.Timestamp(getattr(current, time_column))
            row = _aggregate_daily_interval(daily, start, end)
            row.update(
                {
                    "interval_type": interval_type,
                    "start_source_id": getattr(previous, id_column),
                    "end_source_id": getattr(current, id_column),
                }
            )
            cycles = physical_cycles.loc[
                physical_cycles["end_utc"].gt(start) & physical_cycles["end_utc"].le(end)
            ]
            row.update(
                {
                    "physical_discharge_cycle_count": int(len(cycles)),
                    "strict_100_to_0_cycle_count": int(cycles["observed_100_to_0"].fillna(False).sum()),
                    "near_99_to_1_cycle_count": int(cycles["observed_99_to_1"].fillna(False).sum()),
                    "operational_95_to_5_cycle_count": int(cycles["observed_95_to_5"].fillna(False).sum()),
                    "median_soc_span_pct": float(cycle_values(cycles, "soc_span_pct").median()),
                    "long_rest_episode_count": int(
                        cycle_values(cycles, "rest_duration_s").ge(24 * 3600).sum()
                    ),
                    "maximum_rest_hours": float(
                        cycle_values(cycles, "rest_duration_s").max() / 3600.0
                    ),
                    "sys_cycles_delta": float(
                        cycle_values(cycles, "sys_cycles_delta").sum(min_count=1)
                    ),
                }
            )
            records.append(row)
    return pd.DataFrame.from_records(records)


def build_assessments(
    transitions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    config: Phase4Config,
) -> pd.DataFrame:
    transitions = _as_utc(transitions, ["transition_time_utc"]).sort_values("transition_time_utc")
    observations = _as_utc(observations, ["end_utc"]).sort_values("end_utc")
    rows: list[dict[str, Any]] = []
    for _, transition in transitions.iterrows():
        containing = observations.loc[
            observations["start_utc"].le(transition["transition_time_utc"])
            & observations["end_utc"].ge(transition["transition_time_utc"])
        ].copy()
        if not containing.empty:
            associated = containing.sort_values("start_utc").iloc[-1].copy()
            associated["hours_before"] = 0.0
            association_timing = "transition_inside_discharge_cycle"
        else:
            previous = observations.loc[
                observations["end_utc"].le(transition["transition_time_utc"])
            ].copy()
            if previous.empty:
                associated = None
                association_timing = "none"
            else:
                previous["hours_before"] = (
                    transition["transition_time_utc"] - previous["end_utc"]
                ).dt.total_seconds() / 3600.0
                associated = previous.iloc[-1]
                association_timing = "cycle_completed_before_transition"
        supporting: list[str] = []
        contradicting: list[str] = []
        missing: list[str] = []
        classification = "indeterminate"
        confidence = "low"
        if associated is None or float(associated["hours_before"]) > config.transition_lookback_hours:
            missing.append("no_canonical_deep_discharge_within_72h")
        else:
            supporting.append(f"deep_discharge_within_72h:{associated['physical_cycle_id']}")
            supporting.append(association_timing)
            if associated["bms_reference_gap_significant"]:
                supporting.append("significant_bms_reference_gap")
            if associated["capacity_method"] == "unavailable":
                missing.append("no_calibrated_capacity_soh")
            elif not associated["capacity_reference_gap_significant"]:
                supporting.append("capacity_not_significantly_below_reference")
            else:
                contradicting.append("capacity_significantly_differs_from_reference")
            comparable = observations.loc[
                observations["reference_comparability"].isin(["high", "medium"])
                & observations["capacity_based_soh_nameplate_pct"].notna()
                & observations["end_utc"].le(transition["transition_time_utc"])
            ]
            recent_comparable = comparable.tail(max(3, config.minimum_physical_decline_observations))
            decline_flags = (
                recent_comparable["capacity_reference_gap_significant"]
                & recent_comparable["gap_capacity_reference_pp"].lt(0)
            )
            decline_support = int(decline_flags.sum())
            recalibration_pattern = (
                float(transition["delta_soh_pp"]) < 0
                and not associated["capacity_reference_gap_significant"]
                and associated["capacity_method"] != "unavailable"
            )
            physical_pattern = (
                float(transition["delta_soh_pp"]) < 0
                and
                len(recent_comparable) >= config.minimum_physical_decline_observations
                and bool(decline_flags.tail(config.minimum_physical_decline_observations).all())
            )
            operating_pattern = (
                associated["capacity_reference_gap_significant"]
                and associated["reference_comparability"] == "low"
            )
            if physical_pattern and recalibration_pattern:
                classification = "mixed_behaviour"
            elif physical_pattern:
                classification = "physical_capacity_decline_supported"
            elif recalibration_pattern:
                classification = "likely_bms_recalibration"
            elif operating_pattern:
                classification = "operating_condition_effect_likely"
            confidence = (
                "high"
                if associated["reference_comparability"] == "high" and decline_support >= 2
                else "medium"
                if associated["reference_comparability"] in {"high", "medium"}
                else "low"
            )
        rows.append(
            {
                "transition_id": transition["transition_id"],
                "transition_time_utc": transition["transition_time_utc"],
                "soh_before_pct": transition["previous_soh_pct"],
                "soh_after_pct": transition["new_soh_pct"],
                "soh_change_pp": transition["delta_soh_pp"],
                "associated_physical_cycle_id": None if associated is None else associated["physical_cycle_id"],
                "association_timing": association_timing,
                "classification": classification,
                "confidence": confidence,
                "supporting_evidence": supporting,
                "contradicting_evidence": contradicting,
                "missing_evidence": missing,
                "rule_version": PHASE4_VERSION,
            }
        )
    return pd.DataFrame.from_records(rows)


def run_phase4(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    field_nameplate_ah: float,
    reference_nameplate_ah: float,
    processed_root: Path,
    output_root: Path,
    config: Phase4Config = Phase4Config(),  # noqa: B008 - frozen dataclass, safe as a default
) -> Path:
    if chemistry != "NCA":
        raise ValueError("Phase 4A is NCA-only; LFP requires a separate reference model")
    phase2_run = _latest_run(processed_root / "bms_soh_recalibration", serial, "recalibration_run")
    field_catalog_run = _latest_run(processed_root / "physical_discharge_catalog", serial, "catalog_run")
    reference_catalog_run = _latest_run(processed_root / "physical_discharge_catalog", REFERENCE_SERIAL, "catalog_run")
    field_phase3_run = _latest_run(processed_root / "electrothermal_capacity_context", serial, "phase3_run", chemistry=chemistry)
    reference_phase3_run = _latest_run(processed_root / "electrothermal_capacity_context", REFERENCE_SERIAL, "phase3_run", chemistry=chemistry)
    seasonal_run = _latest_run(processed_root / "seasonal_charge_analysis", serial, "profile_run")
    capacity_runs = sorted((processed_root / "capacity_trend" / f"serial={REFERENCE_SERIAL}").glob("capacity_run=*"))
    if not capacity_runs:
        raise FileNotFoundError("Measured reference capacity trend is missing")
    capacity_run = capacity_runs[-1]

    transitions = pd.read_parquet(phase2_run / "bms_soh_transitions.parquet")
    transition_detail_path = phase2_run / "recalibration_events.parquet"
    if transition_detail_path.exists():
        details = pd.read_parquet(transition_detail_path)
        details = details[
            [
                "event_id",
                "pre_transition_support_fraction",
                "post_transition_support_fraction",
                "largest_data_gap_minutes",
                "transition_detection_status",
                "event_time_resolution",
                "rest_before_event_minutes",
            ]
        ].rename(columns={"event_id": "transition_id"})
        transitions = transitions.merge(details, on="transition_id", how="left", validate="one_to_one")
    def transition_column(name: str) -> pd.Series:
        if name in transitions:
            return transitions[name]
        return pd.Series(np.nan, index=transitions.index)

    transitions["persistence_valid"] = (
        pd.to_numeric(transition_column("pre_transition_support_fraction"), errors="coerce")
        .ge(config.minimum_transition_support_fraction)
        & pd.to_numeric(transition_column("post_transition_support_fraction"), errors="coerce")
        .ge(config.minimum_transition_support_fraction)
    )
    transitions["data_gap_or_reset_risk"] = pd.to_numeric(
        transition_column("largest_data_gap_minutes"), errors="coerce"
    ).ge(config.data_gap_risk_minutes)
    field_catalog = pd.read_parquet(field_catalog_run / "physical_discharge_cycles.parquet")
    reference_catalog = pd.read_parquet(reference_catalog_run / "physical_discharge_cycles.parquet")
    field_phase3 = pd.read_parquet(field_phase3_run / "event_electrothermal_context.parquet")
    reference_phase3 = pd.read_parquet(reference_phase3_run / "event_electrothermal_context.parquet")
    reference_measurements = pd.read_parquet(capacity_run / "capacity_measurements.parquet")
    daily = pd.read_parquet(seasonal_run / "daily_battery_stress.parquet")
    monthly = pd.read_parquet(seasonal_run / "monthly_operating_profile.parquet")

    cumulative_lookup = reference_measurements.set_index("event_discharge_id")[
        "cumulative_all_discharge_ah"
    ]
    reference_catalog = reference_catalog.copy()
    reference_catalog["cumulative_all_discharge_ah"] = pd.to_numeric(
        reference_catalog["source_capacity_event_id"], errors="coerce"
    ).map(cumulative_lookup)
    reference_catalog["cumulative_all_discharge_ah"] = reference_catalog[
        "cumulative_all_discharge_ah"
    ].fillna(reference_catalog["discharged_ah"].fillna(0.0).clip(lower=0.0).cumsum())

    if serial == REFERENCE_SERIAL:
        field_catalog = field_catalog.copy()
        field_catalog["capacity_claim_eligible"] = field_catalog["observed_100_to_0"].fillna(False)
        field_catalog["cumulative_all_discharge_ah"] = pd.to_numeric(
            field_catalog["source_capacity_event_id"], errors="coerce"
        ).map(cumulative_lookup)

    curve = build_reference_curve(
        reference_measurements,
        reference_nameplate_ah=reference_nameplate_ah,
        minimum_cycles=config.minimum_reference_strict_cycles,
    )
    reconciliation = reconcile_candidates(field_catalog, field_phase3)
    field_enriched = _attach_phase3(field_catalog, field_phase3)
    reference_enriched = _attach_phase3(reference_catalog, reference_phase3)
    observations = normalize_capacity_observations(
        field_enriched,
        reference_enriched,
        transitions,
        curve,
        field_nameplate_ah=field_nameplate_ah,
        config=config,
    )
    field_enriched = _as_utc(field_enriched, ["start_utc", "end_utc"])
    stress = build_stress_intervals(daily, transitions, observations, field_enriched)
    assessments = build_assessments(transitions, observations, config=config)
    transition_points = transitions.rename(
        columns={"transition_id": "source_id", "transition_time_utc": "timestamp_utc"}
    ).assign(point_type="soh_transition")
    observation_context = [
        "physical_cycle_id", "evidence_class", "bms_soh_pct", "capacity_based_soh_nameplate_pct",
        "reference_expected_soh_nameplate_pct", "temperature_mean_c", "rms_c_rate",
        "low_soc_voltage_group_spread_p95_mv", "recurring_lowest_voltage_group",
        "pack_voltage_sum_delta_p95_v", "reference_comparability",
    ]
    starts = observations[observation_context + ["start_utc"]].rename(
        columns={"physical_cycle_id": "source_id", "start_utc": "timestamp_utc"}
    ).assign(point_type="discharge_start")
    ends = observations[observation_context + ["end_utc"]].rename(
        columns={"physical_cycle_id": "source_id", "end_utc": "timestamp_utc"}
    ).assign(point_type="discharge_end")
    verification = pd.concat([transition_points, starts, ends], ignore_index=True).sort_values("timestamp_utc")

    identity_payload = {
        "version": PHASE4_VERSION,
        "serial": serial,
        "field_catalog": field_catalog_run.name,
        "reference_catalog": reference_catalog_run.name,
        "phase2": phase2_run.name,
        "field_phase3": field_phase3_run.name,
        "reference_phase3": reference_phase3_run.name,
        "seasonal": seasonal_run.name,
        "capacity": capacity_run.name,
        "config": asdict(config),
        "field_nameplate_ah": field_nameplate_ah,
        "reference_nameplate_ah": reference_nameplate_ah,
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    base = output_root / f"chemistry={chemistry}" / f"serial={serial}"
    for existing in sorted(base.glob("phase4_run=*"), reverse=True):
        manifest_path = existing / "phase4_manifest.json"
        if manifest_path.exists() and _read_json(manifest_path).get("identity_sha256") == identity:
            return existing
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = base / f"phase4_run={stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = {
        "soh_transitions.parquet": transitions,
        "stress_intervals.parquet": stress,
        "device_month_stress.parquet": monthly,
        "reference_normalized_observations.parquet": observations,
        "candidate_reconciliation.parquet": reconciliation,
        "diagnostic_assessments.parquet": assessments,
        "event_verification_points.parquet": verification,
    }
    for name, frame in artifacts.items():
        write_parquet_chunk(frame, run_dir / name, compression="zstd")

    comparison_counts = observations["reference_comparability"].value_counts().to_dict() if not observations.empty else {}
    sensitivity: dict[str, Any] = {}
    for floor in (1.5, 2.0, 3.0):
        thresholds = np.maximum(
            floor,
            config.uncertainty_multiplier
            * pd.to_numeric(observations["reference_uncertainty_pp"], errors="coerce").fillna(0.0),
        )
        sensitivity[f"gap_floor_{floor:.1f}pp"] = {
            "bms_reference_significant_rows": int(
                pd.to_numeric(observations["gap_bms_reference_pp"], errors="coerce").abs().ge(thresholds).sum()
            ),
            "capacity_reference_significant_rows": int((
                pd.to_numeric(observations["gap_capacity_reference_pp"], errors="coerce").abs()
                >= np.maximum(
                    floor,
                    config.uncertainty_multiplier
                    * pd.to_numeric(observations["combined_uncertainty_pp"], errors="coerce").fillna(0.0),
                )
            ).sum()),
        }
    summary = {
        "model_family": "soh_gap_diagnosis",
        "phase4_version": PHASE4_VERSION,
        "serial": serial,
        "reference_serial": REFERENCE_SERIAL,
        "chemistry": chemistry,
        "module_count": module_count,
        "timestamp_semantics": "UTC",
        "field_nameplate_capacity_ah": field_nameplate_ah,
        "reference_nameplate_capacity_ah": reference_nameplate_ah,
        "reference_curve": curve,
        "legacy_phase3_candidate_rows": int(len(field_phase3)),
        "canonical_capacity_observation_rows": int(len(observations)),
        "candidate_reconciliation_counts": reconciliation["reconciliation_status"].value_counts().to_dict(),
        "reference_comparability_counts": comparison_counts,
        "soh_transition_rows": int(len(transitions)),
        "stress_interval_rows": int(len(stress)),
        "device_month_rows": int(len(monthly)),
        "diagnostic_classification_counts": assessments["classification"].value_counts().to_dict(),
        "threshold_sensitivity": sensitivity,
        "uncertainty_scope": "within-reference fit, repeatability, matched-event variability, and measurement uncertainty only",
        "uncertainty_excludes": "manufacturing, batch, hardware, and firmware variability between devices",
        "claim_limit": "Explainable evidence classification; not causal proof, laboratory ground truth, or a black-box ML prediction.",
    }
    (run_dir / "device_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    manifest = {
        **identity_payload,
        "identity_sha256": identity,
        "created_utc": datetime.now(UTC).isoformat(),
        "artifacts": {name: len(frame) for name, frame in artifacts.items()},
        "reference_matching_variables": [
            "evidence_class", "soc_start_pct", "soc_end_pct", "temperature_mean_c",
            "rms_c_rate", "rest_duration_s", "charge_intrusion_ah", "data_coverage_fraction",
        ],
        "diagnostic_outcomes_not_matching_variables": [
            "voltage_group_spread", "weakest_group_identity", "pack_voltage_asymmetry", "temperature_asymmetry",
        ],
    }
    (run_dir / "phase4_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Phase 4A single-device SOH-gap diagnosis")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--field-nameplate-capacity-ah", required=True, type=float)
    parser.add_argument("--reference-nameplate-capacity-ah", required=True, type=float)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--minimum-reference-strict-cycles",
        type=int,
        default=Phase4Config.minimum_reference_strict_cycles,
        help="Reference gate: valid strict reference capacity cycles needed to fit the curve",
    )
    args = parser.parse_args(argv)
    if args.minimum_reference_strict_cycles < 2:
        parser.error("--minimum-reference-strict-cycles must be at least 2")
    storage = StorageConfig.from_env()
    processed = args.processed_root or storage.data_dir / "processed"
    run_dir = run_phase4(
        serial=args.serial,
        chemistry=args.chemistry,
        module_count=args.module_count,
        field_nameplate_ah=args.field_nameplate_capacity_ah,
        reference_nameplate_ah=args.reference_nameplate_capacity_ah,
        processed_root=processed,
        output_root=args.output_root or processed / "soh_gap_diagnosis",
        config=Phase4Config(
            minimum_reference_strict_cycles=args.minimum_reference_strict_cycles
        ),
    )
    print(f"Phase 4A SOH-gap diagnosis complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
