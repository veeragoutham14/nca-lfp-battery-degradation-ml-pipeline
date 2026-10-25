from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectoryConfig:
    maximum_lookback_hours: float = 72.0
    high_soc_pct: float = 95.0
    low_soc_pct: float = 5.0
    empty_soc_pct: float = 0.0
    rest_current_threshold_a: float = 0.5
    maximum_integrated_gap_seconds: float = 5.0
    major_data_gap_seconds: float = 3600.0
    tolerated_charge_intrusion_ah: float = 0.10
    meaningful_recharge_ah: float = 0.50
    meaningful_soc_recovery_pp: float = 5.0


def _split_interval_ah(left: float, right: float, seconds: float) -> tuple[float, float]:
    if seconds <= 0:
        return 0.0, 0.0
    if left <= 0 and right <= 0:
        return abs(left + right) * seconds / 2.0 / 3600.0, 0.0
    if left >= 0 and right >= 0:
        return 0.0, (left + right) * seconds / 2.0 / 3600.0
    negative, positive = (left, right) if left < 0 else (right, left)
    negative_seconds = seconds * abs(negative) / (abs(negative) + positive)
    positive_seconds = seconds - negative_seconds
    return (
        abs(negative) * negative_seconds / 2.0 / 3600.0,
        positive * positive_seconds / 2.0 / 3600.0,
    )


def _evidence_class(start_soc: float, end_soc: float) -> str:
    span = start_soc - end_soc
    if start_soc >= 100.0 and end_soc <= 0.0 and span >= 100.0:
        return "strict_100_to_0"
    if start_soc >= 99.0 and end_soc <= 1.0 and span >= 98.0:
        return "supporting_99_to_1"
    if start_soc >= 95.0 and end_soc <= 5.0 and span >= 90.0:
        return "supporting_95_to_5"
    return "partial"


def detect_discharge_trajectories(
    raw: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
    transition_id: str,
    nominal_capacity_ah: float,
    config: TrajectoryConfig | None = None,
) -> pd.DataFrame:
    """Reconstruct one-second discharge trajectories without compact-data gating."""
    config = config or TrajectoryConfig()
    anchor = pd.Timestamp(anchor_time)
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    else:
        anchor = anchor.tz_convert("UTC")
    start_bound = anchor - pd.Timedelta(hours=config.maximum_lookback_hours)
    frame = raw.loc[raw["time"].between(start_bound, anchor)].copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
    frame["bms_soc_pct"] = pd.to_numeric(frame["bms_soc_pct"], errors="coerce")
    frame["bms_current_a"] = pd.to_numeric(frame["bms_current_a"], errors="coerce")
    voltage = pd.to_numeric(frame.get("bms_voltage_v"), errors="coerce")
    soh = pd.to_numeric(frame.get("bms_soh_pct"), errors="coerce")
    all_zero_dropout = (
        frame["bms_soc_pct"].eq(0.0)
        & frame["bms_current_a"].eq(0.0)
        & voltage.eq(0.0)
        & soh.eq(0.0)
    )
    frame = (
        frame.loc[~all_zero_dropout]
        .dropna(subset=["time", "bms_soc_pct", "bms_current_a"])
        .sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 2:
        return pd.DataFrame()

    high_positions = np.flatnonzero(
        frame["bms_soc_pct"].ge(config.high_soc_pct)
        & ~frame["bms_soc_pct"].ge(config.high_soc_pct).shift(fill_value=False)
    )
    records: list[dict[str, Any]] = []
    normalized_starts: list[int] = []
    for position, high_index in enumerate(high_positions):
        search_end = (
            int(high_positions[position + 1]) if position + 1 < len(high_positions) else len(frame)
        )
        search = frame.iloc[int(high_index) : search_end]
        discharge_offsets = np.flatnonzero(
            search["bms_current_a"].lt(-config.rest_current_threshold_a)
        )
        if not len(discharge_offsets):
            continue
        discharge_onset = int(high_index) + int(discharge_offsets[0])
        pre_discharge = frame.iloc[int(high_index) : discharge_onset + 1]
        normalized_starts.append(int(pre_discharge["bms_soc_pct"].idxmax()))

    for ordinal, start_index in enumerate(sorted(set(normalized_starts)), start=1):
        start_row = frame.iloc[int(start_index)]
        start_soc = float(start_row["bms_soc_pct"])
        minimum_soc = start_soc
        minimum_index = int(start_index)
        discharge_ah = 0.0
        charge_ah = 0.0
        integrated_seconds = 0.0
        elapsed_seconds = 0.0
        rest_seconds = 0.0
        longest_rest_seconds = 0.0
        current_rest_seconds = 0.0
        rest_count = 0
        continuity = {"continuous_discharge"}
        end_reason = "dataset_end"
        milestone_times: dict[str, pd.Timestamp | None] = {
            "soc_95_utc": pd.Timestamp(start_row["time"]) if start_soc <= 95.0 else None,
            "soc_5_utc": None,
            "soc_1_utc": None,
            "soc_0_utc": None,
        }
        previous = frame.iloc[int(start_index)]
        last_index = int(start_index)
        for index in range(int(start_index) + 1, len(frame)):
            current = frame.iloc[index]
            seconds = float((current["time"] - previous["time"]).total_seconds())
            if seconds <= 0:
                previous = current
                continue
            elapsed_seconds += seconds
            if seconds > config.major_data_gap_seconds:
                end_reason = "major_data_gap"
                break
            left_current = float(previous["bms_current_a"])
            right_current = float(current["bms_current_a"])
            midpoint_current = (left_current + right_current) / 2.0
            if seconds <= config.maximum_integrated_gap_seconds:
                discharged, charged = _split_interval_ah(left_current, right_current, seconds)
                discharge_ah += discharged
                charge_ah += charged
                integrated_seconds += seconds
            if abs(midpoint_current) <= config.rest_current_threshold_a:
                if current_rest_seconds == 0:
                    rest_count += 1
                current_rest_seconds += seconds
                rest_seconds += seconds
                longest_rest_seconds = max(longest_rest_seconds, current_rest_seconds)
                continuity.add("rest_continuation")
            else:
                current_rest_seconds = 0.0
            if midpoint_current > config.rest_current_threshold_a:
                continuity.add("tolerated_charge_intrusion")

            soc = float(current["bms_soc_pct"])
            if milestone_times["soc_95_utc"] is None and soc <= 95.0:
                milestone_times["soc_95_utc"] = pd.Timestamp(current["time"])
            if milestone_times["soc_5_utc"] is None and soc <= 5.0:
                milestone_times["soc_5_utc"] = pd.Timestamp(current["time"])
            if milestone_times["soc_1_utc"] is None and soc <= 1.0:
                milestone_times["soc_1_utc"] = pd.Timestamp(current["time"])
            if milestone_times["soc_0_utc"] is None and soc <= config.empty_soc_pct:
                milestone_times["soc_0_utc"] = pd.Timestamp(current["time"])
            if soc < minimum_soc:
                minimum_soc = soc
                minimum_index = index
            recovery = soc - minimum_soc
            last_index = index
            if charge_ah > config.meaningful_recharge_ah:
                end_reason = "meaningful_recharge"
                break
            if recovery >= config.meaningful_soc_recovery_pp:
                end_reason = "soc_recovery"
                break
            if minimum_soc <= config.empty_soc_pct:
                end_reason = "reached_empty_soc"
                last_index = minimum_index
                break
            previous = current

        end_soc = float(frame.iloc[minimum_index]["bms_soc_pct"])
        end_time = pd.Timestamp(frame.iloc[minimum_index]["time"])
        if minimum_index <= int(start_index) or discharge_ah <= 0:
            continue
        coverage = integrated_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        evidence = _evidence_class(start_soc, end_soc)
        observed_95_to_5 = start_soc >= 95.0 and end_soc <= 5.0 and start_soc - end_soc >= 90.0
        observed_99_to_1 = start_soc >= 99.0 and end_soc <= 1.0 and start_soc - end_soc >= 98.0
        observed_100_to_0 = start_soc >= 100.0 and end_soc <= 0.0 and start_soc - end_soc >= 100.0
        delay_hours = (anchor - end_time).total_seconds() / 3600.0
        records.append(
            {
                "transition_id": transition_id,
                "trajectory_id": f"{transition_id}_trajectory_{ordinal:03d}",
                "trajectory_start_utc": pd.Timestamp(start_row["time"]),
                "trajectory_end_utc": end_time,
                "trajectory_end_reason": end_reason,
                "continuity": ",".join(sorted(continuity)),
                "soc_start_pct": start_soc,
                "soc_end_pct": end_soc,
                "soc_span_pp": start_soc - end_soc,
                "evidence_class": evidence,
                "observed_95_to_5": observed_95_to_5,
                "observed_99_to_1": observed_99_to_1,
                "observed_100_to_0": observed_100_to_0,
                **milestone_times,
                "discharged_ah": discharge_ah,
                "charge_intrusion_ah": charge_ah,
                "charge_intrusion_within_tolerance": charge_ah
                <= config.tolerated_charge_intrusion_ah,
                "rest_minutes": rest_seconds / 60.0,
                "longest_rest_minutes": longest_rest_seconds / 60.0,
                "rest_interruption_count": rest_count,
                "elapsed_hours": elapsed_seconds / 3600.0,
                "coverage_fraction": coverage,
                "maximum_gap_s": float(
                    frame.iloc[int(start_index) : last_index + 1]["time"]
                    .diff()
                    .dt.total_seconds()
                    .max()
                ),
                "delay_to_transition_hours": delay_hours,
                "within_12h": 0 <= delay_hours <= 12,
                "within_24h": 0 <= delay_hours <= 24,
                "within_48h": 0 <= delay_hours <= 48,
                "within_72h": 0 <= delay_hours <= 72,
                "direct_capacity_proxy_eligible": evidence == "strict_100_to_0"
                and coverage >= 0.98
                and charge_ah <= config.tolerated_charge_intrusion_ah,
                "supporting_capacity_estimate_eligible": evidence
                in {"supporting_99_to_1", "supporting_95_to_5"}
                and coverage >= 0.90,
                "capacity_soh_pct": (
                    discharge_ah / nominal_capacity_ah * 100.0
                    if evidence == "strict_100_to_0"
                    else discharge_ah
                    / max((start_soc - end_soc) / 100.0, 1e-9)
                    / nominal_capacity_ah
                    * 100.0
                ),
                "raw_rows": last_index - int(start_index) + 1,
            }
        )
    return pd.DataFrame.from_records(records)


def rank_transition_trajectories(
    transitions: pd.DataFrame, trajectories: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_column = "delta_soh_pp" if "delta_soh_pp" in transitions else "soh_delta_pp"
    downward = transitions.loc[transitions[delta_column].lt(0)].copy()
    candidates = trajectories.copy()
    if candidates.empty:
        matches = downward.assign(
            deepest_trajectory_id=None,
            nearest_trajectory_id=None,
            deepest_soc_span_pp=np.nan,
            nearest_delay_hours=np.nan,
            evidence_within_12h=False,
            evidence_within_24h=False,
            evidence_within_48h=False,
            evidence_within_72h=False,
            trajectory_context_status="not_observed_in_raw_lookback",
        )
        return matches, candidates
    evidence_rank = {
        "strict_100_to_0": 0,
        "supporting_99_to_1": 1,
        "supporting_95_to_5": 2,
        "partial": 3,
    }
    candidates["evidence_rank"] = candidates["evidence_class"].map(evidence_rank).fillna(4)
    candidates["deepest_rank"] = candidates.groupby("transition_id")["soc_span_pp"].rank(
        method="first", ascending=False
    )
    candidates["nearest_rank"] = candidates.groupby("transition_id")[
        "delay_to_transition_hours"
    ].rank(method="first", ascending=True)
    rows: list[dict[str, Any]] = []
    for transition in downward.to_dict("records"):
        group = candidates.loc[candidates["transition_id"].eq(transition["transition_id"])]
        if group.empty:
            rows.append({**transition, "trajectory_context_status": "not_observed_in_raw_lookback"})
            continue
        deepest = group.sort_values(
            ["soc_span_pp", "delay_to_transition_hours"], ascending=[False, True]
        ).iloc[0]
        nearest = group.sort_values("delay_to_transition_hours").iloc[0]
        qualifying = group.loc[
            group["evidence_class"].isin(
                {"strict_100_to_0", "supporting_99_to_1", "supporting_95_to_5"}
            )
        ]
        rows.append(
            {
                **transition,
                "deepest_trajectory_id": deepest["trajectory_id"],
                "deepest_evidence_class": deepest["evidence_class"],
                "deepest_soc_start_pct": deepest["soc_start_pct"],
                "deepest_soc_end_pct": deepest["soc_end_pct"],
                "deepest_soc_span_pp": deepest["soc_span_pp"],
                "deepest_delay_hours": deepest["delay_to_transition_hours"],
                "deepest_trajectory_end_reason": deepest["trajectory_end_reason"],
                "deepest_continuity": deepest["continuity"],
                "nearest_trajectory_id": nearest["trajectory_id"],
                "nearest_delay_hours": nearest["delay_to_transition_hours"],
                "evidence_within_12h": bool(qualifying["within_12h"].any()),
                "evidence_within_24h": bool(qualifying["within_24h"].any()),
                "evidence_within_48h": bool(qualifying["within_48h"].any()),
                "evidence_within_72h": bool(qualifying["within_72h"].any()),
                "trajectory_candidate_count": int(len(group)),
                "qualifying_trajectory_count": int(len(qualifying)),
                "trajectory_context_status": (
                    "raw_one_second_qualifying_discharge_resolved"
                    if not qualifying.empty
                    else "raw_one_second_context_only"
                ),
            }
        )
    return pd.DataFrame.from_records(rows), candidates


def reconcile_compact_low_soc_endpoints(
    compact: pd.DataFrame,
    raw: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    if compact.empty:
        return pd.DataFrame()
    end_column = "completion_time_utc" if "completion_time_utc" in compact else "end_time_utc"
    compact_end = pd.to_datetime(compact[end_column], errors="coerce", utc=True)
    compact_soc = pd.to_numeric(compact.get("end_soc_pct"), errors="coerce")
    hints = compact.loc[compact_soc.le(0.0)].copy()
    hints["compact_end_utc"] = compact_end.loc[hints.index]
    records: list[dict[str, Any]] = []
    raw_time = pd.to_datetime(raw.get("time"), errors="coerce", utc=True)
    raw_soc = pd.to_numeric(raw.get("bms_soc_pct"), errors="coerce")
    raw_current = pd.to_numeric(raw.get("bms_current_a"), errors="coerce")
    raw_voltage = pd.to_numeric(raw.get("bms_voltage_v"), errors="coerce")
    raw_soh = pd.to_numeric(raw.get("bms_soh_pct"), errors="coerce")
    for index, hint in hints.iterrows():
        endpoint = pd.Timestamp(hint["compact_end_utc"])
        local = raw.loc[
            raw_time.between(
                endpoint - pd.Timedelta(minutes=10), endpoint + pd.Timedelta(minutes=10)
            )
        ]
        local_soc = raw_soc.loc[local.index] if not local.empty else pd.Series(dtype=float)
        local_all_zero = (
            local_soc.eq(0.0)
            & raw_current.loc[local.index].eq(0.0)
            & raw_voltage.loc[local.index].eq(0.0)
            & raw_soh.loc[local.index].eq(0.0)
        )
        valid_local_soc = local_soc.loc[~local_all_zero]
        matching = (
            trajectories.loc[
                trajectories["trajectory_end_utc"].between(
                    endpoint - pd.Timedelta(hours=6), endpoint + pd.Timedelta(hours=6)
                )
            ]
            if not trajectories.empty
            else trajectories
        )
        if local.empty:
            classification = "one_second_history_missing"
        elif local_all_zero.any() and (
            valid_local_soc.empty or float(valid_local_soc.min()) > 1.0
        ):
            classification = "transient_all_zero_source_row"
        elif valid_local_soc.empty or float(valid_local_soc.min()) > 1.0:
            classification = "compact_raw_disagreement"
        elif not matching.empty and float(matching["soc_start_pct"].max()) >= 95.0:
            classification = "continuation_of_earlier_deep_discharge"
        else:
            classification = "independent_low_soc_discharge"
        raw_min_time = (
            pd.Timestamp(local.loc[valid_local_soc.idxmin(), "time"])
            if not valid_local_soc.empty
            else None
        )
        records.append(
            {
                "compact_endpoint_id": hint.get("discharge_episode_id") or f"compact_{index}",
                "compact_end_utc": endpoint,
                "compact_start_soc_pct": hint.get("start_soc_pct"),
                "compact_end_soc_pct": hint.get("end_soc_pct"),
                "reconciliation_class": classification,
                "raw_min_soc_pct_near_endpoint": float(valid_local_soc.min())
                if not valid_local_soc.empty
                else None,
                "invalid_all_zero_rows_near_endpoint": int(local_all_zero.sum()),
                "raw_min_soc_time_utc": raw_min_time,
                "raw_min_soc_offset_seconds": (
                    (raw_min_time - endpoint).total_seconds()
                    if raw_min_time is not None
                    else None
                ),
                "matched_trajectory_id": (
                    matching.sort_values("soc_span_pp", ascending=False).iloc[0]["trajectory_id"]
                    if not matching.empty
                    else None
                ),
                "one_second_rows_near_endpoint": int(len(local)),
            }
        )
    return pd.DataFrame.from_records(records)
