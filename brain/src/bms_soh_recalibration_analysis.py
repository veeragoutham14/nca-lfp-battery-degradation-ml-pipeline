from __future__ import annotations

import argparse
import html
import json
import logging
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import StorageConfig
from src.io_utils import write_parquet_chunk
from src.seasonal_charge_analysis import (
    _finite,
    _json_safe,
    build_charge_episodes,
    prepare_windows,
    read_compact_history,
    read_latest_soh_audit,
)

logger = logging.getLogger(__name__)

MODEL_FAMILY = "bms_soh_recalibration_evidence"
DETECTOR_VERSION = "v4"
EVIDENCE_HORIZONS_HOURS = (1, 3, 6, 12, 24, 48, 72)
DISCHARGE_HORIZONS_HOURS = EVIDENCE_HORIZONS_HOURS
RAW_ANCHOR_BATCH_SIZE = 8


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "event_id",
            "event_time_utc",
            "event_type",
            "soh_delta_pp",
            "evidence_classification",
            "classification_confidence",
        )
    )


def _serial_raw_root(raw_root: Path, *, serial: str, chemistry: str) -> Path:
    """Resolve legacy or chemistry-partitioned raw storage without mixing layouts."""
    legacy = raw_root / f"serial={serial}"
    canonical = raw_root / f"chemistry={chemistry}" / f"serial={serial}"
    existing = [path for path in (legacy, canonical) if path.exists()]
    if len(existing) > 1:
        raise ValueError(
            f"Battery {serial} exists in both legacy and {chemistry} raw storage"
        )
    if not existing:
        raise FileNotFoundError(f"No raw telemetry exists for serial {serial}")
    return existing[0]


def _overlaps_any(
    start: pd.Timestamp,
    end: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> bool:
    return any(start <= interval_end and end >= interval_start for interval_start, interval_end in intervals)


def read_raw_soh_candidate_windows(
    raw_root: Path,
    *,
    serial: str,
    chemistry: str,
    audit_events: pd.DataFrame,
    search_before_hours: int,
    search_after_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Project only time/SOH row groups overlapping sparse daily candidates."""
    if audit_events.empty:
        return pd.DataFrame(columns=("time", "bms_soh_pct")), {
            "raw_files_considered": 0,
            "raw_row_groups_read": 0,
            "raw_rows_read": 0,
            "raw_rows_retained": 0,
        }
    event_times = pd.to_datetime(audit_events["event_time"], errors="coerce", utc=True).dropna()
    intervals = [
        (
            timestamp - pd.Timedelta(hours=search_before_hours),
            timestamp + pd.Timedelta(hours=search_after_hours),
        )
        for timestamp in event_times
    ]
    serial_root = _serial_raw_root(raw_root, serial=serial, chemistry=chemistry)
    files = sorted(serial_root.rglob("*.parquet"))
    frames: list[pd.DataFrame] = []
    row_groups_read = 0
    rows_read = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        if "time" not in names or "batt_soh_pct" not in names:
            continue
        time_index = names.index("time")
        for row_group_index in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group_index)
            statistics = metadata.column(time_index).statistics
            if statistics and statistics.has_min_max:
                group_start = pd.to_datetime(statistics.min, errors="coerce", utc=True)
                group_end = pd.to_datetime(statistics.max, errors="coerce", utc=True)
                if pd.notna(group_start) and pd.notna(group_end) and not _overlaps_any(
                    group_start, group_end, intervals
                ):
                    continue
            table = parquet.read_row_group(
                row_group_index, columns=("time", "batt_soh_pct")
            )
            row_groups_read += 1
            rows_read += table.num_rows
            frame = table.to_pandas()
            frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
            frame["bms_soh_pct"] = pd.to_numeric(
                frame["batt_soh_pct"], errors="coerce"
            )
            mask = np.zeros(len(frame), dtype=bool)
            for interval_start, interval_end in intervals:
                mask |= frame["time"].ge(interval_start) & frame["time"].le(interval_end)
            retained = frame.loc[mask, ["time", "bms_soh_pct"]]
            if not retained.empty:
                frames.append(retained)
    if not frames:
        result = pd.DataFrame(columns=("time", "bms_soh_pct"))
    else:
        result = (
            pd.concat(frames, ignore_index=True)
            .dropna(subset=("time", "bms_soh_pct"))
            .loc[lambda frame: frame["bms_soh_pct"].between(0, 100)]
            .sort_values("time")
            .drop_duplicates("time", keep="last")
            .reset_index(drop=True)
        )
    return result, {
        "raw_files_considered": len(files),
        "raw_row_groups_read": row_groups_read,
        "raw_rows_read": rows_read,
        "raw_rows_retained": len(result),
    }


def _merge_intervals(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    merged: list[list[pd.Timestamp]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _split_signed_current_ah(
    left_current: np.ndarray,
    right_current: np.ndarray,
    durations: np.ndarray,
    eligible: np.ndarray,
) -> tuple[float, float]:
    """Integrate charge and discharge separately, including mixed-sign intervals."""
    discharge_ah = 0.0
    charge_ah = 0.0
    for left, right, seconds, use_interval in zip(
        left_current, right_current, durations, eligible, strict=True
    ):
        if not use_interval:
            continue
        if left <= 0 and right <= 0:
            discharge_ah += -(left + right) * seconds / 2.0 / 3600.0
        elif left >= 0 and right >= 0:
            charge_ah += (left + right) * seconds / 2.0 / 3600.0
        else:
            negative_current, positive_current = (
                (left, right) if left < 0 else (right, left)
            )
            negative_seconds = seconds * abs(negative_current) / (
                abs(negative_current) + positive_current
            )
            positive_seconds = seconds - negative_seconds
            discharge_ah += abs(negative_current) * negative_seconds / 2.0 / 3600.0
            charge_ah += positive_current * positive_seconds / 2.0 / 3600.0
    return discharge_ah, charge_ah


def read_raw_anchor_windows(
    raw_root: Path,
    *,
    serial: str,
    chemistry: str,
    anchors: pd.Series,
    lookback_hours: int,
    starts: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read projected one-second telemetry for merged, closed raw anchor windows."""
    columns = (
        "time",
        "bms_soc_pct",
        "bms_soh_pct",
        "bms_current_a",
        "bms_voltage_v",
    )
    anchor_times = pd.to_datetime(anchors, errors="coerce", utc=True).dropna()
    if anchor_times.empty:
        return pd.DataFrame(columns=columns), {
            "raw_files_considered": 0,
            "raw_row_groups_read": 0,
            "raw_rows_read": 0,
            "raw_rows_retained": 0,
            "candidate_intervals": 0,
        }
    if starts is None:
        raw_intervals = [
            (
                pd.Timestamp(anchor) - pd.Timedelta(hours=lookback_hours),
                pd.Timestamp(anchor),
            )
            for anchor in anchor_times
        ]
    else:
        start_times = pd.to_datetime(starts, errors="coerce", utc=True)
        paired = pd.DataFrame({"start": start_times, "end": pd.to_datetime(anchors, errors="coerce", utc=True)}).dropna()
        raw_intervals = [
            (pd.Timestamp(row.start), pd.Timestamp(row.end))
            for row in paired.itertuples(index=False)
            if row.start <= row.end
        ]
    intervals = _merge_intervals(raw_intervals)
    serial_root = _serial_raw_root(raw_root, serial=serial, chemistry=chemistry)
    files = sorted(serial_root.rglob("*.parquet"))
    required_source_columns = (
        "time",
        "batt_soc_pct",
        "batt_soh_pct",
        "batt_current_a",
    )
    optional_source_columns = ("batt_voltage_v",)
    frames: list[pd.DataFrame] = []
    row_groups_read = 0
    rows_read = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        if not set(required_source_columns).issubset(names):
            continue
        source_columns = (*required_source_columns, *(name for name in optional_source_columns if name in names))
        time_index = names.index("time")
        for row_group_index in range(parquet.num_row_groups):
            metadata = parquet.metadata.row_group(row_group_index)
            statistics = metadata.column(time_index).statistics
            if statistics and statistics.has_min_max:
                group_start = pd.to_datetime(statistics.min, errors="coerce", utc=True)
                group_end = pd.to_datetime(statistics.max, errors="coerce", utc=True)
                if (
                    pd.notna(group_start)
                    and pd.notna(group_end)
                    and not _overlaps_any(group_start, group_end, intervals)
                ):
                    continue
            table = parquet.read_row_group(row_group_index, columns=source_columns)
            row_groups_read += 1
            rows_read += table.num_rows
            frame = table.to_pandas()
            frame["time"] = pd.to_datetime(frame["time"], errors="coerce", utc=True)
            mask = np.zeros(len(frame), dtype=bool)
            for interval_start, interval_end in intervals:
                mask |= frame["time"].between(interval_start, interval_end)
            retained = frame.loc[mask]
            if not retained.empty:
                frames.append(retained)
    if not frames:
        result = pd.DataFrame(columns=columns)
    else:
        result = pd.concat(frames, ignore_index=True)
        result = result.rename(
            columns={
                "batt_soc_pct": "bms_soc_pct",
                "batt_soh_pct": "bms_soh_pct",
                "batt_current_a": "bms_current_a",
                "batt_voltage_v": "bms_voltage_v",
            }
        )
        if "bms_voltage_v" not in result:
            result["bms_voltage_v"] = np.nan
        for column in columns[1:]:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        result = (
            result.dropna(subset=("time", "bms_soc_pct", "bms_current_a"))
            .loc[lambda frame: frame["bms_soc_pct"].between(0, 100)]
            .sort_values("time")
            .drop_duplicates("time", keep="last")
            .reset_index(drop=True)
        )
    return result.loc[:, columns], {
        "raw_files_considered": len(files),
        "raw_row_groups_read": row_groups_read,
        "raw_rows_read": rows_read,
        "raw_rows_retained": len(result),
        "candidate_intervals": len(intervals),
    }


def _raw_discharge_candidates(
    frame: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
    lookback_hours: int,
    start_soc_pct: float,
    end_soc_pct: float,
    maximum_integrated_gap_seconds: int,
    rest_current_threshold_a: float,
    minimum_coverage_fraction: float,
    minimum_discharge_dominance_fraction: float,
) -> list[dict[str, Any]]:
    """Find raw ordered traversals, retaining rests through the first tied minimum."""
    start = anchor_time - pd.Timedelta(hours=lookback_hours)
    cycle = frame.loc[frame["time"].between(start, anchor_time)].copy()
    cycle = cycle.dropna(subset=("time", "bms_soc_pct", "bms_current_a"))
    cycle = cycle.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if len(cycle) < 2:
        return []
    high = cycle["bms_soc_pct"].ge(start_soc_pct)
    starts = np.flatnonzero(high & ~high.shift(fill_value=False))
    candidates: list[dict[str, Any]] = []
    for position, start_index in enumerate(starts):
        segment_end = starts[position + 1] if position + 1 < len(starts) else len(cycle)
        segment = cycle.iloc[start_index:segment_end].copy()
        low = segment.loc[segment["bms_soc_pct"].le(end_soc_pct)]
        if low.empty:
            continue
        first_low_time = pd.Timestamp(low.iloc[0]["time"])
        minimum_soc = float(low["bms_soc_pct"].min())
        raw_end = pd.Timestamp(low.loc[low["bms_soc_pct"].eq(minimum_soc), "time"].iloc[0])
        bounded = segment.loc[segment["time"].le(raw_end)].copy()
        post_minimum = segment.loc[segment["time"].ge(raw_end)].copy()
        above_low_band = post_minimum["bms_soc_pct"].gt(end_soc_pct).to_numpy()
        tail_stop = int(np.flatnonzero(above_low_band)[0]) if above_low_band.any() else len(post_minimum)
        low_soc_tail = post_minimum.iloc[:tail_stop].copy()
        endpoint_soh = _finite(
            bounded.loc[bounded["time"].eq(raw_end), "bms_soh_pct"].median()
        )
        durations = bounded["time"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
        currents = bounded["bms_current_a"].to_numpy(dtype=float)
        observed = durations > 0
        integrated = observed & (durations <= maximum_integrated_gap_seconds)
        midpoint = (currents[:-1] + currents[1:]) / 2.0
        integrated_seconds = float(durations[integrated].sum())
        elapsed_seconds = float((raw_end - pd.Timestamp(bounded.iloc[0]["time"])).total_seconds())
        unobserved_seconds = max(0.0, elapsed_seconds - integrated_seconds)
        discharge_current = np.clip(-currents, 0.0, None)
        maximum_discharge_current = float(np.nanmax(discharge_current)) if len(discharge_current) else 0.0
        discharge_ah, charge_ah = _split_signed_current_ah(
            currents[:-1], currents[1:], durations, integrated
        )
        confirmed_rest_mask = integrated & (
            np.abs(midpoint) <= rest_current_threshold_a
        )
        potential_gap_rest_mask = (
            observed
            & ~integrated
            & (np.abs(currents[:-1]) <= rest_current_threshold_a)
            & (np.abs(currents[1:]) <= rest_current_threshold_a)
        )
        rest_durations = np.where(integrated, durations, 0.0)
        rest_runs = _true_run_durations_seconds(
            confirmed_rest_mask, rest_durations
        )
        missing_ah_assumption_current_a = maximum_discharge_current
        missing_ah_estimate = (
            missing_ah_assumption_current_a * unobserved_seconds / 3600.0
        )
        discharge_seconds = float(
            durations[integrated & (midpoint < -rest_current_threshold_a)].sum()
        )
        discharge_dominance = (
            discharge_seconds / integrated_seconds if integrated_seconds > 0 else 0.0
        )
        coverage_fraction = (
            integrated_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )
        tail_confirmed_rest_seconds = 0.0
        tail_potential_rest_seconds = 0.0
        if len(low_soc_tail) >= 2:
            tail_durations = (
                low_soc_tail["time"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
            )
            tail_currents = low_soc_tail["bms_current_a"].to_numpy(dtype=float)
            tail_integrated = (tail_durations > 0) & (
                tail_durations <= maximum_integrated_gap_seconds
            )
            tail_midpoint = (tail_currents[:-1] + tail_currents[1:]) / 2.0
            tail_confirmed = tail_integrated & (
                np.abs(tail_midpoint) <= rest_current_threshold_a
            )
            tail_potential = (
                (tail_durations > 0)
                & ~tail_integrated
                & (np.abs(tail_currents[:-1]) <= rest_current_threshold_a)
                & (np.abs(tail_currents[1:]) <= rest_current_threshold_a)
            )
            tail_confirmed_rest_seconds = float(tail_durations[tail_confirmed].sum())
            tail_potential_rest_seconds = float(tail_durations[tail_potential].sum())
        tail_end = (
            pd.Timestamp(low_soc_tail.iloc[-1]["time"])
            if not low_soc_tail.empty
            else raw_end
        )
        candidates.append(
            {
                "raw_start_time_utc": pd.Timestamp(bounded.iloc[0]["time"]),
                "raw_end_time_utc": raw_end,
                "raw_start_soc_pct": float(bounded.iloc[0]["bms_soc_pct"]),
                "raw_end_soc_pct": minimum_soc,
                "raw_bms_soh_at_cycle_completion_pct": endpoint_soh,
                "raw_soc_drop_pct": float(bounded.iloc[0]["bms_soc_pct"]) - minimum_soc,
                "first_low_band_time_utc": first_low_time,
                "minimum_soc_time_utc": raw_end,
                "terminal_low_soc_tail_minutes": (
                    tail_end - first_low_time
                ).total_seconds() / 60.0,
                "post_minimum_low_soc_tail_minutes": (
                    tail_end - raw_end
                ).total_seconds() / 60.0,
                "terminal_low_soc_rest_tail_minutes": (
                    tail_confirmed_rest_seconds / 60.0
                ),
                "terminal_low_soc_potential_rest_tail_minutes": (
                    tail_potential_rest_seconds / 60.0
                ),
                "rest_minutes_inside_cycle": (
                    float(durations[confirmed_rest_mask].sum()) / 60.0
                ),
                "potential_gap_spanning_rest_minutes": (
                    float(durations[potential_gap_rest_mask].sum()) / 60.0
                ),
                "longest_rest_minutes": max(rest_runs, default=0.0) / 60.0,
                "rest_interruption_count": sum(duration >= 60.0 for duration in rest_runs),
                "negative_current_seconds": float(
                    durations[observed & (midpoint < -rest_current_threshold_a)].sum()
                ),
                "positive_current_seconds": float(
                    durations[observed & (midpoint > rest_current_threshold_a)].sum()
                ),
                "raw_sample_rows": int(len(bounded)),
                "raw_coverage_fraction": coverage_fraction,
                "raw_discharge_dominance_fraction": discharge_dominance,
                "raw_discharge_qualified": bool(
                    coverage_fraction >= minimum_coverage_fraction
                    and discharge_dominance
                    >= minimum_discharge_dominance_fraction
                    and discharge_ah > 0
                ),
                "maximum_raw_gap_seconds": float(durations.max()) if len(durations) else 0.0,
                "unobserved_duration_minutes": unobserved_seconds / 60.0,
                "unobserved_discharge_ah_estimate": missing_ah_estimate,
                "missing_ah_assumption_current_a": (
                    missing_ah_assumption_current_a
                ),
                "missing_ah_estimation_method": (
                    "unobserved_duration_at_maximum_observed_discharge_current"
                ),
                "discharged_ah_1s": discharge_ah,
                "charge_intrusion_ah_1s": charge_ah,
                "qualifying_negative_current": bool(
                    np.any(currents < -rest_current_threshold_a)
                ),
            }
        )
    return candidates


def detect_raw_discharge_evidence(
    raw: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    anchor_column: str,
    lookback_hours: int,
    start_soc_pct: float,
    end_soc_pct: float,
    maximum_integrated_gap_seconds: int,
    rest_current_threshold_a: float,
    minimum_coverage_fraction: float = 0.8,
    minimum_discharge_dominance_fraction: float = 0.8,
) -> pd.DataFrame:
    """Apply the identical raw traversal detector to events and matched controls."""
    if anchors.empty:
        result = anchors.copy()
        result["raw_discharge_detection_status"] = pd.Series(dtype=str)
        result["raw_discharge_candidate_count"] = pd.Series(dtype=int)
        result["raw_discharge_ambiguous"] = pd.Series(dtype=bool)
        for column in (
            "raw_start_time_utc",
            "raw_end_time_utc",
            "raw_start_soc_pct",
            "raw_end_soc_pct",
            "raw_soc_drop_pct",
            "first_low_band_time_utc",
            "minimum_soc_time_utc",
            "terminal_low_soc_tail_minutes",
            "post_minimum_low_soc_tail_minutes",
            "terminal_low_soc_rest_tail_minutes",
            "terminal_low_soc_potential_rest_tail_minutes",
            "potential_gap_spanning_rest_minutes",
            "raw_coverage_fraction",
            "raw_discharge_dominance_fraction",
            "raw_discharge_qualified",
            "maximum_raw_gap_seconds",
            "unobserved_discharge_ah_estimate",
            "missing_ah_assumption_current_a",
            "discharged_ah_1s",
            "charge_intrusion_ah_1s",
        ):
            result[column] = pd.Series(dtype=object)
        return result
    records: list[dict[str, Any]] = []
    for row in anchors.to_dict("records"):
        anchor = pd.to_datetime(row.get(anchor_column), errors="coerce", utc=True)
        if pd.isna(anchor):
            records.append({**row, "raw_discharge_detection_status": "invalid_anchor", "raw_discharge_candidate_count": 0, "raw_discharge_ambiguous": False})
            continue
        candidates = _raw_discharge_candidates(
            raw, anchor_time=pd.Timestamp(anchor), lookback_hours=lookback_hours,
            start_soc_pct=start_soc_pct, end_soc_pct=end_soc_pct,
            maximum_integrated_gap_seconds=maximum_integrated_gap_seconds,
            rest_current_threshold_a=rest_current_threshold_a,
            minimum_coverage_fraction=minimum_coverage_fraction,
            minimum_discharge_dominance_fraction=(
                minimum_discharge_dominance_fraction
            ),
        )
        qualified = [
            candidate for candidate in candidates if candidate["qualifying_negative_current"]
        ]
        selected = (
            max(
                qualified,
                key=lambda item: (item["raw_end_time_utc"], item["raw_start_time_utc"]),
            )
            if qualified
            else None
        )
        status = (
            "resolved"
            if selected
            else "soc_traversal_without_negative_discharge"
            if candidates
            else "not_observed"
        )
        records.append({
            **row,
            "raw_discharge_detection_status": status,
            "raw_discharge_candidate_count": len(qualified),
            "raw_discharge_ambiguous": len(qualified) > 1,
            "raw_soc_traversal_count": len(candidates),
            "raw_discharge_selection": "latest_completed_pre_anchor" if selected else None,
            **(selected or {}),
        })
    return pd.DataFrame.from_records(records)


def detect_raw_discharge_evidence_batched(
    raw_root: Path,
    *,
    serial: str,
    chemistry: str,
    anchors: pd.DataFrame,
    anchor_column: str,
    lookback_hours: int,
    start_soc_pct: float,
    end_soc_pct: float,
    maximum_integrated_gap_seconds: int,
    rest_current_threshold_a: float,
    minimum_coverage_fraction: float,
    minimum_discharge_dominance_fraction: float,
    batch_size: int = RAW_ANCHOR_BATCH_SIZE,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan bounded anchor batches so raw telemetry peak memory is capped by batch size."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs: list[pd.DataFrame] = []
    totals = {
        "raw_files_considered": 0,
        "raw_row_groups_read": 0,
        "raw_rows_read": 0,
        "raw_rows_retained": 0,
        "candidate_intervals": 0,
        "anchor_batches": 0,
        "anchors_per_batch_limit": batch_size,
    }
    for offset in range(0, len(anchors), batch_size):
        batch = anchors.iloc[offset : offset + batch_size].copy()
        raw, metrics = read_raw_anchor_windows(
            raw_root,
            serial=serial,
            chemistry=chemistry,
            anchors=batch[anchor_column],
            lookback_hours=lookback_hours,
        )
        outputs.append(
            detect_raw_discharge_evidence(
                raw,
                batch,
                anchor_column=anchor_column,
                lookback_hours=lookback_hours,
                start_soc_pct=start_soc_pct,
                end_soc_pct=end_soc_pct,
                maximum_integrated_gap_seconds=maximum_integrated_gap_seconds,
                rest_current_threshold_a=rest_current_threshold_a,
                minimum_coverage_fraction=minimum_coverage_fraction,
                minimum_discharge_dominance_fraction=(
                    minimum_discharge_dominance_fraction
                ),
            )
        )
        totals["anchor_batches"] += 1
        for key in ("raw_files_considered", "raw_row_groups_read", "raw_rows_read", "raw_rows_retained", "candidate_intervals"):
            totals[key] += int(metrics[key])
    return pd.concat(outputs, ignore_index=True) if outputs else anchors.copy(), totals


def _persistent_transition(
    raw: pd.DataFrame,
    *,
    daily_anchor: pd.Timestamp,
    previous_soh_pct: float,
    new_soh_pct: float,
    search_before_hours: int,
    search_after_hours: int,
    persistence_hours: int,
    minimum_new_level_fraction: float,
    previous_support_minutes: int,
    exact_gap_seconds: int,
) -> dict[str, Any]:
    start = daily_anchor - pd.Timedelta(hours=search_before_hours)
    end = daily_anchor + pd.Timedelta(hours=search_after_hours)
    frame = raw.loc[raw["time"].between(start, end)].copy()
    fallback = {
        "event_time": daily_anchor,
        "event_time_lower_utc": None,
        "event_time_upper_utc": None,
        "event_time_uncertainty_seconds": None,
        "event_time_resolution": "daily_mode_fallback",
        "transition_detection_status": "raw_transition_not_resolved",
        "pre_transition_support_fraction": None,
        "post_transition_support_fraction": None,
    }
    if frame.empty:
        return fallback
    frame = frame.sort_values("time").reset_index(drop=True)
    is_new = frame["bms_soh_pct"].sub(new_soh_pct).abs().le(0.01)
    candidate_indexes = np.flatnonzero(is_new & ~is_new.shift(fill_value=False))
    for candidate_index in candidate_indexes:
        new_start = pd.Timestamp(frame.at[candidate_index, "time"])
        post = frame.loc[
            frame["time"].between(
                new_start,
                new_start + pd.Timedelta(hours=persistence_hours),
                inclusive="both",
            )
        ]
        if post.empty:
            continue
        post_span_minutes = (post["time"].max() - post["time"].min()).total_seconds() / 60.0
        post_fraction = float(post["bms_soh_pct"].sub(new_soh_pct).abs().le(0.01).mean())
        if post_span_minutes < persistence_hours * 60 * 0.8 or post_fraction < minimum_new_level_fraction:
            continue
        pre = frame.loc[
            frame["time"].ge(new_start - pd.Timedelta(hours=2))
            & frame["time"].lt(new_start)
        ]
        if pre.empty:
            continue
        pre_fraction = float(pre["bms_soh_pct"].sub(previous_soh_pct).abs().le(0.01).mean())
        old_samples = pre.loc[pre["bms_soh_pct"].sub(previous_soh_pct).abs().le(0.01)]
        if old_samples.empty or pre_fraction < 0.8:
            continue
        pre_span_minutes = (old_samples["time"].max() - old_samples["time"].min()).total_seconds() / 60.0
        if pre_span_minutes < previous_support_minutes:
            continue
        old_end = pd.Timestamp(old_samples["time"].max())
        uncertainty_seconds = max(0.0, (new_start - old_end).total_seconds())
        exact = uncertainty_seconds <= exact_gap_seconds
        return {
            "event_time": new_start,
            "event_time_lower_utc": old_end,
            "event_time_upper_utc": new_start,
            "event_time_uncertainty_seconds": uncertainty_seconds,
            "event_time_resolution": (
                "persistent_raw_change_point" if exact else "interval_censored_raw_change"
            ),
            "transition_detection_status": "exact" if exact else "interval_censored",
            "pre_transition_support_fraction": pre_fraction,
            "post_transition_support_fraction": post_fraction,
        }
    return fallback


def refine_audit_events(
    audit_events: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    search_before_hours: int,
    search_after_hours: int,
    persistence_hours: int,
    minimum_new_level_fraction: float,
    previous_support_minutes: int,
    exact_gap_seconds: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event in audit_events.to_dict("records"):
        anchor = pd.to_datetime(event.get("event_time"), errors="coerce", utc=True)
        previous_soh = _finite(event.get("previous_soh_pct"))
        new_soh = _finite(event.get("new_soh_pct"))
        if pd.isna(anchor) or previous_soh is None or new_soh is None:
            continue
        refined = _persistent_transition(
            raw,
            daily_anchor=pd.Timestamp(anchor),
            previous_soh_pct=previous_soh,
            new_soh_pct=new_soh,
            search_before_hours=search_before_hours,
            search_after_hours=search_after_hours,
            persistence_hours=persistence_hours,
            minimum_new_level_fraction=minimum_new_level_fraction,
            previous_support_minutes=previous_support_minutes,
            exact_gap_seconds=exact_gap_seconds,
        )
        records.append({**event, "daily_event_anchor_utc": anchor, **refined})
    return pd.DataFrame.from_records(records)


def _state_runs(
    windows: pd.DataFrame,
    *,
    aggregation_minutes: int,
    minimum_window_coverage: float,
) -> pd.DataFrame:
    """Summarise continuous operating state without iterating over raw telemetry."""
    frame = windows.loc[
        :, ["window_start_utc", "operating_state", "coverage_fraction"]
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    frame = frame.sort_values("window_start_utc").reset_index(drop=True)
    frame["state_for_run"] = frame["operating_state"].where(
        frame["coverage_fraction"].ge(minimum_window_coverage), "unknown"
    )
    elapsed_minutes = frame["window_start_utc"].diff().dt.total_seconds().div(60)
    discontinuity = elapsed_minutes.gt(aggregation_minutes * 1.5)
    starts_new_run = discontinuity | frame["state_for_run"].ne(frame["state_for_run"].shift())
    frame["_run"] = starts_new_run.fillna(True).cumsum()
    result = (
        frame.groupby("_run", sort=True, as_index=False)
        .agg(
            operating_state=("state_for_run", "first"),
            start_time_utc=("window_start_utc", "first"),
            last_window_utc=("window_start_utc", "last"),
            mean_coverage_fraction=("coverage_fraction", "mean"),
            window_count=("window_start_utc", "size"),
        )
        .drop(columns="_run")
    )
    result["end_time_utc"] = result["last_window_utc"] + pd.Timedelta(
        minutes=aggregation_minutes
    )
    result["elapsed_minutes"] = (
        result["end_time_utc"] - result["start_time_utc"]
    ).dt.total_seconds().div(60.0)
    return result


def build_calibration_discharge_episodes(
    windows: pd.DataFrame,
    *,
    aggregation_minutes: int,
    episode_gap_minutes: int,
    nominal_capacity_ah: float,
    minimum_window_coverage: float,
    high_soc_start_pct: float,
    low_soc_end_pct: float,
    minimum_soc_drop_pct: float,
    minimum_discharge_dominance_fraction: float,
) -> pd.DataFrame:
    """Build ordered high-to-low discharge sequences from negative-current windows."""
    discharging = windows.loc[
        windows["operating_state"].eq("discharging")
        & windows["coverage_fraction"].ge(minimum_window_coverage)
    ].copy()
    if discharging.empty:
        return pd.DataFrame()
    discharging = discharging.sort_values("window_start_utc")
    start_gap = discharging["window_start_utc"].diff().dt.total_seconds().div(60.0)
    interruption = (start_gap - aggregation_minutes).clip(lower=0.0)
    discharging["_episode_group"] = (
        interruption.gt(episode_gap_minutes).fillna(True).cumsum()
    )
    records: list[dict[str, Any]] = []
    for _, group in discharging.groupby("_episode_group", sort=True):
        group = group.sort_values("window_start_utc")
        start_time = pd.Timestamp(group["window_start_utc"].iloc[0])
        completion_time = pd.Timestamp(group["window_start_utc"].iloc[-1])
        end_time = completion_time + pd.Timedelta(minutes=aggregation_minutes)
        elapsed_minutes = max(
            (end_time - start_time).total_seconds() / 60.0,
            float(aggregation_minutes),
        )
        interval = windows.loc[
            windows["window_start_utc"].ge(start_time)
            & windows["window_start_utc"].le(completion_time)
        ].copy()
        valid_interval = interval.loc[
            interval["coverage_fraction"].ge(minimum_window_coverage)
        ]
        observed_minutes = float(valid_interval["observed_minutes"].sum())
        discharge_minutes = float(group["observed_minutes"].sum())
        discharge_dominance = (
            discharge_minutes / observed_minutes if observed_minutes > 0 else 0.0
        )
        start_soc = _finite(group["bms_soc_max_pct"].iloc[0])
        end_soc = _finite(group["bms_soc_min_pct"].iloc[-1])
        soc_drop = (
            start_soc - end_soc
            if start_soc is not None and end_soc is not None
            else None
        )
        soc_differences = pd.to_numeric(
            group["bms_soc_mean_pct"], errors="coerce"
        ).diff().dropna()
        downward_consistency = (
            float(soc_differences.le(1.0).mean()) if not soc_differences.empty else 1.0
        )
        discharged_ah = float(group["discharge_ah"].sum())
        equivalent_capacity_fraction = discharged_ah / nominal_capacity_ah
        interval_coverage = min(1.0, observed_minutes / elapsed_minutes)
        completed = bool(
            start_soc is not None
            and end_soc is not None
            and start_soc >= high_soc_start_pct
            and end_soc <= low_soc_end_pct
            and soc_drop is not None
            and soc_drop >= minimum_soc_drop_pct
            and discharge_dominance >= minimum_discharge_dominance_fraction
            and interval_coverage >= minimum_window_coverage
            and downward_consistency >= 0.8
        )
        records.append(
            {
                "discharge_episode_id": f"discharge_{len(records) + 1:06d}",
                "start_time_utc": start_time,
                "completion_time_utc": completion_time,
                "end_time_utc": end_time,
                "elapsed_duration_minutes": elapsed_minutes,
                "observed_discharge_minutes": discharge_minutes,
                "episode_coverage_fraction": interval_coverage,
                "discharge_dominance_fraction": discharge_dominance,
                "start_soc_pct": start_soc,
                "end_soc_pct": end_soc,
                "soc_drop_pct": soc_drop,
                "downward_soc_consistency_fraction": downward_consistency,
                "discharged_ah": discharged_ah,
                "discharge_equivalent_capacity_fraction": equivalent_capacity_fraction,
                "complete_high_to_low_discharge": completed,
            }
        )
    return pd.DataFrame.from_records(records)


def build_full_soc_capacity_candidates(
    windows: pd.DataFrame,
    *,
    aggregation_minutes: int,
    nominal_capacity_ah: float,
    start_soc_pct: float,
    end_soc_pct: float,
    maximum_elapsed_hours: int,
    minimum_window_coverage: float,
    maximum_compact_charge_fraction: float = 0.1,
) -> pd.DataFrame:
    """Find separate compact 100-to-0 candidates, including cycles with rest."""
    frame = windows.sort_values("window_start_utc").reset_index(drop=True)
    low = frame["bms_soc_min_pct"].le(end_soc_pct)
    low_crossings = frame.index[low & ~low.shift(fill_value=False)]
    records: list[dict[str, Any]] = []
    used_starts: set[pd.Timestamp] = set()
    for end_index in low_crossings:
        completion_time = pd.Timestamp(frame.at[end_index, "window_start_utc"])
        prior = frame.loc[
            frame["window_start_utc"].between(
                completion_time - pd.Timedelta(hours=maximum_elapsed_hours),
                completion_time,
            )
            & frame["bms_soc_max_pct"].ge(start_soc_pct)
        ]
        if prior.empty:
            continue
        start_time = pd.Timestamp(prior.iloc[-1]["window_start_utc"])
        if start_time in used_starts or start_time >= completion_time:
            continue
        interval = frame.loc[
            frame["window_start_utc"].between(start_time, completion_time)
        ]
        elapsed_minutes = (
            completion_time - start_time
        ).total_seconds() / 60.0 + aggregation_minutes
        observed_minutes = float(interval["observed_minutes"].sum())
        coverage = min(1.0, observed_minutes / elapsed_minutes)
        discharged_ah = float(interval["discharge_ah"].sum())
        charged_ah = float(interval["charge_ah"].sum())
        if (
            coverage < minimum_window_coverage
            or discharged_ah < nominal_capacity_ah * 0.5
            or charged_ah > nominal_capacity_ah * maximum_compact_charge_fraction
        ):
            continue
        start_soc = _finite(interval["bms_soc_max_pct"].iloc[0])
        end_soc = _finite(interval["bms_soc_min_pct"].iloc[-1])
        used_starts.add(start_time)
        records.append(
            {
                "discharge_episode_id": f"full_soc_candidate_{len(records) + 1:06d}",
                "candidate_source": "compact_99_to_1_with_rest",
                "start_time_utc": start_time,
                "completion_time_utc": completion_time,
                "end_time_utc": completion_time
                + pd.Timedelta(minutes=aggregation_minutes),
                "elapsed_duration_minutes": elapsed_minutes,
                "episode_coverage_fraction": coverage,
                "start_soc_pct": start_soc,
                "end_soc_pct": end_soc,
                "soc_drop_pct": (
                    start_soc - end_soc
                    if start_soc is not None and end_soc is not None
                    else None
                ),
                "discharged_ah": discharged_ah,
                "charge_intrusion_ah": charged_ah,
                "discharge_equivalent_capacity_fraction": (
                    discharged_ah / nominal_capacity_ah
                ),
                "complete_high_to_low_discharge": True,
            }
        )
    return pd.DataFrame.from_records(records)


def _last_complete_discharge(
    episodes: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
    lookback_hours: int,
) -> dict[str, Any]:
    empty = {
        "complete_discharge_within_lookback": False,
        "hours_since_complete_discharge": None,
        "complete_discharge_start_time_utc": None,
        "complete_discharge_end_time_utc": None,
        "complete_discharge_start_soc_pct": None,
        "complete_discharge_end_soc_pct": None,
        "complete_discharge_soc_drop_pct": None,
        "complete_discharge_ah": None,
        "complete_discharge_equivalent_capacity_fraction": None,
        "complete_discharge_duration_minutes": None,
        "complete_discharge_coverage_fraction": None,
        "complete_discharge_dominance_fraction": None,
    }
    if episodes.empty:
        return empty
    candidates = episodes.loc[
        episodes["complete_high_to_low_discharge"].fillna(False)
        & episodes["completion_time_utc"].le(anchor_time)
        & episodes["completion_time_utc"].ge(
            anchor_time - pd.Timedelta(hours=lookback_hours)
        )
    ]
    if candidates.empty:
        return empty
    row = candidates.sort_values("completion_time_utc").iloc[-1]
    completion_time = pd.Timestamp(row["completion_time_utc"])
    return {
        "complete_discharge_within_lookback": True,
        "hours_since_complete_discharge": float(
            (anchor_time - completion_time).total_seconds() / 3600.0
        ),
        "complete_discharge_start_time_utc": pd.Timestamp(row["start_time_utc"]),
        "complete_discharge_end_time_utc": completion_time,
        "complete_discharge_start_soc_pct": _finite(row["start_soc_pct"]),
        "complete_discharge_end_soc_pct": _finite(row["end_soc_pct"]),
        "complete_discharge_soc_drop_pct": _finite(row["soc_drop_pct"]),
        "complete_discharge_ah": _finite(row["discharged_ah"]),
        "complete_discharge_equivalent_capacity_fraction": _finite(
            row["discharge_equivalent_capacity_fraction"]
        ),
        "complete_discharge_duration_minutes": _finite(
            row["elapsed_duration_minutes"]
        ),
        "complete_discharge_coverage_fraction": _finite(
            row["episode_coverage_fraction"]
        ),
        "complete_discharge_dominance_fraction": _finite(
            row["discharge_dominance_fraction"]
        ),
    }


def _complete_discharge_horizon_flags(
    episodes: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
) -> dict[str, bool]:
    if episodes.empty:
        return {
            f"complete_discharge_within_{hours}h": False
            for hours in DISCHARGE_HORIZONS_HOURS
        }
    completed = episodes.loc[
        episodes["complete_high_to_low_discharge"].fillna(False)
        & episodes["completion_time_utc"].le(anchor_time)
    ]
    return {
        f"complete_discharge_within_{hours}h": bool(
            completed["completion_time_utc"].ge(
                anchor_time - pd.Timedelta(hours=hours)
            ).any()
        )
        for hours in DISCHARGE_HORIZONS_HOURS
    }


def _longest_true_duration_seconds(mask: np.ndarray, durations: np.ndarray) -> float:
    longest = 0.0
    current = 0.0
    for active, duration in zip(mask, durations, strict=True):
        if active:
            current += float(duration)
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def _true_run_durations_seconds(
    mask: np.ndarray, durations: np.ndarray
) -> list[float]:
    runs: list[float] = []
    current = 0.0
    for active, duration in zip(mask, durations, strict=True):
        if active:
            current += float(duration)
        elif current > 0:
            runs.append(current)
            current = 0.0
    if current > 0:
        runs.append(current)
    return runs


def build_one_second_capacity_validation(
    raw: pd.DataFrame,
    discharge_episodes: pd.DataFrame,
    events: pd.DataFrame,
    *,
    nominal_capacity_ah: float,
    aggregation_minutes: int,
    candidate_start_soc_pct: float,
    candidate_end_soc_pct: float,
    strict_start_soc_pct: float,
    strict_end_soc_pct: float,
    strict_minimum_soc_drop_pct: float,
    minimum_coverage_fraction: float,
    maximum_unobserved_ah: float,
    maximum_charge_intrusion_ah: float,
    maximum_integrated_gap_seconds: int,
    rest_current_threshold_a: float,
    continuous_maximum_rest_minutes: float,
) -> pd.DataFrame:
    """Integrate bounded raw current for qualified high-to-low SOC traversals."""
    qualified = discharge_episodes.loc[
        discharge_episodes.get(
            "complete_high_to_low_discharge",
            pd.Series(False, index=discharge_episodes.index),
        ).fillna(False)
    ].copy()
    if qualified.empty:
        return pd.DataFrame(
            columns=(
                "discharge_episode_id",
                "raw_start_time_utc",
                "raw_end_time_utc",
                "capacity_validation_status",
                "cycle_continuity_class",
                "discharged_ah_1s",
                "direct_capacity_soh_pct",
                "bms_soh_at_cycle_completion_pct",
                "bms_completion_minus_capacity_soh_pp",
                "linked_soh_event_id",
                "bms_soh_after_linked_event_pct",
                "bms_after_event_minus_capacity_soh_pp",
            )
        )
    linkable_events = events.loc[
        events.get(
            "complete_discharge_within_lookback",
            pd.Series(False, index=events.index),
        ).fillna(False)
    ].copy()
    if not linkable_events.empty:
        linkable_events["complete_discharge_end_time_utc"] = pd.to_datetime(
            linkable_events["complete_discharge_end_time_utc"],
            errors="coerce",
            utc=True,
        )
    raw_ordered = (
        raw.sort_values("time")
        .drop_duplicates("time", keep="last")
        .reset_index(drop=True)
    )
    raw_time_values = raw_ordered["time"].to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    records: list[dict[str, Any]] = []
    padding = pd.Timedelta(minutes=aggregation_minutes)
    for episode in qualified.itertuples(index=False):
        compact_start = pd.Timestamp(episode.start_time_utc)
        compact_completion = pd.Timestamp(episode.completion_time_utc)
        lower_ns = (compact_start - padding).value
        upper_ns = (compact_completion + padding).value
        left = int(np.searchsorted(raw_time_values, lower_ns, side="left"))
        right = int(np.searchsorted(raw_time_values, upper_ns, side="right"))
        frame = raw_ordered.iloc[left:right].copy()
        base = {
            "discharge_episode_id": episode.discharge_episode_id,
            "compact_start_time_utc": compact_start,
            "compact_completion_time_utc": compact_completion,
            "raw_start_time_utc": None,
            "raw_end_time_utc": None,
            "cycle_continuity_class": None,
            "linked_soh_event_id": None,
        }
        if frame.empty:
            records.append({**base, "capacity_validation_status": "raw_data_missing"})
            continue
        frame = frame.sort_values("time").drop_duplicates("time", keep="last")
        starts = frame.loc[
            frame["bms_soc_pct"].ge(candidate_start_soc_pct)
            & frame["bms_current_a"].lt(-rest_current_threshold_a)
        ]
        if starts.empty:
            records.append(
                {**base, "capacity_validation_status": "raw_start_boundary_not_resolved"}
            )
            continue
        raw_start = pd.Timestamp(starts.iloc[0]["time"])
        after_start = frame.loc[frame["time"].ge(raw_start)]
        low_rows = after_start.loc[
            after_start["bms_soc_pct"].le(candidate_end_soc_pct)
        ]
        if low_rows.empty:
            records.append(
                {**base, "capacity_validation_status": "raw_end_boundary_not_resolved"}
            )
            continue
        minimum_soc = float(low_rows["bms_soc_pct"].min())
        raw_end = pd.Timestamp(
            low_rows.loc[low_rows["bms_soc_pct"].eq(minimum_soc), "time"].iloc[0]
        )
        cycle = after_start.loc[after_start["time"].le(raw_end)].copy()
        cycle = cycle.dropna(subset=("time", "bms_soc_pct", "bms_current_a"))
        if len(cycle) < 2:
            records.append({**base, "capacity_validation_status": "too_few_raw_rows"})
            continue
        durations = (
            cycle["time"].diff().dt.total_seconds().iloc[1:].to_numpy(dtype=float)
        )
        positive_duration = durations > 0
        integrated = positive_duration & (durations <= maximum_integrated_gap_seconds)
        currents = cycle["bms_current_a"].to_numpy(dtype=float)
        midpoint_current = (currents[:-1] + currents[1:]) / 2.0
        voltages = pd.to_numeric(
            cycle.get("bms_voltage_v", pd.Series(np.nan, index=cycle.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        midpoint_voltage = (voltages[:-1] + voltages[1:]) / 2.0
        integrated_seconds = float(durations[integrated].sum())
        elapsed_seconds = float((raw_end - raw_start).total_seconds())
        unobserved_seconds = max(0.0, elapsed_seconds - integrated_seconds)
        discharge_current = np.clip(-midpoint_current, 0.0, None)
        charge_current = np.clip(midpoint_current, 0.0, None)
        discharge_ah = float(
            np.sum(discharge_current[integrated] * durations[integrated]) / 3600.0
        )
        charge_ah = float(
            np.sum(charge_current[integrated] * durations[integrated]) / 3600.0
        )
        valid_power = integrated & np.isfinite(midpoint_voltage)
        discharged_wh = float(
            np.sum(
                discharge_current[valid_power]
                * midpoint_voltage[valid_power]
                * durations[valid_power]
            )
            / 3600.0
        )
        charge_intrusion_wh = float(
            np.sum(
                charge_current[valid_power]
                * midpoint_voltage[valid_power]
                * durations[valid_power]
            )
            / 3600.0
        )
        maximum_discharge_current = float(np.nanmax(discharge_current))
        unobserved_ah_estimate = (
            maximum_discharge_current * unobserved_seconds / 3600.0
        )
        coverage_fraction = (
            integrated_seconds / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )
        discharge_mask = integrated & (midpoint_current < -rest_current_threshold_a)
        rest_mask = integrated & (np.abs(midpoint_current) <= rest_current_threshold_a)
        charge_mask = integrated & (midpoint_current > rest_current_threshold_a)
        discharge_seconds = float(durations[discharge_mask].sum())
        rest_seconds = float(durations[rest_mask].sum())
        charge_seconds = float(durations[charge_mask].sum())
        longest_rest_seconds = _longest_true_duration_seconds(
            rest_mask, np.where(integrated, durations, 0.0)
        )
        rest_runs = _true_run_durations_seconds(
            rest_mask, np.where(integrated, durations, 0.0)
        )
        rest_interruption_count = sum(duration >= 60.0 for duration in rest_runs)
        start_soc = float(cycle["bms_soc_pct"].iloc[0])
        end_soc = float(cycle["bms_soc_pct"].iloc[-1])
        soc_drop = start_soc - end_soc
        direct_capacity_soh = discharge_ah / nominal_capacity_ah * 100.0
        normalized_capacity = (
            discharge_ah / (soc_drop / 100.0) if soc_drop > 0 else None
        )
        normalized_capacity_soh = (
            normalized_capacity / nominal_capacity_ah * 100.0
            if normalized_capacity is not None
            else None
        )
        strict_boundaries = bool(
            start_soc >= strict_start_soc_pct
            and end_soc <= strict_end_soc_pct
            and soc_drop >= strict_minimum_soc_drop_pct
        )
        if not strict_boundaries:
            status = "exploratory_soc_boundaries"
        elif coverage_fraction < minimum_coverage_fraction:
            status = "insufficient_raw_coverage"
        elif unobserved_ah_estimate > maximum_unobserved_ah:
            status = "excess_missing_ah_uncertainty"
        elif charge_ah > maximum_charge_intrusion_ah:
            status = "charge_intrusion"
        else:
            status = "validation_grade"
        cycle_pattern = (
            "continuous"
            if rest_seconds / 60.0 <= continuous_maximum_rest_minutes
            else "interrupted"
        )
        bms_soh_at_start = _finite(cycle["bms_soh_pct"].dropna().iloc[0]) if cycle["bms_soh_pct"].notna().any() else None
        completion_soh = cycle.loc[
            cycle["time"].ge(raw_end - pd.Timedelta(minutes=1)), "bms_soh_pct"
        ].dropna()
        bms_soh_at_completion = _finite(completion_soh.median())
        linked_event: pd.Series | None = None
        if not linkable_events.empty:
            distances = (
                linkable_events["complete_discharge_end_time_utc"]
                - compact_completion
            ).abs()
            eligible_links = linkable_events.loc[distances.le(padding)]
            if not eligible_links.empty:
                linked_event = eligible_links.loc[distances.loc[eligible_links.index].idxmin()]
        post_update_soh = (
            _finite(linked_event.get("new_soh_pct")) if linked_event is not None else None
        )
        records.append(
            {
                **base,
                "raw_start_time_utc": raw_start,
                "raw_end_time_utc": raw_end,
                "raw_start_soc_pct": start_soc,
                "raw_end_soc_pct": end_soc,
                "raw_soc_drop_pct": soc_drop,
                "raw_sample_rows": int(len(cycle)),
                "elapsed_duration_minutes": elapsed_seconds / 60.0,
                "integrated_duration_minutes": integrated_seconds / 60.0,
                "raw_coverage_fraction": coverage_fraction,
                "maximum_raw_gap_seconds": float(durations.max()),
                "unobserved_duration_minutes": unobserved_seconds / 60.0,
                "unobserved_discharge_ah_estimate": unobserved_ah_estimate,
                "missing_ah_assumption_current_a": maximum_discharge_current,
                "missing_ah_estimation_method": (
                    "unobserved_duration_at_maximum_observed_discharge_current"
                ),
                "discharged_ah_1s": discharge_ah,
                "charge_intrusion_ah_1s": charge_ah,
                "discharged_wh_1s": discharged_wh,
                "charge_intrusion_wh_1s": charge_intrusion_wh,
                "rms_discharge_current_a": (
                    float(np.sqrt(np.average(
                        np.square(discharge_current[discharge_mask]),
                        weights=durations[discharge_mask],
                    )))
                    if discharge_mask.any()
                    else 0.0
                ),
                "p95_discharge_current_a": (
                    float(np.nanpercentile(discharge_current[discharge_mask], 95))
                    if discharge_mask.any()
                    else 0.0
                ),
                "p99_discharge_current_a": (
                    float(np.nanpercentile(discharge_current[discharge_mask], 99))
                    if discharge_mask.any()
                    else 0.0
                ),
                "peak_discharge_current_a": maximum_discharge_current,
                "rms_discharge_c_rate": (
                    float(np.sqrt(np.average(
                        np.square(discharge_current[discharge_mask]),
                        weights=durations[discharge_mask],
                    ))) / nominal_capacity_ah
                    if discharge_mask.any()
                    else 0.0
                ),
                "peak_discharge_c_rate": maximum_discharge_current / nominal_capacity_ah,
                "direct_capacity_soh_pct": direct_capacity_soh,
                "soc_normalized_capacity_ah": normalized_capacity,
                "soc_normalized_capacity_soh_pct": normalized_capacity_soh,
                "strict_full_soc_boundaries": strict_boundaries,
                "capacity_validation_status": status,
                "cycle_continuity_class": cycle_pattern,
                "active_discharge_minutes": discharge_seconds / 60.0,
                "rest_minutes_inside_cycle": rest_seconds / 60.0,
                "charge_minutes_inside_cycle": charge_seconds / 60.0,
                "rest_fraction": rest_seconds / integrated_seconds if integrated_seconds else None,
                "discharge_dominance_fraction_1s": discharge_seconds / integrated_seconds if integrated_seconds else None,
                "rest_interruption_count": rest_interruption_count,
                "longest_rest_minutes": longest_rest_seconds / 60.0,
                "bms_soh_at_cycle_start_pct": bms_soh_at_start,
                "bms_soh_at_cycle_completion_pct": bms_soh_at_completion,
                "bms_completion_minus_capacity_soh_pp": (
                    bms_soh_at_completion - direct_capacity_soh
                    if bms_soh_at_completion is not None
                    else None
                ),
                "bms_completion_minus_normalized_soh_pp": (
                    bms_soh_at_completion - normalized_capacity_soh
                    if bms_soh_at_completion is not None
                    and normalized_capacity_soh is not None
                    else None
                ),
                "linked_soh_event_id": (
                    linked_event.get("event_id") if linked_event is not None else None
                ),
                "linked_soh_event_time_utc": (
                    linked_event.get("event_time_utc") if linked_event is not None else None
                ),
                "hours_to_linked_soh_event": (
                    (pd.Timestamp(linked_event["event_time_utc"]) - raw_end).total_seconds()
                    / 3600.0
                    if linked_event is not None
                    else None
                ),
                "linked_soh_event_delta_pp": (
                    _finite(linked_event.get("soh_delta_pp"))
                    if linked_event is not None
                    else None
                ),
                "bms_soh_after_linked_event_pct": post_update_soh,
                "bms_after_event_minus_capacity_soh_pp": (
                    post_update_soh - direct_capacity_soh
                    if post_update_soh is not None
                    else None
                ),
                "bms_after_event_minus_normalized_soh_pp": (
                    post_update_soh - normalized_capacity_soh
                    if post_update_soh is not None
                    and normalized_capacity_soh is not None
                    else None
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_capacity_validation(
    validations: pd.DataFrame,
    *,
    minimum_coverage_fraction: float,
    maximum_unobserved_ah: float,
    maximum_charge_intrusion_ah: float,
) -> dict[str, Any]:
    if validations.empty:
        return {
            "available": False,
            "reason": "No qualified compact discharge episodes were available.",
            "cycle_rows": 0,
        }
    resolved = validations.loc[
        validations["raw_start_time_utc"].notna()
        & validations["raw_end_time_utc"].notna()
    ] if {"raw_start_time_utc", "raw_end_time_utc"}.issubset(validations) else validations.iloc[0:0]
    validation_grade = resolved.loc[
        resolved.get("capacity_validation_status", pd.Series(index=resolved.index, dtype=str))
        .eq("validation_grade")
    ]
    linked = validation_grade.loc[validation_grade["linked_soh_event_id"].notna()]
    operational = resolved.loc[
        pd.to_numeric(resolved.get("raw_soc_drop_pct"), errors="coerce").ge(90.0)
        & pd.to_numeric(resolved.get("raw_coverage_fraction"), errors="coerce").ge(
            minimum_coverage_fraction
        )
        & pd.to_numeric(
            resolved.get("unobserved_discharge_ah_estimate"), errors="coerce"
        ).le(maximum_unobserved_ah)
        & pd.to_numeric(resolved.get("charge_intrusion_ah_1s"), errors="coerce").le(
            maximum_charge_intrusion_ah
        )
    ]
    operational_linked = operational.loc[operational["linked_soh_event_id"].notna()].copy()
    if not operational_linked.empty:
        operational_linked["bms_completion_minus_normalized_soh_pp"] = (
            pd.to_numeric(
                operational_linked["bms_soh_at_cycle_completion_pct"], errors="coerce"
            )
            - pd.to_numeric(
                operational_linked["soc_normalized_capacity_soh_pct"], errors="coerce"
            )
        )
        operational_linked["bms_after_event_minus_normalized_soh_pp"] = (
            pd.to_numeric(
                operational_linked["bms_soh_after_linked_event_pct"], errors="coerce"
            )
            - pd.to_numeric(
                operational_linked["soc_normalized_capacity_soh_pct"], errors="coerce"
            )
        )

    def error_metrics(column: str, frame: pd.DataFrame) -> dict[str, float | None]:
        values = pd.to_numeric(frame.get(column), errors="coerce").dropna()
        return {
            "rows": int(len(values)),
            "mae_pp": _finite(values.abs().mean()),
            "bias_pp": _finite(values.mean()),
            "maximum_absolute_error_pp": _finite(values.abs().max()),
        }

    continuity: dict[str, dict[str, Any]] = {}
    for cycle_class in ("continuous", "interrupted"):
        frame = operational.loc[
            operational["cycle_continuity_class"].eq(cycle_class)
        ]
        linked_rows = int(frame["linked_soh_event_id"].notna().sum())
        continuity[cycle_class] = {
            "cycle_rows": int(len(frame)),
            "linked_soh_event_rows": linked_rows,
            "linked_soh_event_rate": linked_rows / len(frame) if len(frame) else None,
            "median_rest_minutes": _finite(
                pd.to_numeric(frame.get("rest_minutes_inside_cycle"), errors="coerce").median()
            ),
            "median_longest_rest_minutes": _finite(
                pd.to_numeric(frame.get("longest_rest_minutes"), errors="coerce").median()
            ),
        }
    return {
        "available": bool(len(resolved)),
        "reason": None if len(resolved) else "Raw one-second cycle boundaries were not resolved.",
        "cycle_rows": int(len(validations)),
        "raw_resolved_cycle_rows": int(len(resolved)),
        "strict_full_boundary_rows": int(
            resolved.get(
                "strict_full_soc_boundaries",
                pd.Series(False, index=resolved.index),
            ).fillna(False).sum()
        ),
        "validation_grade_rows": int(len(validation_grade)),
        "linked_validation_grade_rows": int(len(linked)),
        "median_discharged_ah": _finite(
            pd.to_numeric(validation_grade.get("discharged_ah_1s"), errors="coerce").median()
        ),
        "median_capacity_soh_pct": _finite(
            pd.to_numeric(
                validation_grade.get("direct_capacity_soh_pct"), errors="coerce"
            ).median()
        ),
        "bms_at_completion_error": error_metrics(
            "bms_completion_minus_capacity_soh_pp", validation_grade
        ),
        "bms_after_linked_event_error": error_metrics(
            "bms_after_event_minus_capacity_soh_pp", linked
        ),
        "operational_window_rows": int(len(operational)),
        "linked_operational_window_rows": int(len(operational_linked)),
        "median_operational_soc_span_pp": _finite(
            pd.to_numeric(operational.get("raw_soc_drop_pct"), errors="coerce").median()
        ),
        "median_soc_normalized_capacity_ah": _finite(
            pd.to_numeric(
                operational.get("soc_normalized_capacity_ah"), errors="coerce"
            ).median()
        ),
        "median_soc_normalized_capacity_soh_pct": _finite(
            pd.to_numeric(
                operational.get("soc_normalized_capacity_soh_pct"), errors="coerce"
            ).median()
        ),
        "bms_at_completion_normalized_error": error_metrics(
            "bms_completion_minus_normalized_soh_pp", operational_linked
        ),
        "bms_after_linked_event_normalized_error": error_metrics(
            "bms_after_event_minus_normalized_soh_pp", operational_linked
        ),
        "continuity_comparison": continuity,
        "capacity_label_semantics": (
            "Usable discharged-Ah capacity estimated by timestamp-aware integration of "
            "negative one-second BMS pack current between strict raw SOC boundaries."
        ),
        "operational_window_semantics": (
            "Exploratory capacity extrapolated from integrated Ah divided by the raw "
            "BMS SOC fraction traversed, typically 95-100% to 5%. It assumes the BMS "
            "SOC interval is proportional to usable capacity and is not a strict 100-0 test."
        ),
        "claim_limit": (
            "This estimate is independent of the reported BMS SOH value but still depends "
            "on BMS current calibration and BMS SOC boundary accuracy; it is not laboratory "
            "capacity ground truth."
        ),
    }


def _window_context(
    windows: pd.DataFrame,
    *,
    event_time: pd.Timestamp,
    lookback_hours: int,
    aggregation_minutes: int,
    minimum_window_coverage: float,
    low_soc_pct: float,
    near_full_soc_pct: float,
) -> dict[str, Any]:
    start = event_time - pd.Timedelta(hours=lookback_hours)
    context = windows.loc[
        windows["window_start_utc"].ge(start)
        & windows["window_start_utc"].lt(event_time)
    ].copy()
    if context.empty:
        return {
            "context_observed_minutes": 0.0,
            "context_coverage_fraction": 0.0,
            "minimum_soc_pct": None,
            "maximum_soc_pct": None,
            "reached_low_soc": False,
            "reached_near_full_soc": False,
            "maximum_temperature_c": None,
            "mean_temperature_c": None,
            "p95_temperature_c": None,
            "maximum_voltage_spread_mv": None,
            "p95_voltage_spread_mv": None,
            "largest_feature_gap_minutes": None,
            "charge_episode_window_count": 0,
            "charge_ah_in_context": 0.0,
            "discharge_ah_in_context": 0.0,
            "high_soc_hours": 0.0,
            "low_soc_hours": 0.0,
            "maximum_absolute_c_rate": None,
        }
    context = context.sort_values("window_start_utc")
    valid = context["coverage_fraction"].ge(minimum_window_coverage)
    covered_minutes = float(context.loc[valid, "observed_minutes"].sum())
    gap_minutes = (
        context["window_start_utc"].diff().dt.total_seconds().div(60.0) - aggregation_minutes
    ).clip(lower=0.0)
    missing_window_minutes = (
        aggregation_minutes * (1.0 - context["coverage_fraction"].clip(0.0, 1.0))
    )
    largest_gap = max(
        _finite(gap_minutes.max()) or 0.0,
        _finite(missing_window_minutes.max()) or 0.0,
    )
    return {
        "context_observed_minutes": covered_minutes,
        "context_coverage_fraction": min(1.0, covered_minutes / (lookback_hours * 60.0)),
        "minimum_soc_pct": _finite(context.loc[valid, "bms_soc_min_pct"].min()),
        "maximum_soc_pct": _finite(context.loc[valid, "bms_soc_max_pct"].max()),
        "reached_low_soc": bool(context.loc[valid, "bms_soc_min_pct"].le(low_soc_pct).any()),
        "reached_near_full_soc": bool(
            context.loc[valid, "bms_soc_max_pct"].ge(near_full_soc_pct).any()
        ),
        "maximum_temperature_c": _finite(
            context.loc[valid, "stable_pack_temperature_max_c"].max()
        ),
        "mean_temperature_c": _finite(
            context.loc[valid, "stable_pack_temperature_mean_c"].mean()
        ),
        "p95_temperature_c": _finite(
            context.loc[valid, "stable_pack_temperature_max_c"].quantile(0.95)
        ),
        "maximum_voltage_spread_mv": _finite(
            context.loc[valid, "system_voltage_spread_mv"].max()
        ),
        "p95_voltage_spread_mv": _finite(
            context.loc[valid, "system_voltage_spread_mv"].quantile(0.95)
        ),
        "largest_feature_gap_minutes": largest_gap,
        "charge_episode_window_count": int(
            context.loc[valid, "operating_state"].eq("charging").sum()
        ),
        "charge_ah_in_context": float(context.loc[valid, "charge_ah"].sum()),
        "discharge_ah_in_context": float(context.loc[valid, "discharge_ah"].sum()),
        "high_soc_hours": float(
            context.loc[valid & context["bms_soc_mean_pct"].ge(95), "observed_hours"].sum()
        ),
        "low_soc_hours": float(
            context.loc[valid & context["bms_soc_mean_pct"].le(low_soc_pct), "observed_hours"].sum()
        ),
        "maximum_absolute_c_rate": _finite(
            context.loc[valid, "absolute_c_rate"].max()
        ),
    }


def _last_prior_rest_minutes(
    runs: pd.DataFrame, event_time: pd.Timestamp) -> float | None:
    if runs.empty:
        return None
    prior = runs.loc[
        runs["start_time_utc"].lt(event_time)
        & runs["end_time_utc"].ge(event_time - pd.Timedelta(minutes=15))
    ]
    if prior.empty:
        return None
    run = prior.iloc[-1]
    if run["operating_state"] != "rest":
        return 0.0
    return max(
        0.0,
        (min(pd.Timestamp(run["end_time_utc"]), event_time) - pd.Timestamp(run["start_time_utc"])).total_seconds()
        / 60.0,
    )


def _last_near_full_charge(
    episodes: pd.DataFrame,
    *,
    event_time: pd.Timestamp,
    near_full_soc_pct: float,
    lookback_hours: int,
    minimum_window_coverage: float,
) -> dict[str, Any]:
    empty = {
        "near_full_charge_within_lookback": False,
        "hours_since_near_full_charge": None,
        "near_full_charge_start_time_utc": None,
        "near_full_charge_end_time_utc": None,
        "near_full_charge_start_soc_pct": None,
        "near_full_charge_soc_gain_pct": None,
        "near_full_charge_ah": None,
        "near_full_charge_taper_minutes": None,
        "near_full_charge_maximum_c_rate": None,
    }
    if episodes.empty:
        return empty
    candidates = episodes.loc[
        episodes["end_time_utc"].le(event_time)
        & episodes["end_time_utc"].ge(event_time - pd.Timedelta(hours=lookback_hours))
        & episodes["maximum_soc_pct"].ge(near_full_soc_pct)
        & episodes["episode_coverage_fraction"].ge(minimum_window_coverage)
    ]
    if candidates.empty:
        return empty
    row = candidates.sort_values("end_time_utc").iloc[-1]
    end_time = pd.Timestamp(row["end_time_utc"])
    return {
        "near_full_charge_within_lookback": True,
        "hours_since_near_full_charge": float(
            (event_time - end_time).total_seconds() / 3600.0
        ),
        "near_full_charge_start_time_utc": pd.Timestamp(row["start_time_utc"]),
        "near_full_charge_end_time_utc": end_time,
        "near_full_charge_start_soc_pct": _finite(row.get("start_soc_pct")),
        "near_full_charge_soc_gain_pct": _finite(row.get("soc_gain_pct")),
        "near_full_charge_ah": _finite(row.get("charged_ah")),
        "near_full_charge_taper_minutes": _finite(
            row.get("high_soc_taper_proxy_minutes")
        ),
        "near_full_charge_maximum_c_rate": _finite(row.get("maximum_charge_c_rate")),
    }


def _near_full_horizon_flags(
    episodes: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
    near_full_soc_pct: float,
    minimum_window_coverage: float,
) -> dict[str, bool]:
    if episodes.empty:
        return {f"near_full_within_{hours}h": False for hours in EVIDENCE_HORIZONS_HOURS}
    eligible = episodes.loc[
        episodes["end_time_utc"].le(anchor_time)
        & episodes["maximum_soc_pct"].ge(near_full_soc_pct)
        & episodes["episode_coverage_fraction"].ge(minimum_window_coverage)
    ]
    return {
        f"near_full_within_{hours}h": bool(
            eligible["end_time_utc"].ge(anchor_time - pd.Timedelta(hours=hours)).any()
        )
        for hours in EVIDENCE_HORIZONS_HOURS
    }


def _low_before_near_full(
    windows: pd.DataFrame,
    *,
    lookback_start: pd.Timestamp,
    near_full_end: pd.Timestamp | None,
    low_soc_pct: float,
    minimum_window_coverage: float,
) -> bool:
    if near_full_end is None:
        return False
    prior = windows.loc[
        windows["window_start_utc"].ge(lookback_start)
        & windows["window_start_utc"].lt(near_full_end)
        & windows["coverage_fraction"].ge(minimum_window_coverage)
    ]
    return bool(prior["bms_soc_min_pct"].le(low_soc_pct).any())


def _classify_event(
    *,
    event_type: str,
    context_coverage_fraction: float,
    minimum_context_coverage: float,
    data_gap_minutes: float | None,
    data_gap_risk_minutes: int,
    complete_discharge_within_lookback: bool,
    near_full_charge_within_lookback: bool,
    rest_before_event_minutes: float | None,
    minimum_rest_minutes: int,
    reached_low_soc: bool,
    low_before_near_full: bool,
    meaningful_charge: bool,
    maximum_temperature_c: float | None,
    maximum_voltage_spread_mv: float | None,
    maximum_absolute_c_rate: float | None,
    transition_detection_status: str,
    prior_same_direction_events: int,
) -> tuple[str, str, list[str]]:
    tags: list[str] = []
    if event_type == "change_after_gap" or (data_gap_minutes or 0.0) >= data_gap_risk_minutes:
        tags.append("data_gap_or_estimator_reset_risk")
        if context_coverage_fraction < minimum_context_coverage:
            tags.append("compact_context_coverage_below_minimum")
            return "data_gap_or_estimator_reset_risk", "limited", tags
        return "data_gap_or_estimator_reset_risk", "moderate", tags
    if context_coverage_fraction < minimum_context_coverage:
        return "insufficient_context", "limited", ["compact_context_coverage_below_minimum"]
    rested = (rest_before_event_minutes or 0.0) >= minimum_rest_minutes
    if complete_discharge_within_lookback:
        tags.append("complete_high_to_low_discharge_before_update")
    if near_full_charge_within_lookback:
        tags.append("near_full_charge_before_update")
    if rested:
        tags.append("prolonged_rest_before_update")
    if reached_low_soc:
        tags.append("low_soc_observed_before_update")
    if low_before_near_full:
        tags.append("low_to_near_full_sequence")
    if meaningful_charge:
        tags.append("meaningful_charge_before_update")
    if (maximum_temperature_c or -math.inf) >= 35.0:
        tags.append("temperature_at_or_above_35c")
    if (maximum_voltage_spread_mv or -math.inf) >= 30.0:
        tags.append("cell_voltage_spread_at_or_above_30mv")
    if (maximum_absolute_c_rate or -math.inf) >= 0.5:
        tags.append("absolute_c_rate_at_or_above_0_5c")
    if prior_same_direction_events >= 2:
        tags.append("repeated_same_direction_updates")
    if transition_detection_status == "raw_transition_not_resolved":
        tags.append("daily_timestamp_fallback")
    if complete_discharge_within_lookback:
        if rested:
            return "complete_discharge_and_rest_context", "strong", tags
        return "complete_high_to_low_discharge_context", "strong", tags
    independent_matches = sum(
        (
            near_full_charge_within_lookback,
            rested,
            low_before_near_full,
            (maximum_temperature_c or -math.inf) >= 35.0,
            (maximum_voltage_spread_mv or -math.inf) >= 30.0,
            (maximum_absolute_c_rate or -math.inf) >= 0.5,
        )
    )
    strength = "strong" if independent_matches >= 2 else "moderate"
    if transition_detection_status == "raw_transition_not_resolved":
        strength = "limited"
    if low_before_near_full and rested:
        return "low_to_full_charge_and_rest_context", strength, tags
    if near_full_charge_within_lookback and meaningful_charge and rested:
        return "meaningful_charge_completion_and_rest_context", strength, tags
    if near_full_charge_within_lookback and rested:
        return "near_full_charge_and_rest_context", strength, tags
    if reached_low_soc and rested:
        return "low_soc_and_rest_context", strength, tags
    if low_before_near_full:
        return "low_to_near_full_sequence_context", strength, tags
    if near_full_charge_within_lookback and meaningful_charge:
        return "meaningful_charge_completion_context", strength, tags
    if near_full_charge_within_lookback:
        return "near_full_charge_context", strength, tags
    if reached_low_soc:
        return "low_soc_operation_context", strength, tags
    if rested:
        return "prolonged_rest_context", strength, tags
    if prior_same_direction_events >= 2:
        return "recurring_estimator_update_pattern", strength, tags
    return "no_identifiable_operating_anchor", "limited", ["no_matching_context_pattern"]


def build_recalibration_events(
    windows: pd.DataFrame,
    audit_events: pd.DataFrame,
    *,
    aggregation_minutes: int,
    nominal_capacity_ah: float,
    minimum_window_coverage: float,
    minimum_context_coverage: float,
    context_lookback_hours: int,
    near_full_soc_pct: float,
    low_soc_pct: float,
    minimum_rest_minutes: int,
    data_gap_risk_minutes: int,
    recurrence_window_days: int,
    discharge_start_soc_pct: float = 95.0,
    discharge_end_soc_pct: float = 5.0,
    minimum_discharge_soc_drop_pct: float = 90.0,
    minimum_discharge_dominance_fraction: float = 0.8,
    discharge_episode_gap_minutes: int = 30,
    calibration_discharge_episodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach auditable operating evidence to resolved or censored BMS SOH steps."""
    if audit_events.empty:
        return _empty_events()
    candidates = audit_events.loc[
        audit_events["event_type"].isin(("step_drop", "step_recovery", "change_after_gap"))
    ].copy()
    if candidates.empty:
        return _empty_events()
    candidates["event_time"] = pd.to_datetime(
        candidates["event_time"], errors="coerce", utc=True
    )
    candidates = candidates.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)
    runs = _state_runs(
        windows,
        aggregation_minutes=aggregation_minutes,
        minimum_window_coverage=minimum_window_coverage,
    )
    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=aggregation_minutes,
        episode_gap_minutes=15,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_soc_gain_pct=20.0,
    )
    if not episodes.empty:
        episodes["end_time_utc"] = pd.to_datetime(
            episodes["end_time_utc"], errors="coerce", utc=True
        )
        episodes["maximum_soc_pct"] = pd.to_numeric(
            episodes["maximum_soc_pct"], errors="coerce"
        )
        episodes["episode_coverage_fraction"] = pd.to_numeric(
            episodes["episode_coverage_fraction"], errors="coerce"
        ).fillna(0.0)
    discharge_episodes = (
        calibration_discharge_episodes.copy()
        if calibration_discharge_episodes is not None
        else build_calibration_discharge_episodes(
            windows,
            aggregation_minutes=aggregation_minutes,
            episode_gap_minutes=discharge_episode_gap_minutes,
            nominal_capacity_ah=nominal_capacity_ah,
            minimum_window_coverage=minimum_window_coverage,
            high_soc_start_pct=discharge_start_soc_pct,
            low_soc_end_pct=discharge_end_soc_pct,
            minimum_soc_drop_pct=minimum_discharge_soc_drop_pct,
            minimum_discharge_dominance_fraction=(
                minimum_discharge_dominance_fraction
            ),
        )
    )
    records: list[dict[str, Any]] = []
    for index, event in candidates.iterrows():
        event_time = pd.Timestamp(event["event_time"])
        event_type = str(event["event_type"])
        context = _window_context(
            windows,
            event_time=event_time,
            lookback_hours=context_lookback_hours,
            aggregation_minutes=aggregation_minutes,
            minimum_window_coverage=minimum_window_coverage,
            low_soc_pct=low_soc_pct,
            near_full_soc_pct=near_full_soc_pct,
        )
        transition_status = str(
            event.get("transition_detection_status") or "raw_transition_not_resolved"
        )
        rest_minutes = (
            _last_prior_rest_minutes(runs, event_time)
            if transition_status == "exact"
            else None
        )
        charge_evidence = _last_near_full_charge(
            episodes,
            event_time=event_time,
            near_full_soc_pct=near_full_soc_pct,
            lookback_hours=context_lookback_hours,
            minimum_window_coverage=minimum_window_coverage,
        )
        prior = candidates.loc[
            candidates["event_time"].lt(event_time)
            & candidates["event_time"].ge(event_time - pd.Timedelta(days=recurrence_window_days))
            & candidates["event_type"].eq(event_type)
        ]
        horizon_flags = _near_full_horizon_flags(
            episodes,
            anchor_time=event_time,
            near_full_soc_pct=near_full_soc_pct,
            minimum_window_coverage=minimum_window_coverage,
        )
        discharge_evidence = _last_complete_discharge(
            discharge_episodes,
            anchor_time=event_time,
            lookback_hours=context_lookback_hours,
        )
        discharge_horizon_flags = _complete_discharge_horizon_flags(
            discharge_episodes,
            anchor_time=event_time,
        )
        near_full_charge = bool(charge_evidence["near_full_charge_within_lookback"])
        soc_gain = _finite(charge_evidence["near_full_charge_soc_gain_pct"])
        charged_ah = _finite(charge_evidence["near_full_charge_ah"])
        meaningful_charge = bool(
            (soc_gain is not None and soc_gain >= 20.0)
            or (charged_ah is not None and charged_ah >= nominal_capacity_ah * 0.2)
        )
        low_before_near_full = _low_before_near_full(
            windows,
            lookback_start=event_time - pd.Timedelta(hours=context_lookback_hours),
            near_full_end=charge_evidence["near_full_charge_end_time_utc"],
            low_soc_pct=low_soc_pct,
            minimum_window_coverage=minimum_window_coverage,
        )
        audit_gap_minutes = (_finite(event.get("max_gap_seconds")) or 0.0) / 60.0
        gap_minutes = max(audit_gap_minutes, context["largest_feature_gap_minutes"] or 0.0)
        classification, match_strength, tags = _classify_event(
            event_type=event_type,
            context_coverage_fraction=float(context["context_coverage_fraction"]),
            minimum_context_coverage=minimum_context_coverage,
            data_gap_minutes=gap_minutes,
            data_gap_risk_minutes=data_gap_risk_minutes,
            complete_discharge_within_lookback=bool(
                discharge_evidence["complete_discharge_within_lookback"]
            ),
            near_full_charge_within_lookback=near_full_charge,
            rest_before_event_minutes=rest_minutes,
            minimum_rest_minutes=minimum_rest_minutes,
            reached_low_soc=bool(context["reached_low_soc"]),
            low_before_near_full=low_before_near_full,
            meaningful_charge=meaningful_charge,
            maximum_temperature_c=context["maximum_temperature_c"],
            maximum_voltage_spread_mv=context["maximum_voltage_spread_mv"],
            maximum_absolute_c_rate=context["maximum_absolute_c_rate"],
            transition_detection_status=transition_status,
            prior_same_direction_events=len(prior),
        )
        legacy_confidence = {
            "strong": "high",
            "moderate": "medium",
            "limited": "limited",
        }[match_strength]
        records.append(
            {
                "event_id": f"recalibration_{index + 1:05d}",
                "event_time_utc": event_time,
                "daily_event_anchor_utc": event.get("daily_event_anchor_utc"),
                "event_time_lower_utc": event.get("event_time_lower_utc"),
                "event_time_upper_utc": event.get("event_time_upper_utc"),
                "event_time_uncertainty_seconds": _finite(
                    event.get("event_time_uncertainty_seconds")
                ),
                "event_time_resolution": event.get("event_time_resolution")
                or "daily_mode_fallback",
                "transition_detection_status": transition_status,
                "pre_transition_support_fraction": _finite(
                    event.get("pre_transition_support_fraction")
                ),
                "post_transition_support_fraction": _finite(
                    event.get("post_transition_support_fraction")
                ),
                "event_type": event_type,
                "previous_soh_pct": _finite(event.get("previous_soh_pct")),
                "new_soh_pct": _finite(event.get("new_soh_pct")),
                "soh_delta_pp": _finite(event.get("soh_delta_pct")),
                "audit_coverage_fraction": _finite(event.get("coverage_fraction")),
                "audit_cumulative_discharge_ah": _finite(event.get("cumulative_all_discharge_ah")),
                "context_lookback_hours": context_lookback_hours,
                "context_observed_minutes": context["context_observed_minutes"],
                "context_coverage_fraction": context["context_coverage_fraction"],
                "minimum_soc_pct": context["minimum_soc_pct"],
                "maximum_soc_pct": context["maximum_soc_pct"],
                "reached_low_soc": context["reached_low_soc"],
                "reached_near_full_soc": context["reached_near_full_soc"],
                "near_full_charge_within_lookback": near_full_charge,
                "hours_since_near_full_charge": charge_evidence[
                    "hours_since_near_full_charge"
                ],
                "near_full_charge_start_soc_pct": charge_evidence[
                    "near_full_charge_start_soc_pct"
                ],
                "near_full_charge_soc_gain_pct": soc_gain,
                "near_full_charge_ah": charged_ah,
                "near_full_charge_taper_minutes": charge_evidence[
                    "near_full_charge_taper_minutes"
                ],
                "near_full_charge_maximum_c_rate": charge_evidence[
                    "near_full_charge_maximum_c_rate"
                ],
                "meaningful_charge_before_update": meaningful_charge,
                "low_before_near_full": low_before_near_full,
                **discharge_evidence,
                "rest_before_event_minutes": rest_minutes,
                "mean_temperature_c": context["mean_temperature_c"],
                "p95_temperature_c": context["p95_temperature_c"],
                "maximum_temperature_c": context["maximum_temperature_c"],
                "p95_voltage_spread_mv": context["p95_voltage_spread_mv"],
                "maximum_voltage_spread_mv": context["maximum_voltage_spread_mv"],
                "largest_data_gap_minutes": gap_minutes,
                "charging_window_count_in_context": context["charge_episode_window_count"],
                "charge_ah_in_context": context["charge_ah_in_context"],
                "discharge_ah_in_context": context["discharge_ah_in_context"],
                "high_soc_hours": context["high_soc_hours"],
                "low_soc_hours": context["low_soc_hours"],
                "maximum_absolute_c_rate": context["maximum_absolute_c_rate"],
                "prior_same_direction_events": int(len(prior)),
                "evidence_classification": classification,
                "primary_context_pattern": classification,
                "context_match_strength": match_strength,
                "classification_confidence": legacy_confidence,
                "evidence_tags": "; ".join(tags),
                **discharge_horizon_flags,
                **horizon_flags,
                "interpretation": "temporal_evidence_only_not_electrochemical_capacity_loss",
            }
        )
    return pd.DataFrame.from_records(records)


def _anchor_evidence(
    windows: pd.DataFrame,
    charge_episodes: pd.DataFrame,
    discharge_episodes: pd.DataFrame,
    runs: pd.DataFrame,
    *,
    anchor_time: pd.Timestamp,
    aggregation_minutes: int,
    nominal_capacity_ah: float,
    minimum_window_coverage: float,
    context_lookback_hours: int,
    near_full_soc_pct: float,
    low_soc_pct: float,
    minimum_rest_minutes: int,
) -> dict[str, Any]:
    context = _window_context(
        windows,
        event_time=anchor_time,
        lookback_hours=context_lookback_hours,
        aggregation_minutes=aggregation_minutes,
        minimum_window_coverage=minimum_window_coverage,
        low_soc_pct=low_soc_pct,
        near_full_soc_pct=near_full_soc_pct,
    )
    charge = _last_near_full_charge(
        charge_episodes,
        event_time=anchor_time,
        near_full_soc_pct=near_full_soc_pct,
        lookback_hours=context_lookback_hours,
        minimum_window_coverage=minimum_window_coverage,
    )
    soc_gain = _finite(charge["near_full_charge_soc_gain_pct"])
    charged_ah = _finite(charge["near_full_charge_ah"])
    meaningful_charge = bool(
        (soc_gain is not None and soc_gain >= 20.0)
        or (charged_ah is not None and charged_ah >= nominal_capacity_ah * 0.2)
    )
    low_before_full = _low_before_near_full(
        windows,
        lookback_start=anchor_time - pd.Timedelta(hours=context_lookback_hours),
        near_full_end=charge["near_full_charge_end_time_utc"],
        low_soc_pct=low_soc_pct,
        minimum_window_coverage=minimum_window_coverage,
    )
    rest_minutes = _last_prior_rest_minutes(runs, anchor_time)
    discharge = _last_complete_discharge(
        discharge_episodes,
        anchor_time=anchor_time,
        lookback_hours=context_lookback_hours,
    )
    return {
        "anchor_time_utc": anchor_time,
        "context_coverage_fraction": context["context_coverage_fraction"],
        "reached_low_soc": context["reached_low_soc"],
        "low_before_near_full": low_before_full,
        "meaningful_charge_before_update": meaningful_charge,
        "prolonged_rest": bool((rest_minutes or 0.0) >= minimum_rest_minutes),
        "rest_before_event_minutes": rest_minutes,
        "maximum_temperature_c": context["maximum_temperature_c"],
        "maximum_voltage_spread_mv": context["maximum_voltage_spread_mv"],
        "maximum_absolute_c_rate": context["maximum_absolute_c_rate"],
        **discharge,
        **_complete_discharge_horizon_flags(
            discharge_episodes,
            anchor_time=anchor_time,
        ),
        **_near_full_horizon_flags(
            charge_episodes,
            anchor_time=anchor_time,
            near_full_soc_pct=near_full_soc_pct,
            minimum_window_coverage=minimum_window_coverage,
        ),
    }


def _seasonal_distance(left: pd.Timestamp, right: pd.Timestamp) -> int:
    difference = abs(left.dayofyear - right.dayofyear)
    return min(difference, 366 - difference)


def build_matched_control_windows(
    windows: pd.DataFrame,
    audit_daily: pd.DataFrame,
    events: pd.DataFrame,
    *,
    aggregation_minutes: int,
    nominal_capacity_ah: float,
    minimum_window_coverage: float,
    minimum_context_coverage: float,
    context_lookback_hours: int,
    near_full_soc_pct: float,
    low_soc_pct: float,
    minimum_rest_minutes: int,
    controls_per_event: int,
    event_exclusion_days: int,
    discharge_start_soc_pct: float = 95.0,
    discharge_end_soc_pct: float = 5.0,
    minimum_discharge_soc_drop_pct: float = 90.0,
    minimum_discharge_dominance_fraction: float = 0.8,
    discharge_episode_gap_minutes: int = 30,
    calibration_discharge_episodes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events.empty or audit_daily.empty:
        return pd.DataFrame()
    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=aggregation_minutes,
        episode_gap_minutes=15,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_soc_gain_pct=20.0,
    )
    if not episodes.empty:
        for column in ("start_time_utc", "end_time_utc"):
            episodes[column] = pd.to_datetime(episodes[column], errors="coerce", utc=True)
    discharge_episodes = (
        calibration_discharge_episodes.copy()
        if calibration_discharge_episodes is not None
        else build_calibration_discharge_episodes(
            windows,
            aggregation_minutes=aggregation_minutes,
            episode_gap_minutes=discharge_episode_gap_minutes,
            nominal_capacity_ah=nominal_capacity_ah,
            minimum_window_coverage=minimum_window_coverage,
            high_soc_start_pct=discharge_start_soc_pct,
            low_soc_end_pct=discharge_end_soc_pct,
            minimum_soc_drop_pct=minimum_discharge_soc_drop_pct,
            minimum_discharge_dominance_fraction=(
                minimum_discharge_dominance_fraction
            ),
        )
    )
    runs = _state_runs(
        windows,
        aggregation_minutes=aggregation_minutes,
        minimum_window_coverage=minimum_window_coverage,
    )
    daily = audit_daily.copy()
    daily["first_sample_time"] = pd.to_datetime(
        daily["first_sample_time"], errors="coerce", utc=True
    )
    daily["coverage_fraction"] = pd.to_numeric(
        daily["coverage_fraction"], errors="coerce"
    ).fillna(0.0)
    daily = daily.loc[
        daily["first_sample_time"].notna()
        & daily["coverage_fraction"].ge(minimum_context_coverage)
    ].copy()
    event_times = pd.to_datetime(events["event_time_utc"], errors="coerce", utc=True)
    event_days = event_times.dt.floor("D")
    daily = daily.loc[
        ~daily["first_sample_time"].dt.floor("D").map(
            lambda day: any(
                abs((day - event_day).days) <= event_exclusion_days
                for event_day in event_days
            )
        )
    ].copy()
    used_control_days: set[pd.Timestamp] = set()
    records: list[dict[str, Any]] = []
    for event in events.to_dict("records"):
        event_time = pd.Timestamp(event["event_time_utc"])
        event_coverage = float(event.get("context_coverage_fraction") or 0.0)
        candidates = daily.copy()
        candidates["control_day"] = candidates["first_sample_time"].dt.floor("D")
        candidates = candidates.loc[~candidates["control_day"].isin(used_control_days)]
        candidates["match_score"] = candidates.apply(
            lambda row, candidate_time=event_time, candidate_coverage=event_coverage: (
                _seasonal_distance(candidate_time, pd.Timestamp(row["first_sample_time"]))
                + 30.0 * abs(float(row["coverage_fraction"]) - candidate_coverage)
            ),
            axis=1,
        )
        selected = candidates.nsmallest(controls_per_event, "match_score")
        event_offset = event_time - event_time.floor("D")
        for _control_index, control in selected.iterrows():
            control_day = pd.Timestamp(control["control_day"])
            used_control_days.add(control_day)
            anchor_time = control_day + event_offset
            evidence = _anchor_evidence(
                windows,
                episodes,
                discharge_episodes,
                runs,
                anchor_time=anchor_time,
                aggregation_minutes=aggregation_minutes,
                nominal_capacity_ah=nominal_capacity_ah,
                minimum_window_coverage=minimum_window_coverage,
                context_lookback_hours=context_lookback_hours,
                near_full_soc_pct=near_full_soc_pct,
                low_soc_pct=low_soc_pct,
                minimum_rest_minutes=minimum_rest_minutes,
            )
            records.append(
                {
                    "control_id": f"{event['event_id']}_control_{len(records) + 1:04d}",
                    "matched_event_id": event["event_id"],
                    "match_score": float(control["match_score"]),
                    "source_day_coverage_fraction": float(control["coverage_fraction"]),
                    **evidence,
                }
            )
    return pd.DataFrame.from_records(records)


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    z = 1.96
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def build_evidence_comparison(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    minimum_rest_minutes: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    event_features = events.copy()
    event_features["prolonged_rest"] = (
        pd.to_numeric(event_features["rest_before_event_minutes"], errors="coerce")
        .fillna(0.0)
        .ge(minimum_rest_minutes)
    )
    definitions = {
        **{
            f"complete_discharge_within_{hours}h": (
                "Qualified 95-100% to 0-5% discharge completed within "
                f"{hours} {'hour' if hours == 1 else 'hours'}"
            )
            for hours in DISCHARGE_HORIZONS_HOURS
        },
        **{
            f"near_full_within_{hours}h": f"Reached near-full SOC within {hours} hours"
            for hours in EVIDENCE_HORIZONS_HOURS
        },
        "meaningful_charge_before_update": "Meaningful charge ending near full",
        "low_before_near_full": "Low-to-near-full SOC sequence",
        "reached_low_soc": "Reached low SOC in the evidence window",
        "prolonged_rest": "Prolonged rest immediately before the anchor",
        "temperature_at_or_above_35c": "Maximum pack temperature at or above 35 C",
        "spread_at_or_above_30mv": "Maximum cell-voltage spread at or above 30 mV",
        "absolute_c_rate_at_or_above_0_5c": "Absolute C-rate at or above 0.5 C",
    }
    for frame in (event_features, controls):
        if frame.empty:
            continue
        frame["temperature_at_or_above_35c"] = pd.to_numeric(
            frame["maximum_temperature_c"], errors="coerce"
        ).ge(35.0)
        frame["spread_at_or_above_30mv"] = pd.to_numeric(
            frame["maximum_voltage_spread_mv"], errors="coerce"
        ).ge(30.0)
        frame["absolute_c_rate_at_or_above_0_5c"] = pd.to_numeric(
            frame["maximum_absolute_c_rate"], errors="coerce"
        ).ge(0.5)
    rows: list[dict[str, Any]] = []
    for feature, label in definitions.items():
        event_total = len(event_features)
        control_total = len(controls)
        event_count = int(event_features.get(feature, pd.Series(False, index=event_features.index)).fillna(False).sum())
        control_count = int(controls.get(feature, pd.Series(False, index=controls.index)).fillna(False).sum())
        event_rate = event_count / event_total if event_total else None
        control_rate = control_count / control_total if control_total else None
        event_low, event_high = _wilson_interval(event_count, event_total)
        control_low, control_high = _wilson_interval(control_count, control_total)
        rows.append(
            {
                "feature_key": feature,
                "feature_label": label,
                "event_count": event_count,
                "event_total": event_total,
                "event_rate": event_rate,
                "event_rate_ci95_low": event_low,
                "event_rate_ci95_high": event_high,
                "control_count": control_count,
                "control_total": control_total,
                "control_rate": control_rate,
                "control_rate_ci95_low": control_low,
                "control_rate_ci95_high": control_high,
                "risk_difference": (
                    event_rate - control_rate
                    if event_rate is not None and control_rate is not None
                    else None
                ),
                "prevalence_lift": (
                    event_rate / control_rate
                    if event_rate is not None and control_rate not in (None, 0.0)
                    else None
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


MODEL_FEATURES = (
    "complete_discharge_within_12h",
    "complete_discharge_within_24h",
    "complete_discharge_within_48h",
    "complete_discharge_within_72h",
    "near_full_within_1h",
    "near_full_within_3h",
    "near_full_within_6h",
    "near_full_within_12h",
    "meaningful_charge_before_update",
    "low_before_near_full",
    "reached_low_soc",
    "prolonged_rest",
    "maximum_temperature_c",
    "maximum_voltage_spread_mv",
    "maximum_absolute_c_rate",
    "context_coverage_fraction",
)
BOOLEAN_MODEL_FEATURES = frozenset(
    {
        "complete_discharge_within_12h",
        "complete_discharge_within_24h",
        "complete_discharge_within_48h",
        "complete_discharge_within_72h",
        "near_full_within_1h",
        "near_full_within_3h",
        "near_full_within_6h",
        "near_full_within_12h",
        "meaningful_charge_before_update",
        "low_before_near_full",
        "reached_low_soc",
        "prolonged_rest",
    }
)


def _logistic_feature_frame(frame: pd.DataFrame, *, minimum_rest_minutes: int) -> pd.DataFrame:
    result = frame.copy()
    if "prolonged_rest" not in result:
        result["prolonged_rest"] = (
            pd.to_numeric(result.get("rest_before_event_minutes"), errors="coerce")
            .fillna(0.0)
            .ge(minimum_rest_minutes)
        )
    for feature in MODEL_FEATURES:
        if feature not in result:
            result[feature] = False if feature in BOOLEAN_MODEL_FEATURES else np.nan
        if feature in BOOLEAN_MODEL_FEATURES:
            result[feature] = result[feature].fillna(False).astype(float)
        else:
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
    return result


def _prepare_logistic_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, dict[str, float]]]:
    active = [
        feature
        for feature in MODEL_FEATURES
        if train[feature].notna().any() and train[feature].nunique(dropna=True) > 1
    ]
    train_columns: list[np.ndarray] = []
    test_columns: list[np.ndarray] = []
    preprocessing: dict[str, dict[str, float]] = {}
    for feature in active:
        train_values = train[feature].to_numpy(dtype=float)
        test_values = test[feature].to_numpy(dtype=float)
        median = float(np.nanmedian(train_values))
        train_values = np.nan_to_num(train_values, nan=median)
        test_values = np.nan_to_num(test_values, nan=median)
        if feature in BOOLEAN_MODEL_FEATURES:
            centre, scale = 0.0, 1.0
        else:
            centre = float(np.mean(train_values))
            scale = float(np.std(train_values)) or 1.0
        train_columns.append((train_values - centre) / scale)
        test_columns.append((test_values - centre) / scale)
        preprocessing[feature] = {"median": median, "centre": centre, "scale": scale}
    train_matrix = np.column_stack(train_columns) if train_columns else np.empty((len(train), 0))
    test_matrix = np.column_stack(test_columns) if test_columns else np.empty((len(test), 0))
    return train_matrix, test_matrix, active, preprocessing


def _fit_ridge_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    regularization: float,
    maximum_iterations: int = 100,
) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), matrix))
    coefficients = np.zeros(design.shape[1], dtype=float)
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    for _ in range(maximum_iterations):
        linear = np.clip(design @ coefficients, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probability * (1.0 - probability), 1e-6, None)
        gradient = design.T @ (probability - target) + penalty @ coefficients
        hessian = design.T @ (weights[:, None] * design) + penalty
        step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    return coefficients


def _predict_logistic(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(matrix)), matrix))
    return 1.0 / (1.0 + np.exp(-np.clip(design @ coefficients, -35.0, 35.0)))


def _roc_auc(target: np.ndarray, probability: np.ndarray) -> float | None:
    positive = probability[target == 1]
    negative = probability[target == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def build_interpretable_model(
    events: pd.DataFrame,
    controls: pd.DataFrame,
    *,
    minimum_rest_minutes: int,
    regularization: float = 1.0,
) -> dict[str, Any]:
    if events.empty or controls.empty:
        return {"available": False, "reason": "Events and matched controls are required."}
    event_frame = _logistic_feature_frame(events, minimum_rest_minutes=minimum_rest_minutes)
    event_frame["target"] = 1.0
    event_frame["matched_event_id"] = event_frame["event_id"].astype(str)
    control_frame = _logistic_feature_frame(controls, minimum_rest_minutes=minimum_rest_minutes)
    control_frame["target"] = 0.0
    data = pd.concat((event_frame, control_frame), ignore_index=True, sort=False)
    groups = sorted(data["matched_event_id"].dropna().astype(str).unique())
    predictions = np.full(len(data), np.nan, dtype=float)
    for group in groups:
        test_mask = data["matched_event_id"].astype(str).eq(group).to_numpy()
        train = data.loc[~test_mask]
        test = data.loc[test_mask]
        if train["target"].nunique() < 2:
            continue
        train_matrix, test_matrix, _, _ = _prepare_logistic_matrix(train, test)
        coefficients = _fit_ridge_logistic(
            train_matrix,
            train["target"].to_numpy(dtype=float),
            regularization=regularization,
        )
        predictions[test_mask] = _predict_logistic(test_matrix, coefficients)
    valid = np.isfinite(predictions)
    target = data["target"].to_numpy(dtype=float)
    prevalence = float(target.mean())
    cv_auc = _roc_auc(target[valid], predictions[valid]) if valid.any() else None
    cv_brier = float(np.mean((predictions[valid] - target[valid]) ** 2)) if valid.any() else None
    baseline_brier = float(np.mean((prevalence - target) ** 2))
    final_matrix, _, active, preprocessing = _prepare_logistic_matrix(data, data.iloc[:0])
    final_coefficients = _fit_ridge_logistic(
        final_matrix,
        target,
        regularization=regularization,
    )
    coefficient_rows = []
    for feature, coefficient in zip(active, final_coefficients[1:], strict=True):
        coefficient_rows.append(
            {
                "feature_key": feature,
                "coefficient": float(coefficient),
                "odds_ratio": float(math.exp(np.clip(coefficient, -20.0, 20.0))),
                "unit": (
                    "present versus absent"
                    if feature in BOOLEAN_MODEL_FEATURES
                    else "per one training-data standard deviation"
                ),
                **preprocessing[feature],
            }
        )
    coefficient_rows.sort(key=lambda row: abs(row["coefficient"]), reverse=True)
    return {
        "available": True,
        "model_type": "l2_regularized_logistic_regression",
        "purpose": "exploratory event-versus-matched-control discrimination",
        "event_groups": len(groups),
        "rows": len(data),
        "regularization_strength": regularization,
        "validation_method": "leave_one_event_group_out",
        "cross_validated_roc_auc": cv_auc,
        "cross_validated_brier_score": cv_brier,
        "constant_prevalence_brier_score": baseline_brier,
        "coefficients": coefficient_rows,
        "status": "exploratory_small_sample" if len(groups) < 20 else "exploratory",
        "claim_limit": (
            "Coefficients are associations within this battery and control design. They are not "
            "causal effects, fleet generalization evidence, or an independent SOH model."
        ),
    }


def build_soh_timeline(daily: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    timeline = daily.copy()
    timeline["day_utc"] = pd.to_datetime(timeline["day_date"], errors="coerce", utc=True)
    timeline = timeline.loc[
        :, [
            column
            for column in (
                "day_utc",
                "measured_soh_pct",
                "bms_soh_end",
                "coverage_fraction",
                "daily_soh_range_pct",
                "discharge_ah",
                "charge_ah",
                "max_gap_seconds",
            )
            if column in timeline
        ]
    ]
    timeline["recalibration_event_count"] = 0
    timeline["step_drop_count"] = 0
    timeline["step_recovery_count"] = 0
    if events.empty:
        return timeline
    event_days = events.copy()
    event_days["day_utc"] = pd.to_datetime(
        event_days["event_time_utc"], errors="coerce", utc=True
    ).dt.floor("D")
    event_days["step_drop_count"] = event_days["event_type"].eq("step_drop").astype(int)
    event_days["step_recovery_count"] = event_days["event_type"].eq("step_recovery").astype(int)
    event_days = event_days.groupby("day_utc", as_index=False).agg(
        recalibration_event_count=("event_id", "size"),
        step_drop_count=("step_drop_count", "sum"),
        step_recovery_count=("step_recovery_count", "sum"),
    )
    return timeline.drop(
        columns=["recalibration_event_count", "step_drop_count", "step_recovery_count"]
    ).merge(event_days, on="day_utc", how="left").fillna(
        {
            "recalibration_event_count": 0,
            "step_drop_count": 0,
            "step_recovery_count": 0,
        }
    )


def _write_html_report(path: Path, summary: dict[str, Any], events: pd.DataFrame) -> None:
    rows: list[str] = []
    for row in events.tail(100).itertuples(index=False):
        discharge_observed = bool(
            getattr(row, "complete_discharge_within_lookback", False)
        )
        discharge_detail = "not observed"
        if discharge_observed:
            start_soc = getattr(row, "complete_discharge_start_soc_pct", np.nan)
            end_soc = getattr(row, "complete_discharge_end_soc_pct", np.nan)
            discharged_ah = getattr(row, "complete_discharge_ah", np.nan)
            hours_before = getattr(row, "hours_since_complete_discharge", np.nan)
            discharge_detail = (
                f"{start_soc:.1f}% to {end_soc:.1f}%; {discharged_ah:.1f} Ah; "
                f"{hours_before:.2f} h before"
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(pd.Timestamp(row.event_time_utc).isoformat())}</td>"
            f"<td>{html.escape(str(row.event_type))}</td>"
            f"<td>{row.soh_delta_pp:+.2f}</td>"
            f"<td>{html.escape(str(row.evidence_classification).replace('_', ' '))}</td>"
            f"<td>{html.escape(str(row.classification_confidence))}</td>"
            f"<td>{row.context_coverage_fraction:.0%}</td>"
            f"<td>{html.escape(discharge_detail)}</td>"
            f"<td>{'n/a' if pd.isna(row.rest_before_event_minutes) else f'{row.rest_before_event_minutes:.0f}'}</td>"
            f"<td>{html.escape(str(row.evidence_tags or 'none'))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>BMS SOH recalibration evidence - {html.escape(str(summary['serial']))}</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f7;color:#20313b;font:14px Arial,sans-serif}}
main{{max-width:1360px;margin:auto;padding:28px}}h1{{margin:4px 0 8px}}.eyebrow{{font-size:12px;color:#176b87;font-weight:700;text-transform:uppercase}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;background:#d8e0e3;border:1px solid #d8e0e3;margin:22px 0}}.metric,section{{background:#fff;padding:16px}}.metric small{{display:block;color:#66757c;margin-bottom:6px}}.metric strong{{font-size:22px}}
section{{border:1px solid #d8e0e3;margin:18px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e3e8ea;text-align:left;vertical-align:top}}th{{color:#627078;font-size:12px}}.table-wrap{{overflow:auto}}.notice{{border-left:4px solid #d28c1a;background:#fff8ea;padding:12px;line-height:1.5}}@media(max-width:850px){{main{{padding:16px}}.metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<span class="eyebrow">BatteryML evidence report</span><h1>BMS SOH recalibration evidence</h1>
<div>Battery {html.escape(str(summary['serial']))} | {html.escape(str(summary['chemistry']))} | {summary['module_count']} modules | all timestamps UTC</div>
<div class="metrics"><div class="metric"><small>Audited SOH events</small><strong>{summary['event_rows']}</strong></div><div class="metric"><small>Step drops</small><strong>{summary['step_drop_rows']}</strong></div><div class="metric"><small>Step recoveries</small><strong>{summary['step_recovery_rows']}</strong></div><div class="metric"><small>High-confidence evidence</small><strong>{summary['high_confidence_rows']}</strong></div><div class="metric"><small>Limited context</small><strong>{summary['limited_context_rows']}</strong></div></div>
<section><h2>Event evidence</h2><p>The primary test searches first for an ordered negative-current 95-100% to 0-5% discharge in the 72 hours before each SOH transition. Temperature, C-rate, and voltage spread are secondary context.</p><div class="table-wrap"><table><thead><tr><th>Resolved transition UTC</th><th>Type</th><th>Change pp</th><th>Evidence classification</th><th>Confidence</th><th>Context coverage</th><th>Calibration discharge</th><th>Prior rest min</th><th>Evidence</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="9">No BMS SOH steps passed the audit threshold.</td></tr>'}</tbody></table></div></section>
<p class="notice"><strong>Interpretation limit:</strong> {html.escape(str(summary['claim_limit']))}</p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def run_analysis(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
    raw_root: Path,
    feature_root: Path,
    soh_audit_root: Path,
    output_root: Path,
    rest_current_threshold_a: float = 0.5,
    minimum_window_coverage: float = 0.8,
    minimum_context_coverage: float = 0.8,
    context_lookback_hours: int = 72,
    near_full_soc_pct: float = 99.0,
    low_soc_pct: float = 10.0,
    minimum_rest_minutes: int = 120,
    data_gap_risk_minutes: int = 60,
    recurrence_window_days: int = 30,
    raw_search_before_hours: int = 30,
    raw_search_after_hours: int = 30,
    persistence_hours: int = 6,
    minimum_new_level_fraction: float = 0.95,
    previous_support_minutes: int = 30,
    exact_gap_seconds: int = 60,
    controls_per_event: int = 5,
    control_event_exclusion_days: int = 3,
    discharge_start_soc_pct: float = 95.0,
    discharge_end_soc_pct: float = 5.0,
    minimum_discharge_soc_drop_pct: float = 90.0,
    minimum_discharge_dominance_fraction: float = 0.8,
    discharge_episode_gap_minutes: int = 30,
    capacity_validation_start_soc_pct: float = 99.0,
    capacity_validation_end_soc_pct: float = 1.0,
    capacity_validation_minimum_soc_drop_pct: float = 98.0,
    capacity_validation_minimum_coverage_fraction: float = 0.98,
    capacity_validation_maximum_unobserved_ah: float = 0.25,
    capacity_validation_maximum_charge_intrusion_ah: float = 0.1,
    capacity_validation_maximum_integrated_gap_seconds: int = 5,
    capacity_validation_raw_padding_minutes: int = 10,
    continuous_cycle_maximum_rest_minutes: float = 5.0,
    compression: str | None = "zstd",
    run_id: str | None = None,
) -> Path:
    chemistry = chemistry.upper().strip()
    if chemistry not in {"NCA", "LFP"}:
        raise ValueError("chemistry must be NCA or LFP")
    if module_count < 1 or nominal_capacity_ah <= 0:
        raise ValueError("module_count and nominal_capacity_ah must be positive")
    if rest_current_threshold_a < 0:
        raise ValueError("rest_current_threshold_a must be nonnegative")
    for value, name in (
        (minimum_window_coverage, "minimum_window_coverage"),
        (minimum_context_coverage, "minimum_context_coverage"),
        (
            capacity_validation_minimum_coverage_fraction,
            "capacity_validation_minimum_coverage_fraction",
        ),
    ):
        if not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if not 0 <= low_soc_pct < near_full_soc_pct <= 100:
        raise ValueError("SOC thresholds must be ordered within 0 to 100")
    if not 0 <= discharge_end_soc_pct < discharge_start_soc_pct <= 100:
        raise ValueError("Discharge boundary SOC thresholds must be ordered within 0 to 100")
    if not 0 < minimum_discharge_soc_drop_pct <= 100:
        raise ValueError("minimum_discharge_soc_drop_pct must be in (0, 100]")
    if not 0 < minimum_discharge_dominance_fraction <= 1:
        raise ValueError("minimum_discharge_dominance_fraction must be in (0, 1]")
    if not (
        0
        <= capacity_validation_end_soc_pct
        < capacity_validation_start_soc_pct
        <= 100
    ):
        raise ValueError("Capacity-validation SOC boundaries must be ordered")
    if not 0 < capacity_validation_minimum_soc_drop_pct <= 100:
        raise ValueError("capacity_validation_minimum_soc_drop_pct must be in (0, 100]")
    if min(
        capacity_validation_maximum_unobserved_ah,
        capacity_validation_maximum_charge_intrusion_ah,
        continuous_cycle_maximum_rest_minutes,
    ) < 0:
        raise ValueError("Capacity-validation error and rest limits must be nonnegative")
    if min(
        context_lookback_hours,
        minimum_rest_minutes,
        data_gap_risk_minutes,
        recurrence_window_days,
        raw_search_before_hours,
        raw_search_after_hours,
        persistence_hours,
        previous_support_minutes,
        exact_gap_seconds,
        controls_per_event,
        control_event_exclusion_days,
        discharge_episode_gap_minutes,
        capacity_validation_maximum_integrated_gap_seconds,
        capacity_validation_raw_padding_minutes,
    ) < 1:
        raise ValueError("context and evidence windows must be positive")
    if not 0.5 <= minimum_new_level_fraction <= 1.0:
        raise ValueError("minimum_new_level_fraction must be in [0.5, 1]")

    compact, feature_summary = read_compact_history(feature_root, serial)
    feature_module_count = int(feature_summary.get("module_count") or 0)
    if feature_module_count != module_count:
        raise ValueError(
            f"Configured module count {module_count} does not match compact history "
            f"module count {feature_module_count}"
        )
    aggregation_minutes = int(feature_summary["aggregation_minutes"])
    windows = prepare_windows(
        compact,
        aggregation_minutes=aggregation_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        rest_current_threshold_a=rest_current_threshold_a,
    )
    calibration_discharge_episodes = build_calibration_discharge_episodes(
        windows,
        aggregation_minutes=aggregation_minutes,
        episode_gap_minutes=discharge_episode_gap_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_window_coverage=minimum_window_coverage,
        high_soc_start_pct=discharge_start_soc_pct,
        low_soc_end_pct=discharge_end_soc_pct,
        minimum_soc_drop_pct=minimum_discharge_soc_drop_pct,
        minimum_discharge_dominance_fraction=(
            minimum_discharge_dominance_fraction
        ),
    )
    full_soc_capacity_candidates = build_full_soc_capacity_candidates(
        windows,
        aggregation_minutes=aggregation_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        start_soc_pct=capacity_validation_start_soc_pct,
        end_soc_pct=capacity_validation_end_soc_pct,
        maximum_elapsed_hours=context_lookback_hours,
        minimum_window_coverage=minimum_window_coverage,
    )
    calibration_candidates = calibration_discharge_episodes.loc[
        calibration_discharge_episodes.get(
            "complete_high_to_low_discharge",
            pd.Series(False, index=calibration_discharge_episodes.index),
        ).fillna(False)
    ].copy()
    if not calibration_candidates.empty:
        calibration_candidates["candidate_source"] = "compact_95_to_5_calibration"
    capacity_discharge_candidates = pd.concat(
        (calibration_candidates, full_soc_capacity_candidates),
        ignore_index=True,
        sort=False,
    )
    if not capacity_discharge_candidates.empty:
        capacity_discharge_candidates = (
            capacity_discharge_candidates.sort_values(
                ["start_time_utc", "completion_time_utc"]
            )
            .drop_duplicates(
                ["start_time_utc", "completion_time_utc"], keep="last"
            )
            .reset_index(drop=True)
        )
    audit_daily, audit_events, audit_run, audit_summary = read_latest_soh_audit(
        soh_audit_root, serial
    )
    audit_candidates = audit_events.loc[
        audit_events["event_type"].isin(("step_drop", "step_recovery", "change_after_gap"))
    ].copy()
    raw_soh, raw_scan_metrics = read_raw_soh_candidate_windows(
        raw_root,
        serial=serial,
        chemistry=chemistry,
        audit_events=audit_candidates,
        search_before_hours=raw_search_before_hours,
        search_after_hours=raw_search_after_hours,
    )
    refined_events = refine_audit_events(
        audit_candidates,
        raw_soh,
        search_before_hours=raw_search_before_hours,
        search_after_hours=raw_search_after_hours,
        persistence_hours=persistence_hours,
        minimum_new_level_fraction=minimum_new_level_fraction,
        previous_support_minutes=previous_support_minutes,
        exact_gap_seconds=exact_gap_seconds,
    )
    events = build_recalibration_events(
        windows,
        refined_events,
        aggregation_minutes=aggregation_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_window_coverage=minimum_window_coverage,
        minimum_context_coverage=minimum_context_coverage,
        context_lookback_hours=context_lookback_hours,
        near_full_soc_pct=near_full_soc_pct,
        low_soc_pct=low_soc_pct,
        minimum_rest_minutes=minimum_rest_minutes,
        data_gap_risk_minutes=data_gap_risk_minutes,
        recurrence_window_days=recurrence_window_days,
        discharge_start_soc_pct=discharge_start_soc_pct,
        discharge_end_soc_pct=discharge_end_soc_pct,
        minimum_discharge_soc_drop_pct=minimum_discharge_soc_drop_pct,
        minimum_discharge_dominance_fraction=minimum_discharge_dominance_fraction,
        discharge_episode_gap_minutes=discharge_episode_gap_minutes,
        calibration_discharge_episodes=calibration_discharge_episodes,
    )
    controls = build_matched_control_windows(
        windows,
        audit_daily,
        events,
        aggregation_minutes=aggregation_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_window_coverage=minimum_window_coverage,
        minimum_context_coverage=minimum_context_coverage,
        context_lookback_hours=context_lookback_hours,
        near_full_soc_pct=near_full_soc_pct,
        low_soc_pct=low_soc_pct,
        minimum_rest_minutes=minimum_rest_minutes,
        controls_per_event=controls_per_event,
        event_exclusion_days=control_event_exclusion_days,
        discharge_start_soc_pct=discharge_start_soc_pct,
        discharge_end_soc_pct=discharge_end_soc_pct,
        minimum_discharge_soc_drop_pct=minimum_discharge_soc_drop_pct,
        minimum_discharge_dominance_fraction=minimum_discharge_dominance_fraction,
        discharge_episode_gap_minutes=discharge_episode_gap_minutes,
        calibration_discharge_episodes=calibration_discharge_episodes,
    )
    events, event_raw_discharge_scan_metrics = detect_raw_discharge_evidence_batched(
        raw_root,
        serial=serial,
        chemistry=chemistry,
        anchors=events,
        anchor_column="event_time_utc",
        lookback_hours=context_lookback_hours,
        start_soc_pct=discharge_start_soc_pct,
        end_soc_pct=discharge_end_soc_pct,
        maximum_integrated_gap_seconds=capacity_validation_maximum_integrated_gap_seconds,
        rest_current_threshold_a=rest_current_threshold_a,
        minimum_coverage_fraction=minimum_window_coverage,
        minimum_discharge_dominance_fraction=(
            minimum_discharge_dominance_fraction
        ),
    )
    controls, control_raw_discharge_scan_metrics = detect_raw_discharge_evidence_batched(
        raw_root,
        serial=serial,
        chemistry=chemistry,
        anchors=controls,
        anchor_column="anchor_time_utc",
        lookback_hours=context_lookback_hours,
        start_soc_pct=discharge_start_soc_pct,
        end_soc_pct=discharge_end_soc_pct,
        maximum_integrated_gap_seconds=capacity_validation_maximum_integrated_gap_seconds,
        rest_current_threshold_a=rest_current_threshold_a,
        minimum_coverage_fraction=minimum_window_coverage,
        minimum_discharge_dominance_fraction=(
            minimum_discharge_dominance_fraction
        ),
    )
    raw_discharge_scan_metrics = {
        key: event_raw_discharge_scan_metrics.get(key, 0)
        + control_raw_discharge_scan_metrics.get(key, 0)
        for key in (
            "raw_files_considered",
            "raw_row_groups_read",
            "raw_rows_read",
            "raw_rows_retained",
            "candidate_intervals",
            "anchor_batches",
        )
    }
    raw_discharge_scan_metrics["anchors_per_batch_limit"] = RAW_ANCHOR_BATCH_SIZE
    for frame, anchor_column in (
        (events, "event_time_utc"),
        (controls, "anchor_time_utc"),
    ):
        frame["complete_discharge_within_lookback"] = (
            frame.get("raw_discharge_qualified", False)
        )
        frame["complete_discharge_start_time_utc"] = frame.get("raw_start_time_utc")
        frame["complete_discharge_end_time_utc"] = frame.get("raw_end_time_utc")
        frame["complete_discharge_start_soc_pct"] = frame.get("raw_start_soc_pct")
        frame["complete_discharge_end_soc_pct"] = frame.get("raw_end_soc_pct")
        frame["complete_discharge_soc_drop_pct"] = frame.get("raw_soc_drop_pct")
        frame["complete_discharge_ah"] = frame.get("discharged_ah_1s")
        frame["complete_discharge_coverage_fraction"] = frame.get("raw_coverage_fraction")
        frame["hours_since_complete_discharge"] = (
            pd.to_datetime(frame[anchor_column], errors="coerce", utc=True)
            - pd.to_datetime(
                frame.get("raw_end_time_utc", pd.Series(pd.NaT, index=frame.index)),
                errors="coerce",
                utc=True,
            )
        ).dt.total_seconds().div(3600.0)
        frame["complete_discharge_duration_minutes"] = (
            pd.to_datetime(
                frame.get("raw_end_time_utc", pd.Series(pd.NaT, index=frame.index)),
                errors="coerce",
                utc=True,
            )
            - pd.to_datetime(
                frame.get("raw_start_time_utc", pd.Series(pd.NaT, index=frame.index)),
                errors="coerce",
                utc=True,
            )
        ).dt.total_seconds().div(60.0)
        for hours in DISCHARGE_HORIZONS_HOURS:
            frame[f"complete_discharge_within_{hours}h"] = (
                frame["complete_discharge_within_lookback"]
                & frame["hours_since_complete_discharge"].le(hours)
            )
    refreshed_classifications = events.apply(
        lambda event: _classify_event(
            event_type=str(event["event_type"]),
            context_coverage_fraction=float(event.get("context_coverage_fraction") or 0.0),
            minimum_context_coverage=minimum_context_coverage,
            data_gap_minutes=float(event.get("largest_data_gap_minutes") or 0.0),
            data_gap_risk_minutes=data_gap_risk_minutes,
            complete_discharge_within_lookback=bool(
                event["complete_discharge_within_lookback"]
            ),
            near_full_charge_within_lookback=bool(
                event.get("near_full_charge_within_lookback")
            ),
            rest_before_event_minutes=_finite(event.get("rest_before_event_minutes")),
            minimum_rest_minutes=minimum_rest_minutes,
            reached_low_soc=bool(event.get("reached_low_soc")),
            low_before_near_full=bool(event.get("low_before_near_full")),
            meaningful_charge=bool(event.get("meaningful_charge_before_update")),
            maximum_temperature_c=_finite(event.get("maximum_temperature_c")),
            maximum_voltage_spread_mv=_finite(event.get("maximum_voltage_spread_mv")),
            maximum_absolute_c_rate=_finite(event.get("maximum_absolute_c_rate")),
            transition_detection_status=str(event.get("transition_detection_status")),
            prior_same_direction_events=int(event.get("prior_same_direction_events") or 0),
        ),
        axis=1,
    )
    if not events.empty:
        events["evidence_classification"] = refreshed_classifications.map(lambda item: item[0])
        events["primary_context_pattern"] = events["evidence_classification"]
        events["context_match_strength"] = refreshed_classifications.map(lambda item: item[1])
        events["classification_confidence"] = events["context_match_strength"].map(
            {"strong": "high", "moderate": "medium", "limited": "limited"}
        )
        events["evidence_tags"] = refreshed_classifications.map(
            lambda item: "; ".join(item[2])
        )
    capacity_validation = events.loc[
        events["raw_discharge_detection_status"].eq("resolved")
    ].copy()
    if not capacity_validation.empty:
        capacity_validation["discharge_episode_id"] = capacity_validation["event_id"]
        capacity_validation["capacity_validation_status"] = np.select(
            [
                capacity_validation["raw_coverage_fraction"].lt(capacity_validation_minimum_coverage_fraction),
                capacity_validation["unobserved_discharge_ah_estimate"].gt(
                    capacity_validation_maximum_unobserved_ah
                ),
                capacity_validation["charge_intrusion_ah_1s"].gt(capacity_validation_maximum_charge_intrusion_ah),
            ],
            ["insufficient_raw_coverage", "excess_missing_ah_uncertainty", "charge_intrusion"],
            default="validation_grade",
        )
        capacity_validation["strict_full_soc_boundaries"] = (
            capacity_validation["raw_start_soc_pct"].ge(capacity_validation_start_soc_pct)
            & capacity_validation["raw_end_soc_pct"].le(capacity_validation_end_soc_pct)
            & capacity_validation["raw_soc_drop_pct"].ge(capacity_validation_minimum_soc_drop_pct)
        )
        capacity_validation.loc[
            ~capacity_validation["strict_full_soc_boundaries"], "capacity_validation_status"
        ] = "exploratory_soc_boundaries"
        capacity_validation["cycle_continuity_class"] = np.where(
            capacity_validation["longest_rest_minutes"].le(continuous_cycle_maximum_rest_minutes),
            "continuous",
            "interrupted",
        )
        capacity_validation["linked_soh_event_id"] = capacity_validation["event_id"]
        capacity_validation["linked_soh_event_time_utc"] = capacity_validation["event_time_utc"]
        capacity_validation["hours_to_linked_soh_event"] = (
            pd.to_datetime(capacity_validation["event_time_utc"], utc=True)
            - pd.to_datetime(capacity_validation["raw_end_time_utc"], utc=True)
        ).dt.total_seconds().div(3600.0)
        capacity_validation["direct_capacity_soh_pct"] = (
            capacity_validation["discharged_ah_1s"] / nominal_capacity_ah * 100.0
        )
        capacity_validation["soc_normalized_capacity_ah"] = (
            capacity_validation["discharged_ah_1s"]
            / capacity_validation["raw_soc_drop_pct"].div(100.0)
        )
        capacity_validation["soc_normalized_capacity_soh_pct"] = (
            capacity_validation["soc_normalized_capacity_ah"] / nominal_capacity_ah * 100.0
        )
        capacity_validation["bms_soh_at_cycle_completion_pct"] = capacity_validation.get(
            "raw_bms_soh_at_cycle_completion_pct"
        )
        capacity_validation["bms_soh_after_linked_event_pct"] = capacity_validation.get(
            "new_soh_pct"
        )
        capacity_validation["bms_completion_minus_capacity_soh_pp"] = (
            capacity_validation["bms_soh_at_cycle_completion_pct"]
            - capacity_validation["direct_capacity_soh_pct"]
        )
        capacity_validation["bms_after_event_minus_capacity_soh_pp"] = (
            capacity_validation["bms_soh_after_linked_event_pct"]
            - capacity_validation["direct_capacity_soh_pct"]
        )
        capacity_validation["bms_completion_minus_normalized_soh_pp"] = (
            capacity_validation["bms_soh_at_cycle_completion_pct"]
            - capacity_validation["soc_normalized_capacity_soh_pct"]
        )
        capacity_validation["bms_after_event_minus_normalized_soh_pp"] = (
            capacity_validation["bms_soh_after_linked_event_pct"]
            - capacity_validation["soc_normalized_capacity_soh_pct"]
        )
    capacity_validation_summary = summarize_capacity_validation(
        capacity_validation,
        minimum_coverage_fraction=capacity_validation_minimum_coverage_fraction,
        maximum_unobserved_ah=capacity_validation_maximum_unobserved_ah,
        maximum_charge_intrusion_ah=capacity_validation_maximum_charge_intrusion_ah,
    )
    comparison = build_evidence_comparison(
        events,
        controls,
        minimum_rest_minutes=minimum_rest_minutes,
    )
    interpretable_model = build_interpretable_model(
        events,
        controls,
        minimum_rest_minutes=minimum_rest_minutes,
    )
    timeline = build_soh_timeline(audit_daily, events)
    run_stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"serial={serial}" / f"recalibration_run={run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_parquet_chunk(events, run_dir / "recalibration_events.parquet", compression=compression)
    write_parquet_chunk(timeline, run_dir / "soh_daily_timeline.parquet", compression=compression)
    write_parquet_chunk(controls, run_dir / "matched_control_windows.parquet", compression=compression)
    write_parquet_chunk(comparison, run_dir / "evidence_comparison.parquet", compression=compression)
    write_parquet_chunk(
        calibration_discharge_episodes,
        run_dir / "calibration_discharge_episodes.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        capacity_validation,
        run_dir / "one_second_capacity_validation.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        capacity_discharge_candidates,
        run_dir / "capacity_discharge_candidates.parquet",
        compression=compression,
    )
    classification_counts = (
        events["evidence_classification"].value_counts().sort_index().to_dict()
        if not events.empty
        else {}
    )
    evidence_tag_counts: dict[str, int] = {}
    if not events.empty:
        for value in events["evidence_tags"].fillna(""):
            for tag in (item.strip() for item in str(value).split(";") if item.strip()):
                evidence_tag_counts[tag] = evidence_tag_counts.get(tag, 0) + 1
    feature_start = pd.to_datetime(windows["window_start_utc"].min(), utc=True)
    feature_end = pd.to_datetime(
        windows["window_start_utc"].max() + pd.Timedelta(minutes=aggregation_minutes),
        utc=True,
    )
    audit_start = pd.to_datetime(
        audit_daily.get("first_sample_time", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
        utc=True,
    ).min()
    audit_end = pd.to_datetime(
        audit_daily.get("end_anchor_time", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
        utc=True,
    ).max()
    overlap_start = max(feature_start, audit_start) if pd.notna(audit_start) else feature_start
    overlap_end = min(feature_end, audit_end) if pd.notna(audit_end) else feature_end
    delta = pd.to_numeric(events.get("soh_delta_pp", pd.Series(dtype=float)), errors="coerce")
    gross_drop_pp = float(-delta.loc[delta.lt(0)].sum())
    gross_recovery_pp = float(delta.loc[delta.gt(0)].sum())
    net_soh_change_pp = float(delta.sum())
    exact_rows = int(
        events.get("transition_detection_status", pd.Series(dtype=str)).eq("exact").sum()
    )
    interval_rows = int(
        events.get("transition_detection_status", pd.Series(dtype=str))
        .eq("interval_censored")
        .sum()
    )
    fallback_rows = int(len(events) - exact_rows - interval_rows)
    observation_years = max(
        (overlap_end - overlap_start).total_seconds() / (365.25 * 86400.0), 0.0
    )
    cumulative_discharge = _finite(
        pd.to_numeric(
            audit_daily.get("discharge_ah", pd.Series(dtype=float)),
            errors="coerce",
        ).clip(lower=0.0).sum()
    )
    if cumulative_discharge is None or cumulative_discharge <= 0:
        cumulative_discharge = _finite(
            pd.to_numeric(
                events.get("audit_cumulative_discharge_ah", pd.Series(dtype=float)),
                errors="coerce",
            ).max()
        )
    strongest_enrichment = None
    primary_discharge_comparison = None
    if not comparison.empty:
        primary_rows = comparison.loc[
            comparison["feature_key"].eq("complete_discharge_within_72h")
        ]
        if not primary_rows.empty:
            primary_discharge_comparison = _json_safe(primary_rows.iloc[0].to_dict())
        eligible_comparison = comparison.loc[comparison["control_total"].gt(0)].copy()
        if not eligible_comparison.empty:
            strongest_enrichment = _json_safe(
                eligible_comparison.sort_values(
                    ["risk_difference", "prevalence_lift"],
                    ascending=False,
                    na_position="last",
                ).iloc[0].to_dict()
            )
    findings = [
        (
            f"Resolved {exact_rows} of {len(events)} audited SOH transitions to persistent "
            "raw change points."
            if events.empty or exact_rows < len(events)
            else f"All {len(events)} audited SOH transitions were resolved to persistent raw change points."
        ),
        (
            f"Observed {gross_drop_pp:.2f} pp of gross drops, {gross_recovery_pp:.2f} pp "
            f"of recoveries, and {net_soh_change_pp:.2f} pp net BMS SOH movement."
        ),
    ]
    if gross_recovery_pp > 0:
        findings.append(
            "Reversible BMS SOH movement is estimator evidence and cannot be irreversible physical capacity loss."
        )
    if primary_discharge_comparison:
        event_rate = primary_discharge_comparison.get("event_rate")
        control_rate = primary_discharge_comparison.get("control_rate")
        findings.append(
            "A qualified 95-100% to 0-5% discharge completed within 72 hours before "
            f"{100 * event_rate:.1f}% of SOH events and {100 * control_rate:.1f}% "
            "of controls."
            if event_rate is not None and control_rate is not None
            else "The primary 72-hour complete-discharge comparison had insufficient data."
        )
    if strongest_enrichment:
        findings.append(
            f"The strongest matched-control enrichment was {strongest_enrichment['feature_label']}: "
            f"{100 * strongest_enrichment['event_rate']:.1f}% of events versus "
            f"{100 * strongest_enrichment['control_rate']:.1f}% of controls."
        )
    if capacity_validation_summary.get("validation_grade_rows", 0):
        completion_error = capacity_validation_summary["bms_at_completion_error"]
        post_error = capacity_validation_summary["bms_after_linked_event_error"]
        findings.append(
            "One-second current integration retained "
            f"{capacity_validation_summary['validation_grade_rows']} validation-grade "
            "full-SOC cycles with median discharged capacity "
            f"{capacity_validation_summary['median_discharged_ah']:.2f} Ah "
            f"({capacity_validation_summary['median_capacity_soh_pct']:.2f}% of nominal)."
        )
        if completion_error.get("mae_pp") is not None:
            findings.append(
                "BMS SOH at cycle completion differed from the current-integrated "
                f"capacity estimate by {completion_error['mae_pp']:.2f} pp MAE."
            )
        if post_error.get("mae_pp") is not None:
            findings.append(
                "After linked BMS SOH updates, the difference from the current-integrated "
                f"capacity estimate was {post_error['mae_pp']:.2f} pp MAE."
            )
    else:
        findings.append(
            "No one-second cycle passed the strict 99-1% full-cycle capacity criteria; "
            "no direct full-cycle capacity SOH claim is available."
        )
    if capacity_validation_summary.get("operational_window_rows", 0):
        normalized_error = capacity_validation_summary[
            "bms_after_linked_event_normalized_error"
        ]
        findings.append(
            f"Across {capacity_validation_summary['operational_window_rows']} qualified "
            "partial operational windows, SOC-normalized median usable capacity was "
            f"{capacity_validation_summary['median_soc_normalized_capacity_ah']:.2f} Ah "
            f"({capacity_validation_summary['median_soc_normalized_capacity_soh_pct']:.2f}% "
            "of nominal). This extrapolation is not a strict 100-0 measurement."
        )
        if normalized_error.get("mae_pp") is not None:
            findings.append(
                "For linked operational windows, post-update BMS SOH differed from "
                f"the SOC-normalized estimate by {normalized_error['mae_pp']:.2f} pp MAE."
            )
    continuity = capacity_validation_summary.get("continuity_comparison") or {}
    continuous = continuity.get("continuous") or {}
    interrupted = continuity.get("interrupted") or {}
    if continuous.get("cycle_rows") or interrupted.get("cycle_rows"):
        findings.append(
            "A linked SOH update followed "
            f"{continuous.get('linked_soh_event_rows', 0)}/{continuous.get('cycle_rows', 0)} "
            "continuous cycles and "
            f"{interrupted.get('linked_soh_event_rows', 0)}/{interrupted.get('cycle_rows', 0)} "
            "cycles containing more than the configured internal-rest allowance."
        )
    summary = {
        "model_family": MODEL_FAMILY,
        "detector_version": DETECTOR_VERSION,
        "raw_discharge_detector_version": "event_anchor_raw_1s_v1",
        "serial": str(serial),
        "chemistry": chemistry,
        "module_count": module_count,
        "nominal_capacity_ah": nominal_capacity_ah,
        "timestamp_semantics": "UTC",
        "current_sign_convention": {
            "source_signal": "batt_current_a",
            "charging": "positive pack current",
            "discharging": "negative pack current",
            "rest": f"absolute pack current at or below {rest_current_threshold_a:g} A",
            "charge_ah": "time integral of positive pack current",
            "discharge_ah": "time integral of the absolute value of negative pack current",
        },
        "observed_start": overlap_start,
        "observed_end": overlap_end,
        "feature_observed_start": feature_start,
        "feature_observed_end": feature_end,
        "soh_observed_start": audit_start,
        "soh_observed_end": audit_end,
        "analysis_overlap_start": overlap_start,
        "analysis_overlap_end": overlap_end,
        "aggregation_minutes": aggregation_minutes,
        "source_feature_run_ids": feature_summary.get("compatible_feature_run_ids", []),
        "source_soh_audit_run_id": audit_run.name,
        "source_soh_audit_label_status": (audit_summary.get("audit_metrics") or {}).get("label_status"),
        "event_detection_method": (
            "persistent_raw_soh_change_points_refined_from_quality_screened_daily_candidates"
        ),
        "event_time_semantics": (
            "Persistent one-second BMS SOH change point when resolved; otherwise an explicitly "
            "interval-censored or daily-mode fallback timestamp."
        ),
        "event_rows": int(len(events)),
        "step_drop_rows": int(events.get("event_type", pd.Series(dtype=str)).eq("step_drop").sum()),
        "step_recovery_rows": int(events.get("event_type", pd.Series(dtype=str)).eq("step_recovery").sum()),
        "change_after_gap_rows": int(events.get("event_type", pd.Series(dtype=str)).eq("change_after_gap").sum()),
        "high_confidence_rows": int(events.get("classification_confidence", pd.Series(dtype=str)).eq("high").sum()),
        "strong_context_match_rows": int(events.get("context_match_strength", pd.Series(dtype=str)).eq("strong").sum()),
        "limited_context_rows": int(events.get("context_match_strength", pd.Series(dtype=str)).eq("limited").sum()),
        "exact_transition_rows": exact_rows,
        "interval_censored_transition_rows": interval_rows,
        "daily_fallback_transition_rows": fallback_rows,
        "exact_transition_rate": exact_rows / len(events) if len(events) else None,
        "gross_step_drop_pp": gross_drop_pp,
        "gross_step_recovery_pp": gross_recovery_pp,
        "net_bms_soh_change_pp": net_soh_change_pp,
        "event_rate_per_observed_year": len(events) / observation_years if observation_years else None,
        "event_rate_per_1000_discharge_ah": (
            len(events) / (cumulative_discharge / 1000.0)
            if cumulative_discharge and cumulative_discharge > 0
            else None
        ),
        "median_context_coverage_fraction": _finite(
            pd.to_numeric(
                events.get("context_coverage_fraction", pd.Series(dtype=float)),
                errors="coerce",
            ).median()
        ),
        "matched_control_rows": int(len(controls)),
        "controls_per_event_requested": controls_per_event,
        "classification_counts": classification_counts,
        "evidence_tag_counts": dict(sorted(evidence_tag_counts.items())),
        "strongest_matched_control_enrichment": strongest_enrichment,
        "primary_discharge_hypothesis": primary_discharge_comparison,
        "primary_discharge_hypothesis_method": (
            "For every exact or censored BMS SOH event and every matched non-event "
            "anchor, test first whether an ordered negative-current discharge began "
            f"at or above {discharge_start_soc_pct:g}% SOC, ended at or below "
            f"{discharge_end_soc_pct:g}% SOC, crossed at least "
            f"{minimum_discharge_soc_drop_pct:g} percentage points, and completed "
            "within 1, 3, 6, 12, 24, 48, or 72 hours before the anchor. "
            "C-rate, temperature, and cell-voltage spread remain secondary evidence."
        ),
        "complete_discharge_event_rows": int(
            events.get(
                "complete_discharge_within_lookback",
                pd.Series(False, index=events.index),
            )
            .fillna(False)
            .sum()
        ),
        "qualified_calibration_discharge_rows": int(
            calibration_discharge_episodes.get(
                "complete_high_to_low_discharge",
                pd.Series(False, index=calibration_discharge_episodes.index),
            )
            .fillna(False)
            .sum()
        ),
        "interpretable_model": interpretable_model,
        "raw_transition_scan": raw_scan_metrics,
        "raw_capacity_validation_scan": raw_discharge_scan_metrics,
        "one_second_capacity_validation": capacity_validation_summary,
        "findings": findings,
        "configuration": {
            "rest_current_threshold_a": rest_current_threshold_a,
            "minimum_window_coverage": minimum_window_coverage,
            "minimum_context_coverage": minimum_context_coverage,
            "context_lookback_hours": context_lookback_hours,
            "near_full_soc_pct": near_full_soc_pct,
            "low_soc_pct": low_soc_pct,
            "minimum_rest_minutes": minimum_rest_minutes,
            "data_gap_risk_minutes": data_gap_risk_minutes,
            "recurrence_window_days": recurrence_window_days,
            "raw_search_before_hours": raw_search_before_hours,
            "raw_search_after_hours": raw_search_after_hours,
            "persistence_hours": persistence_hours,
            "minimum_new_level_fraction": minimum_new_level_fraction,
            "previous_support_minutes": previous_support_minutes,
            "exact_gap_seconds": exact_gap_seconds,
            "controls_per_event": controls_per_event,
            "control_event_exclusion_days": control_event_exclusion_days,
            "discharge_start_soc_pct": discharge_start_soc_pct,
            "discharge_end_soc_pct": discharge_end_soc_pct,
            "minimum_discharge_soc_drop_pct": minimum_discharge_soc_drop_pct,
            "minimum_discharge_dominance_fraction": (
                minimum_discharge_dominance_fraction
            ),
            "discharge_episode_gap_minutes": discharge_episode_gap_minutes,
            "raw_discharge_lookback_hours": context_lookback_hours,
            "capacity_validation_start_soc_pct": capacity_validation_start_soc_pct,
            "capacity_validation_end_soc_pct": capacity_validation_end_soc_pct,
            "capacity_validation_minimum_soc_drop_pct": (
                capacity_validation_minimum_soc_drop_pct
            ),
            "capacity_validation_minimum_coverage_fraction": (
                capacity_validation_minimum_coverage_fraction
            ),
            "capacity_validation_maximum_unobserved_ah": (
                capacity_validation_maximum_unobserved_ah
            ),
            "capacity_validation_maximum_charge_intrusion_ah": (
                capacity_validation_maximum_charge_intrusion_ah
            ),
            "capacity_validation_maximum_integrated_gap_seconds": (
                capacity_validation_maximum_integrated_gap_seconds
            ),
            "capacity_validation_raw_padding_minutes": (
                capacity_validation_raw_padding_minutes
            ),
            "continuous_cycle_maximum_rest_minutes": (
                continuous_cycle_maximum_rest_minutes
            ),
        },
        "control_method": (
            "Non-event anchors matched without replacement by seasonal day distance and context "
            "coverage; anchors within the configured exclusion distance of any event are removed. "
            "The controls are not matched on complete-discharge exposure because that is the "
            "primary hypothesis being tested. Complete-discharge prevalence is compared before "
            "secondary C-rate, temperature, and cell-voltage-spread evidence."
        ),
        "claim_limit": (
            "This detector measures temporal operating evidence enriched around persistent BMS-reported SOH steps. "
            "It does not prove a firmware cause, instantaneous electrochemical degradation, or "
            "independent physical capacity loss. BMS SOH remains a weak label."
        ),
    }
    manifest = {
        "manifest_version": "4.0.0",
        "created_at": datetime.now(UTC),
        "serial": str(serial),
        "chemistry": chemistry,
        "source_feature_run_ids": summary["source_feature_run_ids"],
        "source_soh_audit_run_id": audit_run.name,
        "artifacts": {
            "recalibration_events.parquet": "persistent or interval-censored BMS SOH transitions with multi-label UTC evidence",
            "soh_daily_timeline.parquet": "daily audited BMS SOH observations with event markers",
            "matched_control_windows.parquet": "seasonally and quality-matched non-event anchors",
            "evidence_comparison.parquet": "event-versus-control prevalence, lift, risk difference, and Wilson intervals",
            "calibration_discharge_episodes.parquet": "ordered negative-current discharge episodes with SOC traversal, coverage, throughput, and qualification evidence",
            "one_second_capacity_validation.parquet": "bounded raw-current integration, usable-capacity SOH comparison, uncertainty bounds, and cycle continuity evidence",
            "capacity_discharge_candidates.parquet": "separate compact 95-5 recalibration and 99-1 full-SOC candidate intervals used for bounded raw validation",
            "interpretable_model.json": "regularized logistic event-versus-control baseline with grouped temporal validation",
        },
    }
    safe_summary = _json_safe(summary)
    (run_dir / "model_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "interpretable_model.json").write_text(
        json.dumps(_json_safe(interpretable_model), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_html_report(run_dir / "bms_soh_recalibration_report.html", safe_summary, events)
    logger.info("BMS SOH recalibration evidence analysis complete: %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify audited BMS SOH step events using compact operating evidence."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--nominal-capacity-ah", required=True, type=float)
    parser.add_argument("--rest-current-threshold-a", type=float, default=0.5)
    parser.add_argument("--minimum-window-coverage", type=float, default=0.8)
    parser.add_argument("--minimum-context-coverage", type=float, default=0.8)
    parser.add_argument("--context-lookback-hours", type=int, default=72)
    parser.add_argument("--near-full-soc-pct", type=float, default=99.0)
    parser.add_argument("--low-soc-pct", type=float, default=10.0)
    parser.add_argument("--minimum-rest-minutes", type=int, default=120)
    parser.add_argument("--data-gap-risk-minutes", type=int, default=60)
    parser.add_argument("--recurrence-window-days", type=int, default=30)
    parser.add_argument("--raw-search-before-hours", type=int, default=30)
    parser.add_argument("--raw-search-after-hours", type=int, default=30)
    parser.add_argument("--persistence-hours", type=int, default=6)
    parser.add_argument("--minimum-new-level-fraction", type=float, default=0.95)
    parser.add_argument("--previous-support-minutes", type=int, default=30)
    parser.add_argument("--exact-gap-seconds", type=int, default=60)
    parser.add_argument("--controls-per-event", type=int, default=5)
    parser.add_argument("--control-event-exclusion-days", type=int, default=3)
    parser.add_argument("--discharge-start-soc-pct", type=float, default=95.0)
    parser.add_argument("--discharge-end-soc-pct", type=float, default=5.0)
    parser.add_argument("--minimum-discharge-soc-drop-pct", type=float, default=90.0)
    parser.add_argument(
        "--minimum-discharge-dominance-fraction", type=float, default=0.8
    )
    parser.add_argument("--discharge-episode-gap-minutes", type=int, default=30)
    parser.add_argument("--capacity-validation-start-soc-pct", type=float, default=99.0)
    parser.add_argument("--capacity-validation-end-soc-pct", type=float, default=1.0)
    parser.add_argument(
        "--capacity-validation-minimum-soc-drop-pct", type=float, default=98.0
    )
    parser.add_argument(
        "--capacity-validation-minimum-coverage-fraction",
        type=float,
        default=0.98,
    )
    parser.add_argument(
        "--capacity-validation-maximum-unobserved-ah", type=float, default=0.25
    )
    parser.add_argument(
        "--capacity-validation-maximum-charge-intrusion-ah",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--capacity-validation-maximum-integrated-gap-seconds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--capacity-validation-raw-padding-minutes", type=int, default=10
    )
    parser.add_argument(
        "--continuous-cycle-maximum-rest-minutes", type=float, default=5.0
    )
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--soh-audit-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--compression", default="zstd", choices=("snappy", "zstd", "gzip", "none")
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    try:
        storage = StorageConfig.from_env(args.env_file)
        run_dir = run_analysis(
            serial=str(args.serial),
            chemistry=args.chemistry,
            module_count=args.module_count,
            nominal_capacity_ah=args.nominal_capacity_ah,
            raw_root=args.raw_root or storage.raw_parquet_dir,
            feature_root=args.feature_root
            or storage.data_dir / "processed" / "bms_hierarchical_features",
            soh_audit_root=args.soh_audit_root
            or storage.data_dir / "processed" / "fleet_soh_audit",
            output_root=args.output_root
            or storage.data_dir / "processed" / "bms_soh_recalibration",
            rest_current_threshold_a=args.rest_current_threshold_a,
            minimum_window_coverage=args.minimum_window_coverage,
            minimum_context_coverage=args.minimum_context_coverage,
            context_lookback_hours=args.context_lookback_hours,
            near_full_soc_pct=args.near_full_soc_pct,
            low_soc_pct=args.low_soc_pct,
            minimum_rest_minutes=args.minimum_rest_minutes,
            data_gap_risk_minutes=args.data_gap_risk_minutes,
            recurrence_window_days=args.recurrence_window_days,
            raw_search_before_hours=args.raw_search_before_hours,
            raw_search_after_hours=args.raw_search_after_hours,
            persistence_hours=args.persistence_hours,
            minimum_new_level_fraction=args.minimum_new_level_fraction,
            previous_support_minutes=args.previous_support_minutes,
            exact_gap_seconds=args.exact_gap_seconds,
            controls_per_event=args.controls_per_event,
            control_event_exclusion_days=args.control_event_exclusion_days,
            discharge_start_soc_pct=args.discharge_start_soc_pct,
            discharge_end_soc_pct=args.discharge_end_soc_pct,
            minimum_discharge_soc_drop_pct=args.minimum_discharge_soc_drop_pct,
            minimum_discharge_dominance_fraction=(
                args.minimum_discharge_dominance_fraction
            ),
            discharge_episode_gap_minutes=args.discharge_episode_gap_minutes,
            capacity_validation_start_soc_pct=(
                args.capacity_validation_start_soc_pct
            ),
            capacity_validation_end_soc_pct=args.capacity_validation_end_soc_pct,
            capacity_validation_minimum_soc_drop_pct=(
                args.capacity_validation_minimum_soc_drop_pct
            ),
            capacity_validation_minimum_coverage_fraction=(
                args.capacity_validation_minimum_coverage_fraction
            ),
            capacity_validation_maximum_unobserved_ah=(
                args.capacity_validation_maximum_unobserved_ah
            ),
            capacity_validation_maximum_charge_intrusion_ah=(
                args.capacity_validation_maximum_charge_intrusion_ah
            ),
            capacity_validation_maximum_integrated_gap_seconds=(
                args.capacity_validation_maximum_integrated_gap_seconds
            ),
            capacity_validation_raw_padding_minutes=(
                args.capacity_validation_raw_padding_minutes
            ),
            continuous_cycle_maximum_rest_minutes=(
                args.continuous_cycle_maximum_rest_minutes
            ),
            compression=None if args.compression == "none" else args.compression,
        )
        print(run_dir)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("BMS SOH recalibration evidence analysis failed")
        else:
            logger.error("BMS SOH recalibration evidence analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
