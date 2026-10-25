from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg2 import sql

from src.config import DBConfig, StorageConfig
from src.db import DEFAULT_SERIAL_COL, DEFAULT_TABLE, DEFAULT_TIMESTAMP_COL, open_db_connection
from src.device_specs import FIELD_NOMINAL_AH
from src.electrothermal_contract import expected_hierarchical_signals
from src.io_utils import write_parquet_chunk

MODEL_FAMILY = "electrothermal_capacity_context"
PHASE3_VERSION = "v2"
REFERENCE_SERIAL_BY_CHEMISTRY = {"NCA": "100000017"}
CAPACITY_CLASSES = {
    "strict_full_discharge",
    "near_full_discharge",
    "operational_full_discharge",
}


def _latest_run(root: Path, serial: str, prefix: str) -> Path | None:
    runs = sorted((root / f"serial={serial}").glob(f"{prefix}=*"))
    return runs[-1] if runs else None


def _reference_capacity_events(capacity_run: Path) -> pd.DataFrame:
    """Adapt the slope-training 100-to-0 catalog to the Phase 3 event contract."""
    measurements = pd.read_parquet(capacity_run / "capacity_measurements.parquet")
    if measurements.empty:
        return measurements
    event_ids = pd.to_numeric(measurements["event_discharge_id"], errors="raise").astype(int)
    quality = measurements["capacity_measurement_quality"].map(
        {"clean_anchor_100_to_0": "high", "long_rest_interrupted": "medium"}
    ).fillna("indeterminate")
    return pd.DataFrame(
        {
            "discharge_event_id": event_ids.map(lambda value: f"reference_full_cycle_{value:06d}"),
            "capacity_claim_eligible": True,
            "measurement_resolution": "one_second",
            "evidence_class": "strict_full_discharge",
            "start_utc": pd.to_datetime(measurements["start_anchor_time"], utc=True),
            "end_utc": pd.to_datetime(measurements["end_anchor_time"], utc=True),
            "soc_start_pct": 100.0,
            "soc_end_pct": 0.0,
            "soc_span_pp": 100.0,
            "discharged_ah": pd.to_numeric(measurements["capacity_ah"], errors="coerce"),
            "discharged_wh": pd.to_numeric(measurements["capacity_wh"], errors="coerce"),
            "measurement_quality": quality,
            "data_coverage_fraction": np.nan,
            "maximum_gap_s": np.nan,
            "rest_duration_s": pd.to_numeric(
                measurements["rest_duration_seconds"], errors="coerce"
            ),
            "num_discharge_events": pd.to_numeric(
                measurements["num_discharge_events"], errors="coerce"
            ),
            "source_capacity_event_id": event_ids,
            "source_capacity_run_id": capacity_run.name,
            "source_semantics": "canonical exact-100-to-0 anchor cycle used by slope training",
        }
    )


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


EVIDENCE_PRIORITY = {
    "strict_full_discharge": 0,
    "near_full_discharge": 1,
    "operational_full_discharge": 2,
}


def _canonicalize_physical_cycles(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse overlapping evidence windows onto one physical discharge cycle."""
    if events.empty:
        return events.copy()
    frame = events.copy()
    frame["start_utc"] = pd.to_datetime(frame["start_utc"], utc=True)
    frame["end_utc"] = pd.to_datetime(frame["end_utc"], utc=True)
    frame = frame.sort_values(["start_utc", "end_utc"])
    cluster_ids: list[int] = []
    cluster = -1
    active_end: pd.Timestamp | None = None
    for row in frame.itertuples(index=False):
        start = pd.Timestamp(row.start_utc)
        end = pd.Timestamp(row.end_utc)
        if active_end is None or start > active_end:
            cluster += 1
            active_end = end
        else:
            active_end = max(active_end, end)
        cluster_ids.append(cluster)
    frame["physical_cycle_id"] = [f"physical_cycle_{value:06d}" for value in cluster_ids]
    frame["_evidence_priority"] = frame["evidence_class"].map(EVIDENCE_PRIORITY).fillna(9)
    frame["_quality_priority"] = frame["measurement_quality"].map(
        {"high": 0, "medium": 1, "low": 2, "indeterminate": 3}
    ).fillna(4)
    if "soc_span_pp" not in frame:
        frame["soc_span_pp"] = np.nan
    canonical = (
        frame.sort_values(
            [
                "physical_cycle_id",
                "_evidence_priority",
                "_quality_priority",
                "data_coverage_fraction",
                "soc_span_pp",
            ],
            ascending=[True, True, True, False, False],
        )
        .groupby("physical_cycle_id", as_index=False)
        .head(1)
        .drop(columns=["_evidence_priority", "_quality_priority"])
    )
    return canonical.sort_values("start_utc")


def select_phase3_events(events: pd.DataFrame, *, serial: str) -> pd.DataFrame:
    selected = events.loc[
        events["capacity_claim_eligible"].fillna(False)
        & events["measurement_resolution"].astype(str).str.startswith("one_second")
        & events["evidence_class"].isin(CAPACITY_CLASSES)
        & events["start_utc"].notna()
        & events["end_utc"].notna()
    ].copy()
    selected = _canonicalize_physical_cycles(selected)
    if serial == REFERENCE_SERIAL_BY_CHEMISTRY.get("NCA") and not selected.empty:
        selected["_month"] = pd.to_datetime(selected["end_utc"], utc=True).dt.strftime("%Y-%m")
        stronger = selected["evidence_class"].isin(
            {"strict_full_discharge", "near_full_discharge"}
        )
        months_with_stronger = set(selected.loc[stronger, "_month"])
        selected = selected.loc[
            stronger
            | (
                selected["evidence_class"].eq("operational_full_discharge")
                & ~selected["_month"].isin(months_with_stronger)
            )
        ].drop(columns="_month")
    return selected.sort_values("start_utc").drop_duplicates("discharge_event_id")


def _extract_event_frame(
    connection: Any,
    *,
    serial: str,
    module_count: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pre_minutes: int,
    post_minutes: int,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> pd.DataFrame:
    signals = expected_hierarchical_signals(module_count)
    source_columns = [signal.source_signal_name for signal in signals]
    core_columns = [
        "batt_soc_pct",
        "batt_soh_pct",
        "batt_current_a",
        "batt_voltage_v",
    ]
    identifiers = [timestamp_col, *core_columns, *source_columns]
    query = sql.SQL(
        "SELECT {columns} FROM {table} WHERE {serial_col} = %s "
        "AND {timestamp_col} >= %s AND {timestamp_col} <= %s "
        "ORDER BY {timestamp_col}"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(item) for item in identifiers),
        table=sql.SQL(".").join(sql.Identifier(item) for item in table.split(".")),
        serial_col=sql.Identifier(serial_col),
        timestamp_col=sql.Identifier(timestamp_col),
    )
    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                serial,
                start - pd.Timedelta(minutes=pre_minutes),
                end + pd.Timedelta(minutes=post_minutes),
            ),
        )
        rows = cursor.fetchall()
        names = [item[0] for item in cursor.description]
    frame = pd.DataFrame.from_records(rows, columns=names)
    if not frame.empty:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
    return frame


def _dwell_minutes(mask: pd.Series, timestamps: pd.Series) -> float:
    if len(mask) < 2:
        return 0.0
    seconds = timestamps.diff().dt.total_seconds().fillna(0).clip(lower=0, upper=5)
    return float(seconds.where(mask.fillna(False), 0).sum() / 60.0)


def _time_weighted_mean(values: pd.Series, timestamps: pd.Series) -> float | None:
    seconds = timestamps.shift(-1).sub(timestamps).dt.total_seconds().clip(lower=0, upper=5)
    valid = values.notna() & seconds.gt(0)
    if not valid.any() or float(seconds.loc[valid].sum()) <= 0:
        return _finite(values.mean())
    return _finite(np.average(values.loc[valid], weights=seconds.loc[valid]))


def _soc_band_metrics(
    *,
    soc: pd.Series,
    current: pd.Series,
    temperature_mean: pd.Series,
    voltage_spread: pd.Series,
    pack_delta: pd.Series,
    target_soc: float,
    tolerance_pp: float = 1.0,
) -> dict[str, Any]:
    mask = (
        soc.between(target_soc - tolerance_pp, target_soc + tolerance_pp)
        & current.lt(-0.5)
    )
    return {
        f"soc_{int(target_soc)}_sample_rows": int(mask.sum()),
        f"soc_{int(target_soc)}_current_median_a": _finite((-current.loc[mask]).median()),
        f"soc_{int(target_soc)}_temperature_median_c": _finite(temperature_mean.loc[mask].median()),
        f"soc_{int(target_soc)}_voltage_spread_median_mv": _finite(voltage_spread.loc[mask].median()),
        f"soc_{int(target_soc)}_voltage_spread_p95_mv": _finite(voltage_spread.loc[mask].quantile(0.95)),
        f"soc_{int(target_soc)}_pack_delta_median_v": _finite(pack_delta.loc[mask].median()),
    }


def summarize_event_frame(
    frame: pd.DataFrame,
    *,
    event: pd.Series,
    module_count: int,
    nominal_capacity_ah: float,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> tuple[dict[str, Any], pd.DataFrame]:
    start = pd.Timestamp(event["start_utc"])
    end = pd.Timestamp(event["end_utc"])
    in_event = frame[timestamp_col].between(start, end)
    data = frame.loc[in_event].copy()
    signals = expected_hierarchical_signals(module_count)
    voltage_signals = [s for s in signals if s.measurement_type == "cell_voltage_channel"]
    temperature_signals = [s for s in signals if s.measurement_type == "pack_temperature_sensor"]
    voltage_columns = [s.source_signal_name for s in voltage_signals]
    temperature_columns = [s.source_signal_name for s in temperature_signals]
    primary_temperature_columns = [
        s.source_signal_name for s in temperature_signals if s.source_channel_index in (0, 1)
    ]
    if data.empty:
        return {
            "discharge_event_id": event["discharge_event_id"],
            "phase3_status": "indeterminate",
            "phase3_reasons": ["no_native_event_rows"],
        }, pd.DataFrame()

    core = data[["batt_soc_pct", "batt_soh_pct", "batt_current_a", "batt_voltage_v"]].apply(
        pd.to_numeric, errors="coerce"
    )
    all_zero_source_row = core.notna().all(axis=1) & core.eq(0.0).all(axis=1)
    data = data.loc[~all_zero_source_row].copy()
    if data.empty:
        return {
            "discharge_event_id": event["discharge_event_id"],
            "phase3_status": "indeterminate",
            "phase3_reasons": ["only_invalid_all_zero_source_rows"],
        }, pd.DataFrame()

    current = pd.to_numeric(data["batt_current_a"], errors="coerce")
    discharge_current = (-current).where(current < 0)
    soc = pd.to_numeric(data["batt_soc_pct"], errors="coerce")
    stack_voltage = pd.to_numeric(data["batt_voltage_v"], errors="coerce")
    voltage = data[voltage_columns].apply(pd.to_numeric, errors="coerce")
    voltage = voltage.where((voltage >= 2000.0) & (voltage <= 5000.0))
    temperature = data[temperature_columns].apply(pd.to_numeric, errors="coerce")
    temperature = temperature.where((temperature >= -40.0) & (temperature <= 100.0))
    primary_temperature = temperature[primary_temperature_columns]
    temperature_mean = primary_temperature.mean(axis=1)
    temperature_spread = primary_temperature.max(axis=1) - primary_temperature.min(axis=1)
    voltage_spread = voltage.max(axis=1) - voltage.min(axis=1)
    lowest_column = voltage.idxmin(axis=1).str.extract(
        r"mod(?P<module>\d+)_pack(?P<pack>\d+)_cellvoltage_mv_(?P<channel>\d+)"
    )
    lowest_keys = lowest_column.apply(
        lambda row: (
            f"m{int(row['module']):02d}_p{int(row['pack']):02d}_cv{int(row['channel']):02d}"
            if row.notna().all()
            else None
        ),
        axis=1,
    )
    low_soc = soc <= 10.0
    recurring_lowest = lowest_keys.loc[low_soc & lowest_keys.notna()].value_counts()
    weakest = str(recurring_lowest.index[0]) if not recurring_lowest.empty else None
    weakest_fraction = (
        float(recurring_lowest.iloc[0] / recurring_lowest.sum())
        if not recurring_lowest.empty
        else None
    )

    pack_sums: dict[str, pd.Series] = {}
    for module in range(1, module_count + 1):
        for pack in (1, 2):
            columns = [
                f"batt_batterymodules_mod{module}_pack{pack}_cellvoltage_mv_{index}"
                for index in range(14)
            ]
            complete = voltage[columns].notna().all(axis=1)
            pack_sums[f"m{module:02d}_p{pack:02d}"] = voltage[columns].sum(axis=1).where(complete) / 1000.0
    pack_frame = pd.DataFrame(pack_sums)
    pack_delta = pack_frame.max(axis=1) - pack_frame.min(axis=1)
    pack_temperature_means: dict[str, pd.Series] = {}
    for module in range(1, module_count + 1):
        for pack in (1, 2):
            columns = [
                signal.source_signal_name
                for signal in temperature_signals
                if signal.module_index == module
                and signal.pack_index == pack
                and signal.source_channel_index in (0, 1)
            ]
            pack_temperature_means[f"m{module:02d}_p{pack:02d}"] = temperature[columns].mean(axis=1)
    pack_temperature_frame = pd.DataFrame(pack_temperature_means)

    timestamps = data[timestamp_col]
    expected_rows = max(1, int((end - start).total_seconds()) + 1)
    observed_distinct = int(timestamps.nunique())
    coverage = min(1.0, observed_distinct / expected_rows)
    temperature_valid_fraction = float(primary_temperature.notna().mean().mean())
    secondary_temperature_valid_fraction = float(
        temperature.drop(columns=primary_temperature_columns).notna().mean().mean()
    )
    voltage_valid_fraction = float(voltage.notna().mean().mean())
    reasons: list[str] = []
    if coverage < 0.95:
        reasons.append("insufficient_native_coverage")
    if temperature_valid_fraction < 0.8:
        reasons.append("insufficient_temperature_coverage")
    if voltage_valid_fraction < 0.8:
        reasons.append("insufficient_voltage_group_coverage")
    mean_temp = _time_weighted_mean(temperature_mean, timestamps)
    rms_current = _finite(np.sqrt(np.nanmean(np.square(discharge_current))))
    rms_c_rate = rms_current / nominal_capacity_ah if rms_current is not None else None
    low_soc_p95_spread = _finite(voltage_spread.loc[low_soc].quantile(0.95))
    p95_pack_delta = _finite(pack_delta.quantile(0.95))
    if mean_temp is not None and mean_temp < 10.0:
        reasons.append("cold_discharge_below_10c")
    elif mean_temp is not None and mean_temp < 15.0:
        reasons.append("cool_discharge_below_15c")
    if rms_c_rate is not None and rms_c_rate > 0.5:
        reasons.append("elevated_analytical_c_rate")
    if low_soc_p95_spread is not None and low_soc_p95_spread >= 50.0:
        reasons.append("meaningful_low_soc_voltage_group_spread")
    elif low_soc_p95_spread is not None and low_soc_p95_spread >= 30.0:
        reasons.append("low_soc_imbalance_candidate")
    if p95_pack_delta is not None and p95_pack_delta >= 0.25:
        reasons.append("pack_voltage_sum_asymmetry")
    if (
        weakest_fraction is not None
        and weakest_fraction >= 0.5
        and low_soc_p95_spread is not None
        and low_soc_p95_spread >= 20.0
    ):
        reasons.append("recurring_lowest_voltage_group")
    quality_blocked = any(reason.startswith("insufficient_") for reason in reasons)
    status = "indeterminate" if quality_blocked else "measured_context_ready"

    record = {
        "discharge_event_id": event["discharge_event_id"],
        "physical_cycle_id": event.get("physical_cycle_id"),
        "evidence_class": event["evidence_class"],
        "start_utc": start,
        "end_utc": end,
        "phase2_measurement_quality": event.get("measurement_quality"),
        "phase3_status": status,
        "phase3_reasons": reasons or ["no_absolute_electrothermal_screen_triggered"],
        "native_rows": len(data),
        "native_coverage_fraction": coverage,
        "temperature_valid_fraction": temperature_valid_fraction,
        "secondary_temperature_valid_fraction": secondary_temperature_valid_fraction,
        "voltage_group_valid_fraction": voltage_valid_fraction,
        "mean_discharge_current_a": _finite(discharge_current.mean()),
        "rms_discharge_current_a": rms_current,
        "p95_discharge_current_a": _finite(discharge_current.quantile(0.95)),
        "peak_discharge_current_a": _finite(discharge_current.max()),
        "rms_c_rate": rms_c_rate,
        "temperature_mean_c": mean_temp,
        "temperature_min_c": _finite(primary_temperature.min(axis=1).min()),
        "temperature_max_c": _finite(primary_temperature.max(axis=1).max()),
        "temperature_sensor_spread_p95_c": _finite(temperature_spread.quantile(0.95)),
        "minutes_below_15c": _dwell_minutes(temperature_mean < 15.0, timestamps),
        "minutes_below_10c": _dwell_minutes(temperature_mean < 10.0, timestamps),
        "minutes_above_30c": _dwell_minutes(temperature_mean > 30.0, timestamps),
        "minutes_above_35c": _dwell_minutes(temperature_mean > 35.0, timestamps),
        "minutes_above_40c": _dwell_minutes(temperature_mean > 40.0, timestamps),
        "stack_voltage_start_v": _finite(stack_voltage.iloc[0]),
        "stack_voltage_end_v": _finite(stack_voltage.iloc[-1]),
        "stack_voltage_min_v": _finite(stack_voltage.min()),
        "pack_voltage_sum_delta_p95_v": p95_pack_delta,
        "voltage_group_spread_p95_mv": _finite(voltage_spread.quantile(0.95)),
        "voltage_group_spread_max_mv": _finite(voltage_spread.max()),
        "low_soc_voltage_group_spread_p95_mv": low_soc_p95_spread,
        "recurring_lowest_voltage_group": weakest,
        "recurring_lowest_voltage_group_fraction": weakest_fraction,
        "pack_voltage_semantics": "derived sum of 14 measured voltage-group channels per pack",
        "temperature_semantics": "BMS pack temperature sensors; not cell-core temperature",
    }
    for key, values in pack_temperature_frame.items():
        record[f"{key}_temperature_mean_c"] = _time_weighted_mean(values, timestamps)
        record[f"{key}_temperature_min_c"] = _finite(values.min())
        record[f"{key}_temperature_max_c"] = _finite(values.max())
        record[f"{key}_minutes_below_15c"] = _dwell_minutes(values < 15.0, timestamps)
        record[f"{key}_minutes_above_30c"] = _dwell_minutes(values > 30.0, timestamps)
    for target_soc in (20.0, 10.0, 5.0):
        record.update(
            _soc_band_metrics(
                soc=soc,
                current=current,
                temperature_mean=temperature_mean,
                voltage_spread=voltage_spread,
                pack_delta=pack_delta,
                target_soc=target_soc,
            )
        )
    final_mask = data.index.isin(data.tail(min(60, len(data))).index)
    record.update(
        {
            "final_soc_pct": _finite(soc.iloc[-1]),
            "final_current_median_a": _finite((-current.loc[final_mask]).median()),
            "final_temperature_median_c": _finite(temperature_mean.loc[final_mask].median()),
            "final_voltage_spread_median_mv": _finite(voltage_spread.loc[final_mask].median()),
            "final_pack_delta_median_v": _finite(pack_delta.loc[final_mask].median()),
        }
    )
    points = pd.DataFrame(
        {
            "discharge_event_id": event["discharge_event_id"],
            "physical_cycle_id": event.get("physical_cycle_id"),
            "event_start_utc": start,
            "event_end_utc": end,
            "event_soc_start_pct": event.get("soc_start_pct"),
            "event_soc_end_pct": event.get("soc_end_pct"),
            "time_utc": timestamps,
            "soc_pct": soc,
            "current_a": current,
            "stack_voltage_v": stack_voltage,
            "temperature_mean_c": temperature_mean,
            "temperature_spread_c": temperature_spread,
            "voltage_group_spread_mv": voltage_spread,
            "lowest_voltage_group": lowest_keys,
            "pack_voltage_sum_delta_v": pack_delta,
        }
    )
    for key, values in pack_frame.items():
        points[f"{key}_derived_voltage_v"] = values
    for key, values in pack_temperature_frame.items():
        points[f"{key}_temperature_mean_c"] = values
    points = points.iloc[::60].copy()
    return record, points


def _add_reference_comparability(
    contexts: pd.DataFrame,
    reference: pd.DataFrame | None,
    *,
    serial: str,
    reference_serial: str | None,
) -> pd.DataFrame:
    result = contexts.copy()
    statuses: list[str] = []
    reasons_out: list[list[str]] = []
    reference_ids: list[str | None] = []
    for _, row in result.iterrows():
        reasons = list(row.get("phase3_reasons") or [])
        selected: pd.Series | None = None
        if row.get("phase3_status") != "measured_context_ready":
            status = "indeterminate"
        elif serial == reference_serial:
            status = "reference_measurement"
            reasons.append("reference_device_event")
        elif reference is None or reference.empty:
            status = "indeterminate"
            reasons.append("phase3_reference_context_missing")
        else:
            required_rank = EVIDENCE_PRIORITY.get(str(row["evidence_class"]), 9)
            reference_rank = reference["evidence_class"].map(EVIDENCE_PRIORITY).fillna(9)
            candidates = reference.loc[
                reference_rank.le(required_rank)
                & reference["phase3_status"].eq("measured_context_ready")
            ].copy()
            if candidates.empty:
                status = "indeterminate"
                reasons.append("matching_reference_evidence_class_missing")
            else:
                candidates["_distance"] = 0.0
                # Voltage spread is an outcome, not a matching variable. Including it
                # here would preferentially select references that agree with the field event.
                for column, scale in (
                    ("temperature_mean_c", 5.0),
                    ("rms_c_rate", 0.1),
                ):
                    value = _finite(row.get(column))
                    if value is not None:
                        candidates["_distance"] += (
                            pd.to_numeric(candidates[column], errors="coerce")
                            .sub(value).abs().fillna(scale * 4) / scale
                        )
                selected = candidates.sort_values("_distance").iloc[0]
                temp_delta = abs(float(row["temperature_mean_c"]) - float(selected["temperature_mean_c"]))
                c_rate_delta = abs(float(row["rms_c_rate"]) - float(selected["rms_c_rate"]))
                spread_value = _finite(row.get("low_soc_voltage_group_spread_p95_mv"))
                reference_spread = _finite(selected.get("low_soc_voltage_group_spread_p95_mv"))
                spread_delta = (
                    abs(spread_value - reference_spread)
                    if spread_value is not None and reference_spread is not None
                    else None
                )
                reasons.extend([
                    f"reference_mean_temperature_delta_c:{temp_delta:.2f}",
                    f"reference_rms_c_rate_delta:{c_rate_delta:.3f}",
                ])
                if spread_delta is not None:
                    reasons.append(f"reference_low_soc_spread_delta_mv:{spread_delta:.2f}")
                matched_bands = 0
                for target_soc in (20, 10, 5):
                    field_rows = _finite(row.get(f"soc_{target_soc}_sample_rows")) or 0
                    reference_rows = _finite(selected.get(f"soc_{target_soc}_sample_rows")) or 0
                    field_current = _finite(row.get(f"soc_{target_soc}_current_median_a"))
                    reference_current = _finite(selected.get(f"soc_{target_soc}_current_median_a"))
                    field_temp = _finite(row.get(f"soc_{target_soc}_temperature_median_c"))
                    reference_temp = _finite(selected.get(f"soc_{target_soc}_temperature_median_c"))
                    if (
                        field_rows > 0
                        and reference_rows > 0
                        and field_current is not None
                        and reference_current is not None
                        and field_temp is not None
                        and reference_temp is not None
                        and abs(field_current - reference_current) / max(float(row.get("nominal_capacity_ah", FIELD_NOMINAL_AH) or FIELD_NOMINAL_AH), 1e-9) <= 0.25
                        and abs(field_temp - reference_temp) <= 10.0
                    ):
                        matched_bands += 1
                reasons.append(f"fixed_soc_matched_bands:{matched_bands}")
                if temp_delta <= 5.0 and c_rate_delta <= 0.1 and matched_bands == 3:
                    status = "high"
                elif temp_delta <= 10.0 and c_rate_delta <= 0.25 and matched_bands >= 2:
                    status = "medium"
                else:
                    status = "low"
                    reasons.append("operating_conditions_differ_materially_from_reference")
        statuses.append(status)
        reasons_out.append(reasons)
        reference_ids.append(str(selected["discharge_event_id"]) if selected is not None else None)
    result["reference_comparability"] = statuses
    result["reference_comparability_reasons"] = reasons_out
    result["comparison_reference_serial"] = reference_serial
    result["comparison_reference_event_id"] = reference_ids
    return result


def run_phase3(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
    recalibration_root: Path,
    capacity_trend_root: Path,
    physical_catalog_root: Path,
    output_root: Path,
    cache_root: Path,
    env_file: Path | None = None,
    pre_minutes: int = 30,
    post_minutes: int = 30,
    table: str = DEFAULT_TABLE,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    serial_col: str = DEFAULT_SERIAL_COL,
    compression: str | None = "zstd",
) -> Path:
    phase2_run = _latest_run(recalibration_root, serial, "recalibration_run")
    if phase2_run is None or not (phase2_run / "phase2_manifest.json").exists():
        raise ValueError(f"Current Phase 2 output is missing for {serial}")
    reference_serial = REFERENCE_SERIAL_BY_CHEMISTRY.get(chemistry)
    reference_capacity_run: Path | None = None
    physical_catalog_run = _latest_run(physical_catalog_root, serial, "catalog_run")
    if serial == reference_serial:
        reference_capacity_run = _latest_run(capacity_trend_root, serial, "capacity_run")
    if physical_catalog_run is not None and (
        physical_catalog_run / "physical_discharge_cycles.parquet"
    ).exists():
        events = pd.read_parquet(physical_catalog_run / "physical_discharge_cycles.parquet")
    elif reference_capacity_run is not None and (
        reference_capacity_run / "capacity_measurements.parquet"
    ).exists():
        events = _reference_capacity_events(reference_capacity_run)
    else:
        events = pd.read_parquet(phase2_run / "capacity_discharge_events.parquet")
    selected = select_phase3_events(events, serial=serial)
    if selected.empty:
        raise ValueError(f"No one-second capacity events are eligible for Phase 3 on {serial}")
    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"chemistry={chemistry}" / f"serial={serial}" / f"phase3_run={run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = cache_root / f"chemistry={chemistry}" / f"serial={serial}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    contexts: list[dict[str, Any]] = []
    point_frames: list[pd.DataFrame] = []
    db_config = DBConfig.from_env(env_file)
    with ExitStack() as stack:
        connection = None
        for _, event in selected.iterrows():
            event_id = str(event["discharge_event_id"])
            cache_identity = hashlib.sha256(
                (
                    f"{event_id}|{event['start_utc']}|{event['end_utc']}|"
                    f"{module_count}|{pre_minutes}|{post_minutes}"
                ).encode("utf-8")
            ).hexdigest()[:12]
            cache_path = cache_dir / f"event={event_id}__window={cache_identity}.parquet"
            if cache_path.exists():
                frame = pd.read_parquet(cache_path)
                frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True)
            else:
                if connection is None:
                    connection = stack.enter_context(open_db_connection(db_config))
                frame = _extract_event_frame(
                    connection,
                    serial=serial,
                    module_count=module_count,
                    start=pd.Timestamp(event["start_utc"]),
                    end=pd.Timestamp(event["end_utc"]),
                    pre_minutes=pre_minutes,
                    post_minutes=post_minutes,
                    table=table,
                    timestamp_col=timestamp_col,
                    serial_col=serial_col,
                )
                if not frame.empty:
                    write_parquet_chunk(frame, cache_path, compression=compression)
            record, points = summarize_event_frame(
                frame,
                event=event,
                module_count=module_count,
                nominal_capacity_ah=nominal_capacity_ah,
                timestamp_col=timestamp_col,
            )
            contexts.append(record)
            if not points.empty:
                point_frames.append(points)
    context_frame = pd.DataFrame.from_records(contexts)
    context_frame["nominal_capacity_ah"] = nominal_capacity_ah
    reference_context: pd.DataFrame | None = None
    if reference_serial and reference_serial != serial:
        reference_runs = sorted(
            (output_root / f"chemistry={chemistry}" / f"serial={reference_serial}").glob("phase3_run=*")
        )
        if reference_runs:
            reference_context = pd.read_parquet(reference_runs[-1] / "event_electrothermal_context.parquet")
    context_frame = _add_reference_comparability(
        context_frame,
        reference_context,
        serial=serial,
        reference_serial=reference_serial,
    )
    write_parquet_chunk(context_frame, run_dir / "event_electrothermal_context.parquet", compression=compression)
    points = pd.concat(point_frames, ignore_index=True) if point_frames else pd.DataFrame()
    if not points.empty:
        write_parquet_chunk(points, run_dir / "event_verification_points.parquet", compression=compression)
    counts = context_frame["reference_comparability"].value_counts().to_dict()
    summary = {
        "model_family": MODEL_FAMILY,
        "phase3_version": PHASE3_VERSION,
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "nominal_capacity_ah": nominal_capacity_ah,
        "timestamp_semantics": "UTC",
        "current_sign_convention": "positive=charge; negative=discharge",
        "source_phase2_run_id": phase2_run.name,
        "source_reference_capacity_run_id": (
            reference_capacity_run.name if reference_capacity_run is not None else None
        ),
        "source_physical_catalog_run_id": (
            physical_catalog_run.name if physical_catalog_run is not None else None
        ),
        "selected_event_rows": len(selected),
        "measured_context_ready_rows": int(context_frame["phase3_status"].eq("measured_context_ready").sum()),
        "reference_comparability_counts": counts,
        "configuration": {
            "pre_event_minutes": pre_minutes,
            "post_event_minutes": post_minutes,
            "temperature_dwell_thresholds_c": ["<15", "<10", ">30", ">35", ">40"],
            "voltage_spread_engineering_bands_mv": [10, 20, 30, 50, 100, 200],
            "reference_temperature_delta_c": {"high": 5, "medium": 10},
            "reference_rms_c_rate_delta": {"high": 0.1, "medium": 0.25},
            "fixed_soc_bands_pct": [20, 10, 5],
            "fixed_soc_tolerance_pp": 1.0,
            "fixed_soc_matching": {
                "high": "all three bands, temperature delta <=5 C, RMS C-rate delta <=0.1",
                "medium": "at least two bands, temperature delta <=10 C, RMS C-rate delta <=0.25",
                "voltage_spread_role": "comparison outcome; never used to choose a reference",
            },
        },
        "artifacts": {
            "event_electrothermal_context.parquet": "one row per Phase 2 capacity event",
            "event_verification_points.parquet": "one-minute display points derived from cached native rows",
        },
        "claim_limit": (
            "Phase 3 assesses whether measured electrothermal conditions make a usable-capacity observation comparable with the chemistry reference. "
            "It does not alter Phase 2 capacity, prove permanent degradation, estimate resistance, or diagnose a failed cell."
        ),
    }
    (run_dir / "model_summary.json").write_text(
        json.dumps(_safe_json(summary), indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "manifest_version": "1.0.0",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_phase2_manifest": str(phase2_run / "phase2_manifest.json"),
        "source_reference_capacity_run": (
            str(reference_capacity_run) if reference_capacity_run is not None else None
        ),
        "cache_root": str(cache_dir),
        "selected_event_ids": selected["discharge_event_id"].astype(str).tolist(),
        "source_semantics": "native BMS database telemetry; independently uncalibrated",
    }
    (run_dir / "phase3_manifest.json").write_text(
        json.dumps(_safe_json(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build measured Phase 3 electrothermal capacity context.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--nominal-capacity-ah", required=True, type=float)
    parser.add_argument("--recalibration-root", type=Path)
    parser.add_argument("--capacity-trend-root", type=Path)
    parser.add_argument("--physical-catalog-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--pre-minutes", type=int, default=30)
    parser.add_argument("--post-minutes", type=int, default=30)
    parser.add_argument("--compression", default="zstd", choices=("zstd", "snappy", "gzip", "none"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = StorageConfig.from_env(args.env_file)
    compression = None if args.compression == "none" else args.compression
    try:
        run_dir = run_phase3(
            serial=str(args.serial),
            chemistry=args.chemistry,
            module_count=args.module_count,
            nominal_capacity_ah=args.nominal_capacity_ah,
            recalibration_root=args.recalibration_root or storage.data_dir / "processed" / "bms_soh_recalibration",
            capacity_trend_root=args.capacity_trend_root or storage.data_dir / "processed" / "capacity_trend",
            physical_catalog_root=args.physical_catalog_root or storage.data_dir / "processed" / "physical_discharge_catalog",
            output_root=args.output_root or storage.data_dir / "processed" / "electrothermal_capacity_context",
            cache_root=args.cache_root or storage.data_dir / "cache" / "electrothermal_event_windows",
            env_file=args.env_file,
            pre_minutes=args.pre_minutes,
            post_minutes=args.post_minutes,
            compression=compression,
        )
        print(f"Phase 3 electrothermal context complete: {run_dir}")
        return 0
    except Exception as exc:
        print(f"Phase 3 electrothermal context failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
