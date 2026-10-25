from __future__ import annotations

import argparse
import calendar
import html
import json
import logging
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import duckdb

from src.config import StorageConfig
from src.io_utils import write_parquet_chunk

logger = logging.getLogger(__name__)

MODEL_FAMILY = "seasonal_charging_stress_profile"
ANALYSIS_VERSION = "v2"
C_RATE_BANDS = (
    ("c0", 0.0, 0.01),
    ("c1", 0.01, 0.25),
    ("c2", 0.25, 0.50),
    ("c3", 0.50, 1.00),
    ("c4", 1.00, math.inf),
)
TEMPERATURE_COLUMN = re.compile(
    r"^module_\d{2}_pack_\d{2}_temperature_sensor_(00|01)_(mean|max)_c$"
)

CONTEXT_COLUMNS = (
    "window_start_utc",
    "serial",
    "bms_soc_mean_pct",
    "bms_soc_min_pct",
    "bms_soc_max_pct",
    "bms_soh_mean_pct",
    "bms_current_mean_a",
    "bms_current_min_a",
    "bms_current_max_a",
    "bms_current_max_abs_a",
    "bms_stack_voltage_mean_v",
    "bms_power_mean_w",
    "coverage_fraction",
)
SYSTEM_COLUMNS = (
    "window_start_utc",
    "serial",
    "system_voltage_spread_mv",
    "system_temperature_spread_c",
    "system_temperature_mean_c",
    "system_temperature_min_c",
    "system_temperature_max_c",
)
SOC_EXPOSURE_MINUTE_COLUMNS = (
    "soc_0_5_minutes",
    "soc_5_20_minutes",
    "soc_20_40_minutes",
    "soc_40_60_minutes",
    "soc_60_80_minutes",
    "soc_80_95_minutes",
    "soc_95_100_minutes",
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _display_number(value: Any, digits: int = 1) -> str:
    number = _finite(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return None
    return float(np.average(values.loc[valid].astype(float), weights=weights.loc[valid]))


def _quantile(values: pd.Series, probability: float) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(probability)) if not numeric.empty else None


def _read_projected(
    paths: list[Path],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> pd.DataFrame:
    if not paths:
        raise ValueError("No compatible compact feature partitions were found")
    schema_names = {name for path in paths for name in pq.ParquetFile(path).schema_arrow.names}
    missing = sorted(set(required) - schema_names)
    if missing:
        raise ValueError("Compact feature history is missing: " + ", ".join(missing))
    selected = tuple(dict.fromkeys((*required, *optional)))
    frames: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [name for name in selected if name in available]
        frame = pq.ParquetFile(path).read(columns=columns).to_pandas()
        for name in selected:
            if name not in frame:
                frame[name] = np.nan
        frames.append(frame.loc[:, selected])
    result = pd.concat(frames, ignore_index=True)
    result["window_start_utc"] = pd.to_datetime(
        result["window_start_utc"], errors="coerce", utc=True
    )
    return (
        result.dropna(subset=["window_start_utc"])
        .sort_values("window_start_utc")
        .drop_duplicates(["window_start_utc", "serial"], keep="last")
        .reset_index(drop=True)
    )


def _compatible_feature_runs(
    feature_root: Path, serial: str
) -> tuple[list[Path], dict[str, Any]]:
    summary_paths = sorted(
        (feature_root / f"serial={serial}").glob("feature_run=*/model_summary.json")
    )
    if not summary_paths:
        raise ValueError(f"No hierarchical feature history exists for serial {serial}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    latest = summaries[-1]
    aggregation = int(latest.get("aggregation_minutes") or 0)
    module_count = int(latest.get("module_count") or 0)
    compatible = [
        path.parent
        for path, summary in zip(summary_paths, summaries, strict=True)
        if int(summary.get("aggregation_minutes") or 0) == aggregation
        and int(summary.get("module_count") or 0) == module_count
    ]
    latest["compatible_feature_run_ids"] = [path.name for path in compatible]
    return compatible, latest


def read_compact_history(
    feature_root: Path, serial: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_dirs, summary = _compatible_feature_runs(feature_root, serial)
    aggregation = int(summary["aggregation_minutes"])
    suffix = f"{aggregation}min.parquet"

    def paths(prefix: str) -> list[Path]:
        return [
            path
            for run_dir in run_dirs
            for path in sorted(run_dir.glob(f"year=*/month=*/{prefix}{suffix}"))
        ]

    context = _read_projected(paths("operating_context_"), required=CONTEXT_COLUMNS)
    system = _read_projected(paths("system_features_"), required=SYSTEM_COLUMNS)
    temperature_paths = paths("temperature_sensor_features_")
    temperature_names = sorted(
        {
            name
            for path in temperature_paths
            for name in pq.ParquetFile(path).schema_arrow.names
            if TEMPERATURE_COLUMN.match(name)
        }
    )
    if not temperature_names:
        raise ValueError("No stable pack-temperature sensor 00/01 features were found")
    temperature = _read_projected(
        temperature_paths,
        required=("window_start_utc", "serial"),
        optional=temperature_names,
    )
    keys = ["window_start_utc", "serial"]
    frame = context.merge(system, on=keys, how="left", validate="one_to_one")
    frame = frame.merge(temperature, on=keys, how="left", validate="one_to_one")
    mean_columns = [name for name in temperature_names if name.endswith("_mean_c")]
    max_columns = [name for name in temperature_names if name.endswith("_max_c")]
    frame["stable_pack_temperature_mean_c"] = frame[mean_columns].mean(axis=1)
    frame["stable_pack_temperature_max_c"] = frame[max_columns].max(axis=1)
    frame["stable_temperature_sensor_count"] = frame[mean_columns].notna().sum(axis=1)
    summary["stable_temperature_feature_columns"] = mean_columns
    summary["stable_temperature_sensor_count"] = len(mean_columns)
    return frame.sort_values("window_start_utc").reset_index(drop=True), summary


def read_latest_soh_audit(
    audit_root: Path, serial: str
) -> tuple[pd.DataFrame, pd.DataFrame, Path, dict[str, Any]]:
    summaries = sorted(
        (audit_root / f"serial={serial}").glob("audit_run=*/model_summary.json")
    )
    usable = [
        path
        for path in summaries
        if (path.parent / "daily_health.parquet").exists()
        and (path.parent / "soh_change_events.parquet").exists()
    ]
    if not usable:
        raise ValueError(f"No completed BMS SOH audit exists for serial {serial}")
    summary_path = usable[-1]
    run_dir = summary_path.parent
    daily = pq.ParquetFile(run_dir / "daily_health.parquet").read().to_pandas()
    events = pq.ParquetFile(run_dir / "soh_change_events.parquet").read().to_pandas()
    for column in ("first_sample_time", "end_anchor_time"):
        if column in daily:
            daily[column] = pd.to_datetime(daily[column], errors="coerce", utc=True)
    if "event_time" in events:
        events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
        events = events.dropna(subset=["event_time"]).reset_index(drop=True)
    return daily, events, run_dir, json.loads(summary_path.read_text(encoding="utf-8"))


def read_latest_recalibration_evidence(
    recalibration_root: Path,
    serial: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path | None, dict[str, Any]]:
    """Read the latest current discharge-first recalibration artifact, when available."""
    summaries = sorted(
        (recalibration_root / f"serial={serial}").glob(
            "recalibration_run=*/model_summary.json"
        ),
        reverse=True,
    )
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("detector_version") or "").lower() != "v4":
            continue
        run_dir = summary_path.parent
        events_path = run_dir / "recalibration_events.parquet"
        capacity_path = run_dir / "one_second_capacity_validation.parquet"
        if not events_path.exists():
            continue
        events = pq.ParquetFile(events_path).read().to_pandas()
        capacity = (
            pq.ParquetFile(capacity_path).read().to_pandas()
            if capacity_path.exists()
            else pd.DataFrame()
        )
        for frame in (events, capacity):
            for name in frame.columns:
                if name.endswith("_utc"):
                    frame[name] = pd.to_datetime(frame[name], errors="coerce", utc=True)
        return events, capacity, run_dir, summary
    return pd.DataFrame(), pd.DataFrame(), None, {}


def _serial_raw_root(raw_root: Path, *, serial: str, chemistry: str) -> Path:
    legacy = raw_root / f"serial={serial}"
    canonical = raw_root / f"chemistry={chemistry}" / f"serial={serial}"
    existing = [path for path in (legacy, canonical) if path.exists()]
    if len(existing) != 1:
        if existing:
            raise ValueError(f"Battery {serial} exists in two raw storage layouts")
        raise FileNotFoundError(f"No raw telemetry exists for serial {serial}")
    return existing[0]


def build_monthly_one_second_current_stress(
    raw_root: Path,
    *,
    serial: str,
    chemistry: str,
    nominal_capacity_ah: float,
    rest_current_threshold_a: float,
    high_current_c_rate_threshold: float,
    maximum_interval_seconds: int = 5,
) -> pd.DataFrame:
    """Aggregate projected raw current into monthly dynamics without Python row loops."""
    serial_root = _serial_raw_root(raw_root, serial=serial, chemistry=chemistry)
    files = [str(path) for path in sorted(serial_root.rglob("*.parquet"))]
    if not files:
        raise FileNotFoundError(f"No raw Parquet files exist for serial {serial}")
    high_current_a = nominal_capacity_ah * high_current_c_rate_threshold
    large_step_a = nominal_capacity_ah * 0.25
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute("SET threads TO 2")
        connection.execute("SET TimeZone = 'UTC'")
        query = """
        WITH projected AS (
            SELECT time, try_cast(batt_current_a AS DOUBLE) AS current_a
            FROM read_parquet(?, union_by_name = true)
            WHERE time IS NOT NULL AND batt_current_a IS NOT NULL
        ), deduplicated AS (
            SELECT time, avg(current_a) AS current_a
            FROM projected
            WHERE isfinite(current_a)
            GROUP BY time
        ), ordered AS (
            SELECT
                time,
                current_a,
                lag(time) OVER (ORDER BY time) AS previous_time,
                lag(current_a) OVER (ORDER BY time) AS previous_current_a,
                avg(abs(current_a)) OVER (
                    ORDER BY time RANGE BETWEEN INTERVAL 9 SECOND PRECEDING AND CURRENT ROW
                ) AS mean_abs_current_10s_a,
                avg(abs(current_a)) OVER (
                    ORDER BY time RANGE BETWEEN INTERVAL 59 SECOND PRECEDING AND CURRENT ROW
                ) AS mean_abs_current_60s_a
            FROM deduplicated
        ), intervals AS (
            SELECT
                *,
                epoch(time - previous_time) AS raw_interval_seconds,
                CASE
                    WHEN epoch(time - previous_time) > 0
                     AND epoch(time - previous_time) <= ?
                    THEN epoch(time - previous_time)
                    ELSE 0
                END AS duration_seconds
            FROM ordered
        ), features AS (
            SELECT
                date_trunc('month', time) AS month_start_utc,
                *,
                abs(current_a) / ? AS absolute_c_rate,
                abs(current_a - previous_current_a) AS current_step_a,
                CASE
                    WHEN abs(current_a) > ? AND abs(previous_current_a) > ?
                     AND sign(current_a) <> sign(previous_current_a)
                    THEN 1 ELSE 0
                END AS reversal,
                CASE
                    WHEN abs(current_a) >= ?
                     AND (
                        coalesce(abs(previous_current_a), 0) < ?
                        OR sign(current_a) <> sign(previous_current_a)
                     )
                    THEN 1 ELSE 0
                END AS high_current_episode_start
            FROM intervals
        )
        SELECT
            month_start_utc,
            count(*) AS raw_current_sample_rows,
            sum(duration_seconds) AS raw_current_observed_seconds,
            sum(CASE WHEN raw_interval_seconds > ? THEN 1 ELSE 0 END) AS raw_current_gap_count,
            sum(greatest(coalesce(raw_interval_seconds, 0) - ?, 0)) AS raw_current_unobserved_seconds,
            sum(greatest(current_a, 0) * duration_seconds) / 3600.0 AS raw_charge_ah,
            sum(greatest(-current_a, 0) * duration_seconds) / 3600.0 AS raw_discharge_ah,
            sqrt(sum(current_a * current_a * duration_seconds) /
                 nullif(sum(duration_seconds), 0)) AS raw_rms_current_a,
            max(current_a) AS raw_peak_charge_current_a,
            max(-current_a) AS raw_peak_discharge_current_a,
            quantile_cont(abs(current_a), 0.95) AS raw_p95_absolute_current_a,
            quantile_cont(abs(current_a), 0.99) AS raw_p99_absolute_current_a,
            max(mean_abs_current_10s_a) AS raw_peak_10s_mean_absolute_current_a,
            max(mean_abs_current_60s_a) AS raw_peak_60s_mean_absolute_current_a,
            quantile_cont(current_step_a, 0.95) AS raw_p95_current_step_a,
            quantile_cont(current_step_a, 0.99) AS raw_p99_current_step_a,
            sum(CASE WHEN current_step_a >= ? THEN 1 ELSE 0 END) AS raw_large_current_step_count,
            sum(reversal) AS raw_current_reversal_count,
            sum(high_current_episode_start) AS raw_high_current_episode_count,
            sum(CASE WHEN absolute_c_rate < 0.01 THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_c0_hours,
            sum(CASE WHEN absolute_c_rate >= 0.01 AND absolute_c_rate < 0.25 THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_c1_hours,
            sum(CASE WHEN absolute_c_rate >= 0.25 AND absolute_c_rate < 0.50 THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_c2_hours,
            sum(CASE WHEN absolute_c_rate >= 0.50 AND absolute_c_rate < 1.00 THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_c3_hours,
            sum(CASE WHEN absolute_c_rate >= 1.00 THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_c4_hours,
            sum(CASE WHEN current_a <= -? THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_high_discharge_hours,
            sum(CASE WHEN current_a >= ? THEN duration_seconds ELSE 0 END) / 3600.0 AS raw_high_charge_hours
        FROM features
        GROUP BY month_start_utc
        ORDER BY month_start_utc
        """
        return connection.execute(
            query,
            [
                files,
                maximum_interval_seconds,
                nominal_capacity_ah,
                rest_current_threshold_a,
                rest_current_threshold_a,
                high_current_a,
                high_current_a,
                maximum_interval_seconds,
                maximum_interval_seconds,
                large_step_a,
                high_current_a,
                high_current_a,
            ],
        ).fetch_df()
    finally:
        connection.close()


def prepare_windows(
    frame: pd.DataFrame,
    *,
    aggregation_minutes: int,
    nominal_capacity_ah: float,
    rest_current_threshold_a: float,
) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = [name for name in result if name != "window_start_utc"]
    for name in numeric_columns:
        if name != "serial":
            result[name] = pd.to_numeric(result[name], errors="coerce")
    result["coverage_fraction"] = result["coverage_fraction"].clip(0.0, 1.0).fillna(0.0)
    result["observed_minutes"] = aggregation_minutes * result["coverage_fraction"]
    result["observed_hours"] = result["observed_minutes"] / 60.0
    current = result["bms_current_mean_a"]
    result["operating_state"] = np.select(
        [current.gt(rest_current_threshold_a), current.lt(-rest_current_threshold_a)],
        ["charging", "discharging"],
        default="rest",
    )
    result["charge_ah"] = current.clip(lower=0) * result["observed_hours"]
    result["discharge_ah"] = (-current.clip(upper=0)) * result["observed_hours"]
    power = pd.to_numeric(result["bms_power_mean_w"], errors="coerce")
    fallback_power = current * result["bms_stack_voltage_mean_v"]
    power = power.where(power.notna(), fallback_power)
    result["charge_wh"] = power.clip(lower=0) * result["observed_hours"]
    result["discharge_wh"] = (-power.clip(upper=0)) * result["observed_hours"]
    result["absolute_c_rate"] = current.abs() / nominal_capacity_ah
    for label, lower, upper in C_RATE_BANDS:
        mask = result["absolute_c_rate"].ge(lower)
        if math.isfinite(upper):
            mask &= result["absolute_c_rate"].lt(upper)
        result[f"{label}_minutes"] = result["observed_minutes"].where(mask, 0.0)
    return result


def _add_joint_exposure_columns(
    frame: pd.DataFrame,
    *,
    high_current_c_rate_threshold: float,
) -> pd.DataFrame:
    """Create duration-weighted, descriptive joint exposure features."""
    result = frame.copy()
    observed = result["observed_minutes"].fillna(0.0)
    c_rate = result["absolute_c_rate"]
    soc = result["bms_soc_mean_pct"]
    temperature = result["stable_pack_temperature_mean_c"]
    spread = result["system_voltage_spread_mv"]
    charging = result["operating_state"].eq("charging")
    discharging = result["operating_state"].eq("discharging")
    high_current = c_rate.ge(high_current_c_rate_threshold)
    masks = {
        "high_discharge_low_soc_minutes": discharging & high_current & soc.lt(10.0),
        "high_charge_high_soc_minutes": charging & high_current & soc.gt(90.0),
        "charging_above_95_soc_minutes": charging & soc.ge(95.0),
        "high_current_hot_minutes": high_current & temperature.ge(30.0),
        "high_charge_cold_minutes": charging & high_current & temperature.lt(10.0),
        "high_current_imbalanced_minutes": high_current & spread.ge(30.0),
        "low_soc_high_discharge_imbalanced_minutes": (
            discharging & high_current & soc.lt(10.0) & spread.ge(30.0)
        ),
    }
    for name, mask in masks.items():
        result[name] = observed.where(mask.fillna(False), 0.0)
    temperature_masks = {
        "temperature_below_10_minutes": temperature.lt(10.0),
        "temperature_10_20_minutes": temperature.ge(10.0) & temperature.lt(20.0),
        "temperature_20_30_minutes": temperature.ge(20.0) & temperature.lt(30.0),
        "temperature_30_40_minutes": temperature.ge(30.0) & temperature.lt(40.0),
        "temperature_above_40_minutes": temperature.ge(40.0),
    }
    for name, mask in temperature_masks.items():
        result[name] = observed.where(mask.fillna(False), 0.0)
    result["temperature_below_15_minutes"] = observed.where(
        temperature.lt(15.0).fillna(False), 0.0
    )
    result["temperature_known_minutes"] = result[list(temperature_masks)].sum(axis=1)
    return result


def _nearby_window_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    if frame.empty:
        return {
            "temperature_mean_c": None,
            "temperature_max_c": None,
            "voltage_spread_p95_mv": None,
            "voltage_spread_max_mv": None,
        }
    return {
        "temperature_mean_c": _weighted_mean(
            frame["stable_pack_temperature_mean_c"], frame["observed_minutes"]
        ),
        "temperature_max_c": _finite(frame["stable_pack_temperature_max_c"].max()),
        "voltage_spread_p95_mv": _quantile(frame["system_voltage_spread_mv"], 0.95),
        "voltage_spread_max_mv": _finite(frame["system_voltage_spread_mv"].max()),
    }


def build_charge_episodes(
    windows: pd.DataFrame,
    *,
    aggregation_minutes: int,
    episode_gap_minutes: int,
    nominal_capacity_ah: float,
    minimum_soc_gain_pct: float,
    context_minutes: int = 30,
) -> pd.DataFrame:
    charging = windows.loc[windows["operating_state"].eq("charging")].copy()
    if charging.empty:
        return pd.DataFrame()
    start_gap = charging["window_start_utc"].diff().dt.total_seconds().div(60)
    interruption = (start_gap - aggregation_minutes).clip(lower=0)
    charging["_episode_group"] = (
        interruption.gt(episode_gap_minutes).fillna(True).cumsum()
    )
    records: list[dict[str, Any]] = []
    for _, group in charging.groupby("_episode_group", sort=True):
        group = group.sort_values("window_start_utc")
        start = group["window_start_utc"].iloc[0]
        end = group["window_start_utc"].iloc[-1] + pd.Timedelta(minutes=aggregation_minutes)
        before = windows.loc[
            windows["window_start_utc"].ge(start - pd.Timedelta(minutes=context_minutes))
            & windows["window_start_utc"].lt(start)
        ]
        after = windows.loc[
            windows["window_start_utc"].ge(end)
            & windows["window_start_utc"].lt(end + pd.Timedelta(minutes=context_minutes))
        ]
        before_metrics = _nearby_window_metrics(before)
        after_metrics = _nearby_window_metrics(after)
        start_soc = _finite(group["bms_soc_mean_pct"].iloc[0])
        end_soc = _finite(group["bms_soc_mean_pct"].iloc[-1])
        soc_gain = end_soc - start_soc if start_soc is not None and end_soc is not None else None
        charged_ah = float(group["charge_ah"].sum())
        maximum_current = _finite(group["bms_current_max_a"].max()) or 0.0
        taper_current = max(0.25 * maximum_current, 0.0)
        taper = group["bms_soc_mean_pct"].ge(95) & group["bms_current_mean_a"].le(taper_current)
        elapsed_minutes = max((end - start).total_seconds() / 60.0, aggregation_minutes)
        partial_capacity = (
            charged_ah / (soc_gain / 100.0)
            if soc_gain is not None and soc_gain >= minimum_soc_gain_pct
            else None
        )
        record = {
            "episode_id": f"charge_{len(records) + 1:06d}",
            "start_time_utc": start,
            "end_time_utc": end,
            "start_day_utc": start.floor("D"),
            "start_hour_utc": float(start.hour + start.minute / 60.0),
            "elapsed_duration_minutes": float(elapsed_minutes),
            "observed_charging_minutes": float(group["observed_minutes"].sum()),
            "episode_coverage_fraction": float(
                min(1.0, group["observed_minutes"].sum() / elapsed_minutes)
            ),
            "charging_window_count": int(len(group)),
            "start_soc_pct": start_soc,
            "end_soc_pct": end_soc,
            "minimum_soc_pct": _finite(group["bms_soc_min_pct"].min()),
            "maximum_soc_pct": _finite(group["bms_soc_max_pct"].max()),
            "soc_gain_pct": soc_gain,
            "charged_ah": charged_ah,
            "charged_wh": float(group["charge_wh"].sum()),
            "mean_charge_current_a": _weighted_mean(
                group["bms_current_mean_a"], group["observed_minutes"]
            ),
            "p95_charge_current_a": _quantile(group["bms_current_mean_a"], 0.95),
            "maximum_charge_current_a": maximum_current,
            "maximum_charge_c_rate": maximum_current / nominal_capacity_ah,
            "reached_90_pct_soc": bool(group["bms_soc_max_pct"].max() >= 90),
            "reached_95_pct_soc": bool(group["bms_soc_max_pct"].max() >= 95),
            "reached_99_pct_soc": bool(group["bms_soc_max_pct"].max() >= 99),
            "reached_100_pct_soc": bool(group["bms_soc_max_pct"].max() >= 100),
            "high_soc_taper_proxy_minutes": float(group.loc[taper, "observed_minutes"].sum()),
            "partial_charge_capacity_estimate_ah": partial_capacity,
            "partial_capacity_estimate_is_weak_label": partial_capacity is not None,
            "temperature_mean_c": _weighted_mean(
                group["stable_pack_temperature_mean_c"], group["observed_minutes"]
            ),
            "temperature_p95_c": _quantile(group["stable_pack_temperature_mean_c"], 0.95),
            "temperature_max_c": _finite(group["stable_pack_temperature_max_c"].max()),
            "voltage_spread_p95_mv": _quantile(group["system_voltage_spread_mv"], 0.95),
            "voltage_spread_max_mv": _finite(group["system_voltage_spread_mv"].max()),
        }
        for prefix, metrics in (("before_30min", before_metrics), ("after_30min", after_metrics)):
            record.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        records.append(record)
    return pd.DataFrame.from_records(records)


def _state_minutes(group: pd.DataFrame, state: str) -> float:
    return float(group.loc[group["operating_state"].eq(state), "observed_minutes"].sum())


def _add_soc_exposure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    soc = pd.to_numeric(result["bms_soc_mean_pct"], errors="coerce")
    observed = pd.to_numeric(result["observed_minutes"], errors="coerce").fillna(0.0)
    plausible = soc.between(0.0, 100.0, inclusive="both")
    masks = {
        "soc_0_5_minutes": plausible & soc.le(5.0),
        "soc_5_20_minutes": plausible & soc.gt(5.0) & soc.le(20.0),
        "soc_20_40_minutes": plausible & soc.gt(20.0) & soc.le(40.0),
        "soc_40_60_minutes": plausible & soc.gt(40.0) & soc.le(60.0),
        "soc_60_80_minutes": plausible & soc.gt(60.0) & soc.le(80.0),
        "soc_80_95_minutes": plausible & soc.gt(80.0) & soc.lt(95.0),
        "soc_95_100_minutes": plausible & soc.ge(95.0),
    }
    for column, mask in masks.items():
        result[column] = observed.where(mask, 0.0)
    result["soc_known_minutes"] = result[list(SOC_EXPOSURE_MINUTE_COLUMNS)].sum(axis=1)
    return result


def build_daily_stress(
    windows: pd.DataFrame,
    episodes: pd.DataFrame,
    audit_events: pd.DataFrame,
    *,
    aggregation_minutes: int,
    nominal_capacity_ah: float,
    high_current_c_rate_threshold: float,
    minimum_daily_coverage: float,
) -> pd.DataFrame:
    frame = _add_joint_exposure_columns(
        _add_soc_exposure_columns(windows),
        high_current_c_rate_threshold=high_current_c_rate_threshold,
    )
    frame["day_utc"] = frame["window_start_utc"].dt.floor("D")
    high_current_a = nominal_capacity_ah * high_current_c_rate_threshold
    episode_days = episodes.copy()
    if not episode_days.empty:
        episode_days["start_day_utc"] = pd.to_datetime(
            episode_days["start_day_utc"], errors="coerce", utc=True
        )
    events = audit_events.copy()
    if not events.empty:
        events["day_utc"] = events["event_time"].dt.floor("D")
    records: list[dict[str, Any]] = []
    for day, group in frame.groupby("day_utc", sort=True):
        observed_minutes = float(group["observed_minutes"].sum())
        daily_episodes = (
            episode_days.loc[episode_days["start_day_utc"].eq(day)]
            if not episode_days.empty
            else episode_days
        )
        daily_events = (
            events.loc[events["day_utc"].eq(day)] if not events.empty else events
        )
        charging = group["operating_state"].eq("charging")
        discharging = group["operating_state"].eq("discharging")
        soh = group["bms_soh_mean_pct"].dropna()
        records.append(
            {
                "day_utc": day,
                "feature_window_count": int(len(group)),
                "expected_window_count": int(round(1440 / aggregation_minutes)),
                "observed_minutes": observed_minutes,
                "telemetry_coverage_fraction": min(1.0, observed_minutes / 1440.0),
                "daily_quality_valid": bool(
                    observed_minutes / 1440.0 >= minimum_daily_coverage
                ),
                "charge_ah": float(group["charge_ah"].sum()),
                "discharge_ah": float(group["discharge_ah"].sum()),
                "charge_wh": float(group["charge_wh"].sum()),
                "discharge_wh": float(group["discharge_wh"].sum()),
                "discharge_equivalent_full_cycles": float(
                    group["discharge_ah"].sum() / nominal_capacity_ah
                ),
                "charging_minutes": _state_minutes(group, "charging"),
                "discharging_minutes": _state_minutes(group, "discharging"),
                "rest_minutes": _state_minutes(group, "rest"),
                "soc_at_or_above_90_minutes": float(
                    group.loc[group["bms_soc_mean_pct"].ge(90), "observed_minutes"].sum()
                ),
                "soc_at_or_above_95_minutes": float(
                    group.loc[group["bms_soc_mean_pct"].ge(95), "observed_minutes"].sum()
                ),
                "soc_at_or_below_10_minutes": float(
                    group.loc[group["bms_soc_mean_pct"].le(10), "observed_minutes"].sum()
                ),
                "soc_at_or_below_5_minutes": float(
                    group.loc[group["bms_soc_mean_pct"].le(5), "observed_minutes"].sum()
                ),
                **{
                    column: float(group[column].sum())
                    for column in SOC_EXPOSURE_MINUTE_COLUMNS
                },
                "soc_known_minutes": float(group["soc_known_minutes"].sum()),
                "soc_unknown_minutes": float(
                    max(0.0, 1440.0 - group["soc_known_minutes"].sum())
                ),
                **{
                    f"{label}_minutes": float(group[f"{label}_minutes"].sum())
                    for label, _, _ in C_RATE_BANDS
                },
                "high_discharge_low_soc_minutes": float(
                    group["high_discharge_low_soc_minutes"].sum()
                ),
                "high_charge_high_soc_minutes": float(
                    group["high_charge_high_soc_minutes"].sum()
                ),
                "charging_above_95_soc_minutes": float(
                    group["charging_above_95_soc_minutes"].sum()
                ),
                "high_current_hot_minutes": float(
                    group["high_current_hot_minutes"].sum()
                ),
                "high_charge_cold_minutes": float(
                    group["high_charge_cold_minutes"].sum()
                ),
                "high_current_imbalanced_minutes": float(
                    group["high_current_imbalanced_minutes"].sum()
                ),
                "low_soc_high_discharge_imbalanced_minutes": float(
                    group["low_soc_high_discharge_imbalanced_minutes"].sum()
                ),
                "temperature_below_10_minutes": float(
                    group["temperature_below_10_minutes"].sum()
                ),
                "temperature_below_15_minutes": float(
                    group["temperature_below_15_minutes"].sum()
                ),
                "temperature_10_20_minutes": float(group["temperature_10_20_minutes"].sum()),
                "temperature_20_30_minutes": float(group["temperature_20_30_minutes"].sum()),
                "temperature_30_40_minutes": float(group["temperature_30_40_minutes"].sum()),
                "temperature_above_40_minutes": float(
                    group["temperature_above_40_minutes"].sum()
                ),
                "temperature_known_minutes": float(group["temperature_known_minutes"].sum()),
                "high_current_charging_minutes": float(
                    group.loc[charging & group["bms_current_mean_a"].ge(high_current_a), "observed_minutes"].sum()
                ),
                "high_current_discharging_minutes": float(
                    group.loc[discharging & group["bms_current_mean_a"].le(-high_current_a), "observed_minutes"].sum()
                ),
                "charge_current_p95_a": _quantile(
                    group.loc[charging, "bms_current_mean_a"], 0.95
                ),
                "discharge_current_p95_abs_a": _quantile(
                    -group.loc[discharging, "bms_current_mean_a"], 0.95
                ),
                "temperature_mean_c": _weighted_mean(
                    group["stable_pack_temperature_mean_c"], group["observed_minutes"]
                ),
                "temperature_p95_c": _quantile(
                    group["stable_pack_temperature_mean_c"], 0.95
                ),
                "temperature_max_c": _finite(group["stable_pack_temperature_max_c"].max()),
                "temperature_at_or_above_35_minutes": float(
                    group.loc[group["stable_pack_temperature_mean_c"].ge(35), "observed_minutes"].sum()
                ),
                "temperature_at_or_above_40_minutes": float(
                    group.loc[group["stable_pack_temperature_mean_c"].ge(40), "observed_minutes"].sum()
                ),
                "temperature_spread_p95_c": _quantile(
                    group["system_temperature_spread_c"], 0.95
                ),
                "temperature_spread_max_c": _finite(group["system_temperature_spread_c"].max()),
                "voltage_spread_p95_mv": _quantile(group["system_voltage_spread_mv"], 0.95),
                "voltage_spread_max_mv": _finite(group["system_voltage_spread_mv"].max()),
                "charging_voltage_spread_p95_mv": _quantile(
                    group.loc[charging, "system_voltage_spread_mv"], 0.95
                ),
                "discharging_voltage_spread_p95_mv": _quantile(
                    group.loc[discharging, "system_voltage_spread_mv"], 0.95
                ),
                "rest_voltage_spread_p95_mv": _quantile(
                    group.loc[~charging & ~discharging, "system_voltage_spread_mv"], 0.95
                ),
                "bms_soh_first_pct": _finite(soh.iloc[0]) if not soh.empty else None,
                "bms_soh_last_pct": _finite(soh.iloc[-1]) if not soh.empty else None,
                "bms_soh_mean_pct": _finite(soh.mean()) if not soh.empty else None,
                "charge_episode_count": int(len(daily_episodes)),
                "near_full_charge_episode_count": int(
                    daily_episodes.get("reached_99_pct_soc", pd.Series(dtype=bool)).sum()
                ),
                "soh_step_drop_count": int(
                    daily_events.get("event_type", pd.Series(dtype=str)).eq("step_drop").sum()
                ),
                "soh_step_recovery_count": int(
                    daily_events.get("event_type", pd.Series(dtype=str)).eq("step_recovery").sum()
                ),
                "soh_change_after_gap_count": int(
                    daily_events.get("event_type", pd.Series(dtype=str)).eq("change_after_gap").sum()
                ),
                "audited_soh_change_pp": float(
                    pd.to_numeric(
                        daily_events.get("soh_delta_pct", pd.Series(dtype=float)),
                        errors="coerce",
                    ).sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _charge_period(hour: float) -> str:
    if hour < 6:
        return "night_00_06_utc"
    if hour < 12:
        return "morning_06_12_utc"
    if hour < 18:
        return "afternoon_12_18_utc"
    return "evening_18_24_utc"


def build_monthly_profiles(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    minimum_observed_days: int,
    minimum_month_coverage: float,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy().sort_values("day_utc")
    frame["month_start_utc"] = pd.to_datetime(
        frame["day_utc"].dt.strftime("%Y-%m-01"), utc=True
    )
    episode_frame = episodes.copy()
    if not episode_frame.empty:
        episode_frame["start_time_utc"] = pd.to_datetime(
            episode_frame["start_time_utc"], errors="coerce", utc=True
        )
        episode_frame["month_start_utc"] = pd.to_datetime(
            episode_frame["start_time_utc"].dt.strftime("%Y-%m-01"), utc=True
        )
        episode_frame["charge_start_period_utc"] = episode_frame["start_hour_utc"].map(_charge_period)
    scope_start = frame["day_utc"].min()
    scope_end = frame["day_utc"].max()
    records: list[dict[str, Any]] = []
    for month_start, group in frame.groupby("month_start_utc", sort=True):
        year, month = int(month_start.year), int(month_start.month)
        calendar_days = calendar.monthrange(year, month)[1]
        month_end = month_start + pd.offsets.MonthBegin(1)
        scoped_start = max(scope_start, month_start)
        scoped_end = min(scope_end, month_end - pd.Timedelta(days=1))
        expected_days = int((scoped_end.floor("D") - scoped_start.floor("D")).days + 1)
        observed_days = int(group["day_utc"].nunique())
        qualified_days = int(group["daily_quality_valid"].sum())
        coverage = min(1.0, float(group["observed_minutes"].sum()) / (expected_days * 1440.0))
        month_episodes = (
            episode_frame.loc[episode_frame["month_start_utc"].eq(month_start)]
            if not episode_frame.empty
            else episode_frame
        )
        period_counts = (
            month_episodes["charge_start_period_utc"].value_counts().to_dict()
            if not month_episodes.empty
            else {}
        )
        valid_soh = group.dropna(subset=["bms_soh_last_pct"])
        angle = 2 * math.pi * (month - 1) / 12.0
        expected_minutes = float(expected_days * 1440)
        known_soc_minutes = float(group["soc_known_minutes"].sum())
        records.append(
            {
                "month_start_utc": month_start,
                "calendar_year": year,
                "calendar_month": month,
                "season": _season(month),
                "month_sin": math.sin(angle),
                "month_cos": math.cos(angle),
                "calendar_days_in_month": calendar_days,
                "expected_days_in_observed_scope": expected_days,
                "observed_days": observed_days,
                "qualified_days": qualified_days,
                "telemetry_coverage_fraction": coverage,
                "seasonal_profile_valid": bool(
                    observed_days >= minimum_observed_days
                    and coverage >= minimum_month_coverage
                ),
                "charge_ah": float(group["charge_ah"].sum()),
                "discharge_ah": float(group["discharge_ah"].sum()),
                "charge_ah_per_observed_day": float(group["charge_ah"].sum() / observed_days),
                "discharge_ah_per_observed_day": float(group["discharge_ah"].sum() / observed_days),
                "charge_wh": float(group["charge_wh"].sum()),
                "discharge_wh": float(group["discharge_wh"].sum()),
                "discharge_equivalent_full_cycles": float(
                    group["discharge_equivalent_full_cycles"].sum()
                ),
                "charging_hours": float(group["charging_minutes"].sum() / 60.0),
                "discharging_hours": float(group["discharging_minutes"].sum() / 60.0),
                "rest_hours": float(group["rest_minutes"].sum() / 60.0),
                "high_soc_90_hours": float(group["soc_at_or_above_90_minutes"].sum() / 60.0),
                "high_soc_95_hours": float(group["soc_at_or_above_95_minutes"].sum() / 60.0),
                "low_soc_10_hours": float(group["soc_at_or_below_10_minutes"].sum() / 60.0),
                "low_soc_5_hours": float(group["soc_at_or_below_5_minutes"].sum() / 60.0),
                **{
                    column.removesuffix("_minutes") + "_hours": float(
                        group[column].sum() / 60.0
                    )
                    for column in SOC_EXPOSURE_MINUTE_COLUMNS
                },
                "soc_known_hours": known_soc_minutes / 60.0,
                "soc_unknown_hours": max(0.0, expected_minutes - known_soc_minutes) / 60.0,
                "soc_known_coverage_fraction": min(
                    1.0, known_soc_minutes / expected_minutes
                ),
                **{
                    f"{label}_hours": float(group[f"{label}_minutes"].sum() / 60.0)
                    for label, _, _ in C_RATE_BANDS
                },
                "high_discharge_low_soc_hours": float(
                    group["high_discharge_low_soc_minutes"].sum() / 60.0
                ),
                "high_charge_high_soc_hours": float(
                    group["high_charge_high_soc_minutes"].sum() / 60.0
                ),
                "charging_above_95_soc_hours": float(
                    group["charging_above_95_soc_minutes"].sum() / 60.0
                ),
                "high_current_hot_hours": float(
                    group["high_current_hot_minutes"].sum() / 60.0
                ),
                "high_charge_cold_hours": float(
                    group["high_charge_cold_minutes"].sum() / 60.0
                ),
                "high_current_imbalanced_hours": float(
                    group["high_current_imbalanced_minutes"].sum() / 60.0
                ),
                "low_soc_high_discharge_imbalanced_hours": float(
                    group["low_soc_high_discharge_imbalanced_minutes"].sum() / 60.0
                ),
                "temperature_below_10_hours": float(
                    group["temperature_below_10_minutes"].sum() / 60.0
                ),
                "temperature_below_15_hours": float(
                    group["temperature_below_15_minutes"].sum() / 60.0
                ),
                "temperature_10_20_hours": float(group["temperature_10_20_minutes"].sum() / 60.0),
                "temperature_20_30_hours": float(group["temperature_20_30_minutes"].sum() / 60.0),
                "temperature_30_40_hours": float(group["temperature_30_40_minutes"].sum() / 60.0),
                "temperature_above_40_hours": float(
                    group["temperature_above_40_minutes"].sum() / 60.0
                ),
                "high_current_charging_hours": float(
                    group["high_current_charging_minutes"].sum() / 60.0
                ),
                "high_current_discharging_hours": float(
                    group["high_current_discharging_minutes"].sum() / 60.0
                ),
                "temperature_mean_c": _weighted_mean(
                    group["temperature_mean_c"], group["observed_minutes"]
                ),
                "temperature_p95_daily_c": _quantile(group["temperature_p95_c"], 0.95),
                "temperature_max_c": _finite(group["temperature_max_c"].max()),
                "hot_35_hours": float(group["temperature_at_or_above_35_minutes"].sum() / 60.0),
                "hot_40_hours": float(group["temperature_at_or_above_40_minutes"].sum() / 60.0),
                "voltage_spread_p95_daily_mv": _quantile(group["voltage_spread_p95_mv"], 0.95),
                "voltage_spread_max_mv": _finite(group["voltage_spread_max_mv"].max()),
                "charge_episode_count": int(len(month_episodes)),
                "charge_episodes_per_observed_day": float(len(month_episodes) / observed_days),
                "near_full_charge_episode_count": int(
                    month_episodes.get("reached_99_pct_soc", pd.Series(dtype=bool)).sum()
                ),
                "median_charge_start_hour_utc": _finite(
                    month_episodes.get("start_hour_utc", pd.Series(dtype=float)).median()
                ),
                "charge_starts_night_00_06_utc": int(period_counts.get("night_00_06_utc", 0)),
                "charge_starts_morning_06_12_utc": int(period_counts.get("morning_06_12_utc", 0)),
                "charge_starts_afternoon_12_18_utc": int(period_counts.get("afternoon_12_18_utc", 0)),
                "charge_starts_evening_18_24_utc": int(period_counts.get("evening_18_24_utc", 0)),
                "bms_soh_first_pct": (
                    _finite(valid_soh["bms_soh_first_pct"].iloc[0]) if not valid_soh.empty else None
                ),
                "bms_soh_last_pct": (
                    _finite(valid_soh["bms_soh_last_pct"].iloc[-1]) if not valid_soh.empty else None
                ),
                "audited_soh_change_pp": float(group["audited_soh_change_pp"].sum()),
                "soh_step_drop_count": int(group["soh_step_drop_count"].sum()),
                "soh_step_recovery_count": int(group["soh_step_recovery_count"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def _period_context(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "observed_days": 0,
            "mean_coverage_fraction": None,
            "charge_ah": 0.0,
            "discharge_ah": 0.0,
            "charge_episode_count": 0,
            "near_full_charge_episode_count": 0,
            "high_soc_95_hours": 0.0,
            "low_soc_5_hours": 0.0,
            "hot_35_hours": 0.0,
            "high_current_charge_hours": 0.0,
            "high_current_discharge_hours": 0.0,
            "temperature_mean_c": None,
            "temperature_max_c": None,
            "voltage_spread_p95_mv": None,
            "voltage_spread_max_mv": None,
        }
    return {
        "observed_days": int(frame["day_utc"].nunique()),
        "mean_coverage_fraction": _finite(frame["telemetry_coverage_fraction"].mean()),
        "charge_ah": float(frame["charge_ah"].sum()),
        "discharge_ah": float(frame["discharge_ah"].sum()),
        "charge_episode_count": int(frame["charge_episode_count"].sum()),
        "near_full_charge_episode_count": int(frame["near_full_charge_episode_count"].sum()),
        "high_soc_95_hours": float(frame["soc_at_or_above_95_minutes"].sum() / 60.0),
        "low_soc_5_hours": float(frame["soc_at_or_below_5_minutes"].sum() / 60.0),
        "hot_35_hours": float(frame["temperature_at_or_above_35_minutes"].sum() / 60.0),
        "high_current_charge_hours": float(frame["high_current_charging_minutes"].sum() / 60.0),
        "high_current_discharge_hours": float(frame["high_current_discharging_minutes"].sum() / 60.0),
        "temperature_mean_c": _weighted_mean(frame["temperature_mean_c"], frame["observed_minutes"]),
        "temperature_max_c": _finite(frame["temperature_max_c"].max()),
        "voltage_spread_p95_mv": _quantile(frame["voltage_spread_p95_mv"], 0.95),
        "voltage_spread_max_mv": _finite(frame["voltage_spread_max_mv"].max()),
    }


def build_transition_context(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: int,
    minimum_daily_coverage: float,
    recalibration_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    episode_frame = episodes.copy()
    if not episode_frame.empty:
        episode_frame["start_time_utc"] = pd.to_datetime(
            episode_frame["start_time_utc"], errors="coerce", utc=True
        )
        episode_frame["end_time_utc"] = pd.to_datetime(
            episode_frame["end_time_utc"], errors="coerce", utc=True
        )
    records: list[dict[str, Any]] = []
    authoritative = (
        recalibration_events.copy()
        if recalibration_events is not None and not recalibration_events.empty
        else pd.DataFrame()
    )
    if not authoritative.empty:
        authoritative["event_time_utc"] = pd.to_datetime(
            authoritative["event_time_utc"], errors="coerce", utc=True
        )
        authoritative["daily_event_anchor_utc"] = pd.to_datetime(
            authoritative.get("daily_event_anchor_utc"), errors="coerce", utc=True
        )
    for index, event in events.sort_values("event_time").reset_index(drop=True).iterrows():
        event_time = pd.Timestamp(event["event_time"])
        event_day = event_time.floor("D")
        pre = daily.loc[
            daily["day_utc"].ge(event_day - pd.Timedelta(days=window_days))
            & daily["day_utc"].lt(event_day)
        ]
        post = daily.loc[
            daily["day_utc"].ge(event_day)
            & daily["day_utc"].lt(event_day + pd.Timedelta(days=window_days))
        ]
        pre_context = _period_context(pre)
        post_context = _period_context(post)
        recent_episodes = (
            episode_frame.loc[
                episode_frame["end_time_utc"].le(event_time)
                & episode_frame["end_time_utc"].ge(event_time - pd.Timedelta(hours=72))
            ]
            if not episode_frame.empty
            else episode_frame
        )
        near_full = (
            recent_episodes.loc[recent_episodes["reached_99_pct_soc"]]
            if not recent_episodes.empty
            else recent_episodes
        )
        last_near_full_end = (
            near_full["end_time_utc"].max() if not near_full.empty else pd.NaT
        )
        event_during_charge = bool(
            (
                episode_frame["start_time_utc"].le(event_time)
                & episode_frame["end_time_utc"].ge(event_time)
            ).any()
        ) if not episode_frame.empty else False
        matched = pd.Series(dtype=object)
        if not authoritative.empty:
            candidates = authoritative.loc[
                authoritative["daily_event_anchor_utc"].dt.floor("D").eq(event_day)
            ].copy()
            for name, audit_name in (
                ("previous_soh_pct", "previous_soh_pct"),
                ("new_soh_pct", "new_soh_pct"),
            ):
                expected = _finite(event.get(audit_name))
                if expected is not None and name in candidates:
                    candidates = candidates.loc[
                        pd.to_numeric(candidates[name], errors="coerce").sub(expected).abs().le(0.01)
                    ]
            if not candidates.empty:
                candidates["_distance"] = (
                    candidates["event_time_utc"] - event_time
                ).abs().dt.total_seconds()
                matched = candidates.sort_values("_distance").iloc[0]
        required_days = min(window_days, 5)
        context_ready = bool(
            pre_context["observed_days"] >= required_days
            and post_context["observed_days"] >= required_days
            and (pre_context["mean_coverage_fraction"] or 0) >= minimum_daily_coverage
            and (post_context["mean_coverage_fraction"] or 0) >= minimum_daily_coverage
        )
        record: dict[str, Any] = {
            "transition_id": f"soh_transition_{index + 1:05d}",
            "event_time_utc": event_time,
            "event_day_utc": event_day,
            "event_type": str(event.get("event_type") or "unknown"),
            "previous_soh_pct": _finite(event.get("previous_soh_pct")),
            "new_soh_pct": _finite(event.get("new_soh_pct")),
            "soh_delta_pp": _finite(event.get("soh_delta_pct")),
            "audit_coverage_fraction": _finite(event.get("coverage_fraction")),
            "audit_cumulative_discharge_ah": _finite(event.get("cumulative_all_discharge_ah")),
            "audit_max_gap_seconds": _finite(event.get("max_gap_seconds")),
            "context_window_days": window_days,
            "context_quality_ready": context_ready,
            "event_during_charge_episode": event_during_charge,
            "near_full_charge_within_72h_before": bool(not near_full.empty),
            "hours_since_near_full_charge": (
                float((event_time - last_near_full_end).total_seconds() / 3600.0)
                if pd.notna(last_near_full_end)
                else None
            ),
            "interpretation": "temporal_association_only_not_causal",
            "authoritative_recalibration_match": bool(not matched.empty),
        }
        if not matched.empty:
            evidence_columns = (
                "event_id",
                "transition_detection_status",
                "event_time_resolution",
                "event_time_uncertainty_seconds",
                "complete_discharge_within_lookback",
                "hours_since_complete_discharge",
                "complete_discharge_start_time_utc",
                "complete_discharge_end_time_utc",
                "complete_discharge_start_soc_pct",
                "complete_discharge_end_soc_pct",
                "complete_discharge_soc_drop_pct",
                "complete_discharge_duration_minutes",
                "complete_discharge_coverage_fraction",
                "complete_discharge_dominance_fraction",
                "raw_discharge_detection_status",
                "raw_discharge_ambiguous",
                "raw_start_time_utc",
                "raw_end_time_utc",
                "raw_start_soc_pct",
                "raw_end_soc_pct",
                "raw_soc_drop_pct",
                "rest_minutes_inside_cycle",
                "longest_rest_minutes",
                "rest_interruption_count",
                "discharged_ah_1s",
                "charge_intrusion_ah_1s",
                "raw_coverage_fraction",
                "raw_discharge_dominance_fraction",
                "raw_discharge_qualified",
                "maximum_raw_gap_seconds",
                "capacity_validation_status",
                "cycle_continuity_class",
                "direct_capacity_soh_pct",
                "soc_normalized_capacity_ah",
                "soc_normalized_capacity_soh_pct",
                "bms_completion_minus_capacity_soh_pp",
                "bms_after_event_minus_capacity_soh_pp",
                "primary_context_pattern",
                "context_match_strength",
                "evidence_tags",
            )
            record.update(
                {
                    f"recalibration_{name}": matched.get(name)
                    for name in evidence_columns
                }
            )
        for prefix, context in (("pre", pre_context), ("post", post_context)):
            record.update({f"{prefix}_{key}": value for key, value in context.items()})
        records.append(record)
    return pd.DataFrame.from_records(records)


def _write_html_report(
    path: Path,
    summary: dict[str, Any],
    monthly: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    month_rows = []
    for row in monthly.tail(24).itertuples(index=False):
        month_rows.append(
            "<tr>"
            f"<td>{html.escape(pd.Timestamp(row.month_start_utc).strftime('%Y-%m'))}</td>"
            f"<td>{html.escape(str(row.season))}</td>"
            f"<td>{row.telemetry_coverage_fraction:.1%}</td>"
            f"<td>{row.charge_ah_per_observed_day:.1f}</td>"
            f"<td>{row.discharge_ah_per_observed_day:.1f}</td>"
            f"<td>{row.high_soc_95_hours:.1f}</td>"
            f"<td>{_display_number(row.temperature_mean_c)}</td>"
            f"<td>{_display_number(row.voltage_spread_p95_daily_mv)}</td>"
            f"<td>{'ready' if row.seasonal_profile_valid else 'limited'}</td>"
            "</tr>"
        )
    transition_rows = []
    for row in transitions.tail(50).itertuples(index=False):
        transition_rows.append(
            "<tr>"
            f"<td>{html.escape(pd.Timestamp(row.event_time_utc).isoformat())}</td>"
            f"<td>{html.escape(str(row.event_type))}</td>"
            f"<td>{_display_number(row.soh_delta_pp)}</td>"
            f"<td>{row.pre_charge_ah:.1f}</td>"
            f"<td>{row.pre_discharge_ah:.1f}</td>"
            f"<td>{_display_number(row.pre_temperature_max_c)}</td>"
            f"<td>{_display_number(row.pre_voltage_spread_max_mv)}</td>"
            f"<td>{'ready' if row.context_quality_ready else 'limited'}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Battery {html.escape(str(summary['serial']))} seasonal charging and stress</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f7;color:#26343d;font:14px Arial,sans-serif}}
main{{max-width:1320px;margin:auto;padding:30px}}h1{{margin:4px 0 8px}}h2{{font-size:18px}}
.eyebrow{{color:#176b87;font-weight:700;text-transform:uppercase;font-size:12px}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;background:#d9e0e3;margin:24px 0;border:1px solid #d9e0e3}}
.metric,section{{background:white;padding:18px}}.metric small{{display:block;color:#68767d;margin-bottom:7px}}.metric strong{{font-size:21px}}
section{{border:1px solid #d9e0e3;margin:18px 0}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid #e4e8ea;text-align:left;white-space:nowrap}}th{{font-size:12px;color:#637078}}
.notice{{border-left:4px solid #d99018;background:#fff8eb;padding:12px;line-height:1.5}}
@media(max-width:850px){{main{{padding:16px}}.metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<span class="eyebrow">BatteryML compact evidence report</span><h1>Battery {html.escape(str(summary['serial']))}</h1>
<div>{html.escape(str(summary['chemistry']))} | {summary['module_count']} modules | all times UTC</div>
<div class="metrics">
<div class="metric"><small>Charge episodes</small><strong>{summary['charge_episode_rows']}</strong></div>
<div class="metric"><small>Daily profiles</small><strong>{summary['daily_stress_rows']}</strong></div>
<div class="metric"><small>Qualified months</small><strong>{summary['valid_monthly_profile_rows']}</strong></div>
<div class="metric"><small>SOH transitions</small><strong>{summary['soh_transition_rows']}</strong></div>
<div class="metric"><small>Discharge EFC</small><strong>{summary['total_discharge_equivalent_full_cycles']:.1f}</strong></div>
</div>
<section><h2>Monthly charging and seasonal exposure</h2><div class="table-wrap"><table><thead><tr><th>Month UTC</th><th>Season</th><th>Coverage</th><th>Charge Ah/day</th><th>Discharge Ah/day</th><th>SOC >=95 h</th><th>Mean temp C</th><th>Daily spread p95 mV</th><th>Quality</th></tr></thead><tbody>{''.join(month_rows) or '<tr><td colspan="9">No monthly profile</td></tr>'}</tbody></table></div></section>
<section><h2>BMS SOH transition context</h2><div class="table-wrap"><table><thead><tr><th>Event UTC</th><th>Type</th><th>SOH change pp</th><th>Pre charge Ah</th><th>Pre discharge Ah</th><th>Pre max temp C</th><th>Pre max spread mV</th><th>Context</th></tr></thead><tbody>{''.join(transition_rows) or '<tr><td colspan="8">No audited SOH transition</td></tr>'}</tbody></table></div></section>
<p class="notice"><strong>Claim limit:</strong> {html.escape(str(summary['claim_limit']))}</p>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def run_analysis(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
    feature_root: Path,
    soh_audit_root: Path,
    recalibration_root: Path,
    raw_root: Path,
    output_root: Path,
    rest_current_threshold_a: float = 0.5,
    high_current_c_rate_threshold: float = 0.5,
    charge_episode_gap_minutes: int = 15,
    minimum_charge_soc_gain_pct: float = 20.0,
    transition_window_days: int = 7,
    minimum_daily_coverage: float = 0.8,
    minimum_month_observed_days: int = 14,
    minimum_month_coverage: float = 0.8,
    compression: str | None = "zstd",
    run_id: str | None = None,
) -> Path:
    chemistry = chemistry.upper().strip()
    if chemistry not in {"NCA", "LFP"}:
        raise ValueError("chemistry must be NCA or LFP")
    if module_count < 1:
        raise ValueError("module_count must be positive")
    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive")
    if rest_current_threshold_a < 0 or high_current_c_rate_threshold <= 0:
        raise ValueError("current thresholds must be positive or zero as documented")
    if charge_episode_gap_minutes < 1 or transition_window_days < 1:
        raise ValueError("episode gap and transition window must be positive")
    if not 0 <= minimum_daily_coverage <= 1 or not 0 <= minimum_month_coverage <= 1:
        raise ValueError("coverage thresholds must be between 0 and 1")

    compact, feature_summary = read_compact_history(feature_root, serial)
    feature_module_count = int(feature_summary.get("module_count") or 0)
    if feature_module_count != module_count:
        raise ValueError(
            f"Configured module count {module_count} does not match compact history "
            f"module count {feature_module_count}"
        )
    aggregation = int(feature_summary["aggregation_minutes"])
    windows = prepare_windows(
        compact,
        aggregation_minutes=aggregation,
        nominal_capacity_ah=nominal_capacity_ah,
        rest_current_threshold_a=rest_current_threshold_a,
    )
    audit_daily, audit_events, audit_run, audit_summary = read_latest_soh_audit(
        soh_audit_root, serial
    )
    recalibration_events, capacity_validation, recalibration_run, recalibration_summary = (
        read_latest_recalibration_evidence(recalibration_root, serial)
    )
    if not recalibration_events.empty and not capacity_validation.empty:
        capacity_columns = [
            name
            for name in (
                "event_id",
                "capacity_validation_status",
                "cycle_continuity_class",
                "direct_capacity_soh_pct",
                "soc_normalized_capacity_ah",
                "soc_normalized_capacity_soh_pct",
                "bms_completion_minus_capacity_soh_pp",
                "bms_after_event_minus_capacity_soh_pp",
            )
            if name in capacity_validation
        ]
        capacity_by_event = (
            capacity_validation.loc[:, capacity_columns]
            .dropna(subset=["event_id"])
            .sort_values("event_id")
            .drop_duplicates("event_id", keep="last")
        )
        recalibration_events = recalibration_events.merge(
            capacity_by_event,
            on="event_id",
            how="left",
            suffixes=("", "_capacity"),
        )
    episodes = build_charge_episodes(
        windows,
        aggregation_minutes=aggregation,
        episode_gap_minutes=charge_episode_gap_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_soc_gain_pct=minimum_charge_soc_gain_pct,
    )
    daily = build_daily_stress(
        windows,
        episodes,
        audit_events,
        aggregation_minutes=aggregation,
        nominal_capacity_ah=nominal_capacity_ah,
        high_current_c_rate_threshold=high_current_c_rate_threshold,
        minimum_daily_coverage=minimum_daily_coverage,
    )
    monthly = build_monthly_profiles(
        daily,
        episodes,
        minimum_observed_days=minimum_month_observed_days,
        minimum_month_coverage=minimum_month_coverage,
    )
    raw_current_monthly = build_monthly_one_second_current_stress(
        raw_root,
        serial=serial,
        chemistry=chemistry,
        nominal_capacity_ah=nominal_capacity_ah,
        rest_current_threshold_a=rest_current_threshold_a,
        high_current_c_rate_threshold=high_current_c_rate_threshold,
    )
    if not raw_current_monthly.empty:
        raw_current_monthly["month_start_utc"] = pd.to_datetime(
            raw_current_monthly["month_start_utc"], errors="coerce", utc=True
        )
        monthly = monthly.merge(raw_current_monthly, on="month_start_utc", how="left")
    transitions = build_transition_context(
        daily,
        episodes,
        audit_events,
        window_days=transition_window_days,
        minimum_daily_coverage=minimum_daily_coverage,
        recalibration_events=recalibration_events,
    )

    run_stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"serial={serial}" / f"profile_run={run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_parquet_chunk(episodes, run_dir / "charge_episodes.parquet", compression=compression)
    write_parquet_chunk(daily, run_dir / "daily_battery_stress.parquet", compression=compression)
    write_parquet_chunk(monthly, run_dir / "monthly_operating_profile.parquet", compression=compression)
    write_parquet_chunk(
        raw_current_monthly,
        run_dir / "monthly_one_second_current_stress.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        transitions, run_dir / "soh_transition_context.parquet", compression=compression
    )
    valid_months = int(monthly.get("seasonal_profile_valid", pd.Series(dtype=bool)).sum())
    latest_soh = _finite(daily["bms_soh_last_pct"].dropna().iloc[-1]) if daily["bms_soh_last_pct"].notna().any() else None
    summary = {
        "model_family": MODEL_FAMILY,
        "analysis_version": ANALYSIS_VERSION,
        "serial": str(serial),
        "chemistry": chemistry,
        "module_count": module_count,
        "nominal_capacity_ah": nominal_capacity_ah,
        "timestamp_semantics": "UTC",
        "observed_start": windows["window_start_utc"].min(),
        "observed_end": windows["window_start_utc"].max() + pd.Timedelta(minutes=aggregation),
        "aggregation_minutes": aggregation,
        "soc_exposure_method": "duration_weighted_mean_bms_soc_by_compact_window",
        "soc_exposure_bands_pct": ["0-5", "5-20", "20-40", "40-60", "60-80", "80-95", "95-100"],
        "soc_unknown_definition": "expected scoped time without a plausible duration-weighted BMS SOC observation",
        "feature_rows": len(windows),
        "charge_episode_rows": len(episodes),
        "daily_stress_rows": len(daily),
        "monthly_profile_rows": len(monthly),
        "valid_monthly_profile_rows": valid_months,
        "soh_transition_rows": len(transitions),
        "authoritative_recalibration_match_rows": int(
            transitions.get(
                "authoritative_recalibration_match", pd.Series(dtype=bool)
            ).sum()
        ),
        "raw_current_monthly_rows": len(raw_current_monthly),
        "context_ready_transition_rows": int(
            transitions.get("context_quality_ready", pd.Series(dtype=bool)).sum()
        ),
        "latest_bms_soh_pct": latest_soh,
        "total_charge_ah": float(daily["charge_ah"].sum()),
        "total_discharge_ah": float(daily["discharge_ah"].sum()),
        "total_discharge_equivalent_full_cycles": float(
            daily["discharge_equivalent_full_cycles"].sum()
        ),
        "total_high_soc_95_hours": float(daily["soc_at_or_above_95_minutes"].sum() / 60.0),
        "maximum_stable_pack_temperature_c": _finite(daily["temperature_max_c"].max()),
        "source_feature_run_ids": feature_summary.get("compatible_feature_run_ids", []),
        "source_soh_audit_run_id": audit_run.name,
        "source_recalibration_run_id": (
            recalibration_run.name if recalibration_run is not None else None
        ),
        "source_recalibration_detector_version": recalibration_summary.get(
            "detector_version"
        ),
        "source_soh_audit_label_status": (audit_summary.get("audit_metrics") or {}).get(
            "label_status"
        ),
        "source_soh_audit_daily_rows": len(audit_daily),
        "stable_temperature_sensor_count": feature_summary.get(
            "stable_temperature_sensor_count"
        ),
        "configuration": {
            "rest_current_threshold_a": rest_current_threshold_a,
            "high_current_c_rate_threshold": high_current_c_rate_threshold,
            "c_rate_bands": {
                label: [lower, None if not math.isfinite(upper) else upper]
                for label, lower, upper in C_RATE_BANDS
            },
            "charge_episode_gap_minutes": charge_episode_gap_minutes,
            "minimum_charge_soc_gain_pct": minimum_charge_soc_gain_pct,
            "transition_window_days": transition_window_days,
            "minimum_daily_coverage": minimum_daily_coverage,
            "minimum_month_observed_days": minimum_month_observed_days,
            "minimum_month_coverage": minimum_month_coverage,
        },
        "value_semantics": {
            "source": "BMS-reported database telemetry",
            "charge_sign": "positive pack current",
            "discharge_sign": "negative pack current",
            "temperature": "mean and maximum across stable pack sensors 00 and 01",
            "partial_capacity": "weak proxy from charge Ah divided by observed SOC gain",
            "season": "meteorological season assigned from UTC calendar month",
            "c_rate_bands": (
                "neutral analytical bins, not manufacturer stress or safety limits"
            ),
            "soh_event_interpretation": (
                "discharge-first recalibration evidence plus longitudinal exposure history"
            ),
        },
        "claim_limit": (
            "Profiles describe temporal associations in BMS-reported telemetry. BMS SOH "
            "steps are weak labels, partial-charge capacity is a proxy, and neither is an "
            "independent physical capacity measurement or a causal aging attribution."
        ),
    }
    manifest = {
        "manifest_version": "1.0.0",
        "created_at": datetime.now(UTC),
        "serial": str(serial),
        "chemistry": chemistry,
        "module_count": module_count,
        "source_feature_run_ids": summary["source_feature_run_ids"],
        "source_soh_audit_run_id": audit_run.name,
        "source_recalibration_run_id": summary["source_recalibration_run_id"],
        "artifacts": {
            "charge_episodes.parquet": "individual observed charging intervals",
            "daily_battery_stress.parquet": "daily exposure and operating-state features",
            "monthly_operating_profile.parquet": "coverage-gated seasonal charging profiles",
            "monthly_one_second_current_stress.parquet": "streamed monthly raw-current pulse and C-rate metrics",
            "soh_transition_context.parquet": "discharge-first recalibration evidence plus pre/post degradation context",
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
    _write_html_report(run_dir / "seasonal_charge_report.html", safe_summary, monthly, transitions)
    logger.info("Seasonal charging and stress analysis complete: %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build charging episodes, daily stress, seasonal profiles, and SOH-step context."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--nominal-capacity-ah", required=True, type=float)
    parser.add_argument("--rest-current-threshold-a", type=float, default=0.5)
    parser.add_argument("--high-current-c-rate-threshold", type=float, default=0.5)
    parser.add_argument("--charge-episode-gap-minutes", type=int, default=15)
    parser.add_argument("--minimum-charge-soc-gain-pct", type=float, default=20.0)
    parser.add_argument("--transition-window-days", type=int, default=7)
    parser.add_argument("--minimum-daily-coverage", type=float, default=0.8)
    parser.add_argument("--minimum-month-observed-days", type=int, default=14)
    parser.add_argument("--minimum-month-coverage", type=float, default=0.8)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--soh-audit-root", type=Path)
    parser.add_argument("--recalibration-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
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
        compression = None if args.compression == "none" else args.compression
        run_dir = run_analysis(
            serial=str(args.serial),
            chemistry=args.chemistry,
            module_count=args.module_count,
            nominal_capacity_ah=args.nominal_capacity_ah,
            feature_root=args.feature_root
            or storage.data_dir / "processed" / "bms_hierarchical_features",
            soh_audit_root=args.soh_audit_root
            or storage.data_dir / "processed" / "fleet_soh_audit",
            recalibration_root=args.recalibration_root
            or storage.data_dir / "processed" / "bms_soh_recalibration",
            raw_root=args.raw_root or storage.raw_parquet_dir,
            output_root=args.output_root
            or storage.data_dir / "processed" / "seasonal_charge_analysis",
            rest_current_threshold_a=args.rest_current_threshold_a,
            high_current_c_rate_threshold=args.high_current_c_rate_threshold,
            charge_episode_gap_minutes=args.charge_episode_gap_minutes,
            minimum_charge_soc_gain_pct=args.minimum_charge_soc_gain_pct,
            transition_window_days=args.transition_window_days,
            minimum_daily_coverage=args.minimum_daily_coverage,
            minimum_month_observed_days=args.minimum_month_observed_days,
            minimum_month_coverage=args.minimum_month_coverage,
            compression=compression,
        )
        print(run_dir)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Seasonal charging analysis failed")
        else:
            logger.error("Seasonal charging analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
