from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.bms_soh_recalibration_analysis import (
    build_calibration_discharge_episodes,
    build_full_soc_capacity_candidates,
    build_one_second_capacity_validation,
    read_raw_anchor_windows,
)
from src.config import StorageConfig
from src.io_utils import write_parquet_chunk
from src.seasonal_charge_analysis import (
    _compatible_feature_runs,
    _finite,
    _json_safe,
    prepare_windows,
    read_compact_history,
)

logger = logging.getLogger(__name__)

MODEL_FAMILY = "discharge_derived_usable_capacity_evidence"
ANALYSIS_VERSION = "v2"
VOLTAGE_MEAN_COLUMN = re.compile(
    r"^module_(\d{2})_pack_(\d{2})_cell_voltage_channel_(\d{2})_mean_mv$"
)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _episode_context(
    validations: pd.DataFrame,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Add bounded five-minute temperature and voltage-spread context."""
    if validations.empty:
        return validations.copy()
    records: list[dict[str, Any]] = []
    for row in validations.to_dict("records"):
        start = pd.to_datetime(row.get("raw_start_time_utc"), errors="coerce", utc=True)
        end = pd.to_datetime(row.get("raw_end_time_utc"), errors="coerce", utc=True)
        if pd.isna(start) or pd.isna(end):
            records.append(row)
            continue
        frame = windows.loc[windows["window_start_utc"].between(start, end)]
        temperatures = pd.to_numeric(
            frame.get("stable_pack_temperature_mean_c"), errors="coerce"
        )
        maximum_temperatures = pd.to_numeric(
            frame.get("stable_pack_temperature_max_c"), errors="coerce"
        )
        spreads = pd.to_numeric(frame.get("system_voltage_spread_mv"), errors="coerce")
        observed_minutes = pd.to_numeric(
            frame.get("observed_minutes", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        temperature_known = temperatures.notna()
        row.update(
            {
                "pack_temperature_mean_c": _finite(temperatures.mean()),
                "pack_temperature_median_c": _finite(temperatures.median()),
                "pack_temperature_min_c": _finite(temperatures.min()),
                "pack_temperature_p95_c": _finite(temperatures.quantile(0.95)),
                "pack_temperature_max_c": _finite(maximum_temperatures.max()),
                "pack_temperature_below_10_minutes": float(
                    observed_minutes.loc[temperature_known & temperatures.lt(10.0)].sum()
                ),
                "pack_temperature_below_15_minutes": float(
                    observed_minutes.loc[temperature_known & temperatures.lt(15.0)].sum()
                ),
                "pack_temperature_at_or_above_30_minutes": float(
                    observed_minutes.loc[temperature_known & temperatures.ge(30.0)].sum()
                ),
                "pack_temperature_at_or_above_35_minutes": float(
                    observed_minutes.loc[temperature_known & temperatures.ge(35.0)].sum()
                ),
                "pack_temperature_at_or_above_40_minutes": float(
                    observed_minutes.loc[temperature_known & temperatures.ge(40.0)].sum()
                ),
                "pack_temperature_context_coverage_fraction": (
                    float(observed_minutes.loc[temperature_known].sum() / observed_minutes.sum())
                    if observed_minutes.sum() > 0
                    else None
                ),
                "cell_voltage_spread_p95_mv": _finite(spreads.quantile(0.95)),
                "cell_voltage_spread_max_mv": _finite(spreads.max()),
            }
        )
        records.append(row)
    return pd.DataFrame.from_records(records)


def _voltage_feature_paths(feature_root: Path, serial: str) -> list[Path]:
    run_dirs, summary = _compatible_feature_runs(feature_root, serial)
    aggregation = int(summary["aggregation_minutes"])
    return [
        path
        for run_dir in run_dirs
        for path in sorted(
            run_dir.glob(
                f"year=*/month=*/voltage_channel_features_{aggregation}min.parquet"
            )
        )
    ]


def build_weakest_channel_evidence(
    feature_root: Path,
    serial: str,
    validations: pd.DataFrame,
    *,
    endpoint_minutes: int = 15,
) -> pd.DataFrame:
    """Identify recurring low-voltage BMS channels near qualified discharge endpoints."""
    resolved = validations.loc[
        validations.get(
            "raw_end_time_utc", pd.Series(pd.NaT, index=validations.index)
        ).notna()
    ].copy()
    if resolved.empty:
        return pd.DataFrame()
    paths = _voltage_feature_paths(feature_root, serial)
    endpoint_times = pd.to_datetime(resolved["raw_end_time_utc"], errors="coerce", utc=True)
    intervals = [
        (time - pd.Timedelta(minutes=endpoint_minutes), time + pd.Timedelta(minutes=endpoint_minutes))
        for time in endpoint_times.dropna()
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        voltage_columns = [name for name in names if VOLTAGE_MEAN_COLUMN.match(name)]
        if "window_start_utc" not in names or not voltage_columns:
            continue
        frame = parquet.read(columns=("window_start_utc", *voltage_columns)).to_pandas()
        frame["window_start_utc"] = pd.to_datetime(
            frame["window_start_utc"], errors="coerce", utc=True
        )
        frame = frame.sort_values("window_start_utc").reset_index(drop=True)
        time_values = frame["window_start_utc"].to_numpy(
            dtype="datetime64[ns]"
        ).astype("int64")
        keep = np.zeros(len(frame), dtype=bool)
        for start, end in intervals:
            left = int(np.searchsorted(time_values, start.value, side="left"))
            right = int(np.searchsorted(time_values, end.value, side="right"))
            keep[left:right] = True
        if keep.any():
            frames.append(frame.loc[keep])
    if not frames:
        return pd.DataFrame()
    voltage = (
        pd.concat(frames, ignore_index=True)
        .sort_values("window_start_utc")
        .drop_duplicates("window_start_utc", keep="last")
    )
    voltage_time_values = voltage["window_start_utc"].to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")
    channel_columns = [
        name for name in voltage.columns if VOLTAGE_MEAN_COLUMN.match(name)
    ]
    records: list[dict[str, Any]] = []
    for row in resolved.to_dict("records"):
        end = pd.Timestamp(row["raw_end_time_utc"])
        start_ns = (end - pd.Timedelta(minutes=endpoint_minutes)).value
        end_ns = (end + pd.Timedelta(minutes=endpoint_minutes)).value
        left = int(np.searchsorted(voltage_time_values, start_ns, side="left"))
        right = int(np.searchsorted(voltage_time_values, end_ns, side="right"))
        endpoint = voltage.iloc[left:right]
        channel_means = endpoint[channel_columns].mean(axis=0, skipna=True).dropna()
        if channel_means.empty:
            continue
        weakest_name = str(channel_means.idxmin())
        match = VOLTAGE_MEAN_COLUMN.match(weakest_name)
        assert match is not None
        records.append(
            {
                "discharge_episode_id": row.get("discharge_episode_id"),
                "endpoint_time_utc": end,
                "module_index": int(match.group(1)),
                "pack_index": int(match.group(2)),
                "channel_index": int(match.group(3)),
                "channel_identity": weakest_name.removesuffix("_mean_mv"),
                "weakest_channel_mean_mv": float(channel_means.min()),
                "pack_channel_median_mv": float(channel_means.median()),
                "weakest_channel_deficit_to_median_mv": float(
                    channel_means.median() - channel_means.min()
                ),
                "endpoint_window_rows": int(len(endpoint)),
                "evidence_semantics": (
                    "Lowest BMS-reported channel mean near the discharge endpoint; "
                    "not an independently verified failed cell."
                ),
            }
        )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        recurrence = result["channel_identity"].value_counts()
        result["weakest_channel_episode_count"] = result["channel_identity"].map(recurrence)
        result["weakest_channel_episode_fraction"] = (
            result["weakest_channel_episode_count"] / len(result)
        )
    return result


def run_analysis(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
    raw_root: Path,
    feature_root: Path,
    output_root: Path,
    rest_current_threshold_a: float = 0.5,
    minimum_window_coverage: float = 0.8,
    maximum_elapsed_hours: int = 72,
    minimum_discharge_dominance_fraction: float = 0.8,
    minimum_raw_coverage_fraction: float = 0.98,
    maximum_unobserved_ah: float = 0.25,
    maximum_charge_intrusion_ah: float = 0.1,
    maximum_integrated_gap_seconds: int = 5,
    continuous_maximum_rest_minutes: float = 5.0,
    compression: str | None = "zstd",
    run_id: str | None = None,
) -> Path:
    chemistry = chemistry.upper().strip()
    if chemistry not in {"NCA", "LFP"}:
        raise ValueError("chemistry must be NCA or LFP")
    if module_count < 1 or nominal_capacity_ah <= 0:
        raise ValueError("module_count and nominal_capacity_ah must be positive")
    compact, feature_summary = read_compact_history(feature_root, serial)
    logger.info("Loaded %d compact five-minute windows", len(compact))
    if int(feature_summary.get("module_count") or 0) != module_count:
        raise ValueError("Configured module count does not match compact feature history")
    aggregation_minutes = int(feature_summary["aggregation_minutes"])
    windows = prepare_windows(
        compact,
        aggregation_minutes=aggregation_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        rest_current_threshold_a=rest_current_threshold_a,
    )
    operational = build_calibration_discharge_episodes(
        windows,
        aggregation_minutes=aggregation_minutes,
        episode_gap_minutes=30,
        nominal_capacity_ah=nominal_capacity_ah,
        minimum_window_coverage=minimum_window_coverage,
        high_soc_start_pct=95.0,
        low_soc_end_pct=5.0,
        minimum_soc_drop_pct=90.0,
        minimum_discharge_dominance_fraction=minimum_discharge_dominance_fraction,
    )
    if not operational.empty:
        operational["candidate_source"] = "compact_operational_95_to_5"
    strict = build_full_soc_capacity_candidates(
        windows,
        aggregation_minutes=aggregation_minutes,
        nominal_capacity_ah=nominal_capacity_ah,
        start_soc_pct=99.0,
        end_soc_pct=1.0,
        maximum_elapsed_hours=maximum_elapsed_hours,
        minimum_window_coverage=minimum_window_coverage,
    )
    candidates = pd.concat((operational, strict), ignore_index=True, sort=False)
    if not candidates.empty:
        candidates = (
            candidates.sort_values(["start_time_utc", "completion_time_utc"])
            .drop_duplicates(["start_time_utc", "completion_time_utc"], keep="last")
            .reset_index(drop=True)
        )
    logger.info(
        "Detected %d compact discharge candidates (%d operational, %d strict)",
        len(candidates),
        len(operational),
        len(strict),
    )
    qualified_compact = candidates.loc[
        candidates.get(
            "complete_high_to_low_discharge", pd.Series(False, index=candidates.index)
        ).fillna(False)
    ]
    anchors = pd.to_datetime(
        qualified_compact.get("completion_time_utc", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
        utc=True,
    ) + pd.Timedelta(minutes=aggregation_minutes)
    starts = pd.to_datetime(
        qualified_compact.get("start_time_utc", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
        utc=True,
    ) - pd.Timedelta(minutes=aggregation_minutes)
    raw, scan = read_raw_anchor_windows(
        raw_root,
        serial=serial,
        chemistry=chemistry,
        anchors=anchors,
        lookback_hours=maximum_elapsed_hours + 1,
        starts=starts,
    )
    logger.info(
        "Read %d bounded raw rows from %d row groups across %d merged intervals",
        scan["raw_rows_retained"],
        scan["raw_row_groups_read"],
        scan["candidate_intervals"],
    )
    validations = build_one_second_capacity_validation(
        raw,
        candidates,
        pd.DataFrame(columns=("complete_discharge_within_lookback",)),
        nominal_capacity_ah=nominal_capacity_ah,
        aggregation_minutes=aggregation_minutes,
        candidate_start_soc_pct=95.0,
        candidate_end_soc_pct=5.0,
        strict_start_soc_pct=99.0,
        strict_end_soc_pct=1.0,
        strict_minimum_soc_drop_pct=98.0,
        minimum_coverage_fraction=minimum_raw_coverage_fraction,
        maximum_unobserved_ah=maximum_unobserved_ah,
        maximum_charge_intrusion_ah=maximum_charge_intrusion_ah,
        maximum_integrated_gap_seconds=maximum_integrated_gap_seconds,
        rest_current_threshold_a=rest_current_threshold_a,
        continuous_maximum_rest_minutes=continuous_maximum_rest_minutes,
    )
    logger.info("Integrated %d raw discharge episodes", len(validations))
    validations = _episode_context(validations, windows)
    weakest = build_weakest_channel_evidence(feature_root, serial, validations)
    logger.info("Resolved weakest-channel evidence for %d episodes", len(weakest))
    if not weakest.empty:
        validations = validations.merge(
            weakest.drop(columns=["endpoint_time_utc"]),
            on="discharge_episode_id",
            how="left",
            validate="one_to_one",
        )
    accepted_statuses = {"validation_grade", "exploratory_soc_boundaries"}
    accepted = validations.loc[
        validations.get("capacity_validation_status", pd.Series(dtype=str)).isin(accepted_statuses)
    ].copy()
    rejected = validations.loc[~validations.index.isin(accepted.index)].copy()
    run_stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"serial={serial}" / f"capacity_evidence_run={run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_parquet_chunk(candidates, run_dir / "discharge_candidates.parquet", compression=compression)
    write_parquet_chunk(validations, run_dir / "discharge_episodes.parquet", compression=compression)
    write_parquet_chunk(accepted, run_dir / "qualified_capacity_points.parquet", compression=compression)
    write_parquet_chunk(rejected, run_dir / "rejected_candidates.parquet", compression=compression)
    write_parquet_chunk(weakest, run_dir / "weakest_channel_evidence.parquet", compression=compression)
    strict_points = accepted.loc[accepted["capacity_validation_status"].eq("validation_grade")]
    operational_points = accepted.loc[
        accepted["capacity_validation_status"].eq("exploratory_soc_boundaries")
    ]
    summary = {
        "model_family": MODEL_FAMILY,
        "analysis_version": ANALYSIS_VERSION,
        "serial": str(serial),
        "chemistry": chemistry,
        "module_count": module_count,
        "nominal_capacity_ah": nominal_capacity_ah,
        "timestamp_semantics": "UTC",
        "current_sign_convention": "positive=charging; negative=discharging",
        "capacity_claim": "discharge-derived usable-capacity/SOH proxy",
        "candidate_rows": int(len(candidates)),
        "resolved_episode_rows": int(validations["raw_end_time_utc"].notna().sum()) if not validations.empty else 0,
        "validation_grade_99_to_1_rows": int(len(strict_points)),
        "strict_full_100_to_0_rows": int(
            (
                pd.to_numeric(strict_points.get("raw_start_soc_pct"), errors="coerce").ge(100.0)
                & pd.to_numeric(strict_points.get("raw_end_soc_pct"), errors="coerce").le(0.0)
            ).sum()
        ),
        "operational_95_to_5_rows": int(len(operational_points)),
        "rejected_rows": int(len(rejected)),
        "median_validation_grade_capacity_ah": _finite(strict_points.get("discharged_ah_1s", pd.Series(dtype=float)).median()),
        "median_operational_normalized_capacity_ah": _finite(operational_points.get("soc_normalized_capacity_ah", pd.Series(dtype=float)).median()),
        "source_feature_run_ids": feature_summary.get("compatible_feature_run_ids", []),
        "raw_scan": scan,
        "configuration": {
            "operational_soc_boundaries_pct": [95.0, 5.0],
            "validation_grade_soc_boundaries_pct": [99.0, 1.0],
            "minimum_window_coverage": minimum_window_coverage,
            "minimum_raw_coverage_fraction": minimum_raw_coverage_fraction,
            "maximum_unobserved_ah": maximum_unobserved_ah,
            "maximum_charge_intrusion_ah": maximum_charge_intrusion_ah,
            "maximum_integrated_gap_seconds": maximum_integrated_gap_seconds,
            "continuous_maximum_rest_minutes": continuous_maximum_rest_minutes,
        },
        "temperature_semantics": (
            "BMS pack temperature sensors 00/01 aggregated over the discharge; "
            "not measured cell-core temperature."
        ),
        "claim_limit": (
            "Current integration and SOC boundaries are BMS-derived. These points can assess "
            "BMS SOH consistency and usable-capacity trend, but are not laboratory capacity ground truth."
        ),
    }
    manifest = {
        "manifest_version": "1.0.0",
        "created_at": datetime.now(UTC),
        "serial": str(serial),
        "chemistry": chemistry,
        "analysis_version": ANALYSIS_VERSION,
        "source_feature_run_ids": feature_summary.get("compatible_feature_run_ids", []),
        "artifacts": [
            "discharge_candidates.parquet",
            "discharge_episodes.parquet",
            "qualified_capacity_points.parquet",
            "rejected_candidates.parquet",
            "weakest_channel_evidence.parquet",
        ],
    }
    (run_dir / "model_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8"
    )
    logger.info("Discharge-capacity evidence complete: %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build canonical discharge-derived usable-capacity evidence."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--nominal-capacity-ah", required=True, type=float)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    try:
        storage = StorageConfig.from_env(args.env_file)
        run_analysis(
            serial=str(args.serial),
            chemistry=str(args.chemistry),
            module_count=args.module_count,
            nominal_capacity_ah=args.nominal_capacity_ah,
            raw_root=args.raw_root or storage.raw_parquet_dir,
            feature_root=args.feature_root
            or storage.data_dir / "processed" / "bms_hierarchical_features",
            output_root=args.output_root
            or storage.data_dir / "processed" / "discharge_capacity_evidence",
            compression=None if args.compression == "none" else args.compression,
        )
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Discharge-capacity evidence failed")
        else:
            logger.error("Discharge-capacity evidence failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
