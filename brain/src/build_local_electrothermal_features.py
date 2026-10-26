from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.bms_hierarchical_features import (
    FEATURE_VERSION,
    feature_column,
    split_feature_tables,
)
from src.config import StorageConfig
from src.electrothermal_contract import (
    contract_sha256,
    expected_hierarchical_signals,
)
from src.io_utils import iter_month_ranges, write_parquet_chunk


def _files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return files


def _paths(files: list[Path]) -> list[str]:
    return [str(path.resolve()).replace("\\", "/") for path in files]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest_validation(data_root: Path, chemistry: str, serial: str, module_count: int) -> tuple[Path, dict]:
    root = (
        data_root
        / "processed"
        / "electrothermal_import_validation"
        / f"chemistry={chemistry}"
        / f"serial={serial}"
    )
    summaries = sorted(root.glob("validation_run=*/model_summary.json"))
    if not summaries:
        raise ValueError("Validate imported temperature and voltage data first")
    path = summaries[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("ready"):
        raise ValueError("Latest electrothermal validation did not pass")
    if int(payload.get("module_count") or 0) != module_count:
        raise ValueError("Validated module count does not match the requested build")
    return path.parent, payload


def _raw_files(data_root: Path, chemistry: str, serial: str) -> list[Path]:
    canonical = data_root / "raw_parquet" / f"chemistry={chemistry}" / f"serial={serial}"
    legacy = data_root / "raw_parquet" / f"serial={serial}"
    canonical_files = sorted(canonical.rglob("*.parquet")) if canonical.exists() else []
    legacy_files = sorted(legacy.rglob("*.parquet")) if legacy.exists() else []
    if canonical_files and legacy_files:
        raise ValueError(
            f"Battery {serial} has both canonical and legacy raw layouts; resolve the duplicate layout first"
        )
    files = canonical_files or legacy_files
    if not files:
        raise FileNotFoundError(f"No one-second capacity telemetry found for {serial}")
    return files


def _read_group(
    connection: duckdb.DuckDBPyConnection,
    files: list[Path],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    return connection.execute(
        """
        SELECT *
        FROM read_parquet(?, union_by_name=true, hive_partitioning=false)
        WHERE time >= ? AND time < ?
        ORDER BY time
        """,
        [_paths(files), start, end],
    ).fetch_df()


def _read_context(
    connection: duckdb.DuckDBPyConnection,
    files: list[Path],
    serial: str,
    module_count: int,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    return connection.execute(
        """
        SELECT time_bucket(INTERVAL '5 minutes', time) AS window_start_utc,
               any_value(serial)::VARCHAR AS serial,
               min(time) AS first_sample_time,
               max(time) AS last_sample_time,
               count(*) AS sample_count,
               count(DISTINCT time) AS distinct_timestamp_count,
               avg(batt_soc_pct) AS bms_soc_mean_pct,
               min(batt_soc_pct) AS bms_soc_min_pct,
               max(batt_soc_pct) AS bms_soc_max_pct,
               avg(batt_soh_pct) AS bms_soh_mean_pct,
               avg(batt_current_a) AS bms_current_mean_a,
               min(batt_current_a) AS bms_current_min_a,
               max(batt_current_a) AS bms_current_max_a,
               max(abs(batt_current_a)) AS bms_current_max_abs_a,
               avg(batt_voltage_v) AS bms_stack_voltage_mean_v,
               avg(batt_current_a * batt_voltage_v) AS bms_power_mean_w,
               ?::INTEGER AS bms_module_count_mode
        FROM read_parquet(?, union_by_name=true, hive_partitioning=false)
        WHERE serial::VARCHAR = ? AND time >= ? AND time < ?
        GROUP BY window_start_utc
        ORDER BY window_start_utc
        """,
        [module_count, _paths(files), serial, start, end],
    ).fetch_df()


def _normalized_group(frame: pd.DataFrame, signals) -> pd.DataFrame:
    rename: dict[str, str] = {"time": "window_start_utc"}
    for signal in signals:
        for source_stat, feature_stat in (
            ("mean", "mean"),
            ("min", "min"),
            ("max", "max"),
            ("std", "std"),
            ("count", "valid_count"),
        ):
            rename[f"{signal.source_signal_name}__{source_stat}"] = feature_column(
                signal, feature_stat
            )
    required = ["time", "serial", *rename.keys() - {"time"}]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError("Electrothermal import is missing feature statistics: " + ", ".join(missing))
    normalized = frame[required].rename(columns=rename)
    normalized["serial"] = normalized["serial"].astype(str)
    for signal in signals:
        lower, upper = (
            (1500.0, 5000.0)
            if signal.measurement_type == "cell_voltage_channel"
            else (-40.0, 100.0)
        )
        minimum = feature_column(signal, "min")
        maximum = feature_column(signal, "max")
        valid_count = feature_column(signal, "valid_count")
        valid = (
            normalized[valid_count].gt(0)
            & normalized[minimum].ge(lower)
            & normalized[maximum].le(upper)
        )
        for statistic in ("mean", "min", "max", "std"):
            normalized.loc[~valid, feature_column(signal, statistic)] = np.nan
        normalized.loc[~valid, valid_count] = 0
    return normalized


def build_features(
    *, data_root: Path, serial: str, chemistry: str, module_count: int, compression: str
) -> Path:
    validation_run, validation = _latest_validation(data_root, chemistry, serial, module_count)
    signals = expected_hierarchical_signals(module_count)
    voltage_signals = [item for item in signals if item.measurement_type == "cell_voltage_channel"]
    temperature_signals = [item for item in signals if item.measurement_type == "pack_temperature_sensor"]
    partition = Path(f"chemistry={chemistry}") / f"serial={serial}"
    voltage_files = _files(data_root / "electrothermal_telemetry" / "cell_voltage" / "5m" / partition)
    temperature_files = _files(data_root / "electrothermal_telemetry" / "temperature" / "5m" / partition)
    raw_files = _raw_files(data_root, chemistry, serial)

    starts = [pd.Timestamp(validation["groups"][group]["start_utc"]) for group in ("cell_voltage", "temperature")]
    ends = [pd.Timestamp(validation["groups"][group]["end_utc"]) for group in ("cell_voltage", "temperature")]
    start = max(starts).to_pydatetime()
    end = (min(ends) + pd.Timedelta(minutes=5)).to_pydatetime()
    if start >= end:
        raise ValueError("Validated temperature and voltage datasets have no overlapping range")

    run_stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = data_root / "processed" / "bms_hierarchical_features" / f"serial={serial}" / f"feature_run={run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    codec = None if compression == "none" else compression
    write_parquet_chunk(pd.DataFrame(item.to_dict() for item in signals), run_dir / "channel_map.parquet", compression=codec)

    partitions: list[dict] = []
    total_rows = 0
    with duckdb.connect() as connection:
        for month_start, month_end in iter_month_ranges(start, end):
            voltage = _normalized_group(
                _read_group(connection, voltage_files, month_start, month_end), voltage_signals
            )
            temperature = _normalized_group(
                _read_group(connection, temperature_files, month_start, month_end), temperature_signals
            )
            if voltage.empty or temperature.empty:
                continue
            context = _read_context(
                connection, raw_files, serial, module_count, month_start, month_end
            )
            frame = voltage.merge(temperature, on=["window_start_utc", "serial"], how="inner")
            frame = frame.merge(context, on=["window_start_utc", "serial"], how="left")
            if frame.empty:
                continue
            for column in (
                "bms_reported_voltage_spread_mean_mv",
                "bms_reported_voltage_spread_max_mv",
                "bms_reported_temperature_spread_mean_c",
                "bms_reported_temperature_spread_max_c",
            ):
                frame[column] = np.nan
            for column in ("first_sample_time", "last_sample_time"):
                frame[column] = pd.to_datetime(frame[column], utc=True)
            context_out, voltage_out, temperature_out, system_out = split_feature_tables(
                frame, signals, aggregation_minutes=5
            )
            partition_dir = run_dir / f"year={month_start.year:04d}" / f"month={month_start.month:02d}"
            outputs = {
                "operating_context_5min.parquet": context_out,
                "voltage_channel_features_5min.parquet": voltage_out,
                "temperature_sensor_features_5min.parquet": temperature_out,
                "system_features_5min.parquet": system_out,
            }
            files: list[dict] = []
            for name, output in outputs.items():
                path = partition_dir / name
                write_parquet_chunk(output, path, compression=codec)
                files.append({"name": str(path.relative_to(run_dir)), "rows": len(output), "sha256": _sha256(path)})
            total_rows += len(frame)
            partitions.append({"start": month_start.isoformat(), "end": month_end.isoformat(), "feature_rows": len(frame), "files": files})

    if not partitions:
        raise ValueError("No aligned electrothermal and one-second operating data were found")
    provenance = {
        "source": "local_chemistry_partitioned_parquet",
        "validation_run": validation_run.name,
        "temperature_root": str((data_root / "electrothermal_telemetry" / "temperature" / "5m" / partition).resolve()),
        "cell_voltage_root": str((data_root / "electrothermal_telemetry" / "cell_voltage" / "5m" / partition).resolve()),
        "operating_context_root": str((data_root / "raw_parquet").resolve()),
        "events_required": False,
    }
    manifest = {
        "manifest_version": "2.0.0",
        "feature_version": FEATURE_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "aggregation_minutes": 5,
        "hierarchical_contract_sha256": contract_sha256(signals),
        "partitions": partitions,
        "provenance": provenance,
    }
    summary = {
        "model_family": "bms_hierarchical_electrothermal_features",
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "aggregation_minutes": 5,
        "feature_rows": total_rows,
        "voltage_channel_count": len(voltage_signals),
        "temperature_sensor_count": len(temperature_signals),
        "source_quality_gate": "passed_local_validation",
        "validation_run": validation_run.name,
        "events_required": False,
        "claim_limit": "Features reproduce imported BMS telemetry; physical sensor accuracy requires independent calibration evidence.",
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local five-minute electrothermal features.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", choices=("NCA", "LFP"), required=True)
    parser.add_argument("--module-count", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--compression", default="zstd", choices=("snappy", "zstd", "gzip", "none"))
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.module_count <= 4:
        raise ValueError("module_count must be between 1 and 4")
    storage = StorageConfig.from_env()
    run_dir = build_features(
        data_root=args.data_root or storage.data_dir,
        serial=str(args.serial),
        chemistry=str(args.chemistry),
        module_count=args.module_count,
        compression=args.compression,
    )
    print(run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
