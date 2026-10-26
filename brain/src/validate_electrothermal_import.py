from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from src.config import StorageConfig
from src.electrothermal_contract import expected_hierarchical_signals

MINIMUM_USABLE_CHANNEL_COVERAGE = 0.5
MAXIMUM_INVALID_BUCKET_FRACTION = 0.01


def _files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return files


def _quoted_paths(files: list[Path]) -> list[str]:
    return [str(path.resolve()).replace("\\", "/") for path in files]


def _validate_group(
    *, root: Path, serial: str, expected_columns: list[str], lower: float, upper: float
) -> dict[str, object]:
    files = _files(root)
    required_statistics = ("mean", "min", "max", "std", "count")
    required = {
        "time",
        "serial",
        *(f"{column}__{statistic}" for column in expected_columns for statistic in required_statistics),
    }
    schema_missing: set[str] = set()
    for path in files:
        schema_missing.update(required - set(pq.read_schema(path).names))
    if schema_missing:
        raise ValueError(f"Missing columns under {root}: {', '.join(sorted(schema_missing))}")

    valid_conditions = {
        column: (
            f'"{column}__count" > 0 AND "{column}__min" >= {lower} '
            f'AND "{column}__max" <= {upper}'
        )
        for column in expected_columns
    }
    coverage_terms = [
        f'avg(CASE WHEN {valid_conditions[column]} '
        f'THEN least("{column}__count"::DOUBLE / 300.0, 1.0) ELSE 0.0 END)'
        for column in expected_columns
    ]
    invalid_fraction_terms = [
        f'avg(CASE WHEN "{column}__count" > 0 AND NOT ({valid_conditions[column]}) '
        "THEN 1.0 ELSE 0.0 END)"
        for column in expected_columns
    ]
    minimum_terms = [f'min("{column}__min")' for column in expected_columns]
    maximum_terms = [f'max("{column}__max")' for column in expected_columns]
    connection = duckdb.connect()
    try:
        base = "read_parquet(?, union_by_name=true, hive_partitioning=false)"
        row = connection.execute(
            f"""
            SELECT count(*) AS rows,
                   count(DISTINCT time) AS distinct_times,
                   min(time)::VARCHAR AS start_time,
                   max(time)::VARCHAR AS end_time,
                   count(*) FILTER (WHERE serial::VARCHAR <> ?) AS wrong_serial_rows,
                   count(*) FILTER (WHERE epoch(time)::BIGINT % 300 <> 0) AS misaligned_rows
            FROM {base}
            """,
            [serial, _quoted_paths(files)],
        ).fetchone()
        metrics = connection.execute(
            f"SELECT {', '.join([*coverage_terms, *invalid_fraction_terms, *minimum_terms, *maximum_terms])} FROM {base}",
            [_quoted_paths(files)],
        ).fetchone()
    finally:
        connection.close()

    rows, distinct, start, end, wrong_serial, misaligned = row
    channel_count = len(expected_columns)
    coverages = metrics[:channel_count]
    invalid_fractions = metrics[channel_count : 2 * channel_count]
    minima = [value for value in metrics[2 * channel_count : 3 * channel_count] if value is not None]
    maxima = [value for value in metrics[3 * channel_count :] if value is not None]
    minimum_value = min(minima) if minima else None
    maximum_value = max(maxima) if maxima else None
    minimum_coverage = min(float(value or 0.0) for value in coverages)
    maximum_invalid_fraction = max(float(value or 0.0) for value in invalid_fractions)
    plausible = maximum_invalid_fraction <= MAXIMUM_INVALID_BUCKET_FRACTION
    return {
        "root": str(root),
        "files": len(files),
        "rows": int(rows),
        "start_utc": str(start) if start else None,
        "end_utc": str(end) if end else None,
        "expected_channels": len(expected_columns),
        "minimum_channel_coverage": minimum_coverage,
        "maximum_invalid_bucket_fraction": maximum_invalid_fraction,
        "duplicate_timestamps": int(rows - distinct),
        "wrong_serial_rows": int(wrong_serial),
        "misaligned_five_minute_rows": int(misaligned),
        "minimum_value": float(minimum_value) if minimum_value is not None else None,
        "maximum_value": float(maximum_value) if maximum_value is not None else None,
        "plausible_range_passed": plausible,
        "ready": bool(
            rows > 0
            and rows == distinct
            and not wrong_serial
            and not misaligned
            and minimum_coverage >= MINIMUM_USABLE_CHANNEL_COVERAGE
            and plausible
        ),
    }


def validate_import(*, data_root: Path, serial: str, chemistry: str, module_count: int) -> Path:
    signals = expected_hierarchical_signals(module_count)
    voltage = [s.source_signal_name for s in signals if s.measurement_type == "cell_voltage_channel"]
    temperature = [s.source_signal_name for s in signals if s.measurement_type == "pack_temperature_sensor"]
    custom = data_root / "electrothermal_telemetry"
    partition = Path(f"chemistry={chemistry}") / f"serial={serial}"
    groups = {
        "cell_voltage": _validate_group(
            root=custom / "cell_voltage" / "5m" / partition,
            serial=serial,
            expected_columns=voltage,
            lower=1500.0,
            upper=5000.0,
        ),
        "temperature": _validate_group(
            root=custom / "temperature" / "5m" / partition,
            serial=serial,
            expected_columns=temperature,
            lower=-40.0,
            upper=100.0,
        ),
    }
    ready = all(bool(group["ready"]) for group in groups.values())
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        data_root
        / "processed"
        / "electrothermal_import_validation"
        / f"chemistry={chemistry}"
        / f"serial={serial}"
        / f"validation_run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "model_family": "electrothermal_import_validation",
        "validator_version": "v1",
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "validated_at": datetime.now(UTC).isoformat(),
        "resolution": "5m",
        "groups": groups,
        "ready": ready,
        "claim_limit": "This validates local structure and telemetry quality; it does not diagnose cell health.",
    }
    (run_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not ready:
        raise ValueError(f"Electrothermal import validation failed; evidence written to {run_dir}")
    print(run_dir, flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate local five-minute electrothermal imports.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", choices=("NCA", "LFP"), required=True)
    parser.add_argument("--module-count", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.module_count <= 4:
        raise ValueError("module_count must be between 1 and 4")
    storage = StorageConfig.from_env()
    validate_import(
        data_root=args.data_root or storage.data_dir,
        serial=args.serial,
        chemistry=args.chemistry,
        module_count=args.module_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
