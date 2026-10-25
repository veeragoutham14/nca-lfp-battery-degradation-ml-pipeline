from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.config import StorageConfig
from src.electrothermal_contract import expected_hierarchical_signals
from src.io_utils import write_parquet_chunk

VALIDATOR_VERSION = "data_validation_v1_1"
CORE_SIGNALS = (
    ("soc", "batt_soc_pct", "SOC", "%", "capacity_bms,monthly_exposure"),
    ("soh", "batt_soh_pct", "BMS SOH", "%", "capacity_bms"),
    (
        "current",
        "batt_current_a",
        "Pack current",
        "A",
        "capacity_bms,monthly_exposure",
    ),
    (
        "pack_voltage",
        "batt_voltage_v",
        "Pack voltage",
        "V",
        "capacity_bms,monthly_exposure",
    ),
)
LIFECYCLE_CANDIDATES = (
    "sys_cycles",
    "sys_cycles_count",
    "sys_cyclecount",
    "sys.cycles",
)


def _utc_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC").to_pydatetime()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return _utc_timestamp(value).isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _resolve_serial_root(root: Path, serial: str, chemistry: str) -> Path:
    candidates = (
        root / f"chemistry={chemistry}" / f"serial={serial}",
        root / f"serial={serial}",
    )
    existing = [candidate for candidate in candidates if candidate.exists()]
    if len(existing) != 1:
        raise ValueError(
            f"Expected one raw telemetry location for {serial}; found {len(existing)}"
        )
    return existing[0]


def _parquet_files(root: Path) -> list[Path]:
    files = sorted(path for path in root.rglob("*.parquet") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return files


def _latest_summaries(root: Path, serial: str, pattern: str) -> list[Path]:
    return sorted((root / f"serial={serial}").glob(f"{pattern}/model_summary.json"))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _input_manifest(
    *,
    source_root: Path,
    raw_files: list[Path],
    supporting_files: list[Path],
    serial: str,
    chemistry: str,
    module_count: int,
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in [*raw_files, *supporting_files]:
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(source_root)).replace("\\", "/"),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "validator_version": VALIDATOR_VERSION,
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return payload | {"fingerprint_sha256": hashlib.sha256(encoded).hexdigest()}


def _cached_run(
    output_root: Path, chemistry: str, serial: str, fingerprint: str
) -> Path | None:
    serial_root = output_root / f"chemistry={chemistry}" / f"serial={serial}"
    for path in sorted(
        serial_root.glob("validation_run=*/input_manifest.json"), reverse=True
    ):
        try:
            if _read_json(path).get("fingerprint_sha256") == fingerprint:
                return path.parent
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _raw_schema(files: list[Path]) -> set[str]:
    columns: set[str] = set()
    for path in (files[0], files[-1]):
        columns.update(pq.read_schema(path).names)
    return columns


def _sql_paths(files: list[Path]) -> list[str]:
    return [str(path.resolve()).replace("\\", "/") for path in files]


def _partition_groups(files: list[Path]) -> list[list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for path in files:
        groups.setdefault(path.parent, []).append(path)
    return [groups[key] for key in sorted(groups)]


def _raw_quality(
    files: list[Path], schema: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [name for _, name, *_ in CORE_SIGNALS if name in schema]
    lifecycle = next((name for name in LIFECYCLE_CANDIDATES if name in schema), None)
    selected = ["time", *required, *([lifecycle] if lifecycle else [])]
    projection = ", ".join(f'"{column}"' for column in selected)
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    paths = _sql_paths(files)
    invalid_soc_sql = (
        "sum(CASE WHEN batt_soc_pct < 0 OR batt_soc_pct > 100 THEN 1 ELSE 0 END)"
        if "batt_soc_pct" in schema
        else "0"
    )
    invalid_soh_sql = (
        "sum(CASE WHEN batt_soh_pct < 0 OR batt_soh_pct > 100 THEN 1 ELSE 0 END)"
        if "batt_soh_pct" in schema
        else "0"
    )
    monthly_frames: list[pd.DataFrame] = []
    for partition_files in _partition_groups(files):
        monthly_frames.append(
            connection.execute(
                f"""
                WITH source AS (
                    SELECT {projection}
                    FROM read_parquet(?, union_by_name=true)
                    WHERE time IS NOT NULL
                ), seconds AS (
                    SELECT time,
                           count(*) AS rows_at_time,
                           {", ".join(f'avg("{column}") AS "{column}"' for column in required)}
                    FROM source
                    GROUP BY time
                ), ordered AS (
                    SELECT *, epoch(time - lag(time) OVER (ORDER BY time)) AS gap_seconds
                    FROM seconds
                )
                SELECT strftime(time AT TIME ZONE 'UTC', '%Y-%m-01T00:00:00Z') AS month,
                       count(*) AS distinct_timestamps,
                       sum(rows_at_time) AS row_count,
                       sum(rows_at_time - 1) AS duplicate_rows,
                       cast(min(time) AS VARCHAR) AS first_timestamp,
                       cast(max(time) AS VARCHAR) AS last_timestamp,
                       max(gap_seconds) AS maximum_gap_seconds,
                       100.0 * count(*) /
                         greatest(1, epoch(max(time) - min(time)) + 1)
                           AS observed_span_coverage_pct,
                       {invalid_soc_sql} AS invalid_soc_rows,
                       {invalid_soh_sql} AS invalid_soh_rows
                FROM ordered
                GROUP BY 1
                ORDER BY 1
                """,
                [_sql_paths(partition_files)],
            ).fetchdf()
        )
    partition_months = pd.concat(monthly_frames, ignore_index=True).sort_values(
        ["month", "first_timestamp"]
    )
    reduced_months: list[dict[str, Any]] = []
    for month, rows in partition_months.groupby("month", sort=True):
        rows = rows.sort_values("first_timestamp")
        starts = pd.to_datetime(rows["first_timestamp"], utc=True).reset_index(
            drop=True
        )
        ends = pd.to_datetime(rows["last_timestamp"], utc=True).reset_index(drop=True)
        boundary_gaps = (
            starts.iloc[1:].reset_index(drop=True) - ends.iloc[:-1]
        ).dt.total_seconds()
        within_gap = rows["maximum_gap_seconds"].max()
        boundary_gap = boundary_gaps.max() if not boundary_gaps.empty else 0
        first = starts.min()
        last = ends.max()
        distinct = int(rows["distinct_timestamps"].sum())
        elapsed = max(1.0, (last - first).total_seconds() + 1)
        reduced_months.append(
            {
                "month": month,
                "distinct_timestamps": distinct,
                "row_count": int(rows["row_count"].sum()),
                "duplicate_rows": int(rows["duplicate_rows"].sum()),
                "first_timestamp": first.isoformat(),
                "last_timestamp": last.isoformat(),
                "maximum_gap_seconds": max(
                    float(within_gap) if pd.notna(within_gap) else 0.0,
                    float(boundary_gap) if pd.notna(boundary_gap) else 0.0,
                ),
                "observed_span_coverage_pct": 100.0 * distinct / elapsed,
                "invalid_soc_rows": int(rows["invalid_soc_rows"].sum()),
                "invalid_soh_rows": int(rows["invalid_soh_rows"].sum()),
            }
        )
    monthly = pd.DataFrame(reduced_months)
    first_times = pd.to_datetime(monthly["first_timestamp"], utc=True)
    last_times = pd.to_datetime(monthly["last_timestamp"], utc=True)
    cross_partition_gaps = (
        first_times.reset_index(drop=True).iloc[1:].reset_index(drop=True)
        - last_times.reset_index(drop=True).iloc[:-1].reset_index(drop=True)
    ).dt.total_seconds()
    within_gap = monthly["maximum_gap_seconds"].max()
    cross_gap = cross_partition_gaps.max() if not cross_partition_gaps.empty else 0
    maximum_gap = max(
        float(within_gap) if pd.notna(within_gap) else 0.0,
        float(cross_gap) if pd.notna(cross_gap) else 0.0,
    )
    first_timestamp = first_times.min()
    last_timestamp = last_times.max()
    distinct_timestamps = int(monthly["distinct_timestamps"].sum())
    elapsed_seconds = max(1.0, (last_timestamp - first_timestamp).total_seconds() + 1)
    sign = {"status": "inconclusive", "agreement_pct": None, "evidence_windows": 0}
    if {"batt_soc_pct", "batt_current_a"}.issubset(schema):
        sign_row = connection.execute(
            """
            WITH source AS (
                SELECT time, batt_soc_pct AS soc, batt_current_a AS current
                FROM read_parquet(?, union_by_name=true)
                WHERE time IS NOT NULL AND soc IS NOT NULL AND current IS NOT NULL
            ), windows AS (
                SELECT time_bucket(INTERVAL '5 minutes', time) AS bucket,
                       avg(current) AS mean_current,
                       arg_min(soc, time) AS start_soc,
                       arg_max(soc, time) AS end_soc
                FROM source GROUP BY 1
            ), evidence AS (
                SELECT *, end_soc - start_soc AS soc_delta
                FROM windows
                WHERE abs(mean_current) >= 1.0 AND abs(end_soc - start_soc) >= 0.1
            )
            SELECT count(*) AS evidence_windows,
                   100.0 * avg(CASE WHEN sign(mean_current) = sign(soc_delta) THEN 1 ELSE 0 END)
                       AS agreement_pct
            FROM evidence
            """,
            [paths],
        ).fetchone()
        evidence = int(sign_row[0] or 0)
        agreement = float(sign_row[1]) if sign_row[1] is not None else None
        status = "inconclusive"
        if evidence >= 100 and agreement is not None:
            status = (
                "verified"
                if agreement >= 80.0
                else "conflict"
                if agreement <= 20.0
                else "inconclusive"
            )
        sign = {
            "status": status,
            "agreement_pct": agreement,
            "evidence_windows": evidence,
            "tested_semantics": "positive current = charge; negative current = discharge",
        }
    connection.close()
    quality = {
        "distinct_timestamps": distinct_timestamps,
        "row_count": int(monthly["row_count"].sum()),
        "duplicate_rows": int(monthly["duplicate_rows"].sum()),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "maximum_gap_seconds": maximum_gap,
        "calendar_coverage_pct": 100.0 * distinct_timestamps / elapsed_seconds,
        "current_sign_verification": sign,
        "lifecycle_counter_column": lifecycle,
    }
    return monthly, quality


def _feature_inventory(
    *,
    processed_root: Path,
    serial: str,
    module_count: int,
) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    audit_paths = _latest_summaries(
        processed_root / "bms_electrothermal_source_quality",
        serial,
        "source_audit_run=*",
    )
    feature_paths = _latest_summaries(
        processed_root / "bms_hierarchical_features", serial, "feature_run=*"
    )
    latest_audit = _read_json(audit_paths[-1]) if audit_paths else {}
    expected = expected_hierarchical_signals(module_count)
    map_paths = [path.parent / "channel_map.parquet" for path in feature_paths]
    map_paths = [path for path in map_paths if path.exists()]
    system_paths = sorted(
        path
        for summary_path in feature_paths
        for path in summary_path.parent.rglob("system_features_5min.parquet")
    )
    present: set[str] = set()
    for path in map_paths:
        table = pq.read_table(path, columns=["source_signal_name"])
        present.update(str(value) for value in table.column(0).to_pylist())
    rows = []
    for signal in expected:
        rows.append(
            signal.to_dict() | {"available": signal.source_signal_name in present}
        )
    topology = pd.DataFrame(rows)
    starts: list[datetime] = []
    ends: list[datetime] = []
    feature_rows = 0
    feature_intervals: list[dict[str, str]] = []
    for path in feature_paths:
        summary = _read_json(path)
        start, end = (
            _utc_timestamp(summary.get("start")),
            _utc_timestamp(summary.get("end")),
        )
        if start:
            starts.append(start)
        if end:
            ends.append(end)
        if start and end:
            feature_intervals.append(
                {"start": start.isoformat(), "end": end.isoformat()}
            )
        feature_rows += int(summary.get("feature_rows") or 0)
    suspicious_temperature_windows: int | None = None
    if system_paths:
        connection = duckdb.connect()
        suspicious_temperature_windows = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT window_start_utc,
                           min(system_temperature_min_c) AS minimum_c,
                           max(system_temperature_max_c) AS maximum_c
                    FROM read_parquet(?, union_by_name=true)
                    GROUP BY 1
                )
                WHERE minimum_c < -40 OR maximum_c > 100
                """,
                [_sql_paths(system_paths)],
            ).fetchone()[0]
        )
        connection.close()
    present_voltage = int(
        topology.loc[
            topology["measurement_type"] == "cell_voltage_channel", "available"
        ].sum()
    )
    present_temperature = int(
        topology.loc[
            topology["measurement_type"] == "pack_temperature_sensor", "available"
        ].sum()
    )
    expected_voltage = module_count * 2 * 14
    expected_temperature = module_count * 2 * 4
    summary = {
        "feature_available": bool(feature_paths),
        "feature_start": min(starts).isoformat() if starts else None,
        "feature_end": max(ends).isoformat() if ends else None,
        "feature_rows": feature_rows,
        "aggregation_minutes": 5,
        "expected_voltage_groups": expected_voltage,
        "available_voltage_groups": present_voltage,
        "expected_temperature_sensors": expected_temperature,
        "available_temperature_sensors": present_temperature,
        "source_audit_available": bool(audit_paths),
        "source_audit_ready": bool(latest_audit.get("source_quality_ready")),
        "source_audit_run_id": audit_paths[-1].parent.name if audit_paths else None,
        "feature_run_ids": [path.parent.name for path in feature_paths],
        "feature_intervals": feature_intervals,
        "suspicious_temperature_windows": suspicious_temperature_windows,
    }
    supporting = [*audit_paths, *feature_paths, *map_paths, *system_paths]
    return topology, summary, supporting


def validate_device(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    source_root: Path,
    raw_root: Path,
    processed_root: Path,
    output_root: Path,
    compression: str = "zstd",
) -> Path:
    chemistry = chemistry.upper()
    raw_serial_root = _resolve_serial_root(raw_root, serial, chemistry)
    raw_files = _parquet_files(raw_serial_root)
    topology, feature_summary, supporting = _feature_inventory(
        processed_root=processed_root, serial=serial, module_count=module_count
    )
    manifest = _input_manifest(
        source_root=source_root,
        raw_files=raw_files,
        supporting_files=supporting,
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
    )
    cached = _cached_run(output_root, chemistry, serial, manifest["fingerprint_sha256"])
    if cached is not None:
        print(f"Reusing unchanged validation: {cached}", flush=True)
        return cached

    schema = _raw_schema(raw_files)
    if "time" not in schema:
        raise ValueError("Raw telemetry has no time column")
    monthly, quality = _raw_quality(raw_files, schema)
    first = _utc_timestamp(quality["first_timestamp"])
    last = _utc_timestamp(quality["last_timestamp"])
    stale_days = (datetime.now(UTC) - last).total_seconds() / 86400.0 if last else None

    inventory_rows = []
    for key, column, label, unit, purposes in CORE_SIGNALS:
        inventory_rows.append(
            {
                "signal_key": key,
                "source_column": column,
                "label": label,
                "unit": unit,
                "resolution": "1 second",
                "available": column in schema,
                "purposes": purposes,
                "semantics": (
                    "Positive = charge; negative = discharge"
                    if key == "current"
                    else "BMS-reported telemetry"
                ),
            }
        )
    lifecycle = quality["lifecycle_counter_column"]
    inventory_rows.append(
        {
            "signal_key": "lifecycle_counter",
            "source_column": lifecycle or "sys.cycles",
            "label": "BMS lifecycle counter",
            "unit": "count",
            "resolution": "source-dependent",
            "available": lifecycle is not None,
            "purposes": "capacity_bms",
            "semantics": "Counter semantics are unverified until documented by the BMS owner",
        }
    )
    inventory = pd.DataFrame(inventory_rows)

    duplicate_rows = int(quality["duplicate_rows"] or 0)
    invalid_rows = int(
        monthly["invalid_soc_rows"].sum() + monthly["invalid_soh_rows"].sum()
    )
    current_status = quality["current_sign_verification"]["status"]
    core_ready = all(column in schema for _, column, *_ in CORE_SIGNALS)
    voltage_ready = (
        feature_summary["available_voltage_groups"]
        == feature_summary["expected_voltage_groups"]
        and feature_summary["source_audit_ready"]
    )
    feature_ready = bool(feature_summary["feature_available"])
    readiness = {
        "capacity_bms": {
            "status": "ready"
            if core_ready and current_status != "conflict"
            else "blocked",
            "reason": (
                "Core BMS signals are present and current direction is usable."
                if core_ready and current_status != "conflict"
                else (
                    "Core BMS signals are missing or current direction conflicts with SOC movement."
                )
            ),
        },
        "monthly_exposure": {
            "status": "ready" if core_ready and feature_ready else "blocked",
            "reason": (
                "One-second operating signals and five-minute electrothermal "
                "features are available."
                if core_ready and feature_ready
                else (
                    "One-second operating signals or five-minute electrothermal "
                    "features are missing."
                )
            ),
        },
        "voltage_group_health": {
            "status": "ready" if voltage_ready else "blocked",
            "reason": (
                "Every expected voltage group is mapped and the source audit passed."
                if voltage_ready
                else (
                    "The hierarchical voltage map is incomplete or its source audit has not passed."
                )
            ),
        },
    }
    findings = [
        {
            "finding_key": "duplicates",
            "severity": "warning" if duplicate_rows else "pass",
            "value": duplicate_rows,
            "message": f"{duplicate_rows:,} duplicate timestamp rows detected.",
        },
        {
            "finding_key": "invalid_core_values",
            "severity": "warning" if invalid_rows else "pass",
            "value": invalid_rows,
            "message": f"{invalid_rows:,} SOC/SOH values are outside 0-100%.",
        },
        {
            "finding_key": "maximum_gap",
            "severity": "info",
            "value": float(quality["maximum_gap_seconds"] or 0),
            "message": "Maximum gap in imported one-second telemetry.",
        },
        {
            "finding_key": "freshness",
            "severity": "warning" if stale_days is None or stale_days > 30 else "pass",
            "value": stale_days,
            "message": "Latest local timestamp is more than 30 days old."
            if stale_days and stale_days > 30
            else "Local telemetry is current within 30 days.",
        },
        {
            "finding_key": "lifecycle_counter",
            "severity": "info" if lifecycle is None else "pass",
            "value": lifecycle,
            "message": "BMS lifecycle counter is not imported; its semantics remain unverified."
            if lifecycle is None
            else "BMS lifecycle counter is present; semantics still require owner verification.",
        },
    ]
    quality_findings = pd.DataFrame(findings)
    monthly["month"] = pd.to_datetime(monthly["month"], utc=True)
    monthly["one_second_available"] = True
    feature_months: set[str] = set()
    for interval in feature_summary["feature_intervals"]:
        start_month = pd.Period(interval["start"][:7], freq="M")
        end_month = pd.Period(interval["end"][:7], freq="M")
        feature_months.update(
            str(month) for month in pd.period_range(start_month, end_month)
        )
    monthly["five_minute_available"] = (
        monthly["month"].dt.strftime("%Y-%m").isin(feature_months)
    )
    monthly["voltage_groups_available"] = monthly["five_minute_available"].map(
        {True: feature_summary["available_voltage_groups"], False: 0}
    )
    monthly["temperature_sensors_available"] = monthly["five_minute_available"].map(
        {True: feature_summary["available_temperature_sensors"], False: 0}
    )

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        output_root
        / f"chemistry={chemistry}"
        / f"serial={serial}"
        / f"validation_run={run_id}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {
        "model_family": "battery_data_validation",
        "validator_version": VALIDATOR_VERSION,
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "validated_at": datetime.now(UTC).isoformat(),
        "timestamp_semantics": "UTC",
        "observed_start": first.isoformat() if first else None,
        "observed_end": last.isoformat() if last else None,
        "raw_file_count": len(raw_files),
        "raw_row_count": int(quality["row_count"] or 0),
        "distinct_timestamp_count": int(quality["distinct_timestamps"] or 0),
        "calendar_coverage_pct": float(quality["calendar_coverage_pct"] or 0),
        "maximum_gap_seconds": float(quality["maximum_gap_seconds"] or 0),
        "duplicate_rows": duplicate_rows,
        "invalid_core_rows": invalid_rows,
        "stale_days": stale_days,
        "current_sign_verification": quality["current_sign_verification"],
        "feature_evidence": feature_summary,
        "suspicious_temperature_windows": feature_summary[
            "suspicious_temperature_windows"
        ],
        "readiness": readiness,
        "resistance_reference": {
            "available": False,
            "interpretation_verified": False,
            "analysis_enabled": False,
            "note": "Resistance metadata is not used for scientific claims in Stage 1.",
        },
        "claim_limit": (
            "Validation establishes local data availability and internal consistency. "
            "It does not establish sensor calibration, physical battery health, or causality."
        ),
    }
    safe_summary = _json_safe(summary)
    write_parquet_chunk(
        inventory, run_dir / "signal_inventory.parquet", compression=compression
    )
    write_parquet_chunk(
        monthly, run_dir / "coverage_by_month.parquet", compression=compression
    )
    write_parquet_chunk(
        topology, run_dir / "topology_inventory.parquet", compression=compression
    )
    write_parquet_chunk(
        quality_findings, run_dir / "quality_findings.parquet", compression=compression
    )
    (run_dir / "validation_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run_dir / "model_summary.json").write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (run_dir / "input_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote data validation artifacts to {run_dir}", flush=True)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local battery analysis data."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--processed-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--compression", default="zstd", choices=("zstd", "snappy", "gzip")
    )
    parser.add_argument("--env-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    storage = StorageConfig.from_env(args.env_file)
    source_root = Path(__file__).resolve().parents[1]
    validate_device(
        serial=args.serial,
        chemistry=args.chemistry,
        module_count=args.module_count,
        source_root=source_root,
        raw_root=args.raw_root or storage.raw_parquet_dir,
        processed_root=args.processed_root or storage.data_dir / "processed",
        output_root=args.output_root
        or storage.data_dir / "processed" / "data_validation",
        compression=args.compression,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
