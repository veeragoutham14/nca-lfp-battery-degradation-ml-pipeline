from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.config import Settings
from battryml.plugins.bms_soh_recalibration import BmsSohRecalibrationPlugin
from battryml.plugins.electrothermal import (
    ElectrothermalSourceAuditPlugin,
    HierarchicalElectrothermalFeaturesPlugin,
)
from battryml.plugins.soh_signal_audit import SohSignalAuditPlugin
from battryml.plugins.veera_ingestion import RawTelemetryImportPlugin


def _run_stage(name: str, command: tuple[str, ...], *, cwd: Path) -> None:
    print(f"BATTRYML_STAGE_START {name}", flush=True)
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    print(f"BATTRYML_STAGE_COMPLETE {name}", flush=True)


def _raw_bounds(root: Path, serial: str, chemistry: str) -> tuple[datetime, datetime]:
    candidates = (
        root / f"chemistry={chemistry}" / f"serial={serial}",
        root / f"serial={serial}",
    )
    existing = [path for path in candidates if path.exists()]
    if len(existing) != 1:
        raise ValueError(f"Expected one raw storage location for {serial}; found {len(existing)}")
    minimum: datetime | None = None
    maximum: datetime | None = None
    for path in existing[0].rglob("*.parquet"):
        parquet = pq.ParquetFile(path)
        if "time" not in parquet.schema_arrow.names:
            continue
        column_index = parquet.schema_arrow.names.index("time")
        for row_group_index in range(parquet.num_row_groups):
            statistics = parquet.metadata.row_group(row_group_index).column(column_index).statistics
            if statistics is None or not statistics.has_min_max:
                continue
            start = statistics.min
            end = statistics.max
            start = start if start.tzinfo else start.replace(tzinfo=UTC)
            end = end if end.tzinfo else end.replace(tzinfo=UTC)
            minimum = start if minimum is None else min(minimum, start)
            maximum = end if maximum is None else max(maximum, end)
    if minimum is None or maximum is None:
        raise ValueError(f"Raw telemetry for {serial} has no usable timestamp statistics")
    return minimum, maximum + timedelta(seconds=1)


def _latest_feature_end(root: Path, serial: str) -> datetime | None:
    summaries = sorted((root / f"serial={serial}").glob("feature_run=*/model_summary.json"))
    for path in reversed(summaries):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("end")
            if value:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _phase3_uses_catalog(reference_root: Path, catalog_run_id: str) -> bool:
    summaries = sorted(reference_root.glob("phase3_run=*/model_summary.json"))
    if not summaries:
        return False
    try:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return summary.get("source_physical_catalog_run_id") == catalog_run_id


def validate_analysis_inputs(args: argparse.Namespace, settings: Settings) -> None:
    raw_start, raw_end = _raw_bounds(
        settings.source_root / "data" / "raw_parquet", args.serial, args.chemistry
    )
    _run_stage(
        "soh_signal_audit",
        SohSignalAuditPlugin().build_command(
            settings,
            {
                "serial": args.serial,
                "nominal_capacity_ah": args.nominal_capacity_ah,
                "start": raw_start.isoformat(),
                "end": raw_end.isoformat(),
            },
        ),
        cwd=settings.source_root,
    )
    _run_stage(
        "data_validation",
        (
            str(settings.source_python),
            "-m",
            "src.data_validation",
            "--serial",
            args.serial,
            "--chemistry",
            args.chemistry,
            "--module-count",
            str(args.module_count),
        ),
        cwd=settings.source_root,
    )


def prepare_data(args: argparse.Namespace, settings: Settings) -> None:
    common = {"serial": args.serial, "module_count": args.module_count}
    if args.update_raw:
        values: dict[str, Any] = {
            "serial": args.serial,
            "chemistry": args.chemistry,
            "soc_resolution": "1s",
            "soh_resolution": "1s",
            "voltage_resolution": "1s",
            "temperature_resolution": "5m",
            "current_resolution": "1s",
            "cycle_count_resolution": "5m",
            "chunk_size": 100_000,
            "compression": "zstd",
            "log_level": "INFO",
        }
        if args.start:
            values["start"] = args.start
        if args.end:
            values["end"] = args.end
        _run_stage(
            "raw_telemetry_import",
            RawTelemetryImportPlugin().build_command(settings, values),
            cwd=settings.source_root,
        )
    raw_start, raw_end = _raw_bounds(
        settings.source_root / "data" / "raw_parquet", args.serial, args.chemistry
    )
    _run_stage(
        "electrothermal_source_audit",
        ElectrothermalSourceAuditPlugin().build_command(settings, common),
        cwd=settings.source_root,
    )
    feature_root = settings.source_root / "data" / "processed" / "bms_hierarchical_features"
    previous_end = _latest_feature_end(feature_root, args.serial)
    if previous_end is None:
        feature_start = raw_start
    elif previous_end < raw_end:
        feature_start = max(raw_start, previous_end)
    else:
        feature_start = max(raw_start, raw_end - timedelta(minutes=10))
    _run_stage(
        "hierarchical_features",
        HierarchicalElectrothermalFeaturesPlugin().build_command(
            settings,
            {
                **common,
                "start": feature_start.isoformat(),
                "end": raw_end.isoformat(),
                "aggregation_minutes": "5",
            },
        ),
        cwd=settings.source_root,
    )
    validate_analysis_inputs(args, settings)


def investigate(args: argparse.Namespace, settings: Settings) -> None:
    common = {
        "serial": args.serial,
        "module_count": args.module_count,
        "nominal_capacity_ah": args.nominal_capacity_ah,
    }
    physical_catalog_root = (
        settings.source_root / "data" / "processed" / "physical_discharge_catalog"
    )

    def physical_catalog_command(
        serial: str, module_count: int, capacity_ah: float
    ) -> tuple[str, ...]:
        return (
            str(settings.source_python),
            "-m",
            "src.physical_discharge_catalog",
            "--serial",
            serial,
            "--chemistry",
            args.chemistry,
            "--module-count",
            str(module_count),
            "--nominal-capacity-ah",
            str(capacity_ah),
            "--events-root",
            str(settings.source_root / "data" / "processed" / "events"),
            "--output-root",
            str(physical_catalog_root),
        )

    _run_stage(
        "physical_discharge_catalog",
        physical_catalog_command(args.serial, args.module_count, args.nominal_capacity_ah),
        cwd=settings.source_root,
    )
    if args.chemistry == "NCA" and args.serial != "100000017":
        reference_catalog_root = physical_catalog_root / "serial=100000017"
        if not any(reference_catalog_root.glob("catalog_run=*/physical_discharge_cycles.parquet")):
            _run_stage(
                "reference_physical_discharge_catalog",
                physical_catalog_command("100000017", 2, 56.0),
                cwd=settings.source_root,
            )
    phase2_command = (
        str(settings.source_python),
        "-m",
        "src.capacity_bms_validation",
        "--serial",
        args.serial,
        "--chemistry",
        args.chemistry,
        "--module-count",
        str(args.module_count),
        "--reference-capacity-ah",
        str(args.nominal_capacity_ah),
        "--reference-capacity-source",
        (
            "measured_first_full_cycle"
            if args.serial == "100000017"
            else "device_configuration"
        ),
        "--validation-root",
        str(settings.source_root / "data" / "processed" / "data_validation"),
        "--recalibration-root",
        str(settings.source_root / "data" / "processed" / "bms_soh_recalibration"),
        "--physical-catalog-root",
        str(physical_catalog_root),
    )
    cache_check = subprocess.run(
        (*phase2_command, "--check-cache"),
        cwd=settings.source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if cache_check.returncode == 0:
        print(cache_check.stdout.strip(), flush=True)
    elif cache_check.returncode != 3:
        detail = cache_check.stderr.strip() or cache_check.stdout.strip()
        raise RuntimeError(f"phase2_cache_check failed: {detail}")
    else:
        _run_stage(
            "bms_soh_recalibration",
            BmsSohRecalibrationPlugin().build_command(settings, common),
            cwd=settings.source_root,
        )
        _run_stage("capacity_bms_validation", phase2_command, cwd=settings.source_root)
    def phase3_command(serial: str, module_count: int, capacity_ah: float) -> tuple[str, ...]:
        return (
            str(settings.source_python),
            "-m",
            "src.electrothermal_capacity_context",
            "--serial",
            serial,
            "--chemistry",
            args.chemistry,
            "--module-count",
            str(module_count),
            "--nominal-capacity-ah",
            str(capacity_ah),
            "--recalibration-root",
            str(settings.source_root / "data" / "processed" / "bms_soh_recalibration"),
            "--capacity-trend-root",
            str(settings.source_root / "data" / "processed" / "capacity_trend"),
            "--physical-catalog-root",
            str(physical_catalog_root),
            "--output-root",
            str(settings.source_root / "data" / "processed" / "electrothermal_capacity_context"),
            "--cache-root",
            str(settings.source_root / "data" / "cache" / "electrothermal_event_windows"),
        )

    if args.chemistry == "NCA" and args.serial != "100000017":
        reference_root = (
            settings.source_root
            / "data"
            / "processed"
            / "electrothermal_capacity_context"
            / "chemistry=NCA"
            / "serial=100000017"
        )
        reference_catalog_run = sorted(
            (physical_catalog_root / "serial=100000017").glob("catalog_run=*")
        )[-1]
        if not _phase3_uses_catalog(reference_root, reference_catalog_run.name):
            _run_stage(
                "electrothermal_reference_context",
                phase3_command("100000017", 2, 56.0),
                cwd=settings.source_root,
            )
    _run_stage(
        "electrothermal_capacity_context",
        phase3_command(args.serial, args.module_count, args.nominal_capacity_ah),
        cwd=settings.source_root,
    )
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a BatteryML device workflow.")
    parser.add_argument("mode", choices=("prepare", "validate", "investigate"))
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--nominal-capacity-ah", required=True, type=float)
    parser.add_argument("--update-raw", action="store_true")
    parser.add_argument("--start")
    parser.add_argument("--end")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    try:
        if args.mode == "prepare":
            prepare_data(args, settings)
        elif args.mode == "validate":
            validate_analysis_inputs(args, settings)
        else:
            investigate(args, settings)
        return 0
    except Exception as exc:
        print(f"Workflow failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
