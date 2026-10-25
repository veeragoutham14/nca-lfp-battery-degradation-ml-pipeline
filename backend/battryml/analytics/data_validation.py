from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.data.storage_paths import normalize_chemistry


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _latest_run(source_root: Path, serial: str, chemistry: str) -> Path | None:
    root = source_root / "data" / "processed" / "data_validation"
    candidates = [
        root / f"chemistry={chemistry}" / f"serial={serial}",
        root / f"serial={serial}",
    ]
    runs = [
        run
        for serial_root in candidates
        if serial_root.exists()
        for run in serial_root.glob("validation_run=*")
        if (run / "validation_summary.json").exists()
    ]
    return max(runs, key=lambda path: path.name) if runs else None


def build_data_validation(
    *, source_root: Path, serial: str, chemistry: str
) -> dict[str, Any] | None:
    chemistry_key = normalize_chemistry(chemistry)
    run = _latest_run(source_root, serial, chemistry_key)
    if run is None:
        return None
    summary = json.loads((run / "validation_summary.json").read_text(encoding="utf-8"))
    return {
        "run_id": run.name,
        "summary": summary,
        "signal_inventory": _records(run / "signal_inventory.parquet"),
        "coverage_by_month": _records(run / "coverage_by_month.parquet"),
        "topology_inventory": _records(run / "topology_inventory.parquet"),
        "quality_findings": _records(run / "quality_findings.parquet"),
    }


def build_electrothermal_validation(
    *, source_root: Path, serial: str, chemistry: str
) -> dict[str, Any] | None:
    chemistry_key = normalize_chemistry(chemistry)
    data_root = source_root / "data"
    validation_root = (
        data_root
        / "processed"
        / "electrothermal_import_validation"
        / f"chemistry={chemistry_key}"
        / f"serial={serial}"
    )
    summaries = sorted(validation_root.glob("validation_run=*/model_summary.json"))
    if not summaries:
        return None
    summary_path = summaries[-1]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    latest_source_mtime_ns = 0
    for group in ("temperature", "cell_voltage"):
        source = (
            data_root
            / "electrothermal_telemetry"
            / group
            / "5m"
            / f"chemistry={chemistry_key}"
            / f"serial={serial}"
        )
        if source.exists():
            latest_source_mtime_ns = max(
                latest_source_mtime_ns,
                max(
                    (path.stat().st_mtime_ns for path in source.rglob("*.parquet")),
                    default=0,
                ),
            )
    stale = latest_source_mtime_ns > summary_path.stat().st_mtime_ns
    groups = {}
    for key in ("temperature", "cell_voltage"):
        evidence = summary.get("groups", {}).get(key, {})
        groups[key] = {
            "ready": bool(evidence.get("ready")),
            "expected_channels": int(evidence.get("expected_channels") or 0),
            "minimum_channel_coverage_pct": round(
                float(evidence.get("minimum_channel_coverage") or 0.0) * 100.0, 2
            ),
            "start_utc": evidence.get("start_utc"),
            "end_utc": evidence.get("end_utc"),
            "duplicate_timestamps": int(evidence.get("duplicate_timestamps") or 0),
            "misaligned_five_minute_rows": int(
                evidence.get("misaligned_five_minute_rows") or 0
            ),
        }
    ready = bool(summary.get("ready")) and all(item["ready"] for item in groups.values())
    build_ready = ready and not stale
    return {
        "run_id": summary_path.parent.name,
        "serial": serial,
        "chemistry": chemistry_key,
        "module_count": int(summary.get("module_count") or 0),
        "validated_at": summary.get("validated_at"),
        "stale": stale,
        "build_ready": build_ready,
        "status": "stale" if stale else "ready" if ready else "failed",
        "reason": (
            "Imported temperature or voltage data is newer; validate again."
            if stale
            else "Temperature and voltage data are ready for feature building."
            if ready
            else "Temperature or voltage validation failed."
        ),
        "groups": groups,
    }
