from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def _finite(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    return value


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = pq.read_table(path).to_pylist()
    return [{str(key): _finite(value) for key, value in row.items()} for row in rows]


def latest_phase4_run(source_root: Path, chemistry: str, serial: str) -> Path | None:
    base = (
        source_root
        / "data"
        / "processed"
        / "soh_gap_diagnosis"
        / f"chemistry={chemistry}"
        / f"serial={serial}"
    )
    runs = sorted(base.glob("phase4_run=*"))
    return runs[-1] if runs else None


def build_soh_gap_diagnosis(
    source_root: Path, chemistry: str, serial: str
) -> dict[str, Any] | None:
    run = latest_phase4_run(source_root, chemistry, serial)
    if run is None:
        return None
    summary = json.loads((run / "device_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "phase4_manifest.json").read_text(encoding="utf-8"))
    return _finite(
        {
            "run_id": run.name,
            "summary": summary,
            "manifest": manifest,
            "observations": _records(run / "reference_normalized_observations.parquet"),
            "transitions": _records(run / "soh_transitions.parquet"),
            "assessments": _records(run / "diagnostic_assessments.parquet"),
            "stress_intervals": _records(run / "stress_intervals.parquet"),
            "device_month_stress": _records(run / "device_month_stress.parquet"),
            "verification_points": _records(run / "event_verification_points.parquet"),
            "candidate_reconciliation": _records(run / "candidate_reconciliation.parquet"),
        }
    )
