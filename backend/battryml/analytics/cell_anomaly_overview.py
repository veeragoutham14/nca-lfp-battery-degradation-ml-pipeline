from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.analytics.nca_cohort import (
    DEFAULT_ACTIVE_MODULE_COUNT,
    DEFAULT_REFERENCE_MODULE_COUNT,
    DEFAULT_REFERENCE_SERIAL,
    classify_chemistry_device,
)


def _latest_run(root: Path, serial: str) -> Path | None:
    serial_root = root / f"serial={serial}"
    runs = sorted(serial_root.glob("anomaly_run=*"))
    return runs[-1] if runs else None


def _latest_feature_summary(root: Path, serial: str) -> Path | None:
    summaries = sorted((root / f"serial={serial}").glob("feature_run=*/model_summary.json"))
    return summaries[-1] if summaries else None


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _downsample(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    step = (len(rows) - 1) / (max_points - 1)
    indexes = sorted({round(index * step) for index in range(max_points)})
    return [rows[index] for index in indexes]


def _hierarchical_series(run_dir: Path, serial: str, max_points: int) -> dict[str, Any] | None:
    summary_path = run_dir / "model_summary.json"
    daily_path = run_dir / "daily_anomaly_summary.parquet"
    episode_path = run_dir / "anomaly_episodes.parquet"
    if not summary_path.exists() or not daily_path.exists():
        return None
    summary = _read_summary(summary_path)
    module_count = int(summary.get("module_count") or 0) or None
    daily_rows = pq.ParquetFile(daily_path).read().to_pylist()
    points = []
    for row in daily_rows:
        maximum_score = max(
            (
                value
                for value in (
                    _finite(row.get("maximum_peer_z")),
                    _finite(row.get("maximum_low_rank_z")),
                    _finite(row.get("maximum_temporal_ewma_score")),
                )
                if value is not None
            ),
            default=None,
        )
        points.append(
            {
                "time": _json_value(row.get("day_utc")),
                "module_count": module_count,
                "module_configuration_epoch": None,
                "spread_p50_mv": _finite(row.get("system_spread_p50_mv")),
                "spread_p95_mv": _finite(row.get("system_spread_p95_mv")),
                "spread_max_mv": _finite(row.get("system_spread_max_mv")),
                "raw_spread_p95_mv": _finite(row.get("system_spread_p95_mv")),
                "compensated_spread_p95_mv": _finite(row.get("system_spread_p95_mv")),
                "baseline_threshold_median_mv": None,
                "baseline_threshold_p95_mv": None,
                "point_candidate_bins": int(row.get("point_candidate_windows") or 0),
                "sustained_candidate_bins": int(row.get("sustained_episode_minutes") or 0) // 5,
                "sustained_episode_minutes": _finite(row.get("sustained_episode_minutes")),
                "sustained_charging_minutes": float(row.get("sustained_charging_minutes") or 0),
                "sustained_discharging_minutes": float(
                    row.get("sustained_discharging_minutes") or 0
                ),
                "sustained_rest_minutes": float(row.get("sustained_rest_minutes") or 0),
                "sustained_mixed_minutes": float(row.get("sustained_mixed_minutes") or 0),
                "maximum_anomaly_score": maximum_score,
                "soc_mean_pct": _finite(row.get("soc_mean_pct")),
                "temperature_mean_c": _finite(row.get("temperature_mean_c")),
            }
        )
    episodes: list[dict[str, Any]] = []
    if episode_path.exists():
        for row in pq.ParquetFile(episode_path).read().to_pylist():
            episode_id = str(row.get("episode_id"))
            item = {key: _json_value(value) for key, value in row.items()}
            item["start_time"] = item.pop("start_time_utc", None)
            item["end_time"] = item.pop("end_time_utc", None)
            item["maximum_anomaly_score"] = max(
                (
                    value
                    for value in (
                        _finite(row.get("maximum_peer_z")),
                        _finite(row.get("maximum_low_rank_z")),
                        _finite(row.get("maximum_temporal_ewma_score")),
                    )
                    if value is not None
                ),
                default=None,
            )
            item["has_raw_evidence"] = (
                run_dir / "evidence" / f"episode_id={episode_id}" / "episode_evidence_1s.parquet"
            ).exists()
            episodes.append(item)
    return {
        "serial": serial,
        "run_id": run_dir.name,
        "detector_version": "hierarchical_v2",
        "summary": {
            "analysis_start": summary.get("analysis_start"),
            "analysis_end": summary.get("analysis_end"),
            "feature_rows": int(summary.get("feature_rows") or 0),
            "quality_valid_rows": int(summary.get("quality_valid_rows") or 0),
            "quality_valid_pct": _finite(summary.get("quality_valid_pct")),
            "sustained_anomaly_episode_count": int(
                summary.get("sustained_anomaly_episode_count") or 0
            ),
            "anomaly_days": int(summary.get("anomaly_days") or 0),
            "p95_analysis_spread_mv": _finite(summary.get("p95_analysis_spread_mv")),
            "maximum_analysis_spread_mv": _finite(
                summary.get("maximum_analysis_spread_mv")
            ),
            "maximum_bms_reported_voltage_spread_mv": _finite(
                summary.get("maximum_bms_reported_voltage_spread_mv")
            ),
            "bms_global_spread_implausible_windows": int(
                summary.get("bms_global_spread_implausible_windows") or 0
            ),
            "spread_basis": summary.get("spread_basis"),
            "episode_severity_counts": summary.get("episode_severity_counts") or {},
            "evidence_episode_count": int(summary.get("evidence_episode_count") or 0),
            "evidence_requested_episode_count": int(
                summary.get("evidence_requested_episode_count") or 0
            ),
            "evidence_status": summary.get("evidence_status"),
            "evidence_error": summary.get("evidence_error"),
            "first_episode_start": summary.get("first_episode_start"),
            "latest_episode_end": summary.get("latest_episode_end"),
            "module_epoch_baselines": [],
            "engineering_bands_mv": (
                (summary.get("detector") or {}).get("fixed_engineering_bands_mv") or {}
            ),
            "ground_truth_status": summary.get("ground_truth_status"),
            "claim_limit": summary.get("claim_limit"),
            "context_value_semantics": summary.get("context_value_semantics"),
        },
        "points": _downsample(points, max_points),
        "episodes": episodes,
    }


def build_cell_anomaly_overview(
    *,
    source_root: Path,
    devices: list[dict[str, Any]],
    chemistry: str = "NCA",
    active_module_count: int = DEFAULT_ACTIVE_MODULE_COUNT,
) -> dict[str, Any]:
    processed_root = source_root / "data" / "processed"
    anomaly_root = processed_root / "hierarchical_cell_anomaly"
    feature_root = processed_root / "bms_hierarchical_features"
    chemistry_key = chemistry.strip().upper()
    generated_at = datetime.now(UTC)
    latest_modified_at: datetime | None = None
    rows: list[dict[str, Any]] = []

    for device in devices:
        membership = classify_chemistry_device(
            device,
            chemistry=chemistry_key,
            active_module_count=active_module_count,
            reference_serial=(DEFAULT_REFERENCE_SERIAL if chemistry_key == "NCA" else None),
            reference_module_count=DEFAULT_REFERENCE_MODULE_COUNT,
        )
        if membership is None:
            continue
        serial = str(device["serial"])
        feature_summary = _latest_feature_summary(feature_root, serial)
        run_dir = _latest_run(anomaly_root, serial)
        summary_path = run_dir / "model_summary.json" if run_dir else None
        summary: dict[str, Any] = {}
        source_modified_at = None
        if summary_path is not None and summary_path.exists():
            summary = _read_summary(summary_path)
            modified = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
            source_modified_at = modified.isoformat()
            latest_modified_at = (
                max(latest_modified_at, modified) if latest_modified_at else modified
            )

        maximum_score = max(
            (
                value
                for value in (
                    _finite(summary.get("maximum_anomaly_score")),
                    _finite(summary.get("maximum_peer_z")),
                    _finite(summary.get("maximum_low_rank_z")),
                    _finite(summary.get("maximum_temporal_ewma_score")),
                )
                if value is not None
            ),
            default=None,
        )
        rows.append(
            {
                "serial": serial,
                "display_name": device.get("display_name") or f"Battery {serial}",
                "variant_key": device.get("variant_key"),
                "device_role": device.get("device_role"),
                "cohort_role": membership.cohort_role,
                "module_count": membership.module_count,
                "module_count_status": device.get("module_count_status"),
                "cohort_status": membership.cohort_status,
                "cohort_reason": membership.reason,
                "in_active_cohort": membership.in_active_cohort,
                "history_built": feature_summary is not None,
                "legacy_history_archived": False,
                "analyzed": bool(summary),
                "legacy_analysis_archived": False,
                "detector_version": "hierarchical_v2",
                "analysis_start": summary.get("analysis_start"),
                "analysis_end": summary.get("analysis_end"),
                "feature_rows": summary.get("feature_rows"),
                "quality_valid_pct": _finite(summary.get("quality_valid_pct")),
                "scored_pct_of_valid": _finite(summary.get("scored_pct_of_valid")),
                "sustained_anomaly_episode_count": int(
                    summary.get("sustained_anomaly_episode_count") or 0
                ),
                "anomaly_days": int(summary.get("anomaly_days") or 0),
                "maximum_anomaly_score": maximum_score,
                "p95_analysis_spread_mv": _finite(summary.get("p95_analysis_spread_mv")),
                "maximum_analysis_spread_mv": _finite(summary.get("maximum_analysis_spread_mv")),
                "module_configuration_epoch_count": int(
                    summary.get("module_configuration_epoch_count") or 0
                ),
                "elevated_module_epoch_shift_count": int(
                    summary.get("elevated_module_epoch_shift_count") or 0
                ),
                "maximum_module_epoch_baseline_shift_mv": _finite(
                    summary.get("maximum_module_epoch_baseline_shift_mv")
                ),
                "first_episode_start": summary.get("first_episode_start"),
                "latest_episode_end": summary.get("latest_episode_end"),
                "source_feature_run_id": summary.get("source_feature_run_id"),
                "source_run_id": run_dir.name if run_dir else None,
                "source_modified_at": source_modified_at,
            }
        )

    rows.sort(
        key=lambda item: (
            item["module_count"] is None,
            item["module_count"] or 0,
            item["serial"],
        )
    )
    active_rows = [item for item in rows if item["in_active_cohort"]]
    deferred_rows = [item for item in rows if not item["in_active_cohort"]]
    active_field_rows = [item for item in active_rows if item["cohort_role"] == "field_device"]
    reference_rows = [item for item in active_rows if item["cohort_role"] == "reference_control"]
    analyzed = [item for item in active_rows if item["analyzed"]]
    return {
        "model_family": "cell_voltage_anomaly_overview",
        "chemistry": chemistry_key,
        "generated_at": generated_at.isoformat(),
        "latest_source_modified_at": (
            latest_modified_at.isoformat() if latest_modified_at else None
        ),
        "cohort": {
            "active_module_count": active_module_count,
            "policy": (
                f"All configured {chemistry_key} module variants remain visible; "
                "analysis is reported within explicit module-count cohorts."
            ),
        },
        "metrics": {
            "inventory_device_count": len(rows),
            "deferred_device_count": len(deferred_rows),
            "configured_device_count": len(active_rows),
            "field_device_count": len(active_field_rows),
            "reference_device_count": len(reference_rows),
            "history_device_count": sum(item["history_built"] for item in active_rows),
            "analyzed_device_count": len(analyzed),
            "archived_legacy_device_count": sum(
                item["legacy_analysis_archived"] for item in active_rows
            ),
            "candidate_device_count": sum(
                item["sustained_anomaly_episode_count"] > 0 for item in analyzed
            ),
            "total_sustained_episode_count": sum(
                item["sustained_anomaly_episode_count"] for item in analyzed
            ),
            "total_anomaly_days": sum(item["anomaly_days"] for item in analyzed),
            "elevated_epoch_shift_device_count": sum(
                item["elevated_module_epoch_shift_count"] > 0 for item in analyzed
            ),
        },
        "devices": rows,
        "evidence_policy": {
            "method": (
                "fixed engineering bands plus pack-peer, low-rank, EWMA, "
                "and persistence evidence"
            ),
            "conditioning": (
                "physical module/pack peers with five-minute BMS current, SOC, "
                "and pack temperature context"
            ),
            "ground_truth_status": "no independent cell-voltage instrumentation labels",
            "claim_limit": (
                "Results identify BMS-reported channel inconsistency candidates and "
                "the affected module, pack, and channel. They do not prove a physical "
                "cell defect without independent measurement."
            ),
        },
    }


def build_cell_anomaly_series(
    *,
    source_root: Path,
    serial: str,
    max_points: int = 1000,
) -> dict[str, Any] | None:
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    processed_root = source_root / "data" / "processed"
    run_dir = _latest_run(processed_root / "hierarchical_cell_anomaly", serial)
    if run_dir is not None:
        return _hierarchical_series(run_dir, serial, max_points)
    return None
