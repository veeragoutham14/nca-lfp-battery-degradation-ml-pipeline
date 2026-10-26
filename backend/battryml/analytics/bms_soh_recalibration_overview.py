from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import median
from typing import Any

import pyarrow.parquet as pq


def _latest_run(root: Path, serial: str) -> Path | None:
    runs = sorted((root / f"serial={serial}").glob("recalibration_run=*"))
    return runs[-1] if runs else None


def _latest_capacity_run(source_root: Path, serial: str) -> Path | None:
    root = source_root / "data" / "processed" / "discharge_capacity_evidence"
    runs = sorted((root / f"serial={serial}").glob("capacity_evidence_run=*"))
    return runs[-1] if runs else None


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


def _read_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "model_summary.json").read_text(encoding="utf-8"))


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in pq.ParquetFile(path).read().to_pylist()
    ]


def _median_value(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for row in rows if (value := _finite(row.get(key))) is not None]
    return median(values) if values else None


def _error_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [value for row in rows if (value := _finite(row.get(key))) is not None]
    return {
        "rows": len(values),
        "mae_pp": sum(abs(value) for value in values) / len(values) if values else None,
        "bias_pp": sum(values) / len(values) if values else None,
        "maximum_absolute_error_pp": max(map(abs, values)) if values else None,
    }


def _link_capacity_evidence(
    rows: list[dict[str, Any]], events: list[dict[str, Any]], *, maximum_hours: float = 72.0
) -> list[dict[str, Any]]:
    dated_events = []
    for event in events:
        value = event.get("event_time")
        if value:
            dated_events.append((datetime.fromisoformat(str(value)), event))
    dated_events.sort(key=lambda item: item[0])
    linked: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        end_value = row.get("raw_end_time_utc")
        end = datetime.fromisoformat(str(end_value)) if end_value else None
        match = None
        if end is not None:
            match = next(
                (
                    (time, event)
                    for time, event in dated_events
                    if 0 <= (time - end).total_seconds() <= maximum_hours * 3600
                ),
                None,
            )
        if match:
            event_time, event = match
            row["linked_soh_event_id"] = event.get("event_id")
            row["linked_soh_event_time_utc"] = event_time.isoformat()
            row["hours_to_linked_soh_event"] = (event_time - end).total_seconds() / 3600
            row["linked_soh_event_delta_pp"] = event.get("soh_delta_pp")
            row["bms_soh_after_linked_event_pct"] = event.get("new_soh_pct")
            after = _finite(event.get("new_soh_pct"))
            direct = _finite(row.get("direct_capacity_soh_pct"))
            normalized = _finite(row.get("soc_normalized_capacity_soh_pct"))
            row["bms_after_event_minus_capacity_soh_pp"] = (
                after - direct if after is not None and direct is not None else None
            )
            row["bms_after_event_minus_normalized_soh_pp"] = (
                after - normalized if after is not None and normalized is not None else None
            )
        linked.append(row)
    return linked


def _continuity_metrics(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("cycle_continuity_class") == label]
    linked = [row for row in selected if row.get("linked_soh_event_id")]
    return {
        "cycle_rows": len(selected),
        "linked_soh_event_rows": len(linked),
        "linked_soh_event_rate": len(linked) / len(selected) if selected else None,
        "median_rest_minutes": _median_value(selected, "rest_minutes_inside_cycle"),
        "median_longest_rest_minutes": _median_value(selected, "longest_rest_minutes"),
    }


def _canonical_capacity_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = [row for row in rows if row.get("capacity_validation_status") == "validation_grade"]
    operational = [
        row
        for row in rows
        if row.get("capacity_validation_status") == "exploratory_soc_boundaries"
    ]
    return {
        "available": bool(rows),
        "reason": None if rows else "No qualified discharge-derived capacity point is available.",
        "cycle_rows": len(rows),
        "raw_resolved_cycle_rows": len(rows),
        "strict_full_boundary_rows": len(strict),
        "validation_grade_rows": len(strict),
        "linked_validation_grade_rows": sum(bool(row.get("linked_soh_event_id")) for row in strict),
        "median_discharged_ah": _median_value(strict, "discharged_ah_1s"),
        "median_capacity_soh_pct": _median_value(strict, "direct_capacity_soh_pct"),
        "operational_window_rows": len(operational),
        "linked_operational_window_rows": sum(
            bool(row.get("linked_soh_event_id")) for row in operational
        ),
        "median_operational_soc_span_pp": _median_value(operational, "raw_soc_drop_pct"),
        "median_soc_normalized_capacity_ah": _median_value(
            operational or strict, "soc_normalized_capacity_ah"
        ),
        "median_soc_normalized_capacity_soh_pct": _median_value(
            operational or strict, "soc_normalized_capacity_soh_pct"
        ),
        "bms_at_completion_error": _error_metrics(
            strict, "bms_completion_minus_capacity_soh_pp"
        ),
        "bms_after_linked_event_error": _error_metrics(
            strict, "bms_after_event_minus_capacity_soh_pp"
        ),
        "bms_at_completion_normalized_error": _error_metrics(
            operational, "bms_completion_minus_normalized_soh_pp"
        ),
        "bms_after_linked_event_normalized_error": _error_metrics(
            operational, "bms_after_event_minus_normalized_soh_pp"
        ),
        "continuity_comparison": {
            "continuous": _continuity_metrics(rows, "continuous"),
            "interrupted": _continuity_metrics(rows, "interrupted"),
        },
        "capacity_label_semantics": "Discharge-derived usable-capacity/SOH proxy from a strict 99-1% BMS SOC traversal.",
        "operational_window_semantics": "SOC-normalized 95-5% discharge evidence is exploratory, not laboratory capacity ground truth.",
        "claim_limit": "Current integration and SOC boundaries are BMS-derived. The result tests BMS consistency and usable-capacity trend; it is not laboratory ground truth.",
    }


def _reference_comparison(root: Path, serial: str, summary: dict[str, Any]) -> dict[str, Any]:
    is_nca = str(summary.get("chemistry") or "").upper() == "NCA"
    if not is_nca:
        return {
            "available": False,
            "reason": "NCA and LFP reference comparisons are kept separate.",
            "reference_serial": "100000017",
        }
    reference_run = _latest_run(root, "100000017")
    if reference_run is None or not (reference_run / "model_summary.json").exists():
        return {
            "available": False,
            "reason": "Run the current recalibration analysis for reference device 100000017.",
            "reference_serial": "100000017",
        }
    reference = _read_summary(reference_run)
    if (
        reference.get("detector_version") != summary.get("detector_version")
        or reference.get("raw_discharge_detector_version")
        != summary.get("raw_discharge_detector_version")
    ):
        return {
            "available": False,
            "reason": (
                "Selected and reference devices must use the same raw-discharge "
                "detector version."
            ),
            "reference_serial": "100000017",
        }
    metric_keys = (
        "event_rows",
        "step_drop_rows",
        "step_recovery_rows",
        "net_bms_soh_change_pp",
        "event_rate_per_observed_year",
        "event_rate_per_1000_discharge_ah",
        "exact_transition_rate",
        "median_context_coverage_fraction",
    )
    return {
        "available": True,
        "reason": None,
        "reference_serial": "100000017",
        "is_reference_device": serial == "100000017",
        "comparison_scope": (
            "matched_reference_configuration"
            if int(summary.get("module_count") or 0) == 2
            else "exploratory_normalized_cross_module"
        ),
        "selected": {key: summary.get(key) for key in metric_keys},
        "reference": {key: reference.get(key) for key in metric_keys},
        "reference_run_id": reference_run.name,
        "claim_limit": (
            "Two-module NCA is configuration-matched. Other NCA module counts are exploratory, "
            "normalized comparisons and not validated degradation transfers. This compares BMS "
            "estimator-event behavior, not independent physical capacity."
        ),
    }


def _downsample(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    step = (len(rows) - 1) / (max_points - 1)
    indexes = sorted({round(index * step) for index in range(max_points)})
    return [rows[index] for index in indexes]


def _normalise_chemistry(value: str) -> str:
    chemistry = value.strip().upper()
    if chemistry not in {"NCA", "LFP"}:
        raise ValueError("chemistry must be NCA or LFP")
    return chemistry


def _capacity_bms_validation_ready(
    source_root: Path, chemistry: str, serial: str
) -> bool:
    summaries = sorted(
        (
            source_root
            / "data"
            / "processed"
            / "data_validation"
            / f"chemistry={chemistry}"
            / f"serial={serial}"
        ).glob("validation_run=*/model_summary.json")
    )
    if not summaries:
        return False
    try:
        summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        summary.get("readiness", {}).get("capacity_bms", {}).get("status")
        == "ready"
    )


def build_bms_soh_recalibration_overview(
    *,
    source_root: Path,
    devices: list[dict[str, Any]],
    chemistry: str,
) -> dict[str, Any]:
    chemistry_key = _normalise_chemistry(chemistry)
    root = source_root / "data" / "processed" / "bms_soh_recalibration"
    rows: list[dict[str, Any]] = []
    latest_modified_at: datetime | None = None
    for device in devices:
        if str(device.get("chemistry") or "").upper() != chemistry_key:
            continue
        serial = str(device["serial"])
        run_dir = _latest_run(root, serial)
        summary: dict[str, Any] = {}
        source_modified_at = None
        if run_dir and (run_dir / "model_summary.json").exists():
            summary = _read_summary(run_dir)
            modified = datetime.fromtimestamp(
                (run_dir / "model_summary.json").stat().st_mtime, tz=UTC
            )
            latest_modified_at = (
                max(latest_modified_at, modified) if latest_modified_at else modified
            )
            source_modified_at = modified.isoformat()
        has_prerequisites = bool(
            (device.get("readiness") or {}).get("hierarchical_features")
            and (device.get("readiness") or {}).get("soh_signal_audit")
            and _capacity_bms_validation_ready(source_root, chemistry_key, serial)
        )
        rows.append(
            {
                "serial": serial,
                "display_name": device.get("display_name") or f"Battery {serial}",
                "variant_key": device.get("variant_key"),
                "device_role": device.get("device_role"),
                "module_count": device.get("module_count"),
                "module_count_status": device.get("module_count_status"),
                "nominal_capacity_ah": device.get("nominal_capacity_ah"),
                "prerequisites_ready": has_prerequisites,
                "analyzed": bool(summary),
                "source_run_id": run_dir.name if run_dir else None,
                "source_modified_at": source_modified_at,
                "observed_start": summary.get("observed_start"),
                "observed_end": summary.get("observed_end"),
                "detector_version": summary.get("detector_version"),
                "event_rows": int(summary.get("event_rows") or 0),
                "step_drop_rows": int(summary.get("step_drop_rows") or 0),
                "step_recovery_rows": int(summary.get("step_recovery_rows") or 0),
                "change_after_gap_rows": int(summary.get("change_after_gap_rows") or 0),
                "high_confidence_rows": int(summary.get("high_confidence_rows") or 0),
                "strong_context_match_rows": int(
                    summary.get("strong_context_match_rows")
                    or summary.get("high_confidence_rows")
                    or 0
                ),
                "exact_transition_rows": int(summary.get("exact_transition_rows") or 0),
                "daily_fallback_transition_rows": int(
                    summary.get("daily_fallback_transition_rows") or 0
                ),
                "limited_context_rows": int(summary.get("limited_context_rows") or 0),
                "classification_counts": summary.get("classification_counts") or {},
            }
        )
    rows.sort(
        key=lambda item: (
            item["module_count"] is None,
            item["module_count"] or 0,
            item["serial"],
        )
    )
    analyzed = [item for item in rows if item["analyzed"]]
    return {
        "model_family": "bms_soh_recalibration_overview",
        "chemistry": chemistry_key,
        "generated_at": datetime.now(UTC).isoformat(),
        "latest_source_modified_at": (
            latest_modified_at.isoformat() if latest_modified_at else None
        ),
        "metrics": {
            "configured_device_count": len(rows),
            "prerequisite_ready_device_count": sum(item["prerequisites_ready"] for item in rows),
            "analyzed_device_count": len(analyzed),
            "total_event_count": sum(item["event_rows"] for item in analyzed),
            "total_step_drop_count": sum(item["step_drop_rows"] for item in analyzed),
            "total_step_recovery_count": sum(item["step_recovery_rows"] for item in analyzed),
            "total_data_gap_risk_count": sum(
                int(item["classification_counts"].get("data_gap_or_estimator_reset_risk") or 0)
                for item in analyzed
            ),
            "high_confidence_event_count": sum(item["high_confidence_rows"] for item in analyzed),
            "strong_context_match_count": sum(
                item["strong_context_match_rows"] for item in analyzed
            ),
            "exact_transition_count": sum(item["exact_transition_rows"] for item in analyzed),
            "daily_fallback_transition_count": sum(
                item["daily_fallback_transition_rows"] for item in analyzed
            ),
            "limited_context_event_count": sum(item["limited_context_rows"] for item in analyzed),
        },
        "devices": rows,
        "evidence_policy": {
            "candidate_source": (
                "quality-screened daily candidates refined against persistent one-second SOH levels"
            ),
            "time_resolution": (
                "exact raw change point, interval-censored transition, or explicit daily fallback"
            ),
            "operating_evidence": (
                "compact five-minute BMS current, SOC, voltage-spread, temperature, "
                "and coverage windows"
            ),
            "claim_limit": (
                "Event/control comparisons describe temporal evidence enriched around "
                "BMS-reported SOH changes. "
                "They do not establish a firmware cause or independent physical capacity loss."
            ),
        },
    }


def build_bms_soh_recalibration_series(
    *,
    source_root: Path,
    serial: str,
    max_points: int = 1000,
) -> dict[str, Any] | None:
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    run_dir = _latest_run(
        source_root / "data" / "processed" / "bms_soh_recalibration", serial
    )
    if run_dir is None:
        return None
    summary_path = run_dir / "model_summary.json"
    timeline_path = run_dir / "soh_daily_timeline.parquet"
    events_path = run_dir / "recalibration_events.parquet"
    if not summary_path.exists() or not timeline_path.exists() or not events_path.exists():
        return None
    summary = _read_summary(run_dir)
    points = _read_parquet_rows(timeline_path)
    for point in points:
        point["time"] = point.pop("day_utc", None)
    events = _read_parquet_rows(events_path)
    for event in events:
        event["event_time"] = event.pop("event_time_utc", None)
    comparison = _read_parquet_rows(run_dir / "evidence_comparison.parquet")
    phase2_capacity_events = _read_parquet_rows(
        run_dir / "capacity_discharge_events.parquet"
    )
    phase2_soh_transitions = _read_parquet_rows(
        run_dir / "bms_soh_transitions.parquet"
    )
    phase2_discharge_matches = _read_parquet_rows(
        run_dir / "soh_discharge_matches.parquet"
    )
    phase2_match_candidates = _read_parquet_rows(
        run_dir / "event_match_candidates.parquet"
    )
    raw_discharge_trajectories = _read_parquet_rows(
        run_dir / "raw_discharge_trajectories.parquet"
    )
    downward_trajectory_matches = _read_parquet_rows(
        run_dir / "downward_transition_trajectory_matches.parquet"
    )
    transition_trajectory_candidates = _read_parquet_rows(
        run_dir / "transition_trajectory_candidates.parquet"
    )
    compact_endpoint_reconciliation = _read_parquet_rows(
        run_dir / "compact_endpoint_reconciliation.parquet"
    )
    phase3_root = (
        source_root
        / "data"
        / "processed"
        / "electrothermal_capacity_context"
        / f"chemistry={summary.get('chemistry', 'NCA')}"
        / f"serial={serial}"
    )
    phase3_runs = sorted(
        phase3_root.glob("phase3_run=*") if phase3_root.exists() else []
    )
    phase3_run = phase3_runs[-1] if phase3_runs else None
    electrothermal_context = (
        _read_parquet_rows(phase3_run / "event_electrothermal_context.parquet")
        if phase3_run is not None
        else []
    )
    electrothermal_summary = (
        _read_summary(phase3_run) if phase3_run is not None else None
    )
    capacity_validation = _read_parquet_rows(
        run_dir / "one_second_capacity_validation.parquet"
    )
    capacity_run = _latest_capacity_run(source_root, serial)
    if capacity_run is not None:
        canonical_rows = _read_parquet_rows(
            capacity_run / "qualified_capacity_points.parquet"
        )
        if canonical_rows:
            capacity_validation = _link_capacity_evidence(canonical_rows, events)
            summary["one_second_capacity_validation"] = _canonical_capacity_summary(
                capacity_validation
            )
            summary["source_capacity_evidence_run_id"] = capacity_run.name
    return {
        "serial": str(serial),
        "run_id": run_dir.name,
        "detector_version": summary.get("detector_version"),
        "summary": {
            key: summary.get(key)
            for key in (
                "chemistry",
                "module_count",
                "nominal_capacity_ah",
                "raw_discharge_detector_version",
                "observed_start",
                "observed_end",
                "feature_observed_start",
                "feature_observed_end",
                "soh_observed_start",
                "soh_observed_end",
                "analysis_overlap_start",
                "analysis_overlap_end",
                "aggregation_minutes",
                "current_sign_convention",
                "event_detection_method",
                "event_time_semantics",
                "event_rows",
                "step_drop_rows",
                "step_recovery_rows",
                "change_after_gap_rows",
                "high_confidence_rows",
                "strong_context_match_rows",
                "limited_context_rows",
                "exact_transition_rows",
                "interval_censored_transition_rows",
                "daily_fallback_transition_rows",
                "exact_transition_rate",
                "gross_step_drop_pp",
                "gross_step_recovery_pp",
                "net_bms_soh_change_pp",
                "event_rate_per_observed_year",
                "event_rate_per_1000_discharge_ah",
                "median_context_coverage_fraction",
                "matched_control_rows",
                "controls_per_event_requested",
                "classification_counts",
                "evidence_tag_counts",
                "strongest_matched_control_enrichment",
                "primary_discharge_hypothesis",
                "primary_discharge_hypothesis_method",
                "complete_discharge_event_rows",
                "qualified_calibration_discharge_rows",
                "one_second_capacity_validation",
                "raw_capacity_validation_scan",
                "interpretable_model",
                "raw_transition_scan",
                "findings",
                "control_method",
                "claim_limit",
                "configuration",
                "source_soh_audit_run_id",
                "source_soh_audit_label_status",
                "source_capacity_evidence_run_id",
                "phase2",
            )
        },
        "points": _downsample(points, max_points),
        "events": events,
        "comparison": comparison,
        "capacity_validation": capacity_validation,
        "capacity_events": phase2_capacity_events,
        "soh_transitions": phase2_soh_transitions,
        "discharge_matches": phase2_discharge_matches,
        "match_candidates": phase2_match_candidates,
        "raw_discharge_trajectories": raw_discharge_trajectories,
        "downward_trajectory_matches": downward_trajectory_matches,
        "transition_trajectory_candidates": transition_trajectory_candidates,
        "compact_endpoint_reconciliation": compact_endpoint_reconciliation,
        "electrothermal_context": electrothermal_context,
        "electrothermal_summary": electrothermal_summary,
        "reference_comparison": _reference_comparison(
            source_root / "data" / "processed" / "bms_soh_recalibration",
            str(serial),
            summary,
        ),
    }
