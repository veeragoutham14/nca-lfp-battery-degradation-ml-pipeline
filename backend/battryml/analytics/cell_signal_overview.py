from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from battryml.analytics.nca_cohort import (
    DEFAULT_ACTIVE_MODULE_COUNT,
    DEFAULT_REFERENCE_MODULE_COUNT,
    DEFAULT_REFERENCE_SERIAL,
    classify_chemistry_device,
)

FRESHNESS_LIMIT_DAYS = 30.0


def _latest_summary(root: Path, serial: str, run_pattern: str) -> Path | None:
    serial_root = root / f"serial={serial}"
    summaries = sorted(serial_root.glob(f"{run_pattern}/model_summary.json"))
    return summaries[-1] if summaries else None


def _read_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _feature_coverage(
    root: Path,
    serial: str,
) -> tuple[Path | None, str | None, str | None]:
    """Return the run reaching furthest in time and total chronological bounds."""
    serial_root = root / f"serial={serial}"
    summaries = sorted(serial_root.glob("feature_run=*/model_summary.json"))
    if not summaries:
        return None, None, None

    starts: list[datetime] = []
    ends: list[tuple[datetime, Path]] = []
    for path in summaries:
        summary = _read_summary(path)
        start = _timestamp(summary.get("start"))
        end = _timestamp(summary.get("end"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append((end, path))

    if not ends:
        return summaries[-1], None, None
    coverage_end, coverage_path = max(ends, key=lambda item: item[0])
    coverage_start = min(starts) if starts else None
    return (
        coverage_path,
        coverage_start.isoformat() if coverage_start is not None else None,
        coverage_end.isoformat(),
    )


def _exclusive_source_end(observed_end: Any, sampled_end: Any) -> datetime | None:
    """Return an exclusive minute boundary that includes the final observed row."""
    observed = _timestamp(observed_end)
    if observed is not None:
        return observed.astimezone(UTC).replace(second=0, microsecond=0) + timedelta(minutes=1)
    sampled = _timestamp(sampled_end)
    return sampled.astimezone(UTC) if sampled is not None else None


def _floor_feature_boundary(value: datetime, aggregation_minutes: int) -> datetime:
    minute = (value.minute // aggregation_minutes) * aggregation_minutes
    return value.replace(minute=minute, second=0, microsecond=0)


def build_cell_signal_overview(
    *,
    source_root: Path,
    devices: list[dict[str, Any]],
    chemistry: str = "NCA",
    active_module_count: int = DEFAULT_ACTIVE_MODULE_COUNT,
) -> dict[str, Any]:
    processed_root = source_root / "data" / "processed"
    audit_root = processed_root / "bms_electrothermal_source_quality"
    feature_root = processed_root / "bms_hierarchical_features"
    chemistry_key = chemistry.strip().upper()
    rows: list[dict[str, Any]] = []
    latest_modified_at: datetime | None = None
    generated_at = datetime.now(UTC)

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
        role = device.get("device_role")
        module_status = str(device.get("module_count_status") or "missing")
        serial = str(device["serial"])
        summary_path = _latest_summary(audit_root, serial, "source_audit_run=*")
        source_contract = "hierarchical_electrothermal_v1"
        feature_summary_path, feature_start, feature_end = _feature_coverage(
            feature_root,
            serial,
        )
        feature_contract = "hierarchical_electrothermal_v1"
        if feature_summary_path is None:
            feature_contract = "missing"
        summary: dict[str, Any] = {}
        source_run_id = None
        source_modified_at = None
        if summary_path is not None:
            summary = _read_summary(summary_path)
            source_run_id = summary_path.parent.name
            modified = datetime.fromtimestamp(summary_path.stat().st_mtime, tz=UTC)
            source_modified_at = modified.isoformat()
            latest_modified_at = (
                max(latest_modified_at, modified) if latest_modified_at else modified
            )

        effective_module_count = membership.module_count
        expected_channels = summary.get(
            "expected_voltage_channels", summary.get("expected_cell_channels")
        )
        if expected_channels is None and effective_module_count is not None:
            expected_channels = int(effective_module_count) * 28
        sample_end = _timestamp(summary.get("sample_end"))
        freshness_lag_days = (
            (generated_at - sample_end).total_seconds() / 86_400.0
            if sample_end is not None
            else None
        )
        telemetry_fresh = bool(
            freshness_lag_days is not None and 0.0 <= freshness_lag_days <= FRESHNESS_LIMIT_DAYS
        )
        new_contract = source_contract == "hierarchical_electrothermal_v1"
        source_ready = bool(
            summary.get("source_quality_ready")
            if new_contract
            else summary.get("cell_voltage_ready", False)
        )
        expected_temperature_sensors = summary.get("expected_temperature_sensors")
        if expected_temperature_sensors is None and effective_module_count is not None:
            expected_temperature_sensors = int(effective_module_count) * 8
        initial_start = _timestamp(
            device.get("observed_start_at")
            or device.get("configured_start")
            or summary.get("sample_start")
        )
        source_end = _exclusive_source_end(
            device.get("observed_end_at"), summary.get("sample_end")
        )
        feature_end_timestamp = _timestamp(feature_end)
        aggregation_minutes = 5
        if (
            feature_summary_path is not None
            and feature_contract == "hierarchical_electrothermal_v1"
        ):
            feature_summary = _read_summary(feature_summary_path)
            aggregation_minutes = max(1, int(feature_summary.get("aggregation_minutes") or 5))

        source_modified = _timestamp(source_modified_at)
        if feature_contract != "hierarchical_electrothermal_v1" or feature_end_timestamp is None:
            feature_build_mode = "full"
            recommended_start = initial_start
        elif (
            (source_modified is not None and source_modified > feature_end_timestamp)
            or (source_end is not None and source_end > feature_end_timestamp)
        ):
            feature_build_mode = "incremental"
            recommended_start = _floor_feature_boundary(
                feature_end_timestamp.astimezone(UTC), aggregation_minutes
            )
        else:
            feature_build_mode = "up_to_date"
            recommended_start = None

        feature_lag_seconds = (
            max(0.0, (source_end - feature_end_timestamp).total_seconds())
            if source_end is not None and feature_end_timestamp is not None
            else None
        )
        rows.append(
            {
                "serial": serial,
                "display_name": device.get("display_name") or f"Battery {serial}",
                "variant_key": device.get("variant_key"),
                "device_role": role,
                "cohort_role": membership.cohort_role,
                "cohort_status": membership.cohort_status,
                "cohort_reason": membership.reason,
                "in_active_cohort": membership.in_active_cohort,
                "module_count": effective_module_count,
                "module_count_status": module_status,
                "configured_start": device.get("configured_start"),
                "observed_start": device.get("observed_start_at"),
                "observed_end": device.get("observed_end_at"),
                "audited": bool(summary),
                "cell_voltage_ready": source_ready,
                "cell_health_built": feature_summary_path is not None,
                "feature_contract": feature_contract,
                "source_contract": source_contract,
                "quality_gate_status": summary.get("quality_gate_status"),
                "blocking_reasons": summary.get("blocking_reasons") or [],
                "quality_warnings": summary.get("quality_warnings") or [],
                "telemetry_fresh": telemetry_fresh,
                "freshness_lag_days": freshness_lag_days,
                "sample_start": summary.get("sample_start"),
                "sample_end": summary.get("sample_end"),
                "total_rows": summary.get("total_rows"),
                "nominal_1hz_coverage_pct": summary.get("nominal_1hz_coverage_pct"),
                "expected_cell_channels": expected_channels,
                "present_cell_channels": summary.get(
                    "present_cell_channels", summary.get("expected_voltage_channels")
                ),
                "ready_cell_channels": summary.get(
                    "ready_voltage_channels", summary.get("ready_cell_channels")
                ),
                "minimum_cell_channel_valid_coverage_pct": summary.get(
                    "minimum_voltage_channel_valid_coverage_pct",
                    summary.get("minimum_cell_channel_valid_coverage_pct"),
                ),
                "expected_temperature_sensors": expected_temperature_sensors,
                "active_temperature_sensors": summary.get(
                    "active_temperature_sensors",
                    summary.get("ready_temperature_sensors"),
                ),
                "inactive_temperature_sensors": summary.get(
                    "inactive_temperature_sensors", 0
                ),
                "ready_temperature_sensors": summary.get("ready_temperature_sensors"),
                "minimum_temperature_sensor_valid_coverage_pct": summary.get(
                    "minimum_temperature_sensor_valid_coverage_pct"
                ),
                "minimum_active_temperature_sensor_valid_coverage_pct": summary.get(
                    "minimum_active_temperature_sensor_valid_coverage_pct",
                    summary.get("minimum_temperature_sensor_valid_coverage_pct"),
                ),
                "global_cell_voltage_signals_ready": bool(
                    summary.get(
                        "voltage_extrema_reconciled",
                        summary.get("global_cell_voltage_signals_ready", False),
                    )
                ),
                "cell_temperature_signals_ready": bool(
                    summary.get(
                        "temperature_extrema_reconciled",
                        summary.get("cell_temperature_signals_ready", False),
                    )
                ),
                "feature_start": feature_start,
                "feature_end": feature_end,
                "recommended_feature_start": (
                    recommended_start.astimezone(UTC).isoformat()
                    if recommended_start is not None
                    else None
                ),
                "recommended_feature_end": (
                    source_end.isoformat() if source_end is not None else None
                ),
                "feature_build_mode": feature_build_mode,
                "feature_lag_seconds": feature_lag_seconds,
                "feature_run_id": (
                    feature_summary_path.parent.name if feature_summary_path is not None else None
                ),
                "source_run_id": source_run_id,
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
    inventory_field_rows = [item for item in rows if item["cohort_role"] == "field_device"]
    active_rows = [item for item in rows if item["in_active_cohort"]]
    field_rows = [item for item in active_rows if item["cohort_role"] == "field_device"]
    deferred_field_rows = [item for item in inventory_field_rows if not item["in_active_cohort"]]
    reference_rows = [item for item in active_rows if item["cohort_role"] == "reference_control"]
    audited = [item for item in active_rows if item["audited"]]
    ready = [item for item in active_rows if item["cell_voltage_ready"]]
    fresh_ready = [
        item for item in active_rows if item["cell_voltage_ready"] and item["telemetry_fresh"]
    ]
    return {
        "model_family": "cell_signal_fleet_overview",
        "chemistry": chemistry_key,
        "generated_at": generated_at.isoformat(),
        "latest_source_modified_at": (
            latest_modified_at.isoformat() if latest_modified_at else None
        ),
        "cohort": {
            "active_module_count": active_module_count,
            "reference_serial": DEFAULT_REFERENCE_SERIAL,
            "reference_module_count": DEFAULT_REFERENCE_MODULE_COUNT,
            "policy": (
                f"All configured {chemistry_key} module variants remain visible; "
                "metrics include every device with a verified module count."
            ),
        },
        "metrics": {
            "inventory_device_count": len(rows),
            "configured_device_count": len(active_rows),
            "field_device_count": len(field_rows),
            "all_field_device_count": len(inventory_field_rows),
            "deferred_field_device_count": len(deferred_field_rows),
            "reference_device_count": len(reference_rows),
            "audited_device_count": len(audited),
            "ready_device_count": len(ready),
            "fresh_ready_device_count": len(fresh_ready),
            "stale_device_count": sum(
                item["audited"] and not item["telemetry_fresh"] for item in active_rows
            ),
            "pending_device_count": len(active_rows) - len(audited),
            "cell_health_built_device_count": sum(
                item["cell_health_built"] for item in active_rows
            ),
            "hierarchical_history_built_device_count": sum(
                item["feature_contract"] == "hierarchical_electrothermal_v1"
                for item in active_rows
            ),
            "verified_module_count_devices": sum(
                item["module_count_status"] == "verified" for item in active_rows
            ),
            "total_expected_active_cell_channels": sum(
                int(item["expected_cell_channels"] or 0) for item in active_rows
            ),
            "total_ready_active_cell_channels": sum(
                int(item["ready_cell_channels"] or 0) for item in active_rows
            ),
            "total_expected_active_temperature_sensors": sum(
                int(item["expected_temperature_sensors"] or 0) for item in active_rows
            ),
            "total_ready_active_temperature_sensors": sum(
                int(item["ready_temperature_sensors"] or 0) for item in active_rows
            ),
        },
        "devices": rows,
    }
