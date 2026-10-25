from __future__ import annotations

import json
import math
import statistics
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.analytics.nca_cohort import classify_nca_device


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _positive_int(value: Any) -> int | None:
    number = _positive(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _latest_verification_summary(source_root: Path, serial: str) -> Path | None:
    root = (
        source_root
        / "data"
        / "processed"
        / "fixed_slope_verification"
        / f"serial={serial}"
    )
    candidates = list(root.glob("verification_run=*/model_summary.json"))
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.parent.name),
    )


def _history_rows(run_dir: Path) -> list[dict[str, Any]]:
    history_path = run_dir / "fixed_slope_all_capacity_rows.parquet"
    if not history_path.exists():
        raise ValueError("fixed_slope_all_capacity_rows.parquet is missing")
    parquet = pq.ParquetFile(history_path)
    names = set(parquet.schema_arrow.names)
    required = {"end_anchor_time", "cumulative_all_discharge_ah"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"verification history is missing columns: {', '.join(missing)}")
    soh_column = "bms_soh_end" if "bms_soh_end" in names else "measured_soh_pct"
    if soh_column not in names:
        raise ValueError("verification history has no BMS SOH column")
    rows = parquet.read(
        columns=["end_anchor_time", "cumulative_all_discharge_ah", soh_column]
    ).to_pylist()
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        soh = _finite(row.get(soh_column))
        cumulative_ah = _finite(row.get("cumulative_all_discharge_ah"))
        timestamp = row.get("end_anchor_time")
        if timestamp is None or soh is None or cumulative_ah is None:
            continue
        cleaned.append(
            {
                "timestamp": timestamp,
                "bms_soh_pct": soh,
                "cumulative_discharge_ah": max(0.0, cumulative_ah),
            }
        )
    cleaned.sort(key=lambda row: _iso(row["timestamp"]) or "")
    if len(cleaned) < 2:
        raise ValueError("verification history has fewer than two usable observations")
    return cleaned


def _device_capacity_row(
    *,
    source_root: Path,
    device: dict[str, Any],
    reference_loss_slope_ah_per_1000ah: float,
    bms_tolerance_pct: float,
    step_threshold_pct: float,
    reference_module_count: int,
) -> dict[str, Any]:
    serial = str(device["serial"])
    summary_path = _latest_verification_summary(source_root, serial)
    if summary_path is None:
        raise ValueError("no fixed-slope verification run is available")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _history_rows(summary_path.parent)

    nominal_capacity_ah = _positive(device.get("nominal_capacity_ah")) or _positive(
        summary.get("nominal_capacity_ah")
    )
    if nominal_capacity_ah is None:
        raise ValueError("nominal capacity is missing")

    first = rows[0]
    latest = rows[-1]
    first_soh = float(first["bms_soh_pct"])
    latest_soh = float(latest["bms_soh_pct"])
    first_cumulative_ah = float(first["cumulative_discharge_ah"])
    cumulative_discharge_ah = float(latest["cumulative_discharge_ah"])
    throughput_span_ah = max(0.0, cumulative_discharge_ah - first_cumulative_ah)
    net_bms_soh_change_pct = latest_soh - first_soh

    reference_capacity_ah = max(
        0.0,
        nominal_capacity_ah
        - reference_loss_slope_ah_per_1000ah * cumulative_discharge_ah / 1000.0,
    )
    reference_soh_pct = 100.0 * reference_capacity_ah / nominal_capacity_ah
    bms_implied_capacity_ah = nominal_capacity_ah * latest_soh / 100.0
    bms_minus_reference_soh_pct = latest_soh - reference_soh_pct

    absolute_errors: list[float] = []
    within_tolerance = 0
    drops = 0
    recoveries = 0
    previous_soh: float | None = None
    for row in rows:
        observed_soh = float(row["bms_soh_pct"])
        predicted_soh = 100.0 * max(
            0.0,
            nominal_capacity_ah
            - reference_loss_slope_ah_per_1000ah
            * float(row["cumulative_discharge_ah"])
            / 1000.0,
        ) / nominal_capacity_ah
        error = observed_soh - predicted_soh
        absolute_errors.append(abs(error))
        within_tolerance += abs(error) <= bms_tolerance_pct
        if previous_soh is not None:
            delta = observed_soh - previous_soh
            drops += delta <= -step_threshold_pct
            recoveries += delta >= step_threshold_pct
        previous_soh = observed_soh

    bms_equivalent_loss_slope = (
        -net_bms_soh_change_pct
        * nominal_capacity_ah
        / 100.0
        / (throughput_span_ah / 1000.0)
        if throughput_span_ah > 0
        else None
    )
    module_count = _positive_int(device.get("module_count"))
    module_status = str(device.get("module_count_status") or "missing")
    coverage_pct = _finite((summary.get("data_quality") or {}).get("telemetry_coverage_pct"))
    flags: list[str] = []
    if module_status != "verified":
        flags.append("module_count_unverified")
    if coverage_pct is not None and coverage_pct < 90.0:
        flags.append("telemetry_coverage_below_90pct")
    if recoveries:
        flags.append("bms_soh_recovery_observed")
    if abs(bms_minus_reference_soh_pct) > bms_tolerance_pct:
        flags.append("outside_bms_tolerance")

    return {
        "serial": serial,
        "display_name": device.get("display_name") or f"Battery {serial}",
        "variant_key": device.get("variant_key"),
        "chemistry": device.get("chemistry"),
        "module_count": module_count,
        "module_count_status": module_status,
        "comparison_scope": (
            "matched_reference_configuration"
            if module_count == reference_module_count
            else "exploratory_efc_normalized_cross_module"
        ),
        "nominal_capacity_ah": nominal_capacity_ah,
        "capacity_per_module_ah": (
            nominal_capacity_ah / module_count if module_count else None
        ),
        "system_energy_kwh": _positive(device.get("system_energy_kwh")),
        "observed_start": _iso(first["timestamp"]),
        "observed_end": _iso(latest["timestamp"]),
        "observation_rows": len(rows),
        "first_bms_soh_pct": first_soh,
        "latest_bms_soh_pct": latest_soh,
        "net_bms_soh_change_pct": net_bms_soh_change_pct,
        "bms_implied_capacity_ah": bms_implied_capacity_ah,
        "cumulative_discharge_ah": cumulative_discharge_ah,
        "throughput_span_ah": throughput_span_ah,
        "equivalent_full_cycles": cumulative_discharge_ah / nominal_capacity_ah,
        "discharge_ah_per_module": (
            cumulative_discharge_ah / module_count if module_count else None
        ),
        "reference_loss_slope_ah_per_1000ah": reference_loss_slope_ah_per_1000ah,
        "reference_capacity_ah": reference_capacity_ah,
        "reference_soh_pct": reference_soh_pct,
        "bms_minus_reference_soh_pct": bms_minus_reference_soh_pct,
        "bms_minus_reference_capacity_ah": (
            bms_implied_capacity_ah - reference_capacity_ah
        ),
        "reference_mae_soh_pct": statistics.fmean(absolute_errors),
        "within_bms_tolerance_pct": 100.0 * within_tolerance / len(rows),
        "bms_equivalent_loss_slope_ah_per_1000ah": bms_equivalent_loss_slope,
        "step_drop_count": drops,
        "step_recovery_count": recoveries,
        "telemetry_coverage_pct": coverage_pct,
        "quality_flags": flags,
        "source_run_id": summary_path.parent.name,
        "source_summary_path": str(summary_path.resolve()),
        "source_modified_at": datetime.fromtimestamp(
            summary_path.stat().st_mtime, tz=UTC
        ).isoformat(),
    }


def build_nca_capacity_overview(
    *,
    source_root: Path,
    devices: list[dict[str, Any]],
    reference_loss_slope_ah_per_1000ah: float = 0.12,
    reference_serial: str = "100000017",
    reference_module_count: int = 2,
    bms_tolerance_pct: float = 1.0,
    step_threshold_pct: float = 0.5,
) -> dict[str, Any]:
    if reference_loss_slope_ah_per_1000ah <= 0:
        raise ValueError("reference loss slope must be positive")
    if reference_module_count <= 0:
        raise ValueError("reference module count must be positive")
    if bms_tolerance_pct < 0:
        raise ValueError("BMS tolerance must be nonnegative")
    if step_threshold_pct <= 0:
        raise ValueError("step threshold must be positive")

    classified_devices = []
    for device in devices:
        membership = classify_nca_device(
            device,
            active_module_count=reference_module_count,
            reference_serial=reference_serial,
            reference_module_count=reference_module_count,
        )
        if membership is None or membership.cohort_role != "field_device":
            continue
        classified_devices.append((device, membership))

    candidates = sorted(
        (
            device
            for device, membership in classified_devices
            if str(device.get("module_count_status") or "") == "verified"
            and membership.module_count is not None
            and _positive(device.get("nominal_capacity_ah")) is not None
        ),
        key=lambda device: str(device["serial"]),
    )
    deferred_devices = sorted(
        (
            {
                "serial": str(device["serial"]),
                "display_name": device.get("display_name") or f"Battery {device['serial']}",
                "variant_key": device.get("variant_key"),
                "module_count": membership.module_count,
                "module_count_status": str(
                    device.get("module_count_status") or "missing"
                ),
                "nominal_capacity_ah": _positive(device.get("nominal_capacity_ah")),
                "system_energy_kwh": _positive(device.get("system_energy_kwh")),
                "cohort_status": membership.cohort_status,
                "cohort_reason": membership.reason,
            }
            for device, membership in classified_devices
            if str(device.get("module_count_status") or "") != "verified"
            or membership.module_count is None
            or _positive(device.get("nominal_capacity_ah")) is None
        ),
        key=lambda device: device["serial"],
    )
    completed: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for device in candidates:
        serial = str(device["serial"])
        try:
            completed.append(
                _device_capacity_row(
                    source_root=source_root,
                    device=device,
                    reference_loss_slope_ah_per_1000ah=(
                        reference_loss_slope_ah_per_1000ah
                    ),
                    bms_tolerance_pct=bms_tolerance_pct,
                    step_threshold_pct=step_threshold_pct,
                    reference_module_count=reference_module_count,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            missing.append(
                {
                    "serial": serial,
                    "display_name": device.get("display_name") or f"Battery {serial}",
                    "reason": str(exc),
                }
            )

    reference_device = next(
        (device for device in devices if str(device.get("serial")) == reference_serial),
        None,
    )
    reference_display_name = (
        reference_device.get("display_name")
        if reference_device is not None
        else f"Battery {reference_serial}"
    )

    latest_soh_values = [float(row["latest_bms_soh_pct"]) for row in completed]
    net_change_values = [float(row["net_bms_soh_change_pct"]) for row in completed]
    gap_values = [float(row["bms_minus_reference_soh_pct"]) for row in completed]
    equivalent_slopes = [
        float(value)
        for row in completed
        if (value := row.get("bms_equivalent_loss_slope_ah_per_1000ah")) is not None
    ]
    coverage_values = [
        float(value)
        for row in completed
        if (value := row.get("telemetry_coverage_pct")) is not None
    ]
    newest_source = max(
        (str(row["source_modified_at"]) for row in completed),
        default=None,
    )

    return {
        "model_family": "nca_capacity_overview",
        "chemistry": "NCA",
        "generated_at": datetime.now(UTC).isoformat(),
        "latest_source_modified_at": newest_source,
        "cohort": {
            "device_role": "field_tester",
            "module_count": None,
            "module_count_status": "verified",
            "comparison_basis": (
                "NCA devices grouped by module count; cross-module results use nominal-capacity "
                "and equivalent-full-cycle normalization"
            ),
            "policy": (
                "Two-module NCA devices are directly matched to the reference configuration. "
                "Verified three- and four-module devices remain comparable only as explicitly "
                "exploratory EFC-normalized evidence until matching controls exist."
            ),
        },
        "field_inventory": {
            "total_device_count": len(classified_devices),
            "active_device_count": len(candidates),
            "deferred_device_count": len(deferred_devices),
        },
        "reference": {
            "serial": reference_serial,
            "display_name": reference_display_name,
            "module_count": reference_module_count,
            "loss_slope_ah_per_1000ah": reference_loss_slope_ah_per_1000ah,
            "source": "controlled_full_cycle_nca_device",
            "transfer_basis": (
                "The two-module benchmark is converted to SOH percentage points per equivalent "
                "full cycle for exploratory cross-module comparison. It is not a validated "
                "three- or four-module degradation law."
            ),
        },
        "metrics": {
            "inventory_device_count": len(classified_devices),
            "deferred_device_count": len(deferred_devices),
            "configured_device_count": len(candidates),
            "available_device_count": len(completed),
            "missing_device_count": len(missing),
            "total_cumulative_discharge_ah": sum(
                float(row["cumulative_discharge_ah"]) for row in completed
            ),
            "median_latest_bms_soh_pct": _median(latest_soh_values),
            "median_net_bms_soh_change_pct": _median(net_change_values),
            "median_bms_minus_reference_soh_pct": _median(gap_values),
            "median_bms_equivalent_loss_slope_ah_per_1000ah": _median(
                equivalent_slopes
            ),
            "median_telemetry_coverage_pct": _median(coverage_values),
            "devices_below_reference_by_more_than_tolerance": sum(
                float(row["bms_minus_reference_soh_pct"]) < -bms_tolerance_pct
                for row in completed
            ),
            "devices_above_twice_reference_loss_rate": sum(
                value > 2.0 * reference_loss_slope_ah_per_1000ah
                for value in equivalent_slopes
            ),
            "devices_with_bms_recoveries": sum(
                int(row["step_recovery_count"]) > 0 for row in completed
            ),
            "devices_with_verified_module_count": sum(
                row["module_count_status"] == "verified" for row in completed
            ),
        },
        "devices": completed,
        "missing_devices": missing,
        "deferred_devices": deferred_devices,
        "evidence_policy": {
            "throughput_source": "measured_all_discharge_events",
            "capacity_signal": "nominal_capacity_times_bms_soh",
            "bms_soh_role": "weak_auxiliary_label",
            "reference_curve_role": "transferred_nca_benchmark",
            "ground_truth_status": "independent_capacity_measurement_missing",
            "claim_limit": (
                "Differences quantify agreement with the BMS and the transferred reference; "
                "they do not prove physical capacity loss."
            ),
        },
    }
