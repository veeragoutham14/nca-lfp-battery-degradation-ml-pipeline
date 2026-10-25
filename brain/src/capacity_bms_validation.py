from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.io_utils import write_parquet_chunk

MODEL_FAMILY = "capacity_bms_validation"
PHASE2_VERSION = "v4"
REQUIRED_RECALIBRATION_DETECTOR_VERSION = "v4"
REFERENCE_SERIAL_BY_CHEMISTRY = {"NCA": "100000017"}
QUALITY_PRIORITY = {"high": 0, "medium": 1, "low": 2, "indeterminate": 3}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_run(root: Path, serial: str, prefix: str) -> Path | None:
    runs = sorted((root / f"serial={serial}").glob(f"{prefix}=*"))
    return runs[-1] if runs else None


def _latest_stage1_run(
    validation_root: Path, chemistry: str, serial: str
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    serial_root = validation_root / f"chemistry={chemistry}" / f"serial={serial}"
    runs = sorted(serial_root.glob("validation_run=*"))
    if not runs:
        raise ValueError(f"Stage 1 data validation is missing for {serial}")
    run = runs[-1]
    summary = _read_json(run / "model_summary.json")
    manifest = _read_json(run / "input_manifest.json")
    readiness = summary.get("readiness", {}).get("capacity_bms", {})
    if readiness.get("status") != "ready":
        reasons = readiness.get("reasons") or ["capacity_bms_not_ready"]
        raise ValueError(
            f"Stage 1 capacity/BMS readiness is not ready for {serial}: "
            + ", ".join(map(str, reasons))
        )
    return run, summary, manifest


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _integer(value: Any, default: int = 0) -> int:
    finite = _finite(value)
    return int(finite) if finite is not None else default


def _evidence_class(start_soc: Any, end_soc: Any, resolved: bool) -> str:
    start = _finite(start_soc)
    end = _finite(end_soc)
    if start is None or end is None:
        return "none" if not resolved else "partial_discharge"
    span = start - end
    if start >= 100.0 and end <= 0.0 and span >= 100.0:
        return "strict_full_discharge"
    if start >= 99.0 and end <= 1.0 and span >= 98.0:
        return "near_full_discharge"
    if start >= 95.0 and end <= 5.0 and span >= 90.0:
        return "operational_full_discharge"
    return "partial_discharge"


def _measurement_quality(row: pd.Series) -> tuple[str, list[str]]:
    evidence = str(row.get("evidence_class") or "none")
    coverage = _finite(row.get("data_coverage_fraction"))
    maximum_gap = _finite(row.get("maximum_gap_s"))
    charge_intrusion = _finite(row.get("charge_intrusion_ah"))
    resolved = pd.notna(row.get("start_utc")) and pd.notna(row.get("end_utc"))
    reasons: list[str] = []
    if not resolved:
        return "indeterminate", ["raw_boundaries_not_resolved"]
    reasons.append(f"evidence_class:{evidence}")
    if coverage is not None:
        reasons.append(f"coverage:{coverage:.4f}")
    if maximum_gap is not None:
        reasons.append(f"maximum_gap_s:{maximum_gap:.3f}")
    if charge_intrusion is not None:
        reasons.append(f"charge_intrusion_ah:{charge_intrusion:.5f}")
    if evidence in {"strict_full_discharge", "near_full_discharge"}:
        reasons.append("complete_soc_traversal")
    elif evidence == "operational_full_discharge":
        reasons.append("supporting_soc_traversal")
    else:
        reasons.append("incomplete_soc_traversal")
    high = (
        evidence in {"strict_full_discharge", "near_full_discharge"}
        and coverage is not None
        and coverage >= 0.98
        and maximum_gap is not None
        and maximum_gap <= 5.0
        and charge_intrusion is not None
        and charge_intrusion <= 0.1
    )
    medium = (
        evidence
        in {
            "strict_full_discharge",
            "near_full_discharge",
            "operational_full_discharge",
        }
        and coverage is not None
        and coverage >= 0.90
        and charge_intrusion is not None
        and charge_intrusion <= 0.25
    )
    if high:
        return "high", reasons
    if medium:
        reasons.append("review_before_reference_comparison")
        return "medium", reasons
    reasons.append("material_measurement_limitation")
    return "low", reasons


def _canonical_discharge_events(
    episodes: pd.DataFrame,
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    reference_capacity_ah: float,
    reference_capacity_source: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, source in episodes.reset_index(drop=True).iterrows():
        source_resolution = str(source.get("phase2_source_resolution") or "one_second")
        raw_resolved = source_resolution == "one_second"
        start_value = (
            source.get("raw_start_time_utc") if raw_resolved else source.get("start_time_utc")
        )
        end_value = (
            source.get("raw_end_time_utc")
            if raw_resolved
            else source.get("completion_time_utc") or source.get("end_time_utc")
        )
        start = pd.to_datetime(start_value, errors="coerce", utc=True)
        end = pd.to_datetime(end_value, errors="coerce", utc=True)
        resolved = pd.notna(start) and pd.notna(end)
        start_soc = source.get("raw_start_soc_pct") if raw_resolved else source.get("start_soc_pct")
        end_soc = source.get("raw_end_soc_pct") if raw_resolved else source.get("end_soc_pct")
        evidence_class = _evidence_class(start_soc, end_soc, resolved)
        discharge_ah = _finite(
            source.get("discharged_ah_1s") if raw_resolved else source.get("discharged_ah")
        )
        charge_intrusion_ah = _finite(source.get("charge_intrusion_ah_1s")) or 0.0
        net_discharge_ah = (
            discharge_ah - charge_intrusion_ah if discharge_ah is not None else None
        )
        soc_span = _finite(
            source.get("raw_soc_drop_pct") if raw_resolved else source.get("soc_drop_pct")
        )
        direct_proxy = (
            net_discharge_ah / reference_capacity_ah * 100.0
            if raw_resolved
            and evidence_class == "strict_full_discharge"
            and net_discharge_ah is not None
            else None
        )
        normalized_proxy = (
            net_discharge_ah / (soc_span / 100.0) / reference_capacity_ah * 100.0
            if raw_resolved
            and net_discharge_ah is not None
            and soc_span is not None
            and soc_span >= 90.0
            else None
        )
        active_seconds = (_finite(source.get("active_discharge_minutes")) or 0.0) * 60.0
        record = {
            "device": str(serial),
            "chemistry": chemistry,
            "module_count": int(module_count),
            "discharge_event_id": source.get("discharge_episode_id")
            or f"discharge_{index + 1:06d}",
            "evidence_class": evidence_class,
            "measurement_resolution": source_resolution,
            "meaningful_deep_discharge": raw_resolved
            and soc_span is not None
            and soc_span >= 80.0,
            "capacity_claim_eligible": raw_resolved
            and soc_span is not None
            and soc_span >= 90.0,
            "capacity_estimate_eligible": raw_resolved
            and soc_span is not None
            and soc_span >= 90.0,
            "start_utc": start if resolved else None,
            "end_utc": end if resolved else None,
            "duration_s": (end - start).total_seconds() if resolved else None,
            "soc_start_pct": _finite(start_soc),
            "soc_end_pct": _finite(end_soc),
            "soc_span_pp": soc_span,
            "discharged_ah": discharge_ah,
            "gross_discharged_ah": discharge_ah,
            "net_discharged_ah": net_discharge_ah,
            "discharged_wh": _finite(source.get("discharged_wh_1s")),
            "charge_intrusion_ah": charge_intrusion_ah,
            "charge_intrusion_wh": _finite(source.get("charge_intrusion_wh_1s")),
            "mean_discharge_current_a": (
                discharge_ah * 3600.0 / active_seconds
                if discharge_ah is not None and active_seconds > 0
                else None
            ),
            "rms_current_a": _finite(source.get("rms_discharge_current_a")),
            "p95_current_a": _finite(source.get("p95_discharge_current_a")),
            "p99_current_a": _finite(source.get("p99_discharge_current_a")),
            "max_discharge_current_a": _finite(source.get("peak_discharge_current_a")),
            "rms_c_rate": _finite(source.get("rms_discharge_c_rate")),
            "p95_c_rate": (
                _finite(source.get("p95_discharge_current_a")) / reference_capacity_ah
                if _finite(source.get("p95_discharge_current_a")) is not None
                else None
            ),
            "data_coverage_fraction": _finite(
                source.get("raw_coverage_fraction")
                if raw_resolved
                else source.get("episode_coverage_fraction")
            ),
            "maximum_gap_s": _finite(source.get("maximum_raw_gap_seconds")),
            "missing_duration_s": (
                (_finite(source.get("unobserved_duration_minutes")) or 0.0) * 60.0
            ),
            "observed_discharge_ah": discharge_ah,
            "missing_ah_lower_bound": 0.0,
            "missing_ah_upper_bound": None,
            "missing_ah_method": "not_available_without_source_backed_current_limit",
            "legacy_missing_ah_screening_estimate": _finite(
                source.get("unobserved_discharge_ah_estimate")
            ),
            "rest_duration_s": (_finite(source.get("rest_minutes_inside_cycle")) or 0.0) * 60.0,
            "longest_rest_s": (_finite(source.get("longest_rest_minutes")) or 0.0) * 60.0,
            "rest_interruption_count": _integer(source.get("rest_interruption_count")),
            "discharge_current_dominance_fraction": _finite(
                source.get("discharge_dominance_fraction_1s")
            ),
            "temperature_mean_c": _finite(source.get("pack_temperature_mean_c")),
            "temperature_median_c": _finite(source.get("pack_temperature_median_c")),
            "temperature_min_c": _finite(source.get("pack_temperature_min_c")),
            "temperature_max_c": _finite(source.get("pack_temperature_max_c")),
            "temperature_coverage_fraction": _finite(
                source.get("pack_temperature_context_coverage_fraction")
            ),
            "sys_cycles_start": _finite(source.get("sys_cycles_start")),
            "sys_cycles_end": _finite(source.get("sys_cycles_end")),
            "sys_cycles_delta": _finite(source.get("sys_cycles_delta")),
            "reference_capacity_ah": reference_capacity_ah,
            "reference_capacity_source": reference_capacity_source,
            "estimated_full_capacity_ah": (
                normalized_proxy / 100.0 * reference_capacity_ah
                if normalized_proxy is not None
                else None
            ),
            "calculated_soh_pct": normalized_proxy,
            "discharge_derived_soh_proxy_pct": direct_proxy,
            "soc_span_normalized_soh_estimate_pct": normalized_proxy,
            "capacity_above_reference": (
                discharge_ah > reference_capacity_ah * 1.03 if discharge_ah is not None else None
            ),
            "source_capacity_validation_status": source.get("capacity_validation_status"),
            "provenance": (
                "bounded one-second BMS current integration; positive=charge; negative=discharge"
                if raw_resolved
                else "five-minute candidate inventory; context only; no capacity claim"
            ),
        }
        quality, reasons = _measurement_quality(pd.Series(record))
        if not raw_resolved:
            quality = "indeterminate"
            reasons.append("five_minute_candidate_requires_one_second_resolution")
        if record["capacity_above_reference"]:
            reasons.append("capacity_above_reference_103pct")
            quality = "low"
        record["measurement_quality"] = quality
        record["measurement_quality_reasons"] = reasons
        records.append(record)
    return pd.DataFrame.from_records(records)


def _complete_episode_inventory(
    candidates: pd.DataFrame,
    resolved_episodes: pd.DataFrame,
) -> pd.DataFrame:
    """Retain all compact candidates while preferring one-second evidence by ID."""
    resolved = resolved_episodes.copy()
    resolved["phase2_source_resolution"] = "one_second"
    if candidates.empty:
        return resolved
    resolved_ids = set(resolved["discharge_episode_id"].dropna().astype(str))
    compact_only = candidates.loc[
        ~candidates["discharge_episode_id"].astype(str).isin(resolved_ids)
    ].copy()
    compact_only["phase2_source_resolution"] = "five_minute_candidate"
    return pd.concat([resolved, compact_only], ignore_index=True, sort=False)


def _canonical_transitions(events: pd.DataFrame, serial: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, source in events.reset_index(drop=True).iterrows():
        delta = _finite(source.get("soh_delta_pp"))
        detection = str(source.get("transition_detection_status") or "approximate")
        timestamp_quality = (
            "exact"
            if detection == "exact"
            else "bounded"
            if detection in {"interval_censored", "bounded"}
            else "approximate"
        )
        records.append(
            {
                "device": str(serial),
                "transition_id": source.get("event_id") or f"soh_transition_{index + 1:05d}",
                "previous_soh_pct": _finite(source.get("previous_soh_pct")),
                "new_soh_pct": _finite(source.get("new_soh_pct")),
                "delta_soh_pp": delta,
                "direction": "down" if delta is not None and delta < 0 else "up",
                "transition_time_utc": pd.to_datetime(
                    source.get("event_time_utc"), errors="coerce", utc=True
                ),
                "timestamp_quality": timestamp_quality,
                "previous_stable_time_utc": pd.to_datetime(
                    source.get("previous_stable_time_utc"), errors="coerce", utc=True
                ),
                "new_stable_time_utc": pd.to_datetime(
                    source.get("new_stable_time_utc"), errors="coerce", utc=True
                ),
                "data_coverage_fraction": _finite(source.get("context_coverage_fraction")),
                "provenance": "persistent or interval-censored BMS SOH transition",
            }
        )
    return pd.DataFrame.from_records(records)


def _reference_comparability(
    events: pd.DataFrame,
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    reference_events: pd.DataFrame | None,
) -> pd.DataFrame:
    result = events.copy()
    columns = {
        "reference_comparability": [],
        "reference_comparability_reasons": [],
        "comparison_reference_device": [],
        "comparison_reference_event_id": [],
        "comparison_scope": [],
    }
    reference_serial = REFERENCE_SERIAL_BY_CHEMISTRY.get(chemistry)
    for _, event in result.iterrows():
        reasons: list[str] = []
        selected: pd.Series | None = None
        scope = "not_available"
        if not bool(event.get("capacity_estimate_eligible")):
            status = "indeterminate"
            reasons.append("not_one_second_capacity_evidence")
        elif reference_serial is None:
            status = "indeterminate"
            reasons.append("chemistry_reference_not_configured")
        elif serial == reference_serial:
            status = "directly_comparable"
            reasons.append("reference_device_event")
            scope = "reference_device"
        elif reference_events is None or reference_events.empty:
            status = "indeterminate"
            reasons.append("reference_capacity_events_missing")
        else:
            if "capacity_estimate_eligible" in reference_events:
                eligible_reference = reference_events["capacity_estimate_eligible"].fillna(
                    False
                )
            elif "capacity_claim_eligible" in reference_events:
                eligible_reference = reference_events["capacity_claim_eligible"].fillna(False)
            else:
                eligible_reference = pd.to_numeric(
                    reference_events.get("soc_span_pp"), errors="coerce"
                ).ge(90.0)
            candidates = reference_events.loc[
                eligible_reference
                & pd.to_numeric(reference_events["soc_span_pp"], errors="coerce").sub(
                    _finite(event.get("soc_span_pp")) or 0.0
                ).abs().le(5.0)
            ].copy()
            if candidates.empty:
                status = "needs_electrothermal_review"
                reasons.append("matching_reference_soc_class_missing")
            else:
                event_temp = _finite(event.get("temperature_mean_c"))
                event_c_rate = _finite(event.get("rms_c_rate"))
                candidates["_distance"] = 0.0
                if event_temp is not None:
                    candidates["_distance"] += (
                        pd.to_numeric(candidates["temperature_mean_c"], errors="coerce")
                        .sub(event_temp)
                        .abs()
                        .fillna(20.0)
                        / 5.0
                    )
                if event_c_rate is not None:
                    candidates["_distance"] += (
                        pd.to_numeric(candidates["rms_c_rate"], errors="coerce")
                        .sub(event_c_rate)
                        .abs()
                        .fillna(2.0)
                        / 0.1
                    )
                selected = candidates.sort_values("_distance").iloc[0]
                temp_difference = (
                    abs(event_temp - float(selected["temperature_mean_c"]))
                    if event_temp is not None and pd.notna(selected.get("temperature_mean_c"))
                    else None
                )
                c_rate_difference = (
                    abs(event_c_rate - float(selected["rms_c_rate"]))
                    if event_c_rate is not None and pd.notna(selected.get("rms_c_rate"))
                    else None
                )
                same_modules = _integer(selected.get("module_count")) == module_count
                scope = (
                    "matched_reference_configuration"
                    if same_modules
                    else "exploratory_normalized_cross_module"
                )
                reasons.append(f"comparison_scope:{scope}")
                if temp_difference is not None:
                    reasons.append(f"mean_temperature_difference_c:{temp_difference:.2f}")
                if c_rate_difference is not None:
                    reasons.append(f"rms_c_rate_difference:{c_rate_difference:.3f}")
                if (
                    same_modules
                    and event.get("measurement_quality") == "high"
                    and temp_difference is not None
                    and temp_difference <= 5.0
                    and c_rate_difference is not None
                    and c_rate_difference <= 0.1
                ):
                    status = "directly_comparable"
                elif (
                    event.get("measurement_quality") in {"high", "medium"}
                    and temp_difference is not None
                    and temp_difference <= 10.0
                    and c_rate_difference is not None
                    and c_rate_difference <= 0.25
                ):
                    status = "comparable_with_caution"
                    reasons.append("phase3_electrothermal_review_recommended")
                else:
                    status = "needs_electrothermal_review"
                    reasons.append("operating_context_differs_from_reference")
        columns["reference_comparability"].append(status)
        columns["reference_comparability_reasons"].append(reasons)
        columns["comparison_reference_device"].append(reference_serial)
        columns["comparison_reference_event_id"].append(
            selected.get("discharge_event_id") if selected is not None else None
        )
        columns["comparison_scope"].append(scope)
    for key, values in columns.items():
        result[key] = values
    return result


def _build_matches(
    transitions: pd.DataFrame,
    discharges: pd.DataFrame,
    *,
    lookback_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_records: list[dict[str, Any]] = []
    match_records: list[dict[str, Any]] = []
    usable = discharges.loc[
        discharges["meaningful_deep_discharge"].fillna(False)
        & discharges["end_utc"].notna()
    ].copy()
    usable["end_utc"] = pd.to_datetime(usable["end_utc"], errors="coerce", utc=True)
    for _, transition in transitions.iterrows():
        transition_time = pd.Timestamp(transition["transition_time_utc"])
        candidates = (
            usable.loc[
                usable["end_utc"].le(transition_time)
                & usable["end_utc"].ge(
                    transition_time - pd.Timedelta(hours=lookback_hours)
                )
            ].copy()
            if transition.get("direction") == "down"
            else usable.iloc[0:0].copy()
        )
        if not candidates.empty:
            candidates["time_to_transition_h"] = (
                transition_time - pd.to_datetime(candidates["end_utc"], utc=True)
            ).dt.total_seconds() / 3600.0
            candidates["_quality_priority"] = candidates["measurement_quality"].map(
                QUALITY_PRIORITY
            ).fillna(QUALITY_PRIORITY["indeterminate"])
            evidence_order = candidates.sort_values(
                [
                    "soc_span_pp",
                    "soc_end_pct",
                    "_quality_priority",
                    "time_to_transition_h",
                ],
                ascending=[False, True, True, True],
            )
            proximity_order = candidates.sort_values("time_to_transition_h")
            evidence_rank = {index: rank + 1 for rank, index in enumerate(evidence_order.index)}
            proximity_rank = {index: rank + 1 for rank, index in enumerate(proximity_order.index)}
            for index, candidate in candidates.iterrows():
                candidate_records.append(
                    {
                        "transition_id": transition["transition_id"],
                        "discharge_event_id": candidate["discharge_event_id"],
                        "evidence_class": candidate["evidence_class"],
                        "measurement_quality": candidate["measurement_quality"],
                        "soc_start_pct": candidate.get("soc_start_pct"),
                        "soc_end_pct": candidate.get("soc_end_pct"),
                        "soc_span_pp": candidate.get("soc_span_pp"),
                        "gross_discharged_ah": candidate.get("gross_discharged_ah"),
                        "charge_intrusion_ah": candidate.get("charge_intrusion_ah"),
                        "net_discharged_ah": candidate.get("net_discharged_ah"),
                        "time_to_transition_h": candidate["time_to_transition_h"],
                        "within_12h": candidate["time_to_transition_h"] <= 12.0,
                        "within_24h": candidate["time_to_transition_h"] <= 24.0,
                        "within_48h": candidate["time_to_transition_h"] <= 48.0,
                        "within_72h": candidate["time_to_transition_h"] <= 72.0,
                        "evidence_rank": evidence_rank[index],
                        "proximity_rank": proximity_rank[index],
                        "is_primary_capacity_evidence": evidence_rank[index] == 1,
                        "is_nearest_qualifying_candidate": proximity_rank[index] == 1,
                    }
                )
            primary = evidence_order.iloc[0]
            nearest = proximity_order.iloc[0]
        else:
            primary = None
            nearest = None
        proxy = primary.get("discharge_derived_soh_proxy_pct") if primary is not None else None
        if primary is not None and pd.isna(proxy):
            proxy = primary.get("soc_span_normalized_soh_estimate_pct")
        match_records.append(
            {
                "transition_id": transition["transition_id"],
                "transition_time_utc": transition_time,
                "primary_preceding_discharge_event_id": (
                    primary.get("discharge_event_id") if primary is not None else None
                ),
                "nearest_qualifying_discharge_event_id": (
                    nearest.get("discharge_event_id") if nearest is not None else None
                ),
                "discharge_evidence_class": (
                    primary.get("evidence_class")
                    if primary is not None
                    else "no_qualifying_discharge"
                ),
                "discharge_end_to_soh_update_hours": (
                    primary.get("time_to_transition_h") if primary is not None else None
                ),
                "discharge_derived_capacity_ah": (
                    primary.get("net_discharged_ah") if primary is not None else None
                ),
                "preceding_discharge_start_soc_pct": (
                    primary.get("soc_start_pct") if primary is not None else None
                ),
                "preceding_discharge_end_soc_pct": (
                    primary.get("soc_end_pct") if primary is not None else None
                ),
                "preceding_discharge_soc_drop_pp": (
                    primary.get("soc_span_pp") if primary is not None else None
                ),
                "preceding_discharge_gross_ah": (
                    primary.get("gross_discharged_ah") if primary is not None else None
                ),
                "preceding_discharge_charge_intrusion_ah": (
                    primary.get("charge_intrusion_ah") if primary is not None else None
                ),
                "preceding_discharge_net_ah": (
                    primary.get("net_discharged_ah") if primary is not None else None
                ),
                "discharge_derived_soh_proxy_pct": proxy,
                "soh_proxy_semantics": (
                    "direct_100_to_0"
                    if primary is not None
                    and primary.get("evidence_class") == "strict_full_discharge"
                    else "soc_span_normalized_supporting_estimate"
                    if proxy is not None
                    else "not_available_for_span_below_90pp"
                    if primary is not None
                    else "not_available"
                ),
                "bms_soh_before_pct": transition.get("previous_soh_pct"),
                "bms_soh_after_pct": transition.get("new_soh_pct"),
                "bms_minus_capacity_proxy_pp": (
                    float(transition["new_soh_pct"]) - float(proxy)
                    if proxy is not None and pd.notna(transition.get("new_soh_pct"))
                    else None
                ),
                "measurement_quality": (
                    primary.get("measurement_quality") if primary is not None else "indeterminate"
                ),
                "reference_comparability": (
                    primary.get("reference_comparability")
                    if primary is not None
                    else "indeterminate"
                ),
                "sys_cycles_delta": (
                    primary.get("sys_cycles_delta") if primary is not None else None
                ),
                "interpretation": (
                    "The deepest meaningful discharge in the configured lookback preceded the "
                    "SOH update; temporal evidence is not causal proof."
                    if primary is not None
                    else "No qualifying discharge was observed in the configured lookback."
                ),
            }
        )
    return pd.DataFrame.from_records(match_records), pd.DataFrame.from_records(candidate_records)


def _configuration_identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_configuration(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    reference_capacity_ah: float,
    reference_capacity_source: str,
    stage1_fingerprint: str | None,
    recalibration_root: Path,
    physical_catalog_root: Path,
    lookback_hours: float,
) -> dict[str, Any]:
    reference_serial = REFERENCE_SERIAL_BY_CHEMISTRY.get(chemistry)
    catalog_run = _latest_run(physical_catalog_root, serial, "catalog_run")
    reference_catalog_run = (
        _latest_run(physical_catalog_root, reference_serial, "catalog_run")
        if reference_serial
        else None
    )
    recalibration_run = _latest_run(recalibration_root, serial, "recalibration_run")
    return {
        "phase2_version": PHASE2_VERSION,
        "stage1_fingerprint": stage1_fingerprint,
        "serial": str(serial),
        "chemistry": chemistry,
        "module_count": module_count,
        "reference_capacity_ah": reference_capacity_ah,
        "reference_capacity_source": reference_capacity_source,
        "lookback_hours": lookback_hours,
        "physical_catalog_run_id": catalog_run.name if catalog_run else None,
        "reference_physical_catalog_run_id": (
            reference_catalog_run.name if reference_catalog_run else None
        ),
        "recalibration_run_id": recalibration_run.name if recalibration_run else None,
        "discharge_detector": {
            "source": "one_second_derived_operating_events",
            "minimum_meaningful_soc_drop_pp": 80.0,
            "minimum_capacity_estimate_soc_drop_pp": 90.0,
            "meaningful_recharge_soc_recovery_pp": 1.0,
            "rest_preserved": True,
            "delay_buckets_hours": [12, 24, 48, 72],
        },
        "recalibration_detector_version": REQUIRED_RECALIBRATION_DETECTOR_VERSION,
    }


def find_cached_phase2_run(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    reference_capacity_ah: float,
    reference_capacity_source: str = "device_configuration",
    validation_root: Path,
    recalibration_root: Path,
    physical_catalog_root: Path,
    lookback_hours: float = 72.0,
) -> Path | None:
    _, _, validation_manifest = _latest_stage1_run(validation_root, chemistry, serial)
    config = _cache_configuration(
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
        reference_capacity_ah=reference_capacity_ah,
        reference_capacity_source=reference_capacity_source,
        stage1_fingerprint=validation_manifest.get("fingerprint_sha256"),
        recalibration_root=recalibration_root,
        physical_catalog_root=physical_catalog_root,
        lookback_hours=lookback_hours,
    )
    identity = _configuration_identity(config)
    runs = sorted((recalibration_root / f"serial={serial}").glob("recalibration_run=*"))
    for run in reversed(runs):
        manifest_path = run / "phase2_manifest.json"
        if (
            manifest_path.exists()
            and _read_json(manifest_path).get("configuration_sha256") == identity
        ):
            return run
    return None


def finalize_phase2(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    reference_capacity_ah: float,
    reference_capacity_source: str,
    validation_root: Path,
    recalibration_root: Path,
    physical_catalog_root: Path,
    lookback_hours: float = 72.0,
    compression: str | None = "zstd",
) -> Path:
    chemistry = chemistry.upper()
    validation_run, _, validation_manifest = _latest_stage1_run(validation_root, chemistry, serial)
    recalibration_run = _latest_run(recalibration_root, serial, "recalibration_run")
    catalog_run = _latest_run(physical_catalog_root, serial, "catalog_run")
    if catalog_run is None or recalibration_run is None:
        raise ValueError("Physical discharge catalog and BMS SOH recalibration runs are required")
    recalibration_summary = _read_json(recalibration_run / "model_summary.json")
    if recalibration_summary.get("detector_version") != REQUIRED_RECALIBRATION_DETECTOR_VERSION:
        raise ValueError(
            f"Phase 2 requires recalibration detector {REQUIRED_RECALIBRATION_DETECTOR_VERSION}; "
            f"found {recalibration_summary.get('detector_version')}"
        )
    catalog_path = catalog_run / "physical_discharge_cycles.parquet"
    transitions_path = recalibration_run / "recalibration_events.parquet"
    if not catalog_path.exists() or not transitions_path.exists():
        raise ValueError("Required Phase 2 source artifacts are missing")
    source_transitions = pd.read_parquet(transitions_path)
    discharges = pd.read_parquet(catalog_path)
    discharges["reference_capacity_source"] = reference_capacity_source
    reference_events: pd.DataFrame | None = None
    reference_serial = REFERENCE_SERIAL_BY_CHEMISTRY.get(chemistry)
    if reference_serial and reference_serial != serial:
        reference_catalog_run = _latest_run(
            physical_catalog_root, reference_serial, "catalog_run"
        )
        if reference_catalog_run is not None and (
            reference_catalog_run / "physical_discharge_cycles.parquet"
        ).exists():
            reference_events = pd.read_parquet(
                reference_catalog_run / "physical_discharge_cycles.parquet"
            )
    discharges = _reference_comparability(
        discharges,
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
        reference_events=reference_events,
    )
    transitions = _canonical_transitions(source_transitions, serial)
    matches, candidates = _build_matches(transitions, discharges, lookback_hours=lookback_hours)
    downward = transitions.loc[transitions["delta_soh_pp"].lt(0)].copy()
    write_parquet_chunk(
        discharges,
        recalibration_run / "capacity_discharge_events.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        transitions,
        recalibration_run / "bms_soh_transitions.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        matches,
        recalibration_run / "soh_discharge_matches.parquet",
        compression=compression,
    )
    write_parquet_chunk(
        candidates,
        recalibration_run / "event_match_candidates.parquet",
        compression=compression,
    )
    for obsolete_name in (
        "raw_discharge_trajectories.parquet",
        "downward_transition_trajectory_matches.parquet",
        "transition_trajectory_candidates.parquet",
        "compact_endpoint_reconciliation.parquet",
    ):
        obsolete_path = recalibration_run / obsolete_name
        if obsolete_path.exists():
            obsolete_path.unlink()
    config = _cache_configuration(
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
        reference_capacity_ah=reference_capacity_ah,
        reference_capacity_source=reference_capacity_source,
        stage1_fingerprint=validation_manifest.get("fingerprint_sha256"),
        recalibration_root=recalibration_root,
        physical_catalog_root=physical_catalog_root,
        lookback_hours=lookback_hours,
    )
    manifest = {
        "manifest_version": "1.0.0",
        "model_family": MODEL_FAMILY,
        "phase2_version": PHASE2_VERSION,
        "created_at_utc": datetime.now(UTC),
        "device": str(serial),
        "chemistry": chemistry,
        "source_validation_artifact": str(validation_run),
        "source_validation_fingerprint": validation_manifest.get("fingerprint_sha256"),
        "source_physical_catalog_run": catalog_run.name,
        "source_recalibration_run": recalibration_run.name,
        "configuration": config,
        "configuration_sha256": _configuration_identity(config),
        "artifacts": [
            "capacity_discharge_events.parquet",
            "bms_soh_transitions.parquet",
            "soh_discharge_matches.parquet",
            "event_match_candidates.parquet",
        ],
    }
    (recalibration_run / "phase2_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8"
    )
    summary_path = recalibration_run / "model_summary.json"
    summary = _read_json(summary_path)
    previous_phase2 = summary.get("phase2")
    if isinstance(previous_phase2, dict):
        previous_phase2.pop("full_discharges_without_soh_update_rows", None)
    summary["phase2"] = {
        "available": True,
        "model_family": MODEL_FAMILY,
        "phase2_version": PHASE2_VERSION,
        "source_validation_run_id": validation_run.name,
        "source_validation_fingerprint": validation_manifest.get("fingerprint_sha256"),
        "source_physical_catalog_run_id": catalog_run.name,
        "discharge_event_rows": int(len(discharges)),
        "capacity_claim_eligible_rows": int(discharges["capacity_claim_eligible"].sum()),
        "strict_full_discharge_rows": int(
            (
                discharges["capacity_claim_eligible"]
                & discharges["evidence_class"].eq("strict_full_discharge")
            ).sum()
        ),
        "near_full_discharge_rows": int(
            (
                discharges["capacity_claim_eligible"]
                & discharges["evidence_class"].eq("near_full_discharge")
            ).sum()
        ),
        "operational_full_discharge_rows": int(
            (
                discharges["capacity_claim_eligible"]
                & discharges["evidence_class"].eq("operational_full_discharge")
            ).sum()
        ),
        "meaningful_80pp_discharge_rows": int(
            discharges["meaningful_deep_discharge"].fillna(False).sum()
        ),
        "capacity_estimate_90pp_rows": int(
            discharges["capacity_estimate_eligible"].fillna(False).sum()
        ),
        "context_only_discharge_rows": int((~discharges["capacity_claim_eligible"]).sum()),
        "high_quality_capacity_rows": int(discharges["measurement_quality"].eq("high").sum()),
        "soh_transition_rows": int(len(transitions)),
        "downward_soh_transition_rows": int(len(downward)),
        "downward_transitions_with_raw_trajectory_rows": int(
            matches.get(
                "primary_preceding_discharge_event_id", pd.Series(dtype=object)
            ).notna().sum()
        ),
        "raw_trajectory_rows": int(len(discharges)),
        "raw_trajectory_evidence_within_12h_rows": int(
            candidates.get("within_12h", pd.Series(dtype=bool)).groupby(
                candidates.get("transition_id", pd.Series(dtype=str))
            ).any().sum()
        ),
        "raw_trajectory_evidence_within_24h_rows": int(
            candidates.get("within_24h", pd.Series(dtype=bool)).groupby(
                candidates.get("transition_id", pd.Series(dtype=str))
            ).any().sum()
        ),
        "raw_trajectory_evidence_within_48h_rows": int(
            candidates.get("within_48h", pd.Series(dtype=bool)).groupby(
                candidates.get("transition_id", pd.Series(dtype=str))
            ).any().sum()
        ),
        "raw_trajectory_evidence_within_72h_rows": int(
            candidates.get("within_72h", pd.Series(dtype=bool)).groupby(
                candidates.get("transition_id", pd.Series(dtype=str))
            ).any().sum()
        ),
        "transitions_with_preceding_discharge_rows": int(
            matches["primary_preceding_discharge_event_id"].notna().sum()
            if not matches.empty
            else 0
        ),
        "qualifying_discharges_without_soh_update_rows": int(
            (
                discharges["meaningful_deep_discharge"]
                & ~discharges["discharge_event_id"].isin(candidates.get("discharge_event_id", []))
            ).sum()
        ),
        "claim_limit": (
            "Phase 2 reports event-derived >=80 pp physical discharges, >=90 pp "
            "SOC-normalized usable-capacity estimates, and temporal precedence. It does "
            "not establish laboratory capacity ground truth or causal recalibration."
        ),
    }
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return recalibration_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finalize canonical Phase 2 capacity evidence.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--reference-capacity-ah", required=True, type=float)
    parser.add_argument("--reference-capacity-source", default="device_configuration")
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--recalibration-root", required=True, type=Path)
    parser.add_argument("--physical-catalog-root", required=True, type=Path)
    parser.add_argument("--lookback-hours", default=72.0, type=float)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="Return 0 only when a matching canonical Phase 2 artifact exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_cache:
        cached = find_cached_phase2_run(
            serial=str(args.serial),
            chemistry=str(args.chemistry),
            module_count=args.module_count,
            reference_capacity_ah=args.reference_capacity_ah,
            reference_capacity_source=args.reference_capacity_source,
            validation_root=args.validation_root,
            recalibration_root=args.recalibration_root,
            physical_catalog_root=args.physical_catalog_root,
            lookback_hours=args.lookback_hours,
        )
        if cached is None:
            print("BATTRYML_PHASE2_CACHE_MISS", flush=True)
            return 3
        print(f"BATTRYML_PHASE2_CACHE_HIT {cached}", flush=True)
        return 0
    finalize_phase2(
        serial=str(args.serial),
        chemistry=str(args.chemistry),
        module_count=args.module_count,
        reference_capacity_ah=args.reference_capacity_ah,
        reference_capacity_source=args.reference_capacity_source,
        validation_root=args.validation_root,
        recalibration_root=args.recalibration_root,
        physical_catalog_root=args.physical_catalog_root,
        lookback_hours=args.lookback_hours,
        compression=None if args.compression == "none" else args.compression,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
