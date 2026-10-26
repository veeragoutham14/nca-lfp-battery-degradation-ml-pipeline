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

from src.config import StorageConfig
from src.discharge_cycle_builder import (
    build_capacity_measurements,
    build_discharge_usage_ledger,
    load_event_table,
)
from src.io_utils import write_parquet_chunk

CATALOG_VERSION = "physical_discharge_catalog_v2"
MINIMUM_MEANINGFUL_SOC_DROP_PP = 80.0
MINIMUM_CAPACITY_ESTIMATE_SOC_DROP_PP = 90.0
MEANINGFUL_RECHARGE_SOC_RECOVERY_PP = 1.0
EMPTY_CATALOG_COLUMNS = (
    "device",
    "chemistry",
    "module_count",
    "physical_cycle_id",
    "discharge_event_id",
    "evidence_class",
    "observed_100_to_0",
    "observed_99_to_1",
    "observed_95_to_5",
    "meaningful_deep_discharge",
    "measurement_resolution",
    "capacity_claim_eligible",
    "capacity_estimate_eligible",
    "start_utc",
    "end_utc",
    "duration_s",
    "soc_start_pct",
    "soc_end_pct",
    "soc_span_pp",
    "discharged_ah",
    "discharged_wh",
    "gross_discharged_ah",
    "gross_discharged_wh",
    "charge_intrusion_ah",
    "charge_intrusion_wh",
    "net_discharged_ah",
    "net_discharged_wh",
    "discharge_duration_s",
    "rest_duration_s",
    "rest_interruption_count",
    "longest_rest_s",
    "charge_intrusion_duration_s",
    "charge_intrusion_count",
    "data_coverage_fraction",
    "maximum_gap_s",
    "measurement_quality",
    "measurement_quality_reasons",
    "reference_capacity_ah",
    "reference_capacity_source",
    "estimated_full_capacity_ah",
    "calculated_soh_pct",
    "discharge_derived_soh_proxy_pct",
    "soc_span_normalized_soh_estimate_pct",
    "capacity_above_reference",
    "source_capacity_event_id",
    "source_event_ids",
    "continuity",
    "trajectory_end_reason",
    "provenance",
)


def _source_identity(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
    events_root: Path,
) -> dict[str, Any]:
    serial_root = events_root / f"serial={serial}"
    files = sorted(serial_root.rglob("events.parquet")) if serial_root.exists() else []
    file_metadata = [
        {
            "path": path.relative_to(events_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    payload = {
        "catalog_version": CATALOG_VERSION,
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "nominal_capacity_ah": nominal_capacity_ah,
        "source_files": file_metadata,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "source_event_file_count": len(files),
        "source_identity_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _cached_catalog(output_root: Path, serial: str, identity: str) -> Path | None:
    serial_root = output_root / f"serial={serial}"
    if not serial_root.exists():
        return None
    for run_dir in sorted(serial_root.glob("catalog_run=*"), reverse=True):
        summary_path = run_dir / "model_summary.json"
        parquet_path = run_dir / "physical_discharge_cycles.parquet"
        if not summary_path.exists() or not parquet_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("source_identity_sha256") == identity:
            return run_dir
    return None


def _evidence(start_soc: float | None, end_soc: float | None) -> str:
    if start_soc is None or end_soc is None or not np.isfinite(start_soc) or not np.isfinite(end_soc):
        return "partial_discharge"
    span = start_soc - end_soc
    if start_soc >= 100.0 and end_soc <= 0.0 and span >= 100.0:
        return "strict_full_discharge"
    if start_soc >= 99.0 and end_soc <= 1.0 and span >= 98.0:
        return "near_full_discharge"
    if start_soc >= 95.0 and end_soc <= 5.0 and span >= 90.0:
        return "operational_full_discharge"
    return "partial_discharge"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _new_episode(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "serial": str(event["serial"]),
        "start_timestamp": pd.Timestamp(event["start_timestamp"]),
        "end_timestamp": pd.Timestamp(event["start_timestamp"]),
        "start_soc": _number(event.get("start_soc"), np.nan),
        "end_soc": _number(event.get("start_soc"), np.nan),
        "start_soh": _number(event.get("start_soh"), np.nan),
        "end_soh": _number(event.get("start_soh"), np.nan),
        "gross_discharge_ah": 0.0,
        "gross_discharge_wh": 0.0,
        "charge_intrusion_ah": 0.0,
        "charge_intrusion_wh": 0.0,
        "discharge_duration_s": 0.0,
        "rest_duration_s": 0.0,
        "charge_intrusion_duration_s": 0.0,
        "rest_interruption_count": 0,
        "charge_intrusion_count": 0,
        "longest_rest_s": 0.0,
        "num_discharge_events": 0,
        "source_event_ids": [],
        "continuity_reasons": {"continuous_discharge"},
        "maximum_event_gap_s": 0.0,
        "last_event_end": pd.Timestamp(event["start_timestamp"]),
        "last_discharge_event_row_id": None,
        "pending_charge": None,
        "trajectory_end_reason": "dataset_end",
    }


def _track_gap(episode: dict[str, Any], event: dict[str, Any]) -> None:
    gap = max(
        0.0,
        (pd.Timestamp(event["start_timestamp"]) - episode["last_event_end"]).total_seconds(),
    )
    episode["maximum_event_gap_s"] = max(episode["maximum_event_gap_s"], gap)
    episode["last_event_end"] = pd.Timestamp(event["end_timestamp"])


def _append_discharge(episode: dict[str, Any], event: dict[str, Any]) -> None:
    _track_gap(episode, event)
    episode["end_timestamp"] = pd.Timestamp(event["end_timestamp"])
    episode["end_soc"] = _number(event.get("end_soc"), episode["end_soc"])
    episode["end_soh"] = _number(event.get("end_soh"), episode["end_soh"])
    episode["gross_discharge_ah"] += _number(event.get("throughput_ah"))
    episode["gross_discharge_wh"] += _number(event.get("energy_wh"))
    episode["discharge_duration_s"] += _number(event.get("duration_seconds"))
    episode["num_discharge_events"] += 1
    episode["source_event_ids"].append(str(event.get("event_id", "")))
    episode["last_discharge_event_row_id"] = event.get("_event_row_id")


def _append_rest(episode: dict[str, Any], event: dict[str, Any]) -> None:
    _track_gap(episode, event)
    duration = _number(event.get("duration_seconds"))
    episode["end_timestamp"] = pd.Timestamp(event["end_timestamp"])
    episode["end_soc"] = _number(event.get("end_soc"), episode["end_soc"])
    episode["end_soh"] = _number(event.get("end_soh"), episode["end_soh"])
    episode["rest_duration_s"] += duration
    episode["longest_rest_s"] = max(episode["longest_rest_s"], duration)
    episode["rest_interruption_count"] += 1
    episode["continuity_reasons"].add("rest_continuation")
    episode["source_event_ids"].append(str(event.get("event_id", "")))


def _start_pending_charge(episode: dict[str, Any], event: dict[str, Any]) -> None:
    episode["pending_charge"] = {
        "start_timestamp": pd.Timestamp(event["start_timestamp"]),
        "baseline_soc": episode["end_soc"],
        "maximum_soc": max(
            _number(event.get("start_soc"), episode["end_soc"]),
            _number(event.get("end_soc"), episode["end_soc"]),
        ),
        "events": [event],
    }


def _extend_pending_charge(episode: dict[str, Any], event: dict[str, Any]) -> None:
    pending = episode["pending_charge"]
    pending["events"].append(event)
    pending["maximum_soc"] = max(
        pending["maximum_soc"],
        _number(event.get("start_soc"), pending["maximum_soc"]),
        _number(event.get("end_soc"), pending["maximum_soc"]),
    )


def _pending_recovery_pp(episode: dict[str, Any], event: dict[str, Any] | None = None) -> float:
    pending = episode.get("pending_charge")
    if pending is None:
        return 0.0
    maximum_soc = pending["maximum_soc"]
    if event is not None:
        maximum_soc = max(
            maximum_soc,
            _number(event.get("start_soc"), maximum_soc),
            _number(event.get("end_soc"), maximum_soc),
        )
    return maximum_soc - pending["baseline_soc"]


def _commit_pending_charge(episode: dict[str, Any]) -> None:
    pending = episode.get("pending_charge")
    if pending is None:
        return
    episode["pending_charge"] = None
    for event in pending["events"]:
        _track_gap(episode, event)
        episode["end_timestamp"] = pd.Timestamp(event["end_timestamp"])
        episode["end_soc"] = _number(event.get("end_soc"), episode["end_soc"])
        episode["end_soh"] = _number(event.get("end_soh"), episode["end_soh"])
        episode["charge_intrusion_ah"] += _number(event.get("throughput_ah"))
        episode["charge_intrusion_wh"] += _number(event.get("energy_wh"))
        episode["charge_intrusion_duration_s"] += _number(event.get("duration_seconds"))
        episode["source_event_ids"].append(str(event.get("event_id", "")))
    episode["charge_intrusion_count"] += 1
    episode["continuity_reasons"].add("tolerated_charge_intrusion")


def build_meaningful_discharge_episodes(
    events: pd.DataFrame,
    usage_ledger: pd.DataFrame,
    *,
    minimum_soc_drop_pp: float = MINIMUM_MEANINGFUL_SOC_DROP_PP,
    meaningful_recharge_soc_recovery_pp: float = MEANINGFUL_RECHARGE_SOC_RECOVERY_PP,
) -> pd.DataFrame:
    """Build one physical discharge across rest and sub-1 pp charge intrusions."""
    if events.empty:
        return pd.DataFrame()
    cumulative_by_event_row = (
        usage_ledger.set_index("_event_row_id")[[
            "cumulative_all_discharge_ah",
            "cumulative_all_discharge_wh",
        ]].to_dict(orient="index")
        if not usage_ledger.empty
        else {}
    )
    records: list[dict[str, Any]] = []
    active_by_serial: dict[str, dict[str, Any]] = {}

    def close(serial_key: str, reason: str) -> None:
        episode = active_by_serial.pop(serial_key, None)
        if episode is None or episode["num_discharge_events"] == 0:
            return
        episode.pop("pending_charge", None)
        episode["trajectory_end_reason"] = reason
        episode["soc_drop"] = episode["start_soc"] - episode["end_soc"]
        if episode["soc_drop"] < minimum_soc_drop_pp:
            return
        cumulative = cumulative_by_event_row.get(episode["last_discharge_event_row_id"], {})
        episode["cumulative_all_discharge_ah"] = cumulative.get(
            "cumulative_all_discharge_ah", np.nan
        )
        episode["cumulative_all_discharge_wh"] = cumulative.get(
            "cumulative_all_discharge_wh", np.nan
        )
        episode["source_event_ids"] = ",".join(episode["source_event_ids"])
        episode["continuity_reasons"] = ",".join(sorted(episode["continuity_reasons"]))
        records.append(episode)

    ordered = events.sort_values(["serial", "start_timestamp"], kind="mergesort")
    for event in ordered.to_dict(orient="records"):
        serial_key = str(event["serial"])
        mode = str(event["mode"])
        active = active_by_serial.get(serial_key)

        if active is None:
            if mode == "discharging":
                active = _new_episode(event)
                active_by_serial[serial_key] = active
                _append_discharge(active, event)
            continue

        if mode == "charging":
            if active["pending_charge"] is None:
                _start_pending_charge(active, event)
            else:
                _extend_pending_charge(active, event)
            if _pending_recovery_pp(active) >= meaningful_recharge_soc_recovery_pp:
                close(serial_key, "meaningful_recharge")
            continue

        if active["pending_charge"] is not None:
            if _pending_recovery_pp(active, event) >= meaningful_recharge_soc_recovery_pp:
                close(serial_key, "meaningful_recharge")
                if mode == "discharging":
                    replacement = _new_episode(event)
                    active_by_serial[serial_key] = replacement
                    _append_discharge(replacement, event)
                continue
            _commit_pending_charge(active)

        if mode == "discharging":
            _append_discharge(active, event)
        elif mode == "rest":
            _append_rest(active, event)
        elif mode == "missing":
            close(serial_key, "major_data_gap")
        else:
            close(serial_key, "session_boundary")

    for serial_key in list(active_by_serial):
        close(serial_key, "dataset_end")
    return (
        pd.DataFrame.from_records(records)
        if records
        else pd.DataFrame(columns=EMPTY_CATALOG_COLUMNS)
    )


def _overlapping_measurement(
    episode: pd.Series, measurements: pd.DataFrame
) -> pd.Series | None:
    if measurements.empty:
        return None
    start = pd.Timestamp(episode["start_timestamp"])
    end = pd.Timestamp(episode["end_timestamp"])
    overlap = measurements.loc[
        pd.to_datetime(measurements["start_anchor_time"], utc=True).le(end)
        & pd.to_datetime(measurements["end_anchor_time"], utc=True).ge(start)
    ].copy()
    if overlap.empty:
        return None
    overlap["_overlap_s"] = overlap.apply(
        lambda row: max(
            0.0,
            (
                min(end, pd.Timestamp(row["end_anchor_time"]))
                - max(start, pd.Timestamp(row["start_anchor_time"]))
            ).total_seconds(),
        ),
        axis=1,
    )
    return overlap.sort_values("_overlap_s", ascending=False).iloc[0]


def build_physical_discharge_catalog(
    events: pd.DataFrame,
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
) -> pd.DataFrame:
    usage = build_discharge_usage_ledger(events)
    episodes = build_meaningful_discharge_episodes(events, usage)
    exact = build_capacity_measurements(events, usage)
    records: list[dict[str, Any]] = []
    for ordinal, episode in episodes.reset_index(drop=True).iterrows():
        episode_start_soc = float(episode["start_soc"])
        episode_end_soc = float(episode["end_soc"])
        episode_span = episode_start_soc - episode_end_soc
        measurement = _overlapping_measurement(episode, exact)
        strict = (
            measurement is not None
            and episode_start_soc >= 100.0
            and episode_end_soc <= 0.0
            and episode_span >= 100.0
        )
        start = (
            pd.Timestamp(measurement["start_anchor_time"])
            if strict
            else pd.Timestamp(episode["start_timestamp"])
        )
        end = (
            pd.Timestamp(measurement["end_anchor_time"])
            if strict
            else pd.Timestamp(episode["end_timestamp"])
        )
        start_soc = 100.0 if strict else float(episode["start_soc"])
        end_soc = 0.0 if strict else float(episode["end_soc"])
        evidence = _evidence(start_soc, end_soc)
        gross_discharged_ah = (
            float(measurement["capacity_ah"])
            if strict
            else float(episode["gross_discharge_ah"])
        )
        gross_discharged_wh = (
            float(measurement["capacity_wh"])
            if strict
            else float(episode["gross_discharge_wh"])
        )
        charge_intrusion_ah = float(episode["charge_intrusion_ah"])
        charge_intrusion_wh = float(episode["charge_intrusion_wh"])
        net_discharged_ah = gross_discharged_ah - charge_intrusion_ah
        net_discharged_wh = gross_discharged_wh - charge_intrusion_wh
        discharge_seconds = (
            float(measurement["discharge_duration_seconds"])
            if strict
            else float(episode["discharge_duration_s"])
        )
        rest_seconds = (
            float(measurement["rest_duration_seconds"])
            if strict
            else float(episode["rest_duration_s"])
        )
        elapsed_seconds = max(0.0, (end - start).total_seconds())
        charge_seconds = float(episode["charge_intrusion_duration_s"])
        total_observed_seconds = discharge_seconds + rest_seconds + charge_seconds
        observed_seconds = min(elapsed_seconds, total_observed_seconds)
        coverage = observed_seconds / elapsed_seconds if elapsed_seconds > 0 else None
        duration_consistent = total_observed_seconds <= elapsed_seconds + 5.0
        span = start_soc - end_soc
        capacity_estimate_eligible = span >= MINIMUM_CAPACITY_ESTIMATE_SOC_DROP_PP
        normalized_capacity = (
            net_discharged_ah / (span / 100.0)
            if capacity_estimate_eligible and span > 0
            else None
        )
        records.append(
            {
                "device": str(serial),
                "chemistry": chemistry,
                "module_count": module_count,
                "physical_cycle_id": f"physical_cycle_{ordinal + 1:06d}",
                "discharge_event_id": (
                    f"reference_full_cycle_{int(measurement['event_discharge_id']):06d}"
                    if strict
                    else f"physical_discharge_{ordinal + 1:06d}"
                ),
                "evidence_class": evidence,
                "observed_100_to_0": evidence == "strict_full_discharge",
                "observed_99_to_1": start_soc >= 99.0 and end_soc <= 1.0 and span >= 98.0,
                "observed_95_to_5": start_soc >= 95.0 and end_soc <= 5.0 and span >= 90.0,
                "meaningful_deep_discharge": span >= MINIMUM_MEANINGFUL_SOC_DROP_PP,
                "measurement_resolution": "one_second_event_anchors",
                "capacity_claim_eligible": capacity_estimate_eligible,
                "capacity_estimate_eligible": capacity_estimate_eligible,
                "start_utc": start,
                "end_utc": end,
                "duration_s": elapsed_seconds,
                "soc_start_pct": start_soc,
                "soc_end_pct": end_soc,
                "soc_span_pp": span,
                "discharged_ah": gross_discharged_ah,
                "discharged_wh": gross_discharged_wh,
                "gross_discharged_ah": gross_discharged_ah,
                "gross_discharged_wh": gross_discharged_wh,
                "charge_intrusion_ah": charge_intrusion_ah,
                "charge_intrusion_wh": charge_intrusion_wh,
                "net_discharged_ah": net_discharged_ah,
                "net_discharged_wh": net_discharged_wh,
                "observed_discharge_ah": gross_discharged_ah,
                "discharge_duration_s": discharge_seconds,
                "rest_duration_s": rest_seconds,
                "rest_interruption_count": int(episode["rest_interruption_count"]),
                "longest_rest_s": float(episode["longest_rest_s"]),
                "charge_intrusion_duration_s": charge_seconds,
                "charge_intrusion_count": int(episode["charge_intrusion_count"]),
                "num_discharge_events": int(episode["num_discharge_events"]),
                "data_coverage_fraction": coverage,
                "maximum_gap_s": float(episode["maximum_event_gap_s"]),
                "rms_current_a": None,
                "p95_current_a": None,
                "p99_current_a": None,
                "max_discharge_current_a": None,
                "rms_c_rate": None,
                "p95_c_rate": None,
                "temperature_mean_c": None,
                "temperature_median_c": None,
                "temperature_min_c": None,
                "temperature_max_c": None,
                "temperature_coverage_fraction": None,
                "sys_cycles_start": None,
                "sys_cycles_end": None,
                "sys_cycles_delta": None,
                "measurement_quality": (
                    "high"
                    if duration_consistent
                    and strict
                    and measurement["capacity_measurement_quality"] == "clean_anchor_100_to_0"
                    else "medium"
                    if duration_consistent and capacity_estimate_eligible
                    else "low"
                ),
                "measurement_quality_reasons": [
                    f"evidence_class:{evidence}",
                    f"soc_span_pp:{span:.1f}",
                    "canonical_event_state_machine",
                    "rest_preserved",
                    "charging_requires_persistent_1pp_soc_recovery_to_terminate",
                    *([] if duration_consistent else ["event_duration_exceeds_elapsed_interval"]),
                ],
                "reference_capacity_ah": nominal_capacity_ah,
                "reference_capacity_source": "device_configuration",
                "estimated_full_capacity_ah": normalized_capacity,
                "calculated_soh_pct": (
                    normalized_capacity / nominal_capacity_ah * 100.0
                    if normalized_capacity is not None
                    else None
                ),
                "discharge_derived_soh_proxy_pct": (
                    net_discharged_ah / nominal_capacity_ah * 100.0
                    if evidence == "strict_full_discharge"
                    else None
                ),
                "soc_span_normalized_soh_estimate_pct": (
                    normalized_capacity / nominal_capacity_ah * 100.0
                    if normalized_capacity is not None
                    else None
                ),
                "capacity_above_reference": (
                    normalized_capacity > nominal_capacity_ah * 1.03
                    if normalized_capacity is not None
                    else None
                ),
                "source_capacity_event_id": (
                    int(measurement["event_discharge_id"]) if strict else None
                ),
                "source_event_ids": episode["source_event_ids"],
                "continuity": episode["continuity_reasons"],
                "trajectory_end_reason": (
                    "reached_empty_soc" if strict else episode["trajectory_end_reason"]
                ),
                "provenance": (
                    "one-second-derived operating events; negative=current discharge; "
                    "rest and sub-1pp charge intrusions retained"
                ),
            }
        )
    if not records:
        # No meaningful deep discharge: keep the schema so summaries count zero rows.
        return pd.DataFrame(columns=EMPTY_CATALOG_COLUMNS)
    return pd.DataFrame.from_records(records)


def run_catalog(
    *,
    serial: str,
    chemistry: str,
    module_count: int,
    nominal_capacity_ah: float,
    events_root: Path,
    output_root: Path,
    compression: str | None = "zstd",
) -> Path:
    source_identity = _source_identity(
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
        nominal_capacity_ah=nominal_capacity_ah,
        events_root=events_root,
    )
    cached = _cached_catalog(
        output_root, serial, source_identity["source_identity_sha256"]
    )
    if cached is not None:
        return cached
    events = load_event_table(events_root, serial=serial)
    if events.empty:
        raise ValueError(f"No classified events are available for {serial}")
    catalog = build_physical_discharge_catalog(
        events,
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
        nominal_capacity_ah=nominal_capacity_ah,
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"serial={serial}" / f"catalog_run={stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_parquet_chunk(catalog, run_dir / "physical_discharge_cycles.parquet", compression=compression)
    summary = {
        "model_family": "physical_discharge_catalog",
        "catalog_version": CATALOG_VERSION,
        "serial": serial,
        "chemistry": chemistry,
        "module_count": module_count,
        "nominal_capacity_ah": nominal_capacity_ah,
        "minimum_meaningful_soc_drop_pp": MINIMUM_MEANINGFUL_SOC_DROP_PP,
        "minimum_capacity_estimate_soc_drop_pp": MINIMUM_CAPACITY_ESTIMATE_SOC_DROP_PP,
        "meaningful_recharge_soc_recovery_pp": MEANINGFUL_RECHARGE_SOC_RECOVERY_PP,
        **source_identity,
        "physical_cycle_rows": len(catalog),
        "meaningful_80pp_rows": int(catalog["meaningful_deep_discharge"].sum()),
        "capacity_estimate_90pp_rows": int(catalog["capacity_estimate_eligible"].sum()),
        "strict_100_to_0_rows": int(catalog["observed_100_to_0"].sum()),
        "supporting_99_to_1_rows": int((catalog["observed_99_to_1"] & ~catalog["observed_100_to_0"]).sum()),
        "supporting_95_to_5_rows": int((catalog["observed_95_to_5"] & ~catalog["observed_99_to_1"]).sum()),
        "rest_continuation_rows": int(catalog["rest_duration_s"].gt(0).sum()),
        "observed_start": catalog["start_utc"].min(),
        "observed_end": catalog["end_utc"].max(),
        "current_sign_convention": "positive=charge; negative=discharge",
        "claim_limit": (
            "BMS SOC and event-integrated current define these usable-capacity trajectories. "
            "Only >=90 pp spans receive SOC-normalized capacity estimates; they are not "
            "laboratory ground truth."
        ),
    }
    (run_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return run_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build canonical physical discharge cycles.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--module-count", required=True, type=int)
    parser.add_argument("--nominal-capacity-ah", required=True, type=float)
    parser.add_argument("--events-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--compression", default="zstd")
    args = parser.parse_args(argv)
    storage = StorageConfig.from_env()
    run_dir = run_catalog(
        serial=str(args.serial),
        chemistry=args.chemistry,
        module_count=args.module_count,
        nominal_capacity_ah=args.nominal_capacity_ah,
        events_root=args.events_root or storage.data_dir / "processed" / "events",
        output_root=args.output_root or storage.data_dir / "processed" / "physical_discharge_catalog",
        compression=None if args.compression == "none" else args.compression,
    )
    print(f"Physical discharge catalog complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
