from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.device_specs import nominal_capacity_ah_for_serial
from src.electrothermal_capacity_context import (
    _canonicalize_physical_cycles,
    _safe_json,
)
from src.io_utils import write_parquet_chunk

CLASS_LABELS = {
    "strict_full_discharge": "strict_100_to_0",
    "near_full_discharge": "supporting_99_to_1",
    "operational_full_discharge": "supporting_95_to_5",
}


def _latest(root: Path, serial: str, prefix: str) -> Path:
    runs = sorted((root / f"serial={serial}").glob(f"{prefix}=*"))
    if not runs:
        raise FileNotFoundError(f"No {prefix} run for {serial}")
    return runs[-1]


def _strict_rejection_reason(row: pd.Series) -> str:
    if float(row["soc_start_pct"]) < 95.0:
        return "start_below_95_pct"
    if float(row["soc_end_pct"]) > 5.0:
        return "end_above_5_pct"
    if row["evidence_class"] == "operational_full_discharge":
        return "accepted_95_to_5_but_not_99_to_1"
    if row["evidence_class"] == "near_full_discharge":
        return "accepted_99_to_1_but_not_exact_100_to_0"
    return "not_capacity_eligible_or_unclassified"


def _cache_for_event(cache_dir: Path, event_id: str) -> Path | None:
    paths = sorted(cache_dir.glob(f"event={event_id}__window=*.parquet"))
    return paths[-1] if paths else None


def _plot_event(cache: Path, event: pd.Series, output: Path, title_prefix: str) -> None:
    frame = pd.read_parquet(
        cache,
        columns=["time", "batt_soc_pct", "batt_current_a", "batt_voltage_v"],
    )
    time = pd.to_datetime(frame["time"], utc=True)
    mask = time.between(pd.Timestamp(event["start_utc"]), pd.Timestamp(event["end_utc"]))
    frame = frame.loc[mask].copy()
    time = time.loc[mask]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    axes[0].plot(time, frame["batt_soc_pct"], linewidth=1.1)
    axes[0].set_ylabel("BMS SOC (%)")
    axes[1].plot(time, frame["batt_current_a"], linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Current (A)")
    axes[2].plot(time, frame["batt_voltage_v"], linewidth=0.8)
    axes[2].set_ylabel("Stack voltage (V)")
    axes[2].set_xlabel("UTC")
    fig.suptitle(
        f"{title_prefix}: {event['discharge_event_id']} | "
        f"{event['soc_start_pct']:.1f}% to {event['soc_end_pct']:.1f}%"
    )
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _independent_metrics(frame: pd.DataFrame, event: pd.Series, nominal_ah: float) -> dict[str, Any]:
    time = pd.to_datetime(frame["time"], utc=True)
    mask = time.between(pd.Timestamp(event["start_utc"]), pd.Timestamp(event["end_utc"]))
    data = frame.loc[mask].copy()
    time = time.loc[mask]
    current = pd.to_numeric(data["batt_current_a"], errors="coerce")
    discharge = (-current).where(current < 0)
    expected = max(1, int((pd.Timestamp(event["end_utc"]) - pd.Timestamp(event["start_utc"])).total_seconds()) + 1)
    temp_cols = [column for column in data if "temperature" in column and column.endswith("_c")]
    primary = [column for column in temp_cols if "temperature0_c" in column or "temperature1_c" in column]
    voltage_cols = [column for column in data if "cellvoltage_mv_" in column]
    temp = data[primary].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    voltage = data[voltage_cols].apply(pd.to_numeric, errors="coerce")
    voltage = voltage.where((voltage >= 2000) & (voltage <= 5000))
    spread = voltage.max(axis=1) - voltage.min(axis=1)
    return {
        "native_rows_recomputed": int(len(data)),
        "native_coverage_fraction_recomputed": min(1.0, float(time.nunique()) / expected),
        "rms_discharge_current_a_recomputed": float(np.sqrt(np.nanmean(np.square(discharge)))),
        "rms_c_rate_recomputed": float(np.sqrt(np.nanmean(np.square(discharge)))) / nominal_ah,
        "temperature_mean_c_recomputed": float(temp.mean()),
        "voltage_group_spread_p95_mv_recomputed": float(spread.quantile(0.95)),
        "low_soc_voltage_group_spread_p95_mv_recomputed": float(
            spread.loc[pd.to_numeric(data["batt_soc_pct"], errors="coerce").le(10)].quantile(0.95)
        ),
    }


def run_audit(
    *,
    serial: str,
    field_serial: str,
    processed_root: Path,
    cache_root: Path,
    output_root: Path,
) -> Path:
    phase2 = _latest(processed_root / "bms_soh_recalibration", serial, "recalibration_run")
    events = pd.read_parquet(phase2 / "capacity_discharge_events.parquet")
    eligible = events.loc[
        events["measurement_resolution"].eq("one_second")
        & events["start_utc"].notna()
        & events["end_utc"].notna()
    ].copy()
    canonical = _canonicalize_physical_cycles(eligible)
    canonical["strict_rejection_reason"] = canonical.apply(_strict_rejection_reason, axis=1)
    strict = canonical.loc[canonical["evidence_class"].eq("strict_full_discharge")].copy()
    run_dir = output_root / f"serial={serial}" / f"phase3_audit={datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True)
    write_parquet_chunk(canonical, run_dir / "canonical_physical_cycles.parquet", compression="zstd")
    write_parquet_chunk(strict, run_dir / "strict_100_to_0_events.parquet", compression="zstd")

    cache_dir = cache_root / "chemistry=NCA" / f"serial={serial}"
    accepted_plots: list[str] = []
    for _, event in strict.head(5).iterrows():
        cache = _cache_for_event(cache_dir, str(event["discharge_event_id"]))
        if cache:
            output = plot_dir / f"accepted_{event['discharge_event_id']}.png"
            _plot_event(cache, event, output, "Accepted strict event")
            accepted_plots.append(output.name)
    rejected_plots: list[str] = []
    rejected = canonical.loc[~canonical["evidence_class"].eq("strict_full_discharge")]
    for _, event in rejected.iterrows():
        if len(rejected_plots) >= 10:
            break
        cache = _cache_for_event(cache_dir, str(event["discharge_event_id"]))
        if cache:
            output = plot_dir / f"strict_rejected_{event['discharge_event_id']}.png"
            _plot_event(cache, event, output, "Rejected from strict class")
            rejected_plots.append(output.name)

    checks: list[dict[str, Any]] = []
    for target_serial in (serial, field_serial):
        phase3_root = processed_root / "electrothermal_capacity_context" / "chemistry=NCA"
        phase3 = _latest(phase3_root, target_serial, "phase3_run")
        context = pd.read_parquet(phase3 / "event_electrothermal_context.parquet").head(3)
        target_cache = cache_root / "chemistry=NCA" / f"serial={target_serial}"
        for _, output_row in context.iterrows():
            cache = _cache_for_event(target_cache, str(output_row["discharge_event_id"]))
            if cache is None:
                continue
            native = pd.read_parquet(cache)
            event = pd.Series(
                {
                    "start_utc": output_row["start_utc"],
                    "end_utc": output_row["end_utc"],
                }
            )
            recomputed = _independent_metrics(
                native,
                event,
                float(output_row.get("nominal_capacity_ah") or nominal_capacity_ah_for_serial(target_serial)),
            )
            check = {"serial": target_serial, "discharge_event_id": output_row["discharge_event_id"]}
            for name, value in recomputed.items():
                source_name = name.removesuffix("_recomputed")
                source_value = output_row.get(source_name)
                check[name] = value
                check[f"{source_name}_artifact"] = source_value
                if isinstance(value, (float, int)) and pd.notna(source_value):
                    check[f"{source_name}_absolute_delta"] = abs(float(value) - float(source_value))
            checks.append(check)
    checks_frame = pd.DataFrame.from_records(checks)
    write_parquet_chunk(checks_frame, run_dir / "metric_recomputation_checks.parquet", compression="zstd")

    validation = _latest(processed_root / "data_validation" / "chemistry=NCA", serial, "validation_run")
    validation_summary = json.loads((validation / "model_summary.json").read_text(encoding="utf-8"))
    first = pd.Timestamp(validation_summary["observed_start"])
    last = pd.Timestamp(validation_summary["observed_end"])
    covered_days_equivalent = (last - first).total_seconds() / 86400 * validation_summary["calendar_coverage_pct"] / 100
    summary = {
        "serial": serial,
        "source_phase2_run": phase2.name,
        "funnel": {
            "raw_downward_soc_candidate_windows": int(len(events)),
            "one_second_resolved_windows": int(len(eligible)),
            "unique_physical_discharge_cycles": int(len(canonical)),
            "strict_100_to_0": int(canonical["evidence_class"].eq("strict_full_discharge").sum()),
            "supporting_99_to_1": int(canonical["evidence_class"].eq("near_full_discharge").sum()),
            "supporting_95_to_5": int(canonical["evidence_class"].eq("operational_full_discharge").sum()),
            "rejected_from_strict_by_reason": Counter(canonical["strict_rejection_reason"]),
        },
        "coverage": {
            "first_timestamp_utc": first,
            "last_timestamp_utc": last,
            "calendar_span_days": (last - first).total_seconds() / 86400,
            "covered_days_equivalent": covered_days_equivalent,
            "missing_days_equivalent": (last - first).total_seconds() / 86400 - covered_days_equivalent,
            "calendar_coverage_pct": validation_summary["calendar_coverage_pct"],
            "raw_rows": validation_summary["raw_row_count"],
            "duplicate_rows": validation_summary["duplicate_rows"],
            "maximum_gap_seconds": validation_summary["maximum_gap_seconds"],
        },
        "plots": {"accepted": accepted_plots, "strict_rejected": rejected_plots},
        "independent_recomputation_rows": len(checks_frame),
        "rules": {
            "strict_100_to_0": "reported start SOC >=100.0%, end SOC <=0.0%, span >=100.0 pp",
            "supporting_99_to_1": "reported start SOC >=99.0%, end SOC <=1.0%, span >=98.0 pp",
            "supporting_95_to_5": "reported start SOC >=95.0%, end SOC <=5.0%, span >=90.0 pp",
            "current_sign": "positive=charge; negative=discharge; rest=|current|<=0.5 A",
            "integration_gap": "trapezoidal integration only for adjacent samples <=5 s",
            "major_gap": "trajectory terminates at a gap >3600 s",
            "charge_intrusion": "0.10 Ah allowed for eligibility; trajectory terminates above 0.50 Ah",
            "soc_recovery": "trajectory terminates after recovery >=5 percentage points from its minimum",
            "rest": "rest is retained inside the physical trajectory and reported; it is not a rejection by itself",
        },
        "claim_limit": "Candidate windows are not independent physical cycles until overlap deduplication. Equivalent covered/missing days are coverage-derived, not a count of wholly present/absent UTC dates.",
    }
    (run_dir / "audit_summary.json").write_text(
        json.dumps(_safe_json(summary), indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Phase 2/3 reference evidence.")
    parser.add_argument("--serial", default="100000017")
    parser.add_argument("--field-serial", default="100000001")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    output = run_audit(
        serial=args.serial,
        field_serial=args.field_serial,
        processed_root=args.data_dir / "processed",
        cache_root=args.data_dir / "cache" / "electrothermal_event_windows",
        output_root=args.data_dir / "processed" / "phase3_scientific_audit",
    )
    print(f"Phase 3 scientific audit complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
