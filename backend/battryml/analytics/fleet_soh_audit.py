from __future__ import annotations

import argparse
import html
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from battryml.analytics.soh_signal_audit import run_audit
from battryml.battery_profiles import battery_profile

logger = logging.getLogger(__name__)


def _serials(value: str) -> list[str]:
    serials = list(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not serials:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated serial")
    return serials


def _fleet_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["audit_metrics"]
    legacy_label_status = metrics.get("label_status")
    label_status = (
        "weak_label_only"
        if legacy_label_status == "screened_proxy_candidate"
        else legacy_label_status
    )
    signal_quality_status = metrics.get("signal_quality_status")
    if (
        signal_quality_status is None
        and legacy_label_status == "screened_proxy_candidate"
    ):
        signal_quality_status = "stable_observed_bms_estimate"
    return {
        "serial": summary["serial"],
        "chemistry": summary.get("chemistry"),
        "device_role": summary.get("device_role"),
        "observed_start": summary.get("observed_start"),
        "observed_end": summary.get("observed_end"),
        "raw_rows": summary.get("raw_rows"),
        "daily_rows": summary.get("daily_rows"),
        "first_bms_soh_pct": metrics.get("first_bms_soh_pct"),
        "last_bms_soh_pct": metrics.get("last_bms_soh_pct"),
        "net_bms_soh_change_pct": metrics.get("net_bms_soh_change_pct"),
        "step_drop_count": metrics.get("step_drop_count"),
        "step_recovery_count": metrics.get("step_recovery_count"),
        "largest_step_drop_pct": metrics.get("largest_step_drop_pct"),
        "unstable_soh_days": metrics.get("unstable_soh_days"),
        "median_coverage_fraction": metrics.get("median_coverage_fraction"),
        "total_discharge_ah": metrics.get("total_discharge_ah"),
        "observed_bms_loss_pp_per_1000ah": metrics.get(
            "observed_bms_loss_pp_per_1000ah"
        ),
        "equivalent_capacity_loss_ah_per_1000ah": metrics.get(
            "equivalent_capacity_loss_ah_per_1000ah"
        ),
        "signal_quality_status": signal_quality_status,
        "ground_truth_status": metrics.get(
            "ground_truth_status", "missing_independent_capacity_label"
        ),
        "label_status": label_status,
        "available_signal_count": summary.get("available_signal_count"),
    }


def _median(values: list[float]) -> float | None:
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _event_evidence(
    completed: list[tuple[dict[str, Any], Path]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    for summary, run_dir in completed:
        event_path = run_dir / "soh_change_events.parquet"
        if not event_path.exists():
            continue
        for event in pq.ParquetFile(event_path).read().to_pylist():
            event["serial"] = str(summary["serial"])
            event["audit_run"] = run_dir.name.removeprefix("audit_run=")
            events.append(event)

    by_day: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    by_month: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        event_type = str(event.get("event_type"))
        day = event.get("day_date")
        by_day[(day, event_type)].append(event)
        by_month[(str(day)[:7], event_type)].append(event)

    day_clusters: list[dict[str, Any]] = []
    for (day, event_type), items in sorted(by_day.items(), key=lambda item: item[0]):
        serials = sorted({str(item["serial"]) for item in items})
        deltas = [float(item["soh_delta_pct"]) for item in items]
        day_clusters.append(
            {
                "day_date": day,
                "event_type": event_type,
                "device_count": len(serials),
                "event_count": len(items),
                "serials": ", ".join(serials),
                "mean_soh_delta_pct": sum(deltas) / len(deltas),
                "median_soh_delta_pct": _median(deltas),
            }
        )

    month_clusters: list[dict[str, Any]] = []
    for (month, event_type), items in sorted(
        by_month.items(), key=lambda item: item[0]
    ):
        serials = sorted({str(item["serial"]) for item in items})
        deltas = [float(item["soh_delta_pct"]) for item in items]
        month_clusters.append(
            {
                "month": month,
                "event_type": event_type,
                "device_count": len(serials),
                "event_count": len(items),
                "serials": ", ".join(serials),
                "mean_soh_delta_pct": sum(deltas) / len(deltas),
            }
        )
    return events, day_clusters, month_clusters


def _context_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "day_count": 0,
            "low_soc_day_fraction": None,
            "high_soc_day_fraction": None,
            "full_soc_span_day_fraction": None,
            "median_discharge_ah": None,
            "median_coverage_fraction": None,
        }

    def fraction(predicate: Callable[[dict[str, Any]], bool]) -> float:
        return sum(bool(predicate(row)) for row in rows) / len(rows)

    return {
        "day_count": len(rows),
        "low_soc_day_fraction": fraction(
            lambda row: float(row.get("min_soc_pct") or 999) <= 5
        ),
        "high_soc_day_fraction": fraction(
            lambda row: float(row.get("max_soc_pct") or -999) >= 99
        ),
        "full_soc_span_day_fraction": fraction(
            lambda row: (
                float(row.get("min_soc_pct") or 999) <= 5
                and float(row.get("max_soc_pct") or -999) >= 99
            )
        ),
        "median_discharge_ah": _median(
            [
                float(row["discharge_ah"])
                for row in rows
                if row.get("discharge_ah") is not None
            ]
        ),
        "median_coverage_fraction": _median(
            [
                float(row["coverage_fraction"])
                for row in rows
                if row.get("coverage_fraction") is not None
            ]
        ),
    }


def _daily_evidence(
    completed: list[tuple[dict[str, Any], Path]],
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    event_days = {
        (str(event["serial"]), str(event["day_date"]))
        for event in events
        if event.get("day_date") is not None
    }
    daily_rows: list[dict[str, Any]] = []
    event_day_rows: list[dict[str, Any]] = []
    ordinary_day_rows: list[dict[str, Any]] = []
    for summary, run_dir in completed:
        daily_path = run_dir / "daily_health.parquet"
        if not daily_path.exists():
            continue
        serial = str(summary["serial"])
        for row in pq.ParquetFile(daily_path).read().to_pylist():
            row["serial"] = serial
            row["audit_run"] = run_dir.name.removeprefix("audit_run=")
            row["is_soh_change_day"] = (serial, str(row.get("day_date"))) in event_days
            daily_rows.append(row)
            if row["is_soh_change_day"]:
                event_day_rows.append(row)
            else:
                ordinary_day_rows.append(row)
    return (
        daily_rows,
        _context_metrics(event_day_rows),
        _context_metrics(ordinary_day_rows),
    )


def _fleet_html(
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    metrics: dict[str, Any],
    day_clusters: list[dict[str, Any]],
) -> str:
    table_rows = []
    for row in sorted(rows, key=lambda item: item.get("last_bms_soh_pct") or 999):
        coverage = row.get("median_coverage_fraction")
        equivalent_loss = row.get("equivalent_capacity_loss_ah_per_1000ah")
        equivalent_text = (
            "n/a" if equivalent_loss is None else f"{float(equivalent_loss):.4f}"
        )
        coverage_text = "n/a" if coverage is None else f"{float(coverage):.1%}"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['serial']))}</td>"
            f"<td>{row.get('first_bms_soh_pct')}</td>"
            f"<td>{row.get('last_bms_soh_pct')}</td>"
            f"<td>{row.get('net_bms_soh_change_pct')}</td>"
            f"<td>{row.get('step_drop_count')}</td>"
            f"<td>{row.get('step_recovery_count')}</td>"
            f"<td>{equivalent_text}</td>"
            f"<td>{coverage_text}</td>"
            f"<td>{html.escape(str(row.get('label_status')).replace('_', ' '))}</td>"
            "</tr>"
        )
    failure_items = "".join(
        f"<li>{html.escape(item['serial'])}: {html.escape(item['error'])}</li>"
        for item in failures
    )
    cluster_rows = []
    for cluster in sorted(
        day_clusters,
        key=lambda item: (item["device_count"], item["day_date"]),
        reverse=True,
    )[:30]:
        cluster_rows.append(
            "<tr>"
            f"<td>{html.escape(str(cluster['day_date']))}</td>"
            f"<td>{html.escape(cluster['event_type'].replace('_', ' '))}</td>"
            f"<td>{cluster['device_count']}</td>"
            f"<td>{float(cluster['mean_soh_delta_pct']):+.2f} pp</td>"
            f"<td>{html.escape(cluster['serials'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>BatteryML fleet SOH audit</title>
<style>*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f7;color:#26343d;font:14px Arial,sans-serif}}main{{max-width:1400px;margin:auto;padding:30px}}h1{{margin:4px 0 24px}}.eyebrow{{color:#176b87;font-weight:700;text-transform:uppercase;font-size:12px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}}.metric,section{{background:white;border:1px solid #dbe1e4;padding:20px}}.metric small{{display:block;color:#637078}}.metric strong{{display:block;font-size:24px;margin-top:8px}}section{{overflow:auto;margin-bottom:18px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #e5e9eb;text-align:left;white-space:nowrap}}th{{font-size:12px;color:#637078}}</style></head>
<body><main><span class="eyebrow">Cross-device evidence</span><h1>Fleet BMS SOH audit</h1>
<div class="metrics"><div class="metric"><small>Devices</small><strong>{metrics["device_count"]}</strong></div><div class="metric"><small>Weak labels</small><strong>{metrics["weak_label_device_count"]}</strong></div><div class="metric"><small>Devices with recoveries</small><strong>{metrics["devices_with_recoveries"]}</strong></div><div class="metric"><small>Median equivalent loss</small><strong>{metrics.get("median_equivalent_capacity_loss_ah_per_1000ah", 0):.4f}</strong></div></div>
<section><table><thead><tr><th>Serial</th><th>First SOH</th><th>Latest SOH</th><th>Net pp</th><th>Drops</th><th>Recoveries</th><th>Equivalent Ah/1k Ah</th><th>Coverage</th><th>Label status</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></section>
<section><h2>Same-day change clusters</h2><table><thead><tr><th>Date</th><th>Event</th><th>Devices</th><th>Mean change</th><th>Serials</th></tr></thead><tbody>{"".join(cluster_rows)}</tbody></table></section>
{f"<section><h2>Failures</h2><ul>{failure_items}</ul></section>" if failures else ""}
</main></body></html>"""


def run_fleet_audit(
    *,
    serials: list[str],
    raw_root: Path,
    output_root: Path,
    default_nominal_capacity_ah: float | None,
    workers: int = 1,
    reuse_existing: bool = False,
    run_id: str | None = None,
) -> Path:
    stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    fleet_dir = output_root / f"fleet_run={stamp}"
    fleet_dir.mkdir(parents=True, exist_ok=False)
    completed: list[tuple[dict[str, Any], Path]] = []
    failures: list[dict[str, str]] = []

    def execute(serial: str) -> tuple[dict[str, Any], Path]:
        profile = battery_profile(serial)
        if reuse_existing:
            existing = sorted(
                (output_root / f"serial={serial}").glob(
                    "audit_run=*/model_summary.json"
                ),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if existing:
                summary = json.loads(existing[0].read_text(encoding="utf-8"))
                summary["chemistry"] = profile.chemistry if profile else None
                summary["device_role"] = profile.device_role if profile else None
                return summary, existing[0].parent
        nominal = (
            profile.nominal_capacity_ah
            if profile and profile.nominal_capacity_ah is not None
            else default_nominal_capacity_ah
        )
        run_dir = run_audit(
            serial=serial,
            raw_root=raw_root,
            output_root=output_root,
            nominal_capacity_ah=nominal,
            run_id=stamp,
        )
        summary = json.loads(
            (run_dir / "model_summary.json").read_text(encoding="utf-8")
        )
        summary["chemistry"] = profile.chemistry if profile else None
        summary["device_role"] = profile.device_role if profile else None
        return summary, run_dir

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(execute, serial): serial for serial in serials}
        for future in as_completed(futures):
            serial = futures[future]
            try:
                summary, run_dir = future.result()
                completed.append((summary, run_dir))
                logger.info("Fleet audit completed for %s", serial)
            except Exception as exc:
                logger.error("Fleet audit failed for %s: %s", serial, exc)
                failures.append({"serial": serial, "error": str(exc)})

    rows = [_fleet_row(summary) for summary, _ in completed]
    events, day_clusters, month_clusters = _event_evidence(completed)
    daily_rows, event_day_context, ordinary_day_context = _daily_evidence(
        completed, events
    )
    if rows:
        pq.write_table(
            pa.Table.from_pylist(rows),
            fleet_dir / "fleet_summary.parquet",
            compression="zstd",
        )
    if events:
        pq.write_table(
            pa.Table.from_pylist(events),
            fleet_dir / "fleet_soh_change_events.parquet",
            compression="zstd",
        )
    if daily_rows:
        pq.write_table(
            pa.Table.from_pylist(daily_rows),
            fleet_dir / "fleet_daily_health.parquet",
            compression="zstd",
        )
    if day_clusters:
        pq.write_table(
            pa.Table.from_pylist(day_clusters),
            fleet_dir / "fleet_daily_change_clusters.parquet",
            compression="zstd",
        )
    if month_clusters:
        pq.write_table(
            pa.Table.from_pylist(month_clusters),
            fleet_dir / "fleet_monthly_change_clusters.parquet",
            compression="zstd",
        )

    reference_slope = 0.12
    equivalent_losses = [
        float(row["equivalent_capacity_loss_ah_per_1000ah"])
        for row in rows
        if row.get("equivalent_capacity_loss_ah_per_1000ah") is not None
    ]
    synchronized_drop_clusters = [
        cluster
        for cluster in day_clusters
        if cluster["event_type"] == "step_drop" and cluster["device_count"] >= 2
    ]
    event_full_span = event_day_context.get("full_soc_span_day_fraction")
    ordinary_full_span = ordinary_day_context.get("full_soc_span_day_fraction")
    event_discharge = event_day_context.get("median_discharge_ah")
    ordinary_discharge = ordinary_day_context.get("median_discharge_ah")
    payload = {
        "model_family": "fleet_bms_soh_signal_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "requested_serials": serials,
        "completed_serials": sorted(row["serial"] for row in rows),
        "failures": failures,
        "device_count": len(rows),
        "weak_label_device_count": sum(
            row.get("label_status") == "weak_label_only" for row in rows
        ),
        "devices_with_recoveries": sum(
            int(row.get("step_recovery_count") or 0) > 0 for row in rows
        ),
        "reference_loss_slope_ah_per_1000ah": reference_slope,
        "median_equivalent_capacity_loss_ah_per_1000ah": _median(equivalent_losses),
        "minimum_equivalent_capacity_loss_ah_per_1000ah": (
            min(equivalent_losses) if equivalent_losses else None
        ),
        "maximum_equivalent_capacity_loss_ah_per_1000ah": (
            max(equivalent_losses) if equivalent_losses else None
        ),
        "devices_above_twice_reference_slope": sum(
            value > 2 * reference_slope for value in equivalent_losses
        ),
        "total_step_drop_events": sum(
            int(row.get("step_drop_count") or 0) for row in rows
        ),
        "total_step_recovery_events": sum(
            int(row.get("step_recovery_count") or 0) for row in rows
        ),
        "same_day_multi_device_drop_clusters": len(synchronized_drop_clusters),
        "maximum_devices_in_same_day_drop_cluster": max(
            (cluster["device_count"] for cluster in synchronized_drop_clusters),
            default=0,
        ),
        "soh_change_day_context": event_day_context,
        "ordinary_day_context": ordinary_day_context,
        "full_soc_span_enrichment_on_change_days": (
            event_full_span / ordinary_full_span
            if event_full_span is not None and ordinary_full_span
            else None
        ),
        "median_discharge_ratio_on_change_days": (
            event_discharge / ordinary_discharge
            if event_discharge is not None and ordinary_discharge
            else None
        ),
        "label_decision": (
            "BMS SOH is a noisy auxiliary target, not physical SOH ground truth. "
            "Independent capacity labels are required for supervised SOH training."
        ),
    }
    (fleet_dir / "fleet_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (fleet_dir / "fleet_soh_audit_report.html").write_text(
        _fleet_html(rows, failures, payload, day_clusters), encoding="utf-8"
    )
    return fleet_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BMS SOH audit across a fleet."
    )
    parser.add_argument("--serials", type=_serials, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--default-nominal-capacity-ah", type=float, default=64.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse the latest completed per-device audit and run only missing devices.",
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        output = run_fleet_audit(
            serials=args.serials,
            raw_root=args.raw_root,
            output_root=args.output_root,
            default_nominal_capacity_ah=args.default_nominal_capacity_ah,
            workers=args.workers,
            reuse_existing=args.reuse_existing,
        )
        print(output)
        return 0
    except Exception as exc:
        logger.error("Fleet SOH audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
