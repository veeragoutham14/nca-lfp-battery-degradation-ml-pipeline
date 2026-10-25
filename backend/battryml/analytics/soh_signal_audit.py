from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
import statistics
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "time",
    "serial",
    "batt_soh_pct",
    "batt_soc_pct",
    "batt_current_a",
    "batt_voltage_v",
)


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _validate_serial(serial: str) -> str:
    value = str(serial).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(
            "serial may contain only letters, numbers, underscores, and hyphens"
        )
    return value


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _compression(value: str) -> str | None:
    return None if value == "none" else value


def _daily_query(
    files: list[Path],
    *,
    serial: str,
    start: datetime | None,
    end: datetime | None,
    rest_current_threshold_a: float,
    gap_threshold_seconds: float,
    max_integration_gap_seconds: float,
    expected_sample_interval_seconds: float,
) -> str:
    parquet_files = ", ".join(_sql_string(path.as_posix()) for path in files)
    filters = [f"CAST(serial AS VARCHAR) = {_sql_string(serial)}", "time IS NOT NULL"]
    if start is not None:
        filters.append(f"time >= TIMESTAMPTZ {_sql_string(start.isoformat())}")
    if end is not None:
        filters.append(f"time < TIMESTAMPTZ {_sql_string(end.isoformat())}")
    where_clause = " AND ".join(filters)
    threshold = float(rest_current_threshold_a)
    gap_threshold = float(gap_threshold_seconds)
    max_gap = float(max_integration_gap_seconds)
    expected_interval = float(expected_sample_interval_seconds)

    return f"""
        WITH typed AS (
            SELECT
                CAST(time AS TIMESTAMPTZ) AS sample_time,
                TRY_CAST(batt_soh_pct AS DOUBLE) AS soh_pct,
                TRY_CAST(batt_soc_pct AS DOUBLE) AS soc_pct,
                TRY_CAST(batt_current_a AS DOUBLE) AS current_a,
                TRY_CAST(batt_voltage_v AS DOUBLE) AS voltage_v
            FROM read_parquet(
                [{parquet_files}],
                union_by_name = true,
                hive_partitioning = false
            )
            WHERE {where_clause}
        ),
        timed AS (
            SELECT
                *,
                EXTRACT(EPOCH FROM sample_time - LAG(sample_time) OVER (
                    ORDER BY sample_time
                )) AS interval_seconds
            FROM typed
        ),
        daily AS (
            SELECT
                CAST(sample_time AS DATE) AS day_date,
                MIN(sample_time) AS first_sample_time,
                MAX(sample_time) AS end_anchor_time,
                COUNT(*) AS sample_count,
                COUNT(*) FILTER (WHERE interval_seconds = 0) AS duplicate_timestamp_rows,
                COUNT(*) FILTER (WHERE interval_seconds > {gap_threshold}) AS gap_count,
                MAX(interval_seconds) AS max_gap_seconds,
                COUNT(*) FILTER (WHERE soh_pct IS NULL) AS null_soh_rows,
                COUNT(*) FILTER (WHERE soc_pct IS NULL) AS null_soc_rows,
                COUNT(*) FILTER (WHERE current_a IS NULL) AS null_current_rows,
                COUNT(*) FILTER (WHERE voltage_v IS NULL) AS null_voltage_rows,
                COUNT(*) FILTER (WHERE soh_pct < 0 OR soh_pct > 100) AS invalid_soh_rows,
                COUNT(*) FILTER (WHERE soc_pct < 0 OR soc_pct > 100) AS invalid_soc_rows,
                ARG_MIN(soh_pct, sample_time) FILTER (WHERE soh_pct IS NOT NULL) AS first_soh_pct,
                ARG_MAX(soh_pct, sample_time) FILTER (WHERE soh_pct IS NOT NULL) AS last_soh_pct,
                MODE(soh_pct) FILTER (WHERE soh_pct IS NOT NULL) AS measured_soh_pct,
                MEDIAN(soh_pct) FILTER (WHERE soh_pct IS NOT NULL) AS median_soh_pct,
                AVG(soh_pct) FILTER (WHERE soh_pct IS NOT NULL) AS mean_soh_pct,
                MIN(soh_pct) AS min_soh_pct,
                MAX(soh_pct) AS max_soh_pct,
                MIN(soc_pct) AS min_soc_pct,
                MAX(soc_pct) AS max_soc_pct,
                AVG(soc_pct) AS mean_soc_pct,
                AVG(CASE WHEN soc_pct >= 95 THEN 1.0 ELSE 0.0 END) AS high_soc_fraction,
                AVG(CASE WHEN soc_pct <= 5 THEN 1.0 ELSE 0.0 END) AS low_soc_fraction,
                MIN(voltage_v) AS min_voltage_v,
                MAX(voltage_v) AS max_voltage_v,
                AVG(voltage_v) AS mean_voltage_v,
                MIN(current_a) AS min_current_a,
                MAX(current_a) AS max_current_a,
                SQRT(AVG(current_a * current_a)) AS rms_current_a,
                SUM(
                    CASE WHEN current_a < -{threshold}
                        THEN -current_a * LEAST(
                            GREATEST(COALESCE(interval_seconds, {expected_interval}), 0),
                            {max_gap}
                        ) / 3600.0
                        ELSE 0.0
                    END
                ) AS discharge_ah,
                SUM(
                    CASE WHEN current_a > {threshold}
                        THEN current_a * LEAST(
                            GREATEST(COALESCE(interval_seconds, {expected_interval}), 0),
                            {max_gap}
                        ) / 3600.0
                        ELSE 0.0
                    END
                ) AS charge_ah,
                LEAST(
                    1.0,
                    COUNT(*) * {expected_interval} /
                    GREATEST(
                        EXTRACT(EPOCH FROM MAX(sample_time) - MIN(sample_time))
                            + {expected_interval},
                        {expected_interval}
                    )
                ) AS coverage_fraction
            FROM timed
            GROUP BY CAST(sample_time AS DATE)
        )
        SELECT * FROM daily ORDER BY day_date
    """


def _derive_daily_records(
    rows: list[dict[str, Any]],
    *,
    nominal_capacity_ah: float | None,
    step_threshold_pct: float,
    unstable_daily_range_pct: float,
    min_label_coverage: float,
    max_transition_gap_days: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    daily: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    previous_soh: float | None = None
    previous_day: date | None = None
    cumulative_discharge_ah = 0.0

    for source in rows:
        row = dict(source)
        day_value = row["day_date"]
        day_date = (
            day_value
            if isinstance(day_value, date)
            else date.fromisoformat(str(day_value)[:10])
        )
        soh = _finite(row.get("measured_soh_pct"))
        discharge_ah = _finite(row.get("discharge_ah")) or 0.0
        cumulative_discharge_ah += max(0.0, discharge_ah)
        days_since_previous = (day_date - previous_day).days if previous_day else None
        soh_delta = (
            soh - previous_soh if soh is not None and previous_soh is not None else None
        )
        daily_range = None
        if row.get("min_soh_pct") is not None and row.get("max_soh_pct") is not None:
            daily_range = float(row["max_soh_pct"]) - float(row["min_soh_pct"])
        unstable = bool(
            (daily_range is not None and daily_range >= unstable_daily_range_pct)
            or int(row.get("invalid_soh_rows") or 0) > 0
        )
        coverage = _finite(row.get("coverage_fraction")) or 0.0
        label_usable = bool(
            soh is not None
            and coverage >= min_label_coverage
            and not unstable
            and int(row.get("null_soh_rows") or 0) == 0
        )
        transition_type = None
        if soh_delta is not None and abs(soh_delta) >= step_threshold_pct:
            if (
                days_since_previous is not None
                and days_since_previous > max_transition_gap_days
            ):
                transition_type = "change_after_gap"
            elif soh_delta < 0:
                transition_type = "step_drop"
            else:
                transition_type = "step_recovery"

        row.update(
            {
                "day_date": day_date,
                "soh_delta_pct": soh_delta,
                "days_since_previous": days_since_previous,
                "daily_soh_range_pct": daily_range,
                "cumulative_all_discharge_ah": cumulative_discharge_ah,
                "capacity_ah": (
                    nominal_capacity_ah * soh / 100.0
                    if nominal_capacity_ah is not None and soh is not None
                    else None
                ),
                "bms_soh_end": _finite(row.get("last_soh_pct")),
                "valid_training_row": label_usable,
                "transition_type": transition_type,
            }
        )
        daily.append(row)
        if transition_type is not None:
            changes.append(
                {
                    "event_time": row.get("first_sample_time"),
                    "day_date": day_date,
                    "event_type": transition_type,
                    "previous_soh_pct": previous_soh,
                    "new_soh_pct": soh,
                    "soh_delta_pct": soh_delta,
                    "days_since_previous": days_since_previous,
                    "daily_soh_range_pct": daily_range,
                    "coverage_fraction": coverage,
                    "discharge_ah_day": discharge_ah,
                    "cumulative_all_discharge_ah": cumulative_discharge_ah,
                    "mean_soc_pct": _finite(row.get("mean_soc_pct")),
                    "min_soc_pct": _finite(row.get("min_soc_pct")),
                    "max_soc_pct": _finite(row.get("max_soc_pct")),
                    "mean_voltage_v": _finite(row.get("mean_voltage_v")),
                    "rms_current_a": _finite(row.get("rms_current_a")),
                    "max_gap_seconds": _finite(row.get("max_gap_seconds")),
                }
            )
        if soh is not None:
            previous_soh = soh
            previous_day = day_date
    return daily, changes


def _audit_summary(
    daily: list[dict[str, Any]],
    changes: list[dict[str, Any]],
    *,
    serial: str,
    nominal_capacity_ah: float | None,
    available_columns: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    soh_days = [row for row in daily if row.get("measured_soh_pct") is not None]
    coverage_values = [
        float(row["coverage_fraction"])
        for row in daily
        if row.get("coverage_fraction") is not None
    ]
    first_soh = _finite(soh_days[0]["measured_soh_pct"]) if soh_days else None
    last_soh = _finite(soh_days[-1]["measured_soh_pct"]) if soh_days else None
    net_soh_change = (
        last_soh - first_soh if first_soh is not None and last_soh is not None else None
    )
    drops = [row for row in changes if row["event_type"] == "step_drop"]
    recoveries = [row for row in changes if row["event_type"] == "step_recovery"]
    gap_changes = [row for row in changes if row["event_type"] == "change_after_gap"]
    unstable_days = [
        row
        for row in daily
        if (row.get("daily_soh_range_pct") or 0) >= config["unstable_daily_range_pct"]
        or int(row.get("invalid_soh_rows") or 0) > 0
    ]
    total_discharge_ah = sum(float(row.get("discharge_ah") or 0) for row in daily)
    total_charge_ah = sum(float(row.get("charge_ah") or 0) for row in daily)
    bms_loss_pp_per_1000ah = (
        -net_soh_change / (total_discharge_ah / 1000.0)
        if net_soh_change is not None and total_discharge_ah > 0
        else None
    )
    equivalent_capacity_loss = (
        bms_loss_pp_per_1000ah * nominal_capacity_ah / 100.0
        if bms_loss_pp_per_1000ah is not None and nominal_capacity_ah is not None
        else None
    )

    electrothermal_available = any(
        "cellvoltage" in name for name in available_columns
    ) and any("temperature" in name or "_temp" in name for name in available_columns)
    missing_research_signals = []
    if "batt_modulecount" not in available_columns:
        missing_research_signals.append("module count")
    if not any("cellvoltage" in name for name in available_columns):
        missing_research_signals.append("per-cell voltage")
    if not any("temperature" in name or "_temp" in name for name in available_columns):
        missing_research_signals.append("module and inverter temperature")
    if not any("availablechargecurrent" in name for name in available_columns):
        missing_research_signals.append("charge/discharge limits")
    if not any("error" in name or "fault" in name for name in available_columns):
        missing_research_signals.append("error and diagnostic state")

    median_coverage = _median(coverage_values)
    if not soh_days or median_coverage is None or median_coverage < 0.5:
        signal_quality_status = "unusable"
        label_status = "unusable"
    elif recoveries or gap_changes or unstable_days:
        signal_quality_status = "unstable_bms_estimate"
        label_status = "weak_label_only"
    elif drops:
        signal_quality_status = "stepwise_bms_estimate"
        label_status = "weak_label_only"
    else:
        signal_quality_status = "stable_observed_bms_estimate"
        label_status = "weak_label_only"

    findings: list[str] = []
    if drops:
        findings.append(
            f"Detected {len(drops)} stable day-to-day SOH drops; the largest was "
            f"{abs(min(float(row['soh_delta_pct']) for row in drops)):.2f} percentage points."
        )
    if recoveries:
        findings.append(
            f"Detected {len(recoveries)} SOH recoveries. Reversible SOH movement cannot be "
            "irreversible cell capacity loss."
        )
    if unstable_days:
        findings.append(
            f"Detected {len(unstable_days)} days with a wide or invalid SOH range, consistent "
            "with transient telemetry or BMS state changes."
        )
    if missing_research_signals:
        findings.append(
            "The current import cannot explain electrothermal causes because it omits: "
            + ", ".join(missing_research_signals)
            + "."
        )
    if not changes and soh_days:
        findings.append(
            "No stable SOH step was observed in this window. A flat BMS estimate is not "
            "independent evidence that physical capacity is unchanged."
        )
    findings.append(
        "BMS SOH is retained as a noisy auxiliary target. It is not accepted as true SOH "
        "without an independent capacity or reference label."
    )

    raw_rows = sum(int(row.get("sample_count") or 0) for row in daily)
    return {
        "model_family": "bms_soh_signal_audit",
        "serial": serial,
        "created_at": datetime.now(UTC).isoformat(),
        "nominal_capacity_ah": nominal_capacity_ah,
        "nominal_capacity_source": "battery_profile" if nominal_capacity_ah else None,
        "raw_rows": raw_rows,
        "daily_rows": len(daily),
        "training_rows": sum(bool(row.get("valid_training_row")) for row in daily),
        "forecast_rows": 0,
        "observed_start": _iso(daily[0].get("first_sample_time")) if daily else None,
        "observed_end": _iso(daily[-1].get("end_anchor_time")) if daily else None,
        "available_signal_count": len(available_columns),
        "available_signals": available_columns,
        "electrothermal_signals_available": electrothermal_available,
        "missing_research_signals": missing_research_signals,
        "audit_configuration": config,
        "audit_metrics": {
            "first_bms_soh_pct": first_soh,
            "last_bms_soh_pct": last_soh,
            "net_bms_soh_change_pct": net_soh_change,
            "step_drop_count": len(drops),
            "step_recovery_count": len(recoveries),
            "change_after_gap_count": len(gap_changes),
            "largest_step_drop_pct": (
                min(float(row["soh_delta_pct"]) for row in drops) if drops else None
            ),
            "unstable_soh_days": len(unstable_days),
            "median_coverage_fraction": median_coverage,
            "total_discharge_ah": total_discharge_ah,
            "total_charge_ah": total_charge_ah,
            "observed_bms_loss_pp_per_1000ah": bms_loss_pp_per_1000ah,
            "equivalent_capacity_loss_ah_per_1000ah": equivalent_capacity_loss,
            "signal_quality_status": signal_quality_status,
            "ground_truth_status": "missing_independent_capacity_label",
            "label_status": label_status,
        },
        "findings": findings,
        "label_policy": {
            "bms_soh_role": "noisy_auxiliary_target",
            "ground_truth_required": True,
            "telemetry_screening_is_not_ground_truth": True,
            "acceptable_ground_truth": [
                "controlled full or reference capacity measurement",
                "trusted BMS recalibration event with documented firmware semantics",
                "laboratory capacity label for matched chemistry and hardware",
            ],
        },
    }


def _event_table_html(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return '<p class="empty">No stable SOH step larger than the configured threshold.</p>'
    rows = []
    for item in changes[:100]:
        tone = "drop" if item["event_type"] == "step_drop" else "recovery"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['day_date']))}</td>"
            f'<td><span class="tag {tone}">{html.escape(item["event_type"])}</span></td>'
            f"<td>{item['previous_soh_pct']:.1f}%</td>"
            f"<td>{item['new_soh_pct']:.1f}%</td>"
            f"<td>{item['soh_delta_pct']:+.1f} pp</td>"
            f"<td>{item['discharge_ah_day']:.1f} Ah</td>"
            f"<td>{item['coverage_fraction']:.1%}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Date</th><th>Class</th>'
        "<th>Previous</th><th>New</th><th>Change</th><th>Day discharge</th>"
        "<th>Coverage</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _chart_svg(daily: list[dict[str, Any]], changes: list[dict[str, Any]]) -> str:
    points = [row for row in daily if _finite(row.get("measured_soh_pct")) is not None]
    if len(points) < 2:
        return '<p class="empty">Not enough SOH observations for a chart.</p>'
    width, height = 1120, 330
    left, right, top, bottom = 58, 18, 20, 46
    values = [float(row["measured_soh_pct"]) for row in points]
    y_min = min(75.0, math.floor(min(values) - 2.0))
    y_max = max(102.0, math.ceil(max(values) + 1.0))

    def x_at(index: int) -> float:
        return left + index * (width - left - right) / max(1, len(points) - 1)

    def y_at(value: float) -> float:
        return top + (y_max - value) * (height - top - bottom) / (y_max - y_min)

    polyline = " ".join(
        f"{x_at(index):.1f},{y_at(float(row['measured_soh_pct'])):.1f}"
        for index, row in enumerate(points)
    )
    event_days = {str(item["day_date"]): item["event_type"] for item in changes}
    markers = []
    for index, row in enumerate(points):
        event_type = event_days.get(str(row["day_date"]))
        if event_type:
            color = "#b63e35" if event_type == "step_drop" else "#2f7b59"
            markers.append(
                f'<circle cx="{x_at(index):.1f}" cy="{y_at(float(row["measured_soh_pct"])):.1f}" '
                f'r="4.5" fill="{color}"><title>{html.escape(str(row["day_date"]))}: '
                f"{html.escape(event_type)}</title></circle>"
            )
    grid = []
    for value in range(int(math.ceil(y_min / 5) * 5), int(y_max) + 1, 5):
        y = y_at(float(value))
        grid.append(
            f'<line x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#e4e8ea"/><text x="48" y="'
            f'{y + 4:.1f}" text-anchor="end">{value}%</text>'
        )
    start_label = html.escape(str(points[0]["day_date"]))
    end_label = html.escape(str(points[-1]["day_date"]))
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Daily BMS SOH">'
        + "".join(grid)
        + f'<line x1="{left}" x2="{width - right}" y1="{height - bottom}" y2="{height - bottom}" '
        'stroke="#78868c"/><polyline fill="none" stroke="#176b87" stroke-width="2.2" '
        f'points="{polyline}"/>{"".join(markers)}'
        f'<text x="{left}" y="{height - 15}">{start_label}</text>'
        f'<text x="{width - right}" y="{height - 15}" text-anchor="end">{end_label}</text>'
        "</svg>"
    )


def _write_html_report(
    path: Path,
    summary: dict[str, Any],
    daily: list[dict[str, Any]],
    changes: list[dict[str, Any]],
) -> None:
    metrics = summary["audit_metrics"]
    findings = "".join(f"<li>{html.escape(item)}</li>" for item in summary["findings"])
    missing = ", ".join(summary["missing_research_signals"]) or "none"
    loss = metrics.get("observed_bms_loss_pp_per_1000ah")
    loss_text = "n/a" if loss is None else f"{loss:.3f} pp"
    coverage = metrics.get("median_coverage_fraction")
    coverage_text = "n/a" if coverage is None else f"{coverage:.1%}"
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Battery {html.escape(summary["serial"])} SOH signal audit</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f4f6f7;color:#26343d;font:14px Arial,sans-serif}}
main{{max-width:1240px;margin:auto;padding:32px}} header{{display:flex;justify-content:space-between;gap:24px;align-items:end}}
h1{{font-size:30px;margin:4px 0}} h2{{font-size:18px;margin:0 0 14px}} .eyebrow{{color:#176b87;font-weight:700;text-transform:uppercase;font-size:12px}}
.status{{padding:8px 10px;border-radius:4px;background:#f5dfb8;color:#6d4a0b;font-weight:700}}
.metrics{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:1px;background:#dbe1e4;border:1px solid #dbe1e4;margin:28px 0}}
.metric{{background:white;padding:16px}} .metric small{{display:block;color:#6b777d;margin-bottom:8px}} .metric strong{{font-size:21px}}
section{{background:white;border:1px solid #dbe1e4;margin:18px 0;padding:22px}} svg{{display:block;width:100%;height:auto}} svg text{{font-size:12px;fill:#68767d}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse}} th,td{{padding:10px;border-bottom:1px solid #e4e8ea;text-align:left;white-space:nowrap}} th{{font-size:12px;color:#637078}}
.tag{{padding:4px 6px;border-radius:3px;font-size:12px}} .drop{{background:#f7d8d5;color:#8d2d27}} .recovery{{background:#dcece3;color:#235e43}}
li{{margin:8px 0;line-height:1.45}} .notice{{border-left:4px solid #d99018;padding:12px 14px;background:#fff8eb;line-height:1.5}} .empty{{color:#6b777d}}
@media(max-width:800px){{main{{padding:18px}} header{{display:block}} .metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<header><div><span class="eyebrow">BatteryML evidence report</span><h1>Battery {html.escape(summary["serial"])}</h1><div>{html.escape(str(summary["observed_start"]))} to {html.escape(str(summary["observed_end"]))}</div></div><span class="status">{html.escape(metrics["label_status"].replace("_", " "))}</span></header>
<div class="metrics"><div class="metric"><small>First BMS SOH</small><strong>{metrics.get("first_bms_soh_pct")}%</strong></div><div class="metric"><small>Latest BMS SOH</small><strong>{metrics.get("last_bms_soh_pct")}%</strong></div><div class="metric"><small>Stable drops</small><strong>{metrics["step_drop_count"]}</strong></div><div class="metric"><small>Recoveries</small><strong>{metrics["step_recovery_count"]}</strong></div><div class="metric"><small>Median coverage</small><strong>{coverage_text}</strong></div></div>
<section><h2>Daily BMS SOH and detected transitions</h2>{_chart_svg(daily, changes)}</section>
<section><h2>Stable transition evidence</h2>{_event_table_html(changes)}</section>
<section><h2>What the evidence supports</h2><ul>{findings}</ul><p class="notice"><strong>Observed association:</strong> {loss_text} BMS-SOH movement per 1,000 discharged Ah. This is not a learned true-capacity degradation law while BMS SOH remains the target.</p></section>
<section><h2>Research signal readiness</h2><p>Available columns: {summary["available_signal_count"]}. Missing from this import: {html.escape(missing)}.</p></section>
</main></body></html>"""
    path.write_text(document, encoding="utf-8")


def run_audit(
    *,
    serial: str,
    raw_root: Path,
    output_root: Path,
    nominal_capacity_ah: float | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    rest_current_threshold_a: float = 0.5,
    gap_threshold_seconds: float = 60.0,
    max_integration_gap_seconds: float = 5.0,
    expected_sample_interval_seconds: float = 1.0,
    step_threshold_pct: float = 0.5,
    unstable_daily_range_pct: float = 5.0,
    min_label_coverage: float = 0.8,
    max_transition_gap_days: int = 7,
    compression: str | None = "zstd",
    run_id: str | None = None,
) -> Path:
    serial = _validate_serial(serial)
    if nominal_capacity_ah is not None and nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive")
    if step_threshold_pct <= 0 or unstable_daily_range_pct <= 0:
        raise ValueError("SOH thresholds must be positive")
    if not 0 <= min_label_coverage <= 1:
        raise ValueError("min_label_coverage must be between 0 and 1")
    if max_transition_gap_days < 1:
        raise ValueError("max_transition_gap_days must be at least 1")

    input_root = raw_root / f"serial={serial}"
    files = sorted(input_root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw Parquet files found under {input_root}")

    run_stamp = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"serial={serial}" / f"audit_run={run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar = false")
    parquet_files = ", ".join(_sql_string(path.as_posix()) for path in files)
    schema_rows = connection.execute(
        "DESCRIBE SELECT * FROM read_parquet("
        f"[{parquet_files}], union_by_name = true, hive_partitioning = false)"
    ).fetchall()
    available_columns = [str(row[0]) for row in schema_rows]
    missing = sorted(set(REQUIRED_COLUMNS) - set(available_columns))
    if missing:
        raise ValueError(
            f"Raw telemetry is missing required columns: {', '.join(missing)}"
        )

    query = _daily_query(
        files,
        serial=serial,
        start=start,
        end=end,
        rest_current_threshold_a=rest_current_threshold_a,
        gap_threshold_seconds=gap_threshold_seconds,
        max_integration_gap_seconds=max_integration_gap_seconds,
        expected_sample_interval_seconds=expected_sample_interval_seconds,
    )
    logger.info("Aggregating %s raw files for battery %s", len(files), serial)
    result = connection.execute(query)
    columns = [str(item[0]) for item in result.description]
    rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    connection.close()
    if not rows:
        raise ValueError("No telemetry remained after applying the requested range")

    config = {
        "rest_current_threshold_a": rest_current_threshold_a,
        "gap_threshold_seconds": gap_threshold_seconds,
        "max_integration_gap_seconds": max_integration_gap_seconds,
        "expected_sample_interval_seconds": expected_sample_interval_seconds,
        "step_threshold_pct": step_threshold_pct,
        "unstable_daily_range_pct": unstable_daily_range_pct,
        "min_label_coverage": min_label_coverage,
        "max_transition_gap_days": max_transition_gap_days,
        "start": _iso(start),
        "end": _iso(end),
    }
    daily, changes = _derive_daily_records(
        rows,
        nominal_capacity_ah=nominal_capacity_ah,
        step_threshold_pct=step_threshold_pct,
        unstable_daily_range_pct=unstable_daily_range_pct,
        min_label_coverage=min_label_coverage,
        max_transition_gap_days=max_transition_gap_days,
    )
    summary = _audit_summary(
        daily,
        changes,
        serial=serial,
        nominal_capacity_ah=nominal_capacity_ah,
        available_columns=available_columns,
        config=config,
    )

    pq.write_table(
        pa.Table.from_pylist(daily),
        run_dir / "daily_health.parquet",
        compression=compression,
    )
    event_schema = pa.schema(
        [
            ("event_time", pa.timestamp("us", tz="UTC")),
            ("day_date", pa.date32()),
            ("event_type", pa.string()),
            ("previous_soh_pct", pa.float64()),
            ("new_soh_pct", pa.float64()),
            ("soh_delta_pct", pa.float64()),
            ("days_since_previous", pa.int64()),
            ("daily_soh_range_pct", pa.float64()),
            ("coverage_fraction", pa.float64()),
            ("discharge_ah_day", pa.float64()),
            ("cumulative_all_discharge_ah", pa.float64()),
            ("mean_soc_pct", pa.float64()),
            ("min_soc_pct", pa.float64()),
            ("max_soc_pct", pa.float64()),
            ("mean_voltage_v", pa.float64()),
            ("rms_current_a", pa.float64()),
            ("max_gap_seconds", pa.float64()),
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(changes, schema=event_schema),
        run_dir / "soh_change_events.parquet",
        compression=compression,
    )
    (run_dir / "model_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_html_report(
        run_dir / "soh_signal_audit_report.html", summary, daily, changes
    )
    logger.info("SOH signal audit written to %s", run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit BMS SOH quality, step changes, and throughput context."
    )
    parser.add_argument("--serial", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nominal-capacity-ah", type=float)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--rest-current-threshold-a", type=float, default=0.5)
    parser.add_argument("--gap-threshold-seconds", type=float, default=60.0)
    parser.add_argument("--max-integration-gap-seconds", type=float, default=5.0)
    parser.add_argument("--expected-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--step-threshold-pct", type=float, default=0.5)
    parser.add_argument("--unstable-daily-range-pct", type=float, default=5.0)
    parser.add_argument("--min-label-coverage", type=float, default=0.8)
    parser.add_argument("--max-transition-gap-days", type=int, default=7)
    parser.add_argument(
        "--compression", choices=("snappy", "zstd", "gzip", "none"), default="zstd"
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
        run_dir = run_audit(
            serial=args.serial,
            raw_root=args.raw_root,
            output_root=args.output_root,
            nominal_capacity_ah=args.nominal_capacity_ah,
            start=_parse_timestamp(args.start),
            end=_parse_timestamp(args.end),
            rest_current_threshold_a=args.rest_current_threshold_a,
            gap_threshold_seconds=args.gap_threshold_seconds,
            max_integration_gap_seconds=args.max_integration_gap_seconds,
            expected_sample_interval_seconds=args.expected_sample_interval_seconds,
            step_threshold_pct=args.step_threshold_pct,
            unstable_daily_range_pct=args.unstable_daily_range_pct,
            min_label_coverage=args.min_label_coverage,
            max_transition_gap_days=args.max_transition_gap_days,
            compression=_compression(args.compression),
        )
        print(run_dir)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("SOH signal audit failed")
        else:
            logger.error("SOH signal audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
