"""Fleet-level validity scan of seasonal_charge_analysis outputs.

For every serial, pick the LATEST profile_run and report:
  - schema: "new" if monthly_one_second_current_stress.parquet exists, else "OLD"
  - coverage: telemetry coverage range across qualified months
  - Ah sanity: total charge vs discharge from model_summary.json
  - capacity-SOH outliers: event rows with soc-normalized capacity SOH > threshold
    (physically impossible; all should carry a non-authoritative validation status)

Usage:
    python tests/scan_seasonal_fleet.py [root] [threshold_pct]
    root defaults to data/processed/seasonal_charge_analysis

Not a pytest test and not part of the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def latest_run(serial_dir: Path) -> Path | None:
    runs = sorted(serial_dir.glob("profile_run=*"))
    return runs[-1] if runs else None


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/processed/seasonal_charge_analysis"
    )
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    if not root.is_dir():
        print(f"no such root: {root}")
        return 2

    metric = "recalibration_soc_normalized_capacity_soh_pct"
    header = (
        f"{'serial':<11} {'schema':<5} {'months':<6} {'charge_Ah':>10} "
        f"{'discharge_Ah':>12} {'net_%':>6} {'capSOH>thr':>10} {'status_of_outliers'}"
    )
    print(header)
    print("-" * len(header))

    for serial_dir in sorted(root.glob("serial=*")):
        serial = serial_dir.name.split("=")[-1]
        run = latest_run(serial_dir)
        if run is None:
            print(f"{serial:<11} (no runs)")
            continue

        new_schema = (run / "monthly_one_second_current_stress.parquet").exists()
        schema = "new" if new_schema else "OLD"

        charge = discharge = float("nan")
        summary_path = run / "model_summary.json"
        months = ""
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            charge = float(s.get("total_charge_ah", float("nan")))
            discharge = float(s.get("total_discharge_ah", float("nan")))
            months = str(s.get("valid_monthly_profile_rows", ""))
        net_pct = (
            f"{100.0 * (charge - discharge) / charge:.1f}"
            if charge == charge and charge > 0 else "n/a"
        )

        outliers = 0
        statuses: dict[str, int] = {}
        ctx = run / "soh_transition_context.parquet"
        if ctx.exists():
            names = set(pq.read_schema(ctx).names)
            status_col = "recalibration_capacity_validation_status"
            if metric in names and status_col in names:
                df = pq.read_table(ctx, columns=[metric, status_col]).to_pandas()
                bad = df[df[metric] > threshold]
                outliers = int(len(bad))
                if outliers:
                    statuses = bad[status_col].value_counts().to_dict()

        status_txt = ",".join(f"{k}:{v}" for k, v in statuses.items()) if statuses else "-"
        print(
            f"{serial:<11} {schema:<5} {months:<6} {charge:>10.0f} "
            f"{discharge:>12.0f} {net_pct:>6} {outliers:>10} {status_txt}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())