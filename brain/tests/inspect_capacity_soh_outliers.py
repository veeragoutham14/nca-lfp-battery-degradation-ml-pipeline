"""Diagnostic: list SOH-event rows whose capacity-based SOH estimate is physically implausible.

The seasonal analysis carries usable-capacity SOH estimates from the BMS SOH
recalibration detector. A value above 100% is physically impossible and marks an
unreliable estimate (partial-discharge SOC extrapolation or a bad traversal).
This tool lists those rows so they can be excluded or inspected before use.

Usage:
    python tests/inspect_capacity_soh_outliers.py <seasonal_run_dir> [threshold_pct]

Example:
    python tests/inspect_capacity_soh_outliers.py \
        "data/processed/seasonal_charge_analysis/serial=100000001/profile_run=20260818T093852Z" 100

Not a pytest test and not part of the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

COLUMNS = [
    "event_time_utc",
    "new_soh_pct",
    "recalibration_discharged_ah_1s",
    "recalibration_raw_soc_drop_pct",
    "recalibration_raw_coverage_fraction",
    "recalibration_maximum_raw_gap_seconds",
    "recalibration_direct_capacity_soh_pct",
    "recalibration_soc_normalized_capacity_soh_pct",
    "recalibration_capacity_validation_status",
]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    run_dir = Path(sys.argv[1])
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

    context_path = run_dir / "soh_transition_context.parquet"
    if not context_path.exists():
        print(f"no soh_transition_context.parquet under {run_dir}")
        return 2

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    df = pq.read_table(context_path).to_pandas()

    summary_path = run_dir / "model_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print(
            f"serial={summary.get('serial')} nominal_capacity_ah="
            f"{summary.get('nominal_capacity_ah')} module_count={summary.get('module_count')}"
        )

    metric = "recalibration_soc_normalized_capacity_soh_pct"
    if metric not in df.columns:
        print(f"column {metric} not present")
        return 2

    bad = df[df[metric] > threshold]
    present = [c for c in COLUMNS if c in df.columns]
    print(f"\nrows with {metric} > {threshold:g}: {len(bad)} of {len(df)}")
    if len(bad):
        print(bad[present].to_string(index=False))
        statuses = bad["recalibration_capacity_validation_status"].value_counts().to_dict()
        print(f"\ncapacity_validation_status breakdown: {statuses}")
    else:
        print("no physically implausible capacity-SOH estimates found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())