"""Compare marginal stress exposures: reference (cycle tester) vs field devices.

Goal: find which operating-stress dimensions actually separate field devices from
the controlled full-cycle reference (100000017), to explain faster field
degradation. Unlike the strict co-occurrence "dangerous scenario" flags (which
are ~0 everywhere), these marginal exposures are populated and comparable.

For each device's LATEST seasonal run, aggregate monthly_operating_profile over
all qualified months and print totals/means of the key stress dimensions.

Usage:
    python tests/compare_field_vs_reference_stress.py [root]

Not a pytest test and not part of the pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

REFERENCE = "100000017"

# (label, column, aggregation)
METRICS = [
    ("full_cycles", "discharge_equivalent_full_cycles", "sum"),
    ("discharge_Ah", "discharge_ah", "sum"),
    ("SOC>=95_h", "soc_95_100_hours", "sum"),
    ("SOC80-95_h", "soc_80_95_hours", "sum"),
    ("SOC<=5_h", "soc_0_5_hours", "sum"),
    ("Tmean_C", "temperature_mean_c", "mean"),
    ("T>=30_h", "temperature_30_40_hours", "sum"),
    ("hot35_h", "hot_35_hours", "sum"),
    ("Tmax_C", "temperature_max_c", "max"),
    ("hiI_disch_h", "high_current_discharging_hours", "sum"),
    ("hiI_charg_h", "high_current_charging_hours", "sum"),
    ("C2_h", "c2_hours", "sum"),
    ("C3_h", "c3_hours", "sum"),
    ("spreadP95_mV", "voltage_spread_p95_daily_mv", "mean"),
    ("chg>=95SOC_h", "charging_above_95_soc_hours", "sum"),
    ("hiI+hot_h", "high_current_hot_hours", "sum"),
]


def latest_run(serial_dir: Path) -> Path | None:
    runs = sorted(serial_dir.glob("profile_run=*"))
    return runs[-1] if runs else None


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/processed/seasonal_charge_analysis"
    )
    if not root.is_dir():
        print(f"no such root: {root}")
        return 2

    rows = []
    for serial_dir in sorted(root.glob("serial=*")):
        serial = serial_dir.name.split("=")[-1]
        run = latest_run(serial_dir)
        if run is None:
            continue
        p = run / "monthly_operating_profile.parquet"
        if not p.exists():
            continue
        df = pq.read_table(p).to_pandas()
        # only qualified months
        if "seasonal_profile_valid" in df.columns:
            df = df[df["seasonal_profile_valid"].fillna(False)]
        if df.empty:
            continue
        rec = {"serial": serial, "months": len(df)}
        for label, col, agg in METRICS:
            if col not in df.columns:
                rec[label] = float("nan")
                continue
            s = df[col]
            rec[label] = float(
                s.sum() if agg == "sum" else (s.mean() if agg == "mean" else s.max())
            )
        rows.append(rec)

    if not rows:
        print("no data")
        return 2

    # reference first, then field devices
    rows.sort(key=lambda r: (r["serial"] != REFERENCE, r["serial"]))
    labels = ["serial", "months"] + [m[0] for m in METRICS]
    header = "  ".join(f"{l:>13}" for l in labels)
    print(header)
    print("-" * len(header))
    for r in rows:
        vals = []
        for l in labels:
            v = r.get(l)
            if isinstance(v, str):
                vals.append(f"{v:>13}")
            elif isinstance(v, int):
                vals.append(f"{v:>13}")
            else:
                vals.append(f"{v:>13.1f}")
        print("  ".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())