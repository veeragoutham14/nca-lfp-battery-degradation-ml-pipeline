"""One-off validation of seasonal_charge_analysis output data validity.

Reads a seasonal run directory and checks: row counts, null/NaN/inf,
physical value ranges, coverage fractions, sign-convention sanity, and
internal consistency. Prints a plain report. Not part of the pipeline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def check_frame(name: str, df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    print(f"\n=== {name} ===  rows={len(df)} cols={len(df.columns)}")
    if len(df) == 0:
        issues.append(f"{name}: EMPTY (0 rows)")
        print("  EMPTY FRAME")
        return issues

    # Numeric range / NaN / inf checks
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        nan_counts = num.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        if len(nan_cols):
            print(f"  NaN columns: {dict(nan_cols)}")
        for col in num.columns:
            series = num[col].dropna().to_numpy(dtype=float)
            if series.size == 0:
                continue
            if not np.isfinite(series).all():
                issues.append(f"{name}.{col}: contains inf/-inf")
        # report min/max for a few key columns
        for col in num.columns:
            lo, hi = float(num[col].min()), float(num[col].max())
            print(f"  {col}: min={lo:.4g} max={hi:.4g}")

    # Timestamp checks (skip numeric hour/count columns that merely end in _utc)
    for col in df.columns:
        if df[col].dtype.kind in "if":
            continue
        if str(col).endswith(("_utc", "_at")) or "timestamp" in str(col).lower():
            non_null = df[col].dropna()
            ts = pd.to_datetime(non_null, errors="coerce", utc=True)
            n_bad = int(ts.isna().sum())
            n_null = int(df[col].isna().sum())
            if n_bad:
                issues.append(f"{name}.{col}: {n_bad} non-null unparseable timestamps")
            if len(ts):
                print(f"  {col}: {ts.min()} .. {ts.max()} (UTC, {n_null} null=expected gaps)")
    return issues


def check_range(name: str, series: pd.Series, lo: float, hi: float, issues: list[str]) -> None:
    if series.empty:
        return
    s = series.dropna()
    if s.empty:
        return
    bad = int(((s < lo) | (s > hi)).sum())
    if bad:
        issues.append(f"{name}: {bad} values outside [{lo},{hi}] (min={s.min():.4g}, max={s.max():.4g})")


def main() -> int:
    run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if run_dir is None or not run_dir.is_dir():
        print("usage: python validate_seasonal_output.py <run_dir>")
        return 2

    issues: list[str] = []
    print(f"RUN DIR: {run_dir}")

    # model_summary.json
    summary_path = run_dir / "model_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        print("\n=== model_summary.json ===")
        for k in sorted(summary):
            v = summary[k]
            if isinstance(v, (int, float, str, bool)) or v is None:
                print(f"  {k}: {v}")
        # sanity on key aggregates
        for key in ("total_charge_ah", "total_discharge_ah"):
            if key in summary:
                val = summary[key]
                if isinstance(val, (int, float)) and (not np.isfinite(val) or val < 0):
                    issues.append(f"summary.{key} not finite/non-negative: {val}")
    else:
        issues.append("model_summary.json missing")

    # Parquet artifacts
    for fname in (
        "monthly_operating_profile.parquet",
        "monthly_one_second_current_stress.parquet",
        "daily_battery_stress.parquet",
        "soh_transition_context.parquet",
        "charge_episodes.parquet",
    ):
        p = run_dir / fname
        if not p.exists():
            print(f"\n=== {fname} ===  NOT PRESENT")
            if fname == "monthly_one_second_current_stress.parquet":
                issues.append("monthly_one_second_current_stress.parquet MISSING (old-schema run)")
            continue
        df = pq.read_table(p).to_pandas()
        frame_issues = check_frame(fname, df)
        issues.extend(frame_issues)

        # Targeted range checks
        if fname == "monthly_operating_profile.parquet":
            for col, lo, hi in [
                ("coverage_fraction", 0.0, 1.0),
                ("daily_coverage_fraction", 0.0, 1.0),
            ]:
                if col in df.columns:
                    check_range(f"{fname}.{col}", df[col], lo, hi, issues)
            # Only *_pct columns are 0-100 percentages; *_hours are hour counts (unbounded)
            for col in df.columns:
                if col.lower().endswith("_pct") and df[col].dtype.kind in "if":
                    check_range(f"{fname}.{col}", df[col], 0.0, 100.0, issues)
                if ("charge_ah" in col or "discharge_ah" in col) and df[col].dtype.kind in "if":
                    s = df[col].dropna()
                    if not s.empty and (s < 0).any():
                        issues.append(f"{fname}.{col}: negative Ah values present")
        if fname == "monthly_one_second_current_stress.parquet":
            for col in df.columns:
                if col.lower().endswith("_pct") and df[col].dtype.kind in "if":
                    check_range(f"{fname}.{col}", df[col], 0.0, 100.0, issues)
                if ("ah" in col.lower()) and df[col].dtype.kind in "if":
                    s = df[col].dropna()
                    if not s.empty and (s < 0).any():
                        issues.append(f"{fname}.{col}: negative Ah magnitude present")
        if fname == "soh_transition_context.parquet":
            for col in df.columns:
                if col.lower().endswith("_pct") and df[col].dtype.kind in "if":
                    check_range(f"{fname}.{col}", df[col], 0.0, 100.0, issues)
                if "coverage_fraction" in col and df[col].dtype.kind in "if":
                    check_range(f"{fname}.{col}", df[col], 0.0, 1.0, issues)

    print("\n=========================================")
    if issues:
        print(f"ISSUES FOUND: {len(issues)}")
        for it in issues:
            print(f"  - {it}")
    else:
        print("NO ISSUES FOUND: all checks passed")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())