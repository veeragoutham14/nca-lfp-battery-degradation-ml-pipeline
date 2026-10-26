#!/usr/bin/env python3
"""Build the small README figures from the synthetic brain run."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "brain"))
from src.config import install_local_parquet_reader  # noqa: E402

install_local_parquet_reader()

OUT = ROOT / "docs" / "images"
TEAL = "#176b78"
INK = "#202a2f"
GOLD = "#c78319"
BLUE = "#416fa5"
RED = "#bb5148"


def _latest_run() -> Path:
    runs = sorted(
        (ROOT / "brain" / "data" / "processed" / "capacity_ml" / "serial=100000017").glob(
            "ml_run=*"
        )
    )
    if not runs:
        raise SystemExit("No capacity_ml run found. Run scripts/demo.sh first.")
    return runs[-1]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor("#f7fafb")
    ax.figure.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#4d5b61")
    ax.grid(True, color="#e1e8ea", linewidth=0.8)
    ax.set_axisbelow(True)


def _stamp(fig: plt.Figure, text: str = "Synthetic data") -> None:
    fig.text(0.99, 0.01, text, ha="right", va="bottom", fontsize=8, color="#68767c")


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    image = Image.open(path).convert("RGB")
    image.save(path, optimize=True)


def plot_capacity_fade(run: Path) -> None:
    import json

    rows = pd.read_parquet(run / "ml_all_capacity_rows.parquet")
    summary = json.loads((run / "model_summary.json").read_text(encoding="utf-8"))
    model = summary["degradation_forecast_model"]
    x = rows["cumulative_all_discharge_ah"].to_numpy(dtype=float)
    y = rows["capacity_ah"].to_numpy(dtype=float)
    grid = np.linspace(x.min(), x.max(), 100)
    fitted = (
        float(model["baseline_capacity_ah"])
        - float(model["intercept_loss_ah"])
        - float(model["loss_slope_per_1000ah"])
        * (grid - float(model["reference_cumulative_ah"]))
        / 1000.0
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    ax.plot(grid, fitted, color=TEAL, linewidth=2.2, label="Usage-linear fit")
    ax.scatter(x, y, s=42, color=GOLD, zorder=3, label="Full 100% to 0% discharges")
    ax.set_xlabel("Cumulative discharge (Ah)")
    ax.set_ylabel("Measured capacity (Ah)")
    ax.set_title(
        "Capacity fade fit  ·  lab reference 100000017", loc="left", color=INK, fontsize=13, pad=10
    )
    ax.legend(frameon=False)
    _stamp(fig)
    _save(fig, "capacity_fade.png")


def plot_soh_forecast(run: Path) -> None:
    rows = pd.read_parquet(run / "ml_all_capacity_rows.parquet")
    forecast = pd.read_parquet(run / "capacity_forecast.parquet")
    forecast = forecast[forecast["scenario"] == "historical_mean"].sort_values("forecast_timestamp")
    nominal = 56.0
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    lower = forecast["predicted_capacity_ah_lower_95"] / nominal * 100.0
    upper = forecast["predicted_capacity_ah_upper_95"] / nominal * 100.0
    ax.fill_between(
        forecast["forecast_timestamp"],
        lower.clip(40, 110),
        upper.clip(40, 110),
        color="#8fd4cf",
        alpha=0.55,
        zorder=1,
        label="95% band",
    )
    ax.plot(
        forecast["forecast_timestamp"],
        lower,
        color="#0e6b66",
        linewidth=1.1,
        linestyle="--",
        zorder=2,
    )
    ax.plot(
        forecast["forecast_timestamp"],
        upper,
        color="#0e6b66",
        linewidth=1.1,
        linestyle="--",
        zorder=2,
    )
    ax.plot(
        forecast["forecast_timestamp"],
        forecast["predicted_future_soh_pct"],
        color=TEAL,
        linewidth=2.2,
        label="10-year forecast",
    )
    ax.scatter(
        rows["end_anchor_time"],
        rows["measured_soh_pct"],
        s=36,
        color=GOLD,
        zorder=3,
        label="Measured SOH",
    )
    ax.axhline(80, color=RED, linewidth=1.1, linestyle="--", label="80% SOH")
    ax.set_ylim(40, 110)
    ax.set_ylabel("SOH (%)")
    ax.set_title(
        "SOH forecast  ·  historical-mean usage", loc="left", color=INK, fontsize=13, pad=10
    )
    ax.legend(frameon=False, ncol=2)
    fig.autofmt_xdate()
    _stamp(fig)
    _save(fig, "soh_forecast.png")


def plot_anomaly() -> None:
    import importlib.util

    module_path = ROOT / "prototype_anomaly" / "src" / "anomaly_detection.py"
    spec = importlib.util.spec_from_file_location("prototype_anomaly_detection", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    # The prototype imports sibling modules as src.*, so give that package the path.
    prototype_root = str(ROOT / "prototype_anomaly")
    sys.path.insert(0, prototype_root)
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    spec.loader.exec_module(module)
    run_isolation_forest = module.run_isolation_forest

    frames = []
    root = ROOT / "brain" / "data" / "raw_parquet" / "serial=100000001"
    for path in sorted(root.glob("year=*/month=*/part-000.parquet")):
        table = pq.ParquetFile(path).read(
            columns=[
                "time",
                "batt_batterymodules_mod1_pack1_cellvoltage_mv_0",
                "batt_batterymodules_mod1_pack1_cellvoltage_mv_7",
                "batt_batterymodules_mod1_pack1_temperature0_c",
                "batt_batterymodules_mod1_pack1_temperature3_c",
            ]
        )
        frame = table.to_pandas().iloc[::30].copy()
        frame["cell_voltage_spread_V"] = (
            frame["batt_batterymodules_mod1_pack1_cellvoltage_mv_7"]
            - frame["batt_batterymodules_mod1_pack1_cellvoltage_mv_0"]
        ) / 1000.0
        frame["pack_temp_spread_C"] = (
            frame["batt_batterymodules_mod1_pack1_temperature3_c"]
            - frame["batt_batterymodules_mod1_pack1_temperature0_c"]
        ).abs()
        frames.append(frame[["time", "cell_voltage_spread_V", "pack_temp_spread_C"]])
    work = pd.concat(frames, ignore_index=True)
    # Keep the matrix to the voltage spread so the planted offset is the outlier.
    work["is_rest"] = 0.0
    work["is_charging"] = 0.0
    work["is_discharging"] = 1.0
    # The offset month is about a quarter of this short trace, so the forest
    # threshold has to sit above that share or the tied scores stay unlabeled.
    scored = run_isolation_forest(
        work[["time", "cell_voltage_spread_V", "is_rest", "is_charging", "is_discharging"]],
        contamination=0.3,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    spread_mv = scored["cell_voltage_spread_V"] * 1000.0
    flagged = scored[scored["anomaly"] == -1]
    if flagged.empty:
        raise SystemExit("Isolation Forest did not flag any synthetic rows.")
    ax.plot(scored["time"], spread_mv, color=BLUE, linewidth=1.6, label="Cell spread", zorder=2)
    ax.scatter(
        flagged["time"],
        flagged["cell_voltage_spread_V"] * 1000.0,
        s=70,
        color=RED,
        edgecolors="#3b1210",
        linewidths=0.8,
        zorder=4,
        label="Isolation Forest flag",
    )
    ax.set_ylabel("Cell 7 offset (mV)")
    ax.set_title(
        "Cell-voltage anomaly  ·  field device 100000001",
        loc="left",
        color=INK,
        fontsize=13,
        pad=10,
    )
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    _stamp(fig, "Synthetic data  ·  prototype Isolation Forest")
    _save(fig, "cell_anomaly.png")


def plot_seasonal() -> None:
    frames = []
    root = ROOT / "brain" / "data" / "raw_parquet" / "serial=100000001"
    for path in sorted(root.glob("year=*/month=*/part-000.parquet")):
        frame = pq.ParquetFile(path).read(
            columns=["time", "batt_current_a", "sys_temperature1_c"]
        ).to_pandas()
        frames.append(frame)
    work = pd.concat(frames, ignore_index=True)
    work["month"] = work["time"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    work["discharge_ah"] = (
        np.where(work["batt_current_a"] < -0.5, -work["batt_current_a"], 0.0) / 3600.0
    )
    monthly = work.groupby("month", as_index=False).agg(
        temperature_c=("sys_temperature1_c", "mean"),
        discharge_ah=("discharge_ah", "sum"),
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    ax.bar(
        monthly["month"], monthly["temperature_c"], color="#d7ecec", label="Mean pack temperature"
    )
    ax.set_ylabel("Temperature (°C)")
    twin = ax.twinx()
    twin.plot(
        monthly["month"],
        monthly["discharge_ah"],
        color=GOLD,
        marker="o",
        linewidth=2.2,
        label="Discharge Ah",
    )
    twin.set_ylabel("Discharge (Ah)")
    twin.spines["top"].set_visible(False)
    ax.set_title(
        "Seasonal temperature and use  ·  field device", loc="left", color=INK, fontsize=13, pad=10
    )
    ax.tick_params(axis="x", rotation=0)
    _stamp(fig)
    fig.tight_layout()
    _save(fig, "seasonal_temperature.png")


def plot_hero() -> None:
    width, height = 1600, 420
    image = Image.new("RGB", (width, height), TEAL)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        shade = int(23 + 18 * y / height)
        draw.line([(0, y), (width, y)], fill=(shade, 70 + y // 12, 82 + y // 18))
    draw.rounded_rectangle((70, 78, 250, 118), radius=8, fill=(255, 255, 255))
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
        sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        chip_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        title_font = ImageFont.load_default()
        sub_font = title_font
        chip_font = title_font
    draw.text((78, 86), "SYNTHETIC DATA", fill=TEAL, font=chip_font)
    draw.text((70, 150), "Home battery state of health", fill="white", font=title_font)
    draw.text(
        (70, 230),
        "From one-second telemetry to a 10-year capacity forecast.",
        fill=(232, 242, 244),
        font=sub_font,
    )
    chips = ["brain", "FastAPI", "React", "DuckDB"]
    x = 70
    for chip in chips:
        box = (x, 310, x + 150, 356)
        draw.rounded_rectangle(box, radius=16, outline=(255, 255, 255), width=2)
        draw.text((x + 28, 322), chip, fill="white", font=sub_font)
        x += 170
    path = OUT / "hero.png"
    image = image.quantize(colors=32, method=Image.Quantize.MEDIANCUT)
    image.save(path, optimize=True)


def main() -> None:
    import argparse

    global OUT
    parser = argparse.ArgumentParser(description="Export synthetic README figures.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "images" / "generated",
        help="Directory for generated figures. Defaults to a gitignored folder.",
    )
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    run = _latest_run()
    plot_capacity_fade(run)
    plot_soh_forecast(run)
    plot_anomaly()
    plot_seasonal()
    plot_hero()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.name} {path.stat().st_size}")


if __name__ == "__main__":
    main()
