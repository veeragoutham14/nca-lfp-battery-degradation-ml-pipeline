#!/usr/bin/env python3
"""Write synthetic 1-second telemetry in the brain's Parquet layout.

Layout:
  brain/data/raw_parquet/serial=<id>/year=YYYY/month=MM/part-000.parquet

Columns follow config/signal_catalog.json. Values are invented for the demo:
the lab reference 100000017 has exact 100% to 0% discharges so capacity_ml
can fit a curve, and the field traces include a gap longer than 60 seconds.
Charge and discharge currents sit outside the +/- 0.5 A rest band.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "signal_catalog.json"
CELL_VOLTAGE = re.compile(
    r"batt_batterymodules_mod(\d)_pack(\d)_cellvoltage_mv_(\d+)$"
)
CELL_TEMP = re.compile(
    r"batt_batterymodules_mod(\d)_pack(\d)_temperature(\d)_c$"
)
MODULE_ID = re.compile(r"_(mod|pack|node|module|packid)(\d)?")

TEXT_VALUES = {
    "batt_stackstatus": "normal",
    "inverter_activeerror": "",
    "inverter_deratingstatus": "none",
    "inverter_updatestatus": "idle",
    "activeErrors_names": "",
}


@dataclass(frozen=True)
class Segment:
    start: datetime
    rows: int
    soc_start: float
    soc_end: float
    current_a: float
    soh_pct: float
    ambient_c: float
    cycle: int
    cell_bias_mv: float = 0.0
    gap_after_s: int = 1


def _load_columns() -> list[dict]:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    columns = payload["columns"]
    if len(columns) != payload["column_count"]:
        raise ValueError("signal catalog column_count does not match the column list")
    return columns


def _mode_label(current_a: float) -> str:
    if current_a > 0.5:
        return "charge"
    if current_a < -0.5:
        return "discharge"
    return "idle"


def _build_table(
    columns: list[dict],
    serial: int,
    segment: Segment,
    *,
    elapsed_offset_s: float,
) -> tuple[pa.Table, float]:
    rows = segment.rows
    soc = np.linspace(segment.soc_start, segment.soc_end, rows, dtype=np.float64)
    current = np.full(rows, segment.current_a, dtype=np.float64)
    voltage = 360.0 + 0.70 * soc
    soh = np.full(rows, segment.soh_pct, dtype=np.float64)
    ambient = np.full(rows, segment.ambient_c, dtype=np.float64)
    start_ns = np.int64(segment.start.timestamp() * 1_000_000_000)
    times = start_ns + np.arange(rows, dtype=np.int64) * 1_000_000_000
    elapsed = elapsed_offset_s + np.arange(rows, dtype=np.float64)

    cell_base_mv = 3200.0 + 9.0 * soc
    arrays: list[pa.Array] = []
    names: list[str] = []
    for column in columns:
        name = column["name"]
        sql_type = column["sql_type"]
        names.append(name)
        if name == "time":
            arrays.append(pa.array(times, type=pa.timestamp("ns", tz="UTC")))
            continue
        if name == "serial":
            arrays.append(pa.array(np.full(rows, serial, dtype=np.int64)))
            continue
        if sql_type == "TEXT":
            if name == "inverter_state":
                value = "running" if abs(segment.current_a) > 0.5 else "standby"
            elif name == "inverter_batterydcdcstate":
                value = _mode_label(segment.current_a)
            else:
                value = TEXT_VALUES.get(name, "")
            arrays.append(pa.array([value] * rows, type=pa.string()))
            continue
        if sql_type == "BOOLEAN":
            value = name != "sys_isresolvingerror"
            arrays.append(pa.array(np.full(rows, value)))
            continue
        if sql_type == "INTEGER":
            if name in {"batt_batterystate", "stack_batterystate"}:
                value = 1 if segment.current_a > 0.5 else 2 if segment.current_a < -0.5 else 0
            else:
                value = 1
            arrays.append(pa.array(np.full(rows, value, dtype=np.int32)))
            continue

        voltage_match = CELL_VOLTAGE.search(name)
        if voltage_match:
            cell_index = int(voltage_match.group(3))
            bias = segment.cell_bias_mv if cell_index == 7 else 0.0
            values = cell_base_mv + cell_index * 1.5 + bias
            arrays.append(pa.array(values))
            continue
        temperature_match = CELL_TEMP.search(name)
        if temperature_match:
            sensor = int(temperature_match.group(3))
            arrays.append(pa.array(ambient + sensor * 0.15 + abs(segment.current_a) * 0.02))
            continue
        if name in {"batt_soc_pct"}:
            arrays.append(pa.array(soc))
        elif name in {"batt_soh_pct"}:
            arrays.append(pa.array(soh))
        elif name in {"batt_current_a"}:
            arrays.append(pa.array(current))
        elif name in {"batt_voltage_v", "batt_intermediatevoltage_v"}:
            arrays.append(pa.array(voltage))
        elif name in {"batt_cellvoltagemax_v", "stack_cellvoltagemax_v", "batt_compensatedcellvoltagemax_v", "stack_compensatedcellvoltagemax_v"}:
            arrays.append(pa.array((cell_base_mv + 13 * 1.5 + segment.cell_bias_mv) / 1000.0))
        elif name in {"batt_cellvoltagemin_v", "stack_cellvoltagemin_v", "batt_compensatedcellvoltagemin_v", "stack_compensatedcellvoltagemin_v"}:
            arrays.append(pa.array(cell_base_mv / 1000.0))
        elif name in {"batt_modulecount"}:
            arrays.append(pa.array(np.full(rows, 2.0)))
        elif name in {"sys_cycles"}:
            arrays.append(pa.array(np.full(rows, float(segment.cycle))))
        elif "temperature" in name:
            arrays.append(pa.array(ambient + (2.0 if "inverter" in name else 0.0)))
        elif name.endswith("current_a"):
            arrays.append(pa.array(np.full(rows, abs(segment.current_a))))
        elif name.endswith("voltage_v"):
            arrays.append(pa.array(voltage))
        elif name.endswith("power_w"):
            arrays.append(pa.array(np.abs(voltage * current)))
        elif name.endswith("_ws") or name.endswith("_wh"):
            arrays.append(pa.array(elapsed * abs(segment.current_a) * 400.0))
        elif name == "inverter_runtime_s":
            arrays.append(pa.array(elapsed))
        elif name.endswith("id") or name.endswith("moduleid") or name.endswith("packid") or name.endswith("nodeid"):
            digits = re.findall(r"\d+", name)
            arrays.append(pa.array(np.full(rows, float(digits[-1]) if digits else 1.0)))
        else:
            arrays.append(pa.array(np.zeros(rows, dtype=np.float64)))

    table = pa.table({name: array for name, array in zip(names, arrays, strict=True)})
    return table, float(elapsed[-1] + 1.0)


def _write_device(columns: list[dict], serial: int, segments: list[Segment], output: Path) -> int:
    grouped: dict[tuple[int, int], list[Segment]] = {}
    for segment in segments:
        grouped.setdefault((segment.start.year, segment.start.month), []).append(segment)
    written = 0
    elapsed = 0.0
    for (year, month), month_segments in sorted(grouped.items()):
        tables = []
        for segment in month_segments:
            table, elapsed = _build_table(columns, serial, segment, elapsed_offset_s=elapsed)
            tables.append(table)
            if segment.gap_after_s > 1:
                elapsed += segment.gap_after_s
        directory = output / f"serial={serial}" / f"year={year:04d}" / f"month={month:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-000.parquet"
        pq.write_table(pa.concat_tables(tables), path, compression="snappy")
        written += sum(segment.rows for segment in month_segments)
    return written


def _full_cycle(start: datetime, *, capacity_ah: float, soh: float, ambient: float, cycle: int, current_a: float = 40.0) -> list[Segment]:
    discharge_steps = int(round(capacity_ah / current_a * 3600))
    charge_steps = discharge_steps
    cursor = start
    segments = [
        Segment(cursor, 20, 100.0, 100.0, 0.2, soh, ambient, cycle, gap_after_s=1),
    ]
    cursor += timedelta(seconds=20)
    segments.append(
        Segment(cursor, discharge_steps + 1, 100.0, 0.0, -current_a, soh, ambient, cycle, gap_after_s=1)
    )
    cursor += timedelta(seconds=discharge_steps + 1)
    segments.append(Segment(cursor, 15, 0.0, 0.0, 0.1, soh, ambient, cycle, gap_after_s=1))
    cursor += timedelta(seconds=15)
    segments.append(
        Segment(cursor, charge_steps + 1, 0.0, 100.0, current_a, soh, ambient, cycle, gap_after_s=120)
    )
    return segments


def _reference_segments() -> list[Segment]:
    months = [(2024, 1), (2024, 3), (2024, 5), (2024, 7), (2024, 9), (2024, 11), (2025, 1), (2025, 3)]
    ambient = {1: 6.0, 3: 11.0, 5: 18.0, 7: 28.0, 9: 19.0, 11: 8.0}
    segments: list[Segment] = []
    for index, (year, month) in enumerate(months):
        start = datetime(year, month, 8, 8, 0, tzinfo=timezone.utc)
        segments.extend(
            _full_cycle(
                start,
                capacity_ah=56.0 - 0.4 * index,
                soh=100.0 - 0.45 * index,
                ambient=ambient[month],
                cycle=index + 1,
            )
        )
    return segments


def _field_segments() -> list[Segment]:
    """Partial home use, one long gap, and one warm-month cell offset."""
    plan = [
        (datetime(2024, 1, 16, 18, 0, tzinfo=timezone.utc), 4.0, 0.0, 90),
        (datetime(2024, 4, 16, 18, 0, tzinfo=timezone.utc), 15.0, 0.0, 90),
        (datetime(2024, 7, 16, 18, 0, tzinfo=timezone.utc), 29.0, 55.0, 120),
        (datetime(2024, 10, 16, 18, 0, tzinfo=timezone.utc), 12.0, 0.0, 90),
    ]
    segments: list[Segment] = []
    for index, (start, ambient, bias, gap) in enumerate(plan, start=1):
        cursor = start
        segments.append(Segment(cursor, 30, 80.0, 80.0, 0.2, 97.0 - index, ambient, index, gap_after_s=1))
        cursor += timedelta(seconds=30)
        segments.append(
            Segment(cursor, 1201, 80.0, 50.0, -12.0, 97.0 - index, ambient, index, cell_bias_mv=bias, gap_after_s=gap)
        )
        cursor += timedelta(seconds=1201 + gap)
        segments.append(
            Segment(cursor, 1201, 50.0, 85.0, 10.0, 97.0 - index, ambient, index, cell_bias_mv=bias, gap_after_s=1)
        )
    return segments


def _lfp_segments() -> list[Segment]:
    start = datetime(2025, 2, 3, 12, 0, tzinfo=timezone.utc)
    return [
        Segment(start, 20, 70.0, 70.0, 0.2, 96.0, 9.0, 1, gap_after_s=1),
        Segment(start + timedelta(seconds=20), 901, 70.0, 55.0, -8.0, 96.0, 9.0, 1, gap_after_s=90),
        Segment(start + timedelta(seconds=20 + 901 + 90), 601, 55.0, 68.0, 8.0, 96.0, 9.0, 1, gap_after_s=1),
    ]


def generate(output: Path) -> None:
    columns = _load_columns()
    output.mkdir(parents=True, exist_ok=True)
    devices = (
        (100000017, _reference_segments()),
        (100000001, _field_segments()),
        (100000020, _lfp_segments()),
    )
    for serial, segments in devices:
        rows = _write_device(columns, serial, segments, output)
        print(f"serial={serial} rows={rows}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic battery telemetry Parquet.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "brain" / "data" / "raw_parquet",
    )
    args = parser.parse_args()
    generate(args.output.resolve())


if __name__ == "__main__":
    main()
