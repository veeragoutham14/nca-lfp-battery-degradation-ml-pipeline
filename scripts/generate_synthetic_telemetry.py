#!/usr/bin/env python3
"""Write synthetic 1-second telemetry in the brain's Parquet layout.

Layout:
  brain/data/raw_parquet/serial=<id>/year=YYYY/month=MM/part-000.parquet
  (the dense field devices write one part-DDD.parquet per day instead)

Columns follow config/signal_catalog.json. Values are invented for the demo:
the lab reference 100000017 has exact 100% to 0% discharges so capacity_ml
can fit a curve, and the field traces include a gap longer than 60 seconds.
Charge and discharge currents sit outside the +/- 0.5 A rest band.

It also writes the five-minute cell-voltage and temperature aggregates that the
database import step (purpose electrothermal_history) would produce:
  brain/data/electrothermal_telemetry/<group>/5m/chemistry=<c>/serial=<id>/year=YYYY/month=MM/
with <signal>__mean, __min, __max, __std and __count per channel.

With --dense-field-device it adds 100000002 (NCA) and 100000021 (LFP), field
devices with continuous 1-second data on 16 days in each of four months of 2024.
The seasonal, BMS SOH recalibration and voltage anomaly analyses need that
coverage. Each is about 9 million rows, so they are opt-in.

Cell voltages follow a chemistry-specific open-circuit voltage curve plus an
ohmic term whose resistance rises in the cold. Voltage windows, nominal
voltages and fade ranges are taken from public descriptions of open datasets
(Severson et al. 2019, Preger et al. 2020 / Sandia, NASA PCoE); see the README
data section. No measured data is copied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "signal_catalog.json"
ELECTROTHERMAL_STATISTICS = (
    ("mean", "avg"),
    ("min", "min"),
    ("max", "max"),
    ("std", "stddev_pop"),
    ("count", "count"),
)
DENSE_FIELD_SERIAL = 100000002
DENSE_FIELD_CAPACITY_AH = 64.0
DENSE_DAYS_PER_MONTH = 26
DENSE_LFP_SERIAL = 100000021
DENSE_LFP_CAPACITY_AH = 50.0

# Open-circuit voltage in mV against SOC in %. NCA spans the 2.5 to 4.2 V window
# with a sloped curve; LFP spans 2.0 to 3.6 V with the flat two-phase plateau
# near its 3.3 V nominal. Both windows are the published cell ratings in the
# Sandia study (Preger et al. 2020) and the Severson et al. 2019 LFP cells.
OCV_SOC_PCT = (0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 100.0)
OCV_MV = {
    "NCA": (2950.0, 3200.0, 3330.0, 3450.0, 3560.0, 3630.0, 3690.0,
            3760.0, 3850.0, 3940.0, 4030.0, 4110.0, 4150.0, 4190.0),
    "LFP": (2650.0, 2950.0, 3120.0, 3210.0, 3255.0, 3280.0, 3290.0,
            3295.0, 3300.0, 3315.0, 3330.0, 3340.0, 3360.0, 3450.0),
}
CELL_VOLTAGE_WINDOW_MV = {"NCA": (2500.0, 4200.0), "LFP": (2000.0, 3600.0)}
# Ohmic cell resistance at 25 C in milliohm, and its Arrhenius activation energy.
CELL_RESISTANCE_MOHM = {"NCA": 1.1, "LFP": 0.9}
RESISTANCE_ACTIVATION_J_PER_MOL = 20_000.0
GAS_CONSTANT = 8.314


def cell_voltage_mv(
    soc_pct: np.ndarray, current_a: float, cell_temp_c: np.ndarray, chemistry: str
) -> np.ndarray:
    """Open-circuit voltage plus the ohmic drop; charge current is positive."""
    ocv = np.interp(soc_pct, OCV_SOC_PCT, OCV_MV[chemistry])
    kelvin = np.asarray(cell_temp_c, dtype=np.float64) + 273.15
    scale = np.exp(RESISTANCE_ACTIVATION_J_PER_MOL / GAS_CONSTANT * (1.0 / kelvin - 1.0 / 298.15))
    low, high = CELL_VOLTAGE_WINDOW_MV[chemistry]
    # Clip at the rated window, as a constant-voltage charge phase would.
    return np.clip(ocv + current_a * CELL_RESISTANCE_MOHM[chemistry] * scale, low, high)
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
    chemistry: str = "NCA",
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

    cell_temp = ambient + abs(segment.current_a) * 0.02
    cell_base_mv = cell_voltage_mv(soc, segment.current_a, cell_temp, chemistry)
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
            # BMS cell voltages are reported in whole millivolts.
            values = np.round(cell_base_mv + cell_index * 1.5 + bias)
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
        elif name in {
            "batt_cellvoltagemax_v",
            "stack_cellvoltagemax_v",
            "batt_compensatedcellvoltagemax_v",
            "stack_compensatedcellvoltagemax_v",
        }:
            arrays.append(pa.array((cell_base_mv + 13 * 1.5 + segment.cell_bias_mv) / 1000.0))
        elif name in {
            "batt_cellvoltagemin_v",
            "stack_cellvoltagemin_v",
            "batt_compensatedcellvoltagemin_v",
            "stack_compensatedcellvoltagemin_v",
        }:
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
        elif (
            name.endswith("id")
            or name.endswith("moduleid")
            or name.endswith("packid")
            or name.endswith("nodeid")
        ):
            digits = re.findall(r"\d+", name)
            arrays.append(pa.array(np.full(rows, float(digits[-1]) if digits else 1.0)))
        else:
            arrays.append(pa.array(np.zeros(rows, dtype=np.float64)))

    table = pa.table({name: array for name, array in zip(names, arrays, strict=True)})
    return table, float(elapsed[-1] + 1.0)


def _write_device(
    columns: list[dict],
    serial: int,
    segments: list[Segment],
    output: Path,
    chemistry: str = "NCA",
) -> int:
    grouped: dict[tuple[int, int], list[Segment]] = {}
    for segment in segments:
        grouped.setdefault((segment.start.year, segment.start.month), []).append(segment)
    written = 0
    elapsed = 0.0
    for (year, month), month_segments in sorted(grouped.items()):
        tables = []
        for segment in month_segments:
            table, elapsed = _build_table(
                columns, serial, segment, elapsed_offset_s=elapsed, chemistry=chemistry
            )
            tables.append(table)
            if segment.gap_after_s > 1:
                elapsed += segment.gap_after_s
        directory = output / f"serial={serial}" / f"year={year:04d}" / f"month={month:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "part-000.parquet"
        pq.write_table(pa.concat_tables(tables), path, compression="zstd")
        written += sum(segment.rows for segment in month_segments)
    return written


def _full_cycle(
    start: datetime,
    *,
    capacity_ah: float,
    soh: float,
    ambient: float,
    cycle: int,
    current_a: float = 40.0,
) -> list[Segment]:
    discharge_steps = int(round(capacity_ah / current_a * 3600))
    charge_steps = discharge_steps
    cursor = start
    segments = [
        Segment(cursor, 20, 100.0, 100.0, 0.2, soh, ambient, cycle, gap_after_s=1),
    ]
    cursor += timedelta(seconds=20)
    segments.append(
        Segment(
            cursor, discharge_steps + 1, 100.0, 0.0, -current_a, soh, ambient, cycle, gap_after_s=1
        )
    )
    cursor += timedelta(seconds=discharge_steps + 1)
    segments.append(Segment(cursor, 15, 0.0, 0.0, 0.1, soh, ambient, cycle, gap_after_s=1))
    cursor += timedelta(seconds=15)
    segments.append(
        Segment(
            cursor, charge_steps + 1, 0.0, 100.0, current_a, soh, ambient, cycle, gap_after_s=120
        )
    )
    return segments


def _reference_segments() -> list[Segment]:
    months = [
        (2024, 1), (2024, 3), (2024, 5), (2024, 7),
        (2024, 9), (2024, 11), (2025, 1), (2025, 3),
    ]
    ambient = {1: 6.0, 3: 11.0, 5: 18.0, 7: 28.0, 9: 19.0, 11: 8.0}
    segments: list[Segment] = []
    for index, (year, month) in enumerate(months):
        start = datetime(year, month, 8, 8, 0, tzinfo=UTC)
        capacity_ah = 56.0 - 0.4 * index
        # The BMS reports SOH in whole percent, so it moves in visible steps.
        bms_soh = float(round(100.0 * capacity_ah / 56.0))
        segments.extend(
            _full_cycle(
                start,
                capacity_ah=capacity_ah,
                soh=bms_soh,
                ambient=ambient[month],
                cycle=index + 1,
            )
        )
        # Two logged rest days in the chamber, well away from the cycle, give the
        # BMS SOH recalibration analysis matched control days.
        rest_start = datetime(year, month, 20, tzinfo=UTC)
        segments.append(
            Segment(
                rest_start, 2 * 86400, 100.0, 100.0, 0.2,
                bms_soh, ambient[month], index + 1,
            )
        )
    return segments


def _field_segments() -> list[Segment]:
    """Partial home use, one long gap, and one warm-month cell offset."""
    plan = [
        (datetime(2024, 1, 16, 18, 0, tzinfo=UTC), 4.0, 0.0, 90),
        (datetime(2024, 4, 16, 18, 0, tzinfo=UTC), 15.0, 0.0, 90),
        (datetime(2024, 7, 16, 18, 0, tzinfo=UTC), 29.0, 55.0, 120),
        (datetime(2024, 10, 16, 18, 0, tzinfo=UTC), 12.0, 0.0, 90),
    ]
    segments: list[Segment] = []
    for index, (start, ambient, bias, gap) in enumerate(plan, start=1):
        cursor = start
        segments.append(
            Segment(cursor, 30, 80.0, 80.0, 0.2, 97.0 - index, ambient, index, gap_after_s=1)
        )
        cursor += timedelta(seconds=30)
        segments.append(
            Segment(
                cursor, 1201, 80.0, 50.0, -12.0, 97.0 - index,
                ambient, index, cell_bias_mv=bias, gap_after_s=gap,
            )
        )
        cursor += timedelta(seconds=1201 + gap)
        segments.append(
            Segment(
                cursor, 1201, 50.0, 85.0, 10.0, 97.0 - index,
                ambient, index, cell_bias_mv=bias, gap_after_s=1,
            )
        )
    return segments


def _lfp_segments() -> list[Segment]:
    start = datetime(2025, 2, 3, 12, 0, tzinfo=UTC)
    return [
        Segment(start, 20, 70.0, 70.0, 0.2, 96.0, 9.0, 1, gap_after_s=1),
        Segment(start + timedelta(seconds=20), 901, 70.0, 55.0, -8.0, 96.0, 9.0, 1, gap_after_s=90),
        Segment(
            start + timedelta(seconds=20 + 901 + 90), 601, 55.0, 68.0, 8.0,
            96.0, 9.0, 1, gap_after_s=1,
        ),
    ]


def _write_device_streaming(
    columns: list[dict],
    serial: int,
    segments: list[Segment],
    output: Path,
    batch_rows: int = 200_000,
    chemistry: str = "NCA",
) -> int:
    """Write one Parquet part per day in row groups, so long traces stay out of memory.

    Daily parts mirror the chunked layout of the raw importer and keep each file
    small enough for the per-file brain stages (row classification reads a whole
    file at once).
    """
    writers: dict[tuple[int, int, int], pq.ParquetWriter] = {}
    buffers: dict[tuple[int, int, int], list[pa.Table]] = {}
    written = 0
    elapsed = 0.0

    def flush(key: tuple[int, int, int]) -> None:
        if buffers.get(key):
            writers[key].write_table(pa.concat_tables(buffers[key]))
            buffers[key] = []

    try:
        for segment in sorted(segments, key=lambda item: item.start):
            table, elapsed = _build_table(
                columns, serial, segment, elapsed_offset_s=elapsed, chemistry=chemistry
            )
            if segment.gap_after_s > 1:
                elapsed += segment.gap_after_s
            key = (segment.start.year, segment.start.month, segment.start.day)
            if key not in writers:
                for other in list(writers):
                    flush(other)
                    writers.pop(other).close()
                directory = (
                    output / f"serial={serial}" / f"year={key[0]:04d}" / f"month={key[1]:02d}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                writers[key] = pq.ParquetWriter(
                    directory / f"part-{key[2]:03d}.parquet", table.schema, compression="zstd"
                )
                buffers[key] = []
            buffers[key].append(table)
            written += table.num_rows
            if sum(item.num_rows for item in buffers[key]) >= batch_rows:
                flush(key)
    finally:
        for key, writer in writers.items():
            flush(key)
            writer.close()
    return written


def _dense_day(
    day_start: datetime,
    *,
    soc_start: float,
    true_capacity_ah: float,
    bms_soh_pct: float,
    ambient_c: float,
    charge_current_a: float,
    full_discharge: bool,
    cell_bias_mv: float,
    cycle: int,
    soh_after_charge_start_pct: float | None = None,
) -> tuple[list[Segment], float]:
    """One continuous day: night rest, morning use, solar charge, evening use.

    On a full-discharge day the pack starts discharging ten minutes after it
    reaches 100 % and runs down to 1 % without a break, so the 95 % to 5 %
    traversal is mostly discharge time, as in a deliberate capacity check.
    soh_after_charge_start_pct, when given, is the BMS SOH reported from the
    start of the day's charge on, so a recalibration lands just after the
    discharge that preceded it has ended.
    """
    segments: list[Segment] = []
    cursor = day_start
    soc = soc_start
    day_end = day_start + timedelta(days=1)

    def hold(until: datetime, ambient: float) -> None:
        nonlocal cursor
        rows = int((until - cursor).total_seconds())
        if rows > 0:
            segments.append(
                Segment(cursor, rows, soc, soc, 0.2, reported_soh, ambient, cycle, cell_bias_mv)
            )
            cursor = until

    def move(target_soc: float, current_a: float, ambient: float) -> None:
        nonlocal cursor, soc
        rows = int(round(abs(target_soc - soc) / 100.0 * true_capacity_ah / abs(current_a) * 3600))
        rows = min(rows, int((day_end - cursor).total_seconds()))
        if rows > 0:
            segments.append(
                Segment(
                    cursor, rows, soc, target_soc, current_a, reported_soh,
                    ambient, cycle, cell_bias_mv,
                )
            )
            cursor += timedelta(seconds=rows)
            soc = target_soc

    reported_soh = bms_soh_pct
    night, noon = ambient_c - 2.0, ambient_c + 3.0
    hold(day_start.replace(hour=7), night)
    if soc > 20.0:
        move(soc - 12.0, -18.0, night)
    hold(day_start.replace(hour=10), ambient_c)
    if soh_after_charge_start_pct is not None:
        reported_soh = soh_after_charge_start_pct
    move(100.0 if full_discharge else 96.0, charge_current_a, noon)
    if full_discharge:
        hold(cursor + timedelta(minutes=10), noon)
    else:
        hold(day_start.replace(hour=17), noon)
    move(1.0 if full_discharge else 30.0, -20.0, ambient_c)
    hold(day_end, night)
    return segments, soc


def _arrhenius_factor(temp_c: float, activation_j_per_mol: float = 31_500.0) -> float:
    """Capacity-fade acceleration relative to 25 C (Wang et al. 2011 activation energy)."""
    kelvin = temp_c + 273.15
    return math.exp(activation_j_per_mol / GAS_CONSTANT * (1.0 / 298.15 - 1.0 / kelvin))


def _dense_field_segments(
    *,
    capacity_ah: float = DENSE_FIELD_CAPACITY_AH,
    fade_pct_per_day_25c: float = 0.012,
    anomaly_month: int = 7,
    anomaly_mv: float = 40.0,
) -> list[Segment]:
    """Continuous 1-second home use on 26 days in January, April, July and October 2024.

    About one equivalent full cycle a day. True capacity fades by
    fade_pct_per_day_25c per day at 25 C, scaled by an Arrhenius factor of the
    month's temperature: 0.012 %/cycle for NCA and 0.006 %/cycle for LFP sit
    inside the 80 %-retention lifetimes of roughly 150 to 2,300 cycles reported
    by Severson et al. 2019 and the Sandia study, with LFP fading slowest. The
    reported BMS SOH steps down after full discharges in April, July and
    October, which gives the recalibration analysis events to find; each step
    appears when the next charge starts, after the full discharge has ended. Charging is
    faster in summer, and cell 7 is offset by anomaly_mv on six days of
    anomaly_month. Twenty-six logged days cover more than 80 % of each month, so
    every month qualifies for the seasonal profile.
    """
    months = ((1, 4.0, 14.0), (4, 13.0, 22.0), (7, 27.0, 36.0), (10, 12.0, 20.0))
    reported_soh = {1: 98.0, 4: 97.0, 7: 96.0, 10: 95.0}
    segments: list[Segment] = []
    soc = 30.0
    day_index = 0
    true_soh = 98.5
    for month, ambient, charge_current in months:
        bms_soh = reported_soh[month] + (0.0 if month == 1 else 1.0)
        for day in range(1, DENSE_DAYS_PER_MONTH + 1):
            day_start = datetime(2024, month, day, tzinfo=UTC)
            if day_index:
                true_soh -= fade_pct_per_day_25c * _arrhenius_factor(ambient)
            full = day in (4, 12, 20)
            step = month != 1 and day == 5
            seasonal_swing = 1.5 * math.sin(2 * math.pi * day / DENSE_DAYS_PER_MONTH)
            day_segments, soc = _dense_day(
                day_start,
                soc_start=soc,
                true_capacity_ah=capacity_ah * true_soh / 100.0,
                bms_soh_pct=bms_soh,
                ambient_c=ambient + seasonal_swing,
                charge_current_a=charge_current,
                full_discharge=full,
                cell_bias_mv=anomaly_mv if month == anomaly_month and 6 <= day <= 11 else 0.0,
                cycle=day_index + 1,
                soh_after_charge_start_pct=reported_soh[month] if step else None,
            )
            if step:
                bms_soh = reported_soh[month]
            segments.extend(day_segments)
            day_index += 1
    return segments


def write_electrothermal_aggregates(
    raw_output: Path,
    electrothermal_output: Path,
    *,
    serial: int,
    chemistry: str,
    module_count: int = 2,
) -> int:
    """Aggregate 1-second cell voltages and temperatures into five-minute buckets.

    This mirrors the database import step: time_bucket of five minutes, then
    AVG, MIN, MAX, STDDEV_POP and COUNT per channel, one dataset per group.
    """
    groups = {"cell_voltage": [], "temperature": []}
    for column in _load_columns():
        voltage = CELL_VOLTAGE.search(column["name"])
        temperature = CELL_TEMP.search(column["name"])
        match = voltage or temperature
        if match and int(match.group(1)) <= module_count:
            groups["cell_voltage" if voltage else "temperature"].append(column["name"])
    written = 0
    with duckdb.connect() as connection:
        for month_dir in sorted((raw_output / f"serial={serial}").glob("year=*/month=*")):
            files = sorted(str(path) for path in month_dir.glob("*.parquet"))
            for group, names in groups.items():
                aggregates = ", ".join(
                    f'{function}("{name}") AS "{name}__{statistic}"'
                    for name in names
                    for statistic, function in ELECTROTHERMAL_STATISTICS
                )
                frame = connection.execute(
                    f"""
                    SELECT time_bucket(INTERVAL '5 minutes', time) AS time,
                           any_value(serial) AS serial, {aggregates}
                    FROM read_parquet(?, hive_partitioning=false)
                    GROUP BY 1 ORDER BY 1
                    """,
                    [files],
                ).fetch_arrow_table()
                target = (
                    electrothermal_output / group / "5m" / f"chemistry={chemistry}"
                    / f"serial={serial}" / month_dir.parent.name / month_dir.name
                )
                target.mkdir(parents=True, exist_ok=True)
                pq.write_table(frame, target / "part-00000.parquet", compression="zstd")
                written += frame.num_rows
    return written


def generate(
    output: Path,
    *,
    electrothermal_output: Path | None = None,
    dense_field_device: bool = False,
) -> None:
    columns = _load_columns()
    output.mkdir(parents=True, exist_ok=True)
    devices = [
        (100000017, "NCA", _reference_segments()),
        (100000001, "NCA", _field_segments()),
        (100000020, "LFP", _lfp_segments()),
    ]
    for serial, chemistry, segments in devices:
        rows = _write_device(columns, serial, segments, output, chemistry=chemistry)
        print(f"serial={serial} rows={rows}")
    if dense_field_device:
        dense = (
            (DENSE_FIELD_SERIAL, "NCA", _dense_field_segments()),
            (
                DENSE_LFP_SERIAL,
                "LFP",
                _dense_field_segments(
                    capacity_ah=DENSE_LFP_CAPACITY_AH,
                    fade_pct_per_day_25c=0.006,
                    anomaly_month=10,
                    anomaly_mv=-25.0,
                ),
            ),
        )
        for serial, chemistry, segments in dense:
            rows = _write_device_streaming(
                columns, serial, segments, output, chemistry=chemistry
            )
            print(f"serial={serial} rows={rows}")
            devices.append((serial, chemistry, []))
    if electrothermal_output is not None:
        for serial, chemistry, _segments in devices:
            buckets = write_electrothermal_aggregates(
                output, electrothermal_output, serial=serial, chemistry=chemistry
            )
            print(f"serial={serial} electrothermal_5m_rows={buckets}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic battery telemetry Parquet.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "brain" / "data" / "raw_parquet",
    )
    parser.add_argument(
        "--electrothermal-output",
        type=Path,
        default=None,
        help="Five-minute electrothermal root. Defaults to electrothermal_telemetry "
        "next to --output.",
    )
    parser.add_argument(
        "--no-electrothermal",
        action="store_true",
        help="Skip the five-minute cell-voltage and temperature aggregates.",
    )
    parser.add_argument(
        "--dense-field-device",
        action="store_true",
        help="Also write the continuous field devices 100000002 (NCA) and 100000021 (LFP), "
        "about 9 M rows each.",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    electrothermal = None
    if not args.no_electrothermal:
        electrothermal = (
            args.electrothermal_output or output.parent / "electrothermal_telemetry"
        ).resolve()
    generate(
        output,
        electrothermal_output=electrothermal,
        dense_field_device=args.dense_field_device,
    )


if __name__ == "__main__":
    main()
