#!/usr/bin/env python3
"""Reproducible fleet benchmark on synthetic data.

1. Writes a synthetic fleet of 1-second telemetry (configured serials from
   config/battery_profiles.json) in the brain's Parquet layout.
2. Runs the brain per device and times every stage:
   - all devices:        src.build_events (row classification + operating events)
   - NCA field devices:  src.fixed_slope_verification (feeds the NCA capacity comparison)
   - lab reference:      src.discharge_cycle_builder + src.capacity_ml
3. Starts the FastAPI backend and times key endpoints, cold (first request
   after start) and warm (repeated requests: median and p95).
4. Reports the production frontend bundle size if frontend/dist exists.

Everything is synthetic. Results go to stdout and to --report (JSON).
Without --keep-data it deletes and regenerates brain/data/ (gitignored);
six months for 30 devices needs about 20 GB of free disk.

    python scripts/benchmark.py --devices 30 --months 6
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "brain"
RAW = BRAIN / "data" / "raw_parquet"
PROCESSED = BRAIN / "data" / "processed"
REFERENCE_SERIAL = 100000017


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "synthetic_generator", ROOT / "scripts" / "generate_synthetic_telemetry.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fleet(device_count: int) -> list[dict]:
    profiles = json.loads((ROOT / "config" / "battery_profiles.json").read_text(encoding="utf-8"))
    devices: list[dict] = []
    for variant_key, variant in profiles["variants"].items():
        for device in variant.get("devices", []):
            devices.append(
                {
                    "serial": int(device["serial"]),
                    "variant": variant_key,
                    "chemistry": variant.get("chemistry", "NCA"),
                    "nominal_capacity_ah": float(variant.get("nominal_capacity_ah") or 50.0),
                    "loss_slopes": variant.get("default_loss_slopes_per_1000ah") or [0.12],
                    "module_count": int(variant.get("loss_slope_reference_module_count") or 2),
                }
            )
    reference = [d for d in devices if d["serial"] == REFERENCE_SERIAL]
    nca = [d for d in devices if d["chemistry"] == "NCA" and d["serial"] != REFERENCE_SERIAL]
    lfp = [d for d in devices if d["chemistry"] == "LFP"]
    # Interleave so any fleet size mixes chemistries.
    ordered = reference[:]
    for index in range(max(len(nca), len(lfp))):
        if index < len(nca):
            ordered.append(nca[index])
        if index < len(lfp) and index % 2 == 0:
            ordered.append(lfp[index])
    for device in lfp:
        if device not in ordered:
            ordered.append(device)
    return ordered[:device_count]


def _field_days(gen, device: dict, index: int, start: datetime, days: int) -> list:
    """Daily partial cycles: evening discharge, next-morning charge."""
    capacity = device["nominal_capacity_ah"]
    depth = 30.0 + (index * 7) % 35  # 30..64 % depth of discharge
    current = 20.0 + (index % 4) * 5.0  # 20..35 A
    fade_per_day = (0.15 + (index % 5) * 0.1) / 30.0  # SOH points per day
    bias_month = 3 if index % 7 == 3 else None
    segments = []
    for day in range(days):
        moment = start + timedelta(days=day)
        soh = 99.5 - fade_per_day * day
        ambient = 12.0 + 10.0 * math.sin(2 * math.pi * (moment.timetuple().tm_yday - 110) / 365.0)
        rows = int(round(depth / 100.0 * capacity / current * 3600))
        bias = (
            45.0 if bias_month is not None and (day // 30) == bias_month and day % 30 < 7 else 0.0
        )
        top = 90.0
        evening = moment.replace(hour=17, minute=0, second=0)
        segments.append(gen.Segment(evening, 60, top, top, 0.2, soh, ambient, day + 1))
        segments.append(
            gen.Segment(
                evening + timedelta(seconds=60),
                rows,
                top,
                top - depth,
                -current,
                soh,
                ambient,
                day + 1,
                cell_bias_mv=bias,
                gap_after_s=120 if day % 9 == 0 else 1,
            )
        )
        morning = moment.replace(hour=9, minute=0, second=0) + timedelta(days=1)
        if day == days - 1 or (morning.month != moment.month):
            # keep each segment inside one calendar month partition
            morning = evening + timedelta(seconds=60 + rows + 600)
        segments.append(
            gen.Segment(
                morning, rows, top - depth, top, current, soh, ambient, day + 1, cell_bias_mv=bias
            )
        )
    return segments


def _reference_cycles(gen, start: datetime, days: int) -> list:
    segments = []
    cycle = 0
    for day in range(0, days, 14):
        moment = (start + timedelta(days=day)).replace(hour=8)
        segments.extend(
            gen._full_cycle(
                moment,
                capacity_ah=56.0 - 0.05 * cycle,
                soh=100.0 - 0.06 * cycle,
                ambient=12.0
                + 10.0 * math.sin(2 * math.pi * (moment.timetuple().tm_yday - 110) / 365.0),
                cycle=cycle + 1,
            )
        )
        cycle += 1
    return segments


def _write_streaming(
    gen, columns, serial: int, segments: list, output: Path, batch_rows: int = 100_000
) -> int:
    """Like the generator's writer, but streams each month in ~100k-row groups.

    That keeps months of 1-second data out of memory while writing.
    """
    import pyarrow as pa

    writers: dict[tuple[int, int], pq.ParquetWriter] = {}
    buffers: dict[tuple[int, int], list] = {}
    rows = 0
    elapsed = 0.0

    def flush(key):
        if buffers.get(key):
            writers[key].write_table(pa.concat_tables(buffers[key]))
            buffers[key] = []

    try:
        for segment in sorted(segments, key=lambda s: s.start):
            table, elapsed = gen._build_table(columns, serial, segment, elapsed_offset_s=elapsed)
            if segment.gap_after_s > 1:
                elapsed += segment.gap_after_s
            key = (segment.start.year, segment.start.month)
            if key not in writers:
                for other in list(writers):
                    flush(other)
                directory = (
                    output / f"serial={serial}" / f"year={key[0]:04d}" / f"month={key[1]:02d}"
                )
                directory.mkdir(parents=True, exist_ok=True)
                writers[key] = pq.ParquetWriter(
                    directory / "part-000.parquet", table.schema, compression="snappy"
                )
                buffers[key] = []
            buffers[key].append(table)
            rows += table.num_rows
            if sum(t.num_rows for t in buffers[key]) >= batch_rows:
                flush(key)
    finally:
        for key, writer in writers.items():
            flush(key)
            writer.close()
    return rows


def _write_module_metadata(fleet: list[dict]) -> Path:
    """Write the module-count snapshot that a metadata import would produce.

    The comparison page only includes field devices whose module count is verified.
    The synthetic fleet has no database, so the snapshot is written from the profiles.
    """
    import pyarrow as pa

    rows = [d for d in fleet if d["chemistry"] == "NCA"]
    path = PROCESSED / "device_metadata" / "module_counts.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "serial": [str(d["serial"]) for d in rows],
            "module_count": [d["module_count"] for d in rows],
            "observed_module_counts": [str(d["module_count"]) for d in rows],
            "module_count_status": ["verified"] * len(rows),
            "module_count_agreement": [1.0] * len(rows),
        }
    )
    pq.write_table(table, path)
    return path


def _dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def _run(cmd: list[str]) -> float:
    started = time.perf_counter()
    result = subprocess.run(
        cmd, cwd=BRAIN, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(BRAIN)}
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(cmd[2:5])} exited with {result.returncode} "
            f"(negative = killed, e.g. out of memory):\n{result.stderr[-2000:]}"
        )
    return time.perf_counter() - started


def _device_pipeline(device: dict) -> dict:
    serial = str(device["serial"])
    py = sys.executable
    stages = {"build_events": _run([py, "-m", "src.build_events", "--serial", serial])}
    if device["serial"] == REFERENCE_SERIAL:
        stages["discharge_cycle_builder"] = _run(
            [py, "-m", "src.discharge_cycle_builder", "--serial", serial]
        )
        stages["capacity_ml"] = _run(
            [
                py,
                "-m",
                "src.capacity_ml",
                "--serial",
                serial,
                "--nominal-capacity-ah",
                str(device["nominal_capacity_ah"]),
                "--future-days",
                "3650",
                "--step-days",
                "30",
            ]
        )
    elif device["chemistry"] == "NCA":
        stages["fixed_slope_verification"] = _run(
            [
                py,
                "-m",
                "src.fixed_slope_verification",
                "--serial",
                serial,
                "--nominal-capacity-ah",
                str(device["nominal_capacity_ah"]),
                "--module-count",
                str(device["module_count"]),
                "--slope-reference-module-count",
                str(device["module_count"]),
                "--loss-slopes-per-1000ah",
                ",".join(str(s) for s in device["loss_slopes"]),
            ]
        )
    return {
        "serial": serial,
        "chemistry": device["chemistry"],
        "stages": stages,
        "seconds": sum(stages.values()),
    }


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _get(url: str) -> tuple[float, int, int]:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=300) as response:
        body = response.read()
        status = response.status
    return (time.perf_counter() - started) * 1000.0, status, len(body)


def _api_bench(port: int, repeats: int, catalog: Path) -> dict:
    env = {
        **os.environ,
        "BATTRYML_SOURCE_ROOT": str(BRAIN),
        "BATTRYML_SOURCE_PYTHON": sys.executable,
        "BATTRYML_AUTH_ENABLED": "false",
        "BATTRYML_CATALOG_PATH": str(catalog),
        "BATTRYML_JOB_ROOT": str(catalog.parent / "jobs"),
        "PYTHONPATH": str(BRAIN),
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "battryml.cli", "serve", "--host", "127.0.0.1", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                _get(base + "/api/health")
                break
            except Exception:
                time.sleep(0.5)
        runs = json.loads(urllib.request.urlopen(base + "/api/runs", timeout=300).read())
        runs = runs.get("items", runs) if isinstance(runs, dict) else runs
        run = next(
            (r for r in runs if str(r.get("serial")) == str(REFERENCE_SERIAL)),
            runs[0] if runs else None,
        )
        run_id = (
            urllib.request.quote(str(run.get("run_id") or run.get("id")), safe="") if run else None
        )
        endpoints = {
            "GET /api/devices": "/api/devices",
            "GET /api/devices?refresh=true": "/api/devices?refresh=true",
            "GET /api/analytics/nca-capacity-overview": (
                "/api/analytics/nca-capacity-overview"
                "?reference_loss_slope_ah_per_1000ah=0.12&bms_tolerance_pct=1"
            ),
            "GET /api/analytics/cell-signal-overview": (
                "/api/analytics/cell-signal-overview?chemistry=NCA"
            ),
        }
        if run_id:
            endpoints["GET /api/runs/{id}/summary"] = f"/api/runs/{run_id}/summary"
            endpoints["GET /api/runs/{id}/series"] = f"/api/runs/{run_id}/series"
        results = {}
        for label, path in endpoints.items():
            cold_ms, status, size = _get(base + path)
            warm = [_get(base + path)[0] for _ in range(repeats)]
            results[label] = {
                "status": status,
                "bytes": size,
                "cold_ms": round(cold_ms, 1),
                "warm_median_ms": round(statistics.median(warm), 1),
                "warm_p95_ms": round(_pct(warm, 0.95), 1),
            }
        return {"runs_in_catalog": len(runs), "endpoints": results}
    finally:
        server.terminate()
        server.wait(timeout=30)


def _frontend_size() -> dict | None:
    dist = ROOT / "frontend" / "dist"
    if not dist.exists():
        return None
    files = [p for p in dist.rglob("*") if p.is_file() and p.suffix in {".js", ".css"}]
    raw = sum(p.stat().st_size for p in files)
    gz = sum(len(gzip.compress(p.read_bytes(), 9)) for p in files)
    return {"js_css_bytes": raw, "js_css_gzip_bytes": gz, "dist_bytes": _dir_bytes(dist)}


def _machine() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_gb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                mem_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        pass
    return {
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_gb": mem_gb,
        "python": platform.python_version(),
        "os": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--devices", type=int, default=30)
    parser.add_argument("--months", type=int, default=6)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help=(
            "parallel devices for the fleet wall-clock run; one brain process peaked at "
            "5.3 GB RSS on six months of 1 s data, so size this to your RAM"
        ),
    )
    parser.add_argument("--repeats", type=int, default=30, help="warm requests per endpoint")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--keep-data", action="store_true", help="reuse existing brain/data instead of regenerating"
    )
    parser.add_argument("--report", type=Path, default=ROOT / "var" / "benchmark.json")
    args = parser.parse_args()

    gen = _load_generator()
    columns = gen._load_columns()
    fleet = _fleet(args.devices)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    days = int(round(args.months * 30.4))
    report: dict = {
        "machine": _machine(),
        "fleet_devices": len(fleet),
        "months": args.months,
        "columns": len(columns),
        "synthetic": True,
    }

    if not args.keep_data:
        shutil.rmtree(BRAIN / "data", ignore_errors=True)
        started = time.perf_counter()
        total_rows = 0
        for index, device in enumerate(fleet):
            segments = (
                _reference_cycles(gen, start, days)
                if device["serial"] == REFERENCE_SERIAL
                else _field_days(gen, device, index, start, days)
            )
            device["rows"] = _write_streaming(gen, columns, device["serial"], segments, RAW)
            total_rows += device["rows"]
        report["generate_seconds"] = round(time.perf_counter() - started, 1)
        report["raw_rows"] = total_rows
    report["raw_parquet_bytes"] = _dir_bytes(RAW)
    report["raw_rows"] = report.get("raw_rows") or sum(
        pq.ParquetFile(p).metadata.num_rows for p in RAW.rglob("*.parquet")
    )
    print(
        f"fleet: {len(fleet)} devices, {report['raw_rows']:,} rows x {len(columns)} columns, "
        f"{report['raw_parquet_bytes'] / 1e6:.0f} MB raw Parquet"
    )

    # Sequential pass: clean per-device timings.
    shutil.rmtree(PROCESSED, ignore_errors=True)
    started = time.perf_counter()
    per_device = [_device_pipeline(device) for device in fleet]
    sequential = time.perf_counter() - started
    rows_by_serial = {}
    for path in RAW.rglob("*.parquet"):
        serial = path.parts[-4].split("=", 1)[1]
        rows_by_serial[serial] = (
            rows_by_serial.get(serial, 0) + pq.ParquetFile(path).metadata.num_rows
        )
    for item in per_device:
        item["rows"] = rows_by_serial.get(item["serial"], 0)
    events = [d["stages"]["build_events"] for d in per_device]
    report["brain_sequential"] = {
        "total_seconds": round(sequential, 1),
        "per_device_median_seconds": round(statistics.median(d["seconds"] for d in per_device), 2),
        "per_device_max_seconds": round(max(d["seconds"] for d in per_device), 2),
        "build_events_rows_per_second": round(report["raw_rows"] / sum(events)),
        "stage_totals_seconds": {
            stage: round(sum(d["stages"].get(stage, 0.0) for d in per_device), 1)
            for stage in sorted({s for d in per_device for s in d["stages"]})
        },
    }
    import resource

    report["brain_sequential"]["peak_process_rss_mb"] = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
    )
    report["per_device"] = per_device
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2)
    )  # keep sequential results even if a later pass fails

    # Parallel pass: fleet wall-clock with N workers.
    shutil.rmtree(PROCESSED, ignore_errors=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(_device_pipeline, fleet))
    report["brain_parallel"] = {
        "workers": args.workers,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    report["processed_bytes"] = _dir_bytes(PROCESSED)

    _write_module_metadata(_fleet(10**6))  # every configured NCA device, as an import would
    catalog = ROOT / "var" / "benchmark" / "catalog.duckdb"
    shutil.rmtree(catalog.parent, ignore_errors=True)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    report["api"] = _api_bench(args.port, args.repeats, catalog)
    report["frontend"] = _frontend_size()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "per_device"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
