from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_synthetic_telemetry.py"


def _generator():
    spec = importlib.util.spec_from_file_location("generate_synthetic_telemetry", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cell_voltage_curves_stay_in_rated_windows_and_keep_chemistry_shape():
    gen = _generator()
    soc = np.linspace(0.0, 100.0, 201)
    warm = np.full_like(soc, 25.0)
    for chemistry, (low, high) in gen.CELL_VOLTAGE_WINDOW_MV.items():
        rest = gen.cell_voltage_mv(soc, 0.0, warm, chemistry)
        assert np.all(np.diff(rest) >= 0.0)
        for current in (-40.0, 40.0):
            loaded = gen.cell_voltage_mv(soc, current, warm, chemistry)
            assert loaded.min() >= low and loaded.max() <= high
    plateau = (soc >= 20.0) & (soc <= 80.0)
    lfp = gen.cell_voltage_mv(soc, 0.0, warm, "LFP")[plateau]
    nca = gen.cell_voltage_mv(soc, 0.0, warm, "NCA")[plateau]
    assert np.ptp(lfp) < 100.0 < np.ptp(nca)


def test_cold_cells_show_a_larger_ohmic_drop():
    gen = _generator()
    soc = np.array([50.0])
    rest = gen.cell_voltage_mv(soc, 0.0, np.array([25.0]), "NCA")[0]
    warm = rest - gen.cell_voltage_mv(soc, -20.0, np.array([25.0]), "NCA")[0]
    cold = rest - gen.cell_voltage_mv(soc, -20.0, np.array([0.0]), "NCA")[0]
    assert cold > warm > 0.0


def test_dense_field_plan_is_deterministic_with_soh_steps_and_a_planted_offset():
    gen = _generator()
    first = gen._dense_field_segments()
    assert first == gen._dense_field_segments()
    assert sorted({segment.soh_pct for segment in first}, reverse=True) == [98.0, 97.0, 96.0, 95.0]
    offsets = {segment.start.month for segment in first if segment.cell_bias_mv}
    assert offsets == {7}
    lfp = gen._dense_field_segments(
        capacity_ah=50.0, fade_pct_per_day_25c=0.006, anomaly_month=10, anomaly_mv=-25.0
    )
    assert {segment.start.month for segment in lfp if segment.cell_bias_mv < 0} == {10}


def test_dense_writer_splits_each_day_into_its_own_part(tmp_path):
    from datetime import UTC, datetime

    import pyarrow.parquet as pq

    gen = _generator()
    segments = [
        gen.Segment(datetime(2024, 1, day, 6, tzinfo=UTC), 120, 90.0, 80.0, -10.0, 98.0, 5.0, 1)
        for day in (4, 5)
    ]
    rows = gen._write_device_streaming(gen._load_columns(), 100000099, segments, tmp_path)
    parts = sorted((tmp_path / "serial=100000099" / "year=2024" / "month=01").glob("*.parquet"))
    assert [path.name for path in parts] == ["part-004.parquet", "part-005.parquet"]
    assert rows == sum(pq.ParquetFile(path).metadata.num_rows for path in parts) == 240


def test_full_discharge_day_is_mostly_discharge_between_95_and_5_percent():
    from datetime import UTC, datetime

    gen = _generator()
    segments, _ = gen._dense_day(
        datetime(2024, 1, 4, tzinfo=UTC), soc_start=30.0, true_capacity_ah=63.0,
        bms_soh_pct=98.0, ambient_c=4.0, charge_current_a=14.0, full_discharge=True,
        cell_bias_mv=0.0, cycle=1,
    )
    peak = max(range(len(segments)), key=lambda index: segments[index].soc_end)
    charge = segments[peak]
    tail = segments[peak + 1 :]
    discharge = next(segment for segment in tail if segment.current_a < 0)
    assert discharge.soc_end == 1.0
    above_95 = charge.rows * (100.0 - 95.0) / (charge.soc_end - charge.soc_start)
    rest = sum(segment.rows for segment in tail[: tail.index(discharge)])
    assert discharge.rows / (discharge.rows + rest + above_95) >= 0.8


def test_bms_soh_steps_when_the_charge_after_a_full_discharge_starts():
    segments = _generator()._dense_field_segments()
    steps = [
        (previous, current)
        for previous, current in zip(segments, segments[1:], strict=False)
        if current.soh_pct != previous.soh_pct
    ]
    assert [current.soh_pct for _, current in steps] == [97.0, 96.0, 95.0]
    for previous, current in steps:
        assert current.start.day == 5 and current.start.hour == 10
        assert current.current_a > 0  # the step arrives with the charge
        assert previous.soc_end > 0.0  # the full discharge ended the day before
        full_discharge = next(
            segment
            for segment in reversed(segments[: segments.index(current)])
            if segment.soc_end == 1.0 and segment.current_a < 0
        )
        assert full_discharge.start.day == 4


def test_module_count_snapshot_matches_the_device_registry_reader(tmp_path):
    from battryml.data.devices import _module_count_snapshot

    raw = tmp_path / "data" / "raw_parquet"
    _generator().write_module_count_snapshot(raw, 100000002)
    snapshot = _module_count_snapshot(
        tmp_path / "data" / "processed" / "device_metadata" / "module_counts.parquet"
    )
    assert snapshot["100000002"]["module_count"] == 2
    assert snapshot["100000002"]["module_count_status"] == "verified"
