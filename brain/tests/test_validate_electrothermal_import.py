from __future__ import annotations

import pandas as pd
import pytest

from src.validate_electrothermal_import import _validate_group


def test_validate_group_reads_parquet_paths_and_filters_the_requested_serial(tmp_path) -> None:
    root = tmp_path / "temperature"
    root.mkdir()
    pd.DataFrame(
        {
            "time": [pd.Timestamp("2026-08-25T10:00:00Z")],
            "serial": ["100000001"],
            "sensor_a__mean": [25.0],
            "sensor_a__min": [24.0],
            "sensor_a__max": [26.0],
            "sensor_a__std": [0.5],
            "sensor_a__count": [300],
        }
    ).to_parquet(root / "part-00000.parquet", index=False)

    result = _validate_group(
        root=root,
        serial="100000001",
        expected_columns=["sensor_a"],
        lower=-40.0,
        upper=100.0,
    )

    assert result["ready"] is True
    assert result["rows"] == 1
    assert result["wrong_serial_rows"] == 0
    assert result["minimum_channel_coverage"] == 1.0


def test_validate_group_quantifies_sparse_invalid_sentinel_buckets(tmp_path) -> None:
    root = tmp_path / "cell_voltage"
    root.mkdir()
    rows = 101
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2026-08-01", periods=rows, freq="5min", tz="UTC"),
            "serial": ["100000001"] * rows,
            "cell_a__mean": [3700.0] * 100 + [5960.0],
            "cell_a__min": [3690.0] * 100 + [3690.0],
            "cell_a__max": [3710.0] * 100 + [5960.0],
            "cell_a__std": [2.0] * 100 + [1000.0],
            "cell_a__count": [300] * rows,
        }
    )
    frame.to_parquet(root / "part-00000.parquet", index=False)

    result = _validate_group(
        root=root,
        serial="100000001",
        expected_columns=["cell_a"],
        lower=1500.0,
        upper=5000.0,
    )

    assert result["ready"] is True
    assert result["minimum_channel_coverage"] == pytest.approx(100 / 101)
    assert result["maximum_invalid_bucket_fraction"] == pytest.approx(1 / 101)
