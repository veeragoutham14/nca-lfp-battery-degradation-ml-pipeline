from __future__ import annotations

import pandas as pd

from src.electrothermal_capacity_context import _extract_local_event_frame
from src.electrothermal_contract import expected_hierarchical_signals


def test_local_event_frame_reads_the_padded_window(tmp_path) -> None:
    signals = [signal.source_signal_name for signal in expected_hierarchical_signals(1)]
    times = pd.date_range("2024-01-01T00:00:00Z", periods=120, freq="min")
    frame = pd.DataFrame(
        {
            "time": times,
            "serial": "100000099",
            "batt_soc_pct": 50.0,
            "batt_soh_pct": 99.0,
            "batt_current_a": -5.0,
            "batt_voltage_v": 50.0,
            **{name: 3600.0 for name in signals},
        }
    )
    partition = tmp_path / "serial=100000099" / "year=2024" / "month=01"
    partition.mkdir(parents=True)
    frame.to_parquet(partition / "part.parquet", index=False)

    window = _extract_local_event_frame(
        tmp_path,
        serial="100000099",
        module_count=1,
        start=pd.Timestamp("2024-01-01T00:40:00Z"),
        end=pd.Timestamp("2024-01-01T00:50:00Z"),
        pre_minutes=10,
        post_minutes=10,
        timestamp_col="time",
    )

    assert window["time"].min() == pd.Timestamp("2024-01-01T00:30:00Z")
    assert window["time"].max() == pd.Timestamp("2024-01-01T01:00:00Z")
    assert set(signals) <= set(window.columns)


def test_local_event_frame_is_empty_without_files(tmp_path) -> None:
    window = _extract_local_event_frame(
        tmp_path,
        serial="100000099",
        module_count=1,
        start=pd.Timestamp("2024-01-01T00:40:00Z"),
        end=pd.Timestamp("2024-01-01T00:50:00Z"),
        pre_minutes=10,
        post_minutes=10,
        timestamp_col="time",
    )
    assert window.empty
