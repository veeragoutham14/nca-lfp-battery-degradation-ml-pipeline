from __future__ import annotations

import pandas as pd

from src.discharge_capacity_evidence import _episode_context


def test_episode_context_uses_only_the_discharge_interval() -> None:
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    validations = pd.DataFrame(
        {
            "discharge_episode_id": ["episode_1"],
            "raw_start_time_utc": [start],
            "raw_end_time_utc": [start + pd.Timedelta(minutes=10)],
        }
    )
    windows = pd.DataFrame(
        {
            "window_start_utc": [
                start - pd.Timedelta(minutes=5),
                start,
                start + pd.Timedelta(minutes=5),
                start + pd.Timedelta(minutes=15),
            ],
            "stable_pack_temperature_mean_c": [99.0, 8.0, 32.0, 99.0],
            "stable_pack_temperature_max_c": [99.0, 9.0, 36.0, 99.0],
            "system_voltage_spread_mv": [999.0, 12.0, 24.0, 999.0],
            "observed_minutes": [5.0, 5.0, 5.0, 5.0],
        }
    )

    result = _episode_context(validations, windows).iloc[0]

    assert result["pack_temperature_mean_c"] == 20.0
    assert result["pack_temperature_median_c"] == 20.0
    assert result["pack_temperature_min_c"] == 8.0
    assert result["pack_temperature_max_c"] == 36.0
    assert result["pack_temperature_below_10_minutes"] == 5.0
    assert result["pack_temperature_below_15_minutes"] == 5.0
    assert result["pack_temperature_at_or_above_30_minutes"] == 5.0
    assert result["cell_voltage_spread_max_mv"] == 24.0
    assert result["pack_temperature_context_coverage_fraction"] == 1.0
