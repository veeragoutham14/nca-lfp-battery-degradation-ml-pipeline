from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from battryml.battery_profiles import battery_profile
from battryml.config import Settings


@pytest.fixture
def platform_settings(tmp_path: Path) -> Settings:
    source_root = tmp_path / "Veera_SOH"
    run_dir = (
        source_root
        / "data"
        / "processed"
        / "capacity_ml"
        / "serial=TEST001"
        / "ml_run=20260102T030405Z"
    )
    run_dir.mkdir(parents=True)
    (source_root / "src").mkdir()
    (source_root / "src" / "capacity_ml.py").write_text("# fixture\n", encoding="utf-8")

    lab_nominal_ah = battery_profile("100000017").nominal_capacity_ah
    example_capacity_ah = 47.3
    measured_capacity_ah = [example_capacity_ah + 0.5, example_capacity_ah]
    forecast_capacity_ah = [example_capacity_ah - 0.2, example_capacity_ah - 0.4]
    forecast_lower_ah = [example_capacity_ah - 0.6, example_capacity_ah - 0.8]
    forecast_upper_ah = [example_capacity_ah + 0.2, example_capacity_ah]
    validation_actual_ah = [example_capacity_ah - 0.1, example_capacity_ah - 0.2]
    validation_predicted_ah = [example_capacity_ah - 0.2, example_capacity_ah - 0.3]
    summary = {
        "nominal_capacity_ah": lab_nominal_ah,
        "nominal_capacity_source": "explicit",
        "training_cutoff": "2025-12-31 23:59:59+00:00",
        "usage_rates_by_model": {
            "historical_mean": {"forecast_ah_per_day": 20.0, "forecast_wh_per_day": 6000.0}
        },
        "degradation_forecast_model": {
            "model_name": "degradation_usage_linear",
            "baseline_capacity_ah": lab_nominal_ah,
            "reference_cumulative_ah": 0.0,
            "intercept_loss_ah": 0.1,
            "loss_slope_per_1000ah": 0.08,
            "train_rows": 20,
            "residual_std": 0.2,
        },
        "cutoff_validation": {
            "rows": 2,
            "mae_soh_pct": 0.4,
            "max_abs_soh_error_pct": 0.8,
            "rmse_ah": 0.25,
        },
        "forecast_rows": 2,
    }
    (run_dir / "model_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    timestamps = [
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 6, 1, tzinfo=UTC),
    ]
    pq.write_table(
        pa.table(
            {
                "serial": [100000017, 100000017],
                "end_anchor_time": timestamps,
                "capacity_ah": measured_capacity_ah,
                "measured_soh_pct": [99.0, 98.0],
                "bms_soh_end": [99.0, 98.0],
                "cumulative_all_discharge_ah": [1000.0, 5000.0],
                "valid_training_row": [True, True],
            }
        ),
        run_dir / "ml_all_capacity_rows.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "forecast_timestamp": [
                    datetime(2026, 2, 1, tzinfo=UTC),
                    datetime(2026, 3, 1, tzinfo=UTC),
                ],
                "scenario": ["historical_mean", "historical_mean"],
                "forecast_days_after_last_measurement": [30, 60],
                "predicted_capacity_ah": forecast_capacity_ah,
                "predicted_capacity_ah_lower_95": forecast_lower_ah,
                "predicted_capacity_ah_upper_95": forecast_upper_ah,
                "predicted_future_soh_pct": [97.6, 97.2],
                "cumulative_all_discharge_ah": [5600.0, 6200.0],
                "forecast_ah_per_day": [20.0, 20.0],
            }
        ),
        run_dir / "capacity_forecast.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "end_anchor_time": timestamps,
                "event_discharge_id": [21, 22],
                "cumulative_all_discharge_ah": [5100.0, 5200.0],
                "actual_capacity_ah": validation_actual_ah,
                "predicted_capacity_ah": validation_predicted_ah,
                "capacity_error_ah": [0.1, 0.1],
                "actual_soh_pct": [97.8, 97.6],
                "predicted_soh_pct": [97.6, 97.4],
                "soh_error_pct": [0.2, 0.2],
                "abs_soh_error_pct": [0.2, 0.2],
            }
        ),
        run_dir / "cutoff_validation.parquet",
    )

    return Settings(
        project_root=tmp_path / "BattryML",
        source_root=source_root,
        source_python=Path(sys.executable),
        catalog_path=tmp_path / "catalog.duckdb",
        job_root=tmp_path / "jobs",
        api_host="127.0.0.1",
        api_port=8000,
        allowed_origins=("http://localhost:5173",),
        worker_threads=1,
    )
