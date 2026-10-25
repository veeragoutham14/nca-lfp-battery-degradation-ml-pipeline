from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.bms_hierarchical_features import feature_column
from src.build_local_electrothermal_features import _normalized_group, build_features
from src.electrothermal_contract import expected_hierarchical_signals, voltage_signal


def test_normalized_group_discards_an_implausible_channel_bucket() -> None:
    signal = voltage_signal(1, 1, 0)
    source = signal.source_signal_name
    result = _normalized_group(
        pd.DataFrame(
            {
                "time": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "serial": [100000001],
                f"{source}__mean": [5960.0],
                f"{source}__min": [3700.0],
                f"{source}__max": [5960.0],
                f"{source}__std": [10.0],
                f"{source}__count": [300],
            }
        ),
        [signal],
    )

    assert pd.isna(result.loc[0, feature_column(signal, "mean")])
    assert result.loc[0, feature_column(signal, "valid_count")] == 0
    assert result.loc[0, "serial"] == "100000001"


def test_builds_feature_contract_from_validated_local_datasets(tmp_path: Path) -> None:
    serial = "TEST001"
    chemistry = "NCA"
    module_count = 1
    start = datetime(2026, 1, 1, tzinfo=UTC)
    signals = expected_hierarchical_signals(module_count)

    raw_dir = tmp_path / "raw_parquet" / f"chemistry={chemistry}" / f"serial={serial}"
    raw_dir.mkdir(parents=True)
    times = [start + timedelta(seconds=index) for index in range(300)]
    pd.DataFrame(
        {
            "time": times,
            "serial": serial,
            "batt_soc_pct": 80.0,
            "batt_soh_pct": 98.0,
            "batt_current_a": -10.0,
            "batt_voltage_v": 100.0,
        }
    ).to_parquet(raw_dir / "part-00000.parquet", index=False)

    for group, measurement_type, value in (
        ("cell_voltage", "cell_voltage_channel", 3700.0),
        ("temperature", "pack_temperature_sensor", 25.0),
    ):
        group_signals = [item for item in signals if item.measurement_type == measurement_type]
        payload: dict[str, object] = {"time": [start], "serial": [serial]}
        for signal in group_signals:
            for statistic, statistic_value in (
                ("mean", value),
                ("min", value - 1),
                ("max", value + 1),
                ("std", 0.5),
                ("count", 300),
            ):
                payload[f"{signal.source_signal_name}__{statistic}"] = [statistic_value]
        path = (
            tmp_path
            / "electrothermal_telemetry"
            / group
            / "5m"
            / f"chemistry={chemistry}"
            / f"serial={serial}"
            / "year=2026"
            / "month=01"
        )
        path.mkdir(parents=True)
        pd.DataFrame(payload).to_parquet(path / "part-00000.parquet", index=False)

    validation_dir = (
        tmp_path
        / "processed"
        / "electrothermal_import_validation"
        / f"chemistry={chemistry}"
        / f"serial={serial}"
        / "validation_run=20260101T000000Z"
    )
    validation_dir.mkdir(parents=True)
    validation_dir.joinpath("model_summary.json").write_text(
        json.dumps(
            {
                "ready": True,
                "module_count": module_count,
                "groups": {
                    "cell_voltage": {"start_utc": start.isoformat(), "end_utc": start.isoformat()},
                    "temperature": {"start_utc": start.isoformat(), "end_utc": start.isoformat()},
                },
            }
        ),
        encoding="utf-8",
    )

    run_dir = build_features(
        data_root=tmp_path,
        serial=serial,
        chemistry=chemistry,
        module_count=module_count,
        compression="zstd",
    )

    summary = json.loads(run_dir.joinpath("model_summary.json").read_text(encoding="utf-8"))
    assert summary["feature_rows"] == 1
    assert summary["events_required"] is False
    assert (run_dir / "year=2026" / "month=01" / "voltage_channel_features_5min.parquet").exists()
    assert (run_dir / "year=2026" / "month=01" / "temperature_sensor_features_5min.parquet").exists()
