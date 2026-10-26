from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from battryml.analytics.cell_anomaly_overview import (
    build_cell_anomaly_overview,
    build_cell_anomaly_series,
)


def test_cell_anomaly_overview_archives_legacy_runs(tmp_path: Path) -> None:
    feature_run = (
        tmp_path
        / "data"
        / "processed"
        / "cell_health"
        / "serial=100000002"
        / "feature_run=20260701T000000Z"
    )
    anomaly_run = (
        tmp_path
        / "data"
        / "processed"
        / "cell_anomaly"
        / "serial=100000002"
        / "anomaly_run=20260702T000000Z"
    )
    feature_run.mkdir(parents=True)
    anomaly_run.mkdir(parents=True)
    (feature_run / "model_summary.json").write_text("{}", encoding="utf-8")
    (anomaly_run / "model_summary.json").write_text(
        json.dumps(
            {
                "model_family": "nca_cell_voltage_anomaly",
                "analysis_start": "2026-04-01T00:00:00+00:00",
                "analysis_end": "2026-05-31T00:00:00+00:00",
                "feature_rows": 100,
                "quality_valid_pct": 99.5,
                "scored_pct_of_valid": 100.0,
                "sustained_anomaly_episode_count": 3,
                "anomaly_days": 2,
                "maximum_anomaly_score": 25.0,
                "p95_analysis_spread_mv": 120.0,
                "maximum_analysis_spread_mv": 170.0,
                "module_configuration_epoch_count": 2,
                "elevated_module_epoch_shift_count": 1,
                "maximum_module_epoch_baseline_shift_mv": 110.0,
                "source_feature_run_id": feature_run.name,
                "module_epoch_baselines": [{"epoch": 2, "baseline_p95_mv": 120.0}],
                "claim_limit": "Candidate only.",
            }
        ),
        encoding="utf-8",
    )
    timestamps = [
        datetime(2026, 4, 1, tzinfo=UTC),
        datetime(2026, 4, 2, tzinfo=UTC),
    ]
    pq.write_table(
        pa.table(
            {
                "time": timestamps,
                "module_count": [2, 3],
                "module_configuration_epoch": [1, 2],
                "spread_p50_mv": [10.0, 100.0],
                "spread_p95_mv": [20.0, 120.0],
                "spread_max_mv": [30.0, 150.0],
                "raw_spread_p95_mv": [20.0, 130.0],
                "compensated_spread_p95_mv": [20.0, 120.0],
                "baseline_threshold_median_mv": [25.0, 125.0],
                "point_candidate_bins": [0, 4],
                "sustained_candidate_bins": [0, 3],
                "sustained_episode_minutes": [0.0, 15.0],
                "maximum_anomaly_score": [0.0, 12.0],
                "soc_mean_pct": [60.0, 55.0],
                "temperature_mean_c": [20.0, 22.0],
            }
        ),
        anomaly_run / "daily_cell_voltage_anomaly.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_id": ["episode-1"],
                "start_time": [timestamps[1]],
                "end_time": [timestamps[1]],
                "duration_minutes": [15.0],
                "severity": ["warning"],
            }
        ),
        anomaly_run / "cell_voltage_anomaly_episodes.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "time": [
                    timestamps[1],
                    timestamps[1] + timedelta(minutes=5),
                    timestamps[1] + timedelta(minutes=10),
                ],
                "operating_state": ["charging", "discharging", "discharging"],
                "in_sustained_episode": [True, True, True],
            }
        ),
        anomaly_run / "cell_voltage_anomaly_scores.parquet",
    )
    devices = [
        {
            "serial": "100000002",
            "variant_key": "nca-home",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
        },
        {
            "serial": "100000006",
            "variant_key": "nca-home-15",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 3,
            "module_count_status": "verified",
        },
        {
            "serial": "100000017",
            "variant_key": "nca-reference",
            "chemistry": "NCA",
            "device_role": "full_cycle_test",
            "module_count": 2,
            "module_count_status": "verified",
        },
        {"serial": "LFP001", "chemistry": "LFP", "device_role": "field_tester"},
    ]

    overview = build_cell_anomaly_overview(source_root=tmp_path, devices=devices)
    series = build_cell_anomaly_series(
        source_root=tmp_path,
        serial="100000002",
        max_points=100,
    )

    assert overview["metrics"]["inventory_device_count"] == 3
    assert overview["metrics"]["configured_device_count"] == 3
    assert overview["metrics"]["deferred_device_count"] == 0
    assert overview["metrics"]["field_device_count"] == 2
    assert overview["metrics"]["reference_device_count"] == 1
    assert overview["metrics"]["analyzed_device_count"] == 0
    assert overview["metrics"]["archived_legacy_device_count"] == 0
    assert overview["metrics"]["elevated_epoch_shift_device_count"] == 0
    assert overview["devices"][0]["history_built"] is False
    assert overview["devices"][0]["legacy_history_archived"] is False
    assert overview["devices"][0]["analyzed"] is False
    assert overview["devices"][0]["legacy_analysis_archived"] is False
    assert overview["devices"][0]["maximum_module_epoch_baseline_shift_mv"] is None
    assert overview["devices"][1]["serial"] == "100000017"
    assert overview["devices"][1]["cohort_role"] == "reference_control"
    assert overview["devices"][2]["serial"] == "100000006"
    assert overview["devices"][2]["cohort_status"] == "included"
    assert series is None


def test_hierarchical_anomaly_results_expose_channel_and_raw_evidence(tmp_path: Path) -> None:
    feature_run = (
        tmp_path
        / "data"
        / "processed"
        / "bms_hierarchical_features"
        / "serial=100000001"
        / "feature_run=20260727T000000Z"
    )
    anomaly_run = (
        tmp_path
        / "data"
        / "processed"
        / "hierarchical_cell_anomaly"
        / "serial=100000001"
        / "anomaly_run=20260728T000000Z"
    )
    feature_run.mkdir(parents=True)
    anomaly_run.mkdir(parents=True)
    (feature_run / "model_summary.json").write_text("{}", encoding="utf-8")
    (anomaly_run / "model_summary.json").write_text(
        json.dumps(
            {
                "model_family": "hierarchical_cell_voltage_anomaly",
                "module_count": 2,
                "analysis_start": "2026-07-01T00:00:00+00:00",
                "analysis_end": "2026-07-02T00:00:00+00:00",
                "feature_rows": 288,
                "quality_valid_rows": 285,
                "quality_valid_pct": 99.0,
                "sustained_anomaly_episode_count": 51,
                "anomaly_days": 1,
                "p95_analysis_spread_mv": 42.0,
                "maximum_analysis_spread_mv": 75.0,
                "maximum_bms_reported_voltage_spread_mv": 18960.0,
                "bms_global_spread_implausible_windows": 3,
                "spread_basis": "hierarchical_channel_means_only",
                "maximum_peer_z": 8.0,
                "source_feature_run_id": feature_run.name,
                "first_episode_start": "2026-07-01T10:00:00+00:00",
                "latest_episode_end": "2026-07-01T10:15:00+00:00",
                "episode_severity_counts": {"engineering_review": 51},
                "evidence_episode_count": 1,
                "detector": {"fixed_engineering_bands_mv": {"engineering_review": [50, 100]}},
                "claim_limit": "Candidate only.",
            }
        ),
        encoding="utf-8",
    )
    timestamp = datetime(2026, 7, 1, tzinfo=UTC)
    pq.write_table(
        pa.table(
            {
                "day_utc": [timestamp],
                "system_spread_p50_mv": [15.0],
                "system_spread_p95_mv": [42.0],
                "system_spread_max_mv": [75.0],
                "maximum_peer_z": [8.0],
                "maximum_low_rank_z": [7.0],
                "maximum_temporal_ewma_score": [6.5],
                "point_candidate_windows": [3],
                "sustained_episode_minutes": [15.0],
                "sustained_charging_minutes": [0.0],
                "sustained_discharging_minutes": [15.0],
                "sustained_rest_minutes": [0.0],
                "sustained_mixed_minutes": [0.0],
                "soc_mean_pct": [55.0],
                "temperature_mean_c": [24.0],
            }
        ),
        anomaly_run / "daily_anomaly_summary.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_id": [f"episode_{index:05d}" for index in range(1, 52)],
                "start_time_utc": [
                    timestamp + timedelta(minutes=20 * index) for index in range(51)
                ],
                "end_time_utc": [
                    timestamp + timedelta(minutes=20 * index + 15) for index in range(51)
                ],
                "duration_minutes": [15.0] * 51,
                "severity": ["engineering_review"] * 51,
                "maximum_spread_mv": [75.0] * 51,
                "affected_channel_key": ["m01_p02_cv07"] * 51,
                "affected_module_index": [1] * 51,
                "affected_pack_index": [2] * 51,
                "affected_channel_index": [7] * 51,
                "maximum_peer_z": [8.0] * 51,
                "maximum_low_rank_z": [7.0] * 51,
                "maximum_temporal_ewma_score": [6.5] * 51,
            }
        ),
        anomaly_run / "anomaly_episodes.parquet",
    )
    evidence = anomaly_run / "evidence" / "episode_id=episode_00001"
    evidence.mkdir(parents=True)
    pq.write_table(pa.table({"time": [timestamp]}), evidence / "episode_evidence_1s.parquet")
    devices = [
        {
            "serial": "100000001",
            "variant_key": "nca-home",
            "chemistry": "NCA",
            "device_role": "field_tester",
            "module_count": 2,
            "module_count_status": "verified",
        }
    ]
    overview = build_cell_anomaly_overview(source_root=tmp_path, devices=devices)
    series = build_cell_anomaly_series(source_root=tmp_path, serial="100000001")
    assert overview["devices"][0]["detector_version"] == "hierarchical_v2"
    assert series is not None
    assert series["detector_version"] == "hierarchical_v2"
    assert series["points"][0]["spread_p95_mv"] == 42.0
    assert series["summary"]["maximum_analysis_spread_mv"] == 75.0
    assert series["summary"]["bms_global_spread_implausible_windows"] == 3
    assert series["summary"]["spread_basis"] == "hierarchical_channel_means_only"
    assert len(series["episodes"]) == 51
    assert series["episodes"][0]["affected_channel_key"] == "m01_p02_cv07"
    assert series["episodes"][0]["has_raw_evidence"] is True
