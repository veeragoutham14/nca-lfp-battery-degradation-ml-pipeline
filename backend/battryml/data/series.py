from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.catalog.repository import CatalogRepository


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(path: Path, requested_columns: list[str]) -> list[dict[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    schema_names = set(parquet_file.schema_arrow.names)
    columns = [column for column in requested_columns if column in schema_names]
    if not columns:
        return []
    # Read the file directly so parent paths such as serial=100000017 are not
    # interpreted as Hive partitions and merged into the physical file schema.
    table = parquet_file.read(columns=columns)
    return [{key: _json_value(value) for key, value in row.items()} for row in table.to_pylist()]


def _thin(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    step = (len(rows) - 1) / (max_points - 1)
    indexes = sorted({round(index * step) for index in range(max_points)})
    return [rows[index] for index in indexes]


class RunSeriesService:
    def __init__(self, repository: CatalogRepository):
        self.repository = repository

    def run_series(self, run_id: str, max_history_points: int = 2000) -> dict[str, Any] | None:
        run = self.repository.get_run(run_id)
        if run is None:
            return None
        summary = self.repository.get_summary(run_id) or {}
        nominal = summary.get("nominal_capacity_ah") or run.get("nominal_capacity_ah")
        history_artifact = self.repository.find_artifact(run_id, ["history"])
        forecast_artifact = self.repository.find_artifact(run_id, ["forecast"])
        seasonal_artifact = self.repository.find_artifact(run_id, ["seasonal_profile"])
        transition_artifact = self.repository.find_artifact(run_id, ["transition_context"])

        history = self._history(history_artifact, max_history_points)
        forecasts = self._forecast(forecast_artifact, nominal)
        seasonal_profile = self._seasonal_profile(
            seasonal_artifact, max_history_points
        )
        soh_transitions = self._soh_transitions(
            transition_artifact, max_history_points
        )
        return {
            "run_id": run_id,
            "serial": run["serial"],
            "model_family": run["model_family"],
            "nominal_capacity_ah": nominal,
            "history": history,
            "forecasts": forecasts,
            "seasonal_profile": seasonal_profile,
            "soh_transitions": soh_transitions,
        }

    def validation_series(self, run_id: str, max_points: int = 2000) -> dict[str, Any] | None:
        run = self.repository.get_run(run_id)
        if run is None:
            return None
        artifact = self.repository.find_artifact(run_id, ["validation"])
        if artifact is None:
            return {"run_id": run_id, "points": []}
        rows = _records(
            Path(artifact["path"]),
            [
                "end_anchor_time",
                "event_discharge_id",
                "cumulative_all_discharge_ah",
                "actual_capacity_ah",
                "predicted_capacity_ah",
                "capacity_error_ah",
                "actual_soh_pct",
                "predicted_soh_pct",
                "soh_error_pct",
                "abs_soh_error_pct",
            ],
        )
        points: list[dict[str, Any]] = []
        for row in rows:
            points.append(
                {
                    "timestamp": row.get("end_anchor_time"),
                    "event_id": row.get("event_discharge_id"),
                    "cumulative_discharge_ah": row.get("cumulative_all_discharge_ah"),
                    "actual_capacity_ah": row.get("actual_capacity_ah"),
                    "predicted_capacity_ah": row.get("predicted_capacity_ah"),
                    "capacity_error_ah": row.get("capacity_error_ah"),
                    "actual_soh_pct": row.get("actual_soh_pct"),
                    "predicted_soh_pct": row.get("predicted_soh_pct"),
                    "soh_error_pct": row.get("soh_error_pct"),
                    "abs_soh_error_pct": row.get("abs_soh_error_pct"),
                }
            )
        points.sort(key=lambda item: item.get("timestamp") or "")
        return {"run_id": run_id, "points": _thin(points, max_points)}

    @staticmethod
    def _history(artifact: dict[str, Any] | None, max_points: int) -> list[dict[str, Any]]:
        if artifact is None:
            return []
        rows = _records(
            Path(artifact["path"]),
            [
                "end_anchor_time",
                "capacity_ah",
                "measured_soh_pct",
                "bms_soh_end",
                "cumulative_all_discharge_ah",
                "valid_training_row",
                "sample_count",
            ],
        )
        history: list[dict[str, Any]] = []
        for row in rows:
            timestamp = row.get("end_anchor_time")
            if timestamp is None or row.get("measured_soh_pct") is None:
                continue
            history.append(
                {
                    "timestamp": timestamp,
                    "capacity_ah": row.get("capacity_ah"),
                    "soh_pct": row.get("measured_soh_pct"),
                    "bms_soh_pct": row.get("bms_soh_end"),
                    "cumulative_discharge_ah": row.get("cumulative_all_discharge_ah"),
                    "valid_training_row": row.get("valid_training_row"),
                    "sample_count": row.get("sample_count"),
                }
            )
        history.sort(key=lambda item: item["timestamp"])
        return _thin(history, max(2, max_points))

    @staticmethod
    def _forecast(
        artifact: dict[str, Any] | None,
        nominal_capacity_ah: float | None,
    ) -> list[dict[str, Any]]:
        if artifact is None:
            return []
        rows = _records(
            Path(artifact["path"]),
            [
                "forecast_timestamp",
                "scenario",
                "usage_rate_model",
                "forecast_ah_per_day",
                "forecast_days_after_last_measurement",
                "cumulative_all_discharge_ah",
                "predicted_capacity_ah",
                "predicted_capacity_ah_lower_95",
                "predicted_capacity_ah_upper_95",
                "predicted_future_soh_pct",
                "projected_soh_error_pct_final",
                "projected_soh_error_pct_slope",
            ],
        )
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            scenario = row.get("scenario") or row.get("usage_rate_model") or "forecast"
            lower_capacity = row.get("predicted_capacity_ah_lower_95")
            upper_capacity = row.get("predicted_capacity_ah_upper_95")
            grouped.setdefault(str(scenario), []).append(
                {
                    "timestamp": row.get("forecast_timestamp"),
                    "days_ahead": row.get("forecast_days_after_last_measurement"),
                    "soh_pct": row.get("predicted_future_soh_pct"),
                    "lower_95_soh_pct": (
                        100.0 * lower_capacity / nominal_capacity_ah
                        if lower_capacity is not None and nominal_capacity_ah
                        else None
                    ),
                    "upper_95_soh_pct": (
                        100.0 * upper_capacity / nominal_capacity_ah
                        if upper_capacity is not None and nominal_capacity_ah
                        else None
                    ),
                    "capacity_ah": row.get("predicted_capacity_ah"),
                    "cumulative_discharge_ah": row.get("cumulative_all_discharge_ah"),
                    "forecast_ah_per_day": row.get("forecast_ah_per_day"),
                    "usage_error_soh_pct_final": row.get("projected_soh_error_pct_final"),
                    "usage_error_soh_pct_slope": row.get("projected_soh_error_pct_slope"),
                }
            )
        return [
            {
                "scenario": scenario,
                "points": sorted(points, key=lambda item: item.get("timestamp") or ""),
            }
            for scenario, points in sorted(grouped.items())
        ]

    @staticmethod
    def _seasonal_profile(
        artifact: dict[str, Any] | None,
        max_points: int,
    ) -> list[dict[str, Any]]:
        if artifact is None:
            return []
        rows = _records(
            Path(artifact["path"]),
            [
                "month_start_utc",
                "season",
                "expected_days_in_observed_scope",
                "observed_days",
                "telemetry_coverage_fraction",
                "seasonal_profile_valid",
                "charge_ah",
                "discharge_ah",
                "charge_ah_per_observed_day",
                "discharge_ah_per_observed_day",
                "charging_hours",
                "discharging_hours",
                "rest_hours",
                "high_soc_95_hours",
                "low_soc_5_hours",
                "soc_0_5_hours",
                "soc_5_20_hours",
                "soc_20_40_hours",
                "soc_40_60_hours",
                "soc_60_80_hours",
                "soc_80_95_hours",
                "soc_95_100_hours",
                "soc_known_hours",
                "soc_unknown_hours",
                "soc_known_coverage_fraction",
                "temperature_mean_c",
                "temperature_p95_daily_c",
                "temperature_max_c",
                "hot_35_hours",
                "voltage_spread_p95_daily_mv",
                "voltage_spread_max_mv",
                "charge_episode_count",
                "near_full_charge_episode_count",
                "median_charge_start_hour_utc",
                "bms_soh_first_pct",
                "bms_soh_last_pct",
                "audited_soh_change_pp",
                "raw_rms_current_a",
                "raw_peak_charge_current_a",
                "raw_peak_discharge_current_a",
                "raw_peak_10s_mean_absolute_current_a",
                "raw_peak_60s_mean_absolute_current_a",
                "raw_p95_current_step_a",
                "raw_large_current_step_count",
                "raw_current_reversal_count",
                "raw_high_current_episode_count",
                "raw_c0_hours",
                "raw_c1_hours",
                "raw_c2_hours",
                "raw_c3_hours",
                "raw_c4_hours",
                "high_discharge_low_soc_hours",
                "high_charge_high_soc_hours",
                "charging_above_95_soc_hours",
                "high_current_hot_hours",
                "high_charge_cold_hours",
                "high_current_imbalanced_hours",
                "low_soc_high_discharge_imbalanced_hours",
                "temperature_below_10_hours",
                "temperature_10_20_hours",
                "temperature_20_30_hours",
                "temperature_30_40_hours",
                "temperature_above_40_hours",
            ],
        )
        rows.sort(key=lambda item: item.get("month_start_utc") or "")
        return _thin(rows, max(2, max_points))

    @staticmethod
    def _soh_transitions(
        artifact: dict[str, Any] | None,
        max_points: int,
    ) -> list[dict[str, Any]]:
        if artifact is None:
            return []
        rows = _records(
            Path(artifact["path"]),
            [
                "transition_id",
                "event_time_utc",
                "event_type",
                "previous_soh_pct",
                "new_soh_pct",
                "soh_delta_pp",
                "context_quality_ready",
                "event_during_charge_episode",
                "near_full_charge_within_72h_before",
                "hours_since_near_full_charge",
                "authoritative_recalibration_match",
                "recalibration_transition_detection_status",
                "recalibration_complete_discharge_within_lookback",
                "recalibration_hours_since_complete_discharge",
                "recalibration_raw_start_soc_pct",
                "recalibration_raw_end_soc_pct",
                "recalibration_raw_soc_drop_pct",
                "recalibration_discharged_ah_1s",
                "recalibration_charge_intrusion_ah_1s",
                "recalibration_raw_coverage_fraction",
                "recalibration_rest_minutes_inside_cycle",
                "recalibration_longest_rest_minutes",
                "recalibration_rest_interruption_count",
                "recalibration_cycle_continuity_class",
                "recalibration_capacity_validation_status",
                "recalibration_direct_capacity_soh_pct",
                "recalibration_soc_normalized_capacity_soh_pct",
                "recalibration_bms_after_event_minus_capacity_soh_pp",
                "recalibration_primary_context_pattern",
                "recalibration_context_match_strength",
                "pre_mean_coverage_fraction",
                "pre_charge_ah",
                "pre_discharge_ah",
                "pre_charge_episode_count",
                "pre_near_full_charge_episode_count",
                "pre_high_soc_95_hours",
                "pre_low_soc_5_hours",
                "pre_temperature_mean_c",
                "pre_temperature_max_c",
                "pre_voltage_spread_p95_mv",
                "pre_voltage_spread_max_mv",
                "post_mean_coverage_fraction",
                "post_charge_ah",
                "post_discharge_ah",
                "post_charge_episode_count",
                "post_near_full_charge_episode_count",
                "post_high_soc_95_hours",
                "post_low_soc_5_hours",
                "post_temperature_mean_c",
                "post_temperature_max_c",
                "post_voltage_spread_p95_mv",
                "post_voltage_spread_max_mv",
            ],
        )
        rows.sort(key=lambda item: item.get("event_time_utc") or "")
        return _thin(rows, max(2, max_points))
