from __future__ import annotations

from datetime import datetime
from typing import Any

from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


def _timestamp(value: Any, field: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc


class CellSignalAuditPlugin(PipelinePlugin):
    key = "veera.cell_signal_audit.v1"
    name = "Audit cell-voltage channels"
    description = (
        "Check every active module's cell channels for schema presence, non-null "
        "coverage, and physically plausible values."
    )
    output_family = "cell_signal_availability"
    requires_database = True
    workflow_stage = "extraction"
    device_roles = ("field_tester", "full_cycle_test")
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition("module_count", "Installed modules", "integer", group="Battery"),
        ParameterDefinition(
            "start",
            "Sample start (optional)",
            "datetime",
            group="Sample window",
            description="Leave both dates empty to select the strongest recent UTC day.",
        ),
        ParameterDefinition(
            "end",
            "Sample end (optional)",
            "datetime",
            group="Sample window",
        ),
        ParameterDefinition(
            "readiness_threshold",
            "Required valid coverage",
            "number",
            0.95,
            group="Quality",
        ),
        ParameterDefinition(
            "compression",
            "Parquet compression",
            "choice",
            "zstd",
            choices=("snappy", "zstd", "gzip", "none"),
            group="Storage",
        ),
        ParameterDefinition(
            "log_level",
            "Log level",
            "choice",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            group="Execution",
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        start = normalized.get("start")
        end = normalized.get("end")
        if bool(start) != bool(end):
            raise ValueError("start and end must be provided together")
        if start and end and _timestamp(start, "start") >= _timestamp(end, "end"):
            raise ValueError("start must be earlier than end")
        module_count = normalized.get("module_count")
        if module_count is not None and not 1 <= int(module_count) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        threshold = float(normalized["readiness_threshold"])
        if not 0 < threshold <= 1:
            raise ValueError("readiness_threshold must be between 0 and 1")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = [
            "--serial",
            str(values["serial"]),
            "--readiness-threshold",
            str(values["readiness_threshold"]),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        ]
        for key, flag in {
            "module_count": "--module-count",
            "start": "--start",
            "end": "--end",
        }.items():
            value = values.get(key)
            if value is not None and str(value).strip():
                args.extend((flag, str(value)))
        return (
            str(settings.source_python),
            "-m",
            "src.cell_signal_audit",
            *args,
        )


class CellHealthFeaturesPlugin(PipelinePlugin):
    key = "veera.cell_health_features.v1"
    name = "Build compact cell-health history"
    description = (
        "Aggregate cell spread, compensated spread, temperature, SOC, and load "
        "into a disk-efficient feature history."
    )
    output_family = "cell_health_features"
    requires_database = True
    workflow_stage = "extraction"
    device_roles = ("field_tester", "full_cycle_test")
    requires_stage = "cell_signal_audit"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition("start", "Start (inclusive)", "datetime", required=True, group="Range"),
        ParameterDefinition("end", "End (exclusive)", "datetime", required=True, group="Range"),
        ParameterDefinition(
            "aggregation_minutes",
            "Aggregation interval",
            "choice",
            "5",
            choices=("1", "5", "15"),
            group="Resolution",
            description="Five minutes is recommended for full-history fleet analysis.",
        ),
        ParameterDefinition(
            "compression",
            "Parquet compression",
            "choice",
            "zstd",
            choices=("snappy", "zstd", "gzip", "none"),
            group="Storage",
        ),
        ParameterDefinition(
            "log_level",
            "Log level",
            "choice",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            group="Execution",
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        if _timestamp(normalized["start"], "start") >= _timestamp(normalized["end"], "end"):
            raise ValueError("start must be earlier than end")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        return (
            str(settings.source_python),
            "-m",
            "src.cell_health_features",
            "--serial",
            str(values["serial"]),
            "--start",
            str(values["start"]),
            "--end",
            str(values["end"]),
            "--aggregation-minutes",
            str(values["aggregation_minutes"]),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        )


class CellVoltageAnomalyPlugin(PipelinePlugin):
    key = "veera.cell_voltage_anomaly.v1"
    name = "Analyze cell-voltage anomalies"
    description = (
        "Learn operating-condition baselines from compact NCA cell-health features, "
        "then report sustained spread anomalies and module-configuration shifts."
    )
    output_family = "nca_cell_voltage_anomaly"
    workflow_stage = "analysis"
    device_roles = ("field_tester", "full_cycle_test")
    requires_stage = "cell_health"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "baseline_days", "Baseline window (days)", "integer", 90, group="Baseline"
        ),
        ParameterDefinition(
            "minimum_baseline_group_rows",
            "Minimum comparable bins",
            "integer",
            100,
            group="Baseline",
        ),
        ParameterDefinition(
            "baseline_quantile", "Baseline high quantile", "number", 0.995, group="Baseline"
        ),
        ParameterDefinition(
            "robust_z_threshold", "Robust z threshold", "number", 6.0, group="Detection"
        ),
        ParameterDefinition(
            "minimum_excess_mv", "Minimum excess", "number", 5.0, group="Detection"
        ),
        ParameterDefinition(
            "scale_floor_mv", "Noise scale floor", "number", 1.0, group="Detection"
        ),
        ParameterDefinition(
            "absolute_threshold_floor_mv",
            "Absolute spread floor",
            "number",
            25.0,
            group="Detection",
        ),
        ParameterDefinition(
            "absolute_critical_spread_mv",
            "Absolute critical spread",
            "number",
            100.0,
            group="Detection",
        ),
        ParameterDefinition(
            "module_epoch_shift_threshold_mv",
            "Module-epoch shift threshold",
            "number",
            20.0,
            group="Detection",
        ),
        ParameterDefinition(
            "minimum_episode_bins", "Minimum episode bins", "integer", 3, group="Episodes"
        ),
        ParameterDefinition("episode_gap_minutes", "Episode gap", "integer", 15, group="Episodes"),
        ParameterDefinition(
            "rest_current_threshold_a", "Rest-current threshold", "number", 0.5, group="Conditions"
        ),
        ParameterDefinition(
            "minimum_sample_coverage", "Minimum sample coverage", "number", 0.5, group="Quality"
        ),
        ParameterDefinition(
            "minimum_cell_voltage_coverage",
            "Minimum cell-voltage coverage",
            "number",
            0.95,
            group="Quality",
        ),
        ParameterDefinition(
            "compression",
            "Parquet compression",
            "choice",
            "zstd",
            choices=("snappy", "zstd", "gzip", "none"),
            group="Storage",
        ),
        ParameterDefinition(
            "log_level",
            "Log level",
            "choice",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR"),
            group="Execution",
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        positive = (
            "baseline_days",
            "minimum_baseline_group_rows",
            "robust_z_threshold",
            "scale_floor_mv",
            "minimum_episode_bins",
            "episode_gap_minutes",
        )
        if any(float(normalized[key]) <= 0 for key in positive):
            raise ValueError("baseline, scale, and episode parameters must be positive")
        nonnegative = (
            "minimum_excess_mv",
            "absolute_threshold_floor_mv",
            "absolute_critical_spread_mv",
            "module_epoch_shift_threshold_mv",
            "rest_current_threshold_a",
        )
        if any(float(normalized[key]) < 0 for key in nonnegative):
            raise ValueError("detection thresholds must be non-negative")
        if float(normalized["absolute_critical_spread_mv"]) <= float(
            normalized["absolute_threshold_floor_mv"]
        ):
            raise ValueError("absolute critical spread must exceed the absolute floor")
        if not 0.5 < float(normalized["baseline_quantile"]) < 1:
            raise ValueError("baseline_quantile must be between 0.5 and 1")
        for key in ("minimum_sample_coverage", "minimum_cell_voltage_coverage"):
            if not 0 < float(normalized[key]) <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = ["--serial", str(values["serial"])]
        flags = {
            "baseline_days": "--baseline-days",
            "minimum_baseline_group_rows": "--minimum-baseline-group-rows",
            "baseline_quantile": "--baseline-quantile",
            "robust_z_threshold": "--robust-z-threshold",
            "minimum_excess_mv": "--minimum-excess-mv",
            "scale_floor_mv": "--scale-floor-mv",
            "absolute_threshold_floor_mv": "--absolute-threshold-floor-mv",
            "absolute_critical_spread_mv": "--absolute-critical-spread-mv",
            "module_epoch_shift_threshold_mv": "--module-epoch-shift-threshold-mv",
            "minimum_episode_bins": "--minimum-episode-bins",
            "episode_gap_minutes": "--episode-gap-minutes",
            "rest_current_threshold_a": "--rest-current-threshold-a",
            "minimum_sample_coverage": "--minimum-sample-coverage",
            "minimum_cell_voltage_coverage": "--minimum-cell-voltage-coverage",
            "compression": "--compression",
            "log_level": "--log-level",
        }
        for key, flag in flags.items():
            args.extend((flag, str(values[key])))
        return (
            str(settings.source_python),
            "-m",
            "src.cell_voltage_anomaly",
            *args,
        )
