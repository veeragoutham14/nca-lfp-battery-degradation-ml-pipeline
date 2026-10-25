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


class ElectrothermalSourceAuditPlugin(PipelinePlugin):
    key = "veera.bms_electrothermal_source_audit.v1"
    name = "Qualify hierarchical BMS signals"
    description = (
        "Verify every installed module, pack, voltage channel, and temperature sensor "
        "against the database schema, coverage, physical ranges, and BMS extrema."
    )
    output_family = "bms_electrothermal_source_quality"
    workflow_stage = "extraction"
    device_roles = ("field_tester", "full_cycle_test")
    requires_database = True
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
        ParameterDefinition(
            "start", "Audit start UTC (optional)", "datetime", group="Sample window"
        ),
        ParameterDefinition("end", "Audit end UTC (optional)", "datetime", group="Sample window"),
        ParameterDefinition(
            "readiness_threshold", "Required valid coverage", "number", 0.95, group="Quality"
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
        start, end = normalized.get("start"), normalized.get("end")
        if bool(start) != bool(end):
            raise ValueError("start and end must be provided together")
        if start and _timestamp(start, "start") >= _timestamp(end, "end"):
            raise ValueError("start must be earlier than end")
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        if not 0 < float(normalized["readiness_threshold"]) <= 1:
            raise ValueError("readiness_threshold must be between 0 and 1")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = [
            "--serial",
            str(values["serial"]),
            "--module-count",
            str(values["module_count"]),
            "--readiness-threshold",
            str(values["readiness_threshold"]),
            "--column-catalog",
            str(settings.project_root / "config" / "signal_catalog.json"),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        ]
        if values.get("start"):
            args.extend(("--start", str(values["start"]), "--end", str(values["end"])))
        return (str(settings.source_python), "-m", "src.bms_electrothermal_source_audit", *args)


class HierarchicalElectrothermalFeaturesPlugin(PipelinePlugin):
    key = "veera.bms_hierarchical_features.v1"
    name = "Build hierarchical electrothermal history"
    description = (
        "Create month-partitioned operating, voltage-channel, temperature-sensor, "
        "and system features while preserving database signal identity."
    )
    output_family = "bms_hierarchical_electrothermal_features"
    workflow_stage = "hierarchical_features"
    device_roles = ("field_tester", "full_cycle_test")
    requires_database = False
    requires_stage = "electrothermal_validation"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
        ParameterDefinition(
            "chemistry", "Chemistry", "choice", required=True,
            choices=("NCA", "LFP"), group="Battery"
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
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        return (
            str(settings.source_python),
            "-m",
            "src.build_local_electrothermal_features",
            "--serial",
            str(values["serial"]),
            "--module-count",
            str(values["module_count"]),
            "--chemistry",
            str(values["chemistry"]).upper(),
            "--data-root",
            str(settings.source_root / "data"),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        )


class HierarchicalVoltageAnomalyPlugin(PipelinePlugin):
    key = "veera.hierarchical_voltage_anomaly.v2"
    name = "Analyze hierarchical voltage anomalies"
    description = (
        "Combine fixed engineering bands with pack-peer residuals, low-rank channel "
        "reconstruction, temporal EWMA, persistent episodes, and bounded raw evidence."
    )
    output_family = "hierarchical_cell_voltage_anomaly"
    workflow_stage = "analysis"
    device_roles = ("field_tester", "full_cycle_test")
    requires_stage = "hierarchical_features"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "baseline_days", "Baseline window (days)", "integer", 90, group="Baseline"
        ),
        ParameterDefinition(
            "low_rank_components", "Low-rank components", "integer", 3, group="Model"
        ),
        ParameterDefinition(
            "robust_z_threshold", "Contextual score threshold", "number", 6.0, group="Detection"
        ),
        ParameterDefinition("ewma_alpha", "EWMA responsiveness", "number", 0.1, group="Detection"),
        ParameterDefinition(
            "minimum_coverage", "Minimum feature coverage", "number", 0.95, group="Quality"
        ),
        ParameterDefinition(
            "rest_current_threshold_a", "Rest-current threshold", "number", 0.5, group="Conditions"
        ),
        ParameterDefinition(
            "minimum_episode_bins", "Minimum episode bins", "integer", 3, group="Episodes"
        ),
        ParameterDefinition("episode_gap_minutes", "Episode gap", "integer", 15, group="Episodes"),
        ParameterDefinition(
            "evidence_padding_minutes", "Raw evidence padding", "integer", 30, group="Evidence"
        ),
        ParameterDefinition(
            "maximum_evidence_episodes",
            "Maximum evidence episodes",
            "integer",
            50,
            group="Evidence",
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
        for key in (
            "baseline_days",
            "low_rank_components",
            "minimum_episode_bins",
            "episode_gap_minutes",
        ):
            if int(normalized[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < float(normalized["ewma_alpha"]) <= 1:
            raise ValueError("ewma_alpha must be between 0 and 1")
        if not 0 < float(normalized["minimum_coverage"]) <= 1:
            raise ValueError("minimum_coverage must be between 0 and 1")
        if int(normalized["maximum_evidence_episodes"]) < 0:
            raise ValueError("maximum_evidence_episodes must be non-negative")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = [
            str(settings.source_python),
            "-m",
            "src.hierarchical_voltage_anomaly",
            "--serial",
            str(values["serial"]),
        ]
        mapping = {
            "baseline_days": "--baseline-days",
            "low_rank_components": "--low-rank-components",
            "robust_z_threshold": "--robust-z-threshold",
            "ewma_alpha": "--ewma-alpha",
            "minimum_coverage": "--minimum-coverage",
            "rest_current_threshold_a": "--rest-current-threshold-a",
            "minimum_episode_bins": "--minimum-episode-bins",
            "episode_gap_minutes": "--episode-gap-minutes",
            "evidence_padding_minutes": "--evidence-padding-minutes",
            "maximum_evidence_episodes": "--maximum-evidence-episodes",
            "compression": "--compression",
            "log_level": "--log-level",
        }
        for key, flag in mapping.items():
            args.extend((flag, str(values[key])))
        return tuple(args)
