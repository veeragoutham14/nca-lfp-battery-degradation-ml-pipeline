from __future__ import annotations

from typing import Any

from battryml.battery_profiles import battery_profile, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin

NCA_REFERENCE_CAPACITY_DEFAULT = ("100000017", 56.0)


class BmsSohRecalibrationPlugin(PipelinePlugin):
    key = "veera.bms_soh_recalibration.v4"
    name = "Investigate BMS SOH recalibration events"
    description = (
        "Resolve persistent BMS SOH transitions from one-second telemetry, test "
        "95-100% to 0-5% discharges first, then compare secondary operating evidence "
        "with matched non-event controls."
    )
    output_family = "bms_soh_recalibration_evidence"
    workflow_stage = "soh_recalibration"
    device_roles = ("field_tester", "full_cycle_test")
    chemistries = ("NCA", "LFP")
    requires_stage = "discharge_capacity_evidence"
    requires_stages = ("soh_signal_audit",)
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "nominal_capacity_ah",
            "Nominal capacity",
            "number",
            required=True,
            group="Battery",
            serial_defaults=(*nominal_capacity_defaults(), NCA_REFERENCE_CAPACITY_DEFAULT),
        ),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
        ParameterDefinition(
            "rest_current_threshold_a",
            "Rest-current threshold",
            "number",
            0.5,
            group="Operating state",
        ),
        ParameterDefinition(
            "context_lookback_hours",
            "Evidence lookback",
            "integer",
            72,
            group="Evidence window",
            description="Hours before the resolved BMS SOH transition to inspect.",
        ),
        ParameterDefinition(
            "discharge_start_soc_pct",
            "Calibration-discharge start SOC",
            "number",
            95.0,
            group="Primary discharge hypothesis",
            description="A qualifying 100-to-0 discharge must begin at or above this SOC.",
        ),
        ParameterDefinition(
            "discharge_end_soc_pct",
            "Calibration-discharge end SOC",
            "number",
            5.0,
            group="Primary discharge hypothesis",
            description="A qualifying 100-to-0 discharge must finish at or below this SOC.",
        ),
        ParameterDefinition(
            "minimum_discharge_soc_drop_pct",
            "Minimum discharge SOC drop",
            "number",
            90.0,
            group="Primary discharge hypothesis",
        ),
        ParameterDefinition(
            "minimum_discharge_dominance_fraction",
            "Minimum discharge dominance",
            "number",
            0.8,
            group="Primary discharge hypothesis",
            description="Minimum fraction of observed episode time with negative current.",
        ),
        ParameterDefinition(
            "discharge_episode_gap_minutes",
            "Allowed discharge interruption",
            "integer",
            30,
            group="Primary discharge hypothesis",
        ),
        ParameterDefinition(
            "capacity_validation_start_soc_pct",
            "Strict capacity start SOC",
            "number",
            99.0,
            group="One-second capacity validation",
        ),
        ParameterDefinition(
            "capacity_validation_end_soc_pct",
            "Strict capacity end SOC",
            "number",
            1.0,
            group="One-second capacity validation",
        ),
        ParameterDefinition(
            "capacity_validation_minimum_coverage_fraction",
            "Minimum raw coverage",
            "number",
            0.98,
            group="One-second capacity validation",
        ),
        ParameterDefinition(
            "capacity_validation_maximum_unobserved_ah",
            "Maximum estimated missing Ah",
            "number",
            0.25,
            group="One-second capacity validation",
        ),
        ParameterDefinition(
            "continuous_cycle_maximum_rest_minutes",
            "Continuous-cycle rest limit",
            "number",
            5.0,
            group="One-second capacity validation",
            description=(
                "Cycles with more internal rest are reported as interrupted."
            ),
        ),
        ParameterDefinition(
            "persistence_hours",
            "New-level persistence",
            "integer",
            6,
            group="Transition detection",
            description="Hours the new BMS SOH level must remain dominant.",
        ),
        ParameterDefinition(
            "minimum_new_level_fraction",
            "New-level support fraction",
            "number",
            0.95,
            group="Transition detection",
        ),
        ParameterDefinition(
            "previous_support_minutes",
            "Previous-level support",
            "integer",
            30,
            group="Transition detection",
        ),
        ParameterDefinition(
            "exact_gap_seconds",
            "Exact transition gap",
            "integer",
            60,
            group="Transition detection",
            description="Larger old-to-new sample gaps are reported as interval-censored.",
        ),
        ParameterDefinition(
            "near_full_soc_pct",
            "Near-full SOC",
            "number",
            99.0,
            group="Evidence window",
        ),
        ParameterDefinition(
            "low_soc_pct",
            "Low SOC",
            "number",
            10.0,
            group="Evidence window",
        ),
        ParameterDefinition(
            "minimum_rest_minutes",
            "Prolonged-rest threshold",
            "integer",
            120,
            group="Evidence window",
        ),
        ParameterDefinition(
            "data_gap_risk_minutes",
            "Data-gap risk threshold",
            "integer",
            60,
            group="Evidence window",
        ),
        ParameterDefinition(
            "recurrence_window_days",
            "Repeated-update window",
            "integer",
            30,
            group="Evidence window",
        ),
        ParameterDefinition(
            "minimum_window_coverage",
            "Minimum compact-window coverage",
            "number",
            0.8,
            group="Quality",
        ),
        ParameterDefinition(
            "minimum_context_coverage",
            "Minimum evidence coverage",
            "number",
            0.8,
            group="Quality",
        ),
        ParameterDefinition(
            "controls_per_event",
            "Matched controls per event",
            "integer",
            5,
            group="Matched controls",
        ),
        ParameterDefinition(
            "control_event_exclusion_days",
            "Control exclusion radius",
            "integer",
            3,
            group="Matched controls",
            description="Non-event anchors this many days around an SOH event are excluded.",
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
        profile = battery_profile(str(normalized["serial"]))
        if profile is None or profile.chemistry.upper() not in {"NCA", "LFP"}:
            raise ValueError("Battery chemistry must be configured as NCA or LFP")
        if float(normalized["nominal_capacity_ah"]) <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        if float(normalized["rest_current_threshold_a"]) < 0:
            raise ValueError("rest_current_threshold_a must be nonnegative")
        for key in (
            "context_lookback_hours",
            "minimum_rest_minutes",
            "data_gap_risk_minutes",
            "recurrence_window_days",
            "persistence_hours",
            "previous_support_minutes",
            "exact_gap_seconds",
            "controls_per_event",
            "control_event_exclusion_days",
            "discharge_episode_gap_minutes",
        ):
            if int(normalized[key]) < 1:
                raise ValueError(f"{key} must be positive")
        low_soc = float(normalized["low_soc_pct"])
        near_full_soc = float(normalized["near_full_soc_pct"])
        if not 0 <= low_soc < near_full_soc <= 100:
            raise ValueError("low_soc_pct and near_full_soc_pct must be ordered within 0 to 100")
        discharge_end = float(normalized["discharge_end_soc_pct"])
        discharge_start = float(normalized["discharge_start_soc_pct"])
        if not 0 <= discharge_end < discharge_start <= 100:
            raise ValueError("Discharge boundary SOC thresholds must be ordered")
        if not 0 < float(normalized["minimum_discharge_soc_drop_pct"]) <= 100:
            raise ValueError("minimum_discharge_soc_drop_pct must be in (0, 100]")
        capacity_end = float(normalized["capacity_validation_end_soc_pct"])
        capacity_start = float(normalized["capacity_validation_start_soc_pct"])
        if not 0 <= capacity_end < capacity_start <= 100:
            raise ValueError("Capacity-validation SOC thresholds must be ordered")
        if float(normalized["capacity_validation_maximum_unobserved_ah"]) < 0:
            raise ValueError(
                "capacity_validation_maximum_unobserved_ah must be nonnegative"
            )
        if float(normalized["continuous_cycle_maximum_rest_minutes"]) < 0:
            raise ValueError(
                "continuous_cycle_maximum_rest_minutes must be nonnegative"
            )
        dominance = float(normalized["minimum_discharge_dominance_fraction"])
        if not 0 < dominance <= 1:
            raise ValueError("minimum_discharge_dominance_fraction must be in (0, 1]")
        for key in (
            "minimum_window_coverage",
            "minimum_context_coverage",
            "capacity_validation_minimum_coverage_fraction",
        ):
            if not 0 < float(normalized[key]) <= 1:
                raise ValueError(f"{key} must be in (0, 1]")
        if not 0.5 <= float(normalized["minimum_new_level_fraction"]) <= 1:
            raise ValueError("minimum_new_level_fraction must be in [0.5, 1]")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        profile = battery_profile(str(values["serial"]))
        assert profile is not None
        return (
            str(settings.source_python),
            "-m",
            "src.bms_soh_recalibration_analysis",
            "--serial",
            str(values["serial"]),
            "--chemistry",
            profile.chemistry.upper(),
            "--module-count",
            str(values["module_count"]),
            "--nominal-capacity-ah",
            str(values["nominal_capacity_ah"]),
            "--rest-current-threshold-a",
            str(values["rest_current_threshold_a"]),
            "--context-lookback-hours",
            str(values["context_lookback_hours"]),
            "--discharge-start-soc-pct",
            str(values["discharge_start_soc_pct"]),
            "--discharge-end-soc-pct",
            str(values["discharge_end_soc_pct"]),
            "--minimum-discharge-soc-drop-pct",
            str(values["minimum_discharge_soc_drop_pct"]),
            "--minimum-discharge-dominance-fraction",
            str(values["minimum_discharge_dominance_fraction"]),
            "--discharge-episode-gap-minutes",
            str(values["discharge_episode_gap_minutes"]),
            "--capacity-validation-start-soc-pct",
            str(values["capacity_validation_start_soc_pct"]),
            "--capacity-validation-end-soc-pct",
            str(values["capacity_validation_end_soc_pct"]),
            "--capacity-validation-minimum-coverage-fraction",
            str(values["capacity_validation_minimum_coverage_fraction"]),
            "--capacity-validation-maximum-unobserved-ah",
            str(values["capacity_validation_maximum_unobserved_ah"]),
            "--continuous-cycle-maximum-rest-minutes",
            str(values["continuous_cycle_maximum_rest_minutes"]),
            "--near-full-soc-pct",
            str(values["near_full_soc_pct"]),
            "--low-soc-pct",
            str(values["low_soc_pct"]),
            "--minimum-rest-minutes",
            str(values["minimum_rest_minutes"]),
            "--data-gap-risk-minutes",
            str(values["data_gap_risk_minutes"]),
            "--recurrence-window-days",
            str(values["recurrence_window_days"]),
            "--persistence-hours",
            str(values["persistence_hours"]),
            "--minimum-new-level-fraction",
            str(values["minimum_new_level_fraction"]),
            "--previous-support-minutes",
            str(values["previous_support_minutes"]),
            "--exact-gap-seconds",
            str(values["exact_gap_seconds"]),
            "--controls-per-event",
            str(values["controls_per_event"]),
            "--control-event-exclusion-days",
            str(values["control_event_exclusion_days"]),
            "--minimum-window-coverage",
            str(values["minimum_window_coverage"]),
            "--minimum-context-coverage",
            str(values["minimum_context_coverage"]),
            "--feature-root",
            str(settings.source_root / "data" / "processed" / "bms_hierarchical_features"),
            "--raw-root",
            str(settings.source_root / "data" / "raw_parquet"),
            "--soh-audit-root",
            str(settings.source_root / "data" / "processed" / "fleet_soh_audit"),
            "--output-root",
            str(settings.source_root / "data" / "processed" / "bms_soh_recalibration"),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        )
