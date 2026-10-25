from __future__ import annotations

from typing import Any

from battryml.battery_profiles import battery_profile, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin

NCA_REFERENCE_CAPACITY_DEFAULT = ("100000017", 56.0)


class SeasonalChargeAnalysisPlugin(PipelinePlugin):
    key = "veera.seasonal_charge_analysis.v1"
    name = "Analyze seasonal use, current stress, and SOH events"
    description = (
        "Combine discharge-first BMS recalibration evidence with monthly current, SOC, "
        "temperature, and cell-imbalance exposure history."
    )
    output_family = "seasonal_charging_stress_profile"
    workflow_stage = "seasonal_charge_profile"
    device_roles = ("field_tester", "full_cycle_test")
    requires_stage = "capacity_soh_investigation"
    requires_stages = ("capacity_soh_investigation",)
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
            "high_current_c_rate_threshold",
            "High-current threshold (C-rate)",
            "number",
            0.5,
            group="Stress exposure",
        ),
        ParameterDefinition(
            "charge_episode_gap_minutes",
            "Maximum charging interruption",
            "integer",
            15,
            group="Charging episodes",
        ),
        ParameterDefinition(
            "minimum_charge_soc_gain_pct",
            "Minimum SOC gain for capacity proxy",
            "number",
            20.0,
            group="Charging episodes",
            description="The resulting partial-charge capacity remains a weak proxy.",
        ),
        ParameterDefinition(
            "transition_window_days",
            "SOH context window (days)",
            "integer",
            7,
            group="SOH transitions",
        ),
        ParameterDefinition(
            "minimum_daily_coverage",
            "Minimum daily coverage",
            "number",
            0.8,
            group="Quality",
        ),
        ParameterDefinition(
            "minimum_month_observed_days",
            "Minimum observed days per month",
            "integer",
            14,
            group="Quality",
        ),
        ParameterDefinition(
            "minimum_month_coverage",
            "Minimum monthly coverage",
            "number",
            0.8,
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
        profile = battery_profile(str(normalized["serial"]))
        if profile is None or profile.chemistry.upper() not in {"NCA", "LFP"}:
            raise ValueError("Battery chemistry must be configured as NCA or LFP")
        if float(normalized["nominal_capacity_ah"]) <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        if float(normalized["rest_current_threshold_a"]) < 0:
            raise ValueError("rest_current_threshold_a must be nonnegative")
        if float(normalized["high_current_c_rate_threshold"]) <= 0:
            raise ValueError("high_current_c_rate_threshold must be positive")
        for key in ("charge_episode_gap_minutes", "transition_window_days"):
            if int(normalized[key]) < 1:
                raise ValueError(f"{key} must be positive")
        if float(normalized["minimum_charge_soc_gain_pct"]) <= 0:
            raise ValueError("minimum_charge_soc_gain_pct must be positive")
        if int(normalized["minimum_month_observed_days"]) < 1:
            raise ValueError("minimum_month_observed_days must be positive")
        for key in ("minimum_daily_coverage", "minimum_month_coverage"):
            if not 0 <= float(normalized[key]) <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        profile = battery_profile(str(values["serial"]))
        assert profile is not None
        return (
            str(settings.source_python),
            "-m",
            "src.seasonal_charge_analysis",
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
            "--high-current-c-rate-threshold",
            str(values["high_current_c_rate_threshold"]),
            "--charge-episode-gap-minutes",
            str(values["charge_episode_gap_minutes"]),
            "--minimum-charge-soc-gain-pct",
            str(values["minimum_charge_soc_gain_pct"]),
            "--transition-window-days",
            str(values["transition_window_days"]),
            "--minimum-daily-coverage",
            str(values["minimum_daily_coverage"]),
            "--minimum-month-observed-days",
            str(values["minimum_month_observed_days"]),
            "--minimum-month-coverage",
            str(values["minimum_month_coverage"]),
            "--feature-root",
            str(settings.source_root / "data" / "processed" / "bms_hierarchical_features"),
            "--soh-audit-root",
            str(settings.source_root / "data" / "processed" / "fleet_soh_audit"),
            "--recalibration-root",
            str(settings.source_root / "data" / "processed" / "bms_soh_recalibration"),
            "--raw-root",
            str(settings.source_root / "data" / "raw_parquet"),
            "--output-root",
            str(settings.source_root / "data" / "processed" / "seasonal_charge_analysis"),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        )
