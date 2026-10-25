from __future__ import annotations

from typing import Any

from battryml.battery_profiles import nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin

REST_RATIO_EXCLUDE_FLAG = "--exclude-high-rest-ratio-capacity-rows"


def _value_args(values: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, flag in mapping.items():
        value = values.get(key)
        if value is not None and str(value).strip() != "":
            args.extend([flag, str(value)])
    return args


def _flag_args(values: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    return [flag for key, flag in mapping.items() if values.get(key) is True]


COMMON_PARAMETERS = (
    ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
    ParameterDefinition(
        "nominal_capacity_ah",
        "Nominal capacity",
        "number",
        group="Battery",
        serial_defaults=nominal_capacity_defaults(),
    ),
    ParameterDefinition("training_cutoff", "Training cutoff", "datetime", group="Validation"),
    ParameterDefinition("validation_end", "Validation end", "datetime", group="Validation"),
    ParameterDefinition(
        "future_days", "Forecast horizon (days)", "integer", 3650, group="Forecast"
    ),
    ParameterDefinition("step_days", "Forecast step (days)", "integer", 30, group="Forecast"),
    ParameterDefinition(
        "degradation_model",
        "Degradation model",
        "choice",
        "usage_linear",
        choices=("usage_linear", "usage_calendar_constrained"),
        group="Model",
    ),
    ParameterDefinition(
        "usage_rate_modes",
        "Usage-rate models",
        "text",
        "historical_mean,historical_median,recent_median",
        group="Forecast",
    ),
    ParameterDefinition(
        "recent_usage_days", "Recent usage window", "integer", 90, group="Forecast"
    ),
)


class FullCycleSohPlugin(PipelinePlugin):
    key = "veera.full_cycle_soh.v1"
    name = "Full-cycle SOH forecast"
    description = "Run the proven Veera_SOH full 100-to-0 capacity degradation model."
    output_family = "full_discharge_capacity"
    workflow_stage = "full_cycle_soh"
    device_roles = ("full_cycle_test",)
    requires_stage = "capacity_trend"
    parameters = COMMON_PARAMETERS + (
        ParameterDefinition("capacity_run_dir", "Capacity run override", "path", group="Source"),
        ParameterDefinition(
            "bms_soh_reliable_after",
            "BMS SOH reliable after",
            "datetime",
            group="Model",
        ),
        ParameterDefinition("min_capacity_ah", "Minimum capacity", "number", 30.0, group="Quality"),
        ParameterDefinition("max_capacity_ah", "Maximum capacity", "number", 80.0, group="Quality"),
        ParameterDefinition(
            "exclude_high_rest_ratio_capacity_rows",
            "Exclude high-rest measurements",
            "boolean",
            True,
            group="Quality",
        ),
        ParameterDefinition(
            "max_rest_to_discharge_ratio",
            "Maximum rest/discharge ratio",
            "number",
            1.0,
            group="Quality",
        ),
        ParameterDefinition(
            "exclude_statistical_outliers",
            "Exclude statistical outliers",
            "boolean",
            False,
            group="Quality",
        ),
        ParameterDefinition(
            "degradation_baseline_capacity_ah",
            "Degradation baseline Ah",
            "number",
            group="Anchor",
        ),
        ParameterDefinition(
            "degradation_baseline_method",
            "Baseline method",
            "choice",
            "max",
            choices=("first", "max", "p95"),
            group="Anchor",
        ),
        ParameterDefinition(
            "degradation_reference_cumulative_ah",
            "Reference cumulative Ah",
            "number",
            group="Anchor",
        ),
        ParameterDefinition(
            "degradation_force_zero_intercept",
            "Force zero-loss intercept",
            "boolean",
            False,
            group="Anchor",
        ),
        ParameterDefinition(
            "internal_holdout_fraction",
            "Internal holdout fraction",
            "number",
            0.0,
            group="Validation",
        ),
    )

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        nominal_was_explicit = values.get("nominal_capacity_ah") not in (None, "")
        values = self.validate(values)
        if not nominal_was_explicit:
            values.pop("nominal_capacity_ah", None)
        args = _value_args(
            values,
            {
                "serial": "--serial",
                "capacity_run_dir": "--capacity-run-dir",
                "nominal_capacity_ah": "--nominal-capacity-ah",
                "bms_soh_reliable_after": "--bms-soh-reliable-after",
                "min_capacity_ah": "--min-capacity-ah",
                "max_capacity_ah": "--max-capacity-ah",
                "max_rest_to_discharge_ratio": "--max-rest-to-discharge-ratio",
                "future_days": "--future-days",
                "step_days": "--step-days",
                "training_cutoff": "--training-cutoff",
                "validation_end": "--validation-end",
                "internal_holdout_fraction": "--internal-holdout-fraction",
                "degradation_model": "--degradation-model",
                "degradation_baseline_capacity_ah": "--degradation-baseline-capacity-ah",
                "degradation_baseline_method": "--degradation-baseline-method",
                "degradation_reference_cumulative_ah": "--degradation-reference-cumulative-ah",
                "usage_rate_modes": "--usage-rate-modes",
                "recent_usage_days": "--recent-usage-days",
            },
        )
        args.extend(
            _flag_args(
                values,
                {
                    "exclude_high_rest_ratio_capacity_rows": REST_RATIO_EXCLUDE_FLAG,
                    "exclude_statistical_outliers": "--exclude-statistical-outliers",
                    "degradation_force_zero_intercept": "--degradation-force-zero-intercept",
                },
            )
        )
        return (str(settings.source_python), "-m", "src.capacity_ml", *args)
