from __future__ import annotations

from typing import Any

from battryml.battery_profiles import loss_slope_defaults, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


def _parse_slopes(value: Any) -> tuple[float, ...]:
    try:
        slopes = tuple(
            float(item.strip()) for item in str(value).split(",") if item.strip()
        )
    except ValueError as exc:
        raise ValueError("Loss slopes must be comma-separated numbers") from exc
    if not slopes or any(slope <= 0 for slope in slopes):
        raise ValueError("Provide at least one positive loss slope")
    if len(set(slopes)) != len(slopes):
        raise ValueError("Loss slopes must be unique")
    return slopes


class FixedSlopeVerificationPlugin(PipelinePlugin):
    key = "veera.fixed_slope_verification.v1"
    name = "Compare transferred loss slopes"
    description = "Compare fixed Ah-loss slopes with BMS SOH using actual cumulative discharge."
    output_family = "fixed_slope_transfer_verification"
    workflow_stage = "fixed_slope_verification"
    device_roles = ("field_tester",)
    chemistries = ("NCA",)
    requires_stage = "events"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "nominal_capacity_ah",
            "Nominal capacity",
            "number",
            required=True,
            group="Battery",
            serial_defaults=nominal_capacity_defaults(),
        ),
        ParameterDefinition(
            "module_count",
            "Installed modules",
            "integer",
            group="Battery",
        ),
        ParameterDefinition(
            "loss_slopes_per_1000ah",
            "Loss slopes (Ah / 1000 Ah)",
            "text",
            "0.072556426063",
            required=True,
            group="Transferred model",
            serial_defaults=loss_slope_defaults(),
        ),
        ParameterDefinition(
            "slope_reference_module_count",
            "Slope reference modules",
            "integer",
            2,
            group="Transferred model",
        ),
        ParameterDefinition(
            "anchor_mode",
            "SOH anchor",
            "choice",
            "absolute",
            choices=("absolute", "first_bms"),
            group="Transferred model",
        ),
        ParameterDefinition(
            "reference_cumulative_ah",
            "Reference cumulative Ah",
            "number",
            0.0,
            group="Transferred model",
        ),
        ParameterDefinition(
            "bms_tolerance_pct",
            "BMS tolerance",
            "number",
            1.0,
            group="Verification",
        ),
        ParameterDefinition(
            "aggregation",
            "Comparison interval",
            "choice",
            "daily",
            choices=("daily", "weekly"),
            group="Verification",
        ),
        ParameterDefinition("start", "Start (optional)", "datetime", group="Range"),
        ParameterDefinition("end", "End (optional)", "datetime", group="Range"),
        ParameterDefinition("event_run_id", "Event run override", "text", group="Source"),
        ParameterDefinition(
            "compression",
            "Parquet compression",
            "choice",
            "snappy",
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
        _parse_slopes(normalized["loss_slopes_per_1000ah"])
        if float(normalized["nominal_capacity_ah"]) <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        for key in ("module_count", "slope_reference_module_count"):
            if normalized.get(key) is not None and int(normalized[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if float(normalized["bms_tolerance_pct"]) < 0:
            raise ValueError("bms_tolerance_pct must be nonnegative")
        if float(normalized.get("reference_cumulative_ah", 0.0)) < 0:
            raise ValueError("reference_cumulative_ah must be nonnegative")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = [
            "--serial",
            str(values["serial"]),
            "--nominal-capacity-ah",
            str(values["nominal_capacity_ah"]),
            "--loss-slopes-per-1000ah",
            str(values["loss_slopes_per_1000ah"]),
            "--anchor-mode",
            str(values["anchor_mode"]),
            "--reference-cumulative-ah",
            str(values["reference_cumulative_ah"]),
            "--bms-tolerance-pct",
            str(values["bms_tolerance_pct"]),
            "--aggregation",
            str(values["aggregation"]),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        ]
        for key, flag in {
            "start": "--start",
            "end": "--end",
            "event_run_id": "--event-run-id",
            "module_count": "--module-count",
            "slope_reference_module_count": "--slope-reference-module-count",
        }.items():
            value = values.get(key)
            if value is not None and str(value).strip():
                args.extend((flag, str(value)))
        return (
            str(settings.source_python),
            "-m",
            "src.fixed_slope_verification",
            *args,
        )
