from __future__ import annotations

import sys
from typing import Any

from battryml.battery_profiles import battery_profile, nominal_capacity_defaults
from battryml.config import Settings
from battryml.data.storage_paths import resolve_dataset_location
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


class SohSignalAuditPlugin(PipelinePlugin):
    key = "battryml.soh_signal_audit.v1"
    name = "Audit BMS SOH signal"
    description = (
        "Detect BMS SOH steps, recoveries, telemetry gaps, and weak-label risk before "
        "training a health model."
    )
    output_family = "bms_soh_signal_audit"
    workflow_stage = "soh_signal_audit"
    device_roles = ("field_tester", "full_cycle_test")
    requires_stage = "raw"
    parameters = (
        ParameterDefinition(
            "serial", "Battery serial", "text", required=True, group="Battery"
        ),
        ParameterDefinition(
            "nominal_capacity_ah",
            "Nominal capacity",
            "number",
            group="Battery",
            serial_defaults=nominal_capacity_defaults(),
        ),
        ParameterDefinition("start", "Start (optional)", "datetime", group="Range"),
        ParameterDefinition("end", "End (optional)", "datetime", group="Range"),
        ParameterDefinition(
            "step_threshold_pct",
            "SOH step threshold",
            "number",
            0.5,
            group="Detection",
            description="Minimum day-to-day BMS SOH movement in percentage points.",
        ),
        ParameterDefinition(
            "unstable_daily_range_pct",
            "Unstable daily range",
            "number",
            5.0,
            group="Detection",
        ),
        ParameterDefinition(
            "min_label_coverage",
            "Minimum label coverage",
            "number",
            0.8,
            group="Quality",
        ),
        ParameterDefinition(
            "gap_threshold_seconds",
            "Telemetry gap threshold",
            "number",
            60.0,
            group="Quality",
        ),
        ParameterDefinition(
            "max_integration_gap_seconds",
            "Maximum integration interval",
            "number",
            5.0,
            group="Quality",
            description="Caps Ah integration across missing telemetry.",
        ),
        ParameterDefinition(
            "rest_current_threshold_a",
            "Rest-current threshold",
            "number",
            0.5,
            group="Operating state",
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
        nominal = normalized.get("nominal_capacity_ah")
        if nominal is not None and float(nominal) <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        for key in (
            "step_threshold_pct",
            "unstable_daily_range_pct",
            "gap_threshold_seconds",
            "max_integration_gap_seconds",
        ):
            if float(normalized[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if float(normalized["rest_current_threshold_a"]) < 0:
            raise ValueError("rest_current_threshold_a must be nonnegative")
        coverage = float(normalized["min_label_coverage"])
        if not 0 <= coverage <= 1:
            raise ValueError("min_label_coverage must be between 0 and 1")
        return normalized

    def build_command(
        self, settings: Settings, values: dict[str, Any]
    ) -> tuple[str, ...]:
        values = self.validate(values)
        profile = battery_profile(str(values["serial"]))
        if profile is None:
            raise ValueError(f"Battery {values['serial']} has no configured chemistry")
        raw_location = resolve_dataset_location(
            settings.source_root / "data" / "raw_parquet",
            serial=str(values["serial"]),
            chemistry=profile.chemistry,
        )
        args = [
            "--serial",
            str(values["serial"]),
            "--raw-root",
            str(raw_location.parent_root),
            "--output-root",
            str(settings.source_root / "data" / "processed" / "fleet_soh_audit"),
            "--step-threshold-pct",
            str(values["step_threshold_pct"]),
            "--unstable-daily-range-pct",
            str(values["unstable_daily_range_pct"]),
            "--min-label-coverage",
            str(values["min_label_coverage"]),
            "--gap-threshold-seconds",
            str(values["gap_threshold_seconds"]),
            "--max-integration-gap-seconds",
            str(values["max_integration_gap_seconds"]),
            "--rest-current-threshold-a",
            str(values["rest_current_threshold_a"]),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        ]
        for key, flag in {
            "nominal_capacity_ah": "--nominal-capacity-ah",
            "start": "--start",
            "end": "--end",
        }.items():
            value = values.get(key)
            if value is not None and str(value).strip():
                args.extend((flag, str(value)))
        return (
            sys.executable,
            "-m",
            "battryml.analytics.soh_signal_audit",
            *args,
        )
