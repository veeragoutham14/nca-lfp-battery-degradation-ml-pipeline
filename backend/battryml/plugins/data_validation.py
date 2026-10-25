from __future__ import annotations

import sys
from typing import Any

from battryml.battery_profiles import battery_profile
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


class DataValidationPlugin(PipelinePlugin):
    key = "veera.data_validation.v1"
    name = "Validate analysis inputs"
    description = (
        "Qualify the BMS SOH signal and validate local telemetry, operating events, "
        "and built electrothermal features as one analysis-readiness gate."
    )
    output_family = "battery_data_validation"
    workflow_stage = "data_validation"
    requires_stages = ("hierarchical_features", "events")
    device_roles = ("field_tester", "full_cycle_test")
    chemistries = ("NCA", "LFP")
    requires_database = False
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        profile = battery_profile(str(values["serial"]))
        if profile is None:
            raise ValueError(f"Battery {values['serial']} has no configured chemistry")
        if profile.nominal_capacity_ah is None:
            raise ValueError(f"Battery {values['serial']} has no configured nominal capacity")
        return (
            sys.executable,
            "-m",
            "battryml.workflows.device_analysis",
            "validate",
            "--serial",
            str(values["serial"]),
            "--chemistry",
            profile.chemistry.upper(),
            "--module-count",
            str(values["module_count"]),
            "--nominal-capacity-ah",
            str(profile.nominal_capacity_ah),
        )


class ElectrothermalImportValidationPlugin(PipelinePlugin):
    key = "veera.electrothermal_import_validation.v1"
    name = "Validate electrothermal history"
    description = "Validate local five-minute temperature and voltage-group datasets."
    output_family = "electrothermal_import_validation"
    workflow_stage = "electrothermal_validation"
    device_roles = ("field_tester", "full_cycle_test")
    chemistries = ("NCA", "LFP")
    requires_database = False
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
        ParameterDefinition(
            "chemistry",
            "Chemistry",
            "choice",
            required=True,
            choices=("NCA", "LFP"),
            group="Battery",
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        profile = battery_profile(str(normalized["serial"]))
        if profile and profile.chemistry.upper() != str(normalized["chemistry"]).upper():
            raise ValueError("Selected chemistry conflicts with the configured battery profile")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        return (
            str(settings.source_python),
            "-m",
            "src.validate_electrothermal_import",
            "--serial",
            str(values["serial"]),
            "--chemistry",
            str(values["chemistry"]).upper(),
            "--module-count",
            str(values["module_count"]),
            "--data-root",
            str(settings.source_root / "data"),
        )
