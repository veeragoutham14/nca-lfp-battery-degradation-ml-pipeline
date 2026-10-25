from __future__ import annotations

import sys
from typing import Any

from battryml.battery_profiles import battery_profile, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


class _DeviceWorkflowPlugin(PipelinePlugin):
    device_roles = ("field_tester", "full_cycle_test")
    chemistries = ("NCA", "LFP")
    produces_analysis_run = False
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
        ParameterDefinition(
            "nominal_capacity_ah",
            "Nominal capacity",
            "number",
            required=True,
            group="Battery",
            serial_defaults=nominal_capacity_defaults(),
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        if float(normalized["nominal_capacity_ah"]) <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        return normalized

    def workflow_command(self, values: dict[str, Any], mode: str) -> list[str]:
        profile = battery_profile(str(values["serial"]))
        if profile is None:
            raise ValueError(f"Battery {values['serial']} has no configured chemistry")
        return [
            sys.executable,
            "-m",
            "battryml.workflows.device_analysis",
            mode,
            "--serial",
            str(values["serial"]),
            "--chemistry",
            profile.chemistry.upper(),
            "--module-count",
            str(values["module_count"]),
            "--nominal-capacity-ah",
            str(values["nominal_capacity_ah"]),
        ]


class PrepareAnalysisDataPlugin(_DeviceWorkflowPlugin):
    key = "battryml.prepare_analysis_data.v1"
    name = "Prepare or update analysis data"
    description = (
        "Run the SOH audit, hierarchical source audit, and incremental five-minute "
        "feature build as one recoverable workflow."
    )
    output_family = "device_analysis_preparation"
    scan_output_families = ("battery_data_validation",)
    workflow_stage = "extraction"
    requires_database = True
    parameters = _DeviceWorkflowPlugin.parameters + (
        ParameterDefinition(
            "update_raw",
            "Import newly available core telemetry first",
            "boolean",
            False,
            group="Update",
        ),
        ParameterDefinition("start", "Raw import start (optional)", "datetime", group="Update"),
        ParameterDefinition("end", "Raw import end (optional)", "datetime", group="Update"),
    )

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        command = self.workflow_command(values, "prepare")
        if bool(values.get("update_raw")):
            command.append("--update-raw")
        for key, flag in (("start", "--start"), ("end", "--end")):
            if values.get(key):
                command.extend((flag, str(values[key])))
        return tuple(command)


class CapacitySohInvestigationPlugin(_DeviceWorkflowPlugin):
    key = "battryml.capacity_soh_investigation.v1"
    name = "Run capacity and BMS validation"
    description = (
        "Build rest-aware physical discharges with at least 80 percentage points of "
        "SOC drop and associate them with persistent downward BMS SOH transitions."
    )
    output_family = "bms_soh_recalibration_evidence"
    scan_output_families = (
        "discharge_derived_usable_capacity_evidence",
        "bms_soh_recalibration_evidence",
    )
    workflow_stage = "capacity_soh_investigation"
    requires_stages = ("data_validation", "events")

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        return tuple(self.workflow_command(values, "investigate"))
