from __future__ import annotations

from typing import Any

from battryml.battery_profiles import battery_profile, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin
from battryml.plugins.seasonal_charge import NCA_REFERENCE_CAPACITY_DEFAULT

REFERENCE_SERIAL, REFERENCE_NAMEPLATE_AH = NCA_REFERENCE_CAPACITY_DEFAULT


class SohGapDiagnosisPlugin(PipelinePlugin):
    """Run the Phase 4 single-device SOH-gap diagnosis behind the gap analytics endpoint."""

    key = "battryml.soh_gap_diagnosis.v1"
    name = "Diagnose the BMS SOH gap against the reference"
    description = (
        "Normalise field capacity observations against the measured reference curve, "
        "classify each BMS SOH transition, and summarise stress between transitions."
    )
    output_family = "soh_gap_diagnosis"
    # Phase 4 writes device_summary.json and is read directly by the gap endpoint.
    produces_analysis_run = False
    workflow_stage = "soh_gap_diagnosis"
    device_roles = ("field_tester",)
    chemistries = ("NCA",)
    requires_stage = "seasonal_charge_profile"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "module_count", "Installed modules", "integer", required=True, group="Battery"
        ),
        ParameterDefinition(
            "field_nameplate_capacity_ah",
            "Field nameplate capacity",
            "number",
            required=True,
            group="Battery",
            serial_defaults=nominal_capacity_defaults(),
        ),
        ParameterDefinition(
            "reference_nameplate_capacity_ah",
            "Reference nameplate capacity",
            "number",
            REFERENCE_NAMEPLATE_AH,
            group="Reference",
            description=f"Nameplate of reference battery {REFERENCE_SERIAL}.",
        ),
        ParameterDefinition(
            "minimum_reference_strict_cycles",
            "Minimum strict reference cycles",
            "integer",
            500,
            group="Reference",
            description="Reference gate before the capacity curve is fitted.",
        ),
    )

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        normalized = super().validate(values)
        if not 1 <= int(normalized["module_count"]) <= 4:
            raise ValueError("module_count must be between 1 and 4")
        for key in ("field_nameplate_capacity_ah", "reference_nameplate_capacity_ah"):
            if float(normalized[key]) <= 0:
                raise ValueError(f"{key} must be positive")
        if int(normalized["minimum_reference_strict_cycles"]) < 2:
            raise ValueError("minimum_reference_strict_cycles must be at least 2")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        profile = battery_profile(str(values["serial"]))
        assert profile is not None
        processed = settings.source_root / "data" / "processed"
        return (
            str(settings.source_python),
            "-m",
            "src.soh_gap_diagnosis",
            "--serial",
            str(values["serial"]),
            "--chemistry",
            profile.chemistry.upper(),
            "--module-count",
            str(values["module_count"]),
            "--field-nameplate-capacity-ah",
            str(values["field_nameplate_capacity_ah"]),
            "--reference-nameplate-capacity-ah",
            str(values["reference_nameplate_capacity_ah"]),
            "--minimum-reference-strict-cycles",
            str(values["minimum_reference_strict_cycles"]),
            "--processed-root",
            str(processed),
            "--output-root",
            str(processed / "soh_gap_diagnosis"),
        )
