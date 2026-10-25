from __future__ import annotations

from typing import Any

from battryml.battery_profiles import battery_profile, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


class DischargeCapacityEvidencePlugin(PipelinePlugin):
    key = "veera.discharge_capacity_evidence.v1"
    name = "Build discharge capacity evidence"
    description = (
        "Find 99-1 and 95-5 discharges, integrate bounded one-second current, and "
        "attach temperature, voltage-spread, continuity, and weakest-channel evidence."
    )
    output_family = "discharge_derived_usable_capacity_evidence"
    workflow_stage = "discharge_capacity_evidence"
    device_roles = ("field_tester", "full_cycle_test")
    chemistries = ("NCA", "LFP")
    requires_stage = "hierarchical_features"
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
        if float(normalized["nominal_capacity_ah"]) <= 0:
            raise ValueError("nominal_capacity_ah must be positive")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        profile = battery_profile(str(values["serial"]))
        if profile is None:
            raise ValueError(f"Battery {values['serial']} has no configured chemistry")
        return (
            str(settings.source_python),
            "-m",
            "src.discharge_capacity_evidence",
            "--serial",
            str(values["serial"]),
            "--chemistry",
            profile.chemistry.upper(),
            "--module-count",
            str(values["module_count"]),
            "--nominal-capacity-ah",
            str(values["nominal_capacity_ah"]),
            "--raw-root",
            str(settings.source_root / "data" / "raw_parquet"),
            "--feature-root",
            str(settings.source_root / "data" / "processed" / "bms_hierarchical_features"),
            "--output-root",
            str(settings.source_root / "data" / "processed" / "discharge_capacity_evidence"),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        )
