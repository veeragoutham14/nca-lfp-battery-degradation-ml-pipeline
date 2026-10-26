from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from battryml.battery_profiles import battery_profile
from battryml.config import Settings


@dataclass(frozen=True)
class ParameterDefinition:
    key: str
    label: str
    type: str
    default: Any = None
    required: bool = False
    choices: tuple[str, ...] = ()
    description: str = ""
    group: str = "Model"
    serial_defaults: tuple[tuple[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["choices"] = list(self.choices)
        value["serial_defaults"] = dict(self.serial_defaults)
        return value

    def default_for_serial(self, serial: str) -> Any:
        return dict(self.serial_defaults).get(serial, self.default)


class PipelinePlugin(ABC):
    key: str
    name: str
    description: str
    output_family: str
    parameters: tuple[ParameterDefinition, ...]
    produces_analysis_run: bool = True
    scan_output_families: tuple[str, ...] = ()
    workflow_stage: str = "analysis"
    device_roles: tuple[str, ...] = ()
    chemistries: tuple[str, ...] = ()
    requires_database: bool = False
    requires_stage: str | None = None
    requires_stages: tuple[str, ...] = ()

    def needs_database(self, values: dict[str, Any]) -> bool:
        """Return True when this run, with these values, reads the source database."""
        return self.requires_database

    def dependency_stages(self) -> tuple[str, ...]:
        stages = ([self.requires_stage] if self.requires_stage else []) + list(
            self.requires_stages
        )
        return tuple(dict.fromkeys(stage for stage in stages if stage))

    def definition(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "output_family": self.output_family,
            "workflow_stage": self.workflow_stage,
            "device_roles": list(self.device_roles),
            "chemistries": list(self.chemistries),
            "requires_database": self.requires_database,
            "requires_stage": self.requires_stage,
            "requires_stages": list(self.dependency_stages()),
            "scan_output_families": list(self.scan_output_families),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

    def validate(self, values: dict[str, Any]) -> dict[str, Any]:
        allowed = {parameter.key: parameter for parameter in self.parameters}
        unknown = sorted(set(values) - set(allowed))
        if unknown:
            raise ValueError(f"Unknown parameters for {self.key}: {', '.join(unknown)}")
        serial = str(values.get("serial", "")).strip()
        if serial and (self.device_roles or self.chemistries):
            profile = battery_profile(serial)
            if profile is None:
                raise ValueError(
                    f"Battery {serial} has no assigned device role. Add it to "
                    "config/battery_profiles.json before running this pipeline."
                )
            if self.device_roles and profile.device_role not in self.device_roles:
                allowed = ", ".join(self.device_roles)
                raise ValueError(
                    f"Pipeline {self.key} is available for {allowed}; battery {serial} "
                    f"is assigned as {profile.device_role}."
                )
            if self.chemistries and profile.chemistry.upper() not in self.chemistries:
                allowed = ", ".join(self.chemistries)
                raise ValueError(
                    f"Pipeline {self.key} is available for {allowed}; battery {serial} "
                    f"is assigned as {profile.chemistry}."
                )
        normalized: dict[str, Any] = {}
        for key, definition in allowed.items():
            supplied = values.get(key)
            missing = (
                key not in values
                or supplied is None
                or (isinstance(supplied, str) and not supplied.strip())
            )
            value = definition.default_for_serial(serial) if missing else supplied
            if definition.required and (value is None or str(value).strip() == ""):
                raise ValueError(f"Missing required parameter: {key}")
            if value is not None and definition.choices and str(value) not in definition.choices:
                raise ValueError(f"{key} must be one of: {', '.join(definition.choices)}")
            if value is not None:
                normalized[key] = value
        return normalized

    @abstractmethod
    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        raise NotImplementedError
