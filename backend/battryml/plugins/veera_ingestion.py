from __future__ import annotations

from datetime import datetime
from typing import Any

from battryml.battery_profiles import battery_profile
from battryml.config import Settings
from battryml.data.import_planning import custom_dataset_root, electrothermal_dataset_root
from battryml.data.signal_registry import (
    CAPACITY_CORE_PURPOSE,
    CUSTOM_GROUP_KEYS,
    CUSTOM_PURPOSE,
    ELECTROTHERMAL_HISTORY_PURPOSE,
    RESOLUTION_CHOICES,
    RESOLUTION_OFF,
    capacity_core_resolutions,
    electrothermal_history_resolutions,
)
from battryml.data.storage_paths import resolve_dataset_location
from battryml.plugins.base import ParameterDefinition, PipelinePlugin


def _timestamp(value: Any, field: str) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc


def _plugin_resolutions(selection: str) -> tuple[str, ...]:
    mapping = {"off": (), "1s": ("1s",), "5m": ("5m",), "both": ("1s", "5m")}
    return mapping[selection]


class RawTelemetryImportPlugin(PipelinePlugin):
    key = "veera.raw_telemetry_import.v1"
    name = "Import raw telemetry"
    description = (
        "Import chemistry-specific telemetry at selectable one-second and five-minute resolutions."
    )
    output_family = "raw_telemetry"
    produces_analysis_run = False
    workflow_stage = "extraction"
    requires_database = True
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "chemistry",
            "Chemistry",
            "choice",
            required=True,
            choices=("NCA", "LFP"),
            group="Battery",
        ),
        ParameterDefinition(
            "start",
            "Start (inclusive, optional)",
            "datetime",
            group="Range",
            description=(
                "Existing devices continue after the latest local row. New devices use the "
                "first timestamp available in the database."
            ),
        ),
        ParameterDefinition(
            "end",
            "End (exclusive, optional)",
            "datetime",
            group="Range",
            description="Leave blank to use the last timestamp available in the database.",
        ),
        *(
            ParameterDefinition(
                f"{field}_resolution",
                label,
                "choice",
                choices=("off", "1s", "5m", "both"),
                group="Signal resolution",
            )
            for field, label in (
                ("soc", "State of charge"),
                ("soh", "BMS state of health"),
                ("voltage", "BMS stack voltage"),
                ("temperature", "BMS temperature summaries"),
                ("current", "BMS stack current"),
                ("cycle_count", "BMS lifecycle cycle counter"),
            )
        ),
        ParameterDefinition(
            "chunk_size",
            "Rows per file chunk",
            "integer",
            100_000,
            group="Storage",
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
            "limit",
            "Maximum rows",
            "integer",
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
        start = _timestamp(normalized["start"], "start") if normalized.get("start") else None
        end = _timestamp(normalized["end"], "end") if normalized.get("end") else None
        if start is not None and end is not None and start >= end:
            raise ValueError("start must be earlier than end")

        chemistry = str(normalized["chemistry"]).upper()
        normalized["chemistry"] = chemistry
        profile = battery_profile(str(normalized["serial"]))
        if profile is not None and profile.chemistry.upper() != chemistry:
            raise ValueError(
                f"Battery {normalized['serial']} is configured as {profile.chemistry}, "
                f"not {chemistry}."
            )

        resolutions = [
            normalized.get(f"{field}_resolution")
            for field in ("soc", "soh", "voltage", "temperature", "current", "cycle_count")
        ]
        supplied_resolutions = [value for value in resolutions if value is not None]
        if supplied_resolutions and all(value == "off" for value in supplied_resolutions):
            raise ValueError("Select at least one signal at one resolution")

        for key in ("chunk_size", "limit"):
            value = normalized.get(key)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{key} must be positive")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        source_data = settings.source_root / "data"
        raw_location = resolve_dataset_location(
            source_data / "raw_parquet",
            serial=str(values["serial"]),
            chemistry=str(values["chemistry"]),
            for_write=True,
        )
        compact_location = resolve_dataset_location(
            source_data / "telemetry_5min",
            serial=str(values["serial"]),
            chemistry=str(values["chemistry"]),
            for_write=True,
        )
        manifest_location = resolve_dataset_location(
            source_data / "import_manifests",
            serial=str(values["serial"]),
            chemistry=str(values["chemistry"]),
            for_write=True,
        )
        args = [
            "--serial",
            str(values["serial"]),
            "--chemistry",
            str(values["chemistry"]),
            "--chunk-size",
            str(values["chunk_size"]),
            "--import-profile-catalog",
            str(settings.project_root / "config" / "telemetry_import_profiles.json"),
            "--output-dir",
            str(raw_location.parent_root),
            "--five-minute-output-dir",
            str(compact_location.parent_root),
            "--manifest-output-dir",
            str(manifest_location.parent_root),
            "--compression",
            str(values["compression"]),
            "--log-level",
            str(values["log_level"]),
        ]
        if values.get("start") is not None:
            args.extend(("--start", str(values["start"])))
        if values.get("end") is not None:
            args.extend(("--end", str(values["end"])))
        for field in ("soc", "soh", "voltage", "temperature", "current", "cycle_count"):
            resolution = values.get(f"{field}_resolution")
            if resolution is not None:
                args.extend(("--field-resolution", f"{field}={resolution}"))
        if values.get("limit") is not None:
            args.extend(("--limit", str(values["limit"])))
        return (
            str(settings.source_python),
            "-m",
            "src.extract_configurable_telemetry",
            *args,
        )


class RawTelemetryImportV2Plugin(PipelinePlugin):
    """Purpose-driven import: fixed capacity preset or controlled custom groups."""

    key = "veera.raw_telemetry_import.v2"
    name = "Import telemetry (purpose-based)"
    description = (
        "Import the fixed capacity-degradation core preset or selectable custom "
        "signal groups with per-signal resolutions and incremental watermarks."
    )
    output_family = "raw_telemetry"
    produces_analysis_run = False
    workflow_stage = "extraction"
    requires_database = True
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "chemistry",
            "Chemistry",
            "choice",
            required=True,
            choices=("NCA", "LFP"),
            group="Battery",
        ),
        ParameterDefinition(
            "purpose",
            "Import purpose",
            "choice",
            required=True,
            choices=(CAPACITY_CORE_PURPOSE, ELECTROTHERMAL_HISTORY_PURPOSE, CUSTOM_PURPOSE),
            group="Battery",
        ),
        ParameterDefinition(
            "module_count",
            "Installed modules",
            "integer",
            group="Battery",
            description=(
                "Required when importing temperature or cell-voltage groups; "
                "resolves pack channels per module."
            ),
        ),
        ParameterDefinition(
            "start",
            "Start override (optional)",
            "datetime",
            group="Range",
            description=(
                "Blank uses watermark continuation; new signals begin at the "
                "device's configured start date."
            ),
        ),
        ParameterDefinition(
            "end",
            "End (optional, exclusive)",
            "datetime",
            group="Range",
            description="Blank means up to the newest database data.",
        ),
        *(
            ParameterDefinition(
                f"{group}_resolution",
                label,
                "choice",
                choices=RESOLUTION_CHOICES,
                group="Custom signals",
            )
            for group, label in (
                ("soc", "SOC"),
                ("soh", "BMS SOH"),
                ("current", "Current"),
                ("voltage", "Stack voltage"),
                ("temperature", "Temperature channels"),
                ("cell_voltage", "Cell/voltage-group channels"),
                ("cycle_count", "Lifecycle cycle counter"),
            )
        ),
        ParameterDefinition(
            "chunk_size", "Rows per file chunk", "integer", 100_000, group="Storage"
        ),
        ParameterDefinition(
            "compression",
            "Parquet compression",
            "choice",
            "zstd",
            choices=("snappy", "zstd", "gzip", "none"),
            group="Storage",
        ),
        ParameterDefinition("limit", "Maximum rows", "integer", group="Storage"),
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
        start = normalized.get("start")
        end = normalized.get("end")
        if start is not None and end is not None and start >= end:
            raise ValueError("start must be earlier than end")

        chemistry = str(normalized["chemistry"]).upper()
        normalized["chemistry"] = chemistry
        profile = battery_profile(str(normalized["serial"]))
        if profile is not None and profile.chemistry.upper() != chemistry:
            raise ValueError(
                f"Battery {normalized['serial']} is configured as {profile.chemistry}, "
                f"not {chemistry}."
            )

        purpose = str(normalized["purpose"])
        active = {
            field: normalized.get(f"{field}_resolution")
            for field in CUSTOM_GROUP_KEYS
            if normalized.get(f"{field}_resolution") is not None
            and normalized[f"{field}_resolution"] != RESOLUTION_OFF
        }
        if purpose in {CAPACITY_CORE_PURPOSE, ELECTROTHERMAL_HISTORY_PURPOSE} and active:
            raise ValueError(
                "The selected import preset is fixed; custom signal "
                "selections are not allowed for this purpose."
            )
        if purpose == CUSTOM_PURPOSE and not active:
            raise ValueError("Select at least one telemetry group")
        topology_groups_selected = bool(
            purpose in {CUSTOM_PURPOSE, ELECTROTHERMAL_HISTORY_PURPOSE}
            and {"temperature", "cell_voltage"} & set(active)
        )
        if purpose == ELECTROTHERMAL_HISTORY_PURPOSE:
            topology_groups_selected = True
        module_count_value = normalized.get("module_count")
        if topology_groups_selected:
            if module_count_value is None:
                raise ValueError(
                    "module_count is required when importing temperature or "
                    "cell-voltage channels"
                )
            if not 1 <= int(module_count_value) <= 4:
                raise ValueError("module_count must be between 1 and 4")
        elif module_count_value is not None and not (
            1 <= int(module_count_value) <= 4
        ):
            raise ValueError("module_count must be between 1 and 4")
        for key in ("chunk_size", "limit"):
            value = normalized.get(key)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{key} must be positive")
        return normalized

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        source_data = settings.source_root / "data"
        chemistry = str(values["chemistry"])
        purpose = str(values["purpose"])
        profile = battery_profile(str(values["serial"]))

        args = [
            "--serial",
            str(values["serial"]),
            "--chemistry",
            chemistry,
            "--chunk-size",
            str(values.get("chunk_size") or 100_000),
            "--import-profile-catalog",
            str(settings.project_root / "config" / "telemetry_import_profiles.json"),
            "--compression",
            str(values.get("compression") or "zstd"),
            "--log-level",
            str(values.get("log_level") or "INFO"),
            "--purpose",
            purpose,
        ]

        def _manifest_dir() -> str:
            location = resolve_dataset_location(
                source_data / "import_manifests",
                serial=str(values["serial"]),
                chemistry=chemistry,
                for_write=True,
            )
            return str(location.parent_root)

        if purpose == CAPACITY_CORE_PURPOSE:
            raw_location = resolve_dataset_location(
                source_data / "raw_parquet",
                serial=str(values["serial"]),
                chemistry=chemistry,
                for_write=True,
            )
            compact_location = resolve_dataset_location(
                source_data / "telemetry_5min",
                serial=str(values["serial"]),
                chemistry=chemistry,
                for_write=True,
            )
            args.extend(
                [
                    "--output-dir",
                    str(raw_location.parent_root),
                    "--five-minute-output-dir",
                    str(compact_location.parent_root),
                    "--manifest-output-dir",
                    _manifest_dir(),
                ]
            )
            for field, resolution in capacity_core_resolutions().items():
                args.extend(("--field-resolution", f"{field}={resolution}"))
        else:
            args.extend(("--manifest-output-dir", _manifest_dir()))
            module_count = values.get("module_count")
            if module_count is not None:
                args.extend(("--module-count", str(module_count)))
            fixed_electrothermal = (
                electrothermal_history_resolutions()
                if purpose == ELECTROTHERMAL_HISTORY_PURPOSE
                else {}
            )
            for field in CUSTOM_GROUP_KEYS:
                resolution = fixed_electrothermal.get(field) or values.get(f"{field}_resolution")
                if resolution is None:
                    continue
                args.extend(("--field-resolution", f"{field}={resolution}"))
                for res in _plugin_resolutions(resolution):
                    group_root = (
                        electrothermal_dataset_root(source_data, field, chemistry)
                        if purpose == ELECTROTHERMAL_HISTORY_PURPOSE
                        else custom_dataset_root(source_data, field, res, chemistry)
                    )
                    args.extend((
                        "--group-output-root",
                        f"{field}={res}={group_root}",
                    ))

        effective_start = values.get("start")
        if effective_start is None and profile is not None and profile.data_available_from:
            effective_start = f"{profile.data_available_from}T00:00:00+00:00"
        if effective_start is not None:
            args.extend(("--start", str(effective_start)))
        if values.get("end") is not None:
            args.extend(("--end", str(values["end"])))
        if values.get("limit") is not None:
            args.extend(("--limit", str(values["limit"])))
        return (
            str(settings.source_python),
            "-m",
            "src.extract_configurable_telemetry",
            *args,
        )
