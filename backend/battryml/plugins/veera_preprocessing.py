from __future__ import annotations

from typing import Any

from battryml.battery_profiles import battery_profile
from battryml.config import Settings
from battryml.data.storage_paths import resolve_dataset_location
from battryml.plugins.base import ParameterDefinition, PipelinePlugin

ALL_DEVICE_ROLES = ("field_tester", "full_cycle_test")


def _optional_args(values: dict[str, Any], mapping: dict[str, str]) -> list[str]:
    args: list[str] = []
    for key, flag in mapping.items():
        value = values.get(key)
        if value is not None and str(value).strip():
            args.extend((flag, str(value)))
    return args


class ClassifyRowsPlugin(PipelinePlugin):
    key = "veera.classify_rows.v1"
    name = "Classify telemetry rows"
    description = "Classify imported telemetry as charging, discharging, or rest."
    output_family = "classified_telemetry"
    produces_analysis_run = False
    workflow_stage = "classified"
    device_roles = ALL_DEVICE_ROLES
    requires_stage = "raw"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition("start", "Start (optional)", "datetime", group="Range"),
        ParameterDefinition("end", "End (optional)", "datetime", group="Range"),
        ParameterDefinition(
            "rest_current_threshold_a",
            "Rest current threshold",
            "number",
            0.5,
            group="Classification",
        ),
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

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
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
            "--input",
            str(raw_location.serial_root),
        ]
        args.extend(
            _optional_args(
                values,
                {
                    "start": "--start",
                    "end": "--end",
                    "rest_current_threshold_a": "--rest-current-threshold-a",
                    "compression": "--compression",
                    "log_level": "--log-level",
                },
            )
        )
        return (str(settings.source_python), "-m", "src.classify_rows", *args)


class BuildEventsPlugin(PipelinePlugin):
    key = "veera.build_events.v1"
    name = "Build operating events"
    description = "Merge classified telemetry into charging, discharging, rest, and gap events."
    output_family = "operating_events"
    produces_analysis_run = False
    workflow_stage = "events"
    device_roles = ALL_DEVICE_ROLES
    requires_stage = "classified"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition(
            "classified_run_id",
            "Classified run override",
            "text",
            group="Source",
        ),
        ParameterDefinition(
            "rest_current_threshold_a",
            "Rest current threshold",
            "number",
            0.5,
            group="Events",
        ),
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

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = ["--serial", str(values["serial"])]
        args.extend(
            _optional_args(
                values,
                {
                    "classified_run_id": "--classified-run-id",
                    "rest_current_threshold_a": "--rest-current-threshold-a",
                    "compression": "--compression",
                    "log_level": "--log-level",
                },
            )
        )
        return (
            str(settings.source_python),
            "-m",
            "src.build_events_from_classified",
            *args,
        )


class CapacityTrendPlugin(PipelinePlugin):
    key = "veera.capacity_trend.v1"
    name = "Build full-cycle capacity trend"
    description = "Build 100-to-0 capacity measurements and the all-discharge usage ledger."
    output_family = "capacity_trend"
    produces_analysis_run = False
    workflow_stage = "capacity_trend"
    device_roles = ("full_cycle_test",)
    requires_stage = "events"
    parameters = (
        ParameterDefinition("serial", "Battery serial", "text", required=True, group="Battery"),
        ParameterDefinition("event_run_id", "Event run override", "text", group="Source"),
        ParameterDefinition(
            "future_events",
            "Future full-cycle events",
            "integer",
            20,
            group="Trend",
        ),
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

    def build_command(self, settings: Settings, values: dict[str, Any]) -> tuple[str, ...]:
        values = self.validate(values)
        args = ["--serial", str(values["serial"])]
        args.extend(
            _optional_args(
                values,
                {
                    "event_run_id": "--event-run-id",
                    "future_events": "--future-events",
                    "compression": "--compression",
                    "log_level": "--log-level",
                },
            )
        )
        return (str(settings.source_python), "-m", "src.discharge_cycle_builder", *args)
