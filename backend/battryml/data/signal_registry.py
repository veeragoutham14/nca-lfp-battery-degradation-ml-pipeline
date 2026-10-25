"""Import-purpose and telemetry-signal registry for the redesigned extraction page.

The registry is the single authority for which signals exist, which resolutions
each supports, and how a selected signal resolves into database columns. The
frontend renders from this contract; it never invents column names.
"""

from __future__ import annotations

from dataclasses import dataclass

CAPACITY_CORE_PURPOSE = "capacity_core"
ELECTROTHERMAL_HISTORY_PURPOSE = "electrothermal_history"
CUSTOM_PURPOSE = "custom"
IMPORT_PURPOSES = (CAPACITY_CORE_PURPOSE, ELECTROTHERMAL_HISTORY_PURPOSE, CUSTOM_PURPOSE)

RESOLUTION_OFF = "off"
RESOLUTION_1S = "1s"
RESOLUTION_5M = "5m"
RESOLUTION_BOTH = "both"
RESOLUTION_CHOICES = (RESOLUTION_OFF, RESOLUTION_1S, RESOLUTION_5M, RESOLUTION_BOTH)

CORE_RESOLUTION = RESOLUTION_1S

#: Voltage-group topology from docs/DATA_VALIDATION.md: two packs per module,
#: fourteen voltage-group channels per pack, four pack-temperature sensors per
#: pack. Pack-temperature channels are BMS sensors, not cell-core temperatures.
PACKS_PER_MODULE = 2
VOLTAGE_GROUPS_PER_PACK = 14
TEMPERATURE_SENSORS_PER_PACK = 4


@dataclass(frozen=True)
class CoreSignal:
    """Fixed member of the capacity-degradation preset. Not user-configurable."""

    key: str
    label: str
    description: str


@dataclass(frozen=True)
class CustomSignalGroup:
    """Selectable custom-extraction group with configurable resolution."""

    key: str
    label: str
    description: str
    default_resolution: str
    #: Group needs chemistry + verified module count to resolve columns.
    requires_module_topology: bool = False
    #: Field key in config/telemetry_import_profiles.json, when the group maps
    #: onto system-level catalog fields instead of hierarchical channel sets.
    profile_field: str | None = None
    #: Requires the grouped-output extractor (implementation order step 6).
    requires_grouped_extractor: bool = False


CORE_PRESET: tuple[CoreSignal, ...] = (
    CoreSignal("time", "Timestamp", "One-second UTC source timestamps"),
    CoreSignal("serial", "Serial number", "Device identifier column"),
    CoreSignal(
        "soc",
        "SOC",
        "State of charge; anchors discharge evidence classes and SOC boundaries",
    ),
    CoreSignal(
        "soh",
        "BMS SOH",
        "Weak auxiliary label used for persistent-transition detection",
    ),
    CoreSignal(
        "current",
        "Current",
        "Signed stack current (positive charging); integrates Ah and Wh",
    ),
    CoreSignal(
        "voltage",
        "Stack voltage",
        "Stack voltage for Wh integration and plausibility checks",
    ),
)

CUSTOM_GROUPS: tuple[CustomSignalGroup, ...] = (
    CustomSignalGroup(
        "soc",
        "SOC",
        "State of charge",
        default_resolution=RESOLUTION_1S,
        profile_field="soc",
    ),
    CustomSignalGroup(
        "soh",
        "BMS SOH",
        "BMS state-of-health estimate",
        default_resolution=RESOLUTION_1S,
        profile_field="soh",
    ),
    CustomSignalGroup(
        "current",
        "Current",
        "Signed BMS stack current",
        default_resolution=RESOLUTION_1S,
        profile_field="current",
    ),
    CustomSignalGroup(
        "voltage",
        "Stack voltage",
        "BMS stack voltage",
        default_resolution=RESOLUTION_1S,
        profile_field="voltage",
    ),
    CustomSignalGroup(
        "temperature",
        "Temperature channels",
        "Pack/module temperature sensors resolved per module and pack",
        default_resolution=RESOLUTION_5M,
        requires_module_topology=True,
        requires_grouped_extractor=True,
    ),
    CustomSignalGroup(
        "cell_voltage",
        "Cell/voltage-group channels",
        "Voltage-group channels resolved per module, pack, and group",
        default_resolution=RESOLUTION_5M,
        requires_module_topology=True,
        requires_grouped_extractor=True,
    ),
    CustomSignalGroup(
        "cycle_count",
        "Lifecycle cycle counter",
        "BMS sys_cycles counter; optional freshness signal",
        default_resolution=RESOLUTION_5M,
        profile_field="cycle_count",
    ),
)

CUSTOM_GROUP_KEYS = tuple(group.key for group in CUSTOM_GROUPS)
CUSTOM_GROUPS_BY_KEY = {group.key: group for group in CUSTOM_GROUPS}
CORE_KEYS = tuple(signal.key for signal in CORE_PRESET)


def capacity_core_resolutions() -> dict[str, str]:
    """Fixed resolution assignment for the capacity-degradation preset."""

    return {key: CORE_RESOLUTION for key in ("soc", "soh", "current", "voltage")}


def electrothermal_history_resolutions() -> dict[str, str]:
    """Fixed five-minute contract shared by electrothermal analyses."""

    return {"temperature": RESOLUTION_5M, "cell_voltage": RESOLUTION_5M}


def validate_custom_selection(selections: dict[str, str]) -> dict[str, str]:
    """Normalize and validate a custom {group: resolution} mapping."""

    normalized: dict[str, str] = {}
    for group_key, resolution in selections.items():
        group = CUSTOM_GROUPS_BY_KEY.get(group_key)
        if group is None:
            raise ValueError(f"Unknown telemetry group: {group_key}")
        if resolution not in RESOLUTION_CHOICES:
            raise ValueError(
                f"{group.label} resolution must be one of: {', '.join(RESOLUTION_CHOICES)}"
            )
        if resolution != RESOLUTION_OFF:
            normalized[group_key] = resolution
    if not normalized:
        raise ValueError("Select at least one telemetry group")
    return normalized


def expected_column_count(
    group_key: str, module_count: int | None
) -> int | None:
    """Deterministic expected column count for topology-resolved groups."""

    if module_count is None or module_count <= 0:
        return None
    if group_key == "temperature":
        return module_count * PACKS_PER_MODULE * TEMPERATURE_SENSORS_PER_PACK
    if group_key == "cell_voltage":
        return module_count * PACKS_PER_MODULE * VOLTAGE_GROUPS_PER_PACK
    return None
