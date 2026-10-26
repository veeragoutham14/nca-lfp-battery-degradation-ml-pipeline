"""Per-signal/per-resolution import planning: local ranges, watermarks, plans.

Watermarks are derived from Parquet row-group statistics of each signal
dataset — no telemetry rows are read. The core preset keeps the legacy
``data/raw_parquet`` layout; every custom group owns a dedicated root so
signals can continue independently without overlapping writes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.data.signal_registry import (
    CAPACITY_CORE_PURPOSE,
    CORE_RESOLUTION,
    CUSTOM_GROUPS_BY_KEY,
    ELECTROTHERMAL_HISTORY_PURPOSE,
    capacity_core_resolutions,
    electrothermal_history_resolutions,
    expected_column_count,
)

CUSTOM_TELEMETRY_ROOT = "custom_telemetry"
ELECTROTHERMAL_TELEMETRY_ROOT = "electrothermal_telemetry"


def dataset_root(
    source_data: Path,
    group: str,
    resolution: str,
    chemistry: str | None = None,
) -> Path:
    """Filesystem root for one signal group at one resolution.

    Custom groups are chemistry-partitioned so per-device watermarks can be
    scoped by chemistry + serial without touching sibling fleets.
    """

    if group == "core":
        return source_data / "raw_parquet"
    if chemistry is None:
        raise ValueError("Custom telemetry roots require a chemistry")
    return (
        source_data
        / CUSTOM_TELEMETRY_ROOT
        / group
        / resolution
        / f"chemistry={chemistry}"
    )


def core_dataset_root(source_data: Path) -> Path:
    return dataset_root(source_data, "core", CORE_RESOLUTION)


def custom_dataset_root(
    source_data: Path, group: str, resolution: str, chemistry: str
) -> Path:
    return dataset_root(source_data, group, resolution, chemistry)


def electrothermal_dataset_root(
    source_data: Path, group: str, chemistry: str
) -> Path:
    return (
        source_data
        / ELECTROTHERMAL_TELEMETRY_ROOT
        / group
        / "5m"
        / f"chemistry={chemistry}"
    )


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def serial_scan_dirs(
    source_data: Path,
    *,
    group: str,
    resolution: str,
    chemistry: str,
    serial: str,
) -> list[Path]:
    """Every directory that may hold THIS device's rows for one signal set.

    Core data spans two layouts (legacy ``serial=`` and canonical
    ``chemistry=/serial=``); custom groups are always canonical.
    """

    if group == "core":
        base = source_data / "raw_parquet"
        return [
            base / f"chemistry={chemistry}" / f"serial={serial}",
            base / f"serial={serial}",
        ]
    return [
        dataset_root(source_data, group, resolution, chemistry)
        / f"serial={serial}"
    ]


def local_time_range(dirs) -> tuple[datetime | None, datetime | None]:
    """Bounded [start, end_exclusive] from row-group statistics of ``dirs``.

    ``dirs`` must already be serial-scoped by the caller. Opens only Parquet
    footers/metadata; never materializes rows.
    """

    minimum: datetime | None = None
    maximum: datetime | None = None
    for root in dirs:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            try:
                parquet = pq.ParquetFile(path)
            except OSError:
                continue
            names = parquet.schema_arrow.names
            if "time" not in names:
                continue
            time_index = names.index("time")
            metadata = parquet.metadata
            for row_group_index in range(metadata.num_row_groups):
                statistics = (
                    metadata.row_group(row_group_index).column(time_index).statistics
                )
                if statistics is None or not statistics.has_min_max:
                    continue
                start = _aware(statistics.min)
                end = _aware(statistics.max)
                minimum = start if minimum is None else min(minimum, start)
                maximum = end if maximum is None else max(maximum, end)
    return minimum, maximum


def count_rows_and_months(dirs) -> tuple[int, int] | None:
    """Total rows and distinct month partitions under serial-scoped ``dirs``."""

    months: set[tuple[int, int]] = set()
    total_rows = 0
    seen_any = False
    for root in dirs:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.parquet")):
            try:
                parquet = pq.ParquetFile(path)
                years = [part for part in path.parts if part.startswith("year=")]
                months_ = [part for part in path.parts if part.startswith("month=")]
                if years and months_:
                    months.add((int(years[0][5:]), int(months_[0][6:])))
                    seen_any = True
                total_rows += parquet.metadata.num_rows
            except OSError:
                continue
    if not seen_any:
        return None
    return total_rows, len(months)


def estimated_rows_per_month(dirs) -> int | None:
    totals = count_rows_and_months(dirs)
    if totals is None or totals[1] == 0:
        return None
    return max(1, round(totals[0] / totals[1]))


def planned_start(
    *,
    group_key: str,
    resolution: str,
    watermark_end: datetime | None,
    device_start: datetime | None,
) -> tuple[datetime | None, str]:
    """Continuation rule: existing signals resume after their watermark;
    new signals begin at the device's configured start date."""

    if watermark_end is not None:
        interval = timedelta(minutes=5) if resolution == "5m" else timedelta(seconds=1)
        return watermark_end + interval, "continue_after_watermark"
    if device_start is not None:
        return device_start, "device_configuration_start"
    return None, "database_first_timestamp"


def _resolutions_for(selection: str) -> tuple[str, ...]:
    mapping = {
        "off": (),
        "1s": ("1s",),
        "5m": ("5m",),
        "both": ("1s", "5m"),
    }
    return mapping[selection]


def build_import_plan(
    *,
    purpose: str,
    selections: dict[str, str],
    source_data: Path,
    serial: str,
    device_start: datetime | None,
    chemistry: str,
    module_count: int | None,
    database_end: datetime | None = None,
) -> dict[str, Any]:
    """Compose the per-signal plan shown to the operator before execution."""

    if purpose == CAPACITY_CORE_PURPOSE:
        effective = capacity_core_resolutions()
        entries: list[tuple[str, str]] = [("core", CORE_RESOLUTION)]
    elif purpose == ELECTROTHERMAL_HISTORY_PURPOSE:
        effective = electrothermal_history_resolutions()
        entries = list(effective.items())
    else:
        effective = dict(selections)
        entries = [
            (group_key, resolution)
            for group_key, resolution in effective.items()
            for resolution in _resolutions_for(resolution)
        ]

    signals: list[dict[str, Any]] = []
    for group_key, resolution in entries:
        group_meta = CUSTOM_GROUPS_BY_KEY.get(group_key)
        if purpose == ELECTROTHERMAL_HISTORY_PURPOSE:
            dirs = [electrothermal_dataset_root(source_data, group_key, chemistry) / f"serial={serial}"]
        else:
            dirs = serial_scan_dirs(
                source_data,
                group=group_key,
                resolution=resolution,
                chemistry=chemistry,
                serial=serial,
            )
        local_start, watermark_end = local_time_range(dirs)
        planned, rule = planned_start(
            group_key=group_key,
            resolution=resolution,
            watermark_end=watermark_end,
            device_start=device_start,
        )
        topology_group = group_key in {"temperature", "cell_voltage"}
        signals.append(
            {
                "group": group_key,
                "label": (
                    "Core capacity-analysis signals"
                    if group_key == "core"
                    else group_meta.label
                ),
                "resolution": resolution,
                "chemistry": chemistry,
                "column_count": expected_column_count(group_key, module_count)
                if topology_group
                else None,
                "local_range": {
                    "start": local_start.isoformat() if local_start else None,
                    "end": watermark_end.isoformat() if watermark_end else None,
                },
                "planned_start": planned.isoformat() if planned else None,
                "planned_rule": rule,
                "planned_end_open": database_end is None,
                "database_end": database_end.isoformat() if database_end else None,
                "estimated_rows_per_month": estimated_rows_per_month(dirs),
                "requires_module_topology": bool(topology_group),
                "topology_resolved": bool(not topology_group or module_count),
            }
        )
    return {"purpose": purpose, "signals": signals}
