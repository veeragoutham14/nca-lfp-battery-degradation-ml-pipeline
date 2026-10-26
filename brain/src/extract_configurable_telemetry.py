from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
from psycopg2 import sql

from src.config import DBConfig, StorageConfig
from src.db import (
    DEFAULT_SERIAL_COL,
    DEFAULT_TABLE,
    DEFAULT_TIMESTAMP_COL,
    cursor_column_names,
    open_db_connection,
)
from src.import_module_counts import (
    build_snapshot as build_module_count_snapshot,
)
from src.import_module_counts import (
    fetch_module_counts,
    write_serial_snapshot,
)
from src.io_utils import (
    build_partition_dir,
    iter_month_ranges,
    normalize_timestamp_columns,
    parse_timestamp,
    write_parquet_chunk,
)
from src.logging_utils import configure_logging

logger = logging.getLogger(__name__)

FIELD_KEYS = ("soc", "soh", "voltage", "temperature", "current", "cycle_count")
#: Groups whose columns are resolved from device module topology instead of the
#: static profile catalog. Definitions are injected at import time from
#: src.electrothermal_contract; the catalog may omit them.
HIERARCHICAL_GROUP_KEYS = ("cell_voltage",)
GROUP_KEYS = (*FIELD_KEYS, *HIERARCHICAL_GROUP_KEYS)
PURPOSE_CHOICES = ("capacity_core", "electrothermal_history", "custom")
RESOLUTIONS = ("1s", "5m")
PART_FILE_PATTERN = re.compile(r"^part-(\d+)\.parquet$")
#: The capacity-degradation preset is exactly these signals at one second.
#: Catalog defaults (temperature/cycle_count at 5m) must never leak into it;
#: enforcement lives here in the extractor, not in the calling plugin.
CAPACITY_CORE_FIELDS = ("soc", "soh", "current", "voltage")
BUCKET_SECONDS = {"1s": 1, "5m": 300}
SAMPLE_COUNT_SUFFIX = "__sample_count"
ELECTROTHERMAL_STATISTICS = ("mean", "min", "max", "std", "count")
EXPECTED_SAMPLES_PER_BUCKET = {"1s": 1, "5m": 300}


def _bucket_seconds(resolution: str) -> int:
    try:
        return BUCKET_SECONDS[resolution]
    except KeyError as exc:
        raise ValueError(f"Unknown resolution {resolution!r}") from exc


def _next_bucket_start(timestamp: datetime, resolution: str) -> datetime:
    """Continuation point after an existing complete bucket."""

    return timestamp + timedelta(seconds=_bucket_seconds(resolution))


def _completed_end(end: datetime, resolution: str) -> datetime:
    """Clamp an exclusive end so partially filled buckets are excluded."""

    seconds = _bucket_seconds(resolution)
    if seconds == 1:
        return end
    epoch = datetime(1970, 1, 1, tzinfo=end.tzinfo or UTC)
    elapsed_seconds = (end - epoch).total_seconds()
    completed_seconds = int(elapsed_seconds // seconds) * seconds
    return epoch + timedelta(seconds=completed_seconds)


def _range_end_exclusive(actual_end, resolution: str):
    """Manifest logical exclusive end consistent with dataset resolution."""

    if actual_end is None:
        return None
    return actual_end + timedelta(seconds=_bucket_seconds(resolution))


@dataclass(frozen=True)
class FieldDefinition:
    key: str
    columns: tuple[str, ...]
    aggregation: str


@dataclass(frozen=True)
class ImportContract:
    chemistry: str
    fields: dict[str, FieldDefinition]
    selections: dict[str, tuple[str, ...]]
    catalog_version: int

    def fields_for(self, resolution: str) -> tuple[FieldDefinition, ...]:
        return tuple(
            self.fields[key]
            for key in GROUP_KEYS
            if resolution in self.selections.get(key, ())
        )


@dataclass(frozen=True)
class ExtractResult:
    resolution: str
    rows: int
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ExtractionTarget:
    """One signal dataset to write: a group at one resolution under one root.

    ``group=None`` marks the legacy mixed-column dataset used before
    purpose-based imports existed.
    """

    group: str | None
    resolution: str
    fields: tuple[FieldDefinition, ...]
    output_root: Path
    statistics: tuple[str, ...] = ()


def _parse_group_root(value: str) -> tuple[tuple[str, str], Path]:
    parts = [part.strip() for part in value.split("=", 2)]
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("Use GROUP=RESOLUTION=PATH")
    group, resolution, path = parts
    if group not in GROUP_KEYS:
        raise argparse.ArgumentTypeError(f"Unknown group {group!r}")
    if resolution not in RESOLUTIONS:
        raise argparse.ArgumentTypeError(f"Invalid resolution {resolution!r}")
    return (group, resolution), Path(path)


def _qualified_identifier(name: str) -> sql.Composed:
    parts = [part.strip() for part in name.split(".")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return sql.SQL(".").join(sql.Identifier(part) for part in parts)


def _dataset_watermark(root: Path, serial: str) -> datetime | None:
    """Newest timestamp already written for THIS serial; footer reads only.

    Scanning the unscoped root would let a newer sibling device push this
    device's continuation forward and silently skip history.
    """

    serial_root = root / f"serial={serial}"
    if not serial_root.exists():
        return None
    newest: datetime | None = None
    for path in sorted(serial_root.rglob("*.parquet")):
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
            candidate = statistics.max
            if candidate.tzinfo is None:
                candidate = candidate.replace(tzinfo=UTC)
            newest = candidate if newest is None else max(newest, candidate)
    return newest


def _drop_overlapping(frame: pd.DataFrame, time_col: str, last_written):
    """Guard against duplicate timestamps inside one signal dataset."""

    if frame.empty:
        return frame, last_written
    frame = frame.drop_duplicates(subset=[time_col], keep="last")
    if last_written is not None:
        frame = frame[frame[time_col] > pd.Timestamp(last_written)]
    if frame.empty:
        return frame, last_written
    newest = frame[time_col].iloc[-1]
    newest_dt = newest.to_pydatetime() if hasattr(newest, "to_pydatetime") else newest
    if newest_dt.tzinfo is None:
        newest_dt = newest_dt.replace(tzinfo=UTC)
    return frame, newest_dt


def _validate_frame(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    serial_col: str,
    serial: str,
    start: datetime,
    end: datetime,
) -> None:
    """Pre-write integrity gate before anything touches disk."""

    if frame.empty:
        return
    times = frame[timestamp_col]
    if not times.is_monotonic_increasing:
        raise ValueError("Extracted timestamps are not monotonically increasing")
    if times.duplicated().any():
        raise ValueError("Duplicate timestamps within extracted batch")
    if serial_col in frame.columns:
        identities = frame[serial_col].dropna().astype(str).unique()
        if len(identities) != 1 or identities[0] != str(serial):
            raise ValueError(
                f"Serial mismatch in extracted rows: {sorted(identities)[:3]}"
            )
    first = times.iloc[0]
    last = times.iloc[-1]
    first = first.to_pydatetime() if hasattr(first, "to_pydatetime") else first
    last = last.to_pydatetime() if hasattr(last, "to_pydatetime") else last
    if first.tzinfo is None:
        first = first.replace(tzinfo=UTC)
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    if first < start or last >= end:
        raise ValueError(
            "Extracted timestamps fall outside the requested range: "
            f"[{first.isoformat()}, {last.isoformat()}] not within "
            f"[{start.isoformat()}, {end.isoformat()})"
        )


def _verify_written_file(
    part: Path, *, expected_columns, expected_rows: int, timestamp_col: str
) -> None:
    parquet = pq.ParquetFile(part)
    schema_names = set(parquet.schema_arrow.names)
    missing = [column for column in expected_columns if column not in schema_names]
    if missing:
        raise ValueError(f"Written file {part.name} lacks columns: {', '.join(missing)}")
    if timestamp_col not in schema_names:
        raise ValueError(f"Written file {part.name} lacks timestamp column")
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"Written file {part.name} has {parquet.metadata.num_rows} rows; "
            f"expected {expected_rows}"
        )


def _parse_resolution(value: str) -> tuple[str, tuple[str, ...]]:
    try:
        field, raw_resolution = (part.strip().lower() for part in value.split("=", 1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use FIELD=off|1s|5m|both") from exc
    if field not in GROUP_KEYS:
        raise argparse.ArgumentTypeError(f"Unknown field {field!r}")
    mapping = {
        "off": (),
        "1s": ("1s",),
        "5m": ("5m",),
        "both": ("1s", "5m"),
    }
    if raw_resolution not in mapping:
        raise argparse.ArgumentTypeError("Resolution must be off, 1s, 5m, or both")
    return field, mapping[raw_resolution]


def load_import_contract(
    path: Path,
    *,
    chemistry: str,
    resolution_values: Sequence[tuple[str, tuple[str, ...]]],
) -> ImportContract:
    payload = json.loads(path.read_text(encoding="utf-8"))
    chemistry_key = chemistry.strip().upper()
    chemistry_definition = payload.get("chemistries", {}).get(chemistry_key)
    if not isinstance(chemistry_definition, dict):
        available = ", ".join(sorted(payload.get("chemistries", {})))
        raise ValueError(f"Unknown chemistry {chemistry_key!r}; available: {available}")

    raw_fields = chemistry_definition.get("fields", {})
    fields: dict[str, FieldDefinition] = {}
    defaults: dict[str, tuple[str, ...]] = {}
    for key in GROUP_KEYS:
        definition = raw_fields.get(key)
        if not isinstance(definition, dict):
            if key in HIERARCHICAL_GROUP_KEYS:
                # Resolved from module topology at import time; the static
                # catalog intentionally does not enumerate 56+ channel names.
                fields[key] = FieldDefinition(key, (), "mean")
                defaults[key] = ()
                continue
            raise ValueError(f"Chemistry {chemistry_key} has no {key!r} field definition")
        columns = definition.get("columns")
        aggregation = str(definition.get("aggregation", "mean")).lower()
        if not isinstance(columns, list) or not columns or not all(
            isinstance(column, str) and column.strip() for column in columns
        ):
            raise ValueError(f"Invalid columns for {chemistry_key}.{key}")
        if aggregation not in {"mean", "last"}:
            raise ValueError(f"Unsupported aggregation for {chemistry_key}.{key}: {aggregation}")
        fields[key] = FieldDefinition(key, tuple(dict.fromkeys(columns)), aggregation)
        default = str(definition.get("default_resolution", "off")).lower()
        defaults[key] = _parse_resolution(f"{key}={default}")[1]

    selections = dict(defaults)
    selections.update(dict(resolution_values))
    if not any(selections.values()):
        raise ValueError("Select at least one signal at one resolution")
    return ImportContract(
        chemistry=chemistry_key,
        fields=fields,
        selections=selections,
        catalog_version=int(payload.get("schema_version", 1)),
    )


def discover_time_bounds(
    *,
    db_config: DBConfig,
    serial: str,
    table: str,
    timestamp_col: str,
    serial_col: str,
) -> tuple[datetime, datetime]:
    query = sql.SQL(
        "SELECT MIN({time}), MAX({time}) FROM {table} WHERE {serial} = %s"
    ).format(
        time=sql.Identifier(timestamp_col),
        table=_qualified_identifier(table),
        serial=sql.Identifier(serial_col),
    )
    with open_db_connection(db_config) as conn, conn.cursor() as cursor:
        cursor.execute(query, [serial])
        first, last = cursor.fetchone()
    if first is None or last is None:
        raise ValueError(f"No database telemetry exists for serial {serial}")
    return first, last + timedelta(microseconds=1)


def hierarchical_group_fields(group: str, module_count: int | None) -> FieldDefinition:
    """Resolve temperature/cell-voltage columns from device module topology."""

    from src.electrothermal_contract import expected_hierarchical_signals

    if group not in HIERARCHICAL_GROUP_KEYS and group != "temperature":
        raise ValueError(f"{group!r} is not a topology-resolved group")
    if module_count is None:
        raise ValueError(f"{group} requires --module-count to resolve pack channels")
    signals = expected_hierarchical_signals(module_count)
    wanted = (
        "cell_voltage_channel"
        if group == "cell_voltage"
        else "pack_temperature_sensor"
    )
    columns = [
        signal.source_signal_name
        for signal in signals
        if signal.measurement_type == wanted
    ]
    if not columns:
        raise ValueError(f"No channels resolved for group {group!r}")
    return FieldDefinition(group, tuple(dict.fromkeys(columns)), "mean")


def build_extraction_targets(
    *,
    purpose,
    contract: ImportContract,
    raw_root: Path,
    five_minute_root: Path,
    group_roots: dict,
    module_count,
    requested_groups: set[str] | None = None,
) -> tuple[ExtractionTarget, ...]:
    """Route selections to output datasets.

    Legacy/core mode keeps one mixed dataset per resolution. Custom mode writes
    every explicitly requested group into its dedicated root — the precondition
    for independent per-signal watermarks and non-overlapping continuation.
    Catalog default resolutions do NOT activate groups in custom mode; only
    ``requested_groups`` (the CLI-supplied selections) are imported.
    """

    targets: list[ExtractionTarget] = []
    if purpose == "capacity_core":
        # Extractor-side enforcement of the fixed preset: exactly the four
        # analysis signals at one second. Catalog defaults (temperature,
        # cycle_count at 5m) must never leak into a capacity-core dataset.
        fields = []
        for key in CAPACITY_CORE_FIELDS:
            definition = contract.fields.get(key)
            if definition is None or not definition.columns:
                raise ValueError(
                    f"Profile catalog has no columns for capacity-core signal {key!r}"
                )
            fields.append(definition)
        return (
            ExtractionTarget(
                group=None,
                resolution="1s",
                fields=tuple(fields),
                output_root=raw_root,
            ),
        )
    if purpose == "electrothermal_history":
        requested_groups = {"temperature", "cell_voltage"}
    if purpose is None:
        for resolution, root in (("1s", raw_root), ("5m", five_minute_root)):
            fields = contract.fields_for(resolution)
            if fields:
                targets.append(ExtractionTarget(None, resolution, fields, root))
        return tuple(targets)

    if requested_groups is None:
        requested_groups = {
            key for key in GROUP_KEYS if contract.selections.get(key)
        }
    for key in sorted(requested_groups):
        if key not in GROUP_KEYS:
            raise ValueError(f"Unknown telemetry group {key!r}")
        resolutions = contract.selections.get(key, ())
        if not resolutions:
            continue
        for resolution in resolutions:
            root = group_roots.get((key, resolution))
            if root is None:
                raise ValueError(
                    f"Missing --group-output-root for {key}={resolution}; custom "
                    "imports must write each group to its own dataset"
                )
            if key in HIERARCHICAL_GROUP_KEYS or key == "temperature":
                fields = (hierarchical_group_fields(key, module_count),)
            else:
                definition = contract.fields.get(key)
                fields = () if definition is None else (definition,)
            if not fields or not fields[0].columns:
                raise ValueError(f"No columns resolved for custom group {key!r}")
            targets.append(
                ExtractionTarget(
                    key,
                    resolution,
                    fields,
                    root,
                    ELECTROTHERMAL_STATISTICS
                    if purpose == "electrothermal_history"
                    else (),
                )
            )
    if not targets:
        raise ValueError("Select at least one telemetry group")
    return tuple(targets)


def _next_part_index(output_dir: Path) -> int:
    highest = -1
    if output_dir.exists():
        for path in output_dir.iterdir():
            match = PART_FILE_PATTERN.fullmatch(path.name) if path.is_file() else None
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _one_second_query(
    *,
    table: str,
    timestamp_col: str,
    serial_col: str,
    columns: Sequence[str],
    limit: int | None,
) -> sql.Composed:
    projected = [timestamp_col, serial_col, *columns]
    query = sql.SQL(
        "SELECT {columns} FROM {table} WHERE {serial} = %s "
        "AND {time} >= %s AND {time} < %s ORDER BY {time} ASC"
    ).format(
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in projected),
        table=_qualified_identifier(table),
        serial=sql.Identifier(serial_col),
        time=sql.Identifier(timestamp_col),
    )
    if limit is not None:
        query += sql.SQL(" LIMIT %s")
    return query


def _has_timescaledb(conn: Any) -> bool:
    """Return True when the TimescaleDB extension is installed on the connection."""
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
        return cursor.fetchone() is not None


def _five_minute_query(
    *,
    table: str,
    timestamp_col: str,
    serial_col: str,
    fields: Sequence[FieldDefinition],
    limit: int | None,
    include_sample_counts: bool = False,
    statistics: Sequence[str] = (),
    timescale: bool = True,
) -> sql.Composed:
    """Build the five-minute aggregate query.

    With ``timescale`` the query uses TimescaleDB's time_bucket and last. Without
    it, the same five-minute buckets and last-value semantics are written in
    plain PostgreSQL so the import also runs on a database without the extension.
    """
    aggregates: list[sql.Composed] = []
    for field in fields:
        for column in field.columns:
            if statistics:
                aggregate_sql = {
                    "mean": "AVG({column})",
                    "min": "MIN({column})",
                    "max": "MAX({column})",
                    "std": "STDDEV_POP({column})",
                    "count": "COUNT({column})",
                }
                for statistic in statistics:
                    aggregates.append(
                        sql.SQL(
                            aggregate_sql[statistic]
                            + " FILTER (WHERE {column} IS NOT NULL) AS {alias}"
                        ).format(
                            column=sql.Identifier(column),
                            alias=sql.Identifier(f"{column}__{statistic}"),
                        )
                    )
                continue
            last_sql = (
                "last({column}, {time})"
                if timescale
                else "(array_agg({column} ORDER BY {time} DESC)"
                " FILTER (WHERE {column} IS NOT NULL))[1]"
            )
            expression = (
                sql.SQL(last_sql)
                if field.aggregation == "last"
                else sql.SQL("AVG({column})")
            ).format(column=sql.Identifier(column), time=sql.Identifier(timestamp_col))
            aggregates.append(
                sql.SQL("{expression} FILTER (WHERE {column} IS NOT NULL) AS {alias}").format(
                    expression=expression,
                    column=sql.Identifier(column),
                    alias=sql.Identifier(column),
                )
            )
            if include_sample_counts:
                # Coverage evidence for electrothermal analysis: a 5-minute
                # mean from two one-second samples is not comparable with one
                # backed by all 300.
                aggregates.append(
                    sql.SQL(
                        "COUNT({column}) FILTER (WHERE {column} IS NOT NULL)"
                        " AS {alias}"
                    ).format(
                        column=sql.Identifier(column),
                        alias=sql.Identifier(column + SAMPLE_COUNT_SUFFIX),
                    )
                )
    bucket_sql = (
        "time_bucket(INTERVAL '5 minutes', {time})"
        if timescale
        else "({time} - mod(extract(epoch FROM {time})::numeric, 300)"
        " * INTERVAL '1 second')"
    )
    query = sql.SQL(
        "SELECT " + bucket_sql + " AS {time_alias}, "
        "{serial} AS {serial_alias}, {aggregates} FROM {table} "
        "WHERE {serial} = %s AND {time} >= %s AND {time} < %s "
        "GROUP BY 1, 2 ORDER BY 1 ASC"
    ).format(
        time=sql.Identifier(timestamp_col),
        time_alias=sql.Identifier(timestamp_col),
        serial=sql.Identifier(serial_col),
        serial_alias=sql.Identifier(serial_col),
        aggregates=sql.SQL(", ").join(aggregates),
        table=_qualified_identifier(table),
    )
    if limit is not None:
        query += sql.SQL(" LIMIT %s")
    return query


def extract_resolution(
    *,
    db_config: DBConfig,
    serial: str,
    start: datetime,
    end: datetime,
    resolution: str,
    fields: Sequence[FieldDefinition],
    output_root: Path,
    chunk_size: int,
    compression: str | None,
    table: str,
    timestamp_col: str,
    serial_col: str,
    limit: int | None,
) -> ExtractResult:
    columns = tuple(dict.fromkeys(column for field in fields for column in field.columns))
    query = (
        _one_second_query(
            table=table,
            timestamp_col=timestamp_col,
            serial_col=serial_col,
            columns=columns,
            limit=limit,
        )
        if resolution == "1s"
        else _five_minute_query(
            table=table,
            timestamp_col=timestamp_col,
            serial_col=serial_col,
            fields=fields,
            limit=limit,
        )
    )
    params: list[Any] = [serial, start, end]
    if limit is not None:
        params.append(limit)

    files: list[Path] = []
    total_rows = 0
    with open_db_connection(db_config) as conn:
        if resolution != "1s" and not _has_timescaledb(conn):
            query = _five_minute_query(
                table=table,
                timestamp_col=timestamp_col,
                serial_col=serial_col,
                fields=fields,
                limit=limit,
                timescale=False,
            )
        cursor_name = f"telemetry_{resolution}_{uuid4().hex}"
        with conn.cursor(name=cursor_name) as cursor:
            cursor.itersize = chunk_size
            cursor.execute(query, params)
            names: list[str] | None = None
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                if names is None:
                    names = cursor_column_names(cursor)
                frame = normalize_timestamp_columns(pd.DataFrame.from_records(rows, columns=names))
                interval_start = frame[timestamp_col].iloc[0].to_pydatetime()
                output_dir = build_partition_dir(
                    output_root,
                    serial=serial,
                    interval_start=interval_start,
                )
                part = output_dir / f"part-{_next_part_index(output_dir):05d}.parquet"
                write_parquet_chunk(frame, part, compression=compression)
                files.append(part)
                total_rows += len(frame)
    return ExtractResult(resolution=resolution, rows=total_rows, files=tuple(files))


@dataclass(frozen=True)
class SignalImportResult:
    """Per-signal-dataset outcome recorded in manifest schema 2."""

    group: str | None
    resolution: str
    columns: tuple[str, ...]
    aggregation: str | None
    watermark_before: datetime | None
    actual_start: datetime | None
    actual_end: datetime | None
    rows: int
    files: tuple[Path, ...]
    coverage_columns: tuple[str, ...] = ()
    expected_samples_per_bucket: int | None = None


def verify_source_columns(
    *, db_config: DBConfig, table: str, needed
) -> None:
    """Fail fast when the source table lacks any projected column (§5 step 2)."""

    if "." in table:
        schema_name, table_name = table.split(".", 1)
    else:
        schema_name, table_name = "public", table
    query = sql.SQL(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s"
    )
    with open_db_connection(db_config) as conn, conn.cursor() as cursor:
        cursor.execute(query, [schema_name, table_name])
        existing = {row[0] for row in cursor.fetchall()}
    missing = sorted(set(needed) - existing)
    if missing:
        raise ValueError(
            f"Source table {table} lacks required columns: {', '.join(missing)}"
        )


def extract_target(
    *,
    db_config: DBConfig,
    serial: str,
    start: datetime,
    end: datetime,
    target: ExtractionTarget,
    chunk_size: int,
    compression,
    table: str,
    timestamp_col: str,
    serial_col: str,
    limit: int | None,
) -> SignalImportResult:
    """Extract one signal dataset from its continuation point.

    ``start`` must already be the resolution-aware continuation boundary
    computed by the caller. Frames are validated before anything touches disk;
    files land under a temporary name and are atomically renamed only after
    footer verification passes.
    """

    columns = tuple(
        dict.fromkeys(column for field in target.fields for column in field.columns)
    )
    watermark_before = _dataset_watermark(target.output_root, serial)
    include_counts = target.resolution == "5m" and target.group is not None
    query = (
        _one_second_query(
            table=table,
            timestamp_col=timestamp_col,
            serial_col=serial_col,
            columns=columns,
            limit=limit,
        )
        if target.resolution == "1s"
        else _five_minute_query(
            table=table,
            timestamp_col=timestamp_col,
            serial_col=serial_col,
            fields=target.fields,
            limit=limit,
            include_sample_counts=include_counts,
            statistics=target.statistics,
        )
    )
    params: list[Any] = [serial, start, end]
    if limit is not None:
        params.append(limit)

    coverage_columns = tuple(
        f"{column}__count" if target.statistics else column + SAMPLE_COUNT_SUFFIX
        for column in columns
    ) if include_counts else ()
    value_columns = tuple(
        f"{column}__{statistic}"
        for column in columns
        for statistic in target.statistics
    ) if target.statistics else columns
    expected_columns = (timestamp_col, serial_col, *value_columns, *coverage_columns)
    expected_samples = EXPECTED_SAMPLES_PER_BUCKET.get(
        target.resolution
    ) if include_counts else None

    files: list[Path] = []
    total_rows = 0
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    last_written = watermark_before
    aggregation = (
        target.fields[0].aggregation if len(target.fields) == 1 else None
    )
    with open_db_connection(db_config) as conn:
        if target.resolution != "1s" and not _has_timescaledb(conn):
            query = _five_minute_query(
                table=table,
                timestamp_col=timestamp_col,
                serial_col=serial_col,
                fields=target.fields,
                limit=limit,
                include_sample_counts=include_counts,
                statistics=target.statistics,
                timescale=False,
            )
        cursor_name = f"telemetry_{target.resolution}_{uuid4().hex}"
        with conn.cursor(name=cursor_name) as cursor:
            cursor.itersize = chunk_size
            cursor.execute(query, params)
            names: list[str] | None = None
            while True:
                rows = cursor.fetchmany(chunk_size)
                if not rows:
                    break
                if names is None:
                    names = cursor_column_names(cursor)
                frame = normalize_timestamp_columns(
                    pd.DataFrame.from_records(rows, columns=names)
                )
                frame, last_written = _drop_overlapping(frame, timestamp_col, last_written)
                _validate_frame(
                    frame,
                    timestamp_col=timestamp_col,
                    serial_col=serial_col,
                    serial=serial,
                    start=start,
                    end=end,
                )
                if frame.empty:
                    continue
                interval_start = frame[timestamp_col].iloc[0].to_pydatetime()
                output_dir = build_partition_dir(
                    target.output_root,
                    serial=serial,
                    interval_start=interval_start,
                )
                part_index = _next_part_index(output_dir)
                final_part = output_dir / f"part-{part_index:05d}.parquet"
                # Verify-then-rename: a failed verification must never leave a
                # readable file behind in its final position.
                temporary_part = output_dir / (
                    f"part-{part_index:05d}.{uuid4().hex[:8]}.parquet.tmp"
                )
                write_parquet_chunk(frame, temporary_part, compression=compression)
                try:
                    _verify_written_file(
                        temporary_part,
                        expected_columns=expected_columns,
                        expected_rows=len(frame),
                        timestamp_col=timestamp_col,
                    )
                    os.replace(temporary_part, final_part)
                except Exception:
                    temporary_part.unlink(missing_ok=True)
                    raise
                first_ts = frame[timestamp_col].iloc[0].to_pydatetime()
                if actual_start is None or first_ts < actual_start:
                    actual_start = first_ts
                actual_end = last_written
                files.append(final_part)
                total_rows += len(frame)
    return SignalImportResult(
        group=target.group,
        resolution=target.resolution,
        columns=columns,
        aggregation=aggregation,
        watermark_before=watermark_before,
        actual_start=actual_start,
        actual_end=actual_end,
        rows=total_rows,
        files=tuple(files),
        coverage_columns=coverage_columns,
        expected_samples_per_bucket=expected_samples,
    )


def _write_manifest(
    *,
    root: Path,
    serial: str,
    contract: ImportContract,
    start: datetime,
    end: datetime,
    requested_start: datetime | None,
    requested_end: datetime | None,
    results: Sequence[ExtractResult],
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"serial={serial}" / f"import_run={run_id}" / "import_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "serial": serial,
        "chemistry": contract.chemistry,
        "profile_catalog_version": contract.catalog_version,
        "requested_start": requested_start.isoformat() if requested_start else None,
        "requested_end": requested_end.isoformat() if requested_end else None,
        "resolved_start": start.isoformat(),
        "resolved_end_exclusive": end.isoformat(),
        "range_source": "database_bounds" if requested_start is None and requested_end is None else "mixed_or_explicit",
        "field_resolutions": {
            key: list(contract.selections[key]) for key in FIELD_KEYS
        },
        "fields": {
            key: {
                "columns": list(definition.columns),
                "aggregation": definition.aggregation,
            }
            for key, definition in contract.fields.items()
            if contract.selections[key]
        },
        "results": [
            {
                "resolution": result.resolution,
                "rows": result.rows,
                "files": [str(file) for file in result.files],
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_manifest_v2(
    *,
    root: Path,
    serial: str,
    contract: ImportContract,
    purpose: str,
    module_count: int | None,
    database_start: datetime | None,
    database_end: datetime | None,
    requested_start: datetime | None,
    requested_end: datetime | None,
    signals: Sequence[SignalImportResult],
) -> Path:
    """Immutable manifest schema 2: purpose + per-signal evidence."""

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = root / f"serial={serial}" / f"import_run={run_id}" / "import_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "serial": serial,
        "chemistry": contract.chemistry,
        "purpose": purpose,
        "module_count": module_count,
        "profile_catalog_version": contract.catalog_version,
        "requested_start": requested_start.isoformat() if requested_start else None,
        "requested_end": requested_end.isoformat() if requested_end else None,
        "database_bounds": {
            "start": database_start.isoformat() if database_start else None,
            "end_exclusive": database_end.isoformat() if database_end else None,
        },
        "signals": [
            {
                "group": signal.group,
                "resolution": signal.resolution,
                "columns": list(signal.columns),
                "aggregation": signal.aggregation,
                "watermark_before": (
                    signal.watermark_before.isoformat()
                    if signal.watermark_before
                    else None
                ),
                "actual_range": {
                    "start": signal.actual_start.isoformat()
                    if signal.actual_start
                    else None,
                    # Resolution-consistent exclusive end: watermark+1s for
                    # one-second data, last bucket + 5 minutes for aggregates.
                    "end_exclusive": (
                        _range_end_exclusive(signal.actual_end, signal.resolution).isoformat()
                        if _range_end_exclusive(signal.actual_end, signal.resolution)
                        is not None
                        else None
                    ),
                },
                "rows": signal.rows,
                "coverage": {
                    "columns": list(signal.coverage_columns),
                    "expected_samples_per_bucket": signal.expected_samples_per_bucket,
                },
                "files": [str(file) for file in signal.files],
            }
            for signal in signals
        ],
        "status": "completed",
        "completed_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def matching_completed_manifest(
    *,
    root: Path,
    serial: str,
    contract: ImportContract,
    start: datetime,
    end: datetime,
) -> Path | None:
    expected_selections = {
        key: list(contract.selections[key]) for key in FIELD_KEYS
    }
    serial_root = root / f"serial={serial}"
    for path in sorted(serial_root.glob("import_run=*/import_manifest.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            files = [
                Path(file)
                for result in payload.get("results", [])
                for file in result.get("files", [])
            ]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (
            payload.get("chemistry") == contract.chemistry
            and payload.get("resolved_start") == start.isoformat()
            and payload.get("resolved_end_exclusive") == end.isoformat()
            and payload.get("field_resolutions") == expected_selections
            and files
            and all(file.is_file() for file in files)
        ):
            return path
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chemistry-aware configurable telemetry import.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--start", type=parse_timestamp)
    parser.add_argument("--end", type=parse_timestamp)
    parser.add_argument("--field-resolution", action="append", type=_parse_resolution, default=[])
    parser.add_argument("--import-profile-catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--five-minute-output-dir", type=Path)
    parser.add_argument("--manifest-output-dir", type=Path)
    parser.add_argument("--purpose", choices=PURPOSE_CHOICES)
    parser.add_argument(
        "--group-output-root",
        action="append",
        type=_parse_group_root,
        default=[],
    )
    parser.add_argument("--module-count", type=int)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--compression", choices=("snappy", "zstd", "gzip", "none"), default="zstd")
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--timestamp-col", default=DEFAULT_TIMESTAMP_COL)
    parser.add_argument("--serial-col", default=DEFAULT_SERIAL_COL)
    parser.add_argument("--module-count-col", default="batt_modulecount")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        if args.start and args.end and args.start >= args.end:
            raise ValueError("start must be earlier than end")

        db_config = DBConfig.from_env(args.env_file)
        storage = StorageConfig.from_env(args.env_file, raw_parquet_dir=args.output_dir)
        storage.ensure_dirs()
        contract = load_import_contract(
            args.import_profile_catalog,
            chemistry=args.chemistry,
            resolution_values=args.field_resolution,
        )
        database_start, database_end = discover_time_bounds(
            db_config=db_config,
            serial=args.serial,
            table=args.table,
            timestamp_col=args.timestamp_col,
            serial_col=args.serial_col,
        )
        start = args.start or database_start
        end = args.end or database_end
        if start >= end:
            raise ValueError("Resolved start must be earlier than resolved end")

        five_minute_root = args.five_minute_output_dir or storage.data_dir / "telemetry_5min"
        manifest_root = args.manifest_output_dir or storage.data_dir / "import_manifests"
        compression = None if args.compression == "none" else args.compression

        if args.purpose is None:
            # Legacy single-dataset import; unchanged behavior and manifest v1.
            existing_manifest = matching_completed_manifest(
                root=manifest_root,
                serial=args.serial,
                contract=contract,
                start=start,
                end=end,
            )
            if existing_manifest is not None:
                logger.info("Matching import already exists; no rows rewritten: %s", existing_manifest)
                return 0
            results: list[ExtractResult] = []
            for resolution, output_root in (
                ("1s", storage.raw_parquet_dir),
                ("5m", five_minute_root),
            ):
                fields = contract.fields_for(resolution)
                if not fields:
                    continue
                resolution_rows = 0
                resolution_files: list[Path] = []
                remaining_limit = args.limit
                for interval_start, interval_end in iter_month_ranges(start, end):
                    if remaining_limit is not None and remaining_limit <= 0:
                        break
                    result = extract_resolution(
                        db_config=db_config,
                        serial=args.serial,
                        start=interval_start,
                        end=interval_end,
                        resolution=resolution,
                        fields=fields,
                        output_root=output_root,
                        chunk_size=args.chunk_size,
                        compression=compression,
                        table=args.table,
                        timestamp_col=args.timestamp_col,
                        serial_col=args.serial_col,
                        limit=remaining_limit,
                    )
                    resolution_rows += result.rows
                    resolution_files.extend(result.files)
                    if remaining_limit is not None:
                        remaining_limit -= result.rows
                results.append(ExtractResult(resolution, resolution_rows, tuple(resolution_files)))
                logger.info("Imported %s rows at %s", f"{resolution_rows:,}", resolution)

            manifest = _write_manifest(
                root=manifest_root,
                serial=args.serial,
                contract=contract,
                start=start,
                end=end,
                requested_start=args.start,
                requested_end=args.end,
                results=results,
            )
            logger.info("Import manifest: %s", manifest)
            return 0

        # PURPOSE_BASED_FLOW_MARKER
        targets = build_extraction_targets(
            purpose=args.purpose,
            contract=contract,
            raw_root=args.output_dir or storage.raw_parquet_dir,
            five_minute_root=five_minute_root,
            group_roots=dict(args.group_output_root),
            module_count=args.module_count,
            requested_groups={
                field for field, resolutions in args.field_resolution if resolutions
            },
        )

        # Capacity imports are also the authoritative module-metadata entry
        # point. Refresh this before the potentially long telemetry transfer so
        # a later interruption cannot leave an otherwise known device without
        # the module count required by downstream electrothermal imports.
        resolved_module_count = args.module_count
        if args.purpose == "capacity_core":
            observed_module_counts = fetch_module_counts(
                db_config=db_config,
                serials=[str(args.serial)],
                table=args.table,
                serial_col=args.serial_col,
                module_count_col=args.module_count_col,
                timestamp_col=args.timestamp_col,
                recent_samples=1_000,
            )
            module_snapshot = build_module_count_snapshot(
                [str(args.serial)],
                observed_module_counts,
                minimum_agreement=0.95,
            )
            module_output = write_serial_snapshot(
                module_snapshot,
                storage.data_dir / "processed" / "device_metadata" / "module_counts",
                str(args.serial),
            )
            module_row = module_snapshot.iloc[0]
            if module_row["module_count_status"] == "verified":
                resolved_module_count = int(module_row["module_count"])
            logger.info(
                "Module metadata refreshed for %s: %s (%s)",
                args.serial,
                module_row["module_count_status"],
                module_output,
            )

        needed_columns: set[str] = {args.timestamp_col, args.serial_col}
        for target in targets:
            for field in target.fields:
                needed_columns.update(field.columns)
        verify_source_columns(db_config=db_config, table=args.table, needed=needed_columns)

        def _merge(base, extra: SignalImportResult) -> SignalImportResult:
            if base is None:
                return extra
            return SignalImportResult(
                group=base.group,
                resolution=base.resolution,
                columns=base.columns,
                aggregation=base.aggregation,
                watermark_before=base.watermark_before,
                actual_start=(
                    base.actual_start
                    if base.actual_start is not None
                    and (extra.actual_start is None or base.actual_start <= extra.actual_start)
                    else extra.actual_start
                ),
                actual_end=(
                    base.actual_end
                    if base.actual_end is not None
                    and (extra.actual_end is None or base.actual_end >= extra.actual_end)
                    else extra.actual_end
                ),
                rows=base.rows + extra.rows,
                files=base.files + extra.files,
            )

        signal_results: list[SignalImportResult] = []
        for target in targets:
            watermark = _dataset_watermark(target.output_root, args.serial)
            # Resolution-aware continuation: 1s continues at watermark+1s; a
            # 5-minute dataset continues at watermark+300s so a partially
            # filled final bucket is fully rebuilt from source instead of
            # being silently dropped by the duplicate guard.
            effective_start = args.start
            if effective_start is None:
                effective_start = (
                    _next_bucket_start(watermark, target.resolution)
                    if watermark is not None
                    else database_start
                )
            elif watermark is not None and effective_start <= watermark:
                logger.warning(
                    "Requested start overlaps existing %s/%s data for %s; "
                    "continuing after watermark %s",
                    target.group,
                    target.resolution,
                    args.serial,
                    watermark.isoformat(),
                )
                effective_start = _next_bucket_start(watermark, target.resolution)
            # Exclude a partially filled newest bucket for aggregated data.
            target_end = _completed_end(end, target.resolution)
            merged = SignalImportResult(
                group=target.group,
                resolution=target.resolution,
                columns=tuple(
                    dict.fromkeys(
                        column for field in target.fields for column in field.columns
                    )
                ),
                aggregation=(
                    target.fields[0].aggregation if len(target.fields) == 1 else None
                ),
                watermark_before=watermark,
                actual_start=None,
                actual_end=None,
                rows=0,
                files=(),
            )
            if effective_start >= target_end:
                logger.info(
                    "No new %s/%s rows for %s after watermark",
                    target.group,
                    target.resolution,
                    args.serial,
                )
                signal_results.append(merged)
                continue
            remaining_limit = args.limit
            for interval_start, interval_end in iter_month_ranges(effective_start, target_end):
                if remaining_limit is not None and remaining_limit <= 0:
                    break
                result = extract_target(
                    db_config=db_config,
                    serial=args.serial,
                    start=interval_start,
                    end=interval_end,
                    target=target,
                    chunk_size=args.chunk_size,
                    compression=compression,
                    table=args.table,
                    timestamp_col=args.timestamp_col,
                    serial_col=args.serial_col,
                    limit=remaining_limit,
                )
                merged = _merge(merged, result)
                if remaining_limit is not None:
                    remaining_limit -= result.rows
            coverage_columns = (
                tuple(
                    f"{column}__count" if target.statistics else column + SAMPLE_COUNT_SUFFIX
                    for column in merged.columns
                )
                if merged.resolution == "5m" and merged.group is not None
                else ()
            )
            merged = replace(
                merged,
                coverage_columns=coverage_columns,
                expected_samples_per_bucket=(
                    EXPECTED_SAMPLES_PER_BUCKET.get(merged.resolution)
                    if coverage_columns
                    else None
                ),
            )
            signal_results.append(merged)
            logger.info(
                "Imported %s rows for %s/%s",
                f"{merged.rows:,}",
                target.group,
                target.resolution,
            )

        manifest = _write_manifest_v2(
            root=manifest_root,
            serial=args.serial,
            contract=contract,
            purpose=args.purpose,
            module_count=resolved_module_count,
            database_start=database_start,
            database_end=database_end,
            requested_start=args.start,
            requested_end=args.end,
            signals=signal_results,
        )
        logger.info("Import manifest: %s", manifest)
        return 0
    except Exception as exc:
        if args.log_level == "DEBUG":
            logger.exception("Configurable telemetry import failed")
        else:
            logger.error("Configurable telemetry import failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
