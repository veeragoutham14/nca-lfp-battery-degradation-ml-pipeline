from __future__ import annotations

from pathlib import Path

from src.extract_configurable_telemetry import (
    _five_minute_query,
    _parse_resolution,
    load_import_contract,
)

CATALOG = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "telemetry_import_profiles.json"
)


def test_contract_keeps_nca_and_lfp_defaults_separate() -> None:
    nca = load_import_contract(CATALOG, chemistry="NCA", resolution_values=[])
    lfp = load_import_contract(CATALOG, chemistry="LFP", resolution_values=[])

    assert nca.selections["soc"] == ("1s",)
    assert nca.selections["temperature"] == ("5m",)
    assert nca.selections["cycle_count"] == ("5m",)
    assert lfp.selections["soc"] == ("1s",)
    assert lfp.selections["current"] == ("1s",)
    assert lfp.selections["cycle_count"] == ("5m",)


def test_cycle_count_resolution_is_supported() -> None:
    assert _parse_resolution("cycle_count=5m") == ("cycle_count", ("5m",))


def test_explicit_resolution_overrides_chemistry_default() -> None:
    contract = load_import_contract(
        CATALOG,
        chemistry="LFP",
        resolution_values=[_parse_resolution("soc=both"), _parse_resolution("soh=off")],
    )

    assert contract.selections["soc"] == ("1s", "5m")
    assert contract.selections["soh"] == ()
    assert {field.key for field in contract.fields_for("1s")} == {
        "soc",
        "voltage",
        "current",
    }


def test_five_minute_query_uses_mean_and_last_aggregations() -> None:
    contract = load_import_contract(
        CATALOG,
        chemistry="NCA",
        resolution_values=[_parse_resolution("soh=5m")],
    )
    query = _five_minute_query(
        table="public.battery_telemetry",
        timestamp_col="time",
        serial_col="serial",
        fields=contract.fields_for("5m"),
        limit=None,
    )
    rendered = str(query)

    assert "time_bucket" in rendered
    assert "AVG" in rendered
    assert "last" in rendered


# --- Purpose-based grouped extraction (manifest schema 2) -------------------


def test_parse_group_root_accepts_and_rejects(tmp_path: Path) -> None:
    import argparse

    import pytest

    from src.extract_configurable_telemetry import _parse_group_root

    (group, resolution), path = _parse_group_root(
        f"temperature=5m={tmp_path / 'temp'}"
    )
    assert (group, resolution) == ("temperature", "5m")
    assert path == tmp_path / "temp"

    with pytest.raises(argparse.ArgumentTypeError):
        _parse_group_root("temperature")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_group_root("made_up=1s=/tmp/x")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_group_root("soc=30s=/tmp/x")


def test_dataset_watermark_reads_footer_only(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.extract_configurable_telemetry import _dataset_watermark

    root = tmp_path / "custom_telemetry" / "cycle_count" / "5m"
    path = root / "serial=100000020" / "year=2026" / "month=07" / "part-00000.parquet"
    path.parent.mkdir(parents=True)
    times = [
        datetime(2026, 7, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 7, 1, 12, 5, tzinfo=UTC),
    ]
    pq.write_table(
        pa.table(
            {
                "time": pa.array(times, type=pa.timestamp("us", tz="UTC")),
                "sys_cycles": [10.0, 11.0],
            }
        ),
        path,
    )
    # A newer sibling device must NOT move this device's watermark.
    other = (
        root
        / "serial=100000021"
        / "year=2026"
        / "month=08"
        / "part-00000.parquet"
    )
    other.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "time": pa.array(
                    [datetime(2026, 8, 1, 0, 0, tzinfo=UTC)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "sys_cycles": [99.0],
            }
        ),
        other,
    )

    watermark = _dataset_watermark(root, "100000020")
    assert watermark == datetime(2026, 7, 1, 12, 5, tzinfo=UTC)
    # The sibling's own watermark is later; scoping is what keeps them apart.
    assert _dataset_watermark(root, "100000021") == datetime(
        2026, 8, 1, 0, 0, tzinfo=UTC
    )
    assert _dataset_watermark(tmp_path / "missing", "100000020") is None


def test_bucket_continuation_and_completion_math() -> None:
    from datetime import UTC, datetime

    from src.extract_configurable_telemetry import (
        _completed_end,
        _next_bucket_start,
        _range_end_exclusive,
    )

    one_second = datetime(2026, 7, 17, 11, 1, tzinfo=UTC)
    five_minute = datetime(2026, 7, 17, 11, 5, tzinfo=UTC)

    assert _next_bucket_start(one_second, "1s") == datetime(
        2026, 7, 17, 11, 1, 1, tzinfo=UTC
    )
    # Five-minute data continues a full bucket later, not one second later.
    assert _next_bucket_start(five_minute, "5m") == datetime(
        2026, 7, 17, 11, 10, tzinfo=UTC
    )
    # An exclusive end inside an incomplete bucket is pulled back to the last
    # completed boundary so partial buckets are never materialized.
    assert _completed_end(datetime(2026, 7, 17, 11, 7, 30, tzinfo=UTC), "5m") == (
        datetime(2026, 7, 17, 11, 5, tzinfo=UTC)
    )
    assert _completed_end(
        datetime(2026, 7, 17, 11, 7, 29, 1, tzinfo=UTC), "5m"
    ) == datetime(2026, 7, 17, 11, 5, tzinfo=UTC)
    assert _completed_end(one_second, "1s") == one_second
    # Manifest ranges match dataset resolution.
    assert _range_end_exclusive(five_minute, "5m") == datetime(
        2026, 7, 17, 11, 10, tzinfo=UTC
    )
    assert _range_end_exclusive(one_second, "1s") == datetime(
        2026, 7, 17, 11, 1, 1, tzinfo=UTC
    )
    assert _range_end_exclusive(None, "5m") is None


def test_drop_overlapping_removes_duplicate_and_stale_rows() -> None:
    from datetime import UTC, datetime

    import pandas as pd

    from src.extract_configurable_telemetry import _drop_overlapping

    def stamp(minute: int) -> pd.Timestamp:
        return pd.Timestamp(datetime(2026, 7, 1, 0, minute, tzinfo=UTC))

    frame = pd.DataFrame({"time": [stamp(0), stamp(0), stamp(1)]})
    kept, last_written = _drop_overlapping(frame, "time", None)
    assert list(kept["time"]) == [stamp(0), stamp(1)]
    assert last_written == datetime(2026, 7, 1, 0, 1, tzinfo=UTC)

    later = pd.DataFrame({"time": [stamp(0), stamp(2)]})
    kept, last_written = _drop_overlapping(later, "time", last_written)
    assert list(kept["time"]) == [stamp(2)]
    assert last_written == datetime(2026, 7, 1, 0, 2, tzinfo=UTC)


def _contract(**kwargs):
    from src.extract_configurable_telemetry import load_import_contract

    return load_import_contract(CATALOG, **kwargs)


def test_build_targets_custom_mode_requires_dedicated_roots(tmp_path: Path) -> None:
    import pytest

    from src.extract_configurable_telemetry import build_extraction_targets

    contract = _contract(
        chemistry="LFP",
        resolution_values=[("cycle_count", ("5m",))],
    )
    with pytest.raises(ValueError, match="Missing --group-output-root"):
        build_extraction_targets(
            purpose="custom",
            contract=contract,
            raw_root=tmp_path / "raw",
            five_minute_root=tmp_path / "5min",
            group_roots={},
            module_count=None,
        )


def test_build_targets_routes_each_group_to_its_own_root(tmp_path: Path) -> None:
    from src.extract_configurable_telemetry import build_extraction_targets

    contract = _contract(
        chemistry="LFP",
        resolution_values=[
            ("cycle_count", ("5m",)),
            ("cell_voltage", ("1s",)),
        ],
    )
    targets = build_extraction_targets(
        purpose="custom",
        contract=contract,
        raw_root=tmp_path / "raw",
        five_minute_root=tmp_path / "5min",
        group_roots={
            ("cycle_count", "5m"): tmp_path / "g1",
            ("cell_voltage", "1s"): tmp_path / "g2",
        },
        module_count=2,
        requested_groups={"cycle_count", "cell_voltage"},
    )
    by_key = {(target.group, target.resolution): target for target in targets}
    assert set(by_key) == {("cycle_count", "5m"), ("cell_voltage", "1s")}
    # Hierarchical cell-voltage channels resolved from module topology.
    cv_columns = by_key[("cell_voltage", "1s")].fields[0].columns
    assert len(cv_columns) == 56
    assert cv_columns[0] == "batt_batterymodules_mod1_pack1_cellvoltage_mv_0"
    cycle_target = by_key[("cycle_count", "5m")]
    assert cycle_target.fields[0].columns == ("sys_cycles",)
    assert cycle_target.fields[0].aggregation == "last"


def test_build_targets_capacity_core_enforces_fixed_schema(tmp_path: Path) -> None:
    """Catalog defaults must not leak temperature/cycle_count into core."""

    from src.extract_configurable_telemetry import build_extraction_targets

    contract = _contract(chemistry="NCA", resolution_values=[])
    targets = build_extraction_targets(
        purpose="capacity_core",
        contract=contract,
        raw_root=tmp_path / "raw",
        five_minute_root=tmp_path / "5min",
        group_roots={},
        module_count=None,
    )
    assert len(targets) == 1
    target = targets[0]
    assert target.resolution == "1s"
    assert target.group is None
    columns = tuple(
        column for field in target.fields for column in field.columns
    )
    assert set(columns) == {
        "batt_soc_pct",
        "batt_soh_pct",
        "batt_current_a",
        "batt_voltage_v",
    }


def test_build_targets_legacy_purpose_keeps_catalog_defaults(tmp_path: Path) -> None:
    from src.extract_configurable_telemetry import build_extraction_targets

    contract = _contract(chemistry="NCA", resolution_values=[])
    targets = build_extraction_targets(
        purpose=None,
        contract=contract,
        raw_root=tmp_path / "raw",
        five_minute_root=tmp_path / "5min",
        group_roots={},
        module_count=None,
    )
    assert {target.group for target in targets} == {None}
    assert sorted(target.resolution for target in targets) == ["1s", "5m"]
    # Legacy behavior intentionally retains catalog defaults.
    five_min = next(t for t in targets if t.resolution == "5m")
    five_columns = {
        column for field in five_min.fields for column in field.columns
    }
    assert "batt_bmstemperaturemax_c" in five_columns
    assert "sys_cycles" in five_columns


def test_hierarchical_temperature_fields_resolved_from_module_count() -> None:
    import pytest

    from src.extract_configurable_telemetry import hierarchical_group_fields

    fields = hierarchical_group_fields("temperature", 2)
    assert len(fields.columns) == 16
    assert fields.aggregation == "mean"
    assert fields.columns[0] == "batt_batterymodules_mod1_pack1_temperature0_c"

    with pytest.raises(ValueError, match="module-count"):
        hierarchical_group_fields("temperature", None)


def test_verify_written_file_detects_missing_column(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from src.extract_configurable_telemetry import _verify_written_file

    part = tmp_path / "part-00000.parquet"
    pq.write_table(pa.table({"time": [1], "value": [2.0]}), part)
    _verify_written_file(
        part, expected_columns=("value",), expected_rows=1, timestamp_col="time"
    )
    with pytest.raises(ValueError, match="lacks columns"):
        _verify_written_file(
            part,
            expected_columns=("value", "batt_soc_pct"),
            expected_rows=1,
            timestamp_col="time",
        )


def test_manifest_v2_writer_records_per_signal_evidence(tmp_path: Path) -> None:
    import json
    from datetime import UTC, datetime

    from src.extract_configurable_telemetry import (
        SignalImportResult,
        _write_manifest_v2,
    )

    contract = _contract(chemistry="LFP", resolution_values=[])
    signals = [
        SignalImportResult(
            group="cycle_count",
            resolution="5m",
            columns=("sys_cycles",),
            aggregation="last",
            watermark_before=datetime(2026, 7, 1, tzinfo=UTC),
            actual_start=datetime(2026, 7, 1, 0, 5, tzinfo=UTC),
            actual_end=datetime(2026, 8, 1, 0, 0, tzinfo=UTC),
            rows=8888,
            files=(tmp_path / "part-00000.parquet",),
        )
    ]
    path = _write_manifest_v2(
        root=tmp_path,
        serial="100000020",
        contract=contract,
        purpose="custom",
        module_count=2,
        database_start=datetime(2025, 6, 30, tzinfo=UTC),
        database_end=datetime(2026, 8, 20, tzinfo=UTC),
        requested_start=None,
        requested_end=None,
        signals=signals,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["purpose"] == "custom"
    assert payload["module_count"] == 2
    assert payload["status"] == "completed"
    signal = payload["signals"][0]
    assert signal["group"] == "cycle_count"
    assert signal["columns"] == ["sys_cycles"]
    assert signal["watermark_before"] == "2026-07-01T00:00:00+00:00"
    assert signal["actual_range"]["start"] == "2026-07-01T00:05:00+00:00"
    # Five-minute manifests use bucket-consistent exclusive ends.
    assert signal["actual_range"]["end_exclusive"] == "2026-08-01T00:05:00+00:00"
    assert signal["rows"] == 8888


def test_validate_frame_rejects_corrupt_batches() -> None:
    from datetime import UTC, datetime

    import pandas as pd
    import pytest

    from src.extract_configurable_telemetry import _validate_frame

    start = datetime(2026, 7, 1, tzinfo=UTC)
    end = datetime(2026, 7, 2, tzinfo=UTC)

    def stamp(minute: int) -> pd.Timestamp:
        return pd.Timestamp(datetime(2026, 7, 1, 0, minute, tzinfo=UTC))

    good = pd.DataFrame({"time": [stamp(0), stamp(1)], "serial": ["83", "83"]})
    _validate_frame(
        good,
        timestamp_col="time",
        serial_col="serial",
        serial="83",
        start=start,
        end=end,
    )

    unsorted = pd.DataFrame({"time": [stamp(1), stamp(0)]})
    with pytest.raises(ValueError, match="monotonically"):
        _validate_frame(
            unsorted,
            timestamp_col="time",
            serial_col="serial",
            serial="83",
            start=start,
            end=end,
        )

    duplicated = pd.DataFrame({"time": [stamp(0), stamp(0)]})
    with pytest.raises(ValueError, match="Duplicate timestamps"):
        _validate_frame(
            duplicated,
            timestamp_col="time",
            serial_col="serial",
            serial="83",
            start=start,
            end=end,
        )

    wrong_serial = pd.DataFrame({"time": [stamp(0)], "serial": ["999"]})
    with pytest.raises(ValueError, match="Serial mismatch"):
        _validate_frame(
            wrong_serial,
            timestamp_col="time",
            serial_col="serial",
            serial="83",
            start=start,
            end=end,
        )

    out_of_range = pd.DataFrame(
        {"time": [pd.Timestamp(datetime(2026, 7, 3, tzinfo=UTC))]}
    )
    with pytest.raises(ValueError, match="outside the requested range"):
        _validate_frame(
            out_of_range,
            timestamp_col="time",
            serial_col="serial",
            serial="83",
            start=start,
            end=end,
        )


def test_five_minute_query_has_a_plain_postgres_form() -> None:
    contract = load_import_contract(
        CATALOG,
        chemistry="NCA",
        resolution_values=[_parse_resolution("soh=5m")],
    )
    query = _five_minute_query(
        table="public.battery_telemetry",
        timestamp_col="time",
        serial_col="serial",
        fields=contract.fields_for("5m"),
        limit=None,
        timescale=False,
    )
    rendered = str(query)

    assert "time_bucket" not in rendered
    assert "last(" not in rendered
    assert "extract(epoch FROM" in rendered
    assert "array_agg" in rendered
    assert "AVG" in rendered


def test_has_timescaledb_reads_pg_extension() -> None:
    from src.extract_configurable_telemetry import _has_timescaledb

    class Cursor:
        def __init__(self, row):
            self.row = row
            self.sql = ""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement):
            self.sql = statement

        def fetchone(self):
            return self.row

    class Connection:
        def __init__(self, row):
            self.row = row

        def cursor(self):
            return Cursor(self.row)

    assert _has_timescaledb(Connection((1,))) is True
    assert _has_timescaledb(Connection(None)) is False
