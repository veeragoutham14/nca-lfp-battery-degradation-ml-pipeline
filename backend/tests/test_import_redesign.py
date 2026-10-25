from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from battryml.config import Settings
from battryml.data.import_planning import (
    build_import_plan,
    core_dataset_root,
    count_rows_and_months,
    custom_dataset_root,
    dataset_root,
    estimated_rows_per_month,
    local_time_range,
    planned_start,
    serial_scan_dirs,
)
from battryml.data.signal_registry import (
    CAPACITY_CORE_PURPOSE,
    CORE_PRESET,
    CUSTOM_PURPOSE,
    ELECTROTHERMAL_HISTORY_PURPOSE,
    expected_column_count,
    validate_custom_selection,
)


def _write_parquet(path: Path, *, start: datetime, seconds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [start + timedelta(seconds=i) for i in range(seconds)]
    table = pa.table(
        {
            "time": pa.array(times, type=pa.timestamp("us", tz="UTC")),
            "value": [float(i) for i in range(seconds)],
        }
    )
    pq.write_table(table, path)


def test_capacity_core_preset_is_fixed_and_complete() -> None:
    assert {signal.key for signal in CORE_PRESET} == {
        "time",
        "serial",
        "soc",
        "soh",
        "current",
        "voltage",
    }
    # Temperature, cell voltage, and cycle count are excluded by design.
    assert {"temperature", "cell_voltage", "cycle_count"}.isdisjoint(
        {signal.key for signal in CORE_PRESET}
    )


def test_electrothermal_plan_is_fixed_to_two_five_minute_groups(tmp_path: Path) -> None:
    plan = build_import_plan(
        purpose=ELECTROTHERMAL_HISTORY_PURPOSE,
        selections={},
        source_data=tmp_path,
        serial="100000001",
        device_start=datetime(2023, 8, 1, tzinfo=UTC),
        chemistry="NCA",
        module_count=2,
    )
    assert {(item["group"], item["resolution"]) for item in plan["signals"]} == {
        ("temperature", "5m"),
        ("cell_voltage", "5m"),
    }
    assert all(item["chemistry"] == "NCA" for item in plan["signals"])


def test_expected_column_counts_follow_topology() -> None:
    assert expected_column_count("temperature", 2) == 16
    assert expected_column_count("cell_voltage", 2) == 56
    assert expected_column_count("temperature", 4) == 32
    assert expected_column_count("cell_voltage", 3) == 84
    assert expected_column_count("temperature", None) is None


def test_validate_custom_selection_normalizes_and_rejects() -> None:
    selections = validate_custom_selection(
        {"temperature": "5m", "cycle_count": "off", "soh": "both"}
    )
    assert selections == {"temperature": "5m", "soh": "both"}
    with pytest.raises(ValueError, match="Unknown telemetry group"):
        validate_custom_selection({"made_up": "1s"})
    with pytest.raises(ValueError, match="resolution must be one of"):
        validate_custom_selection({"soc": "30s"})
    with pytest.raises(ValueError, match="Select at least one"):
        validate_custom_selection({"soc": "off"})


def test_planned_start_rules() -> None:
    watermark = datetime(2026, 7, 17, 11, 1, tzinfo=UTC)
    device_start = datetime(2023, 8, 1, tzinfo=UTC)

    planned, rule = planned_start(
        group_key="core",
        resolution="1s",
        watermark_end=watermark,
        device_start=device_start,
    )
    assert rule == "continue_after_watermark"
    assert planned == datetime(2026, 7, 17, 11, 1, 1, tzinfo=UTC)

    planned, rule = planned_start(
        group_key="temperature",
        resolution="5m",
        watermark_end=None,
        device_start=device_start,
    )
    assert (rule, planned) == ("device_configuration_start", device_start)

    planned, rule = planned_start(
        group_key="temperature",
        resolution="5m",
        watermark_end=None,
        device_start=None,
    )
    assert (rule, planned) == ("database_first_timestamp", None)

    planned, rule = planned_start(
        group_key="temperature",
        resolution="5m",
        watermark_end=watermark,
        device_start=device_start,
    )
    assert rule == "continue_after_watermark"
    assert planned == datetime(2026, 7, 17, 11, 6, tzinfo=UTC)


def test_dataset_roots_keep_core_legacy_and_group_custom(tmp_path: Path) -> None:
    source_data = tmp_path / "data"
    assert core_dataset_root(source_data) == source_data / "raw_parquet"
    assert dataset_root(source_data, "core", "1s") == source_data / "raw_parquet"
    assert custom_dataset_root(source_data, "temperature", "5m", "LFP") == (
        source_data
        / "custom_telemetry"
        / "temperature"
        / "5m"
        / "chemistry=LFP"
    )
    # Custom roots are chemistry-scoped so per-device watermarks cannot see a
    # sibling fleet's data.
    assert custom_dataset_root(
        source_data, "temperature", "5m", "NCA"
    ) != custom_dataset_root(source_data, "temperature", "5m", "LFP")


def test_serial_scan_dirs_cover_both_core_layouts(tmp_path: Path) -> None:
    source_data = tmp_path / "data"
    core_dirs = serial_scan_dirs(
        source_data,
        group="core",
        resolution="1s",
        chemistry="NCA",
        serial="100000001",
    )
    assert core_dirs == [
        source_data / "raw_parquet" / "chemistry=NCA" / "serial=100000001",
        source_data / "raw_parquet" / "serial=100000001",
    ]
    custom_dirs = serial_scan_dirs(
        source_data,
        group="cycle_count",
        resolution="5m",
        chemistry="LFP",
        serial="100000020",
    )
    assert custom_dirs == [
        source_data
        / "custom_telemetry"
        / "cycle_count"
        / "5m"
        / "chemistry=LFP"
        / "serial=100000020"
    ]


def test_local_range_is_scoped_to_requested_serial(tmp_path: Path) -> None:
    source_data = tmp_path / "data"
    own = serial_scan_dirs(
        source_data,
        group="cycle_count",
        resolution="5m",
        chemistry="LFP",
        serial="100000020",
    )[0]
    other = serial_scan_dirs(
        source_data,
        group="cycle_count",
        resolution="5m",
        chemistry="LFP",
        serial="100000021",
    )[0]
    _write_parquet(
        own / "year=2026" / "month=07" / "part-00000.parquet",
        start=datetime(2026, 7, 1, tzinfo=UTC),
        seconds=10,
    )
    # A NEWER row belonging to another device must not move this watermark.
    _write_parquet(
        other / "year=2026" / "month=09" / "part-00000.parquet",
        start=datetime(2026, 9, 1, tzinfo=UTC),
        seconds=10,
    )

    start, end = local_time_range([own])
    assert start == datetime(2026, 7, 1, tzinfo=UTC)
    assert end == datetime(2026, 7, 1, 0, 0, 9, tzinfo=UTC)
    # Sanity: the other device's data exists and would have poisoned an
    # unscoped scan.
    _, other_end = local_time_range([other])
    assert other_end == datetime(2026, 9, 1, 0, 0, 9, tzinfo=UTC)
    assert count_rows_and_months([own]) == (10, 1)
    assert estimated_rows_per_month([own]) == 10


def test_build_import_plan_core_uses_fixed_resolutions(tmp_path: Path) -> None:
    plan = build_import_plan(
        purpose=CAPACITY_CORE_PURPOSE,
        selections={},
        source_data=tmp_path,
        serial="100000001",
        device_start=datetime(2023, 8, 1, tzinfo=UTC),
        chemistry="NCA",
        module_count=2,
    )
    assert plan["purpose"] == CAPACITY_CORE_PURPOSE
    assert [(s["group"], s["resolution"]) for s in plan["signals"]] == [("core", "1s")]
    signal = plan["signals"][0]
    assert signal["planned_rule"] == "device_configuration_start"
    assert signal["planned_start"] == "2023-08-01T00:00:00+00:00"
    assert signal["planned_end_open"] is True


def test_build_import_plan_custom_groups_split_resolutions(tmp_path: Path) -> None:
    plan = build_import_plan(
        purpose=CUSTOM_PURPOSE,
        selections={"temperature": "both", "cycle_count": "5m"},
        source_data=tmp_path,
        serial="100000020",
        device_start=None,
        chemistry="LFP",
        module_count=2,
    )
    entries = [(s["group"], s["resolution"]) for s in plan["signals"]]
    assert entries == [
        ("temperature", "1s"),
        ("temperature", "5m"),
        ("cycle_count", "5m"),
    ]
    temperature = plan["signals"][0]
    assert temperature["requires_module_topology"] is True
    assert temperature["column_count"] == 16
    assert temperature["topology_resolved"] is True


def test_v2_plugin_rejects_custom_signals_for_core_purpose(
    platform_settings: Settings,
) -> None:
    from battryml.plugins.veera_ingestion import RawTelemetryImportV2Plugin

    with pytest.raises(ValueError, match="preset is fixed"):
        RawTelemetryImportV2Plugin().build_command(
            platform_settings,
            {
                "serial": "100000020",
                "chemistry": "LFP",
                "purpose": CAPACITY_CORE_PURPOSE,
                "temperature_resolution": "5m",
            },
        )


def test_v2_plugin_requires_selection_for_custom_purpose(
    platform_settings: Settings,
) -> None:
    from battryml.plugins.veera_ingestion import RawTelemetryImportV2Plugin

    with pytest.raises(ValueError, match="Select at least one telemetry group"):
        RawTelemetryImportV2Plugin().build_command(
            platform_settings,
            {"serial": "100000020", "chemistry": "LFP", "purpose": CUSTOM_PURPOSE},
        )


def test_v2_plugin_core_command_targets_legacy_layout(
    platform_settings: Settings,
) -> None:
    from battryml.plugins.veera_ingestion import RawTelemetryImportV2Plugin

    command = RawTelemetryImportV2Plugin().build_command(
        platform_settings,
        {
            "serial": "100000020",
            "chemistry": "LFP",
            "purpose": CAPACITY_CORE_PURPOSE,
        },
    )
    assert "--purpose" in command and "capacity_core" in command
    output_dir = Path(command[command.index("--output-dir") + 1])
    assert output_dir.name == "chemistry=LFP"
    assert output_dir.parent.name == "raw_parquet"
    assert "--group-output-root" not in command
    field_resolutions = [
        command[i + 1] for i, flag in enumerate(command) if flag == "--field-resolution"
    ]
    assert sorted(field_resolutions) == ["current=1s", "soc=1s", "soh=1s", "voltage=1s"]
    assert command[command.index("--start") + 1] == "2024-11-01T00:00:00+00:00"


def test_v2_plugin_custom_command_writes_dedicated_group_roots(
    platform_settings: Settings,
) -> None:
    from battryml.plugins.veera_ingestion import RawTelemetryImportV2Plugin

    command = RawTelemetryImportV2Plugin().build_command(
        platform_settings,
        {
            "serial": "100000020",
            "chemistry": "LFP",
            "purpose": CUSTOM_PURPOSE,
            "module_count": 2,
            "temperature_resolution": "both",
            "cycle_count_resolution": "5m",
        },
    )
    assert command[command.index("--purpose") + 1] == CUSTOM_PURPOSE
    assert command[command.index("--module-count") + 1] == "2"
    assert command[command.index("--start") + 1] == "2024-11-01T00:00:00+00:00"
    group_roots = [
        command[i + 1]
        for i, flag in enumerate(command)
        if flag == "--group-output-root"
    ]
    mapped = {}
    for entry in group_roots:
        group, resolution, path = entry.split("=", 2)
        mapped[(group, resolution)] = Path(path)
    assert set(mapped) == {
        ("temperature", "1s"),
        ("temperature", "5m"),
        ("cycle_count", "5m"),
    }
    for (group, resolution), path in mapped.items():
        parts = path.parts
        assert group in parts and resolution in parts
        assert "custom_telemetry" in parts
        assert "chemistry=LFP" in parts


def test_v2_plugin_requires_module_count_for_topology_groups(
    platform_settings: Settings,
) -> None:
    from battryml.plugins.veera_ingestion import RawTelemetryImportV2Plugin

    with pytest.raises(ValueError, match="module_count is required"):
        RawTelemetryImportV2Plugin().build_command(
            platform_settings,
            {
                "serial": "100000020",
                "chemistry": "LFP",
                "purpose": CUSTOM_PURPOSE,
                "cell_voltage_resolution": "5m",
            },
        )
    with pytest.raises(ValueError, match="between 1 and 4"):
        RawTelemetryImportV2Plugin().build_command(
            platform_settings,
            {
                "serial": "100000020",
                "chemistry": "LFP",
                "purpose": CUSTOM_PURPOSE,
                "module_count": 7,
                "cell_voltage_resolution": "5m",
            },
        )
