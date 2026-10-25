from __future__ import annotations

import json
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from battryml.catalog.database import CatalogDatabase
from battryml.catalog.repository import CatalogRepository
from battryml.config import Settings
from battryml.data import devices as devices_module
from battryml.data.devices import DeviceLibraryService


def test_unconfigured_import_serial_is_visible_in_selected_chemistry(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    repository.create_job(
        {
            "job_id": "lfp-import",
            "pipeline_key": "veera.raw_telemetry_import.v1",
            "serial": "100000042",
            "status": "queued",
            "request_json": json.dumps({"serial": "100000042", "chemistry": "LFP"}),
            "created_at": datetime.now(UTC),
        }
    )

    item = DeviceLibraryService(platform_settings, repository).get_device("100000042")

    assert item is not None
    assert item["chemistry"] == "LFP"
    assert item["device_role"] == "unassigned"
    assert item["display_name"] == "Battery 100000042"


def test_device_display_name_override_persists_and_can_be_restored(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    service = DeviceLibraryService(platform_settings, repository)

    configured = service.get_device("100000001")
    assert configured is not None
    assert configured["display_name"] == "NCA Field 01"
    assert configured["display_name_source"] == "profile"

    updated = service.update_display_name("100000001", "  Cellar   Alpha  ")
    assert updated is not None
    assert updated["display_name"] == "Cellar Alpha"
    assert updated["display_name_source"] == "custom"

    reloaded = DeviceLibraryService(platform_settings, repository).get_device("100000001")
    assert reloaded is not None
    assert reloaded["display_name"] == "Cellar Alpha"

    restored = service.update_display_name("100000001", None)
    assert restored is not None
    assert restored["display_name"] == "NCA Field 01"
    assert restored["display_name_source"] == "profile"


def test_device_display_names_must_be_unique(platform_settings: Settings) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    service = DeviceLibraryService(platform_settings, CatalogRepository(database))

    try:
        service.update_display_name("100000001", "NCA Field 02")
    except ValueError as exc:
        assert "already used by serial 100000002" in str(exc)
    else:
        raise AssertionError("Duplicate display name should be rejected")


def test_module_count_above_two_derives_system_capacity(platform_settings: Settings) -> None:
    metadata = (
        platform_settings.source_root / "data" / "processed" / "device_metadata"
    )
    metadata.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "serial": ["100000002"],
                "module_count": [3],
                "observed_module_counts": ["3"],
                "module_count_sample_rows": [1000],
                "module_count_agreement": [1.0],
                "module_count_status": ["verified"],
                "imported_at": [datetime.now(UTC)],
            }
        ),
        metadata / "module_counts.parquet",
    )
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    service = DeviceLibraryService(platform_settings, CatalogRepository(database))

    item = service.get_device("100000002")

    assert item is not None
    assert item["module_count"] == 3
    assert item["module_count_status"] == "verified"
    assert item["nominal_capacity_ah"] == 96.0
    assert item["system_energy_kwh"] == 21.0


def test_unchanged_device_refresh_reuses_persisted_metadata(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    service = DeviceLibraryService(platform_settings, repository)
    original = devices_module._parquet_inventory
    calls = 0

    def counting_inventory(files):
        nonlocal calls
        calls += 1
        return original(files)

    monkeypatch.setattr(devices_module, "_parquet_inventory", counting_inventory)
    service.list_devices()
    assert calls > 0

    calls = 0
    service.list_devices(refresh=True)
    assert calls == 0


def test_device_inventory_recommends_first_unstored_timestamp(
    platform_settings: Settings,
) -> None:
    raw_dir = platform_settings.source_root / "data" / "raw_parquet" / "serial=100000001"
    compact_dir = platform_settings.source_root / "data" / "telemetry_5min" / "serial=100000001"
    raw_dir.mkdir(parents=True)
    compact_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"time": [datetime(2026, 7, 17, 10, 4, 59, tzinfo=UTC)]}),
        raw_dir / "part-00000.parquet",
    )
    pq.write_table(
        pa.table({"time": [datetime(2026, 7, 17, 10, 0, 0, tzinfo=UTC)]}),
        compact_dir / "part-00000.parquet",
    )
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()

    item = DeviceLibraryService(platform_settings, CatalogRepository(database)).get_device(
        "100000001"
    )

    assert item is not None
    assert item["recommended_import_start_at"] == "2026-07-17T10:05:00+00:00"


def test_device_inventory_discovers_chemistry_partitioned_lfp_data(
    platform_settings: Settings,
) -> None:
    raw_dir = (
        platform_settings.source_root
        / "data"
        / "raw_parquet"
        / "chemistry=LFP"
        / "serial=100000020"
    )
    raw_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"time": [datetime(2026, 7, 20, 10, 0, 0, tzinfo=UTC)]}),
        raw_dir / "part-00000.parquet",
    )
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()

    item = DeviceLibraryService(platform_settings, CatalogRepository(database)).get_device(
        "100000020"
    )

    assert item is not None
    assert item["chemistry"] == "LFP"
    assert item["raw_file_count"] == 1
    assert item["recommended_import_start_at"] == "2026-07-20T10:00:01+00:00"


def test_single_device_refresh_does_not_refresh_every_serial(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    repository = CatalogRepository(database)
    service = DeviceLibraryService(platform_settings, repository)
    service.list_devices()
    original = service._refresh_inventory
    refreshed: list[str] = []

    def counting_refresh(serial: str, *, force: bool):
        refreshed.append(serial)
        return original(serial, force=force)

    monkeypatch.setattr(service, "_refresh_inventory", counting_refresh)
    item = service.get_device("100000001", refresh=True)

    assert item is not None
    assert refreshed == ["100000001"]


def test_cached_inventory_detects_new_soh_audit_artifact(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    service = DeviceLibraryService(platform_settings, CatalogRepository(database))
    service.list_devices()

    audit_run = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "fleet_soh_audit"
        / "serial=100000001"
        / "audit_run=20260727T120000Z"
    )
    audit_run.mkdir(parents=True)
    pq.write_table(
        pa.table({"time": [datetime.now(UTC)], "soh_pct": [98.0]}),
        audit_run / "daily_health.parquet",
    )

    item = service.get_device("100000001")

    assert item is not None
    assert item["readiness"]["soh_signal_audit"] is False
    assert item["workflow"]["soh_signal_audit"]["status"] == "blocked"
    assert item["workflow"]["soh_signal_audit"]["output_modified_at"] is not None


def test_cached_inventory_detects_hierarchical_prerequisite_artifacts(
    platform_settings: Settings,
) -> None:
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    service = DeviceLibraryService(platform_settings, CatalogRepository(database))
    service.list_devices()

    audit_run = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "bms_electrothermal_source_quality"
        / "serial=100000001"
        / "source_audit_run=20260727T120000Z"
    )
    audit_run.mkdir(parents=True)
    (audit_run / "model_summary.json").write_text(
        '{"source_quality_ready": true}',
        encoding="utf-8",
    )
    feature_run = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "bms_hierarchical_features"
        / "serial=100000001"
        / "feature_run=20260727T121000Z"
    )
    feature_run.mkdir(parents=True)
    pq.write_table(
        pa.table({"window_start_utc": [datetime.now(UTC)]}),
        feature_run / "operating_context_5min.parquet",
    )

    item = service.get_device("100000001")

    assert item is not None
    assert item["readiness"]["electrothermal_source_audit"] is False
    # A hierarchical feature file with no separate validation run is accepted
    # by the legacy electrothermal bridge, so this stage is current.
    assert item["readiness"]["hierarchical_features"] is True
    assert item["workflow"]["electrothermal_source_audit"]["status"] == "blocked"
    assert item["workflow"]["hierarchical_features"]["status"] == "current"


def test_device_inventory_detects_seasonal_charge_profile(
    platform_settings: Settings,
) -> None:
    run_dir = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "seasonal_charge_analysis"
        / "serial=100000001"
        / "profile_run=20260727T130000Z"
    )
    run_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"day_utc": [datetime.now(UTC)], "charge_ah": [10.0]}),
        run_dir / "daily_battery_stress.parquet",
    )
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    service = DeviceLibraryService(platform_settings, CatalogRepository(database))

    item = service.get_device("100000001")

    assert item is not None
    assert item["readiness"]["seasonal_charge_profile"] is False
    assert item["workflow"]["seasonal_charge_profile"]["status"] == "blocked"


def test_device_inventory_detects_bms_soh_recalibration_analysis(
    platform_settings: Settings,
) -> None:
    run_dir = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "bms_soh_recalibration"
        / "serial=100000001"
        / "recalibration_run=20260807T120000Z"
    )
    run_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"event_id": ["recalibration_00001"]}),
        run_dir / "recalibration_events.parquet",
    )
    database = CatalogDatabase(platform_settings.catalog_path)
    database.initialize()
    service = DeviceLibraryService(platform_settings, CatalogRepository(database))

    item = service.get_device("100000001")

    assert item is not None
    assert item["readiness"]["soh_recalibration"] is False
    assert item["workflow"]["soh_recalibration"]["status"] == "blocked"
