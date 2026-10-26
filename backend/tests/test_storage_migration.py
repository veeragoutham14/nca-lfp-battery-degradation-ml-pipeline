from __future__ import annotations

import json
from pathlib import Path

import pytest

from battryml.data.storage_migration import migrate_serial_storage


def test_migration_moves_families_and_rewrites_manifest_paths(tmp_path: Path) -> None:
    source_root = tmp_path / "Veera_SOH"
    raw = source_root / "data" / "raw_parquet" / "serial=100000020"
    compact = source_root / "data" / "telemetry_5min" / "serial=100000020"
    manifests = source_root / "data" / "import_manifests" / "serial=100000020"
    raw.mkdir(parents=True)
    compact.mkdir(parents=True)
    run = manifests / "import_run=20260807T120000Z"
    run.mkdir(parents=True)
    raw_file = raw / "part-00000.parquet"
    compact_file = compact / "part-00000.parquet"
    raw_file.write_bytes(b"raw")
    compact_file.write_bytes(b"compact")
    manifest = run / "import_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "results": [
                    {"resolution": "1s", "files": [str(raw_file)]},
                    {"resolution": "5m", "files": [str(compact_file)]},
                ]
            }
        ),
        encoding="utf-8",
    )

    migrate_serial_storage(
        source_root,
        serial="100000020",
        chemistry="LFP",
        apply=True,
    )

    migrated_raw = (
        source_root
        / "data"
        / "raw_parquet"
        / "chemistry=LFP"
        / "serial=100000020"
        / raw_file.name
    )
    migrated_manifest = (
        source_root
        / "data"
        / "import_manifests"
        / "chemistry=LFP"
        / "serial=100000020"
        / run.name
        / manifest.name
    )
    payload = json.loads(migrated_manifest.read_text(encoding="utf-8"))
    assert migrated_raw.read_bytes() == b"raw"
    assert payload["results"][0]["files"] == [str(migrated_raw)]
    assert not raw.exists()


def test_migration_dry_run_does_not_move_data(tmp_path: Path) -> None:
    source_root = tmp_path / "Veera_SOH"
    raw = source_root / "data" / "raw_parquet" / "serial=100000020"
    raw.mkdir(parents=True)
    (raw / "part-00000.parquet").write_bytes(b"raw")

    plans = migrate_serial_storage(
        source_root,
        serial="100000020",
        chemistry="LFP",
        apply=False,
    )

    assert len(plans) == 1
    assert raw.exists()


def test_migration_rejects_existing_destination(tmp_path: Path) -> None:
    source_root = tmp_path / "Veera_SOH"
    legacy = source_root / "data" / "raw_parquet" / "serial=100000020"
    canonical = (
        source_root
        / "data"
        / "raw_parquet"
        / "chemistry=LFP"
        / "serial=100000020"
    )
    legacy.mkdir(parents=True)
    canonical.mkdir(parents=True)

    with pytest.raises(ValueError, match="destination already exists"):
        migrate_serial_storage(
            source_root,
            serial="100000020",
            chemistry="LFP",
            apply=True,
        )
