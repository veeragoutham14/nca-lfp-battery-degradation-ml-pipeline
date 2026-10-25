from __future__ import annotations

from pathlib import Path

import pytest
from battryml.data.storage_paths import (
    discover_serial_chemistries,
    resolve_dataset_location,
)


def test_new_dataset_uses_chemistry_partition(tmp_path: Path) -> None:
    location = resolve_dataset_location(
        tmp_path,
        serial="100000020",
        chemistry="LFP",
        for_write=True,
    )

    assert location.parent_root == tmp_path / "chemistry=LFP"
    assert location.serial_root == tmp_path / "chemistry=LFP" / "serial=100000020"
    assert location.layout == "chemistry_partitioned"


def test_existing_legacy_dataset_remains_authoritative(tmp_path: Path) -> None:
    legacy = tmp_path / "serial=100000001"
    legacy.mkdir()

    location = resolve_dataset_location(
        tmp_path,
        serial="100000001",
        chemistry="NCA",
        for_write=True,
    )

    assert location.serial_root == legacy
    assert location.layout == "legacy"


def test_dual_layout_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "serial=100000001").mkdir()
    (tmp_path / "chemistry=NCA" / "serial=100000001").mkdir(parents=True)

    with pytest.raises(ValueError, match="both legacy and NCA"):
        resolve_dataset_location(
            tmp_path,
            serial="100000001",
            chemistry="NCA",
            for_write=True,
        )


def test_wrong_chemistry_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "chemistry=LFP" / "serial=100000020").mkdir(parents=True)

    with pytest.raises(ValueError, match="cannot be opened as NCA"):
        resolve_dataset_location(
            tmp_path,
            serial="100000020",
            chemistry="NCA",
            for_write=True,
        )


def test_partitioned_serials_report_chemistry(tmp_path: Path) -> None:
    (tmp_path / "chemistry=LFP" / "serial=100000020").mkdir(parents=True)

    assert discover_serial_chemistries(tmp_path) == {"100000020": "LFP"}
