from __future__ import annotations

import os
from pathlib import Path

from battryml.data.devices import _workflow_stage_states


def _artifact(root: Path, stage_folder: str, serial: str, mtime_ns: int) -> None:
    path = root / "processed" / stage_folder / f"serial={serial}" / "run=1"
    path.mkdir(parents=True, exist_ok=True)
    artifact = path / "result.parquet"
    artifact.write_bytes(b"test")
    os.utime(artifact, ns=(mtime_ns, mtime_ns))


def test_new_raw_watermark_stales_direct_stage_and_blocks_descendant(tmp_path: Path) -> None:
    serial = "TEST001"
    _artifact(tmp_path, "classified_rows", serial, 100)
    _artifact(tmp_path, "events", serial, 150)

    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=200,
        electrothermal_quality_ready=False,
        capacity_bms_validation_ready=False,
    )

    assert states["classified"]["status"] == "stale"
    assert states["events"]["status"] == "blocked"


def test_updating_each_stage_unlocks_next_stage_in_order(tmp_path: Path) -> None:
    serial = "TEST001"
    _artifact(tmp_path, "classified_rows", serial, 250)
    _artifact(tmp_path, "events", serial, 150)

    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=200,
        electrothermal_quality_ready=False,
        capacity_bms_validation_ready=False,
    )
    assert states["classified"]["status"] == "current"
    assert states["events"]["status"] == "stale"

    _artifact(tmp_path, "events", serial, 300)
    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=200,
        electrothermal_quality_ready=False,
        capacity_bms_validation_ready=False,
    )
    assert states["events"]["status"] == "current"
    assert states["capacity_trend"]["status"] == "ready"
    assert states["fixed_slope_verification"]["status"] == "ready"


def test_electrothermal_features_require_fresh_passed_local_validation(tmp_path: Path) -> None:
    serial = "TEST001"
    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=100,
        electrothermal_quality_ready=False,
        capacity_bms_validation_ready=False,
        electrothermal_validation_ready=False,
        electrothermal_source_mtime_ns=200,
    )
    assert states["electrothermal_validation"]["status"] == "ready"
    assert states["hierarchical_features"]["status"] == "blocked"

    _artifact(tmp_path, "electrothermal_import_validation", serial, 250)
    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=100,
        electrothermal_quality_ready=False,
        capacity_bms_validation_ready=False,
        electrothermal_validation_ready=True,
        electrothermal_source_mtime_ns=200,
    )
    assert states["electrothermal_validation"]["status"] == "current"
    assert states["hierarchical_features"]["status"] == "ready"

    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=100,
        electrothermal_quality_ready=False,
        capacity_bms_validation_ready=False,
        electrothermal_validation_ready=True,
        electrothermal_source_mtime_ns=300,
    )
    assert states["electrothermal_validation"]["status"] == "stale"
    assert states["hierarchical_features"]["status"] == "blocked"


def test_phase2_requires_current_purpose_specific_data_validation(tmp_path: Path) -> None:
    serial = "TEST001"
    _artifact(tmp_path, "classified_rows", serial, 205)
    _artifact(tmp_path, "events", serial, 210)
    _artifact(tmp_path, "bms_electrothermal_source_quality", serial, 210)
    _artifact(tmp_path, "electrothermal_import_validation", serial, 215)
    _artifact(tmp_path, "bms_hierarchical_features", serial, 220)
    _artifact(tmp_path, "fleet_soh_audit", serial, 230)
    validation = (
        tmp_path
        / "processed"
        / "data_validation"
        / "chemistry=NCA"
        / f"serial={serial}"
        / "validation_run=1"
    )
    validation.mkdir(parents=True)
    artifact = validation / "result.parquet"
    artifact.write_bytes(b"test")
    os.utime(artifact, ns=(240, 240))
    _artifact(tmp_path, "discharge_capacity_evidence", serial, 250)

    states = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=200,
        electrothermal_quality_ready=True,
        electrothermal_validation_ready=True,
        capacity_bms_validation_ready=True,
    )
    assert states["data_validation"]["status"] == "current"
    assert states["capacity_soh_investigation"]["status"] == "ready"

    failed = _workflow_stage_states(
        tmp_path,
        serial,
        raw_ready=True,
        raw_mtime_ns=200,
        electrothermal_quality_ready=True,
        electrothermal_validation_ready=True,
        capacity_bms_validation_ready=False,
    )
    assert failed["data_validation"]["status"] == "failed"
    assert failed["capacity_soh_investigation"]["status"] == "blocked"
