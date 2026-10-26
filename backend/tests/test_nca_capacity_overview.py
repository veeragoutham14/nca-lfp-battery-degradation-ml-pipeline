from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from battryml.analytics.nca_capacity_overview import build_nca_capacity_overview
from battryml.battery_profiles import battery_profile

FIELD_NOMINAL_AH = battery_profile("100000001").nominal_capacity_ah
FIELD_MODULE_KWH = battery_profile("100000001").module_energy_kwh


def _verification_run(root: Path, serial: str) -> None:
    run_dir = (
        root
        / "data"
        / "processed"
        / "fixed_slope_verification"
        / f"serial={serial}"
        / "verification_run=20260701T120000Z"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_summary.json").write_text(
        json.dumps(
            {
                "model_family": "fixed_slope_transfer_verification",
                "serial": serial,
                "nominal_capacity_ah": FIELD_NOMINAL_AH,
                "data_quality": {"telemetry_coverage_pct": 96.0},
            }
        ),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "end_anchor_time": [
                    datetime(2025, 1, 1, tzinfo=UTC),
                    datetime(2026, 1, 1, tzinfo=UTC),
                ],
                "bms_soh_end": [100.0, 98.0],
                "cumulative_all_discharge_ah": [100.0, 2_100.0],
            }
        ),
        run_dir / "fixed_slope_all_capacity_rows.parquet",
    )


def _device(
    serial: str,
    *,
    chemistry: str = "NCA",
    role: str = "field_tester",
    module_count: int = 2,
):
    return {
        "serial": serial,
        "variant_key": "test_variant",
        "chemistry": chemistry,
        "device_role": role,
        "nominal_capacity_ah": FIELD_NOMINAL_AH if chemistry == "NCA" else None,
        "module_count": module_count,
        "module_count_status": "verified",
        "system_energy_kwh": module_count * FIELD_MODULE_KWH if chemistry == "NCA" else None,
    }


def test_capacity_overview_uses_actual_throughput_and_reference_slope(
    tmp_path: Path,
) -> None:
    _verification_run(tmp_path, "100000001")

    overview = build_nca_capacity_overview(
        source_root=tmp_path,
        devices=[_device("100000001")],
        reference_loss_slope_ah_per_1000ah=0.1,
    )

    row = overview["devices"][0]
    nominal = FIELD_NOMINAL_AH
    cumulative = 2_100.0
    reference = nominal - 0.1 * cumulative / 1000.0
    assert row["cumulative_discharge_ah"] == cumulative
    assert row["equivalent_full_cycles"] == cumulative / nominal
    assert row["reference_capacity_ah"] == reference
    assert row["reference_soh_pct"] == 100.0 * reference / nominal
    assert round(row["bms_minus_reference_soh_pct"], 2) == round(98.0 - row["reference_soh_pct"], 2)
    assert row["bms_equivalent_loss_slope_ah_per_1000ah"] == (2.0 * nominal / 100.0) / 2.0
    assert overview["metrics"]["devices_above_twice_reference_loss_rate"] == 1
    assert overview["evidence_policy"]["bms_soh_role"] == "weak_auxiliary_label"


def test_capacity_overview_separates_nca_and_reports_missing_runs(
    tmp_path: Path,
) -> None:
    _verification_run(tmp_path, "100000001")

    overview = build_nca_capacity_overview(
        source_root=tmp_path,
        devices=[
            _device("100000001"),
            _device("100000002", module_count=3),
            _device("100000003"),
            _device("LFP001", chemistry="LFP"),
            _device("100000017", role="full_cycle_test"),
        ],
    )

    assert overview["metrics"]["configured_device_count"] == 3
    assert overview["metrics"]["inventory_device_count"] == 3
    assert overview["metrics"]["deferred_device_count"] == 0
    assert overview["metrics"]["available_device_count"] == 1
    assert overview["metrics"]["missing_device_count"] == 2
    assert overview["cohort"]["module_count"] is None
    assert {row["serial"] for row in overview["missing_devices"]} == {
        "100000002",
        "100000003",
    }
    assert overview["deferred_devices"] == []
