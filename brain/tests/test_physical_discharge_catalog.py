from __future__ import annotations

import pandas as pd

from src.device_specs import FIELD_NOMINAL_AH, LAB_NOMINAL_AH
from src.physical_discharge_catalog import (
    _cached_catalog,
    _source_identity,
    build_physical_discharge_catalog,
)


def _event(
    event_id: str,
    mode: str,
    start: str,
    end: str,
    start_soc: float,
    end_soc: float,
    ah: float = 0.0,
) -> dict[str, object]:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    return {
        "event_id": event_id,
        "serial": "field",
        "mode": mode,
        "start_timestamp": start_time,
        "end_timestamp": end_time,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "start_soc": start_soc,
        "end_soc": end_soc,
        "start_soh": 95.0,
        "end_soh": 95.0,
        "throughput_ah": ah,
        "energy_wh": ah * 200.0,
        "_event_row_id": 0,
    }


def test_catalog_preserves_rest_and_emits_one_nested_exact_cycle() -> None:
    rows = [
        _event("rest_at_full", "rest", "2026-01-01T00:00Z", "2026-01-01T01:00Z", 100, 100),
        _event("d1", "discharging", "2026-01-01T01:00Z", "2026-01-01T03:00Z", 100, 50, 24.5),
        _event("rest_mid", "rest", "2026-01-01T03:00Z", "2026-01-01T04:00Z", 50, 50),
        _event("d2", "discharging", "2026-01-01T04:00Z", "2026-01-01T06:00Z", 50, 0, 24.0),
    ]
    events = pd.DataFrame(rows)
    events["_event_row_id"] = range(len(events))

    catalog = build_physical_discharge_catalog(
        events,
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )

    assert len(catalog) == 1
    row = catalog.iloc[0]
    assert row["evidence_class"] == "strict_full_discharge"
    assert bool(row["observed_100_to_0"])
    assert bool(row["observed_99_to_1"])
    assert bool(row["observed_95_to_5"])
    assert row["rest_duration_s"] == 7200.0
    assert row["discharged_ah"] == 48.5


def test_catalog_keeps_95_to_5_as_one_supporting_physical_cycle() -> None:
    rows = [
        _event("d1", "discharging", "2026-01-01T01:00Z", "2026-01-01T02:00Z", 95, 40, 25.0),
        _event("rest", "rest", "2026-01-01T02:00Z", "2026-01-01T03:00Z", 40, 40),
        _event("d2", "discharging", "2026-01-01T03:00Z", "2026-01-01T04:00Z", 40, 5, 16.0),
        _event("charge", "charging", "2026-01-01T04:00Z", "2026-01-01T05:00Z", 5, 20, 0.0),
    ]
    events = pd.DataFrame(rows)
    events["_event_row_id"] = range(len(events))

    catalog = build_physical_discharge_catalog(
        events,
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )

    row = catalog.iloc[0]
    assert row["evidence_class"] == "operational_full_discharge"
    assert bool(row["observed_95_to_5"])
    assert not bool(row["observed_99_to_1"])
    assert not bool(row["observed_100_to_0"])


def test_catalog_retains_sub_one_pp_charge_intrusion_and_uses_net_ah_for_soh() -> None:
    rows = [
        _event("d1", "discharging", "2026-01-01T00:00Z", "2026-01-01T02:00Z", 96, 50, 22.0),
        _event("pulse", "charging", "2026-01-01T02:00Z", "2026-01-01T02:05Z", 50, 50, 0.2),
        _event("rest", "rest", "2026-01-01T02:05Z", "2026-01-01T03:00Z", 50, 50),
        _event("d2", "discharging", "2026-01-01T03:00Z", "2026-01-01T05:00Z", 50, 4, 22.0),
    ]
    events = pd.DataFrame(rows)
    events["_event_row_id"] = range(len(events))

    catalog = build_physical_discharge_catalog(
        events,
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )

    assert len(catalog) == 1
    row = catalog.iloc[0]
    assert row["soc_span_pp"] == 92.0
    assert row["gross_discharged_ah"] == 44.0
    assert row["charge_intrusion_ah"] == 0.2
    assert row["net_discharged_ah"] == 43.8
    assert row["charge_intrusion_count"] == 1
    assert "tolerated_charge_intrusion" in row["continuity"]
    expected_capacity = 43.8 / 0.92
    assert row["estimated_full_capacity_ah"] == expected_capacity
    assert row["calculated_soh_pct"] == expected_capacity / FIELD_NOMINAL_AH * 100.0


def test_catalog_meaningful_one_pp_recharge_splits_physical_discharges() -> None:
    rows = [
        _event("d1", "discharging", "2026-01-01T00:00Z", "2026-01-01T02:00Z", 100, 50, 24.0),
        _event("charge", "charging", "2026-01-01T02:00Z", "2026-01-01T02:30Z", 50, 52, 1.0),
        _event("d2", "discharging", "2026-01-01T02:30Z", "2026-01-01T05:00Z", 52, 0, 25.0),
    ]
    events = pd.DataFrame(rows)
    events["_event_row_id"] = range(len(events))

    catalog = build_physical_discharge_catalog(
        events,
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )

    assert catalog.empty


def test_catalog_keeps_80pp_for_recalibration_but_calculates_soh_only_from_90pp() -> None:
    rows = [
        _event("d80", "discharging", "2026-01-01T00:00Z", "2026-01-01T04:00Z", 90, 10, 38.0),
        _event("charge", "charging", "2026-01-01T04:00Z", "2026-01-01T05:00Z", 10, 20, 4.0),
    ]
    events = pd.DataFrame(rows)
    events["_event_row_id"] = range(len(events))

    catalog = build_physical_discharge_catalog(
        events,
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )

    assert len(catalog) == 1
    row = catalog.iloc[0]
    assert bool(row["meaningful_deep_discharge"])
    assert not bool(row["capacity_estimate_eligible"])
    assert pd.isna(row["estimated_full_capacity_ah"])
    assert pd.isna(row["calculated_soh_pct"])


def test_catalog_cache_identity_changes_only_when_source_or_configuration_changes(
    tmp_path,
) -> None:
    events_root = tmp_path / "events"
    source = events_root / "serial=field" / "month=2026-01" / "events.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first")

    first = _source_identity(
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        events_root=events_root,
    )
    unchanged = _source_identity(
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        events_root=events_root,
    )
    different_capacity = _source_identity(
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=LAB_NOMINAL_AH,
        events_root=events_root,
    )

    assert first == unchanged
    assert first["source_identity_sha256"] != different_capacity["source_identity_sha256"]

    output_root = tmp_path / "catalog"
    run_dir = output_root / "serial=field" / "catalog_run=20260101T000000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "physical_discharge_cycles.parquet").write_bytes(b"catalog")
    (run_dir / "model_summary.json").write_text(
        '{"source_identity_sha256": "' + first["source_identity_sha256"] + '"}',
        encoding="utf-8",
    )
    assert _cached_catalog(output_root, "field", first["source_identity_sha256"]) == run_dir


def test_catalog_without_deep_discharges_writes_an_empty_run(tmp_path, monkeypatch) -> None:
    import json

    from src import physical_discharge_catalog as module

    rows = [
        _event("d1", "discharging", "2026-01-01T01:00Z", "2026-01-01T02:00Z", 80, 60, 12.0),
        _event("c1", "charging", "2026-01-01T02:00Z", "2026-01-01T03:00Z", 60, 80, 12.0),
    ]
    events = pd.DataFrame(rows)
    events["_event_row_id"] = range(len(events))
    catalog = build_physical_discharge_catalog(
        events, serial="field", chemistry="NCA", module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
    )
    assert catalog.empty
    assert "meaningful_deep_discharge" in catalog.columns

    monkeypatch.setattr(module, "load_event_table", lambda root, serial: events)
    monkeypatch.setattr(
        module, "_source_identity", lambda **kwargs: {"source_identity_sha256": "x"}
    )
    run_dir = module.run_catalog(
        serial="field",
        chemistry="NCA",
        module_count=2,
        nominal_capacity_ah=FIELD_NOMINAL_AH,
        events_root=tmp_path / "events",
        output_root=tmp_path / "out",
    )
    summary = json.loads((run_dir / "model_summary.json").read_text())
    assert summary["physical_cycle_rows"] == 0
    assert summary["meaningful_80pp_rows"] == 0
