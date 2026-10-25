from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from battryml.analytics.fleet_soh_audit import _context_metrics, _event_evidence


def _event_file(run_dir: Path, serial: str) -> None:
    run_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "event_time": datetime(2026, 1, 2, tzinfo=UTC),
                    "day_date": date(2026, 1, 2),
                    "event_type": "step_drop",
                    "soh_delta_pct": -1.0,
                    "serial": serial,
                    "audit_run": "source",
                },
                {
                    "event_time": datetime(2026, 1, 3, tzinfo=UTC),
                    "day_date": date(2026, 1, 3),
                    "event_type": "step_recovery",
                    "soh_delta_pct": 1.0,
                    "serial": serial,
                    "audit_run": "source",
                },
            ]
        ),
        run_dir / "soh_change_events.parquet",
    )


def test_event_evidence_finds_cross_device_change_clusters(tmp_path: Path) -> None:
    first = tmp_path / "serial=one" / "audit_run=first"
    second = tmp_path / "serial=two" / "audit_run=second"
    _event_file(first, "one")
    _event_file(second, "two")

    events, daily, monthly = _event_evidence(
        [({"serial": "one"}, first), ({"serial": "two"}, second)]
    )

    assert len(events) == 4
    drop_cluster = next(row for row in daily if row["event_type"] == "step_drop")
    assert drop_cluster["device_count"] == 2
    assert drop_cluster["serials"] == "one, two"
    recovery_month = next(
        row for row in monthly if row["event_type"] == "step_recovery"
    )
    assert recovery_month["device_count"] == 2
    assert recovery_month["event_count"] == 2


def test_context_metrics_measure_full_soc_span() -> None:
    metrics = _context_metrics(
        [
            {
                "min_soc_pct": 5.0,
                "max_soc_pct": 100.0,
                "discharge_ah": 40.0,
                "coverage_fraction": 0.99,
            },
            {
                "min_soc_pct": 30.0,
                "max_soc_pct": 90.0,
                "discharge_ah": 20.0,
                "coverage_fraction": 0.97,
            },
        ]
    )

    assert metrics["day_count"] == 2
    assert metrics["low_soc_day_fraction"] == 0.5
    assert metrics["high_soc_day_fraction"] == 0.5
    assert metrics["full_soc_span_day_fraction"] == 0.5
    assert metrics["median_discharge_ah"] == 30.0
