"""End-to-end run of the synthetic demo fleet through the jobs API.

Slow (about 8 minutes on 8 vCPUs, peak memory about 3 GB): it generates the dense NCA and
LFP field devices and the lab reference, then runs every local stage through
POST /api/jobs exactly as the dashboard does. Run it with

    python -m pytest -m slow backend/tests/test_e2e_demo_chain.py

The raw telemetry import is not part of the chain: it reads PostgreSQL, and the
generator writes the same raw Parquet layout instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from battryml.api.app import create_app
from battryml.config import Settings

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
REFERENCE = {
    "serial": "100000017",
    "chemistry": "NCA",
    "module_count": 2,
    "nominal_capacity_ah": 56,
}
NCA_FIELD = {
    "serial": "100000002",
    "chemistry": "NCA",
    "module_count": 2,
    "nominal_capacity_ah": 64,
}
LFP_FIELD = {
    "serial": "100000021",
    "chemistry": "LFP",
    "module_count": 2,
    "nominal_capacity_ah": 50,
}
PREPARE = (
    "veera.classify_rows.v1",
    "veera.build_events.v1",
    # electrothermal import validation, hierarchical features and data validation
    "battryml.prepare_analysis_data.v1",
)
EVIDENCE = (
    "battryml.soh_signal_audit.v1",
    "veera.discharge_capacity_evidence.v1",
    "veera.bms_soh_recalibration.v4",
    # physical discharge catalog, recalibration, capacity/BMS validation, Phase 3 context
    "battryml.capacity_soh_investigation.v1",
    "veera.seasonal_charge_analysis.v1",
)
CHAIN: tuple[tuple[dict[str, Any], tuple[str, ...]], ...] = (
    (REFERENCE, (*PREPARE, "veera.capacity_trend.v1", *EVIDENCE)),
    (NCA_FIELD, (*PREPARE, "veera.hierarchical_voltage_anomaly.v2", *EVIDENCE)),
    (LFP_FIELD, (*PREPARE, "veera.hierarchical_voltage_anomaly.v2", *EVIDENCE)),
)
# The demo reference logs a handful of strict capacity cycles, not the 500 the
# production gate asks for, so the gap diagnosis lowers the gate explicitly.
GAP_DIAGNOSIS = (
    "battryml.soh_gap_diagnosis.v1",
    {**NCA_FIELD, "field_nameplate_capacity_ah": 64, "minimum_reference_strict_cycles": 5},
)


def _source_root(tmp_path: Path) -> Path:
    """A private brain checkout whose data directory lives under tmp_path."""
    source_root = tmp_path / "brain"
    shutil.copytree(
        REPO / "brain" / "src", source_root / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copytree(REPO / "config", tmp_path / "config")
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "generate_synthetic_telemetry.py"),
            "--output",
            str(source_root / "data" / "raw_parquet"),
            "--dense-field-device",
        ],
        check=True,
        capture_output=True,
        timeout=900,
    )
    return source_root


def _run(client: TestClient, key: str, params: dict[str, Any]) -> dict[str, Any]:
    plugin = next(item for item in client.get("/api/pipelines").json() if item["key"] == key)
    accepted = {parameter["key"] for parameter in plugin["parameters"]}
    response = client.post(
        "/api/jobs",
        json={
            "pipeline_key": key,
            "parameters": {name: value for name, value in params.items() if name in accepted},
        },
    )
    assert response.status_code in {200, 202}, f"{key} {params['serial']}: {response.text}"
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 900
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] not in {"queued", "running"}:
            break
        assert time.monotonic() < deadline, f"{key} {params['serial']} timed out"
        time.sleep(2)
    assert job["status"] == "completed", f"{key} {params['serial']}: {job.get('error_message')}"
    return job


def _latest(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    assert matches, f"no output for {root}/{pattern}"
    return matches[-1]


def test_demo_fleet_runs_every_local_stage_through_the_jobs_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = _source_root(tmp_path)
    # Jobs inherit this environment, and the device workflow re-reads its
    # settings from it, so point it at the private copy and keep the checkout's
    # brain/ off the import path.
    monkeypatch.setenv("PYTHONPATH", str(REPO / "backend"))
    monkeypatch.setenv("BATTRYML_SOURCE_ROOT", str(source_root))
    monkeypatch.setenv("BATTRYML_SOURCE_PYTHON", sys.executable)
    monkeypatch.setenv("BATTRYML_CATALOG_PATH", str(tmp_path / "catalog.duckdb"))
    monkeypatch.setenv("BATTRYML_JOB_ROOT", str(tmp_path / "jobs"))
    monkeypatch.setenv("BATTRYML_AUTH_ENABLED", "false")
    settings = Settings(
        project_root=tmp_path,
        source_root=source_root,
        source_python=Path(sys.executable),
        catalog_path=tmp_path / "catalog.duckdb",
        job_root=tmp_path / "jobs",
        api_host="127.0.0.1",
        api_port=8000,
        allowed_origins=("http://localhost:5173",),
        worker_threads=1,
        auth_enabled=False,
    )
    processed = source_root / "data" / "processed"
    with TestClient(create_app(settings)) as client:
        for params, keys in CHAIN:
            for key in keys:
                _run(client, key, params)
        _run(client, *GAP_DIAGNOSIS)

        gap = client.get("/api/analytics/soh-gap-diagnosis/100000002", params={"chemistry": "NCA"})
        assert gap.status_code == 200
        assert len(gap.json()["assessments"]) == 3
        series = client.get("/api/analytics/bms-soh-recalibration-series/100000002")
        assert series.status_code == 200

    for serial, planted_month in (("100000002", 7), ("100000021", 10)):
        seasonal = _latest(
            processed / "seasonal_charge_analysis" / f"serial={serial}", "profile_run=*"
        )
        months = pd.read_parquet(seasonal / "monthly_operating_profile.parquet")
        assert months["seasonal_profile_valid"].all() and len(months) == 4
        assert months["temperature_max_c"].max() > 30.0 > months["temperature_mean_c"].min()

        anomaly = json.loads(
            next(
                (processed / "hierarchical_cell_anomaly").rglob(
                    f"serial={serial}/**/model_summary.json"
                )
            ).read_text(encoding="utf-8")
        )
        assert anomaly["sustained_anomaly_episode_count"] >= 1
        assert pd.Timestamp(anomaly["first_episode_start"]).month == planted_month

        recalibration = _latest(
            processed / "bms_soh_recalibration" / f"serial={serial}", "recalibration_run=*"
        )
        summary = json.loads((recalibration / "model_summary.json").read_text(encoding="utf-8"))
        assert summary["event_rows"] == 3
        assert summary["matched_control_rows"] == 15

        catalog = _latest(
            processed / "physical_discharge_catalog" / f"serial={serial}", "catalog_run=*"
        )
        cycles = pd.read_parquet(catalog / "physical_discharge_cycles.parquet")
        assert len(cycles) >= 12
