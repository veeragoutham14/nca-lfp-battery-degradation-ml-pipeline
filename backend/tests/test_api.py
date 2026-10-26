from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from battryml.api.app import create_app
from battryml.battery_profiles import battery_profile
from battryml.config import Settings

LFP_SERIALS = {
    "100000020",
    "100000021",
    "100000022",
    "100000023",
    "100000024",
    "100000025",
    "100000026",
    "100000027",
    "100000028",
    "100000032",
    "100000033",
    "100000034",
    "100000035",
    "100000036",
    "100000037",
    "100000038",
    "100000039",
    "100000040",
    "100000041",
}


def test_api_exposes_catalog_and_chart_series(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["counts"]["runs"] == 1
        assert health.json()["database"]["configured"] is False
        assert health.json()["database"]["reachable"] is False

        batteries = client.get("/api/batteries").json()
        assert batteries[0]["serial"] == "TEST001"

        runs = client.get("/api/runs", params={"serial": "TEST001"}).json()
        run_id = runs[0]["run_id"]
        series = client.get(f"/api/runs/{run_id}/series")
        assert series.status_code == 200
        assert len(series.json()["history"]) == 2
        assert series.json()["forecasts"][0]["points"][1]["soh_pct"] == 97.2

        validation = client.get(f"/api/runs/{run_id}/validation").json()
        assert len(validation["points"]) == 2
        artifacts = client.get(f"/api/runs/{run_id}/artifacts").json()
        assert {item["kind"] for item in artifacts} >= {
            "summary",
            "history",
            "forecast",
            "validation",
        }


def test_api_exposes_seasonal_profile_and_soh_transition_series(
    platform_settings: Settings,
) -> None:
    run_dir = (
        platform_settings.source_root
        / "data"
        / "processed"
        / "seasonal_charge_analysis"
        / "serial=TEST001"
        / "profile_run=20260701T000000Z"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "model_summary.json").write_text(
        '{"model_family":"seasonal_charging_stress_profile",'
        f'"nominal_capacity_ah":{battery_profile("100000001").nominal_capacity_ah},'
        '"monthly_profile_rows":2}',
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "month_start_utc": [
                    datetime(2026, 1, 1, tzinfo=UTC),
                    datetime(2026, 2, 1, tzinfo=UTC),
                ],
                "season": ["winter", "winter"],
                "expected_days_in_observed_scope": [31, 28],
                "observed_days": [31, 28],
                "telemetry_coverage_fraction": [0.99, 0.98],
                "seasonal_profile_valid": [True, True],
                "charge_ah": [620.0, 560.0],
                "discharge_ah": [590.0, 530.0],
                "charge_ah_per_observed_day": [20.0, 20.0],
                "discharge_ah_per_observed_day": [19.0, 18.93],
                "high_soc_95_hours": [40.0, 35.0],
                "low_soc_5_hours": [12.0, 10.0],
                "soc_0_5_hours": [12.0, 10.0],
                "soc_5_20_hours": [90.0, 80.0],
                "soc_20_40_hours": [130.0, 115.0],
                "soc_40_60_hours": [190.0, 170.0],
                "soc_60_80_hours": [160.0, 145.0],
                "soc_80_95_hours": [105.0, 92.0],
                "soc_95_100_hours": [40.0, 35.0],
                "soc_known_hours": [727.0, 647.0],
                "soc_unknown_hours": [17.0, 25.0],
                "soc_known_coverage_fraction": [727.0 / 744.0, 647.0 / 672.0],
                "temperature_mean_c": [8.0, 9.0],
                "temperature_max_c": [16.0, 17.0],
                "voltage_spread_p95_daily_mv": [18.0, 19.0],
                "voltage_spread_max_mv": [30.0, 31.0],
                "bms_soh_first_pct": [98.0, 98.0],
                "bms_soh_last_pct": [98.0, 96.0],
                "audited_soh_change_pp": [0.0, -2.0],
            }
        ),
        run_dir / "monthly_operating_profile.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "transition_id": ["soh_transition_00001"],
                "event_time_utc": [datetime(2026, 2, 20, tzinfo=UTC)],
                "event_type": ["step_drop"],
                "previous_soh_pct": [98.0],
                "new_soh_pct": [96.0],
                "soh_delta_pp": [-2.0],
                "context_quality_ready": [True],
                "near_full_charge_within_72h_before": [True],
                "hours_since_near_full_charge": [7.5],
                "pre_charge_ah": [120.0],
                "post_charge_ah": [150.0],
                "pre_temperature_max_c": [16.0],
                "post_temperature_max_c": [17.0],
                "pre_voltage_spread_p95_mv": [18.0],
                "post_voltage_spread_p95_mv": [19.0],
            }
        ),
        run_dir / "soh_transition_context.parquet",
    )

    with TestClient(create_app(platform_settings)) as client:
        latest = client.get(
            "/api/latest-runs",
            params={"model_family": "seasonal_charging_stress_profile"},
        )
        runs = client.get("/api/runs", params={"serial": "TEST001"}).json()
        seasonal_run = next(
            item
            for item in runs
            if item["model_family"] == "seasonal_charging_stress_profile"
        )
        series = client.get(f"/api/runs/{seasonal_run['run_id']}/series")

    assert latest.status_code == 200
    assert latest.json()[0]["run_id"] == seasonal_run["run_id"]
    assert series.status_code == 200
    payload = series.json()
    assert payload["seasonal_profile"][1]["charge_ah"] == 560.0
    assert payload["seasonal_profile"][1]["bms_soh_last_pct"] == 96.0
    assert payload["seasonal_profile"][1]["soc_40_60_hours"] == 170.0
    assert payload["seasonal_profile"][1]["soc_unknown_hours"] == 25.0
    assert payload["soh_transitions"][0]["hours_since_near_full_charge"] == 7.5


def test_api_updates_and_restores_device_display_name(
    platform_settings: Settings,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        updated = client.patch(
            "/api/devices/100000001/display-name",
            json={"display_name": "Garage East"},
        )
        assert updated.status_code == 200
        assert updated.json()["display_name"] == "Garage East"
        assert updated.json()["display_name_source"] == "custom"

        fetched = client.get("/api/devices/100000001")
        assert fetched.json()["display_name"] == "Garage East"

        restored = client.patch(
            "/api/devices/100000001/display-name",
            json={"display_name": None},
        )
        assert restored.status_code == 200
        assert restored.json()["display_name"] == "NCA Field 01"
        assert restored.json()["display_name_source"] == "profile"


def test_api_lists_versioned_plugins(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        plugins = client.get("/api/pipelines").json()
        assert {plugin["key"] for plugin in plugins} == {
            "battryml.capacity_soh_investigation.v1",
            "battryml.prepare_analysis_data.v1",
            "battryml.soh_signal_audit.v1",
            "battryml.soh_gap_diagnosis.v1",
            "veera.seasonal_charge_analysis.v1",
            "veera.build_events.v1",
            "veera.cell_health_features.v1",
            "veera.cell_signal_audit.v1",
            "veera.cell_voltage_anomaly.v1",
            "veera.bms_electrothermal_source_audit.v1",
            "veera.bms_hierarchical_features.v1",
            "veera.bms_soh_recalibration.v4",
            "veera.data_validation.v1",
            "veera.electrothermal_import_validation.v1",
            "veera.discharge_capacity_evidence.v1",
            "veera.capacity_trend.v1",
            "veera.classify_rows.v1",
            "veera.fixed_slope_verification.v1",
            "veera.full_cycle_soh.v1",
            "veera.hierarchical_voltage_anomaly.v2",
            "veera.raw_telemetry_import.v1",
            "veera.raw_telemetry_import.v2",
        }
        fixed_slope = next(
            plugin
            for plugin in plugins
            if plugin["key"] == "veera.fixed_slope_verification.v1"
        )
        assert fixed_slope["device_roles"] == ["field_tester"]
        assert fixed_slope["chemistries"] == ["NCA"]
        assert fixed_slope["requires_database"] is False
        assert fixed_slope["requires_stage"] == "events"
        raw_import = next(
            plugin
            for plugin in plugins
            if plugin["key"] == "veera.raw_telemetry_import.v1"
        )
        assert raw_import["requires_database"] is True
        seasonal = next(
            plugin
            for plugin in plugins
            if plugin["key"] == "veera.seasonal_charge_analysis.v1"
        )
        assert seasonal["requires_stages"] == ["capacity_soh_investigation"]


def test_api_exposes_role_aware_device_library(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        devices = client.get("/api/devices").json()
        by_serial = {device["serial"]: device for device in devices}

        assert by_serial["100000001"]["device_role"] == "field_tester"
        assert by_serial["100000001"]["nominal_capacity_ah"] == 64.0
        assert by_serial["100000017"]["device_role"] == "full_cycle_test"
        assert by_serial["TEST001"]["device_role"] == "unassigned"


def test_api_exposes_chemistry_separated_nca_capacity_overview(
    platform_settings: Settings,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        response = client.get("/api/analytics/nca-capacity-overview")

        assert response.status_code == 200
        payload = response.json()
        assert payload["chemistry"] == "NCA"
        assert payload["reference"]["loss_slope_ah_per_1000ah"] == 0.12
        assert payload["metrics"]["configured_device_count"] == 0
        assert payload["metrics"]["available_device_count"] == 0
        assert payload["cohort"]["module_count"] is None


def test_api_exposes_nca_cell_signal_readiness(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        response = client.get("/api/analytics/cell-signal-overview")

        assert response.status_code == 200
        payload = response.json()
        assert payload["chemistry"] == "NCA"
        assert payload["metrics"]["configured_device_count"] == 1
        assert payload["metrics"]["field_device_count"] == 0
        assert payload["metrics"]["reference_device_count"] == 1
        assert payload["metrics"]["audited_device_count"] == 0
        assert payload["metrics"]["ready_device_count"] == 0


def test_api_exposes_nca_cell_anomaly_overview(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        response = client.get("/api/analytics/cell-anomaly-overview")

        assert response.status_code == 200
        payload = response.json()
        assert payload["chemistry"] == "NCA"
        assert payload["metrics"]["configured_device_count"] == 1
        assert payload["metrics"]["field_device_count"] == 0
        assert payload["metrics"]["reference_device_count"] == 1
        assert payload["metrics"]["analyzed_device_count"] == 0


def test_api_exposes_chemistry_scoped_cell_overviews(
    platform_settings: Settings,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        signal = client.get("/api/analytics/cell-signal-overview?chemistry=LFP")
        anomaly = client.get("/api/analytics/cell-anomaly-overview?chemistry=LFP")
        invalid = client.get("/api/analytics/cell-signal-overview?chemistry=OTHER")

        assert signal.status_code == 200
        assert signal.json()["chemistry"] == "LFP"
        # LFP devices stay discoverable before any analysis has run; their
        # readiness flags must show unprepared rather than hiding them.
        signal_devices = signal.json()["devices"]
        assert {device["serial"] for device in signal_devices} == set(LFP_SERIALS)
        assert all(not device["audited"] for device in signal_devices)
        assert anomaly.status_code == 200
        assert anomaly.json()["chemistry"] == "LFP"
        assert {device["serial"] for device in anomaly.json()["devices"]} == set(LFP_SERIALS)
        assert invalid.status_code == 422


def test_api_rejects_pipeline_for_wrong_device_role(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        response = client.post(
            "/api/jobs",
            json={
                "pipeline_key": "veera.full_cycle_soh.v1",
                "parameters": {"serial": "100000001"},
            },
        )

        assert response.status_code == 422
        assert "assigned as field_tester" in response.json()["detail"]


def test_api_login_protects_workspace_routes(platform_settings: Settings) -> None:
    protected_settings = replace(
        platform_settings,
        auth_enabled=True,
        auth_username="operator",
        auth_password="test-password",
        auth_secret="test-secret",
    )
    with TestClient(create_app(protected_settings)) as client:
        assert client.get("/api/health").status_code == 200
        session = client.get("/api/auth/me")
        assert session.status_code == 200
        assert session.json()["authenticated"] is False
        assert client.get("/api/devices").status_code == 401
        assert client.post(
            "/api/auth/login",
            json={"username": "operator", "password": "wrong"},
        ).status_code == 401

        login = client.post(
            "/api/auth/login",
            json={"username": "operator", "password": "test-password"},
        )
        assert login.status_code == 200
        assert login.json()["authenticated"] is True
        assert client.get("/api/devices").status_code == 200

        assert client.post("/api/auth/logout").status_code == 204
        assert client.get("/api/devices").status_code == 401


def test_api_archives_and_removes_completed_jobs(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        repository = client.app.state.runtime.repository
        repository.create_job(
            {
                "job_id": "completed-job",
                "pipeline_key": "veera.raw_telemetry_import.v1",
                "serial": "TEST001",
                "status": "completed",
                "request_json": "{}",
                "created_at": datetime.now(UTC),
            }
        )

        assert len(client.get("/api/jobs").json()) == 1
        archived = client.patch("/api/jobs/completed-job/archive")
        assert archived.status_code == 200
        assert archived.json()["archived_at"] is not None
        assert client.get("/api/jobs").json() == []
        assert len(client.get("/api/jobs?include_archived=true").json()) == 1

        removed = client.delete("/api/jobs/completed-job")
        assert removed.status_code == 204
        assert client.get("/api/jobs?include_archived=true").json() == []


def test_api_keeps_active_jobs_out_of_archive(platform_settings: Settings) -> None:
    with TestClient(create_app(platform_settings)) as client:
        client.app.state.runtime.repository.create_job(
            {
                "job_id": "running-job",
                "pipeline_key": "veera.raw_telemetry_import.v1",
                "serial": "TEST001",
                "status": "running",
                "request_json": "{}",
                "created_at": datetime.now(UTC),
            }
        )

        assert client.patch("/api/jobs/running-job/archive").status_code == 409
        assert client.delete("/api/jobs/running-job").status_code == 409


def test_api_rejects_overlapping_pipeline_jobs_for_same_serial(
    platform_settings: Settings,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        client.app.state.runtime.repository.create_job(
            {
                "job_id": "active-stage",
                "pipeline_key": "veera.classify_rows.v1",
                "serial": "TEST001",
                "status": "running",
                "request_json": "{}",
                "created_at": datetime.now(UTC),
            }
        )

        response = client.post(
            "/api/jobs",
            json={
                "pipeline_key": "veera.raw_telemetry_import.v1",
                "parameters": {
                    "serial": "TEST001",
                    "chemistry": "NCA",
                    "start": "2025-01-01T00:00:00+00:00",
                    "end": "2025-01-02T00:00:00+00:00",
                },
            },
        )

        assert response.status_code == 422
        assert "already has an active pipeline task" in response.json()["detail"]


def test_api_applies_cached_continuation_to_raw_import(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        runtime = client.app.state.runtime
        submitted: dict[str, object] = {}
        monkeypatch.setattr(
            runtime.devices,
            "get_device",
            lambda serial, refresh=False: {
                "recommended_import_start_at": "2026-07-17T10:05:00+00:00"
            },
        )

        def capture_submit(pipeline_key, parameters):
            submitted.update(parameters)
            return {"pipeline_key": pipeline_key, "request_json": "{}"}

        monkeypatch.setattr(runtime.jobs, "submit", capture_submit)
        response = client.post(
            "/api/jobs",
            json={
                "pipeline_key": "veera.raw_telemetry_import.v1",
                "parameters": {"serial": "100000001", "chemistry": "NCA"},
            },
        )

        assert response.status_code == 202
        assert submitted["start"] == "2026-07-17T10:05:00+00:00"


def test_api_applies_configured_start_to_first_raw_import(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        runtime = client.app.state.runtime
        submitted: dict[str, object] = {}
        monkeypatch.setattr(
            runtime.devices,
            "get_device",
            lambda serial, refresh=False: {
                "recommended_import_start_at": None,
                "configured_start": "2023-08-01",
            },
        )

        def capture_submit(pipeline_key, parameters):
            submitted.update(parameters)
            return {"pipeline_key": pipeline_key, "request_json": "{}"}

        monkeypatch.setattr(runtime.jobs, "submit", capture_submit)
        response = client.post(
            "/api/jobs",
            json={
                "pipeline_key": "veera.raw_telemetry_import.v1",
                "parameters": {"serial": "100000001", "chemistry": "NCA"},
            },
        )

        assert response.status_code == 202
        assert submitted["start"] == "2023-08-01"


def test_api_rejects_raw_import_overlap(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        monkeypatch.setattr(
            client.app.state.runtime.devices,
            "get_device",
            lambda serial, refresh=False: {
                "recommended_import_start_at": "2026-07-17T10:05:00+00:00"
            },
        )
        response = client.post(
            "/api/jobs",
            json={
                "pipeline_key": "veera.raw_telemetry_import.v1",
                "parameters": {
                    "serial": "100000001",
                    "chemistry": "NCA",
                    "start": "2026-07-17T00:00:00+00:00",
                },
            },
        )

        assert response.status_code == 422
        assert "to avoid overlap" in response.json()["detail"]


def test_api_refreshes_stage_readiness_before_dependency_validation(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        runtime = client.app.state.runtime
        stale_inventory = runtime.devices.refresh_serial("100000016")
        assert stale_inventory["has_electrothermal_source_audit"] is False

        raw_dir = (
            platform_settings.source_root
            / "data"
            / "raw_parquet"
            / "serial=100000016"
        )
        raw_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"time": [datetime.now(UTC)]}), raw_dir / "part-00000.parquet"
        )

        audit_dir = (
            platform_settings.source_root
            / "data"
            / "processed"
            / "electrothermal_import_validation"
            / "chemistry=NCA"
            / "serial=100000016"
            / "validation_run=20260727T100000Z"
        )
        audit_dir.mkdir(parents=True)
        (audit_dir / "model_summary.json").write_text(
            '{"ready": true, "module_count": 2}',
            encoding="utf-8",
        )

        monkeypatch.setattr(
            runtime.jobs,
            "submit",
            lambda pipeline_key, parameters: {
                "pipeline_key": pipeline_key,
                "serial": parameters["serial"],
                "status": "queued",
            },
        )
        response = client.post(
            "/api/jobs",
            json={
                "pipeline_key": "veera.bms_hierarchical_features.v1",
                "parameters": {
                        "serial": "100000016",
                        "module_count": 2,
                        "chemistry": "NCA",
                },
            },
        )

        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        refreshed = runtime.repository.get_device_inventory("100000016")
        assert refreshed is not None
        assert refreshed["has_electrothermal_source_audit"] is False


def test_api_enforces_all_seasonal_profile_dependencies(
    platform_settings: Settings,
    monkeypatch,
) -> None:
    with TestClient(create_app(platform_settings)) as client:
        runtime = client.app.state.runtime
        request = {
            "pipeline_key": "veera.seasonal_charge_analysis.v1",
            "parameters": {"serial": "100000001", "module_count": 2},
        }
        raw_dir = (
            platform_settings.source_root
            / "data"
            / "raw_parquet"
            / "serial=100000001"
        )
        raw_dir.mkdir(parents=True)
        pq.write_table(
            pa.table({"time": [datetime.now(UTC)]}), raw_dir / "part-00000.parquet"
        )
        missing = client.post("/api/jobs", json=request)
        assert missing.status_code == 422
        assert "capacity_soh_investigation" in missing.json()["detail"]

        feature_dir = (
            platform_settings.source_root
            / "data"
            / "processed"
            / "bms_hierarchical_features"
            / "serial=100000001"
            / "feature_run=20260727T100000Z"
        )
        audit_dir = (
            platform_settings.source_root
            / "data"
            / "processed"
            / "fleet_soh_audit"
            / "serial=100000001"
            / "audit_run=20260727T100000Z"
        )
        recalibration_dir = (
            platform_settings.source_root
            / "data"
            / "processed"
            / "bms_soh_recalibration"
            / "serial=100000001"
            / "recalibration_run=20260727T100000Z"
        )
        source_audit_dir = (
            platform_settings.source_root
            / "data"
            / "processed"
            / "electrothermal_import_validation"
            / "chemistry=NCA"
            / "serial=100000001"
            / "validation_run=20260727T100000Z"
        )
        discharge_dir = (
            platform_settings.source_root
            / "data"
            / "processed"
            / "discharge_capacity_evidence"
            / "serial=100000001"
            / "capacity_evidence_run=20260727T100000Z"
        )
        source_audit_dir.mkdir(parents=True)
        feature_dir.mkdir(parents=True)
        audit_dir.mkdir(parents=True)
        discharge_dir.mkdir(parents=True)
        recalibration_dir.mkdir(parents=True)
        (source_audit_dir / "model_summary.json").write_text(
            '{"ready": true, "module_count": 2}', encoding="utf-8"
        )
        pq.write_table(pa.table({"time": [datetime.now(UTC)]}), feature_dir / "features.parquet")
        pq.write_table(pa.table({"time": [datetime.now(UTC)]}), audit_dir / "daily_health.parquet")
        pq.write_table(
            pa.table({"discharge_episode_id": ["discharge_000001"]}),
            discharge_dir / "discharge_episodes.parquet",
        )
        pq.write_table(
            pa.table({"event_id": ["recalibration_00001"]}),
            recalibration_dir / "recalibration_events.parquet",
        )
        monkeypatch.setattr(
            runtime.jobs,
            "submit",
            lambda pipeline_key, parameters: {
                "pipeline_key": pipeline_key,
                "serial": parameters["serial"],
                "status": "queued",
            },
        )
        monkeypatch.setattr(
            runtime.devices,
            "get_device",
            lambda serial, refresh=False: {
                "serial": serial,
                "readiness": {},
                "workflow": {"capacity_soh_investigation": {"status": "current"}},
            },
        )

        ready = client.post("/api/jobs", json=request)
        assert ready.status_code == 202
        assert ready.json()["status"] == "queued"
