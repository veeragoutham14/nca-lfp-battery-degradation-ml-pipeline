from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from battryml import __version__
from battryml.analytics.bms_soh_recalibration_overview import (
    build_bms_soh_recalibration_overview,
    build_bms_soh_recalibration_series,
)
from battryml.analytics.cell_anomaly_overview import (
    build_cell_anomaly_overview,
    build_cell_anomaly_series,
)
from battryml.analytics.cell_signal_overview import build_cell_signal_overview
from battryml.analytics.data_validation import (
    build_data_validation,
    build_electrothermal_validation,
)
from battryml.analytics.nca_capacity_overview import build_nca_capacity_overview
from battryml.analytics.soh_gap_diagnosis import build_soh_gap_diagnosis
from battryml.api.runtime import Runtime
from battryml.api.schemas import DeviceDisplayNameUpdate, JobCreate, LoginRequest
from battryml.auth import SESSION_COOKIE, AuthManager
from battryml.battery_profiles import battery_profile
from battryml.config import Settings
from battryml.data.import_planning import build_import_plan
from battryml.data.signal_registry import (
    CUSTOM_PURPOSE,
    IMPORT_PURPOSES,
    validate_custom_selection,
)


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def require_user(request: Request) -> dict[str, Any]:
    username = getattr(request.state, "username", None)
    if not username:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {"username": username}


def _public_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in artifact.items() if key not in {"fingerprint"}} | {
        "download_url": f"/api/artifacts/{artifact['artifact_id']}/download"
    }


def _parse_timestamp(value: Any) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    auth = AuthManager.from_settings(resolved_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = Runtime.build(resolved_settings)
        app.state.runtime = runtime
        app.state.initial_scan = runtime.scanner.scan()
        yield
        runtime.jobs.shutdown()

    app = FastAPI(
        title="BattryML API",
        version=__version__,
        description="Catalog, orchestration, and analytics API for modular battery pipelines.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_session(request: Request, call_next):
        public_paths = {"/api/health", "/api/auth/login", "/api/auth/me"}
        if (
            request.method != "OPTIONS"
            and request.url.path.startswith("/api/")
            and request.url.path not in public_paths
        ):
            username = auth.verify(request.cookies.get(SESSION_COOKIE))
            if username is None:
                return JSONResponse(status_code=401, content={"detail": "Authentication required"})
            request.state.username = username
        return await call_next(request)

    @app.post("/api/auth/login")
    def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
        if not auth.authenticate(payload.username.strip(), payload.password):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        username = payload.username.strip() if auth.enabled else auth.username
        response.set_cookie(
            SESSION_COOKIE,
            auth.issue(username),
            max_age=auth.session_seconds,
            httponly=True,
            samesite="lax",
            secure=False,
            path="/",
        )
        return {
            "username": username,
            "auth_enabled": auth.enabled,
            "authenticated": True,
        }

    @app.get("/api/auth/me")
    def current_user(request: Request) -> dict[str, Any]:
        username = auth.verify(request.cookies.get(SESSION_COOKIE))
        return {
            "username": username or "",
            "auth_enabled": auth.enabled,
            "authenticated": username is not None,
        }

    @app.post("/api/auth/logout", status_code=204)
    def logout(response: Response) -> Response:
        response.delete_cookie(SESSION_COOKIE, path="/")
        response.status_code = 204
        return response

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        source_problems = runtime.settings.validate_source()
        database = runtime.settings.source_database_status()
        return {
            "status": "degraded" if source_problems or not database["reachable"] else "ok",
            "version": __version__,
            "source_root": str(runtime.settings.source_root),
            "catalog_path": str(runtime.settings.catalog_path),
            "auth_enabled": auth.enabled,
            "source_problems": source_problems,
            "database": database,
            "counts": runtime.repository.counts(),
        }

    @app.get("/api/batteries")
    def batteries(request: Request) -> list[dict[str, Any]]:
        return _runtime(request).repository.list_batteries()

    @app.get("/api/devices")
    def devices(
        request: Request,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        return _runtime(request).devices.list_devices(refresh=refresh)

    @app.patch("/api/devices/{serial}/display-name")
    def update_device_display_name(
        serial: str,
        payload: DeviceDisplayNameUpdate,
        request: Request,
    ) -> dict[str, Any]:
        try:
            item = _runtime(request).devices.update_display_name(
                serial,
                payload.display_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="Device not found")
        return item

    @app.get("/api/devices/{serial}")
    def device(serial: str, request: Request, refresh: bool = False) -> dict[str, Any]:
        item = _runtime(request).devices.get_device(serial, refresh=refresh)
        if item is None:
            raise HTTPException(status_code=404, detail="Device not found")
        return item

    @app.get("/api/import/plan")
    def import_plan(
        serial: str,
        purpose: str,
        request: Request,
        chemistry: str | None = None,
        module_count: int | None = None,
        selections: str | None = None,
        _: dict[str, Any] = Depends(require_user),
    ) -> dict[str, Any]:
        """Per-signal import plan: watermarks, continuation rules, estimates."""

        runtime = _runtime(request)
        profile = battery_profile(serial)
        resolved_chemistry = (chemistry or (profile.chemistry if profile else "")).upper()
        if resolved_chemistry not in {"NCA", "LFP"}:
            raise HTTPException(status_code=400, detail="Unknown chemistry")
        if purpose not in IMPORT_PURPOSES:
            raise HTTPException(status_code=400, detail="Unknown import purpose")

        parsed_selections: dict[str, str] = {}
        if purpose == CUSTOM_PURPOSE:
            try:
                raw_selections = json.loads(selections) if selections else {}
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail="Invalid selections JSON") from exc
            try:
                parsed_selections = validate_custom_selection(raw_selections)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        device_start = None
        if profile is not None and profile.data_available_from:
            device_start = datetime.fromisoformat(profile.data_available_from).replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
            )

        return build_import_plan(
            purpose=purpose,
            selections=parsed_selections,
            source_data=runtime.settings.source_root / "data",
            serial=serial,
            device_start=device_start,
            chemistry=resolved_chemistry,
            module_count=module_count,
        )

    @app.get("/api/analytics/nca-capacity-overview")
    def nca_capacity_overview(
        request: Request,
        reference_loss_slope_ah_per_1000ah: float = Query(0.12, gt=0),
        bms_tolerance_pct: float = Query(1.0, ge=0),
    ) -> dict[str, Any]:
        runtime = _runtime(request)
        return build_nca_capacity_overview(
            source_root=runtime.settings.source_root,
            devices=runtime.devices.list_devices(),
            reference_loss_slope_ah_per_1000ah=(
                reference_loss_slope_ah_per_1000ah
            ),
            bms_tolerance_pct=bms_tolerance_pct,
        )

    @app.get("/api/analytics/cell-signal-overview")
    def cell_signal_overview(
        request: Request,
        chemistry: str = Query("NCA", pattern="^(NCA|LFP)$"),
    ) -> dict[str, Any]:
        runtime = _runtime(request)
        return build_cell_signal_overview(
            source_root=runtime.settings.source_root,
            devices=runtime.devices.list_devices(),
            chemistry=chemistry,
        )

    @app.get("/api/analytics/data-validation/{serial}")
    def data_validation(
        serial: str,
        request: Request,
        chemistry: str = Query("NCA", pattern="^(NCA|LFP)$"),
    ) -> dict[str, Any]:
        result = build_data_validation(
            source_root=_runtime(request).settings.source_root,
            serial=serial,
            chemistry=chemistry,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Data validation has not been run")
        return result

    @app.get("/api/analytics/electrothermal-validation/{serial}")
    def electrothermal_validation(
        serial: str,
        request: Request,
        chemistry: str = Query("NCA", pattern="^(NCA|LFP)$"),
    ) -> dict[str, Any]:
        result = build_electrothermal_validation(
            source_root=_runtime(request).settings.source_root,
            serial=serial,
            chemistry=chemistry,
        )
        if result is None:
            raise HTTPException(
                status_code=404, detail="Electrothermal validation has not been run"
            )
        return result

    @app.get("/api/analytics/cell-anomaly-overview")
    def cell_anomaly_overview(
        request: Request,
        chemistry: str = Query("NCA", pattern="^(NCA|LFP)$"),
    ) -> dict[str, Any]:
        runtime = _runtime(request)
        return build_cell_anomaly_overview(
            source_root=runtime.settings.source_root,
            devices=runtime.devices.list_devices(),
            chemistry=chemistry,
        )

    @app.get("/api/analytics/cell-anomaly-series/{serial}")
    def cell_anomaly_series(
        serial: str,
        request: Request,
        max_points: int = Query(1000, ge=50, le=5000),
    ) -> dict[str, Any]:
        runtime = _runtime(request)
        series = build_cell_anomaly_series(
            source_root=runtime.settings.source_root,
            serial=serial,
            max_points=max_points,
        )
        if series is None:
            raise HTTPException(status_code=404, detail="Cell anomaly run not found")
        return series

    @app.get("/api/analytics/bms-soh-recalibration-overview")
    def bms_soh_recalibration_overview(
        request: Request,
        chemistry: str = Query("NCA", pattern="^(NCA|LFP)$"),
    ) -> dict[str, Any]:
        runtime = _runtime(request)
        return build_bms_soh_recalibration_overview(
            source_root=runtime.settings.source_root,
            devices=runtime.devices.list_devices(),
            chemistry=chemistry,
        )

    @app.get("/api/analytics/bms-soh-recalibration-series/{serial}")
    def bms_soh_recalibration_series(
        serial: str,
        request: Request,
        max_points: int = Query(1000, ge=50, le=5000),
    ) -> dict[str, Any]:
        runtime = _runtime(request)
        series = build_bms_soh_recalibration_series(
            source_root=runtime.settings.source_root,
            serial=serial,
            max_points=max_points,
        )
        if series is None:
            raise HTTPException(status_code=404, detail="BMS SOH recalibration run not found")
        return series

    @app.get("/api/analytics/soh-gap-diagnosis/{serial}")
    def soh_gap_diagnosis(
        serial: str,
        request: Request,
        chemistry: str = Query(default="NCA", pattern="^(NCA|LFP)$"),
        _: dict[str, Any] = Depends(require_user),
    ) -> dict[str, Any]:
        result = build_soh_gap_diagnosis(_runtime(request).settings.source_root, chemistry, serial)
        if result is None:
            raise HTTPException(status_code=404, detail="SOH-gap diagnosis run not found")
        return result

    @app.get("/api/runs")
    def runs(
        request: Request,
        serial: str | None = None,
        model_family: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        return _runtime(request).repository.list_runs(
            serial=serial,
            model_family=model_family,
            limit=limit,
        )

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str, request: Request) -> dict[str, Any]:
        run = _runtime(request).repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        run.pop("summary_json", None)
        return run

    @app.get("/api/latest-runs")
    def latest_runs(
        request: Request,
        model_family: str = Query(min_length=1),
    ) -> list[dict[str, Any]]:
        return _runtime(request).repository.list_latest_runs(model_family=model_family)

    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str, request: Request) -> dict[str, Any]:
        summary = _runtime(request).repository.get_summary(run_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return summary

    @app.get("/api/runs/{run_id}/series")
    def run_series(
        run_id: str,
        request: Request,
        max_history_points: int = Query(2000, ge=50, le=10000),
    ) -> dict[str, Any]:
        series = _runtime(request).series.run_series(run_id, max_history_points)
        if series is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return series

    @app.get("/api/runs/{run_id}/validation")
    def run_validation(
        run_id: str,
        request: Request,
        max_points: int = Query(2000, ge=50, le=10000),
    ) -> dict[str, Any]:
        series = _runtime(request).series.validation_series(run_id, max_points)
        if series is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return series

    @app.get("/api/runs/{run_id}/artifacts")
    def run_artifacts(run_id: str, request: Request) -> list[dict[str, Any]]:
        runtime = _runtime(request)
        if runtime.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return [_public_artifact(item) for item in runtime.repository.list_artifacts(run_id)]

    @app.get("/api/runs/{run_id}/datasets")
    def run_datasets(run_id: str, request: Request) -> list[dict[str, Any]]:
        runtime = _runtime(request)
        if runtime.repository.get_run(run_id) is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return runtime.repository.list_datasets(run_id)

    @app.get("/api/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str, request: Request) -> FileResponse:
        artifact = _runtime(request).repository.get_artifact(artifact_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Artifact not found")
        path = Path(artifact["path"])
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=410, detail="Artifact is no longer available")
        return FileResponse(path, media_type=artifact["media_type"], filename=artifact["name"])

    @app.post("/api/catalog/scan")
    def scan_catalog(request: Request) -> dict[str, Any]:
        return _runtime(request).scanner.scan()

    @app.get("/api/pipelines")
    def pipelines(request: Request) -> list[dict[str, Any]]:
        return _runtime(request).registry.definitions()

    @app.post("/api/jobs", status_code=202)
    def create_job(payload: JobCreate, request: Request) -> dict[str, Any]:
        runtime = _runtime(request)
        problems = runtime.settings.validate_source()
        if problems:
            raise HTTPException(status_code=409, detail={"source_problems": problems})
        try:
            plugin = runtime.registry.get(payload.pipeline_key)
            normalized = plugin.validate(payload.parameters)
            serial = str(normalized.get("serial", "")).strip()
            if plugin.key == "veera.raw_telemetry_import.v1" and serial:
                device_item = runtime.devices.get_device(serial, refresh=True)
                continuation = (device_item or {}).get("recommended_import_start_at")
                if continuation:
                    requested_start = normalized.get("start")
                    if requested_start and _parse_timestamp(requested_start) < _parse_timestamp(
                        continuation
                    ):
                        raise ValueError(
                            f"Battery {serial} already has local telemetry. Start must be "
                            f"{continuation} or later to avoid overlap."
                        )
                    if not requested_start:
                        normalized["start"] = continuation
                    requested_end = normalized.get("end")
                    if requested_end and _parse_timestamp(normalized["start"]) >= _parse_timestamp(
                        requested_end
                    ):
                        raise ValueError(
                            f"Battery {serial} has no new timestamps before the selected end. "
                            f"The next import starts at {normalized['start']}."
                        )
                elif not normalized.get("start"):
                    configured_start = (device_item or {}).get("configured_start")
                    if configured_start:
                        normalized["start"] = configured_start
            if plugin.dependency_stages() and serial:
                device_item = runtime.devices.get_device(serial, refresh=True)
                readiness = (device_item or {}).get("readiness", {})
                workflow = (device_item or {}).get("workflow", {})
                missing_stages = [
                    stage
                    for stage in plugin.dependency_stages()
                    if workflow.get(stage, {}).get("status") != "current"
                    and not readiness.get(stage, False)
                ]
                if missing_stages:
                    reasons = "; ".join(
                        str(workflow.get(stage, {}).get("reason") or stage)
                        for stage in missing_stages
                    )
                    raise ValueError(
                        f"Pipeline {plugin.key} requires current prerequisite stages for battery "
                        f"{serial}: {', '.join(missing_stages)}. {reasons}"
                    )
            return runtime.jobs.submit(payload.pipeline_key, normalized)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/jobs")
    def jobs(
        request: Request,
        limit: int = Query(100, ge=1, le=500),
        include_archived: bool = False,
    ) -> list[dict[str, Any]]:
        return _runtime(request).repository.list_jobs(
            limit,
            include_archived=include_archived,
        )

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request) -> dict[str, Any]:
        item = _runtime(request).repository.get_job(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return item

    @app.patch("/api/jobs/{job_id}/archive")
    def archive_job(job_id: str, request: Request) -> dict[str, Any]:
        try:
            item = _runtime(request).repository.archive_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if item is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return item

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str, request: Request) -> Response:
        try:
            removed = _runtime(request).repository.delete_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not removed:
            raise HTTPException(status_code=404, detail="Job not found")
        return Response(status_code=204)

    @app.get("/api/jobs/{job_id}/logs")
    def job_logs(job_id: str, request: Request) -> dict[str, Any]:
        item = _runtime(request).repository.get_job(job_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Job not found")

        def tail(path_value: str | None) -> str:
            if not path_value:
                return ""
            path = Path(path_value)
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8", errors="replace")[-20000:]

        return {
            "job_id": job_id,
            "stdout": tail(item.get("stdout_path")),
            "stderr": tail(item.get("stderr_path")),
        }

    frontend_dist = resolved_settings.project_root / "frontend" / "dist"
    if frontend_dist.exists():
        assets = frontend_dist / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str) -> FileResponse:
            candidate = (frontend_dist / path).resolve()
            if path and frontend_dist.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
