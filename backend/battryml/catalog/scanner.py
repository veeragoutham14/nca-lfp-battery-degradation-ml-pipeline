from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.compute as pc
import pyarrow.parquet as pq

from battryml.catalog.repository import CatalogRepository, utc_now
from battryml.config import Settings


@dataclass(frozen=True)
class RunFamily:
    pipeline_key: str
    model_family: str
    root_name: str
    run_prefix: str
    forecast_names: tuple[str, ...]
    history_names: tuple[str, ...]
    validation_names: tuple[str, ...]


RUN_FAMILIES = (
    RunFamily(
        pipeline_key="veera.data_validation.v1",
        model_family="battery_data_validation",
        root_name="data_validation",
        run_prefix="validation_run",
        forecast_names=(),
        history_names=("coverage_by_month.parquet",),
        validation_names=("quality_findings.parquet",),
    ),
    RunFamily(
        pipeline_key="veera.bms_electrothermal_source_audit.v1",
        model_family="bms_electrothermal_source_quality",
        root_name="bms_electrothermal_source_quality",
        run_prefix="source_audit_run",
        forecast_names=(),
        history_names=("signal_availability.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.bms_hierarchical_features.v1",
        model_family="bms_hierarchical_electrothermal_features",
        root_name="bms_hierarchical_features",
        run_prefix="feature_run",
        forecast_names=(),
        history_names=("operating_context_*min.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.hierarchical_voltage_anomaly.v2",
        model_family="hierarchical_cell_voltage_anomaly",
        root_name="hierarchical_cell_anomaly",
        run_prefix="anomaly_run",
        forecast_names=(),
        history_names=("daily_anomaly_summary.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.cell_signal_audit.v1",
        model_family="cell_signal_availability",
        root_name="cell_signal_audit",
        run_prefix="cell_audit_run",
        forecast_names=(),
        history_names=("cell_signal_availability.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.cell_health_features.v1",
        model_family="cell_health_features",
        root_name="cell_health",
        run_prefix="feature_run",
        forecast_names=(),
        history_names=("cell_health_features.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.cell_voltage_anomaly.v1",
        model_family="nca_cell_voltage_anomaly",
        root_name="cell_anomaly",
        run_prefix="anomaly_run",
        forecast_names=(),
        history_names=("daily_cell_voltage_anomaly.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="battryml.soh_signal_audit.v1",
        model_family="bms_soh_signal_audit",
        root_name="fleet_soh_audit",
        run_prefix="audit_run",
        forecast_names=(),
        history_names=("daily_health.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.seasonal_charge_analysis.v1",
        model_family="seasonal_charging_stress_profile",
        root_name="seasonal_charge_analysis",
        run_prefix="profile_run",
        forecast_names=(),
        history_names=("daily_battery_stress.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.discharge_capacity_evidence.v1",
        model_family="discharge_derived_usable_capacity_evidence",
        root_name="discharge_capacity_evidence",
        run_prefix="capacity_evidence_run",
        forecast_names=(),
        history_names=("qualified_capacity_points.parquet",),
        validation_names=("rejected_candidates.parquet",),
    ),
    RunFamily(
        pipeline_key="veera.bms_soh_recalibration.v4",
        model_family="bms_soh_recalibration_evidence",
        root_name="bms_soh_recalibration",
        run_prefix="recalibration_run",
        forecast_names=(),
        history_names=("soh_daily_timeline.parquet",),
        validation_names=(),
    ),
    RunFamily(
        pipeline_key="veera.full_cycle_soh.v1",
        model_family="full_discharge_capacity",
        root_name="capacity_ml",
        run_prefix="ml_run",
        forecast_names=("capacity_forecast.parquet",),
        history_names=("ml_all_capacity_rows.parquet",),
        validation_names=("cutoff_validation.parquet",),
    ),
    RunFamily(
        pipeline_key="veera.fixed_slope_verification.v1",
        model_family="fixed_slope_transfer_verification",
        root_name="fixed_slope_verification",
        run_prefix="verification_run",
        forecast_names=("capacity_forecast.parquet",),
        history_names=("fixed_slope_all_capacity_rows.parquet",),
        validation_names=("cutoff_validation.parquet",),
    ),
)


MEDIA_TYPES = {
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".png": "image/png",
    ".html": "text/html",
    ".log": "text/plain",
}


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:20]
    return digest


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return _stable_id(str(path.resolve()), str(stat.st_size), str(stat.st_mtime_ns))


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _clean_json(json.load(handle))


def _parse_run_time(run_stamp: str) -> datetime | None:
    try:
        return datetime.strptime(run_stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _as_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None


def _parquet_metadata(path: Path) -> tuple[int | None, str | None]:
    try:
        parquet = pq.ParquetFile(path)
        schema = {field.name: str(field.type) for field in parquet.schema_arrow}
        return parquet.metadata.num_rows, json.dumps(schema, sort_keys=True)
    except Exception:
        return None, None


def _parquet_date_range(path: Path) -> tuple[datetime | None, datetime | None]:
    try:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        column = next(
            (
                name
                for name in (
                    "window_start_utc",
                    "month",
                    "day_utc",
                    "start_time_utc",
                    "end_anchor_time",
                    "capacity_window_start",
                    "forecast_timestamp",
                    "time",
                )
                if name in schema_names
            ),
            None,
        )
        if column is None:
            return None, None
        values = pq.read_table(path, columns=[column])[column]
        if values.null_count == len(values) or len(values) == 0:
            return None, None
        return _as_datetime(pc.min(values).as_py()), _as_datetime(pc.max(values).as_py())
    except Exception:
        return None, None


def _artifact_kind(name: str) -> str:
    lower = name.lower()
    if lower == "model_summary.json":
        return "summary"
    if "forecast" in lower and lower.endswith(".parquet") and "validation" not in lower:
        return "forecast"
    if "cutoff_validation" in lower and "forecast" not in lower:
        return "validation"
    if "cutoff_forecast_validation" in lower:
        return "forecast_validation"
    if "all_capacity_rows" in lower or "capacity_all_rows" in lower:
        return "history"
    if lower == "daily_health.parquet":
        return "history"
    if lower == "soh_change_events.parquet":
        return "change_events"
    if lower == "charge_episodes.parquet":
        return "charge_episodes"
    if lower == "daily_battery_stress.parquet":
        return "daily_stress"
    if lower == "monthly_operating_profile.parquet":
        return "seasonal_profile"
    if lower == "soh_transition_context.parquet":
        return "transition_context"
    if lower == "recalibration_events.parquet":
        return "recalibration_events"
    if lower == "capacity_discharge_events.parquet":
        return "capacity_discharge_events"
    if lower == "bms_soh_transitions.parquet":
        return "bms_soh_transitions"
    if lower == "soh_discharge_matches.parquet":
        return "soh_discharge_matches"
    if lower == "event_match_candidates.parquet":
        return "soh_discharge_match_candidates"
    if lower == "raw_discharge_trajectories.parquet":
        return "raw_discharge_trajectories"
    if lower == "downward_transition_trajectory_matches.parquet":
        return "downward_transition_trajectory_matches"
    if lower == "transition_trajectory_candidates.parquet":
        return "transition_trajectory_candidates"
    if lower == "compact_endpoint_reconciliation.parquet":
        return "compact_endpoint_reconciliation"
    if lower == "phase2_manifest.json":
        return "phase2_manifest"
    if lower == "matched_control_windows.parquet":
        return "matched_control_windows"
    if lower == "evidence_comparison.parquet":
        return "recalibration_evidence_comparison"
    if lower == "interpretable_model.json":
        return "recalibration_interpretable_model"
    if lower == "soh_daily_timeline.parquet":
        return "history"
    if lower == "bms_soh_recalibration_report.html":
        return "interactive_report"
    if lower == "cell_signal_availability.parquet":
        return "signal_availability"
    if lower == "cell_health_features.parquet":
        return "history"
    if lower == "daily_cell_voltage_anomaly.parquet":
        return "history"
    if lower == "cell_voltage_anomaly_episodes.parquet":
        return "anomaly_episodes"
    if lower == "cell_voltage_anomaly_scores.parquet":
        return "anomaly_scores"
    if lower == "signal_reconciliation.parquet":
        return "signal_reconciliation"
    if lower == "channel_map.parquet":
        return "channel_map"
    if lower.startswith("operating_context_"):
        return "operating_context"
    if lower.startswith("voltage_channel_features_"):
        return "voltage_channel_features"
    if lower.startswith("temperature_sensor_features_"):
        return "temperature_sensor_features"
    if lower.startswith("system_features_"):
        return "system_features"
    if lower == "anomaly_windows.parquet":
        return "anomaly_scores"
    if lower == "anomaly_episodes.parquet":
        return "anomaly_episodes"
    if lower == "daily_anomaly_summary.parquet":
        return "history"
    if lower == "episode_evidence_1s.parquet":
        return "episode_evidence"
    if "training_table" in lower:
        return "training"
    if "backtest" in lower:
        return "backtest"
    if "degradation_model_summary" in lower:
        return "model"
    if lower.endswith(".png"):
        return "report_image"
    if lower.endswith(".html"):
        return "interactive_report"
    return "dataset" if lower.endswith(".parquet") else "artifact"


class LegacyRunScanner:
    def __init__(self, settings: Settings, repository: CatalogRepository):
        self.settings = settings
        self.repository = repository

    def scan(
        self,
        *,
        serial: str | None = None,
        model_family: str | None = None,
    ) -> dict[str, Any]:
        discovered = 0
        examined = 0
        failures: list[dict[str, str]] = []
        for family in RUN_FAMILIES:
            if model_family and family.model_family != model_family:
                continue
            root = self.settings.source_root / "data" / "processed" / family.root_name
            if not root.exists():
                continue
            serial_partition = f"serial={serial}" if serial else "serial=*"
            patterns = (
                f"{serial_partition}/{family.run_prefix}=*/model_summary.json",
                f"chemistry=*/{serial_partition}/{family.run_prefix}=*/model_summary.json",
            )
            summary_paths = sorted(
                {path for pattern in patterns for path in root.glob(pattern)}
            )
            for summary_path in summary_paths:
                examined += 1
                try:
                    if self._ingest_run(family, summary_path):
                        discovered += 1
                except Exception as exc:
                    failures.append({"path": str(summary_path), "error": str(exc)})
        self.repository.refresh_batteries(str(self.settings.source_root))
        return {
            "source_root": str(self.settings.source_root),
            "runs_discovered": discovered,
            "runs_examined": examined,
            "failures": failures,
            "scanned_at": utc_now(),
        }

    def _ingest_run(self, family: RunFamily, summary_path: Path) -> bool:
        run_dir = summary_path.parent
        serial = run_dir.parent.name.removeprefix("serial=")
        source_run_id = run_dir.name
        run_stamp = source_run_id.split("=", 1)[-1]
        run_id = f"{family.pipeline_key}:{serial}:{run_stamp}"
        fingerprint = _fingerprint(summary_path)
        existing = self.repository.get_run(run_id)
        if existing and existing.get("fingerprint") == fingerprint:
            return False
        summary = _read_summary(summary_path)
        artifacts = self._collect_artifacts(run_id, run_dir, serial, run_stamp)
        datasets = self._collect_datasets(run_id, serial, artifacts)
        observed_start, observed_end = self._observed_range(family, run_dir)
        validation = summary.get("cutoff_validation") or {}
        degradation = summary.get("degradation_forecast_model") or {}
        audit = summary.get("audit_metrics") or {}

        run = {
            "run_id": run_id,
            "source_run_id": source_run_id,
            "serial": serial,
            "pipeline_key": family.pipeline_key,
            "model_family": summary.get("model_family") or family.model_family,
            "status": "completed",
            "source_path": str(run_dir.resolve()),
            "summary_path": str(summary_path.resolve()),
            "nominal_capacity_ah": summary.get("nominal_capacity_ah"),
            "training_rows": (
                degradation.get("train_rows")
                or summary.get("valid_monthly_profile_rows")
                or summary.get("valid_training_rows")
                or summary.get("training_rows")
                or summary.get("feature_rows")
                or summary.get("total_rows")
            ),
            "validation_rows": validation.get("rows"),
            "validation_mae_soh_pct": validation.get("mae_soh_pct"),
            "validation_max_error_soh_pct": validation.get("max_abs_soh_error_pct"),
            "loss_slope_per_1000ah": (
                degradation.get("loss_slope_per_1000ah")
                or audit.get("equivalent_capacity_loss_ah_per_1000ah")
            ),
            "forecast_rows": summary.get("forecast_rows"),
            "observed_start_at": observed_start,
            "observed_end_at": observed_end,
            "run_created_at": _parse_run_time(run_stamp),
            "ingested_at": utc_now(),
            "fingerprint": fingerprint,
            "summary_json": json.dumps(summary, allow_nan=False, sort_keys=True),
        }
        self.repository.replace_run_bundle(run, artifacts, datasets)
        return True

    def _collect_artifacts(
        self,
        run_id: str,
        run_dir: Path,
        serial: str,
        run_stamp: str,
    ) -> list[dict[str, Any]]:
        files = [path for path in run_dir.rglob("*") if path.is_file()]
        validation_root = self.settings.source_root / "data" / "validation_plots"
        for plot_root in (validation_root / f"serial={serial}", validation_root):
            if plot_root.exists():
                files.extend(
                    path for path in plot_root.glob(f"*{serial}*{run_stamp}*") if path.is_file()
                )

        artifacts: list[dict[str, Any]] = []
        for path in sorted(set(files)):
            stat = path.stat()
            artifacts.append(
                {
                    "artifact_id": _stable_id(run_id, str(path.resolve())),
                    "run_id": run_id,
                    "kind": _artifact_kind(path.name),
                    "name": path.name,
                    "path": str(path.resolve()),
                    "media_type": MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC),
                    "fingerprint": _fingerprint(path),
                }
            )
        return artifacts

    def _collect_datasets(
        self,
        run_id: str,
        serial: str,
        artifacts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        datasets: list[dict[str, Any]] = []
        for artifact in artifacts:
            if not artifact["name"].endswith(".parquet"):
                continue
            path = Path(artifact["path"])
            row_count, schema_json = _parquet_metadata(path)
            observed_start, observed_end = _parquet_date_range(path)
            datasets.append(
                {
                    "dataset_id": _stable_id(run_id, "dataset", str(path.resolve())),
                    "run_id": run_id,
                    "serial": serial,
                    "stage": artifact["kind"],
                    "name": path.name,
                    "path": str(path.resolve()),
                    "format": "parquet",
                    "row_count": row_count,
                    "schema_json": schema_json,
                    "observed_start_at": observed_start,
                    "observed_end_at": observed_end,
                    "fingerprint": artifact["fingerprint"],
                }
            )
        return datasets

    @staticmethod
    def _observed_range(
        family: RunFamily, run_dir: Path
    ) -> tuple[datetime | None, datetime | None]:
        for name in family.history_names:
            paths = sorted(run_dir.rglob(name))
            ranges = [_parquet_date_range(path) for path in paths]
            starts = [start for start, _ in ranges if start is not None]
            ends = [end for _, end in ranges if end is not None]
            if starts or ends:
                return min(starts) if starts else None, max(ends) if ends else None
        return None, None
