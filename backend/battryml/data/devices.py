from __future__ import annotations

import json
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from battryml.battery_profiles import battery_profiles, effective_nominal_capacity_ah
from battryml.catalog.repository import CatalogRepository
from battryml.config import Settings
from battryml.data.storage_paths import discover_serial_chemistries, serial_dataset_paths


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _next_import_timestamp(
    raw_end: str | None,
    five_minute_end: str | None,
) -> str | None:
    """Return the first timestamp not represented by either local dataset."""
    candidates: list[datetime] = []
    for value, step in ((raw_end, timedelta(seconds=1)), (five_minute_end, timedelta(minutes=5))):
        if not value:
            continue
        try:
            timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        candidates.append(timestamp + step)
    return max(candidates).isoformat() if candidates else None


def _partition_serial(path: Path) -> str | None:
    if path.is_dir() and path.name.startswith("serial="):
        return path.name.removeprefix("serial=")
    return None


def _parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet")) if root.exists() else []


def _single_serial_dataset_path(root: Path, serial: str) -> Path | None:
    paths = serial_dataset_paths(root, serial)
    if len(paths) > 1:
        locations = ", ".join(str(path) for path in paths)
        raise ValueError(
            f"Battery {serial} has multiple storage locations under {root}: {locations}"
        )
    return paths[0] if paths else None


def _file_signature(files: list[Path]) -> tuple[int, int]:
    total_bytes = 0
    latest_mtime_ns = 0
    for path in files:
        try:
            stat = path.stat()
        except OSError:
            continue
        total_bytes += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return total_bytes, latest_mtime_ns


def _parquet_inventory(
    files: list[Path],
) -> tuple[int, str | None, str | None]:
    row_count = 0
    minimum: Any = None
    maximum: Any = None
    for path in files:
        try:
            parquet = pq.ParquetFile(path)
            row_count += parquet.metadata.num_rows
            names = parquet.schema_arrow.names
            timestamp_name = next(
                (name for name in ("time", "timestamp", "end_timestamp") if name in names),
                None,
            )
            if timestamp_name is None:
                continue
            column_index = names.index(timestamp_name)
            for row_group_index in range(parquet.metadata.num_row_groups):
                statistics = (
                    parquet.metadata.row_group(row_group_index).column(column_index).statistics
                )
                if statistics is None or not statistics.has_min_max:
                    continue
                minimum = statistics.min if minimum is None else min(minimum, statistics.min)
                maximum = statistics.max if maximum is None else max(maximum, statistics.max)
        except Exception:
            continue
    return row_count, _iso(minimum), _iso(maximum)


def _has_parquet(root: Path) -> bool:
    return root.exists() and next(root.rglob("*.parquet"), None) is not None


def _latest_quality_gate_passed(root: Path) -> bool:
    summaries = sorted(root.glob("source_audit_run=*/model_summary.json"))
    if not summaries:
        return False
    try:
        return bool(
            json.loads(summaries[-1].read_text(encoding="utf-8")).get("source_quality_ready")
        )
    except (OSError, json.JSONDecodeError):
        return False


_WORKFLOW_STAGE_SPECS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "raw": (None, ()),
    "classified": ("classified_rows", ("raw",)),
    "events": ("events", ("classified",)),
    "cell_signal_audit": ("cell_signal_audit", ("raw",)),
    "cell_health": ("cell_health", ("cell_signal_audit",)),
    "electrothermal_source_audit": ("bms_electrothermal_source_quality", ("raw",)),
    "electrothermal_validation": ("electrothermal_import_validation", ()),
    "hierarchical_features": (
        "bms_hierarchical_features",
        ("electrothermal_validation",),
    ),
    "soh_signal_audit": ("fleet_soh_audit", ("raw",)),
    "data_validation": (
        "data_validation",
        ("hierarchical_features", "events"),
    ),
    "discharge_capacity_evidence": (
        "discharge_capacity_evidence",
        ("hierarchical_features",),
    ),
    "soh_recalibration": (
        "bms_soh_recalibration",
        ("discharge_capacity_evidence", "soh_signal_audit"),
    ),
    "capacity_soh_investigation": (
        "electrothermal_capacity_context",
        ("data_validation", "events"),
    ),
    "seasonal_charge_profile": (
        "seasonal_charge_analysis",
        ("capacity_soh_investigation",),
    ),
    "soh_gap_diagnosis": ("soh_gap_diagnosis", ("seasonal_charge_profile",)),
    "capacity_trend": ("capacity_trend", ("events",)),
    "fixed_slope_verification": (
        "fixed_slope_verification",
        ("events",),
    ),
    "full_cycle_soh": ("capacity_ml", ("capacity_trend", "soh_signal_audit")),
}


def _workflow_stage_root(processed: Path, folder: str, serial: str) -> Path:
    if folder not in {
        "data_validation",
        "electrothermal_capacity_context",
        "electrothermal_import_validation",
        "soh_gap_diagnosis",
    }:
        return processed / folder / f"serial={serial}"
    matches = list((processed / folder).glob(f"chemistry=*/serial={serial}"))
    if not matches:
        return processed / folder / f"serial={serial}"
    return max(matches, key=_latest_output_mtime_ns)


def _capacity_bms_validation_ready(root: Path) -> bool:
    summaries = sorted(root.glob("validation_run=*/model_summary.json"))
    if not summaries:
        return False
    try:
        payload = json.loads(summaries[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("readiness", {}).get("capacity_bms", {}).get("status") == "ready"


def _electrothermal_validation_ready(root: Path) -> bool:
    summaries = sorted(root.glob("validation_run=*/model_summary.json"))
    if not summaries:
        return False
    try:
        payload = json.loads(summaries[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("ready"))


def _phase2_contract_current(root: Path) -> bool:
    summaries = list(root.glob("recalibration_run=*/model_summary.json"))
    if not summaries:
        return False
    latest = max(summaries, key=lambda path: path.stat().st_mtime_ns)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("phase2", {}).get("phase2_version") == "v4"


def _latest_output_mtime_ns(root: Path) -> int:
    latest = 0
    if not root.exists():
        return latest
    for pattern in ("*.parquet", "model_summary.json", "dataset_manifest.json"):
        for path in root.rglob(pattern):
            try:
                latest = max(latest, path.stat().st_mtime_ns)
            except OSError:
                continue
    return latest


def _mtime_iso(value: int) -> str | None:
    return datetime.fromtimestamp(value / 1_000_000_000, UTC).isoformat() if value else None


def _electrothermal_import_mtime_ns(
    source_data: Path, chemistry: str, serial: str
) -> int:
    latest = 0
    for group in ("temperature", "cell_voltage"):
        root = (
            source_data
            / "electrothermal_telemetry"
            / group
            / "5m"
            / f"chemistry={chemistry}"
            / f"serial={serial}"
        )
        latest = max(latest, _latest_output_mtime_ns(root))
    return latest


def _workflow_directory_signature(source_data: Path, serial: str) -> tuple[int, ...]:
    """Detect newly created stage runs with a bounded number of directory stats."""
    processed = source_data / "processed"
    signature: list[int] = []
    for folder, _ in _WORKFLOW_STAGE_SPECS.values():
        if folder is None:
            continue
        root = _workflow_stage_root(processed, folder, serial)
        try:
            signature.append(root.stat().st_mtime_ns)
        except OSError:
            signature.append(0)
    return tuple(signature)


def _workflow_stage_states(
    source_data: Path,
    serial: str,
    *,
    raw_ready: bool,
    raw_mtime_ns: int,
    electrothermal_quality_ready: bool,
    capacity_bms_validation_ready: bool,
    electrothermal_validation_ready: bool = False,
    electrothermal_source_mtime_ns: int = 0,
) -> dict[str, dict[str, Any]]:
    """Evaluate freshness from persisted filesystem watermarks without reading row data."""
    states: dict[str, dict[str, Any]] = {
        "raw": {
            "status": "current" if raw_ready else "missing",
            "reason": "Raw telemetry is available." if raw_ready else "Import raw telemetry first.",
            "output_modified_at": _mtime_iso(raw_mtime_ns),
            "required_modified_at": None,
            "_output_mtime_ns": raw_mtime_ns,
        }
    }
    processed = source_data / "processed"
    legacy_hierarchical_mtime_ns = _latest_output_mtime_ns(
        _workflow_stage_root(processed, "bms_hierarchical_features", serial)
    )
    for stage, (folder, dependencies) in _WORKFLOW_STAGE_SPECS.items():
        if stage == "raw":
            continue
        blocked = [
            dependency
            for dependency in dependencies
            if states[dependency]["status"] != "current"
        ]
        root = _workflow_stage_root(processed, str(folder), serial)
        output_mtime_ns = _latest_output_mtime_ns(root)
        legacy_electrothermal_bridge = (
            stage == "electrothermal_validation"
            and output_mtime_ns == 0
            and electrothermal_source_mtime_ns == 0
            and legacy_hierarchical_mtime_ns > 0
        )
        if legacy_electrothermal_bridge:
            # Portable/legacy runs may predate the dedicated import-validation
            # stage. Their already-built hierarchical history is accepted only
            # until a new electrothermal dataset is imported; new data always
            # re-enables the explicit validation gate.
            output_mtime_ns = legacy_hierarchical_mtime_ns
        required_mtime_ns = max(
            (
                int(states[item].get("_output_mtime_ns") or 0)
                for item in dependencies
            ),
            default=0,
        )
        if stage == "electrothermal_validation":
            required_mtime_ns = electrothermal_source_mtime_ns
        if blocked:
            status = "blocked"
            reason = "Update required upstream: " + ", ".join(blocked) + "."
        elif legacy_electrothermal_bridge:
            status = "current"
            reason = (
                "Legacy hierarchical history is available; the separate "
                "electrothermal validation stage did not exist for this run."
            )
        elif not output_mtime_ns:
            status = "ready"
            reason = "Prerequisites are current; this stage can run."
        elif stage == "electrothermal_source_audit" and not electrothermal_quality_ready:
            status = "failed"
            reason = "The latest source-quality audit did not pass."
        elif stage == "electrothermal_validation" and not electrothermal_validation_ready:
            status = "failed" if output_mtime_ns else "ready"
            reason = (
                "The latest temperature and voltage validation did not pass."
                if output_mtime_ns
                else "Validate imported temperature and voltage data first."
            )
        elif stage == "data_validation" and not capacity_bms_validation_ready:
            status = "failed"
            reason = "The latest Stage 1 capacity/BMS validation did not pass."
        elif stage == "soh_recalibration" and not _phase2_contract_current(root):
            status = "stale"
            reason = "The latest run uses an older Phase 2 evidence contract; rerun required."
        elif output_mtime_ns < required_mtime_ns:
            status = "stale"
            reason = "Upstream data is newer than this stage output; rerun required."
        else:
            status = "current"
            reason = "Output matches the current upstream filesystem watermark."
        states[stage] = {
            "status": status,
            "reason": reason,
            "output_modified_at": _mtime_iso(output_mtime_ns),
            "required_modified_at": _mtime_iso(required_mtime_ns),
            "_output_mtime_ns": output_mtime_ns,
        }
    for item in states.values():
        item.pop("_output_mtime_ns", None)
    return states


def _module_count_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        try:
            rows.extend(pq.ParquetFile(path).read().to_pylist())
        except Exception:
            pass
    serial_root = path.parent / "module_counts"
    if serial_root.exists():
        for serial_path in sorted(serial_root.glob("serial=*/module_count.parquet")):
            try:
                rows.extend(pq.ParquetFile(serial_path).read().to_pylist())
            except Exception:
                continue
    return {
        str(row["serial"]): {
            "module_count": (None if row.get("module_count") is None else row.get("module_count")),
            "observed_module_counts": str(row.get("observed_module_counts") or ""),
            "module_count_status": str(row.get("module_count_status") or "missing"),
            "module_count_agreement": float(row.get("module_count_agreement") or 0.0),
        }
        for row in rows
    }


def _module_counts_from_capacity_manifests(root: Path) -> dict[str, dict[str, Any]]:
    """Recover DB-verified module metadata recorded by capacity imports.

    The dedicated snapshot is primary. This fallback keeps portable/new
    installations usable if a completed import manifest was retained while its
    small metadata snapshot was omitted or not yet discovered by the catalog.
    """

    if not root.exists():
        return {}
    serial_roots = list(root.glob("serial=*"))
    serial_roots.extend(root.glob("chemistry=*/serial=*"))
    recovered: dict[str, dict[str, Any]] = {}
    for serial_root in serial_roots:
        serial = serial_root.name.removeprefix("serial=")
        manifests = sorted(
            serial_root.glob("import_run=*/import_manifest.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for manifest in manifests:
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                module_count = int(payload.get("module_count"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if (
                payload.get("purpose") != "capacity_core"
                or payload.get("status") != "completed"
                or not 1 <= module_count <= 4
            ):
                continue
            recovered[serial] = {
                "module_count": module_count,
                "observed_module_counts": str(module_count),
                "module_count_status": "verified",
                "module_count_agreement": 1.0,
            }
            break
    return recovered


class DeviceLibraryService:
    def __init__(
        self,
        settings: Settings,
        repository: CatalogRepository,
    ):
        self.settings = settings
        self.repository = repository
        self._lock = threading.Lock()
        self._compact_inventory_cache: dict[
            str, tuple[tuple[int, int], tuple[int, str | None, str | None]]
        ] = {}
        self._workflow_state_cache: dict[
            str, tuple[tuple[Any, ...], dict[str, dict[str, Any]]]
        ] = {}

    def _compact_inventory(
        self,
        serial: str,
        root: Path,
    ) -> tuple[int, int, str | None, str | None]:
        serial_root = _single_serial_dataset_path(root, serial)
        files = _parquet_files(serial_root) if serial_root is not None else []
        total_bytes, latest_mtime_ns = _file_signature(files)
        signature = (total_bytes, latest_mtime_ns)
        cached = self._compact_inventory_cache.get(serial)
        if cached is None or cached[0] != signature:
            self._compact_inventory_cache[serial] = (
                signature,
                _parquet_inventory(files),
            )
        rows, observed_start, observed_end = self._compact_inventory_cache[serial][1]
        return len(files), rows, observed_start, observed_end

    def list_devices(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            return self._scan(refresh=refresh)

    def get_device(self, serial: str, *, refresh: bool = False) -> dict[str, Any] | None:
        if refresh:
            self.refresh_serial(serial)
        return next(
            (item for item in self.list_devices() if item["serial"] == serial),
            None,
        )

    def update_display_name(
        self,
        serial: str,
        display_name: str | None,
    ) -> dict[str, Any] | None:
        current = self.get_device(serial)
        if current is None:
            return None
        normalized = " ".join((display_name or "").split())
        if display_name is not None and not normalized:
            raise ValueError("Display name cannot be empty")
        if len(normalized) > 80:
            raise ValueError("Display name cannot exceed 80 characters")
        if normalized:
            duplicate = next(
                (
                    device
                    for device in self.list_devices()
                    if device["serial"] != serial
                    and str(device["display_name"]).casefold() == normalized.casefold()
                ),
                None,
            )
            if duplicate is not None:
                raise ValueError(
                    f"Display name is already used by serial {duplicate['serial']}"
                )
            self.repository.set_device_name_override(serial, normalized)
        else:
            self.repository.clear_device_name_override(serial)
        return self.get_device(serial)

    def refresh_serial(self, serial: str) -> dict[str, Any]:
        with self._lock:
            self._workflow_state_cache.pop(serial, None)
            return self._refresh_inventory(serial, force=False)

    def _scan(self, *, refresh: bool) -> list[dict[str, Any]]:
        source_data = self.settings.source_root / "data"
        raw_root = source_data / "raw_parquet"
        five_minute_root = source_data / "telemetry_5min"
        partition_chemistries = discover_serial_chemistries(raw_root)
        for serial, chemistry in discover_serial_chemistries(five_minute_root).items():
            existing = partition_chemistries.get(serial)
            if existing is not None and existing != chemistry:
                raise ValueError(
                    f"Battery {serial} is present in both {existing} and {chemistry} storage"
                )
            partition_chemistries[serial] = chemistry
        profiles = battery_profiles()
        name_overrides = self.repository.list_device_name_overrides()
        module_counts = _module_count_snapshot(
            source_data / "processed" / "device_metadata" / "module_counts.parquet"
        )
        manifest_module_counts = _module_counts_from_capacity_manifests(
            source_data / "import_manifests"
        )
        for serial, metadata in manifest_module_counts.items():
            module_counts.setdefault(serial, metadata)
        catalog_batteries = {str(item["serial"]): item for item in self.repository.list_batteries()}
        inventory = {str(item["serial"]): item for item in self.repository.list_device_inventory()}
        jobs = self.repository.list_jobs(500)
        imported_chemistry: dict[str, str] = {}
        for item in jobs:
            if item.get("pipeline_key") != "veera.raw_telemetry_import.v1":
                continue
            serial = str(item.get("serial", "")).strip()
            if not serial or serial in imported_chemistry:
                continue
            try:
                request = json.loads(str(item.get("request_json") or "{}"))
            except json.JSONDecodeError:
                continue
            chemistry = str(request.get("chemistry") or "").strip().upper()
            if chemistry in {"NCA", "LFP"}:
                imported_chemistry[serial] = chemistry
        serials = set(profiles) | set(catalog_batteries) | set(inventory)
        serials.update(imported_chemistry)
        serials.update(partition_chemistries)
        if raw_root.exists():
            serials.update(
                serial
                for path in raw_root.iterdir()
                if (serial := _partition_serial(path)) is not None
            )
        if five_minute_root.exists():
            serials.update(
                serial
                for path in five_minute_root.iterdir()
                if (serial := _partition_serial(path)) is not None
            )

        devices: list[dict[str, Any]] = []
        for serial in sorted(serials):
            profile = profiles.get(serial)
            inferred_chemistry = imported_chemistry.get(serial)
            partition_chemistry = partition_chemistries.get(serial)
            if (
                inferred_chemistry
                and partition_chemistry
                and inferred_chemistry != partition_chemistry
            ):
                raise ValueError(
                    f"Battery {serial} import chemistry {inferred_chemistry} conflicts with "
                    f"storage chemistry {partition_chemistry}"
                )
            inferred_chemistry = partition_chemistry or inferred_chemistry
            if profile and partition_chemistry and profile.chemistry != partition_chemistry:
                raise ValueError(
                    f"Battery {serial} profile chemistry {profile.chemistry} conflicts with "
                    f"storage chemistry {partition_chemistry}"
                )
            module_count = module_counts.get(serial, {})
            observed_module_count = module_count.get("module_count")
            module_count_status = module_count.get(
                "module_count_status",
                "missing" if profile and profile.module_capacity_ah else "unconfigured",
            )
            module_count_source = "database_snapshot" if module_count else "missing"
            if (
                not module_count
                and profile is not None
                and profile.configured_module_count is not None
            ):
                observed_module_count = profile.configured_module_count
                module_count_status = "verified"
                module_count_source = "configured_profile"
            nominal_capacity_ah = effective_nominal_capacity_ah(
                profile,
                observed_module_count,
                module_count_status,
            )
            current_inventory = inventory.get(serial)
            if refresh or current_inventory is None:
                current_inventory = self._refresh_inventory(
                    serial,
                    force=current_inventory is None,
                )
            has_electrothermal_source_audit = _latest_quality_gate_passed(
                source_data
                / "processed"
                / "bms_electrothermal_source_quality"
                / f"serial={serial}"
            )
            validation_root = _workflow_stage_root(
                source_data / "processed", "data_validation", serial
            )
            has_capacity_bms_validation = _capacity_bms_validation_ready(validation_root)
            electrothermal_validation_root = _workflow_stage_root(
                source_data / "processed", "electrothermal_import_validation", serial
            )
            has_electrothermal_validation = _electrothermal_validation_ready(
                electrothermal_validation_root
            )
            electrothermal_source_mtime_ns = _electrothermal_import_mtime_ns(
                source_data, inferred_chemistry or profile.chemistry, serial
            ) if (inferred_chemistry or profile) else 0
            file_count = int(current_inventory.get("raw_file_count") or 0)
            row_count = int(current_inventory.get("raw_row_count") or 0)
            observed_start = _iso(current_inventory.get("observed_start_at"))
            observed_end = _iso(current_inventory.get("observed_end_at"))
            serial_jobs = [item for item in jobs if str(item.get("serial")) == serial]
            import_jobs = [
                item
                for item in serial_jobs
                if item.get("pipeline_key") == "veera.raw_telemetry_import.v1"
            ]
            active_import = next(
                (item for item in import_jobs if item.get("status") in {"queued", "running"}),
                None,
            )
            catalog = catalog_batteries.get(serial, {})
            raw_ready = file_count > 0
            (
                five_minute_file_count,
                five_minute_row_count,
                five_minute_start,
                five_minute_end,
            ) = self._compact_inventory(serial, five_minute_root)
            telemetry_ready = raw_ready or five_minute_file_count > 0
            raw_mtime_ns = int(current_inventory.get("raw_latest_mtime_ns") or 0)
            workflow_cache_key = (
                raw_mtime_ns,
                has_electrothermal_source_audit,
                has_capacity_bms_validation,
                has_electrothermal_validation,
                electrothermal_source_mtime_ns,
                *_workflow_directory_signature(source_data, serial),
            )
            cached_workflow = self._workflow_state_cache.get(serial)
            if refresh or cached_workflow is None or cached_workflow[0] != workflow_cache_key:
                workflow = _workflow_stage_states(
                    source_data,
                    serial,
                    raw_ready=raw_ready,
                    raw_mtime_ns=raw_mtime_ns,
                    electrothermal_quality_ready=has_electrothermal_source_audit,
                    capacity_bms_validation_ready=has_capacity_bms_validation,
                    electrothermal_validation_ready=has_electrothermal_validation,
                    electrothermal_source_mtime_ns=electrothermal_source_mtime_ns,
                )
                self._workflow_state_cache[serial] = (workflow_cache_key, workflow)
            else:
                workflow = cached_workflow[1]
            current_stages = {
                stage
                for stage, state in workflow.items()
                if state.get("status") == "current"
            }
            recommended_import_start = _next_import_timestamp(
                observed_end,
                _iso(five_minute_end),
            )
            devices.append(
                {
                    "serial": serial,
                    "display_name": name_overrides.get(
                        serial,
                        profile.display_name if profile else f"Battery {serial}",
                    ),
                    "display_name_source": (
                        "custom"
                        if serial in name_overrides
                        else "profile"
                        if profile
                        else "fallback"
                    ),
                    "variant_key": profile.variant_key if profile else None,
                    "chemistry": profile.chemistry if profile else inferred_chemistry,
                    "device_role": profile.device_role if profile else "unassigned",
                    "nominal_capacity_ah": nominal_capacity_ah,
                    "configured_nominal_capacity_ah": (
                        profile.nominal_capacity_ah if profile else None
                    ),
                    "module_capacity_ah": profile.module_capacity_ah if profile else None,
                    "module_energy_kwh": profile.module_energy_kwh if profile else None,
                    "loss_slope_reference_module_count": (
                        profile.loss_slope_reference_module_count if profile else None
                    ),
                    "module_count": observed_module_count,
                    "module_count_source": module_count_source,
                    "module_count_agreement": module_count.get("module_count_agreement", 0.0),
                    "system_energy_kwh": (
                        observed_module_count * profile.module_energy_kwh
                        if module_count_status == "verified"
                        and observed_module_count is not None
                        and profile
                        and profile.module_energy_kwh is not None
                        else None
                    ),
                    "observed_module_counts": module_count.get("observed_module_counts", ""),
                    "module_count_status": module_count_status,
                    "configured_start": profile.data_available_from if profile else None,
                    "default_loss_slopes_per_1000ah": (
                        list(profile.default_loss_slopes_per_1000ah) if profile else []
                    ),
                    "raw_status": (
                        "importing"
                        if active_import
                        else "ready"
                        if telemetry_ready
                        else "not_imported"
                    ),
                    "raw_file_count": file_count,
                    "raw_row_count": row_count,
                    "five_minute_file_count": five_minute_file_count,
                    "five_minute_row_count": five_minute_row_count,
                    "five_minute_start_at": _iso(five_minute_start),
                    "five_minute_end_at": _iso(five_minute_end),
                    "observed_start_at": observed_start,
                    "observed_end_at": observed_end,
                    "recommended_import_start_at": recommended_import_start,
                    "latest_import_job_status": (
                        import_jobs[0].get("status") if import_jobs else None
                    ),
                    "run_count": int(catalog.get("run_count") or 0),
                    "latest_run_id": catalog.get("latest_run_id"),
                    "workflow": workflow,
                    "readiness": {
                        "raw": raw_ready,
                        "cell_signal_audit": "cell_signal_audit" in current_stages,
                        "cell_health": "cell_health" in current_stages,
                        "electrothermal_source_audit": (
                            "electrothermal_source_audit" in current_stages
                        ),
                        "electrothermal_validation": (
                            "electrothermal_validation" in current_stages
                        ),
                        "hierarchical_features": "hierarchical_features" in current_stages,
                        "soh_signal_audit": "soh_signal_audit" in current_stages,
                        "data_validation": "data_validation" in current_stages,
                        "discharge_capacity_evidence": (
                            "discharge_capacity_evidence" in current_stages
                        ),
                        "soh_recalibration": "soh_recalibration" in current_stages,
                        "capacity_soh_investigation": (
                            "capacity_soh_investigation" in current_stages
                        ),
                        "seasonal_charge_profile": "seasonal_charge_profile" in current_stages,
                        "soh_gap_diagnosis": "soh_gap_diagnosis" in current_stages,
                        "classified": "classified" in current_stages,
                        "events": "events" in current_stages,
                        "capacity_trend": "capacity_trend" in current_stages,
                        "analysis": bool(catalog.get("run_count")),
                    },
                }
            )
        devices.sort(
            key=lambda item: (
                item["module_count"] is None,
                item["module_count"] or 0,
                item["serial"],
            )
        )
        return devices

    def _refresh_inventory(self, serial: str, *, force: bool) -> dict[str, Any]:
        source_data = self.settings.source_root / "data"
        raw_root = source_data / "raw_parquet"
        serial_root = _single_serial_dataset_path(raw_root, serial)
        files = _parquet_files(serial_root) if serial_root is not None else []
        total_bytes, latest_mtime_ns = _file_signature(files)
        current = self.repository.get_device_inventory(serial)
        signature_changed = (
            current is None
            or int(current.get("raw_file_count") or 0) != len(files)
            or int(current.get("raw_total_bytes") or 0) != total_bytes
            or int(current.get("raw_latest_mtime_ns") or 0) != latest_mtime_ns
        )
        if force or signature_changed:
            row_count, observed_start, observed_end = _parquet_inventory(files)
        else:
            row_count = int(current.get("raw_row_count") or 0)
            observed_start = current.get("observed_start_at")
            observed_end = current.get("observed_end_at")

        item = {
            "serial": serial,
            "raw_file_count": len(files),
            "raw_row_count": row_count,
            "observed_start_at": observed_start,
            "observed_end_at": observed_end,
            "raw_total_bytes": total_bytes,
            "raw_latest_mtime_ns": latest_mtime_ns,
            "has_cell_signal_audit": _has_parquet(
                source_data / "processed" / "cell_signal_audit" / f"serial={serial}"
            ),
            "has_cell_health": _has_parquet(
                source_data / "processed" / "cell_health" / f"serial={serial}"
            ),
            "has_electrothermal_source_audit": _latest_quality_gate_passed(
                source_data / "processed" / "bms_electrothermal_source_quality" / f"serial={serial}"
            ),
            "has_hierarchical_features": _has_parquet(
                source_data / "processed" / "bms_hierarchical_features" / f"serial={serial}"
            ),
            "has_soh_signal_audit": _has_parquet(
                source_data / "processed" / "fleet_soh_audit" / f"serial={serial}"
            ),
            "has_seasonal_charge_profile": _has_parquet(
                source_data / "processed" / "seasonal_charge_analysis" / f"serial={serial}"
            ),
            "has_soh_recalibration": _has_parquet(
                source_data / "processed" / "bms_soh_recalibration" / f"serial={serial}"
            ),
            "has_classified": _has_parquet(
                source_data / "processed" / "classified_rows" / f"serial={serial}"
            ),
            "has_events": _has_parquet(source_data / "processed" / "events" / f"serial={serial}"),
            "has_capacity_trend": _has_parquet(
                source_data / "processed" / "capacity_trend" / f"serial={serial}"
            ),
            "scanned_at": datetime.now(UTC),
        }
        self.repository.upsert_device_inventory(item)
        return item
