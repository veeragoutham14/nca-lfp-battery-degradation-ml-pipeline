from __future__ import annotations

from dataclasses import dataclass

from battryml.catalog.database import CatalogDatabase
from battryml.catalog.repository import CatalogRepository
from battryml.catalog.scanner import LegacyRunScanner
from battryml.config import Settings
from battryml.data.devices import DeviceLibraryService
from battryml.data.series import RunSeriesService
from battryml.jobs import PipelineJobRunner
from battryml.plugins.registry import PluginRegistry, build_default_registry


@dataclass
class Runtime:
    settings: Settings
    database: CatalogDatabase
    repository: CatalogRepository
    scanner: LegacyRunScanner
    devices: DeviceLibraryService
    series: RunSeriesService
    registry: PluginRegistry
    jobs: PipelineJobRunner

    @classmethod
    def build(cls, settings: Settings) -> Runtime:
        settings.ensure_runtime_dirs()
        database = CatalogDatabase(settings.catalog_path)
        database.initialize()
        repository = CatalogRepository(database)
        registry = build_default_registry()
        scanner = LegacyRunScanner(settings, repository)
        devices = DeviceLibraryService(settings, repository)
        series = RunSeriesService(repository)
        jobs = PipelineJobRunner(settings, repository, scanner, registry, devices)
        jobs.recover_incomplete_jobs()
        return cls(settings, database, repository, scanner, devices, series, registry, jobs)
