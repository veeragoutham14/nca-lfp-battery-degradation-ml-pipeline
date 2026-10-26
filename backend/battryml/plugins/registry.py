from __future__ import annotations

from battryml.plugins.base import PipelinePlugin
from battryml.plugins.bms_soh_recalibration import BmsSohRecalibrationPlugin
from battryml.plugins.data_validation import (
    DataValidationPlugin,
    ElectrothermalImportValidationPlugin,
)
from battryml.plugins.device_workflow import (
    CapacitySohInvestigationPlugin,
    PrepareAnalysisDataPlugin,
)
from battryml.plugins.discharge_capacity import DischargeCapacityEvidencePlugin
from battryml.plugins.electrothermal import (
    ElectrothermalSourceAuditPlugin,
    HierarchicalElectrothermalFeaturesPlugin,
    HierarchicalVoltageAnomalyPlugin,
)
from battryml.plugins.seasonal_charge import SeasonalChargeAnalysisPlugin
from battryml.plugins.soh_signal_audit import SohSignalAuditPlugin
from battryml.plugins.veera_fixed_slope import FixedSlopeVerificationPlugin
from battryml.plugins.veera_ingestion import (
    RawTelemetryImportPlugin,
    RawTelemetryImportV2Plugin,
)
from battryml.plugins.veera_preprocessing import (
    BuildEventsPlugin,
    CapacityTrendPlugin,
    ClassifyRowsPlugin,
)
from battryml.plugins.veera_soh import FullCycleSohPlugin


class PluginRegistry:
    def __init__(self, plugins: list[PipelinePlugin] | None = None):
        self._plugins = {plugin.key: plugin for plugin in (plugins or [])}

    def register(self, plugin: PipelinePlugin) -> None:
        if plugin.key in self._plugins:
            raise ValueError(f"Pipeline plugin already registered: {plugin.key}")
        self._plugins[plugin.key] = plugin

    def get(self, key: str) -> PipelinePlugin:
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise KeyError(f"Unknown pipeline plugin: {key}") from exc

    def definitions(self) -> list[dict]:
        return [self._plugins[key].definition() for key in sorted(self._plugins)]


def build_default_registry() -> PluginRegistry:
    return PluginRegistry(
        [
            RawTelemetryImportPlugin(),
            RawTelemetryImportV2Plugin(),
            DataValidationPlugin(),
            ElectrothermalImportValidationPlugin(),
            PrepareAnalysisDataPlugin(),
            ElectrothermalSourceAuditPlugin(),
            HierarchicalElectrothermalFeaturesPlugin(),
            HierarchicalVoltageAnomalyPlugin(),
            SohSignalAuditPlugin(),
            DischargeCapacityEvidencePlugin(),
            CapacitySohInvestigationPlugin(),
            BmsSohRecalibrationPlugin(),
            SeasonalChargeAnalysisPlugin(),
            ClassifyRowsPlugin(),
            BuildEventsPlugin(),
            CapacityTrendPlugin(),
            FullCycleSohPlugin(),
            FixedSlopeVerificationPlugin(),
        ]
    )
