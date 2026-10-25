from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from battryml.battery_profiles import battery_profile, battery_profiles, nominal_capacity_defaults
from battryml.config import Settings
from battryml.plugins.bms_soh_recalibration import BmsSohRecalibrationPlugin
from battryml.plugins.cell_health import (
    CellHealthFeaturesPlugin,
    CellSignalAuditPlugin,
    CellVoltageAnomalyPlugin,
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
from battryml.plugins.veera_ingestion import RawTelemetryImportPlugin
from battryml.plugins.veera_preprocessing import (
    BuildEventsPlugin,
    CapacityTrendPlugin,
    ClassifyRowsPlugin,
)
from battryml.plugins.veera_soh import FullCycleSohPlugin


def _path_ends(value: str, *parts: str) -> bool:
    """Compare a path tail without assuming a Windows separator."""
    return Path(value).parts[-len(parts) :] == parts


def test_raw_import_plugin_builds_source_extractor_command(
    platform_settings: Settings,
) -> None:
    command = RawTelemetryImportPlugin().build_command(
        platform_settings,
        {
            "serial": "100000011",
            "chemistry": "NCA",
            "start": "2023-08-01T00:00:00+00:00",
            "end": "2026-06-01T00:00:00+00:00",
            "chunk_size": 50_000,
            "compression": "zstd",
            "soc_resolution": "1s",
            "soh_resolution": "5m",
            "voltage_resolution": "both",
            "temperature_resolution": "5m",
            "current_resolution": "off",
            "limit": 1_000_000,
        },
    )

    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.extract_configurable_telemetry",
    )
    assert command[command.index("--serial") + 1] == "100000011"
    assert command[command.index("--chemistry") + 1] == "NCA"
    assert command[command.index("--limit") + 1] == "1000000"
    assert "voltage=both" in command
    assert "current=off" in command
    assert _path_ends(
        command[command.index("--import-profile-catalog") + 1],
        "config",
        "telemetry_import_profiles.json",
    )
    assert _path_ends(
        command[command.index("--output-dir") + 1],
        "data",
        "raw_parquet",
        "chemistry=NCA",
    )
    assert _path_ends(
        command[command.index("--five-minute-output-dir") + 1],
        "data",
        "telemetry_5min",
        "chemistry=NCA",
    )


def test_raw_import_plugin_allows_database_discovered_range(
    platform_settings: Settings,
) -> None:
    command = RawTelemetryImportPlugin().build_command(
        platform_settings,
        {
            "serial": "100000011",
            "chemistry": "NCA",
            "soc_resolution": "5m",
        },
    )

    assert "--start" not in command
    assert "--end" not in command
    assert "soc=5m" in command


def test_soh_signal_audit_plugin_uses_platform_interpreter(
    platform_settings: Settings,
) -> None:
    command = SohSignalAuditPlugin().build_command(
        platform_settings,
        {"serial": "100000002"},
    )

    assert command[:3] == (
        command[0],
        "-m",
        "battryml.analytics.soh_signal_audit",
    )
    assert _path_ends(
        command[command.index("--raw-root") + 1],
        "data",
        "raw_parquet",
        "chemistry=NCA",
    )
    assert command[command.index("--nominal-capacity-ah") + 1] == "64.0"


def test_raw_import_plugin_rejects_invalid_range(platform_settings: Settings) -> None:
    with pytest.raises(ValueError, match="start must be earlier than end"):
        RawTelemetryImportPlugin().build_command(
            platform_settings,
            {
                "serial": "100000011",
                "chemistry": "NCA",
                "start": "2026-06-01T00:00:00+00:00",
                "end": "2023-08-01T00:00:00+00:00",
            },
        )


def test_raw_import_plugin_rejects_configured_chemistry_mismatch(
    platform_settings: Settings,
) -> None:
    with pytest.raises(ValueError, match="configured as NCA"):
        RawTelemetryImportPlugin().build_command(
            platform_settings,
            {
                "serial": "100000011",
                "chemistry": "LFP",
                "soc_resolution": "5m",
            },
        )


def test_cell_signal_audit_plugin_builds_source_command(
    platform_settings: Settings,
) -> None:
    command = CellSignalAuditPlugin().build_command(
        platform_settings,
        {
            "serial": "100000001",
            "module_count": 2,
            "start": "2026-07-01T00:00:00+00:00",
            "end": "2026-07-02T00:00:00+00:00",
        },
    )

    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.cell_signal_audit",
    )
    assert command[command.index("--module-count") + 1] == "2"
    assert command[command.index("--readiness-threshold") + 1] == "0.95"


def test_cell_health_feature_plugin_requires_valid_range(
    platform_settings: Settings,
) -> None:
    plugin = CellHealthFeaturesPlugin()
    command = plugin.build_command(
        platform_settings,
        {
            "serial": "100000001",
            "start": "2023-08-01T00:00:00+00:00",
            "end": "2026-07-02T00:00:00+00:00",
            "aggregation_minutes": "5",
        },
    )

    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.cell_health_features",
    )
    assert command[command.index("--aggregation-minutes") + 1] == "5"

    with pytest.raises(ValueError, match="start must be earlier than end"):
        plugin.validate(
            {
                "serial": "100000001",
                "start": "2026-07-02T00:00:00+00:00",
                "end": "2026-07-01T00:00:00+00:00",
            }
        )


def test_cell_voltage_anomaly_plugin_builds_tuned_source_command(
    platform_settings: Settings,
) -> None:
    command = CellVoltageAnomalyPlugin().build_command(
        platform_settings,
        {"serial": "100000001"},
    )

    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.cell_voltage_anomaly",
    )
    assert command[command.index("--baseline-quantile") + 1] == "0.995"
    assert command[command.index("--absolute-threshold-floor-mv") + 1] == "25.0"
    assert command[command.index("--absolute-critical-spread-mv") + 1] == "100.0"
    assert command[command.index("--minimum-episode-bins") + 1] == "3"

    with pytest.raises(ValueError, match="baseline_quantile"):
        CellVoltageAnomalyPlugin().validate({"serial": "100000001", "baseline_quantile": 1.0})

    reference_command = CellVoltageAnomalyPlugin().build_command(
        platform_settings,
        {"serial": "100000017"},
    )
    assert reference_command[reference_command.index("--serial") + 1] == "100000017"


def test_hierarchical_electrothermal_plugins_build_versioned_commands(
    platform_settings: Settings,
) -> None:
    audit = ElectrothermalSourceAuditPlugin().build_command(
        platform_settings,
        {"serial": "100000001", "module_count": 2},
    )
    features = HierarchicalElectrothermalFeaturesPlugin().build_command(
        platform_settings,
        {
            "serial": "100000001",
            "module_count": 2,
            "chemistry": "NCA",
        },
    )
    anomaly = HierarchicalVoltageAnomalyPlugin().build_command(
        platform_settings,
        {"serial": "100000001"},
    )

    assert audit[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.bms_electrothermal_source_audit",
    )
    assert features[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.build_local_electrothermal_features",
    )
    assert anomaly[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.hierarchical_voltage_anomaly",
    )
    assert HierarchicalElectrothermalFeaturesPlugin.requires_stage == "electrothermal_validation"
    assert HierarchicalElectrothermalFeaturesPlugin.requires_database is False
    assert HierarchicalVoltageAnomalyPlugin.requires_stage == "hierarchical_features"


def test_seasonal_charge_plugin_builds_chemistry_tagged_command(
    platform_settings: Settings,
) -> None:
    plugin = SeasonalChargeAnalysisPlugin()
    command = plugin.build_command(
        platform_settings,
        {"serial": "100000001", "module_count": 2},
    )

    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.seasonal_charge_analysis",
    )
    assert command[command.index("--chemistry") + 1] == "NCA"
    assert command[command.index("--nominal-capacity-ah") + 1] == "64.0"
    assert command[command.index("--module-count") + 1] == "2"
    assert plugin.dependency_stages() == ("capacity_soh_investigation",)
    assert "--raw-root" in command
    assert "--recalibration-root" in command

    reference = plugin.validate({"serial": "100000017", "module_count": 2})
    assert reference["nominal_capacity_ah"] == 56.0

    with pytest.raises(ValueError, match="module_count"):
        plugin.validate({"serial": "100000001", "module_count": 0})


def test_bms_soh_recalibration_plugin_builds_compact_evidence_command(
    platform_settings: Settings,
) -> None:
    plugin = BmsSohRecalibrationPlugin()
    command = plugin.build_command(
        platform_settings,
        {"serial": "100000001", "module_count": 2},
    )

    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.bms_soh_recalibration_analysis",
    )
    assert command[command.index("--chemistry") + 1] == "NCA"
    assert command[command.index("--nominal-capacity-ah") + 1] == "64.0"
    assert command[command.index("--context-lookback-hours") + 1] == "72"
    assert plugin.dependency_stages() == (
        "discharge_capacity_evidence",
        "soh_signal_audit",
    )

    with pytest.raises(ValueError, match="ordered"):
        plugin.validate(
            {
                "serial": "100000001",
                "module_count": 2,
                "low_soc_pct": 99.0,
                "near_full_soc_pct": 99.0,
            }
        )


def test_capacity_investigation_plugins_form_one_recoverable_workflow(
    platform_settings: Settings,
) -> None:
    evidence = DischargeCapacityEvidencePlugin()
    command = evidence.build_command(
        platform_settings,
        {"serial": "100000001", "module_count": 2},
    )
    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.discharge_capacity_evidence",
    )
    assert evidence.dependency_stages() == ("hierarchical_features",)

    investigation = CapacitySohInvestigationPlugin()
    command = investigation.build_command(
        platform_settings,
        {"serial": "100000001", "module_count": 2},
    )
    assert command[2:5] == (
        "battryml.workflows.device_analysis",
        "investigate",
        "--serial",
    )
    assert investigation.scan_output_families == (
        "discharge_derived_usable_capacity_evidence",
        "bms_soh_recalibration_evidence",
    )
    assert investigation.dependency_stages() == ("data_validation", "events")

    preparation = PrepareAnalysisDataPlugin()
    command = preparation.build_command(
        platform_settings,
        {
            "serial": "100000001",
            "module_count": 2,
            "update_raw": True,
            "start": "2026-08-01T00:00:00+00:00",
        },
    )
    assert command[2:5] == (
        "battryml.workflows.device_analysis",
        "prepare",
        "--serial",
    )
    assert "--update-raw" in command
    assert command[command.index("--start") + 1] == "2026-08-01T00:00:00+00:00"


def test_nca_home_storage_profiles_cover_the_19_new_devices() -> None:
    profiles = battery_profiles()
    new_field_devices = {
        "100000001",
        "100000002",
        "100000004",
        "100000005",
        "100000006",
        "100000007",
        "100000008",
        "100000009",
        "100000010",
        "100000012",
        "100000013",
        "100000014",
        "100000015",
        "100000016",
        "100000018",
        "100000019",
        "100000029",
        "100000030",
        "100000031",
    }
    assert new_field_devices <= set(profiles)
    assert len(new_field_devices) == 19
    assert {profiles[serial].display_name for serial in new_field_devices} == {
        f"NCA Field {index:02d}" for index in range(1, 20)
    }
    assert len({profile.display_name for profile in profiles.values()}) == len(profiles)
    # LFP field devices are configured without a verified nameplate; NCA keeps
    # its own chemistry-specific assumptions.
    assert {profile.chemistry for profile in profiles.values()} == {"NCA", "LFP"}
    nca_serials = {
        serial for serial, profile in profiles.items() if profile.chemistry == "NCA"
    }
    lfp_serials = {
        serial for serial, profile in profiles.items() if profile.chemistry == "LFP"
    }
    assert new_field_devices <= nca_serials
    assert len(lfp_serials) == 19
    assert {profiles[serial].nominal_capacity_ah for serial in lfp_serials} == {None}
    assert {profiles[serial].nominal_capacity_ah for serial in new_field_devices} == {64.0}
    assert {profiles[serial].module_capacity_ah for serial in new_field_devices} == {32.0}
    assert {profiles[serial].module_energy_kwh for serial in new_field_devices} == {7.0}
    assert battery_profile("100000001").data_available_from == "2023-08-01"
    assert battery_profile("100000011").data_available_from == "2023-03-15"
    assert battery_profile("100000017").data_available_from == "2022-05-01"
    assert battery_profile("100000020").data_available_from == "2024-11-01"
    assert {profiles[serial].loss_slope_reference_module_count for serial in new_field_devices} == {
        2
    }
    assert battery_profile("100000011").device_role == "field_tester"
    assert battery_profile("100000011").display_name == "NCA Field Reference"
    assert battery_profile("100000017").device_role == "full_cycle_test"
    assert battery_profile("100000017").display_name == "NCA Full-Cycle Reference"
    assert battery_profile("100000017").nominal_capacity_ah == 56.0


def test_fixed_slope_plugin_applies_field_device_defaults(
    platform_settings: Settings,
) -> None:
    plugin = FixedSlopeVerificationPlugin()
    normalized = plugin.validate({"serial": "100000001"})
    assert normalized["nominal_capacity_ah"] == 64.0
    assert normalized["loss_slopes_per_1000ah"] == "0.12"

    command = plugin.build_command(platform_settings, {"serial": "100000001"})
    assert command[command.index("--nominal-capacity-ah") + 1] == "64.0"
    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.fixed_slope_verification",
    )

    nominal_parameter = next(
        parameter
        for parameter in plugin.definition()["parameters"]
        if parameter["key"] == "nominal_capacity_ah"
    )
    assert nominal_parameter["serial_defaults"]["100000001"] == 64.0


def test_fixed_slope_plugin_rejects_lfp_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "battryml.plugins.base.battery_profile",
        lambda _serial: SimpleNamespace(device_role="field_tester", chemistry="LFP"),
    )

    with pytest.raises(ValueError, match="available for NCA"):
        FixedSlopeVerificationPlugin().validate(
            {
                "serial": "100000042",
                "nominal_capacity_ah": battery_profile("100000001").nominal_capacity_ah,
                "loss_slopes_per_1000ah": "0.08",
            }
        )


def test_plugins_enforce_device_roles() -> None:
    with pytest.raises(ValueError, match="assigned as field_tester"):
        FullCycleSohPlugin().validate({"serial": "100000001"})
    with pytest.raises(ValueError, match="assigned as full_cycle_test"):
        FixedSlopeVerificationPlugin().validate({"serial": "100000017"})
    with pytest.raises(ValueError, match="no assigned device role"):
        ClassifyRowsPlugin().validate({"serial": "100000003"})


def test_full_cycle_pipeline_is_available_for_reference_device(
    platform_settings: Settings,
) -> None:
    command = FullCycleSohPlugin().build_command(
        platform_settings,
        {"serial": "100000017"},
    )
    assert command[:3] == (
        str(platform_settings.source_python),
        "-m",
        "src.capacity_ml",
    )
    assert "--nominal-capacity-ah" not in command


@pytest.mark.parametrize(
    ("plugin", "module"),
    [
        (ClassifyRowsPlugin(), "src.classify_rows"),
        (BuildEventsPlugin(), "src.build_events_from_classified"),
        (CapacityTrendPlugin(), "src.discharge_cycle_builder"),
    ],
)
def test_preprocessing_plugins_build_expected_source_commands(
    platform_settings: Settings,
    plugin: ClassifyRowsPlugin | BuildEventsPlugin | CapacityTrendPlugin,
    module: str,
) -> None:
    serial = "100000017" if isinstance(plugin, CapacityTrendPlugin) else "100000001"
    command = plugin.build_command(platform_settings, {"serial": serial})

    assert command[:3] == (str(platform_settings.source_python), "-m", module)
    assert command[command.index("--serial") + 1] == serial


LFP_SERIALS = (
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
)


def test_lfp_variants_load_as_field_testers_without_nameplate_defaults() -> None:
    profiles = battery_profiles()
    lfp = {serial: profiles[serial] for serial in LFP_SERIALS}

    assert len(lfp) == len(LFP_SERIALS)
    for serial, profile in lfp.items():
        assert profile.chemistry == "LFP", serial
        assert profile.device_role == "field_tester", serial
        # No verified LFP nameplate exists yet: capacity defaults must stay
        # unset so operators supply the value per job instead of inheriting
        # an NCA assumption.
        assert profile.nominal_capacity_ah is None, serial
        assert profile.module_capacity_ah is None, serial
        assert profile.default_loss_slopes_per_1000ah == (), serial


def test_lfp_serial_has_no_nominal_capacity_default() -> None:
    assert ("100000020", battery_profile("100000001").nominal_capacity_ah) not in nominal_capacity_defaults()


def test_prepare_analysis_plugin_builds_lfp_command() -> None:
    plugin = PrepareAnalysisDataPlugin()
    values = plugin.validate(
        {
            "serial": "100000020",
            "module_count": 2,
            "nominal_capacity_ah": 51.2,
        }
    )
    command = plugin.build_command(None, values)

    assert command[command.index("--chemistry") + 1] == "LFP"
    assert "--update-raw" not in command


def test_prepare_analysis_plugin_requires_capacity_for_lfp_serial() -> None:
    with pytest.raises(ValueError, match="Missing required parameter: nominal_capacity_ah"):
        PrepareAnalysisDataPlugin().validate({"serial": "100000020", "module_count": 2})


def test_raw_import_plugin_builds_lfp_storage_command(
    platform_settings: Settings,
) -> None:
    command = RawTelemetryImportPlugin().build_command(
        platform_settings,
        {
            "serial": "100000020",
            "chemistry": "LFP",
            "soc_resolution": "1s",
            "soh_resolution": "1s",
            "current_resolution": "1s",
        },
    )

    assert command[command.index("--serial") + 1] == "100000020"
    assert command[command.index("--chemistry") + 1] == "LFP"
    output_dir = Path(command[command.index("--output-dir") + 1])
    assert output_dir.name == "chemistry=LFP"
    assert output_dir.parent.name == "raw_parquet"
    assert "--field-resolution" in command
