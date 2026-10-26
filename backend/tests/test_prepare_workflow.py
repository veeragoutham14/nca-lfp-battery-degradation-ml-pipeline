from __future__ import annotations

import argparse
from datetime import UTC, datetime

from battryml.config import Settings
from battryml.workflows import device_analysis


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        serial="100000002",
        chemistry="NCA",
        module_count=2,
        update_raw=False,
        start=None,
        end=None,
    )


def test_prepare_runs_local_validation_and_features_without_database(
    platform_settings: Settings, monkeypatch
) -> None:
    stages: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(
        device_analysis,
        "_run_stage",
        lambda name, command, *, cwd: stages.append((name, command)),
    )
    monkeypatch.setattr(
        device_analysis,
        "_raw_bounds",
        lambda *args: (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
    )
    monkeypatch.setattr(device_analysis, "validate_analysis_inputs", lambda args, settings: None)

    device_analysis.prepare_data(_args(), platform_settings)

    names = [name for name, _ in stages]
    assert names == ["electrothermal_validation", "hierarchical_features"]
    features = dict(stages)["hierarchical_features"]
    assert "--start" not in features and "--aggregation-minutes" not in features
    assert features[features.index("--chemistry") + 1] == "NCA"


def test_prepare_keeps_the_source_audit_when_a_database_is_configured(
    platform_settings: Settings, monkeypatch
) -> None:
    stages: list[str] = []
    monkeypatch.setattr(
        device_analysis, "_run_stage", lambda name, command, *, cwd: stages.append(name)
    )
    monkeypatch.setattr(
        device_analysis,
        "_raw_bounds",
        lambda *args: (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)),
    )
    monkeypatch.setattr(device_analysis, "validate_analysis_inputs", lambda args, settings: None)
    monkeypatch.setattr(
        Settings, "source_database_status", lambda self, **kwargs: {"configured": True}
    )

    device_analysis.prepare_data(_args(), platform_settings)

    assert stages[0] == "electrothermal_source_audit"


def test_prepare_needs_the_database_only_for_the_raw_import() -> None:
    from battryml.plugins.device_workflow import PrepareAnalysisDataPlugin
    from battryml.plugins.electrothermal import ElectrothermalSourceAuditPlugin

    plugin = PrepareAnalysisDataPlugin()
    assert plugin.needs_database({"update_raw": False}) is False
    assert plugin.needs_database({"update_raw": True}) is True
    assert ElectrothermalSourceAuditPlugin().needs_database({}) is True
