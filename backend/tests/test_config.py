from __future__ import annotations

from pathlib import Path

from battryml.config import _source_python


def test_configured_source_python_keeps_venv_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base" / "python3"
    base.parent.mkdir()
    base.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base)

    resolved = _source_python(tmp_path, str(venv_python))

    assert resolved == venv_python
    assert resolved != base


def test_login_is_required_unless_explicitly_disabled(monkeypatch, tmp_path: Path) -> None:
    from battryml import config

    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("BATTRYML_AUTH_ENABLED", raising=False)
    assert config.Settings.from_env().auth_enabled is True
    monkeypatch.setenv("BATTRYML_AUTH_ENABLED", "false")
    assert config.Settings.from_env().auth_enabled is False
