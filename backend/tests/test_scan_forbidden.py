from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "scan_forbidden.py"


def _load_scanner():
    spec = importlib.util.spec_from_file_location("scan_forbidden", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_scan_tree_skips_nested_skip_dirs_but_reports_other_files(tmp_path: Path) -> None:
    scanner = _load_scanner()
    # Built at runtime so this file never contains a 9-digit literal itself.
    outside_placeholder_range = str(2 * 10**8 + 12_345)
    for relative in (
        "brain/data/processed/report.html",
        "docs/images/generated/plot.html",
        "frontend/dist/assets/index.js",
    ):
        _write(tmp_path, relative, f"id {outside_placeholder_range}\n")
    _write(tmp_path, "brain/src/module.py", f"value = '{outside_placeholder_range}'\n")
    _write(tmp_path, "brain/src/clean.py", "value = 'placeholder'\n")

    hits = scanner.scan_tree(tmp_path, [])

    assert hits == [f"brain/src/module.py: 9-digit id {outside_placeholder_range}"]


def test_scan_tree_reports_configured_needles_without_echoing_them(tmp_path: Path) -> None:
    scanner = _load_scanner()
    _write(tmp_path, "README.md", "mentions SecretMarker here\n")

    hits = scanner.scan_tree(tmp_path, ["secretmarker"])

    assert hits == ["README.md: contains a configured needle"]
