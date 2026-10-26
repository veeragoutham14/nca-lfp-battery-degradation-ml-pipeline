#!/usr/bin/env python3
"""Fail if generic or locally configured markers appear in the tree or history.

Built-in checks are generic: Windows user paths, non-localhost IPv4 addresses,
the tokens project_memory and handover, and 9-digit ids outside the
100000000-100000099 placeholder range.

Extra markers are read from the FORBIDDEN_NEEDLES environment variable and from
a gitignored .forbidden_needles file. Separate entries with newlines or commas.
Lines starting with # are comments. If neither source is set, those checks are
skipped.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNER = Path(__file__).resolve()
NEEDLE_FILE = ROOT / ".forbidden_needles"
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "frontend/dist",
    "brain/data",
    "var",
    "out",
    "docs/images/generated",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".css",
    ".html",
    ".txt",
    ".csv",
    ".example",
    ".sh",
    ".mjs",
    ".gitignore",
    "",
}
GENERIC_NEEDLES = (
    "project_memory",
    "handover",
    "C:\\Users",
    "C:/Users",
)
NINE_DIGIT = re.compile(r"(?<!\d)(\d{9})(?!\d)")
IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)
# 100000000 .. 100000099, built without a 9-digit literal.
_PLACEHOLDER_BASE = 10**8
ALLOWED_SERIALS = {str(_PLACEHOLDER_BASE + offset) for offset in range(100)}


def _split_needles(text: str) -> list[str]:
    pieces: list[str] = []
    for line in text.replace(",", "\n").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            pieces.append(item)
    return pieces


def extra_needles() -> list[str]:
    found = _split_needles(os.environ.get("FORBIDDEN_NEEDLES", ""))
    if NEEDLE_FILE.is_file():
        try:
            found.extend(_split_needles(NEEDLE_FILE.read_text(encoding="utf-8")))
        except OSError:
            pass
    unique: list[str] = []
    seen: set[str] = set()
    for needle in found:
        key = needle.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(needle)
    return unique


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == SCANNER or path.resolve() == NEEDLE_FILE.resolve():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        # Multi-segment entries such as "brain/data" never equal a single path
        # part, so compare them as prefixes of the repo-relative path.
        prefixes = ("/".join(relative.parts[:depth]) for depth in range(1, len(relative.parts)))
        if any(prefix in SKIP_DIRS for prefix in prefixes):
            continue
        if "node_modules" in path.parts or "generated" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".env.example",
            "Makefile",
            "LICENSE",
        }:
            if path.suffix.lower() in {
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".parquet",
                ".pyc",
                ".woff",
                ".woff2",
                ".ico",
                ".duckdb",
                ".tsbuildinfo",
            }:
                continue
        yield path


def _serial_hits(text: str) -> list[str]:
    hits = []
    for match in NINE_DIGIT.finditer(text):
        value = match.group(1)
        if value not in ALLOWED_SERIALS:
            hits.append(value)
    return hits


def _ipv4_hits(text: str) -> list[str]:
    hits = []
    for match in IPV4.finditer(text):
        value = match.group(0)
        if value.startswith("127.") or value == "0.0.0.0":
            continue
        hits.append(value)
    return hits


def _needle_hits(text: str, needles: list[str]) -> list[str]:
    haystack = text.casefold()
    return [needle for needle in needles if needle.casefold() in haystack]


def scan_tree(root: Path, configured: list[str]) -> list[str]:
    hits: list[str] = []
    for path in _iter_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(root)
        for needle in _needle_hits(text, list(GENERIC_NEEDLES)):
            hits.append(f"{relative}: contains {needle}")
        for _needle in _needle_hits(text, configured):
            hits.append(f"{relative}: contains a configured needle")
        for value in _serial_hits(text):
            hits.append(f"{relative}: 9-digit id {value}")
        for value in _ipv4_hits(text):
            hits.append(f"{relative}: non-localhost ipv4 {value}")
    return hits


def scan_history(root: Path, configured: list[str]) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--format=%B"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    text = result.stdout
    hits = []
    for needle in _needle_hits(text, list(GENERIC_NEEDLES) + configured):
        label = needle if needle in GENERIC_NEEDLES else "a configured needle"
        hits.append(f"git history: contains {label}")
    for value in _serial_hits(text):
        hits.append(f"git history: 9-digit id {value}")
    for value in _ipv4_hits(text):
        hits.append(f"git history: non-localhost ipv4 {value}")
    return hits


def main() -> int:
    configured = extra_needles()
    hits = scan_tree(ROOT, configured) + scan_history(ROOT, configured)
    if hits:
        print("Forbidden-string scan failed:")
        for hit in hits:
            print(f"  {hit}")
        return 1
    if configured:
        print(f"Forbidden-string scan: clean ({len(configured)} extra needles)")
    else:
        print("Forbidden-string scan: clean (no extra needles configured)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
