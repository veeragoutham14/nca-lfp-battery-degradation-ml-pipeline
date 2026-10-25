from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_CHEMISTRIES = ("NCA", "LFP")


@dataclass(frozen=True)
class DatasetLocation:
    parent_root: Path
    serial_root: Path
    layout: str


def normalize_chemistry(chemistry: str) -> str:
    normalized = str(chemistry).strip().upper()
    if normalized not in SUPPORTED_CHEMISTRIES:
        raise ValueError(
            f"Unsupported chemistry {chemistry!r}; expected one of: "
            f"{', '.join(SUPPORTED_CHEMISTRIES)}"
        )
    return normalized


def chemistry_parent(root: Path, chemistry: str) -> Path:
    return root / f"chemistry={normalize_chemistry(chemistry)}"


def serial_dataset_paths(root: Path, serial: str) -> tuple[Path, ...]:
    paths: list[Path] = []
    legacy = root / f"serial={serial}"
    if legacy.exists():
        paths.append(legacy)
    for chemistry in SUPPORTED_CHEMISTRIES:
        candidate = chemistry_parent(root, chemistry) / f"serial={serial}"
        if candidate.exists():
            paths.append(candidate)
    return tuple(paths)


def resolve_dataset_location(
    root: Path,
    *,
    serial: str,
    chemistry: str,
    for_write: bool = False,
) -> DatasetLocation:
    normalized = normalize_chemistry(chemistry)
    legacy = root / f"serial={serial}"
    canonical_parent = chemistry_parent(root, normalized)
    canonical = canonical_parent / f"serial={serial}"
    wrong_chemistry = [
        chemistry_parent(root, other) / f"serial={serial}"
        for other in SUPPORTED_CHEMISTRIES
        if other != normalized
        and (chemistry_parent(root, other) / f"serial={serial}").exists()
    ]
    if wrong_chemistry:
        raise ValueError(
            f"Battery {serial} already has {wrong_chemistry[0].parent.name} storage and "
            f"cannot be opened as {normalized}."
        )
    if legacy.exists() and canonical.exists():
        raise ValueError(
            f"Battery {serial} exists in both legacy and {normalized} chemistry storage. "
            "Resolve the duplicate layout before continuing."
        )
    if canonical.exists():
        return DatasetLocation(canonical_parent, canonical, "chemistry_partitioned")
    if legacy.exists():
        return DatasetLocation(root, legacy, "legacy")
    if for_write:
        return DatasetLocation(canonical_parent, canonical, "chemistry_partitioned")
    return DatasetLocation(canonical_parent, canonical, "missing")


def discover_serial_chemistries(root: Path) -> dict[str, str]:
    discovered: dict[str, str] = {}
    if not root.exists():
        return discovered
    for chemistry in SUPPORTED_CHEMISTRIES:
        parent = chemistry_parent(root, chemistry)
        if not parent.exists():
            continue
        for path in parent.glob("serial=*"):
            if path.is_dir():
                discovered[path.name.removeprefix("serial=")] = chemistry
    return discovered
