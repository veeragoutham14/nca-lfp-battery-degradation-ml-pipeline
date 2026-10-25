from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from battryml.data.storage_paths import chemistry_parent, normalize_chemistry

DATASET_FAMILIES = ("raw_parquet", "telemetry_5min", "import_manifests")


def _inventory(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _rewrite_manifest_paths(
    manifest_root: Path,
    replacements: dict[str, str],
) -> dict[Path, str]:
    originals: dict[Path, str] = {}
    for path in manifest_root.rglob("import_manifest.json"):
        original = path.read_text(encoding="utf-8")
        payload = json.loads(original)
        changed = False
        for result in payload.get("results", []):
            files = result.get("files", [])
            rewritten: list[str] = []
            for value in files:
                updated = str(value)
                for old, new in replacements.items():
                    if updated.startswith(old):
                        updated = new + updated[len(old) :]
                        changed = True
                        break
                rewritten.append(updated)
            result["files"] = rewritten
        if not changed:
            continue
        originals[path] = original
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)
    return originals


def migrate_serial_storage(
    source_root: Path,
    *,
    serial: str,
    chemistry: str,
    apply: bool,
) -> list[tuple[Path, Path]]:
    normalized = normalize_chemistry(chemistry)
    data_root = source_root / "data"
    plans: list[tuple[Path, Path]] = []
    for family in DATASET_FAMILIES:
        family_root = data_root / family
        source = family_root / f"serial={serial}"
        destination = chemistry_parent(family_root, normalized) / f"serial={serial}"
        if not source.exists():
            continue
        if destination.exists():
            raise ValueError(f"Migration destination already exists: {destination}")
        plans.append((source, destination))
    if not plans:
        raise ValueError(f"No legacy telemetry storage found for battery {serial}")
    if not apply:
        return plans

    before = {source: _inventory(source) for source, _ in plans}
    moved: list[tuple[Path, Path]] = []
    manifest_originals: dict[Path, str] = {}
    try:
        for source, destination in plans:
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moved.append((source, destination))
            if _inventory(destination) != before[source]:
                raise RuntimeError(f"Post-move inventory mismatch for {destination}")

        path_replacements = {str(source): str(destination) for source, destination in plans}
        manifest_plan = next(
            (
                (source, destination)
                for source, destination in plans
                if source.parent.name == "import_manifests"
            ),
            None,
        )
        if manifest_plan is not None:
            manifest_originals = _rewrite_manifest_paths(
                manifest_plan[1],
                path_replacements,
            )
        return plans
    except Exception:
        for path, original in manifest_originals.items():
            path.write_text(original, encoding="utf-8")
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                destination.rename(source)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move one legacy serial into chemistry-partitioned telemetry storage."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--chemistry", required=True, choices=("NCA", "LFP"))
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plans = migrate_serial_storage(
        args.source_root,
        serial=args.serial,
        chemistry=args.chemistry,
        apply=args.apply,
    )
    action = "Moved" if args.apply else "Would move"
    for source, destination in plans:
        print(f"{action}: {source} -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
