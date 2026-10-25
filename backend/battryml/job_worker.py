from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one durable BattryML pipeline command")
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--stdout-path", type=Path, required=True)
    parser.add_argument("--stderr-path", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("No pipeline command was supplied.")

    args.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    args.stderr_path.parent.mkdir(parents=True, exist_ok=True)
    return_code = -1
    error_message: str | None = None
    try:
        with (
            args.stdout_path.open("w", encoding="utf-8") as stdout_file,
            args.stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            completed = subprocess.run(
                command,
                cwd=args.cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                check=False,
            )
        return_code = completed.returncode
    except Exception as exc:
        error_message = str(exc)
        args.stderr_path.write_text(error_message, encoding="utf-8")

    _write_result(
        args.result_path,
        {
            "return_code": return_code,
            "completed_at": datetime.now(UTC).isoformat(),
            "error_message": error_message,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
