from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

import uvicorn

from battryml.api.runtime import Runtime
from battryml.config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BattryML local platform")
    parser.add_argument("--env-file", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scan", help="Import existing Veera_SOH runs into the catalog")
    commands.add_parser("status", help="Show catalog and source status")
    commands.add_parser("pipelines", help="List registered pipeline plugins")
    serve = commands.add_parser("serve", help="Start the FastAPI server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--reload", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(args.env_file)
    if args.command == "serve":
        uvicorn.run(
            "battryml.api.app:create_app",
            factory=True,
            host=args.host or settings.api_host,
            port=args.port or settings.api_port,
            reload=args.reload,
        )
        return 0

    runtime = Runtime.build(settings)
    try:
        if args.command == "scan":
            print(json.dumps(runtime.scanner.scan(), indent=2, default=str))
        elif args.command == "status":
            print(
                json.dumps(
                    {
                        "source_root": str(settings.source_root),
                        "source_problems": settings.validate_source(),
                        "catalog": runtime.repository.counts(),
                    },
                    indent=2,
                )
            )
        elif args.command == "pipelines":
            print(json.dumps(runtime.registry.definitions(), indent=2))
        return 0
    finally:
        runtime.jobs.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
