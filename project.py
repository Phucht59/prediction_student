"""Safe command line interface for the canonical frozen-evidence release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command_status(_args: argparse.Namespace) -> int:
    path = ROOT / "artifacts" / "final" / "final_results.json"
    if not path.is_file():
        print(
            json.dumps({"status": "NOT_BUILT", "training_performed": False}, indent=2)
        )
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "READY",
                "schema_version": payload.get("schema_version"),
                "training_performed": payload.get("training_performed"),
                "comparator_completion_performed": payload.get(
                    "comparator_completion_performed"
                ),
                "dataset_model_rows": {
                    name: len(dataset.get("models", []))
                    for name, dataset in payload.get("datasets", {}).items()
                },
                "future_oulad": payload.get("future_oulad"),
                "expert_status": payload.get("recommendation", {})
                .get("expert_status", {})
                .get("value"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_report(_args: argparse.Namespace) -> int:
    from src.final_release.reports import generate

    generate()
    print(
        json.dumps(
            {
                "status": "PASS",
                "training_performed": True,
                "official_deep_models_retrained": False,
                "report_root": "reports/final",
            },
            indent=2,
        )
    )
    return 0


def command_validate(_args: argparse.Namespace) -> int:
    from src.final_release.validate import main

    return int(main())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical student prediction release."
    )
    root = parser.add_subparsers(dest="command", required=True)
    final = root.add_parser("final", help="Final validated-evidence release.")
    commands = final.add_subparsers(dest="final_command", required=True)
    status = commands.add_parser("status", help="Show canonical release state.")
    status.set_defaults(handler=command_status)
    report = commands.add_parser(
        "report", help="Regenerate reports from canonical JSON."
    )
    report.set_defaults(handler=command_report)
    validate = commands.add_parser(
        "validate", help="Validate evidence, tables and checksums."
    )
    validate.set_defaults(handler=command_validate)
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "db-final":
        from scripts.database_final import main as database_final_main

        return int(database_final_main(sys.argv[2:]))
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
