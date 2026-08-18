"""Read-only CLI for the restored Phase8 prediction authority.

The prediction CLI deliberately exposes status, registry and validation only.
Training, hyperparameter search and outer evaluation are not implicit commands
in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command_status(_args: argparse.Namespace) -> int:
    path = ROOT / "artifacts" / "migration" / "MIGRATION_TEST_SUMMARY.json"
    if not path.is_file():
        print(
            json.dumps({"status": "NOT_MIGRATED", "training_performed": False}, indent=2)
        )
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": payload.get("restore_status"),
                "active_model_family": payload.get("active_model_family"),
                "active_public_model_class": payload.get("active_public_model_class"),
                "training_performed": payload.get("retraining", False),
                "hpo_performed": payload.get("hpo", False),
                "outer_evaluation_rerun": payload.get("outer_rerun", False),
                "kltn_modified": payload.get("kltn_modified"),
                "scientific_phase8_checkpoints_available": payload.get(
                    "scientific_phase8_checkpoints_available"
                ),
                "recommendation_status": payload.get("recommendation_evidence_status"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def command_report(_args: argparse.Namespace) -> int:
    report = ROOT / "reports" / "migration" / "PHASE8_PREDICTION_RESTORE.md"
    print(json.dumps({"status": "PASS" if report.is_file() else "MISSING", "report": str(report.relative_to(ROOT))}, indent=2))
    return 0 if report.is_file() else 1


def command_validate(_args: argparse.Namespace) -> int:
    summary = ROOT / "artifacts" / "migration" / "MIGRATION_TEST_SUMMARY.json"
    registry = ROOT / "configs" / "prediction" / "registry.json"
    if not summary.is_file() or not registry.is_file():
        print(json.dumps({"status": "MISSING_ACTIVE_AUTHORITY"}, indent=2))
        return 1
    payload = json.loads(summary.read_text(encoding="utf-8"))
    status = payload.get("restore_status") == "PASS" and payload.get("model_equivalence", {}).get("status") == "PASS"
    print(json.dumps({"status": "PASS" if status else "FAIL", "active_registry": str(registry.relative_to(ROOT)), "summary": str(summary.relative_to(ROOT))}, indent=2))
    return 0 if status else 1


def command_registry(_args: argparse.Namespace) -> int:
    path = ROOT / "configs" / "prediction" / "registry.json"
    if not path.is_file():
        print(json.dumps({"status": "MISSING_ACTIVE_REGISTRY"}, indent=2))
        return 1
    print(path.read_text(encoding="utf-8"), end="")
    return 0


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
    prediction = root.add_parser("prediction", help="Restored Phase8 prediction authority.")
    prediction_commands = prediction.add_subparsers(dest="prediction_command", required=True)
    prediction_status = prediction_commands.add_parser("status", help="Show read-only migration state.")
    prediction_status.set_defaults(handler=command_status)
    prediction_validate = prediction_commands.add_parser("validate", help="Validate active authority artifacts.")
    prediction_validate.set_defaults(handler=command_validate)
    prediction_registry = prediction_commands.add_parser("registry", help="Print the active model registry.")
    prediction_registry.set_defaults(handler=command_registry)
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "db-final":
        from scripts.database_final import main as database_final_main

        return int(database_final_main(sys.argv[2:]))
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
