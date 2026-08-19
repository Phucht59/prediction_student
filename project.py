"""Read-only CLI for the thesis-final Phase 4 Hybrid authority.

Training, HPO and outer evaluation are not implicit commands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command_status(_args: argparse.Namespace) -> int:
    path = ROOT / "artifacts" / "prediction" / "final" / "FINALIZATION_DECISION.json"
    if not path.is_file():
        print(json.dumps({"status": "NOT_FINALIZED", "training_performed": False}, indent=2))
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": "THESIS_FINAL",
                "model_id": payload.get("model_id"),
                "display_name": payload.get("display_name"),
                "architecture_id": payload.get("architecture_id"),
                "source_phase": payload.get("source_phase"),
                "previous_phase4_gate_status": payload.get("previous_phase4_gate_status"),
                "final_authority_selected": payload.get("final_authority_selected"),
                "evaluation_status": payload.get("evaluation_status"),
                "outer_test_used": payload.get("outer_test_used"),
                "training_performed": False,
                "hpo_performed": False,
            },
            indent=2,
        )
    )
    return 0


def command_report(_args: argparse.Namespace) -> int:
    report = ROOT / "reports" / "prediction" / "final" / "FINAL_PREDICTION_MODEL_REPORT.md"
    print(json.dumps({"status": "PASS" if report.is_file() else "MISSING", "report": str(report.relative_to(ROOT))}, indent=2))
    return 0 if report.is_file() else 1


def command_validate(_args: argparse.Namespace) -> int:
    decision = ROOT / "artifacts" / "prediction" / "final" / "FINALIZATION_DECISION.json"
    registry = ROOT / "configs" / "prediction" / "registry.json"
    report = ROOT / "reports" / "prediction" / "final" / "FINAL_PREDICTION_MODEL_REPORT.md"
    if not decision.is_file() or not registry.is_file() or not report.is_file():
        print(json.dumps({"status": "MISSING_ACTIVE_AUTHORITY"}, indent=2))
        return 1
    payload = json.loads(decision.read_text(encoding="utf-8"))
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    ok = (
        payload.get("final_authority_selected") is True
        and payload.get("outer_test_used") is False
        and payload.get("previous_phase4_gate_status") == "NOT_READY_FOR_FINAL_EVAL"
        and registry_payload.get("xgboost_active") is False
        and registry_payload.get("prediction_model", {}).get("architecture_id") == "C0"
    )
    print(
        json.dumps(
            {
                "status": "PASS" if ok else "FAIL",
                "active_registry": str(registry.relative_to(ROOT)),
                "decision": str(decision.relative_to(ROOT)),
            },
            indent=2,
        )
    )
    return 0 if ok else 1


def command_registry(_args: argparse.Namespace) -> int:
    path = ROOT / "configs" / "prediction" / "registry.json"
    if not path.is_file():
        print(json.dumps({"status": "MISSING_ACTIVE_REGISTRY"}, indent=2))
        return 1
    print(path.read_text(encoding="utf-8"), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical student prediction release.")
    root = parser.add_subparsers(dest="command", required=True)
    final = root.add_parser("final", help="Thesis-final Phase 4 Hybrid authority.")
    commands = final.add_subparsers(dest="final_command", required=True)
    status = commands.add_parser("status", help="Show final authority state.")
    status.set_defaults(handler=command_status)
    report = commands.add_parser("report", help="Show canonical final report path.")
    report.set_defaults(handler=command_report)
    validate = commands.add_parser("validate", help="Validate final authority artifacts.")
    validate.set_defaults(handler=command_validate)
    prediction = root.add_parser("prediction", help="Thesis-final Phase 4 Hybrid authority.")
    prediction_commands = prediction.add_subparsers(dest="prediction_command", required=True)
    prediction_status = prediction_commands.add_parser("status", help="Show final authority state.")
    prediction_status.set_defaults(handler=command_status)
    prediction_validate = prediction_commands.add_parser("validate", help="Validate final authority artifacts.")
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
