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
                "training_performed": False,
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


def command_teacher_feedback_prepare(_args: argparse.Namespace) -> int:
    from src.studies.teacher_feedback import prepare_regression_guard

    result = prepare_regression_guard()
    print(
        json.dumps(
            {
                "status": "PASS",
                "guard": "artifacts/final/teacher_feedback_validation/regression_guard_before.json",
                "official_macro_f1": result["official_macro_f1"],
                "training_performed": False,
            },
            indent=2,
        )
    )
    return 0


def command_teacher_feedback_all(_args: argparse.Namespace) -> int:
    from src.studies.teacher_feedback import run_all

    result = run_all()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def command_teacher_feedback_validate(_args: argparse.Namespace) -> int:
    from src.studies.teacher_feedback import validate_study

    result = validate_study()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


def command_unified_stage(args: argparse.Namespace) -> int:
    from src.studies import unified_stage

    handlers = {
        "prepare": unified_stage.prepare,
        "train": unified_stage.train,
        "evaluate": unified_stage.evaluate,
        "report": unified_stage.report,
        "validate": unified_stage.validate,
        "all": unified_stage.all_steps,
    }
    result = handlers[args.unified_stage_command]()
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "PASS" else 1


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
    study = root.add_parser(
        "study", help="Explicit training/diagnostic studies; never implicit."
    )
    study_commands = study.add_subparsers(dest="study_command", required=True)
    teacher = study_commands.add_parser(
        "teacher-feedback", help="Teacher-feedback evidence completion."
    )
    teacher_commands = teacher.add_subparsers(
        dest="teacher_feedback_command", required=True
    )
    prepare = teacher_commands.add_parser(
        "prepare", help="Freeze the official regression guard without training."
    )
    prepare.set_defaults(handler=command_teacher_feedback_prepare)
    run_all = teacher_commands.add_parser(
        "all",
        help="Explicitly train missing timing/MLP comparators on frozen folds.",
    )
    run_all.set_defaults(handler=command_teacher_feedback_all)
    study_validate = teacher_commands.add_parser(
        "validate", help="Validate already-generated teacher-feedback evidence."
    )
    study_validate.set_defaults(handler=command_teacher_feedback_validate)
    unified = study_commands.add_parser(
        "unified-stage",
        help="One-estimator, three-stage UCI authority.",
    )
    unified_commands = unified.add_subparsers(
        dest="unified_stage_command", required=True
    )
    for name in ("prepare", "train", "evaluate", "report", "validate", "all"):
        command = unified_commands.add_parser(name)
        command.set_defaults(handler=command_unified_stage)
    return parser


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "db-final":
        from scripts.database_final import main as database_final_main

        return int(database_final_main(sys.argv[2:]))
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
