from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.studies.v6.audit import run_knowledge_audit  # noqa: E402
from src.studies.v6.reproduction import reproduce_v5_1  # noqa: E402
from src.studies.v6.pretraining import screen_pretraining  # noqa: E402
from src.studies.v6.multitask import screen_multitask  # noqa: E402
from src.studies.v6.ranking import screen_ranking  # noqa: E402
from src.studies.v6.evaluation import evaluate_final_prediction  # noqa: E402
from src.studies.v6.calibration import calibrate_final_predictions  # noqa: E402
from src.studies.v6.domain import evaluate_domain_generalization  # noqa: E402
from src.studies.v6.expert import export_expert_casebook, import_expert_scores  # noqa: E402
from src.studies.v6.governance import audit_database, build_registries  # noqa: E402
from src.studies.v6.linkage import analyze_linkage  # noqa: E402
from src.studies.v6.recommendation import generate_recommendations  # noqa: E402
from src.studies.v6.reporting import generate_final_report  # noqa: E402
from src.studies.v6.risk_profile import generate_risk_profiles  # noqa: E402
from src.studies.v6.validation import validate_v6  # noqa: E402


def _status() -> dict:
    stages = {
        "audit": "audit/knowledge_audit.json",
        "reproduction": "prediction/v5_1_reproduction.json",
        "pretraining": "prediction/pretraining/gate.json",
        "multitask": "prediction/multitask/gate.json",
        "ranking": "prediction/ranking/gate.json",
        "final_prediction": "prediction/final/run_state.json",
        "domain_generalization": "prediction/domain_generalization/run_state.json",
        "calibration": "prediction/calibration.json",
        "risk_profiles": "prediction/risk_profile_state.json",
        "recommendation": "recommendation/run_state.json",
        "validation": "validation_report.json",
    }
    result = {}
    root = ROOT / "artifacts/v6"
    for name, relative in stages.items():
        path = root / relative
        value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        result[name] = value.get("status", "NOT_STARTED")
    result["future_oulad"] = "LOCKED_NOT_EXECUTED"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V6 stage-gated workflow")
    parser.add_argument(
        "stage",
        choices=[
            "status",
            "audit",
            "reproduce-v5-1",
            "pretrain",
            "train",
            "rank",
            "evaluate",
            "calibrate",
            "risk-profiles",
            "recommend",
            "export-expert-review",
            "import-expert-review",
            "validate",
            "report",
            "run-all",
        ],
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()
    if args.stage == "status":
        result = _status()
    elif args.stage == "audit":
        result = run_knowledge_audit(args.device)
    elif args.stage == "reproduce-v5-1":
        result = reproduce_v5_1(args.device)
    elif args.stage == "pretrain":
        result = screen_pretraining(args.device)
    elif args.stage == "train":
        result = screen_multitask(args.device)
    elif args.stage == "rank":
        result = screen_ranking(args.device)
    elif args.stage == "evaluate":
        result = {
            "prediction": evaluate_final_prediction(args.device),
            "domain": evaluate_domain_generalization(args.device),
        }
    elif args.stage == "calibrate":
        result = calibrate_final_predictions()
    elif args.stage == "risk-profiles":
        result = generate_risk_profiles()
    elif args.stage == "recommend":
        result = generate_recommendations()
    elif args.stage == "export-expert-review":
        result = export_expert_casebook()
    elif args.stage == "import-expert-review":
        result = import_expert_scores(args.file)
    elif args.stage == "validate":
        result = {
            "database": audit_database(),
            "linkage": analyze_linkage(),
            "registry": build_registries(),
            "validation": validate_v6(),
        }
    elif args.stage == "report":
        result = generate_final_report()
    elif args.stage == "run-all":
        result = {
            "audit": run_knowledge_audit(args.device),
            "reproduction": reproduce_v5_1(args.device),
            "pretraining": screen_pretraining(args.device),
            "multitask": screen_multitask(args.device),
            "ranking": screen_ranking(args.device),
            "prediction": evaluate_final_prediction(args.device),
            "domain": evaluate_domain_generalization(args.device),
            "calibration": calibrate_final_predictions(),
            "risk_profiles": generate_risk_profiles(),
            "recommendation": generate_recommendations(),
            "casebook": export_expert_casebook(),
            "expert": import_expert_scores(args.file),
            "database": audit_database(),
            "linkage": analyze_linkage(),
        }
        result["registry"] = build_registries()
        result["validation"] = validate_v6()
        result["report"] = generate_final_report()
    else:  # pragma: no cover - argparse enforces the current stage set
        raise ValueError(args.stage)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
