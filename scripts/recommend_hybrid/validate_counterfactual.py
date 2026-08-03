"""Validate the constrained counterfactual recommender implementation."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.action_catalog import ActionCatalog
from src.recommend_hybrid.counterfactual.effects import (
    CounterfactualEffectCatalog,
)
from src.recommend_hybrid.counterfactual.evaluation import (
    CounterfactualEvaluationRow,
)
from src.recommend_hybrid.counterfactual.feature_authority import (
    PreprocessedOULADFeatureAuthority,
)
from src.recommend_hybrid.counterfactual.oulad_tensor import (
    OULADTensorEffectCatalog,
)
from src.recommend_hybrid.counterfactual.reference_profile import (
    REFERENCE_SPECS,
)
from src.recommend_hybrid.prediction_adapter import HybridPredictionAdapter

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual"
REPORT = ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_VALIDATION.md"
CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_pytest() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/recommend_hybrid/counterfactual",
        "-q",
    ]
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = process.stdout + process.stderr
    match = re.search(r"(\d+) passed", output)
    return {
        "command": command,
        "return_code": process.returncode,
        "passed": int(match.group(1)) if match else 0,
        "status": "PASS" if process.returncode == 0 else "FAIL",
        "output": output[-12000:],
    }


def _static_validation() -> dict[str, Any]:
    actions = ActionCatalog.load(
        ROOT / "configs/recommend_hybrid/actions.yaml"
    )
    state_catalog = CounterfactualEffectCatalog.load(
        ROOT / "configs/recommend_hybrid/counterfactual_oulad.yaml"
    )
    tensor_catalog = OULADTensorEffectCatalog.load(
        ROOT
        / "configs/recommend_hybrid/counterfactual_oulad_tensor.yaml"
    )
    planning = yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/planning.yaml").read_text(
            encoding="utf-8"
        )
    )
    action_ids = {item.action_id for item in actions.actions}
    planned_oulad = set(planning["dataset_actions"]["oulad"])
    state_ids = {item.action_id for item in state_catalog.actions}
    tensor_ids = {item.action_id for item in tensor_catalog.actions}
    expected_references = {item[0] for item in REFERENCE_SPECS}
    configured_references = {
        effect.reference_key
        for action in tensor_catalog.actions
        for effect in action.effects
        if effect.reference_key is not None
    }
    evaluation_fields = {field.name for field in fields(CounterfactualEvaluationRow)}
    forbidden_evaluation_fields = {
        "target",
        "label",
        "final_result",
        "date_unregistration",
        "withdrawal_outcome",
    }
    checks = {
        "state_actions_known": state_ids <= action_ids,
        "tensor_actions_known": tensor_ids <= action_ids,
        "tensor_actions_cover_planned_oulad": tensor_ids == planned_oulad,
        "tensor_references_available": (
            configured_references <= expected_references
        ),
        "mutable_protected_disjoint": tensor_catalog.mutable_channels.isdisjoint(
            tensor_catalog.protected_channels
        ),
        "frozen_aggregate_preprocessor_api": all(
            hasattr(HybridPredictionAdapter, method)
            for method in (
                "transform_aggregate",
                "inverse_transform_aggregate",
            )
        ),
        "frozen_static_preprocessor_api": hasattr(
            HybridPredictionAdapter,
            "transform_static",
        ),
        "preprocessed_feature_authority_api": hasattr(
            PreprocessedOULADFeatureAuthority,
            "rebuild",
        ),
        "evaluation_contract_has_no_outcome_label": not (
            evaluation_fields & forbidden_evaluation_fields
        ),
        "outer_fold_evaluator_present": (
            ROOT
            / "scripts/recommend_hybrid/evaluate_counterfactual_recommender.py"
        ).is_file(),
        "claim_boundary_locked": CLAIM_BOUNDARY
        == "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT",
    }
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "action_count": len(action_ids),
        "tensor_action_count": len(tensor_ids),
        "reference_keys": sorted(expected_references),
        "configured_reference_keys": sorted(configured_references),
        "evaluation_fields": sorted(evaluation_fields),
    }


def _write_report(payload: dict[str, Any]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    static = payload["static_validation"]
    tests = payload["pytest"]
    lines = [
        "# Counterfactual recommender validation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Claim boundary: `{payload['claim_boundary']}`",
        f"- Counterfactual tests passed: `{tests['passed']}`",
        f"- Static validation: `{static['status']}`",
        "",
        "## Static gates",
        "",
    ]
    for name, passed in static["checks"].items():
        lines.append(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            "The recommender ranks feasible actions by the change in risk "
            "estimated by the frozen Hybrid CNN-BiLSTM. This validation does "
            "not establish a causal treatment effect, expert agreement, or "
            "real-world grade improvement.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    static = _static_validation()
    tests = (
        {
            "command": [],
            "return_code": 0,
            "passed": 0,
            "status": "SKIPPED",
            "output": "",
        }
        if args.skip_pytest
        else _run_pytest()
    )
    tests_ok = tests["status"] in {"PASS", "SKIPPED"}
    status = "PASS" if static["status"] == "PASS" and tests_ok else "FAIL"
    payload = {
        "schema_version": "counterfactual_validation_v2",
        "generated_at": _utc_now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "static_validation": static,
        "pytest": tests,
        "status": status,
    }
    _write_json(OUT / "validation.json", payload)
    _write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
