"""Validate the complete four-stage ranking and causal evidence release."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANKER = ROOT / "artifacts/recommend_hybrid/final_stage_aware_v2/FOUR_STAGE_ACTION_HEAD_EVIDENCE.json"
DEFAULT_CAUSAL = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json"
DEFAULT_OUTPUT = ROOT / "reports/recommend_hybrid/STAGE_AWARE_COMPLETE_VALIDATION.json"
STAGES = ["EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"]
ACTIONS = [
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
]


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate(ranker_path: Path, causal_path: Path) -> dict[str, object]:
    ranker = _load(ranker_path)
    causal = _load(causal_path)
    failures: list[str] = []
    if ranker.get("status") != "COMPLETE":
        failures.append("FOUR_STAGE_RANKER_NOT_COMPLETE")
    if ranker.get("stage_order") != STAGES:
        failures.append("FOUR_STAGE_RANKER_STAGE_ORDER_INVALID")
    if ranker.get("action_order") != ACTIONS:
        failures.append("FOUR_STAGE_RANKER_ACTION_ORDER_INVALID")
    if sorted((ranker.get("per_stage") or {}).keys()) != sorted(STAGES):
        failures.append("FOUR_STAGE_RANKER_STAGE_EVIDENCE_INCOMPLETE")
    if ranker.get("frozen_hybrid_modified") is not False:
        failures.append("FROZEN_HYBRID_WAS_MODIFIED")
    if ranker.get("release", {}).get("runtime_authorized") is not False:
        failures.append("RANKER_RUNTIME_AUTHORITY_ESCALATED")
    if len(ranker.get("checkpoint_hashes") or {}) != 15:
        failures.append("FOUR_STAGE_CHECKPOINT_SET_INCOMPLETE")
    if causal.get("status") != "PASS":
        failures.append("CAUSAL_ARTIFACT_VALIDATION_FAILED")
    if causal.get("stage_order") != STAGES:
        failures.append("CAUSAL_STAGE_ORDER_INVALID")
    if causal.get("action_order") != ACTIONS:
        failures.append("CAUSAL_ACTION_ORDER_INVALID")
    trial_count = int(causal.get("trial_count", -1))
    if trial_count != len(STAGES) * len(ACTIONS):
        failures.append("CAUSAL_TRIAL_MATRIX_INCOMPLETE")

    ranker_gate_pass = bool(ranker.get("release", {}).get("main_gates_pass"))
    identifiable_trials = int(causal.get("identifiable_trial_count", 0))
    scientific_status = (
        "RANKING_AND_OBSERVATIONAL_CAUSAL_EVIDENCE_AVAILABLE"
        if ranker_gate_pass and identifiable_trials > 0
        else "TECHNICAL_PIPELINE_COMPLETE_WITH_LIMITED_SCIENTIFIC_EVIDENCE"
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "branch_scope": "FOUR_STAGE_CONDITIONAL_RANKING_PLUS_STAGE_AWARE_TARGET_TRIALS",
        "stage_order": STAGES,
        "action_order": ACTIONS,
        "ranker_main_gates_pass": ranker_gate_pass,
        "ranker_release_status": ranker.get("release", {}).get("status"),
        "causal_identifiable_trial_count": identifiable_trials,
        "causal_trial_count": trial_count,
        "scientific_status": scientific_status,
        "runtime_authorized": False,
        "claim_boundary": (
            "Conditional ranking is offline. Causal estimates, where identifiable, "
            "are observational under stated assumptions and do not prove the effect "
            "of displaying a recommendation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranker", type=Path, default=DEFAULT_RANKER)
    parser.add_argument("--causal", type=Path, default=DEFAULT_CAUSAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = validate(args.ranker, args.causal)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
