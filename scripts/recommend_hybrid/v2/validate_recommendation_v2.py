"""Validate Recommendation V2 artefacts and scientific claim boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TAXONOMY = ROOT / "artifacts/recommend_hybrid/v2/taxonomy_audit.json"
DEFAULT_TIMELINESS = ROOT / "artifacts/recommend_hybrid/v2/assessment_timeliness_audit.json"
DEFAULT_SIMULATION = ROOT / "artifacts/recommend_hybrid/v2/simulation_summary.json"
DEFAULT_EVIDENCE = ROOT / "artifacts/recommend_hybrid/v2/FULL_POPULATION_EVIDENCE.json"
DEFAULT_BREAKDOWN = ROOT / "artifacts/recommend_hybrid/v2/STAGE_ACTION_BREAKDOWN.json"
DEFAULT_OUTPUT = ROOT / "reports/recommend_hybrid/v2/RECOMMENDATION_V2_VALIDATION.json"
STAGES = {"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"}
ACTIONS = {
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    taxonomy_path: Path,
    timeliness_path: Path,
    simulation_path: Path,
    evidence_path: Path,
    breakdown_path: Path,
    output_path: Path,
) -> dict[str, object]:
    taxonomy = _load(taxonomy_path)
    timeliness = _load(timeliness_path)
    simulation = _load(simulation_path)
    evidence = _load(evidence_path)
    breakdown = _load(breakdown_path)
    stage_rows = set(evidence.get("eligibility", {}).get("per_stage_test", {}))
    ranking_stages = set(breakdown.get("stages", {}))
    breakdown_actions = {row.get("action_id") for row in breakdown.get("actions", [])}
    action_order = set(evidence.get("action_order", []))
    gates = {
        "taxonomy_complete": taxonomy.get("status") in {"PASS", "REVIEW_REQUIRED"},
        "five_learned_actions": action_order == ACTIONS,
        "governance_outside_ranker": evidence.get("governance_routes_are_outside_ranker") is True,
        "timeliness_candidate_audited": timeliness.get("candidate") == "ASSESSMENT_TIMELINESS",
        "timeliness_candidate_not_auto_activated": timeliness.get("activated_as_learned_action") is False,
        "full_population_evaluated": int(evidence.get("population", {}).get("groups", 0)) > 0,
        "student_split_leakage": int(
            evidence.get("population", {}).get("student_leakage_count", -1)
        )
        == 0,
        "four_stage_eligibility_evidence": stage_rows == STAGES,
        "four_stage_ranking_breakdown": ranking_stages == STAGES,
        "all_action_breakdowns": breakdown_actions == ACTIONS,
        "ranking_baseline_comparison": bool(breakdown.get("baselines")),
        "test_not_used_for_selection": evidence.get("test_used_for_policy_or_weight_selection") is False,
        "breakdown_test_not_used_for_selection": breakdown.get("test_used_for_weight_selection") is False,
        "simulation_complete": simulation.get("status") == "COMPLETE",
        "simulation_all_actions": set(simulation.get("learned_actions", [])) == ACTIONS,
        "simulation_no_constraint_violations": int(
            simulation.get("constraint_violation_count", -1)
        )
        == 0,
        "frozen_hybrid_unchanged": simulation.get("frozen_hybrid_modified") is False,
        "runtime_not_authorized": evidence.get("runtime_authorized") is False,
        "no_causal_claim": evidence.get("claim_boundary")
        == "OFFLINE_EVALUATION_AND_MODEL_SENSITIVITY_NOT_CAUSAL_EFFECT",
    }
    failures = [name for name, passed in gates.items() if not passed]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "gates": gates,
        "failures": failures,
        "scientific_status": (
            "FULL_POPULATION_OFFLINE_EVALUATION_WITH_MODEL_SENSITIVITY"
            if not failures
            else "RECOMMENDATION_V2_NOT_RELEASEABLE"
        ),
        "runtime_authorized": False,
        "claim_boundary": "NO_DEPLOYED_OR_CAUSAL_EFFECT_CLAIM",
        "artefacts": {
            "taxonomy": str(taxonomy_path.relative_to(ROOT)),
            "assessment_timeliness": str(timeliness_path.relative_to(ROOT)),
            "simulation": str(simulation_path.relative_to(ROOT)),
            "evidence": str(evidence_path.relative_to(ROOT)),
            "stage_action_breakdown": str(breakdown_path.relative_to(ROOT)),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"Recommendation V2 validation failed: {failures}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--timeliness", type=Path, default=DEFAULT_TIMELINESS)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--breakdown", type=Path, default=DEFAULT_BREAKDOWN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = run(
        args.taxonomy,
        args.timeliness,
        args.simulation,
        args.evidence,
        args.breakdown,
        args.output,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
