"""Generate agent pseudo reviews for offline experimentation ONLY.

IMPORTANT: These annotations are AGENT_GENERATED_PSEUDO_REVIEW.
They are NOT eligible for final Snorkel LabelModel or any scientific claim.
They are NOT independent external LLM reviews.
Output goes to annotations/pseudo_agent_experiments/ — NOT imports/raw/.
"""
from __future__ import annotations
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.explainable_v2.contracts import CanonicalAction

ANNOTATION_TYPE = "AGENT_GENERATED_PSEUDO_REVIEW"
NOT_ELIGIBLE_FOR_FINAL_LABEL_MODEL = True
NOT_INDEPENDENT = True
NOT_REAL_EXTERNAL_LLM = True


def _rule_score_reviewer_a(case: dict, action: str) -> dict:
    ev = case.get("observed_pre_cutoff_evidence", {})
    inactivity = ev.get("inactivity_streak", 0)
    active_rate = ev.get("active_day_rate", 0.5)
    due_soon = ev.get("assessment_due_soon", ev.get("due_soon_count", 0) > 0)
    abstain = False
    score = 1
    evidence_codes: list[str] = []
    if action == "RECOVER_ENGAGEMENT":
        if inactivity > 3:
            score = 3; evidence_codes.append("INACTIVITY_STREAK_EXCEEDS_THRESHOLD")
        elif inactivity >= 1:
            score = 2; evidence_codes.append("MODERATE_INACTIVITY_OBSERVED")
        else:
            score = 0; evidence_codes.append("ZERO_INACTIVITY_STREAK")
    elif action == "ASSESSMENT_COMPLETION":
        if due_soon:
            score = 3; evidence_codes.append("ASSESSMENT_DUE_SOON_FLAG")
        elif active_rate < 0.4:
            score = 2; evidence_codes.append("LOW_ACTIVE_DAY_RATE")
    elif action == "STUDY_REGULARITY":
        if active_rate < 0.4:
            score = 3; evidence_codes.append("IRREGULAR_VLE_ACCESS_PATTERN")
        elif active_rate <= 0.6:
            score = 2; evidence_codes.append("MODERATE_ACCESS_REGULARITY")
    elif action == "TARGETED_CONTENT_REVIEW":
        score = 2 if active_rate < 0.5 else 1
    elif action == "QUIZ_RETRIEVAL_PRACTICE":
        score = 2 if active_rate >= 0.5 else 1
    return {"score": -1 if abstain else score, "abstain": abstain, "evidence_codes": evidence_codes}


def _rule_score_reviewer_b(case: dict, action: str) -> dict:
    stage = case.get("stage", "EARLY_20")
    ev = case.get("observed_pre_cutoff_evidence", {})
    due_soon = ev.get("assessment_due_soon", ev.get("due_soon_count", 0) > 0)
    contraindications = case.get("contraindications", [])
    score = 1; evidence_codes: list[str] = []; contra = False
    if action in contraindications:
        score = 0; contra = True; evidence_codes.append("ACTION_CONTRAINDICATED")
    elif stage in ("EARLY_20", "EARLY_35"):
        if action in ("STUDY_REGULARITY", "RECOVER_ENGAGEMENT"):
            score = 3; evidence_codes.append("EARLY_STAGE_HABIT_FORMATION")
        elif action == "ASSESSMENT_COMPLETION" and due_soon:
            score = 3; evidence_codes.append("EARLY_ASSESSMENT_DEADLINE")
        elif action == "TARGETED_CONTENT_REVIEW":
            score = 2
    elif stage == "MIDDLE_50":
        score = 3 if action in ("TARGETED_CONTENT_REVIEW", "ASSESSMENT_COMPLETION") else 2
    else:
        score = 3 if action in ("ASSESSMENT_COMPLETION", "QUIZ_RETRIEVAL_PRACTICE") else (2 if action == "TARGETED_CONTENT_REVIEW" else 1)
    return {"score": score, "abstain": False, "evidence_codes": evidence_codes}


def _rule_score_reviewer_c(case: dict, action: str) -> dict:
    risk_band = case.get("risk_band", "MIDDLE")
    uncertainty = case.get("uncertainty_band", "LOW")
    abstain = uncertainty == "HIGH" and random.random() < 0.15
    score = 1; evidence_codes: list[str] = []
    ev = case.get("observed_pre_cutoff_evidence", {})
    due_soon = ev.get("assessment_due_soon", ev.get("due_soon_count", 0) > 0)
    if not abstain:
        if risk_band == "HIGH":
            score = 3 if action in ("ASSESSMENT_COMPLETION", "RECOVER_ENGAGEMENT") else (2 if action == "STUDY_REGULARITY" else 1)
        elif risk_band == "MIDDLE":
            if action in ("STUDY_REGULARITY", "TARGETED_CONTENT_REVIEW", "QUIZ_RETRIEVAL_PRACTICE"):
                score = 2
            elif action == "ASSESSMENT_COMPLETION" and due_soon:
                score = 3
        else:
            score = 0 if action == "RECOVER_ENGAGEMENT" else (2 if action in ("QUIZ_RETRIEVAL_PRACTICE", "STUDY_REGULARITY") else 1)
    return {"score": -1 if abstain else score, "abstain": abstain, "evidence_codes": evidence_codes}


def generate_pseudo_annotations() -> None:
    out_dir = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/pseudo_agent_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_a_file = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/panel_a_cases.jsonl"
    panel_b_file = ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/panel_b_cases.jsonl"

    def load(p: Path) -> list[dict]:
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    panel_a = load(panel_a_file)
    panel_b = load(panel_b_file)

    random.seed(42)
    selected_a = random.sample(panel_a, min(300, len(panel_a)))
    selected_b = random.sample(panel_b, min(150, len(panel_b)))

    configs = [
        ("panel_a", selected_a, "PSEUDO_REVIEWER_A", _rule_score_reviewer_a),
        ("panel_a", selected_a, "PSEUDO_REVIEWER_B", _rule_score_reviewer_b),
        ("panel_a", selected_a, "PSEUDO_REVIEWER_C", _rule_score_reviewer_c),
        ("panel_b", selected_b, "PSEUDO_REVIEWER_A", _rule_score_reviewer_a),
        ("panel_b", selected_b, "PSEUDO_REVIEWER_B", _rule_score_reviewer_b),
        ("panel_b", selected_b, "PSEUDO_REVIEWER_C", _rule_score_reviewer_c),
    ]

    total = 0
    for panel_name, cases, reviewer_id, score_fn in configs:
        out_file = out_dir / f"{panel_name}_{reviewer_id.lower()}_pseudo.jsonl"
        records = []
        for case in cases:
            for act in case.get("feasible_candidate_actions", [a.value for a in CanonicalAction]):
                result = score_fn(case, act)
                rec = {
                    "case_id": case.get("query_id", case.get("case_id", "")),
                    "action_id": act,
                    "reviewer_id": reviewer_id,
                    "reviewer_type": ANNOTATION_TYPE,
                    "model_name": "ANTIGRAVITY_INTERNAL_RULE_AGENT",
                    "provider": "NONE_INTERNAL",
                    "request_id": None,
                    "response_id": None,
                    "not_eligible_for_final_label_model": NOT_ELIGIBLE_FOR_FINAL_LABEL_MODEL,
                    "not_independent": NOT_INDEPENDENT,
                    "not_real_external_llm": NOT_REAL_EXTERNAL_LLM,
                    "relevance_score": result["score"],
                    "abstain": result["abstain"],
                    "evidence_ids": result["evidence_codes"],
                    "rationale": f"Rule-based pseudo review: {reviewer_id} for {act}",
                    "contraindication_detected": False,
                    "safety_flag": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                records.append(rec)
        out_file.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        total += len(records)

    print(f"PSEUDO_ANNOTATION_TOTAL={total}")
    print(f"PSEUDO_ANNOTATION_TYPE=AGENT_GENERATED_PSEUDO_REVIEW")
    print(f"ELIGIBLE_FOR_FINAL_LABEL_MODEL=FALSE")


if __name__ == "__main__":
    generate_pseudo_annotations()
