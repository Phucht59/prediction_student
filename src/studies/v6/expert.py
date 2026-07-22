from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score

from .contract import ARTIFACT_ROOT, REPORT_ROOT, atomic_json, atomic_text
from .recommendation import load_plans


EXPERT_ROOT = ARTIFACT_ROOT / "recommendation/expert_evaluation"
CASEBOOK_ROOT = ARTIFACT_ROOT / "recommendation/casebook"
SCORE_FIELDS = [
    "action_relevant",
    "action_safe",
    "action_feasible",
    "priority_correct",
    "escalation_correct",
    "reason_code_clear",
    "weekly_minutes_reasonable",
    "overall_plan_approved",
]


def export_expert_casebook(case_count: int = 60) -> dict[str, Any]:
    state_path = CASEBOOK_ROOT / "export_state.json"
    if state_path.is_file():
        return json.loads(state_path.read_text(encoding="utf-8"))
    plans = load_plans()
    profiles = pd.read_parquet(ARTIFACT_ROOT / "prediction/risk_profiles.parquet").set_index(
        "record_id"
    )
    ordered = sorted(
        plans,
        key=lambda plan: (
            plan["risk_mechanism"],
            profiles.loc[plan["record_id"], "confidence_level"],
            plan["plan_id"],
        ),
    )
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in ordered:
        key = (
            plan["risk_mechanism"],
            str(profiles.loc[plan["record_id"], "confidence_level"]),
        )
        groups.setdefault(key, []).append(plan)
    selected: list[dict[str, Any]] = []
    while len(selected) < min(case_count, len(plans)):
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < case_count:
                selected.append(groups[key].pop(0))
                progressed = True
        if not progressed:
            break
    cases: list[dict[str, Any]] = []
    for number, plan in enumerate(selected, 1):
        profile = profiles.loc[plan["record_id"]]
        cases.append(
            {
                "case_id": f"CASE-{number:03d}",
                "student_state_summary": {
                    "forecast_id": profile.forecast_id,
                    "course_progress_day": int(profile.cutoff_day),
                    "risk_percentile_band": profile.top_k_bucket,
                    "confidence_level": profile.confidence_level,
                },
                "risk_evidence": {
                    "risk_probability_band": plan["risk_level"],
                    "risk_mechanism": plan["risk_mechanism"],
                    "withdrawal_horizon_band": "ELEVATED"
                    if profile.withdrawal_risk_horizon >= 0.45
                    else "NOT_ELEVATED",
                    "fail_probability_band": "ELEVATED"
                    if profile.probability_fail >= 0.55
                    else "NOT_ELEVATED",
                    "reason_codes": plan["reason_codes"],
                },
                "recommendation_actions": plan["recommended_actions"],
                "weekly_minutes": plan["expected_weekly_minutes"],
                "priority": plan["priority"],
                "requires_expert_review": plan["requires_expert_review"],
                "escalation_reason": plan["escalation_reason"],
            }
        )
    CASEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    (CASEBOOK_ROOT / "blinded_cases.jsonl").write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    template = pd.DataFrame(
        [
            {
                "case_id": case["case_id"],
                **{field: "" for field in SCORE_FIELDS},
                "free_text_comment": "",
            }
            for case in cases
        ]
    )
    template.to_csv(CASEBOOK_ROOT / "expert_a_scores.csv", index=False)
    template.to_csv(CASEBOOK_ROOT / "expert_b_scores.csv", index=False)
    state = {
        "schema_version": "v6_blinded_expert_casebook_v1",
        "status": "PENDING_EXPERT_LABELS",
        "cases": len(cases),
        "required_experts": 2,
        "templates": ["expert_a_scores.csv", "expert_b_scores.csv"],
        "model_identity_exposed": False,
        "synthetic_labels_created": False,
    }
    atomic_json(state_path, state)
    return state


def import_expert_scores(path: Path | None = None) -> dict[str, Any]:
    output = EXPERT_ROOT / "metrics.json"
    if path is None or not path.is_file():
        result = {
            "schema_version": "v6_expert_evaluation_v1",
            "status": "PENDING_EXPERT_LABELS",
            "experts": 0,
            "cases_scored": 0,
            "action_f1": None,
            "top_3_action_recall": None,
            "plan_approval_rate": None,
            "escalation_f1": None,
            "agreement": None,
            "synthetic_labels_created": False,
        }
        atomic_json(output, result)
        atomic_text(
            REPORT_ROOT / "EXPERT_EVALUATION_REPORT.md",
            """# V6 expert evaluation report

Status: **PENDING_EXPERT_LABELS**.

A blinded 60-case package and two independent scoring templates were exported.
No expert score was fabricated or inferred. Recommendation effectiveness is not
claimed; metrics remain pending until a real completed file is supplied.
""",
        )
        return result
    scores = pd.read_csv(path)
    required = {"expert_id", "case_id", *SCORE_FIELDS}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Expert score fields missing: {sorted(missing)}")
    if scores.expert_id.nunique() < 2:
        raise ValueError("At least two independent experts are required")
    for field in SCORE_FIELDS:
        if not scores[field].isin([0, 1]).all():
            raise ValueError(f"Expert field must contain real binary scores: {field}")
    expert_ids = sorted(scores.expert_id.unique())[:2]
    first = scores[scores.expert_id.eq(expert_ids[0])].set_index("case_id")
    second = scores[scores.expert_id.eq(expert_ids[1])].set_index("case_id")
    shared = first.index.intersection(second.index)
    agreement = float(
        cohen_kappa_score(
            first.loc[shared, "overall_plan_approved"],
            second.loc[shared, "overall_plan_approved"],
        )
    )
    relevant = scores.action_relevant.to_numpy(dtype=int)
    approved = scores.overall_plan_approved.to_numpy(dtype=int)
    result = {
        "schema_version": "v6_expert_evaluation_v1",
        "status": "COMPLETE",
        "experts": int(scores.expert_id.nunique()),
        "cases_scored": int(scores.case_id.nunique()),
        "action_precision": float(precision_score(approved, relevant, zero_division=0)),
        "action_recall": float(recall_score(approved, relevant, zero_division=0)),
        "action_f1": float(f1_score(approved, relevant, zero_division=0)),
        "top_3_action_recall": None,
        "plan_approval_rate": float(scores.overall_plan_approved.mean()),
        "escalation_f1": float(scores.escalation_correct.mean()),
        "agreement": agreement,
        "synthetic_labels_created": False,
    }
    atomic_json(output, result)
    return result


__all__ = ["export_expert_casebook", "import_expert_scores"]
