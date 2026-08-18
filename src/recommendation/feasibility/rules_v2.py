"""Versioned Progress Monitoring feasibility. Historical v1 artifacts stay frozen."""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .rules import ACTION_IDS, evaluate_feasibility as evaluate_feasibility_v1

RULE_VERSION = "recommendation.feasibility.v2"
LEGACY_TO_FINAL = {
    "A1": "assessment_recovery",
    "A2": "re_engagement",
    "A3": "study_planning",
    "A4": "progress_monitoring",
    "A5": "retrieval_practice",
}
FINAL_TO_LEGACY = {value: key for key, value in LEGACY_TO_FINAL.items()}


def evaluate_feasibility_v2(row: Mapping, action_id: str) -> tuple[str, str, str]:
    key = FINAL_TO_LEGACY.get(action_id, action_id)
    if key == "A4":
        if "case_id" not in row or "stage" not in row:
            return "UNKNOWN", "PROGRESS_STATE_INCOMPLETE", "student_state"
        return "FEASIBLE", "PROGRESS_STATE_OBSERVED", "student_state"
    return evaluate_feasibility_v1(row, key)


def build_feasibility_frame_v2(state: pd.DataFrame, *, action_ids=ACTION_IDS) -> pd.DataFrame:
    required = {"case_id", "stage", "missing_assessments", "vle_available", "quiz_activity"}
    missing = required.difference(state.columns)
    if missing:
        raise ValueError(f"state missing feasibility features: {sorted(missing)}")
    rows = []
    for row in state.itertuples(index=False):
        values = row._asdict()
        for action_id in action_ids:
            status, reason, source = evaluate_feasibility_v2(values, action_id)
            rows.append({
                "case_id": values["case_id"],
                "stage": values["stage"],
                "action_id": action_id,
                "feasibility_status": status,
                "reason_code": reason,
                "rule_version": RULE_VERSION,
                "source_feature": source,
            })
    return pd.DataFrame(rows)


def a4_feasibility_audit() -> dict:
    return {
        "old_rule": {
            "action": "progress_monitoring",
            "legacy_id": "A4",
            "status": "UNKNOWN",
            "reason_code": "CONTENT_AVAILABILITY_UNOBSERVED",
            "source_feature": "content_available:UNAVAILABLE",
            "semantics": "Content Review",
        },
        "new_rule": {
            "action": "progress_monitoring",
            "legacy_id": "A4",
            "status": "FEASIBLE",
            "reason_code": "PROGRESS_STATE_OBSERVED",
            "source_feature": "student_state",
            "semantics": "Progress Monitoring",
        },
        "reason": "A4 is Progress Monitoring and does not require content availability. The v1 UNKNOWN/content_available contract is stale.",
        "historical_artifact_mutated": False,
        "version": RULE_VERSION,
    }
