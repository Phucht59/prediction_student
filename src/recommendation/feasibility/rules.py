"""Minimal evidence-backed feasibility rules for Phase 3."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


ACTION_IDS = ("A1", "A2", "A3", "A4", "A5")
RULE_VERSION = "recommendation.feasibility.v1"


def evaluate_feasibility(row: Mapping, action_id: str) -> tuple[str, str, str]:
    """Return status, deterministic reason and source feature."""
    if action_id not in ACTION_IDS:
        raise ValueError(f"unknown action_id: {action_id}")
    if action_id == "A1":
        if float(row["missing_assessments"]) == 0:
            return "INFEASIBLE", "NO_MISSING_ASSESSMENTS_NOT_APPLICABLE", "missing_assessments"
        return "FEASIBLE", "MISSING_ASSESSMENT_OBJECT_PRESENT", "missing_assessments"
    if action_id == "A2":
        if bool(row["vle_available"]):
            return "FEASIBLE", "OBSERVED_VLE_WINDOW_AVAILABLE", "vle_available"
        return "INFEASIBLE", "OBSERVED_VLE_WINDOW_UNAVAILABLE", "vle_available"
    if action_id == "A3":
        return "FEASIBLE", "STUDY_PLANNING_HAS_NO_REQUIRED_RESOURCE_GATE", "system_contract"
    if action_id == "A4":
        return "UNKNOWN", "CONTENT_AVAILABILITY_UNOBSERVED", "content_available:UNAVAILABLE"
    if float(row["quiz_activity"]) > 0:
        return "FEASIBLE", "QUIZ_ACTIVITY_OBSERVED", "quiz_activity"
    return "UNKNOWN", "QUIZ_AVAILABILITY_UNOBSERVED_NO_INFERENCE_FROM_ZERO", "quiz_available:UNAVAILABLE"


def build_feasibility_frame(state: pd.DataFrame) -> pd.DataFrame:
    required = {"case_id", "stage", "missing_assessments", "vle_available", "quiz_activity"}
    missing = required.difference(state.columns)
    if missing:
        raise ValueError(f"state missing feasibility features: {sorted(missing)}")
    rows: list[dict] = []
    for row in state.itertuples(index=False):
        values = row._asdict()
        for action_id in ACTION_IDS:
            status, reason, source = evaluate_feasibility(values, action_id)
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
