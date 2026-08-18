"""Build and validate pseudonymous, unlabeled panel payloads only."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .constants import ACTION_IDS, FEASIBILITY_STATUSES, PROMPT_VERSION, RUBRIC

ALLOWED_FIELDS = (
    "stage", "risk_probability", "risk_band", "inactive_streak", "active_days_ratio",
    "recent_activity", "activity_trend", "assessment_completion", "missing_assessments",
    "course_progress", "quiz_activity", "vle_available",
)
FORBIDDEN_FIELDS = {"student_id", "module", "presentation", "enrollment_identity", "record_id", "target", "final_result", "score", "outer_fold", "future_features", "raw_features"}
def build_label_payload(row: Mapping, *, panel: str, feasibility_statuses: Mapping[str, str]) -> dict:
    if panel not in {"Panel A", "Panel B"}: raise ValueError("panel must be Panel A or Panel B")
    if "case_id" not in row: raise ValueError("case_id is required")
    missing_feasibility = set(ACTION_IDS).difference(feasibility_statuses)
    if missing_feasibility: raise ValueError(f"missing feasibility statuses: {sorted(missing_feasibility)}")
    if any(feasibility_statuses[action_id] not in FEASIBILITY_STATUSES for action_id in ACTION_IDS):
        raise ValueError("invalid feasibility status")
    payload = {"case_id": str(row["case_id"]), "prompt_version": PROMPT_VERSION}
    def _json_value(value):
        return value.item() if hasattr(value, "item") else value
    payload.update({field: _json_value(row[field]) for field in ALLOWED_FIELDS})
    payload["actions"] = list(ACTION_IDS)
    payload["feasibility"] = {action_id: str(feasibility_statuses[action_id]) for action_id in ACTION_IDS}
    payload["rubric"] = dict(RUBRIC)
    return payload


def validate_label_payload(payload: Mapping) -> list[str]:
    errors: list[str] = []
    required = {"case_id", "prompt_version", "actions", "rubric", "feasibility", *ALLOWED_FIELDS}
    errors.extend(f"missing:{field}" for field in sorted(required.difference(payload)))
    errors.extend(f"unexpected:{field}" for field in sorted(set(payload).difference(required)))
    forbidden = FORBIDDEN_FIELDS.intersection(payload)
    if forbidden: errors.append(f"forbidden:{','.join(sorted(forbidden))}")
    if tuple(payload.get("actions", ())) != ACTION_IDS: errors.append("invalid_action_taxonomy")
    if payload.get("prompt_version") != PROMPT_VERSION: errors.append("invalid_prompt_version")
    feasibility = payload.get("feasibility", {})
    if set(feasibility) != set(ACTION_IDS): errors.append("invalid_feasibility_actions")
    if any(value not in FEASIBILITY_STATUSES for value in feasibility.values()): errors.append("invalid_feasibility_status")
    return errors


def build_label_job(rows: Iterable[Mapping], *, panel: str, feasibility_by_case: Mapping[str, Mapping[str, str]]) -> list[dict]:
    payloads = [build_label_payload(row, panel=panel, feasibility_statuses=feasibility_by_case[str(row["case_id"])]) for row in rows]
    errors = [validate_label_payload(payload) for payload in payloads]
    if any(errors): raise ValueError(f"invalid label payload: {next(error for error in errors if error)}")
    return payloads
