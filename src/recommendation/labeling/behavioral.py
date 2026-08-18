"""Deterministic, config-driven behavioral labeling functions."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

FINAL_ACTIONS = (
    "assessment_recovery", "re_engagement", "study_planning", "progress_monitoring", "retrieval_practice",
)
ACTION_KEYS = {"A1": "assessment_recovery", "A2": "re_engagement", "A3": "study_planning", "A4": "progress_monitoring", "A5": "retrieval_practice"}
BEHAVIOR_LF_NAMES = {
    "assessment_recovery": "LF_BEHAVIOR_A1",
    "re_engagement": "LF_BEHAVIOR_A2",
    "study_planning": "LF_BEHAVIOR_A3",
    "progress_monitoring": "LF_BEHAVIOR_A4",
    "retrieval_practice": "LF_BEHAVIOR_A5",
}


def _valid_number(value) -> bool:
    try:
        return value is not None and not pd.isna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _q(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"q25": None, "q50": None, "q75": None}
    return {key: float(numeric.quantile(level)) for key, level in (("q25", .25), ("q50", .50), ("q75", .75))}


def _bucket(value, cuts: Mapping) -> int | None:
    if not _valid_number(value) or any(cuts.get(key) is None for key in ("q25", "q50", "q75")):
        return None
    return int(sum(float(value) > float(cuts[key]) for key in ("q25", "q50", "q75")))


def _mean(values: list[int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def _score_label(score: float | None, cuts: Mapping) -> tuple[str, float | None]:
    if score is None:
        return "ABSTAIN", None
    if any(cuts.get(key) is None for key in ("q25", "q50", "q75")):
        return "ABSTAIN", score
    label = int(sum(score > float(cuts[key]) for key in ("q25", "q50", "q75")))
    return str(min(3, max(0, label))), float(score)


def derive_thresholds(panel: pd.DataFrame) -> dict:
    required = {
        "inactive_streak", "active_days_ratio", "recent_activity", "activity_trend",
        "assessment_completion", "missing_assessments", "course_progress", "quiz_activity",
    }
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Panel A missing behavioral fields: {sorted(missing)}")
    quantiles = {field: _q(panel[field]) for field in sorted(required)}
    assessment_scores = []
    engagement_scores = []
    planning_scores = []
    positive_quiz_scores = []
    for _, row in panel.iterrows():
        assessment_scores.append(_mean([
            _bucket(row["missing_assessments"], quantiles["missing_assessments"]),
            None if not _valid_number(row["assessment_completion"]) else 3 - (_bucket(row["assessment_completion"], quantiles["assessment_completion"]) or 0),
        ]))
        engagement_scores.append(_mean([
            _bucket(row["inactive_streak"], quantiles["inactive_streak"]),
            None if _bucket(row["active_days_ratio"], quantiles["active_days_ratio"]) is None else 3 - _bucket(row["active_days_ratio"], quantiles["active_days_ratio"]),
            None if _bucket(row["recent_activity"], quantiles["recent_activity"]) is None else 3 - _bucket(row["recent_activity"], quantiles["recent_activity"]),
            None if _bucket(row["activity_trend"], quantiles["activity_trend"]) is None else 3 - _bucket(row["activity_trend"], quantiles["activity_trend"]),
        ]))
        planning_scores.append(None if _bucket(row["activity_trend"], quantiles["activity_trend"]) is None else 3 - _bucket(row["activity_trend"], quantiles["activity_trend"]))
        if _valid_number(row["quiz_activity"]) and float(row["quiz_activity"]) > 0:
            positive_quiz_scores.append(float(row["quiz_activity"]))
    score_cutpoints = {
        "assessment_recovery": _q(pd.Series(assessment_scores)),
        "re_engagement": _q(pd.Series(engagement_scores)),
        "study_planning": _q(pd.Series(planning_scores)),
        "retrieval_practice_positive_quiz": _q(pd.Series(positive_quiz_scores)),
    }
    return {
        "quantiles": quantiles,
        "operational_score_cutpoints": score_cutpoints,
        "rule_parameters": {
            "planning_min_active_level": 1,
            "planning_min_recent_level": 1,
            "planning_max_inactive_level": 1,
        },
        "source_audit": {
            "course_progress": "stage indicator only; progress-gap formula disabled",
            "study_regularness": "UNAVAILABLE; no proxy created",
            "quiz_available": "UNAVAILABLE; quiz_activity=0 is not treated as unavailable by itself",
        },
    }


def _abstain(reason: str) -> dict:
    return {"label": "ABSTAIN", "abstain": True, "evidence_score": None, "reason_code": reason}


def behavioral_label(row: Mapping, action_id: str, feasibility_status: str, thresholds: Mapping) -> dict:
    if action_id not in FINAL_ACTIONS:
        raise ValueError(f"unknown final action: {action_id}")
    if feasibility_status == "INFEASIBLE":
        return _abstain("INFEASIBLE")
    q = thresholds["quantiles"]
    cuts = thresholds["operational_score_cutpoints"]
    params = thresholds["rule_parameters"]
    if action_id == "assessment_recovery":
        if feasibility_status != "FEASIBLE":
            return _abstain("FEASIBILITY_UNKNOWN")
        missing = _bucket(row.get("missing_assessments"), q["missing_assessments"])
        completion = _bucket(row.get("assessment_completion"), q["assessment_completion"])
        if missing is None or completion is None or not (0 <= float(row.get("assessment_completion")) <= 1):
            return _abstain("NO_ASSESSMENT_EVIDENCE")
        label, evidence_score = _score_label(_mean([missing, 3 - completion]), cuts["assessment_recovery"])
        return {"label": label, "abstain": False, "evidence_score": evidence_score, "reason_code": "OBSERVED_ASSESSMENT_GAP"}
    if action_id == "re_engagement":
        if feasibility_status != "FEASIBLE":
            return _abstain("FEASIBILITY_UNKNOWN")
        if row.get("vle_available") is not True:
            return _abstain("VLE_UNAVAILABLE")
        levels = [
            _bucket(row.get("inactive_streak"), q["inactive_streak"]),
            None if _bucket(row.get("active_days_ratio"), q["active_days_ratio"]) is None else 3 - _bucket(row.get("active_days_ratio"), q["active_days_ratio"]),
            None if _bucket(row.get("recent_activity"), q["recent_activity"]) is None else 3 - _bucket(row.get("recent_activity"), q["recent_activity"]),
            None if _bucket(row.get("activity_trend"), q["activity_trend"]) is None else 3 - _bucket(row.get("activity_trend"), q["activity_trend"]),
        ]
        score = _mean(levels)
        if score is None or sum(level is not None for level in levels) < 3:
            return _abstain("NO_ENGAGEMENT_EVIDENCE")
        label, evidence_score = _score_label(score, cuts["re_engagement"])
        return {"label": label, "abstain": False, "evidence_score": evidence_score, "reason_code": "OBSERVED_ENGAGEMENT_PATTERN"}
    if action_id == "study_planning":
        if feasibility_status != "FEASIBLE":
            return _abstain("FEASIBILITY_UNKNOWN")
        active = _bucket(row.get("active_days_ratio"), q["active_days_ratio"])
        recent = _bucket(row.get("recent_activity"), q["recent_activity"])
        inactive = _bucket(row.get("inactive_streak"), q["inactive_streak"])
        trend = _bucket(row.get("activity_trend"), q["activity_trend"])
        if None in (active, recent, inactive, trend):
            return _abstain("NO_PLANNING_EVIDENCE")
        if active < params["planning_min_active_level"] or recent < params["planning_min_recent_level"] or inactive > params["planning_max_inactive_level"]:
            return _abstain("NO_DISTINCTION_FROM_REENGAGEMENT")
        label, evidence_score = _score_label(3 - trend, cuts["study_planning"])
        return {"label": label, "abstain": False, "evidence_score": evidence_score, "reason_code": "PARTICIPATING_PATTERN_ORGANIZATION"}
    if action_id == "progress_monitoring":
        return _abstain("UNSUPPORTED_BEHAVIOR_SIGNAL")
    quiz = row.get("quiz_activity")
    if not _valid_number(quiz) or float(quiz) <= 0:
        return _abstain("QUIZ_AVAILABILITY_UNKNOWN")
    if feasibility_status not in ("FEASIBLE", "UNKNOWN"):
        return _abstain("INFEASIBLE")
    level = _bucket(quiz, q["quiz_activity"])
    if level is None:
        return _abstain("NO_QUIZ_EVIDENCE")
    label, evidence_score = _score_label(float(quiz), cuts["retrieval_practice_positive_quiz"])
    reason = "OBSERVED_POSITIVE_QUIZ_ACTIVITY" if feasibility_status == "FEASIBLE" else "OBSERVED_POSITIVE_QUIZ_ACTIVITY_FEASIBILITY_UNKNOWN"
    return {"label": label, "abstain": False, "evidence_score": evidence_score, "reason_code": reason}
