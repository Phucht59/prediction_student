from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.studies.oulad_v3.data import BASE_CHANNELS


def _identifier(prefix: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()[:16]
    return f"{prefix}-{digest}"


def validate_scores(seed_scores: list[float]) -> np.ndarray:
    values = np.asarray(seed_scores, dtype=float)
    if (
        values.shape != (5,)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Exactly five finite model scores in [0,1] are required")
    return values


def evidence_codes(sequence: np.ndarray, valid_length: int, policy: dict) -> list[str]:
    index = {name: position for position, name in enumerate(BASE_CHANNELS)}
    current = sequence[valid_length - 1]
    previous = sequence[max(0, valid_length - 2)]
    codes = []
    if current[index["weeks_without_activity"]] >= policy["inactivity_streak_weeks"]:
        codes.append("sustained_inactivity")
    previous_clicks = previous[index["total_clicks"]]
    current_clicks = current[index["total_clicks"]]
    if (
        previous_clicks >= policy["activity_change_minimum_clicks"]
        and current_clicks < previous_clicks * policy["activity_drop_ratio"]
    ):
        codes.append("recent_activity_drop")
    if current[index["submitted_assessment_count"]] <= 0:
        codes.append("recent_no_recorded_submission")
    submissions = sequence[:valid_length, index["submitted_assessment_count"]].sum()
    late = sequence[:valid_length, index["late_submission_count"]].sum()
    if (
        submissions >= policy["late_pattern_minimum_submissions"]
        and late / max(submissions, 1) >= policy["late_pattern_ratio"]
    ):
        codes.append("late_submission_pattern")
    missing = current[index["score_missing_mask"]] >= 0.5
    if missing:
        codes.append("score_unavailable")
    else:
        change = (
            current[index["cumulative_mean_score"]]
            - previous[index["cumulative_mean_score"]]
        )
        if change <= policy["score_decline_points"]:
            codes.append("score_decline")
    return codes or ["model_only_signal"]


ACTION_BY_REASON = {
    "sustained_inactivity": "Advisor reviews the recent inactivity period and agrees a feasible re-engagement schedule with the learner.",
    "recent_activity_drop": "Advisor checks the recent activity decline and identifies one concrete barrier before agreeing the next study action.",
    "recent_no_recorded_submission": "Advisor verifies whether any currently available assessment needs planning; this does not assert that a deadline was missed.",
    "late_submission_pattern": "Advisor reviews submission planning and agrees one reminder or workload adjustment with the learner.",
    "score_decline": "Advisor reviews available formative feedback and selects one weak topic for targeted practice.",
    "score_unavailable": "Advisor first verifies whether sufficient assessment evidence is available before proposing an intervention.",
    "model_only_signal": "Advisor performs a diagnostic review because the model signal is not supported by a specific observed reason code.",
}


def build_recommendation(
    *,
    record_id: str,
    seed_scores: list[float],
    threshold: float,
    sequence: np.ndarray,
    valid_length: int,
    policy_parameters: dict[str, Any],
    policy_version: str = "oulad-v4-policy-1",
) -> dict[str, Any]:
    scores = validate_scores(seed_scores)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold outside [0,1]")
    mean = float(scores.mean())
    predictions = scores >= threshold
    disagreement = float(min(predictions.mean(), 1 - predictions.mean()) * 2)
    score_sd = float(scores.std(ddof=0))
    distance = abs(mean - threshold)
    uncertain = (
        disagreement > policy_parameters["max_seed_disagreement"]
        or score_sd > policy_parameters["max_seed_score_sd"]
        or distance < policy_parameters["minimum_threshold_distance"]
    )
    reasons = evidence_codes(sequence, valid_length, policy_parameters)
    trajectory = (
        "declining"
        if "recent_activity_drop" in reasons or "score_decline" in reasons
        else ("inactive" if "sustained_inactivity" in reasons else "stable_or_unknown")
    )
    actions = (
        []
        if uncertain
        else [
            {
                "action_id": f"action-{index + 1}",
                "reason_code": reason,
                "description": ACTION_BY_REASON[reason],
                "advisor_approval_required": True,
                "status": "draft",
            }
            for index, reason in enumerate(reasons)
        ]
    )
    payload = {
        "record_id": record_id,
        "policy_version": policy_version,
        "uncalibrated_model_score": mean,
        "seed_scores": [float(value) for value in scores],
        "seed_disagreement": disagreement,
        "seed_score_sd": score_sd,
        "threshold_distance": distance,
        "predicted_at_risk": bool(mean >= threshold),
        "reason_codes": reasons,
        "trajectory": trajectory,
        "true_abstention": uncertain,
        "advisor_review_required": True,
        "actions": actions,
        "score_semantics": "uncalibrated_model_score",
        "causal_claim": "PROHIBITED",
        "effectiveness": "NOT_ESTABLISHED",
        "deployment_status": "RESEARCH_PROTOTYPE_NOT_APPROVED_FOR_AUTOMATED_STUDENT_CONTACT",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["recommendation_id"] = _identifier(
        "oulad-rec",
        {
            key: payload[key]
            for key in ["record_id", "policy_version", "seed_scores", "reason_codes"]
        },
    )
    return payload


__all__ = [
    "ACTION_BY_REASON",
    "build_recommendation",
    "evidence_codes",
    "validate_scores",
]
