from __future__ import annotations

from typing import Any


POLICY_VERSION = "v6_risk_to_recommendation_policy_v1"
WITHDRAWAL_MECHANISM_STATUS = "EXPLORATORY_DISABLED_FOR_RECOMMENDATION"
POLICY_THRESHOLDS = {
    "risk": {"medium": 0.35, "high": 0.60, "critical": 0.80},
    "hazard": {"high": 0.45},
    "fail": {"high": 0.55},
    "uncertainty": {"low_confidence": 0.18},
    "deep_ml_disagreement": {"expert_review": 0.25},
    "workload": {"max_actions": 4, "max_weekly_minutes": 180},
}


def risk_level(probability: float, percentile: float) -> str:
    if probability >= POLICY_THRESHOLDS["risk"]["critical"] or percentile >= 0.98:
        return "CRITICAL_RISK"
    if probability >= POLICY_THRESHOLDS["risk"]["high"] or percentile >= 0.90:
        return "HIGH_RISK"
    if probability >= POLICY_THRESHOLDS["risk"]["medium"] or percentile >= 0.70:
        return "MEDIUM_RISK"
    return "LOW_RISK"


def risk_mechanism(profile: dict[str, Any]) -> str:
    if profile["confidence_level"] == "LOW_CONFIDENCE" or profile[
        "decision_status"
    ] == "ABSTAIN_REVIEW_REQUIRED":
        return "UNCERTAIN_RISK"
    # The registered withdrawal head has near-zero recall. Its probability can
    # remain in the risk profile as exploratory evidence, but it is not reliable
    # enough to assert an engagement mechanism or select mechanism-specific care.
    academic = profile["probability_fail"] >= POLICY_THRESHOLDS["fail"]["high"]
    if academic:
        return "ACADEMIC_RISK"
    return "GENERAL_RISK"


def priority(profile: dict[str, Any], level: str) -> str:
    if level == "CRITICAL_RISK" or profile["top_k_bucket"] == "TOP_5_PERCENT":
        return "IMMEDIATE"
    if level == "HIGH_RISK" or profile["top_k_bucket"] == "TOP_10_PERCENT":
        return "HIGH"
    if level == "MEDIUM_RISK":
        return "ROUTINE"
    return "MONITOR"


def escalation_reasons(
    profile: dict[str, Any], level: str, mechanism: str, *, conflict: bool = False
) -> list[str]:
    reasons: list[str] = []
    if level == "CRITICAL_RISK":
        reasons.append("CRITICAL_RISK")
    if profile["confidence_level"] == "LOW_CONFIDENCE":
        reasons.append("LOW_CONFIDENCE")
    if profile["deep_ml_disagreement"] >= POLICY_THRESHOLDS["deep_ml_disagreement"][
        "expert_review"
    ]:
        reasons.append("DEEP_ML_DISAGREEMENT")
    if profile["decision_status"] == "ABSTAIN_REVIEW_REQUIRED":
        reasons.append("PREDICTION_ABSTAINED")
    if mechanism == "UNCERTAIN_RISK":
        reasons.append("INSUFFICIENT_AUTOMATIC_EVIDENCE")
    if conflict:
        reasons.append("RECOMMENDATION_CONFLICT")
    return sorted(set(reasons))


def apply_decision_policy(profile: dict[str, Any]) -> dict[str, Any]:
    level = risk_level(profile["probability_at_risk"], profile["risk_percentile"])
    mechanism = risk_mechanism(profile)
    reasons = escalation_reasons(profile, level, mechanism)
    return {
        "policy_version": POLICY_VERSION,
        "risk_level": level,
        "risk_mechanism": mechanism,
        "priority": priority(profile, level),
        "requires_expert_review": bool(reasons),
        "escalation_reasons": reasons,
    }


__all__ = [
    "POLICY_THRESHOLDS",
    "POLICY_VERSION",
    "WITHDRAWAL_MECHANISM_STATUS",
    "apply_decision_policy",
    "escalation_reasons",
    "priority",
    "risk_level",
    "risk_mechanism",
]
