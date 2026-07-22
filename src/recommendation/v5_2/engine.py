from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from .taxonomy import ACTIONS_CAP, ACTION_TAXONOMY, POLICY_VERSION, WEEKLY_MINUTES_CAP


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _probability(values: Sequence[float], classes: int) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.shape != (classes,) or not np.isfinite(probability).all():
        raise ValueError("Invalid prediction probability shape or value")
    if (probability < 0).any() or (probability > 1).any() or not np.isclose(
        probability.sum(), 1.0, atol=1e-5
    ):
        raise ValueError("Invalid prediction probability contract")
    return probability / probability.sum()


def _jensen_shannon(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)
    kl_first = np.sum(first * np.log(np.clip(first / midpoint, 1e-12, None)))
    kl_second = np.sum(second * np.log(np.clip(second / midpoint, 1e-12, None)))
    return float(0.5 * (kl_first + kl_second))


def disagreement_metrics(
    dataset: str,
    deep_probability: Sequence[float],
    ml_probability: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | bool]:
    classes = 2 if dataset == "oulad" else 3
    deep = _probability(deep_probability, classes)
    ml = _probability(ml_probability, classes)
    if classes == 3:
        deep_label = int(deep.argmax())
        ml_label = int(ml.argmax())
        return {
            "jensen_shannon_divergence": _jensen_shannon(deep, ml),
            "label_disagreement": deep_label != ml_label,
            "confidence_gap": float(abs(deep.max() - ml.max())),
        }
    deep_label = bool(deep[1] >= threshold)
    ml_label = bool(ml[1] >= threshold)
    return {
        "probability_difference": float(abs(deep[1] - ml[1])),
        "label_disagreement": deep_label != ml_label,
        "confidence_gap": float(abs(deep.max() - ml.max())),
    }


def _large_disagreement(dataset: str, disagreement: dict[str, float | bool]) -> bool:
    divergence = float(
        disagreement.get(
            "jensen_shannon_divergence", disagreement.get("probability_difference", 0.0)
        )
    )
    cutoff = 0.12 if dataset != "oulad" else 0.25
    return bool(disagreement["label_disagreement"] and divergence >= cutoff)


def _risk(dataset: str, deep: np.ndarray) -> tuple[str, float]:
    risk_probability = float(deep[1] if dataset == "oulad" else deep[0])
    if risk_probability >= 0.65:
        return "high", risk_probability
    if risk_probability >= 0.35:
        return "medium", risk_probability
    return "low", risk_probability


def _confidence(deep: np.ndarray) -> str:
    maximum = float(deep.max())
    return "high" if maximum >= 0.75 else "medium" if maximum >= 0.55 else "low"


def _ranked_candidates(
    dataset: str,
    risk_level: str,
    confidence: str,
    features: dict[str, float | int | bool | None],
    *,
    large_disagreement: bool,
) -> list[tuple[float, str, list[str]]]:
    risk_value = {"low": 0.2, "medium": 0.6, "high": 1.0}[risk_level]
    confidence_value = {"low": 0.2, "medium": 0.6, "high": 1.0}[confidence]
    trend = float(features.get("grade_trend") or 0.0)
    activity = float(features.get("activity_level") or 0.0)
    inactivity = float(features.get("inactivity_streak") or 0.0)
    assessment = float(features.get("assessment_progress") or 0.0)
    candidates: list[tuple[float, str, list[str]]] = []

    def add(action: str, score: float, reasons: list[str]) -> None:
        if dataset in ACTION_TAXONOMY[action]["dataset_applicability"]:
            candidates.append((score, action, reasons))

    if dataset != "oulad":
        add("FOUNDATION_REVIEW", 2.0 * risk_value + (0.5 if trend < 0 else 0), ["LOW_ACHIEVEMENT_RISK"])
        if trend <= 0:
            add("TARGETED_PRACTICE", 1.4 + risk_value, ["NON_POSITIVE_GRADE_TREND"])
    else:
        if activity < 0.35 or inactivity >= 2:
            add("VLE_ENGAGEMENT", 1.5 + risk_value + min(inactivity / 10, 1), ["LOW_VLE_ENGAGEMENT"])
        if assessment < 0.5:
            add("ASSESSMENT_COMPLETION", 1.2 + risk_value + (0.5 - assessment), ["ASSESSMENT_PROGRESS_DEFICIT"])
    if risk_level in {"medium", "high"}:
        add("STUDY_SCHEDULE", 1.0 + risk_value, ["PLAN_STRUCTURE_NEEDED"])
        add("INSTRUCTOR_CONTACT", 0.8 + risk_value, ["PERSISTENT_RISK_SIGNAL"])
    if risk_level == "medium" and confidence != "low":
        add("PEER_STUDY", 0.6 + confidence_value, ["COLLABORATIVE_SUPPORT_OPTION"])
    if large_disagreement or confidence == "low" or risk_level == "high":
        add("ADVISOR_ESCALATION", 3.0 + risk_value, ["ADVISOR_REVIEW_REQUIRED"])
    add("PROGRESS_MONITORING", 0.5 + risk_value, ["FOLLOW_UP_REQUIRED"])
    return sorted(candidates, key=lambda row: (-row[0], row[1]))


def build_recommendation(
    *,
    student_or_enrollment_id: str,
    dataset: str,
    prediction_set_id: str,
    deep_model_registry_id: str,
    ml_model_registry_id: str,
    deep_probability: Sequence[float],
    ml_probability: Sequence[float],
    features: dict[str, float | int | bool | None],
    input_snapshot_at: str,
    prediction_created_at: str,
    created_at: str = "2026-07-19T00:00:00+00:00",
    stale_after_days: int = 30,
) -> dict[str, Any]:
    if dataset not in {"student-mat", "student-por", "oulad"}:
        raise ValueError(f"Unknown recommendation dataset: {dataset}")
    classes = 2 if dataset == "oulad" else 3
    deep = _probability(deep_probability, classes)
    ml = _probability(ml_probability, classes)
    snapshot_time = datetime.fromisoformat(input_snapshot_at.replace("Z", "+00:00"))
    prediction_time = datetime.fromisoformat(prediction_created_at.replace("Z", "+00:00"))
    creation_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    if any(value.tzinfo is None for value in (snapshot_time, prediction_time, creation_time)):
        raise ValueError("Recommendation timestamps must be timezone-aware")
    if snapshot_time > prediction_time:
        raise ValueError("Post-cutoff feature timestamp entered recommendation")
    missing = sorted(name for name, value in features.items() if value is None)
    stale = (creation_time - prediction_time).total_seconds() > stale_after_days * 86400
    disagreement = disagreement_metrics(dataset, deep, ml)
    large = _large_disagreement(dataset, disagreement)
    risk_level, risk_probability = _risk(dataset, deep)
    confidence = _confidence(deep)
    abstained = bool(missing or stale)
    abstention_reason = (
        f"missing_required_features:{','.join(missing)}"
        if missing
        else "stale_prediction"
        if stale
        else None
    )
    ranked_actions = []
    if not abstained:
        for rank, (score, action_id, reasons) in enumerate(
            _ranked_candidates(
                dataset, risk_level, confidence, features, large_disagreement=large
            )[:ACTIONS_CAP],
            1,
        ):
            specification = ACTION_TAXONOMY[action_id]
            ranked_actions.append(
                {
                    "rank": rank,
                    "action_id": action_id,
                    "score": round(float(score), 6),
                    "priority": int(specification["default_priority"]),
                    "weekly_minutes": int(specification["weekly_minutes"]),
                    "target_week": int(specification["target_week"]),
                    "reason_codes": reasons,
                    "requires_advisor": bool(specification["requires_advisor"]),
                    "safety_notes": specification["safety_notes"],
                }
            )
    weekly_minutes = sum(int(action["weekly_minutes"]) for action in ranked_actions)
    if weekly_minutes > WEEKLY_MINUTES_CAP:
        raise RuntimeError("Recommendation workload cap exceeded")
    advisor_review = bool(
        abstained
        or large
        or confidence == "low"
        or risk_level == "high"
        or any(action["requires_advisor"] for action in ranked_actions)
    )
    weekly_plan = [
        {
            "week": week,
            "actions": [
                action["action_id"] for action in ranked_actions if action["target_week"] == week
            ],
            "minutes": sum(
                int(action["weekly_minutes"])
                for action in ranked_actions
                if action["target_week"] == week
            ),
        }
        for week in range(1, 5)
    ]
    lineage = {
        "prediction_set_id": prediction_set_id,
        "deep_model_registry_id": deep_model_registry_id,
        "ml_model_registry_id": ml_model_registry_id,
        "input_snapshot_at": input_snapshot_at,
        "prediction_created_at": prediction_created_at,
    }
    body: dict[str, Any] = {
        "student_or_enrollment_id": student_or_enrollment_id,
        "dataset": dataset,
        **lineage,
        "deep_prediction": deep.tolist(),
        "ml_prediction": ml.tolist(),
        "agreement_score": float(
            1.0
            - disagreement.get(
                "jensen_shannon_divergence",
                disagreement.get("probability_difference", 0.0),
            )
        ),
        "disagreement": disagreement,
        "risk_probability": risk_probability,
        "risk_level": risk_level,
        "confidence_level": confidence,
        "reason_codes": sorted(
            {reason for action in ranked_actions for reason in action["reason_codes"]}
        ),
        "ranked_actions": ranked_actions,
        "weekly_plan": weekly_plan,
        "weekly_minutes": weekly_minutes,
        "requires_advisor_review": advisor_review,
        "abstention_status": "ABSTAIN" if abstained else "GENERATED",
        "abstention_reason": abstention_reason,
        "policy_version": POLICY_VERSION,
        "created_at": created_at,
        "revision_no": 1,
        "revision_history": [{"revision_no": 1, "supersedes": None, "reason": "initial"}],
        "effectiveness": "NOT_ESTABLISHED",
        "causal_claim": "PROHIBITED",
        "post_cutoff_features_used": False,
    }
    body["recommendation_id"] = canonical_hash(body)[:24]
    validate_recommendation(body)
    return body


def validate_recommendation(value: dict[str, Any]) -> None:
    required = {
        "recommendation_id",
        "student_or_enrollment_id",
        "dataset",
        "prediction_set_id",
        "deep_model_registry_id",
        "ml_model_registry_id",
        "deep_prediction",
        "ml_prediction",
        "agreement_score",
        "risk_level",
        "confidence_level",
        "reason_codes",
        "ranked_actions",
        "weekly_plan",
        "weekly_minutes",
        "requires_advisor_review",
        "abstention_status",
        "abstention_reason",
        "policy_version",
        "created_at",
    }
    if required - set(value):
        raise ValueError(f"Recommendation schema missing: {sorted(required - set(value))}")
    if value["policy_version"] != POLICY_VERSION:
        raise ValueError("Recommendation policy version mismatch")
    actions = value["ranked_actions"]
    action_ids = [action["action_id"] for action in actions]
    if len(action_ids) != len(set(action_ids)) or len(actions) > ACTIONS_CAP:
        raise ValueError("Recommendation contains duplicate or excess actions")
    if any(action_id not in ACTION_TAXONOMY for action_id in action_ids):
        raise ValueError("Recommendation contains unknown action")
    if int(value["weekly_minutes"]) > WEEKLY_MINUTES_CAP:
        raise ValueError("Recommendation exceeds workload cap")
    if actions and any(not action["reason_codes"] for action in actions):
        raise ValueError("Recommendation action lacks a reason code")
    if not all(
        value.get(name)
        for name in ("prediction_set_id", "deep_model_registry_id", "ml_model_registry_id")
    ):
        raise ValueError("Recommendation model lineage is incomplete")
    if value["post_cutoff_features_used"]:
        raise ValueError("Recommendation used post-cutoff features")
    if value["abstention_status"] == "ABSTAIN" and actions:
        raise ValueError("Abstained recommendation cannot contain actions")
    if not math.isfinite(float(value["agreement_score"])):
        raise ValueError("Recommendation agreement is non-finite")


__all__ = [
    "build_recommendation",
    "canonical_hash",
    "disagreement_metrics",
    "validate_recommendation",
]
