from __future__ import annotations

from typing import Any

from src.recommendation.v5_2.taxonomy import ACTION_TAXONOMY

from .contract import ARTIFACT_ROOT, SCHEMA_VERSION, atomic_json, canonical_sha256


OBSERVED_FEATURES: dict[str, dict[str, Any]] = {
    "activity_level": {
        "semantic_type": "OBSERVED_PRE_CUTOFF_BEHAVIOR",
        "raw_sources": ["studentVle.sum_click", "studentVle.date"],
        "cutoff_rule": "date <= registered cutoff_day",
        "transformation": "log1p(mean total clicks over last two valid weeks) / log1p(100), clipped [0,1]",
        "valid_range": [0.0, 1.0],
        "rules": [
            {
                "predicate": "activity_level < 0.35",
                "reason_code": "LOW_VLE_ENGAGEMENT",
                "action_id": "VLE_ENGAGEMENT",
            }
        ],
    },
    "inactivity_streak": {
        "semantic_type": "OBSERVED_PRE_CUTOFF_BEHAVIOR",
        "raw_sources": ["studentVle.sum_click", "studentVle.date"],
        "cutoff_rule": "date <= registered cutoff_day",
        "transformation": "consecutive valid weeks ending at cutoff with total_clicks == 0",
        "valid_range": [0.0, None],
        "rules": [
            {
                "predicate": "inactivity_streak >= 2",
                "reason_code": "LOW_VLE_ENGAGEMENT",
                "action_id": "VLE_ENGAGEMENT",
            }
        ],
    },
    "assessment_progress": {
        "semantic_type": "OBSERVED_PRE_CUTOFF_BEHAVIOR",
        "raw_sources": [
            "assessments.id_assessment",
            "assessments.date",
            "studentAssessment.id_assessment",
            "studentAssessment.date_submitted",
        ],
        "cutoff_rule": "assessment date and submission date <= registered cutoff_day",
        "transformation": "unique due assessments submitted by cutoff / unique assessments due by cutoff",
        "valid_range": [0.0, 1.0],
        "not_applicable_rule": "missing when no assessment is due by cutoff",
        "rules": [
            {
                "predicate": "assessment_progress < 0.50 and due_assessment_count > 0",
                "reason_code": "ASSESSMENT_PROGRESS_DEFICIT",
                "action_id": "ASSESSMENT_COMPLETION",
            }
        ],
    },
    "grade_trend": {
        "semantic_type": "OBSERVED_PRE_CUTOFF_BEHAVIOR",
        "raw_sources": [
            "studentAssessment.score",
            "studentAssessment.date_submitted",
        ],
        "cutoff_rule": "date_submitted <= registered cutoff_day",
        "transformation": "change in cumulative mean score between last two valid weeks, divided by 100",
        "valid_range": [-1.0, 1.0],
        "not_applicable_rule": "missing unless a score is available in both comparison weeks",
        "rules": [],
    },
}


PREDICTION_SUPPORT: dict[str, dict[str, Any]] = {
    "risk_band": {
        "semantic_type": "PREDICTION_DECISION_SUPPORT",
        "source": "frozen V6 risk profile probability_at_risk and percentile",
        "allowed_use": ["risk tier", "generic support priority", "human escalation"],
        "prohibited_use": ["assert observed behavior", "assert intervention effectiveness"],
    },
    "fail_risk_band": {
        "semantic_type": "PREDICTION_DECISION_SUPPORT",
        "source": "frozen V6 outcome head probability_fail",
        "allowed_use": ["academic/general mechanism band"],
        "prohibited_use": ["assert assessment non-completion", "assert observed grade decline"],
    },
    "confidence_and_disagreement": {
        "semantic_type": "PREDICTION_DECISION_SUPPORT",
        "source": "frozen V6 confidence/abstention/disagreement fields",
        "allowed_use": ["abstention", "human review"],
        "prohibited_use": ["observed reason code"],
    },
    "withdrawal_horizon": {
        "semantic_type": "EXPLORATORY_PREDICTION_ONLY",
        "source": "frozen V6 withdrawal head",
        "reliability": "UNRELIABLE_NEAR_ZERO_RECALL",
        "allowed_use": ["offline audit display outside expert package"],
        "prohibited_use": [
            "risk mechanism",
            "LOW_VLE_ENGAGEMENT",
            "priority",
            "urgency",
            "action selection",
        ],
        "outgoing_rules": [],
    },
}


def build_feature_lineage() -> dict[str, Any]:
    action_records = {
        action_id: {
            "action_id": action_id,
            "dataset_applicability": spec["dataset_applicability"],
            "registered_reason_code": spec["reason_code"],
            "evidence_source": spec["evidence_source"],
            "weekly_minutes": spec["weekly_minutes"],
            "requires_advisor": spec["requires_advisor"],
        }
        for action_id, spec in ACTION_TAXONOMY.items()
        if "oulad" in spec["dataset_applicability"]
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "lineage_schema": "recommendation_feature_lineage_v1",
        "target_scope": "OULAD F2_MIDDLE historical pre-cutoff recommendation validation",
        "observed_features": OBSERVED_FEATURES,
        "prediction_support": PREDICTION_SUPPORT,
        "actions": action_records,
        "invariants": {
            "observed_behavior_from_prediction_probability": False,
            "post_cutoff_features": False,
            "future_oulad_accessed": False,
            "withdrawal_can_trigger_action": False,
            "sensitive_attributes_in_reasoning": False,
        },
    }
    value["lineage_sha256"] = canonical_sha256(value)
    atomic_json(ARTIFACT_ROOT / "recommendation_feature_lineage.json", value)
    return value


__all__ = [
    "OBSERVED_FEATURES",
    "PREDICTION_SUPPORT",
    "build_feature_lineage",
]
