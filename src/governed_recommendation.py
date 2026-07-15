"""Phase D governed, non-causal, advisor-first recommendation policy.

The only active builder in this module consumes a frozen prediction snapshot.
It never accepts targets/outcomes, never imputes missing academic inputs, and
never promotes a recommendation beyond a draft requiring advisor review.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any

import numpy as np


POLICY_ID = "strategy_b_governed_learning_path"
POLICY_VERSION = "strategy_b_governed_learning_path_v1"
SCHEMA_VERSION = "phase_d_recommendation_v1"
WORKLOAD_CAP_MINUTES = 180
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}
REVIEW_STATUSES = {"eligible_for_draft", "advisor_review_required", "insufficient_information", "invalid_prediction", "stale_prediction"}
FORBIDDEN_RECOMMENDATION_KEYS = {"G3", "G3_raw", "true_label", "outcome", "target", "record_id", "fold_id", "dataset_version_id"}


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def feature_registry() -> list[dict[str, Any]]:
    core = [
        ("G1", "student-mat", "first-period grade", "number", "0..20", "grade", False, "before_prediction", True, "prediction_and_explanation", True),
        ("G2", "student-mat", "second-period grade", "number", "0..20", "grade", False, "before_prediction", True, "prediction_and_explanation", True),
        ("G2_minus_G1", "derived", "deterministic grade trajectory", "number", "-20..20", "grade_delta", True, "at_prediction", False, "explanation_and_trajectory_planning", True),
    ]
    contexts = ["studytime", "failures", "schoolsup", "famsup", "activities", "internet", "absences"]
    rows = []
    for name, source, definition, dtype, values, unit, modifiable, capture, allowed_prediction, use, expert in core:
        rows.append({"feature_name": name, "source": source, "semantic_definition": definition, "dtype": dtype, "allowed_values": values, "unit": unit, "capture_time": capture, "available_before_prediction": True, "freshness_limit": "30_days", "modifiable": modifiable, "sensitive": False, "allowed_for_prediction": allowed_prediction, "allowed_for_recommendation": True, "recommendation_use": use, "expert_approved": expert, "reason": "phase_d_core_feature"})
    for name in contexts:
        rows.append({"feature_name": name, "source": "student-mat", "semantic_definition": "unverified_context_field", "dtype": "unknown", "allowed_values": "unverified", "unit": None, "capture_time": "unverified", "available_before_prediction": False, "freshness_limit": None, "modifiable": None, "sensitive": name in {"famsup", "internet"}, "allowed_for_prediction": False, "allowed_for_recommendation": False, "recommendation_use": "none", "expert_approved": False, "reason": "timing_or_semantic_contract_unverified"})
    return rows


def action_catalog() -> list[dict[str, Any]]:
    return [
        {"action_id": "diagnose_topics", "category": "diagnostic", "description_template": "Complete an advisor-reviewed topic diagnostic and identify two priority topics.", "eligible_evidence_codes": ["trajectory_downward", "low_predicted_achievement"], "contraindications": [], "prerequisites": ["advisor_review"], "minimum_frequency": 1, "maximum_frequency": 1, "maximum_weekly_minutes": 45, "required_resource": "advisor_or_topic_diagnostic", "expert_approval_status": "expert_review_pending"},
        {"action_id": "targeted_practice", "category": "practice", "description_template": "Complete targeted practice for the advisor-confirmed priority topics and record completion.", "eligible_evidence_codes": ["trajectory_downward", "trajectory_stable", "medium_predicted_achievement"], "contraindications": [], "prerequisites": ["advisor_review"], "minimum_frequency": 2, "maximum_frequency": 3, "maximum_weekly_minutes": 90, "required_resource": "practice_materials", "expert_approval_status": "expert_review_pending"},
        {"action_id": "retrieval_quiz", "category": "retrieval", "description_template": "Complete one short retrieval quiz and record common errors for advisor discussion.", "eligible_evidence_codes": ["trajectory_downward", "trajectory_stable"], "contraindications": [], "prerequisites": ["targeted_practice"], "minimum_frequency": 1, "maximum_frequency": 1, "maximum_weekly_minutes": 30, "required_resource": "quiz_material", "expert_approval_status": "expert_review_pending"},
        {"action_id": "schedule_review", "category": "planning", "description_template": "Create a realistic weekly study schedule with the advisor and revise it when commitments conflict.", "eligible_evidence_codes": ["trajectory_downward", "trajectory_stable", "trajectory_upward"], "contraindications": [], "prerequisites": ["advisor_review"], "minimum_frequency": 1, "maximum_frequency": 1, "maximum_weekly_minutes": 20, "required_resource": "schedule_template", "expert_approval_status": "expert_review_pending"},
        {"action_id": "consolidation", "category": "consolidation", "description_template": "Consolidate completed topics with an advanced or mixed practice set; avoid unnecessary overload.", "eligible_evidence_codes": ["trajectory_upward", "high_predicted_achievement"], "contraindications": [], "prerequisites": ["advisor_review"], "minimum_frequency": 1, "maximum_frequency": 2, "maximum_weekly_minutes": 60, "required_resource": "advanced_practice_material", "expert_approval_status": "expert_review_pending"},
        {"action_id": "progress_review", "category": "review", "description_template": "Review completion, difficulty and current measurement with the advisor; continue, revise or close the plan.", "eligible_evidence_codes": ["trajectory_downward", "trajectory_stable", "trajectory_upward"], "contraindications": [], "prerequisites": ["advisor_review"], "minimum_frequency": 1, "maximum_frequency": 1, "maximum_weekly_minutes": 20, "required_resource": "advisor", "expert_approval_status": "expert_review_pending"},
    ]


def validate_scores(seed_scores: list[list[float]]) -> np.ndarray:
    values = np.asarray(seed_scores, dtype=float)
    if values.shape != (5, 3) or not np.isfinite(values).all():
        raise ValueError("N0 must provide five finite three-class seed score vectors.")
    if values.min() < 0 or values.max() > 1 or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("N0 seed scores violate probability range/sum contract.")
    return values


def prediction_snapshot(*, student_source_reference: str, features: dict[str, Any], seed_scores: list[list[float]], r0_reference_class: int, model_bundle: dict[str, str], policy_version: str, input_snapshot_timestamp: str, prediction_timestamp: str | None = None) -> dict[str, Any]:
    if set(features) - {"G1", "G2"}:
        raise ValueError("Prediction snapshot feature input may contain only G1/G2.")
    if any(key in features for key in FORBIDDEN_RECOMMENDATION_KEYS):
        raise ValueError("Target/outcome/lineage metadata cannot enter recommendation features.")
    if not all(isinstance(features.get(key), (int, float)) and np.isfinite(features[key]) for key in ["G1", "G2"]):
        raise ValueError("Missing or invalid G1/G2 produces insufficient_information; it is never defaulted.")
    scores = validate_scores(seed_scores)
    ensemble = scores.mean(axis=0)
    if not np.allclose(ensemble, np.mean(scores, axis=0), atol=1e-12):
        raise ValueError("Ensemble must be the arithmetic mean of five seed scores.")
    predicted = int(ensemble.argmax())
    disagreement = float(np.mean(np.argmax(scores, axis=1) != predicted))
    entropy = float(-np.sum(ensemble * np.log(np.clip(ensemble, 1e-12, 1.0))))
    body = {"model_bundle_id": model_bundle["model_bundle_id"], "model_candidate_id": "N0", "model_version": model_bundle["model_version"], "policy_version": policy_version, "student_source_reference": student_source_reference, "input_snapshot_timestamp": input_snapshot_timestamp, "features": {"G1": float(features["G1"]), "G2": float(features["G2"]), "G2_minus_G1": float(features["G2"] - features["G1"])}, "predicted_class": predicted, "class_scores": ensemble.tolist(), "ensemble_seed_predictions": scores.tolist(), "ensemble_seed_disagreement": disagreement, "predictive_entropy": entropy, "max_model_score": float(ensemble.max()), "r0_reference_class": int(r0_reference_class), "n0_r0_agreement": bool(predicted == int(r0_reference_class)), "feature_contract_hash": model_bundle["feature_contract_hash"], "preprocessor_hash": model_bundle["preprocessor_hash"], "checkpoint_bundle_hash": model_bundle["checkpoint_bundle_hash"], "r0": {"probability_available": False, "uncertainty_available": False, "deterministic_rule": True}}
    return {"prediction_snapshot_id": canonical_hash(body)[:24], "prediction_timestamp": prediction_timestamp or utc_now(), **body}


def assess_snapshot(snapshot: dict[str, Any], uncertainty_policy: dict[str, float], *, now: datetime | None = None) -> dict[str, Any]:
    try:
        validate_scores(snapshot["ensemble_seed_predictions"])
        scores = np.asarray(snapshot["class_scores"], dtype=float)
        if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1 or not np.allclose(scores.sum(), 1.0, atol=1e-6):
            raise ValueError("invalid class scores")
    except (KeyError, ValueError, TypeError):
        return {"recommendation_review_status": "invalid_prediction", "reasons": ["invalid_probability_contract"]}
    features = snapshot.get("features", {})
    if not all(key in features and np.isfinite(features[key]) for key in ["G1", "G2", "G2_minus_G1"]):
        return {"recommendation_review_status": "insufficient_information", "reasons": ["missing_or_invalid_core_feature"]}
    timestamp = datetime.fromisoformat(str(snapshot["input_snapshot_timestamp"]).replace("Z", "+00:00"))
    current = now or datetime.now(timezone.utc)
    if (current - timestamp).total_seconds() > uncertainty_policy["freshness_seconds"]:
        return {"recommendation_review_status": "stale_prediction", "reasons": ["input_snapshot_stale"]}
    reasons = []
    if not snapshot["n0_r0_agreement"]: reasons.append("n0_r0_disagreement")
    if snapshot["ensemble_seed_disagreement"] > uncertainty_policy["max_seed_disagreement"]: reasons.append("seed_disagreement_high")
    if snapshot["max_model_score"] < uncertainty_policy["minimum_max_model_score"]: reasons.append("model_score_low")
    if snapshot["predictive_entropy"] > uncertainty_policy["maximum_entropy"]: reasons.append("entropy_high")
    return {"recommendation_review_status": "advisor_review_required" if reasons else "eligible_for_draft", "reasons": reasons}


def _evidence(snapshot: dict[str, Any]) -> list[str]:
    delta = snapshot["features"]["G2_minus_G1"]
    trajectory = "trajectory_downward" if delta < 0 else "trajectory_upward" if delta > 0 else "trajectory_stable"
    return [trajectory, f"{CLASS_NAMES[snapshot['predicted_class']].lower()}_predicted_achievement"]


def _action_ids(evidence: list[str], band: str) -> list[str]:
    if band == "Low": return ["diagnose_topics", "schedule_review", "targeted_practice", "retrieval_quiz", "progress_review"]
    if band == "Medium": return ["schedule_review", "targeted_practice", "retrieval_quiz", "progress_review"]
    return ["schedule_review", "consolidation", "progress_review"]


def _validate_actions(actions: list[dict[str, Any]]) -> list[str]:
    catalog = {row["action_id"]: row for row in action_catalog()}
    errors = []
    ids = [action["action_id"] for action in actions]
    if len(ids) != len(set(ids)): errors.append("duplicate_action_id")
    workload = sum(int(action["weekly_workload_minutes"]) for action in actions)
    if workload > WORKLOAD_CAP_MINUTES: errors.append("weekly_workload_cap_exceeded")
    for action in actions:
        definition = catalog.get(action["action_id"])
        if definition is None: errors.append("unknown_action")
        elif action["weekly_workload_minutes"] > definition["maximum_weekly_minutes"]: errors.append("action_weekly_limit_exceeded")
        if any(item not in ids and item != "advisor_review" for item in action["prerequisites"]): errors.append("missing_prerequisite")
        if not action.get("required_resource"): errors.append("unavailable_resource")
    return sorted(set(errors))


def build_governed_recommendation(snapshot: dict[str, Any], assessment: dict[str, Any], *, created_by: str = "phase_d_policy_engine") -> dict[str, Any]:
    if assessment["recommendation_review_status"] not in REVIEW_STATUSES:
        raise ValueError("Unknown recommendation review status.")
    if assessment["recommendation_review_status"] in {"invalid_prediction", "insufficient_information", "stale_prediction"}:
        return {"recommendation_revision_id": canonical_hash({"snapshot": snapshot["prediction_snapshot_id"], "status": assessment["recommendation_review_status"]})[:24], "supersedes_revision_id": None, "revision_reason": "draft_blocked", "created_at": snapshot["prediction_timestamp"], "created_by": created_by, "policy_version": POLICY_VERSION, "recommendation_review_status": assessment["recommendation_review_status"], "goals": [], "actions": [], "explanation": {"what_was_predicted": None, "what_evidence_was_used": [], "what_evidence_was_not_used": ["context features", "true outcome", "target labels"], "why_actions_suggested": "Insufficient valid prediction evidence; no draft actions were generated.", "what_can_student_change": [], "uncertainty_remaining": assessment["reasons"], "advisor_approval_remains_required": True}}
    evidence = _evidence(snapshot)
    band = CLASS_NAMES[snapshot["predicted_class"]]
    catalog = {row["action_id"]: row for row in action_catalog()}
    action_ids = _action_ids(evidence, band)
    goal = {"goal_id": "goal_week4_progress", "goal_type": "learning_path", "title": "Complete the advisor-confirmed four-week learning path", "description": "Complete at least 80% of scheduled learning-path actions over four weeks.", "baseline": "completion_not_started", "target": "at_least_80_percent_action_completion", "measurement_method": "follow_up_completion_records", "start_date": "advisor_to_set", "target_date": "four_weeks_after_advisor_approval", "priority": 1, "status": "draft", "evidence_codes": evidence, "policy_version": POLICY_VERSION}
    actions = []
    for index, action_id in enumerate(action_ids, start=1):
        item = catalog[action_id]
        weekly = min(item["maximum_weekly_minutes"], 45 if action_id == "targeted_practice" else item["maximum_weekly_minutes"])
        actions.append({"action_id": action_id, "goal_id": goal["goal_id"], "action_type": item["category"], "description": item["description_template"], "frequency": f"{item['minimum_frequency']} per week", "duration_minutes": weekly, "weekly_workload_minutes": weekly, "schedule": f"week_{min(index, 4)}", "owner": "student_with_advisor_review", "required_resource": item["required_resource"], "prerequisites": item["prerequisites"], "evidence_codes": evidence, "rationale": "Rule-based action mapped to G1/G2 trajectory and predicted achievement band; non-causal.", "status": "draft"})
    conflicts = _validate_actions(actions)
    if conflicts:
        assessment = {"recommendation_review_status": "advisor_review_required", "reasons": sorted(set(assessment["reasons"] + conflicts))}
    return {"recommendation_revision_id": canonical_hash({"snapshot": snapshot["prediction_snapshot_id"], "policy": POLICY_VERSION, "actions": actions})[:24], "supersedes_revision_id": None, "revision_reason": "initial_draft", "created_at": snapshot["prediction_timestamp"], "created_by": created_by, "policy_version": POLICY_VERSION, "recommendation_review_status": "advisor_review_required", "goals": [goal], "actions": actions, "explanation": {"what_was_predicted": f"Frozen N0 five-seed ensemble predicted {band} from G1/G2; R0 reference predicted {CLASS_NAMES[snapshot['r0_reference_class']]}", "what_evidence_was_used": ["G1", "G2", "G2_minus_G1", "N0/R0 agreement", "N0 model-score diagnostics"], "what_evidence_was_not_used": ["true outcome", "target labels", "unverified context features", "legacy observed records"], "why_actions_suggested": "Actions are selected by an expert-guided, rule-based policy from the registered trajectory and achievement band.", "what_can_student_change": ["complete agreed practice", "follow a realistic schedule", "report completion and difficulty"], "uncertainty_remaining": assessment["reasons"] or ["N0 is uncalibrated; all drafts require advisor approval"], "advisor_approval_remains_required": True, "non_causal_limitation": "This policy does not establish that any action will change G3 or cause academic improvement."}, "policy_conflicts": conflicts}


def validate_recommendation(recommendation: dict[str, Any]) -> None:
    required = {"recommendation_revision_id", "policy_version", "recommendation_review_status", "goals", "actions", "explanation"}
    if required - set(recommendation): raise ValueError("Recommendation schema incomplete.")
    if recommendation["policy_version"] != POLICY_VERSION: raise ValueError("Incorrect policy version.")
    if recommendation["recommendation_review_status"] not in REVIEW_STATUSES | {"advisor_review_required"}: raise ValueError("Invalid review status.")
    if recommendation["recommendation_review_status"] == "advisor_review_required" and not recommendation["explanation"].get("advisor_approval_remains_required"): raise ValueError("Advisor review must remain required.")
    if _validate_actions(recommendation["actions"]): raise ValueError("Action safety validation failed.")
    text = json.dumps(recommendation, ensure_ascii=False).lower()
    for phrase in ["chắc chắn", "guarantee", "causes", "will increase g3"]:
        if phrase in text: raise ValueError("Causal or guarantee wording is prohibited.")


def advisor_decision(revision_id: str, decision: str, advisor_reference: str, reason: str, modified_fields: list[str] | None = None) -> dict[str, Any]:
    if decision not in {"approve", "modify", "reject", "request_more_information"}: raise ValueError("Invalid advisor decision.")
    return {"advisor_decision_id": canonical_hash([revision_id, decision, advisor_reference, reason])[:24], "recommendation_revision_id": revision_id, "decision": decision, "advisor_reference": advisor_reference, "decision_timestamp": utc_now(), "reason": reason, "modified_fields": modified_fields or []}


def follow_up(action_id: str, scheduled_date: str) -> dict[str, Any]:
    return {"follow_up_id": canonical_hash([action_id, scheduled_date])[:24], "action_id": action_id, "scheduled_date": scheduled_date, "completion_status": "scheduled", "adherence_value": None, "student_feedback": None, "advisor_feedback": None, "difficulty": None, "adverse_event": None, "recorded_at": utc_now()}
