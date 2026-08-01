"""Deterministic interpreter for declared dataset action-policy rules."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .abstention import automation_status
from .evidence import detect_contradictions, evaluate_eligibility
from .explanation import explain_action
from .policy_contracts import (
    AutomationStatus,
    DatasetId,
    EligibilityStatus,
    PolicyActionDecision,
    PolicyRecommendationResult,
    PredictionAnchor,
    Priority,
    RecommendationRequest,
)
from .priority import ordinal_priority
from .uncertainty import UncertaintyDisposition, uncertainty_disposition


def load_policy(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "policy_version" not in payload:
        raise ValueError(f"invalid policy config: {path}")
    return payload


def evaluation_only_result(
    *,
    dataset_id: DatasetId,
    student_key: str,
    requested_cutoff: float,
    anchor: PredictionAnchor,
    policy_version: str,
) -> PolicyRecommendationResult:
    return PolicyRecommendationResult(
        dataset_id=dataset_id,
        student_key=student_key,
        requested_cutoff=requested_cutoff,
        prediction_anchor=anchor,
        automation_status=AutomationStatus.EVALUATION_ONLY,
        action_decisions=(),
        explanation=(),
        abstention_reasons=("FINAL_STAGE_EVALUATION_ONLY",),
        policy_version=policy_version,
    )


def abstained_result(
    *,
    dataset_id: DatasetId,
    student_key: str,
    requested_cutoff: float,
    anchor: PredictionAnchor,
    reasons: tuple[str, ...],
    policy_version: str,
) -> PolicyRecommendationResult:
    return PolicyRecommendationResult(
        dataset_id=dataset_id,
        student_key=student_key,
        requested_cutoff=requested_cutoff,
        prediction_anchor=anchor,
        automation_status=AutomationStatus.ABSTAIN,
        action_decisions=(),
        explanation=(),
        abstention_reasons=reasons,
        policy_version=policy_version,
    )


def run_declared_policy(
    request: RecommendationRequest,
    *,
    anchor: PredictionAnchor,
    stage: str,
    dataset_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
) -> PolicyRecommendationResult:
    evidence_by_name = {item.feature_name: item for item in request.observed_state}
    contradictions = detect_contradictions(
        evidence_by_name, tuple(dataset_config.get("contradiction_rules", ()))
    )
    if contradictions:
        return abstained_result(
            dataset_id=request.dataset_id,
            student_key=request.student_key,
            requested_cutoff=request.requested_cutoff,
            anchor=anchor,
            reasons=contradictions,
            policy_version=dataset_config["policy_version"],
        )
    uncertainty = uncertainty_disposition(request.prediction_context, common_config)
    decisions: list[PolicyActionDecision] = []
    for action_id in dataset_config["allowed_actions"]:
        action_rule = dataset_config["actions"][action_id]
        status, supporting, missing = evaluate_eligibility(
            action_rule, stage=stage, evidence_by_name=evidence_by_name
        )
        eligible = status in {
            EligibilityStatus.ELIGIBLE,
            EligibilityStatus.REQUIRES_HUMAN_CONTACT,
        }
        priority = (
            ordinal_priority(
                supporting,
                predicted_class=request.prediction_context.predicted_class,
                class_probabilities=request.prediction_context.class_probabilities,
                dataset_config=dataset_config,
                common_config=common_config,
                stage=stage,
                action_rule=action_rule,
                uncertainty_caution=uncertainty is UncertaintyDisposition.CAUTION,
            )
            if eligible
            else Priority.NOT_APPLICABLE
        )
        decisions.append(
            PolicyActionDecision(
                action_id=action_id,
                eligibility_status=status,
                priority=priority,
                reason_codes=tuple(action_rule["reason_codes"]) if supporting else (),
                supporting_evidence=supporting,
                missing_evidence=missing,
                requires_human_contact=status is EligibilityStatus.REQUIRES_HUMAN_CONTACT,
                policy_version=dataset_config["policy_version"],
            )
        )
    supported = sum(
        decision.eligibility_status
        in {EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_HUMAN_CONTACT}
        for decision in decisions
    )
    automation, reasons = automation_status(
        uncertainty=uncertainty,
        core_features=tuple(dataset_config["core_evidence"][stage]),
        evidence_by_name=evidence_by_name,
        supported_actions=supported,
    )
    if automation is AutomationStatus.ABSTAIN:
        decisions = [
            replace(
                decision,
                eligibility_status=EligibilityStatus.NOT_APPLICABLE,
                priority=Priority.NOT_APPLICABLE,
                reason_codes=("POLICY_ABSTENTION_SUPPRESSED_ACTION",)
                if decision.supporting_evidence
                else (),
                requires_human_contact=False,
            )
            for decision in decisions
        ]
    explanations = tuple(
        explain_action(decision, request.prediction_context)
        for decision in decisions
        if decision.eligibility_status
        in {EligibilityStatus.ELIGIBLE, EligibilityStatus.REQUIRES_HUMAN_CONTACT}
    )
    return PolicyRecommendationResult(
        dataset_id=request.dataset_id,
        student_key=request.student_key,
        requested_cutoff=request.requested_cutoff,
        prediction_anchor=anchor,
        automation_status=automation,
        action_decisions=tuple(decisions),
        explanation=explanations,
        abstention_reasons=reasons,
        policy_version=dataset_config["policy_version"],
    )


__all__ = [
    "abstained_result",
    "evaluation_only_result",
    "load_policy",
    "run_declared_policy",
]
