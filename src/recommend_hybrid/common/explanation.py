"""Faithful explanations built only from supporting evidence lineage."""

from __future__ import annotations

from .plan_contracts import PlanExplanation, SelectedAction
from .policy_contracts import (
    ActionExplanation,
    PolicyActionDecision,
    PolicyPredictionContext,
    PolicyRecommendationResult,
)


def explain_action(
    decision: PolicyActionDecision,
    prediction: PolicyPredictionContext,
) -> ActionExplanation:
    observed = tuple(
        f"{item.feature_name}={item.observed_value} "
        f"(severity={item.severity.value}, cutoff={item.cutoff}, source={item.source_lineage})"
        for item in decision.supporting_evidence
    )
    prediction_text = (
        f"Frozen CNN-BiLSTM class={prediction.predicted_class}; "
        f"uncertainty={prediction.uncertainty:.4f}; "
        f"seed_disagreement={prediction.seed_disagreement:.4f}."
    )
    reason = "; ".join(decision.reason_codes)
    limitation = (
        "Missing evidence: " + ", ".join(decision.missing_evidence)
        if decision.missing_evidence
        else "Priority is policy-based and is not a probability of action effectiveness."
    )
    return ActionExplanation(
        action=decision.action_id,
        observed_evidence=observed,
        prediction_context=prediction_text,
        reason=reason,
        limitation=limitation,
    )


def build_plan_explanation(
    policy_result: PolicyRecommendationResult,
    selected_actions: tuple[SelectedAction, ...],
    *,
    rejected_actions: tuple[str, ...],
    constraint_reasons: tuple[str, ...],
) -> PlanExplanation:
    """Describe only observed evidence, routing and explicit constraint outcomes."""
    observed = tuple(
        dict.fromkeys(
            f"{item.feature_name}={item.observed_value} ({item.source_lineage})"
            for action in selected_actions
            for item in action.supporting_evidence
        )
    )
    issues = tuple(
        dict.fromkeys(code for action in selected_actions for code in action.reason_codes)
    )
    selected = tuple(
        f"{action.action_id}: phù hợp với bằng chứng học tập hiện tại ({', '.join(action.reason_codes)})"
        for action in selected_actions
    )
    exclusion_details = tuple(
        reason
        for action_id in rejected_actions
        for reason in constraint_reasons
        if reason.startswith(f"{action_id}:")
    )
    exclusions = exclusion_details or tuple(
        f"{action_id}: excluded by eligibility or planning constraints"
        for action_id in rejected_actions
    )
    routing_limits = tuple(
        reason
        for reason in constraint_reasons
        if not any(reason.startswith(f"{action_id}:") for action_id in rejected_actions)
    )
    limitations = routing_limits + (
        "This policy-based plan does not establish educational or causal effectiveness.",
    )
    anchor = policy_result.prediction_anchor
    routing = (
        f"requested_cutoff={policy_result.requested_cutoff}; "
        f"stage={anchor.anchor_stage}; prediction_anchor={anchor.anchor_cutoff}; "
        f"prediction_age={anchor.prediction_age}"
    )
    return PlanExplanation(
        current_state=observed,
        main_issues=issues,
        selected_reasons=selected,
        excluded_actions=exclusions,
        limitations=limitations,
        routing=routing,
    )


__all__ = ["build_plan_explanation", "explain_action"]
