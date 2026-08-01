"""Faithful explanations built only from supporting evidence lineage."""

from __future__ import annotations

from .policy_contracts import ActionExplanation, PolicyActionDecision, PolicyPredictionContext


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


__all__ = ["explain_action"]
