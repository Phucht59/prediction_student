"""Ordinal-only priority computation; no probabilities or relevance scores."""

from __future__ import annotations

from typing import Any, Mapping

from .evidence import SEVERITY_RANK
from .policy_contracts import EvidenceItem, Priority

PRIORITY_RANK = {
    Priority.NOT_APPLICABLE: 0,
    Priority.LOW: 1,
    Priority.MEDIUM: 2,
    Priority.HIGH: 3,
    Priority.CRITICAL: 4,
}
RANK_PRIORITY = {value: key for key, value in PRIORITY_RANK.items()}


def ordinal_priority(
    supporting_evidence: tuple[EvidenceItem, ...],
    *,
    predicted_class: int,
    class_probabilities: tuple[float, ...],
    dataset_config: Mapping[str, Any],
    common_config: Mapping[str, Any],
    stage: str,
    action_rule: Mapping[str, Any],
    uncertainty_caution: bool,
) -> Priority:
    if not supporting_evidence:
        return Priority.NOT_APPLICABLE
    strongest = max(supporting_evidence, key=lambda item: SEVERITY_RANK[item.severity])
    priority = Priority(common_config["priority_from_severity"][strongest.severity.value])
    rank = PRIORITY_RANK[priority]
    risk_index = int(dataset_config["risk_probability_index"])
    risk_probability = class_probabilities[risk_index]
    if (
        predicted_class in set(dataset_config["at_risk_prediction_classes"])
        and risk_probability >= float(dataset_config["risk_probability_priority_threshold"])
    ):
        rank = min(
            4,
            rank + int(common_config["risk_context"]["maximum_priority_steps"]),
        )
    stage_cap = Priority(dataset_config["stage_priority_caps"][stage])
    rank = min(rank, PRIORITY_RANK[stage_cap])
    if "priority_cap" in action_rule:
        rank = min(rank, PRIORITY_RANK[Priority(action_rule["priority_cap"])])
    if uncertainty_caution:
        uncertainty_cap = Priority(common_config["uncertainty"]["caution_priority_cap"])
        rank = min(rank, PRIORITY_RANK[uncertainty_cap])
    return RANK_PRIORITY[max(rank, 1)]


__all__ = ["PRIORITY_RANK", "ordinal_priority"]
