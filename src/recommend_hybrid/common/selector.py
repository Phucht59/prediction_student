"""Deterministic ordering for Phase 3 policy decisions; no scores are created."""

from __future__ import annotations

from typing import Any, Mapping

from .policy_contracts import PolicyActionDecision, Priority

PRIORITY_ORDER = {
    Priority.CRITICAL: 0,
    Priority.HIGH: 1,
    Priority.MEDIUM: 2,
    Priority.LOW: 3,
}


class DeterministicActionSelector:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def key(self, decision: PolicyActionDecision, *, stage: str, dataset_group: str) -> tuple:
        metadata = self.config["action_metadata"][decision.action_id]
        urgency = int(self.config[dataset_group]["stage_urgency"].get(stage, 0))
        return (
            PRIORITY_ORDER[decision.priority],
            len(decision.missing_evidence),
            -len(decision.supporting_evidence),
            -urgency,
            -int(metadata["directness"]),
            int(metadata["weekly_minutes"]),
            decision.action_id,
        )

    def order(
        self,
        decisions: tuple[PolicyActionDecision, ...],
        *,
        stage: str,
        dataset_group: str,
    ) -> tuple[PolicyActionDecision, ...]:
        eligible = tuple(
            item for item in decisions if item.priority is not Priority.NOT_APPLICABLE
        )
        return tuple(sorted(eligible, key=lambda item: self.key(item, stage=stage, dataset_group=dataset_group)))


__all__ = ["DeterministicActionSelector", "PRIORITY_ORDER"]
