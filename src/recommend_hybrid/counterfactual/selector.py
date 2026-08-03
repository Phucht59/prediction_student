"""Counterfactual preference ordering layered over the safe policy selector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol


class BaseActionSelector(Protocol):
    def key(self, decision: Any, *, stage: str, dataset_group: str) -> tuple:
        """Return the existing deterministic policy key."""

    def order(
        self,
        decisions: Sequence[Any],
        *,
        stage: str,
        dataset_group: str,
    ) -> tuple[Any, ...]:
        """Return the existing eligible deterministic ordering."""


class CounterfactualActionSelector:
    """Prefer risk-reducing actions while preserving urgent human escalation.

    The existing selector remains the final deterministic tie-breaker. Critical
    or high-priority human support is never demoted below model-scored actions.
    """

    _URGENT = frozenset({"CRITICAL", "HIGH"})

    def __init__(
        self,
        base_selector: BaseActionSelector,
        preferred_action_order: Sequence[str],
    ) -> None:
        self.base_selector = base_selector
        self.preferred_action_order = tuple(
            dict.fromkeys(str(item) for item in preferred_action_order)
        )
        self._rank = {
            action_id: index
            for index, action_id in enumerate(self.preferred_action_order)
        }

    def key(self, decision: Any, *, stage: str, dataset_group: str) -> tuple:
        priority = getattr(decision.priority, "value", str(decision.priority))
        urgent_human = bool(
            decision.requires_human_contact and priority in self._URGENT
        )
        action_id = str(decision.action_id)
        is_ranked = action_id in self._rank
        preferred_index = self._rank.get(action_id, len(self._rank))
        return (
            0 if urgent_human else 1,
            0 if is_ranked else 1,
            preferred_index,
            *self.base_selector.key(
                decision,
                stage=stage,
                dataset_group=dataset_group,
            ),
        )

    def order(
        self,
        decisions: Sequence[Any],
        *,
        stage: str,
        dataset_group: str,
    ) -> tuple[Any, ...]:
        eligible = self.base_selector.order(
            decisions,
            stage=stage,
            dataset_group=dataset_group,
        )
        return tuple(
            sorted(
                eligible,
                key=lambda item: self.key(
                    item,
                    stage=stage,
                    dataset_group=dataset_group,
                ),
            )
        )


__all__ = ["BaseActionSelector", "CounterfactualActionSelector"]
