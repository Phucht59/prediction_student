from dataclasses import dataclass
from enum import Enum

from src.recommend_hybrid.counterfactual.selector import (
    CounterfactualActionSelector,
)


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class Decision:
    action_id: str
    priority: Priority
    requires_human_contact: bool = False


class BaseSelector:
    def key(self, decision, *, stage, dataset_group):
        del stage, dataset_group
        order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        return (order[decision.priority], decision.action_id)

    def order(self, decisions, *, stage, dataset_group):
        return tuple(
            sorted(
                decisions,
                key=lambda item: self.key(
                    item, stage=stage, dataset_group=dataset_group
                ),
            )
        )


def test_counterfactual_order_beats_policy_tie_breaking():
    decisions = (
        Decision("STUDY_SCHEDULE", Priority.HIGH),
        Decision("VLE_ENGAGEMENT", Priority.MEDIUM),
    )
    selector = CounterfactualActionSelector(
        BaseSelector(),
        ("VLE_ENGAGEMENT", "STUDY_SCHEDULE"),
    )
    assert [
        item.action_id
        for item in selector.order(
            decisions, stage="MIDDLE_50", dataset_group="oulad"
        )
    ] == ["VLE_ENGAGEMENT", "STUDY_SCHEDULE"]


def test_urgent_human_support_is_never_demoted():
    decisions = (
        Decision("ADVISOR_ESCALATION", Priority.CRITICAL, True),
        Decision("VLE_ENGAGEMENT", Priority.HIGH),
    )
    selector = CounterfactualActionSelector(
        BaseSelector(),
        ("VLE_ENGAGEMENT",),
    )
    ordered = selector.order(
        decisions, stage="MIDDLE_50", dataset_group="oulad"
    )
    assert ordered[0].action_id == "ADVISOR_ESCALATION"


def test_counterfactual_order_is_deterministic_for_unranked_actions():
    decisions = (
        Decision("PROGRESS_MONITORING", Priority.LOW),
        Decision("DIAGNOSTIC_CHECK", Priority.MEDIUM),
    )
    selector = CounterfactualActionSelector(BaseSelector(), ())
    first = selector.order(
        decisions, stage="MIDDLE_50", dataset_group="oulad"
    )
    second = selector.order(
        decisions, stage="MIDDLE_50", dataset_group="oulad"
    )
    assert first == second
