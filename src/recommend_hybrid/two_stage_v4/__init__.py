"""Action-aware integrated recommendation heads over frozen hybrid state."""

from .metrics import (
    ActionAwareDecision,
    ActionAwareThresholds,
    evaluate_action_aware,
    make_decisions,
)
from .model import (
    ACTION_COUNT,
    ActionAwareHeadConfig,
    ActionAwareOutput,
    HybridActionAwareRecommendationHeads,
    action_aware_loss,
)
from .selection import select_action_aware_thresholds

__all__ = [
    "ACTION_COUNT",
    "ActionAwareDecision",
    "ActionAwareHeadConfig",
    "ActionAwareOutput",
    "ActionAwareThresholds",
    "HybridActionAwareRecommendationHeads",
    "action_aware_loss",
    "evaluate_action_aware",
    "make_decisions",
    "select_action_aware_thresholds",
]
