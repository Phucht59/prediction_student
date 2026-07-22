"""Public recommendation schemas."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationPlan:
    record_id: str
    risk_probability: float
    actions: tuple[str, ...]
    requires_expert_review: bool
