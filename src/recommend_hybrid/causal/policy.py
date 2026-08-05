"""Causal-evidence gate and recommendation lifecycle rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .protocol import STAGE_ORDER


@dataclass(frozen=True)
class CausalPolicyThresholds:
    minimum_cate: float = 0.05
    minimum_confidence_lower_bound: float = 0.0
    propensity_bounds: tuple[float, float] = (0.10, 0.90)

    def __post_init__(self) -> None:
        low, high = self.propensity_bounds
        if not 0.0 < low < high < 1.0:
            raise ValueError("propensity_bounds must lie strictly inside (0, 1)")
        if not np.isfinite(self.minimum_cate):
            raise ValueError("minimum_cate must be finite")
        if not np.isfinite(self.minimum_confidence_lower_bound):
            raise ValueError("minimum_confidence_lower_bound must be finite")


@dataclass(frozen=True)
class CausalActionDecision:
    issued: bool
    status: str
    action_id: str | None
    stage: str
    cate: float | None
    confidence_interval: tuple[float, float] | None


def gate_causal_action(
    *,
    action_id: str,
    stage: str,
    ranking_authorized: bool,
    identifiable: bool,
    cate: float | None,
    confidence_interval: tuple[float, float] | None,
    propensity: float | None,
    thresholds: CausalPolicyThresholds = CausalPolicyThresholds(),
) -> CausalActionDecision:
    """Fail closed unless ranking and causal evidence both authorize issuance."""

    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage}")
    if not ranking_authorized:
        return CausalActionDecision(
            False, "RANKING_NOT_AUTHORIZED", None, stage, cate, confidence_interval
        )
    if not identifiable:
        return CausalActionDecision(
            False,
            "CAUSAL_EVIDENCE_NOT_IDENTIFIABLE",
            None,
            stage,
            cate,
            confidence_interval,
        )
    if cate is None or confidence_interval is None or propensity is None:
        return CausalActionDecision(
            False, "CAUSAL_EVIDENCE_REQUIRED", None, stage, cate, confidence_interval
        )
    values = (float(cate), float(confidence_interval[0]), float(confidence_interval[1]), float(propensity))
    if not np.isfinite(values).all():
        return CausalActionDecision(
            False, "NON_FINITE_CAUSAL_EVIDENCE", None, stage, cate, confidence_interval
        )
    low, high = thresholds.propensity_bounds
    if not low <= float(propensity) <= high:
        return CausalActionDecision(
            False, "OUTSIDE_CAUSAL_OVERLAP", None, stage, cate, confidence_interval
        )
    if float(cate) < thresholds.minimum_cate:
        return CausalActionDecision(
            False, "EXPECTED_BENEFIT_TOO_SMALL", None, stage, cate, confidence_interval
        )
    if float(confidence_interval[0]) < thresholds.minimum_confidence_lower_bound:
        return CausalActionDecision(
            False, "EXPECTED_BENEFIT_UNCERTAIN", None, stage, cate, confidence_interval
        )
    return CausalActionDecision(
        True,
        "CAUSALLY_SUPPORTED_ACTION",
        action_id,
        stage,
        float(cate),
        (float(confidence_interval[0]), float(confidence_interval[1])),
    )


@dataclass(frozen=True)
class RecommendationEvent:
    stage: str
    action_id: str | None
    issued: bool


def resolve_recommendation_lifecycle(
    events: Iterable[RecommendationEvent],
) -> list[dict[str, object]]:
    """Apply latest-valid-recommendation-wins while retaining audit history."""

    ordered = sorted(events, key=lambda item: STAGE_ORDER.index(item.stage))
    history: list[dict[str, object]] = []
    active_index: int | None = None
    for event in ordered:
        if event.stage not in STAGE_ORDER:
            raise ValueError(f"unknown stage {event.stage}")
        if active_index is not None:
            history[active_index]["status"] = "SUPERSEDED"
            active_index = None
        if event.issued and event.action_id:
            status = "ACTIVE"
            active_index = len(history)
        else:
            status = "ABSTAINED"
        history.append(
            {
                "stage": event.stage,
                "action_id": event.action_id,
                "issued": event.issued,
                "status": status,
            }
        )
    return history


__all__ = [
    "CausalActionDecision",
    "CausalPolicyThresholds",
    "RecommendationEvent",
    "gate_causal_action",
    "resolve_recommendation_lifecycle",
]
