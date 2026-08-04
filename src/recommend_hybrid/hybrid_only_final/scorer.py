"""Deterministic hybrid-only action scoring with selective abstention.

The scorer consumes only outputs from the frozen residual CNN-BiLSTM and
cutoff-safe policy evidence.  It does not fit or invoke an auxiliary machine-
learning model.  Future silver labels are used only by offline evaluation
scripts to select a frozen configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


@dataclass(frozen=True)
class HybridOnlyScoreConfig:
    """Frozen deterministic score and selective-decision thresholds."""

    version: str
    risk_weight: float
    evidence_weight: float
    need_weight: float
    certainty_weight: float
    workload_weight: float
    minimum_risk_reduction: float
    maximum_uncertainty: float
    minimum_evidence: float
    minimum_top_margin: float
    minimum_top_score: float
    risk_scale: float = 0.10
    need_scale: float = 1.00
    uncertainty_scale: float = 0.10
    workload_scale_minutes: float = 150.0

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("score config version is required")
        for name in (
            "risk_weight",
            "evidence_weight",
            "need_weight",
            "certainty_weight",
            "workload_weight",
            "minimum_risk_reduction",
            "maximum_uncertainty",
            "minimum_evidence",
            "minimum_top_margin",
            "minimum_top_score",
            "risk_scale",
            "need_scale",
            "uncertainty_scale",
            "workload_scale_minutes",
        ):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if min(
            self.risk_weight,
            self.evidence_weight,
            self.need_weight,
            self.certainty_weight,
            self.workload_weight,
        ) < 0:
            raise ValueError("score weights must be non-negative")
        if self.risk_scale <= 0 or self.need_scale <= 0:
            raise ValueError("normalization scales must be positive")
        if self.uncertainty_scale <= 0 or self.workload_scale_minutes <= 0:
            raise ValueError("normalization scales must be positive")
        if not 0 <= self.maximum_uncertainty <= 1:
            raise ValueError("maximum_uncertainty must be in [0, 1]")
        if not 0 <= self.minimum_evidence <= 1:
            raise ValueError("minimum_evidence must be in [0, 1]")


@dataclass(frozen=True)
class HybridActionEvidence:
    """Cutoff-safe evidence for one deterministic action candidate."""

    action_id: str
    risk_reduction: float
    evidence_strength: float
    need_score: float
    uncertainty: float
    workload_minutes: int
    available: bool = True
    prerequisite_met: bool = True
    contraindicated: bool = False

    def __post_init__(self) -> None:
        if not self.action_id:
            raise ValueError("action_id is required")
        if self.workload_minutes < 0:
            raise ValueError("workload_minutes must be non-negative")
        for name in ("risk_reduction", "evidence_strength", "need_score", "uncertainty"):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.evidence_strength <= 1:
            raise ValueError("evidence_strength must be in [0, 1]")
        if not 0 <= self.uncertainty <= 1:
            raise ValueError("uncertainty must be in [0, 1]")


@dataclass(frozen=True)
class ScoredHybridAction:
    action_id: str
    score: float
    risk_component: float
    evidence_component: float
    need_component: float
    certainty_component: float
    workload_component: float
    risk_reduction: float
    uncertainty: float


@dataclass(frozen=True)
class HybridOnlyDecision:
    selected_action: ScoredHybridAction | None
    ranked_actions: tuple[ScoredHybridAction, ...]
    abstention_reason: str | None
    top_margin: float

    @property
    def issued(self) -> bool:
        return self.selected_action is not None


def _score(candidate: HybridActionEvidence, config: HybridOnlyScoreConfig) -> ScoredHybridAction:
    risk_component = _clip(max(candidate.risk_reduction, 0.0) / config.risk_scale)
    evidence_component = _clip(candidate.evidence_strength)
    need_component = _clip(max(candidate.need_score, 0.0) / config.need_scale)
    certainty_component = 1.0 - _clip(candidate.uncertainty / config.uncertainty_scale)
    workload_component = _clip(candidate.workload_minutes / config.workload_scale_minutes)
    score = (
        config.risk_weight * risk_component
        + config.evidence_weight * evidence_component
        + config.need_weight * need_component
        + config.certainty_weight * certainty_component
        - config.workload_weight * workload_component
    )
    return ScoredHybridAction(
        action_id=candidate.action_id,
        score=float(score),
        risk_component=risk_component,
        evidence_component=evidence_component,
        need_component=need_component,
        certainty_component=certainty_component,
        workload_component=workload_component,
        risk_reduction=float(candidate.risk_reduction),
        uncertainty=float(candidate.uncertainty),
    )


def score_hybrid_actions(
    candidates: Iterable[HybridActionEvidence],
    config: HybridOnlyScoreConfig,
) -> HybridOnlyDecision:
    """Rank eligible candidates and abstain unless the top decision is reliable.

    Tie-breaking is deterministic and transparent.  No action is issued when
    minimum hybrid risk reduction, evidence, uncertainty, score, or score-margin
    requirements are not satisfied.
    """

    eligible: list[HybridActionEvidence] = []
    for candidate in candidates:
        if not candidate.available or not candidate.prerequisite_met:
            continue
        if candidate.contraindicated:
            continue
        if candidate.risk_reduction < config.minimum_risk_reduction:
            continue
        if candidate.uncertainty > config.maximum_uncertainty:
            continue
        if candidate.evidence_strength < config.minimum_evidence:
            continue
        eligible.append(candidate)

    if not eligible:
        return HybridOnlyDecision(None, (), "NO_ELIGIBLE_CONFIDENT_ACTION", 0.0)

    ranked = sorted(
        (_score(item, config) for item in eligible),
        key=lambda item: (
            -item.score,
            -item.risk_reduction,
            item.uncertainty,
            item.workload_component,
            item.action_id,
        ),
    )
    top = ranked[0]
    second_score = ranked[1].score if len(ranked) > 1 else 0.0
    margin = float(top.score - second_score)
    if top.score < config.minimum_top_score:
        return HybridOnlyDecision(None, tuple(ranked), "TOP_SCORE_BELOW_THRESHOLD", margin)
    if len(ranked) > 1 and margin < config.minimum_top_margin:
        return HybridOnlyDecision(None, tuple(ranked), "TOP_MARGIN_BELOW_THRESHOLD", margin)
    return HybridOnlyDecision(top, tuple(ranked), None, margin)


__all__ = [
    "HybridActionEvidence",
    "HybridOnlyDecision",
    "HybridOnlyScoreConfig",
    "ScoredHybridAction",
    "score_hybrid_actions",
]
