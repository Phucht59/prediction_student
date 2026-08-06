"""Full-population eligibility policy for Recommendation V2."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Iterable

import numpy as np

from .evaluation import eligibility_metrics


class EligibilityDecision(str, Enum):
    NO_ACTION = "NO_ACTION"
    BEHAVIOURAL_ACTION = "BEHAVIOURAL_ACTION"
    DEFER_TO_HUMAN = "DEFER_TO_HUMAN"


@dataclass(frozen=True)
class EligibilityPolicy:
    risk_threshold: float
    minimum_need: float
    defer_entropy: float
    defer_disagreement: float

    def __post_init__(self) -> None:
        for name in (
            "risk_threshold",
            "minimum_need",
            "defer_entropy",
            "defer_disagreement",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {
            "risk_threshold": float(self.risk_threshold),
            "minimum_need": float(self.minimum_need),
            "defer_entropy": float(self.defer_entropy),
            "defer_disagreement": float(self.defer_disagreement),
        }


@dataclass(frozen=True)
class EligibilityUtility:
    true_support: float = 1.0
    false_issue: float = -0.35
    missed_support: float = -1.0
    defer_case: float = -0.10


def normalized_binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must be finite and in [0, 1]")
    clipped = np.clip(p, 1.0e-12, 1.0 - 1.0e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return entropy / np.log(2.0)


def maximum_behaviour_need(baseline_measures: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(baseline_measures, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.shape != valid.shape:
        raise ValueError("baseline measures and mask must align [groups, actions]")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("baseline measures must be normalized to [0, 1]")
    need = np.where(valid, 1.0 - values, -np.inf)
    result = np.max(need, axis=1)
    return np.where(np.isfinite(result), result, 0.0)


def apply_eligibility_policy(
    *,
    risk_probability: np.ndarray,
    need_score: np.ndarray,
    policy: EligibilityPolicy,
    predictive_entropy: np.ndarray | None = None,
    seed_disagreement: np.ndarray | None = None,
) -> np.ndarray:
    risk = np.asarray(risk_probability, dtype=np.float64).reshape(-1)
    need = np.asarray(need_score, dtype=np.float64).reshape(-1)
    if len(risk) != len(need) or not len(risk):
        raise ValueError("risk and need must be non-empty and aligned")
    if not np.isfinite(risk).all() or np.any((risk < 0.0) | (risk > 1.0)):
        raise ValueError("risk probability must be in [0, 1]")
    if not np.isfinite(need).all() or np.any((need < 0.0) | (need > 1.0)):
        raise ValueError("need score must be in [0, 1]")
    entropy = (
        normalized_binary_entropy(risk)
        if predictive_entropy is None
        else np.asarray(predictive_entropy, dtype=np.float64).reshape(-1)
    )
    disagreement = (
        np.zeros(len(risk), dtype=np.float64)
        if seed_disagreement is None
        else np.asarray(seed_disagreement, dtype=np.float64).reshape(-1)
    )
    if not (len(entropy) == len(disagreement) == len(risk)):
        raise ValueError("uncertainty arrays must align with risk")
    if not np.isfinite(entropy).all() or not np.isfinite(disagreement).all():
        raise ValueError("uncertainty arrays must be finite")

    support_candidate = (risk >= policy.risk_threshold) & (need >= policy.minimum_need)
    defer = support_candidate & (
        (entropy >= policy.defer_entropy)
        | (disagreement >= policy.defer_disagreement)
    )
    behaviour = support_candidate & ~defer
    result = np.full(len(risk), EligibilityDecision.NO_ACTION.value, dtype=object)
    result[behaviour] = EligibilityDecision.BEHAVIOURAL_ACTION.value
    result[defer] = EligibilityDecision.DEFER_TO_HUMAN.value
    return result.astype(str)


def policy_utility(
    target: np.ndarray,
    decisions: np.ndarray,
    utility: EligibilityUtility,
) -> float:
    y = np.asarray(target, dtype=np.int8).reshape(-1)
    decision = np.asarray(decisions, dtype=str).reshape(-1)
    if len(y) != len(decision) or not np.isin(y, [0, 1]).all():
        raise ValueError("target and decisions must be aligned binary rows")
    behaviour = decision == EligibilityDecision.BEHAVIOURAL_ACTION.value
    deferred = decision == EligibilityDecision.DEFER_TO_HUMAN.value
    value = np.zeros(len(y), dtype=np.float64)
    value[behaviour & (y == 1)] = utility.true_support
    value[behaviour & (y == 0)] = utility.false_issue
    value[~behaviour & ~deferred & (y == 1)] = utility.missed_support
    value[deferred] = utility.defer_case
    return float(value.mean())


def select_eligibility_policy(
    *,
    validation_target: np.ndarray,
    validation_risk_probability: np.ndarray,
    validation_need_score: np.ndarray,
    validation_entropy: np.ndarray | None = None,
    validation_seed_disagreement: np.ndarray | None = None,
    risk_thresholds: Iterable[float] = (0.35, 0.45, 0.55, 0.65, 0.75),
    minimum_needs: Iterable[float] = (0.10, 0.20, 0.30, 0.40),
    defer_entropies: Iterable[float] = (0.70, 0.80, 0.90),
    defer_disagreements: Iterable[float] = (0.05, 0.10, 0.15),
    utility: EligibilityUtility = EligibilityUtility(),
) -> tuple[EligibilityPolicy, dict[str, object]]:
    """Select one policy on validation only; no test rows are accepted here."""

    best_policy: EligibilityPolicy | None = None
    best_key: tuple[float, ...] | None = None
    best_metrics: dict[str, object] | None = None
    for risk, need, entropy, disagreement in product(
        risk_thresholds,
        minimum_needs,
        defer_entropies,
        defer_disagreements,
    ):
        policy = EligibilityPolicy(
            float(risk),
            float(need),
            float(entropy),
            float(disagreement),
        )
        decisions = apply_eligibility_policy(
            risk_probability=validation_risk_probability,
            need_score=validation_need_score,
            predictive_entropy=validation_entropy,
            seed_disagreement=validation_seed_disagreement,
            policy=policy,
        )
        metrics = eligibility_metrics(
            target=validation_target,
            risk_probability=validation_risk_probability,
            decisions=decisions,
        )
        objective = policy_utility(validation_target, decisions, utility)
        key = (
            objective,
            float(metrics["recall"]),
            float(metrics["balanced_accuracy"]),
            -float(metrics["false_issue_rate"]),
            -float(metrics["defer_rate"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_policy = policy
            best_metrics = {**metrics, "mean_utility": objective}
    if best_policy is None or best_metrics is None:
        raise RuntimeError("eligibility policy grid produced no candidate")
    return best_policy, best_metrics


__all__ = [
    "EligibilityDecision",
    "EligibilityPolicy",
    "EligibilityUtility",
    "apply_eligibility_policy",
    "maximum_behaviour_need",
    "normalized_binary_entropy",
    "policy_utility",
    "select_eligibility_policy",
]
