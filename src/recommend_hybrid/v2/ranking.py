"""Validation-selected multi-objective action ranking for Recommendation V2."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np

from .evaluation import ranking_metrics


@dataclass(frozen=True)
class RankingWeights:
    action_probability: float
    need_severity: float
    simulated_risk_reduction: float
    evidence_confidence: float
    workload_penalty: float
    uncertainty_penalty: float

    def __post_init__(self) -> None:
        if any(float(value) < 0.0 for value in self.__dict__.values()):
            raise ValueError("ranking weights must be non-negative")
        positive = (
            self.action_probability
            + self.need_severity
            + self.simulated_risk_reduction
            + self.evidence_confidence
        )
        if positive <= 0.0:
            raise ValueError("at least one positive ranking component is required")

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class MinMaxNormalizer:
    minimum: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> "MinMaxNormalizer":
        array = np.asarray(values, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool)
        if array.ndim != 2 or array.shape != valid.shape:
            raise ValueError("normalizer inputs must align [groups, actions]")
        minimum = np.zeros(array.shape[1], dtype=np.float64)
        maximum = np.ones(array.shape[1], dtype=np.float64)
        for column in range(array.shape[1]):
            selected = array[valid[:, column], column]
            if len(selected):
                minimum[column] = float(np.min(selected))
                maximum[column] = float(np.max(selected))
        scale = np.where(maximum - minimum < 1.0e-12, 1.0, maximum - minimum)
        return cls(minimum, scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != len(self.minimum):
            raise ValueError("normalizer transform dimension mismatch")
        return np.clip((array - self.minimum) / self.scale, 0.0, 1.0)


def _component(values: np.ndarray, name: str, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return array


def utility_scores(
    *,
    action_probability: np.ndarray,
    need_severity: np.ndarray,
    simulated_risk_reduction: np.ndarray,
    evidence_confidence: np.ndarray,
    workload: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    weights: RankingWeights,
) -> np.ndarray:
    probability = np.asarray(action_probability, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    if probability.ndim != 2 or probability.shape != valid.shape:
        raise ValueError("action probability and mask must align [groups, actions]")
    shape = probability.shape
    probability = _component(probability, "action_probability", shape)
    need = _component(need_severity, "need_severity", shape)
    reduction = _component(simulated_risk_reduction, "simulated_risk_reduction", shape)
    confidence = _component(evidence_confidence, "evidence_confidence", shape)
    burden = _component(workload, "workload", shape)
    uncertainty_array = _component(uncertainty, "uncertainty", shape)
    for name, array in (
        ("action_probability", probability),
        ("need_severity", need),
        ("simulated_risk_reduction", reduction),
        ("evidence_confidence", confidence),
        ("workload", burden),
        ("uncertainty", uncertainty_array),
    ):
        if np.any((array < 0.0) | (array > 1.0)):
            raise ValueError(f"{name} must be normalized to [0, 1]")
    score = (
        weights.action_probability * probability
        + weights.need_severity * need
        + weights.simulated_risk_reduction * reduction
        + weights.evidence_confidence * confidence
        - weights.workload_penalty * burden
        - weights.uncertainty_penalty * uncertainty_array
    )
    return np.where(valid, score, -np.inf)


def ranking_baselines(
    *,
    action_probability: np.ndarray,
    need_severity: np.ndarray,
    simulated_risk_reduction: np.ndarray,
    evidence_confidence: np.ndarray,
    workload: np.ndarray,
    mask: np.ndarray,
    prevalence: np.ndarray | None = None,
    random_state: int = 20260806,
) -> dict[str, np.ndarray]:
    probability = np.asarray(action_probability, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    shape = probability.shape
    need = _component(need_severity, "need_severity", shape)
    reduction = _component(simulated_risk_reduction, "simulated_risk_reduction", shape)
    confidence = _component(evidence_confidence, "evidence_confidence", shape)
    burden = _component(workload, "workload", shape)
    rng = np.random.default_rng(random_state)
    prevalence_vector = (
        np.asarray(prevalence, dtype=np.float64).reshape(-1)
        if prevalence is not None
        else np.ones(shape[1], dtype=np.float64)
    )
    if len(prevalence_vector) != shape[1]:
        raise ValueError("prevalence must match action count")
    return {
        "random": np.where(valid, rng.random(shape), -np.inf),
        "most_prevalent_action": np.where(valid, prevalence_vector[None, :], -np.inf),
        "lowest_workload": np.where(valid, 1.0 - burden, -np.inf),
        "need_severity_only": np.where(valid, need, -np.inf),
        "frozen_action_probability_only": np.where(valid, probability, -np.inf),
        "simulated_risk_reduction_only": np.where(valid, reduction, -np.inf),
        "evidence_confidence_only": np.where(valid, confidence, -np.inf),
    }


def select_ranking_weights(
    *,
    action_probability: np.ndarray,
    need_severity: np.ndarray,
    simulated_risk_reduction: np.ndarray,
    evidence_confidence: np.ndarray,
    workload: np.ndarray,
    uncertainty: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
    action_probability_values: Iterable[float] = (0.30, 0.45, 0.60),
    need_values: Iterable[float] = (0.10, 0.20, 0.30),
    risk_reduction_values: Iterable[float] = (0.00, 0.15, 0.30),
    evidence_values: Iterable[float] = (0.10, 0.20),
    workload_values: Iterable[float] = (0.05, 0.10, 0.15),
    uncertainty_values: Iterable[float] = (0.05, 0.10),
) -> tuple[RankingWeights, dict[str, object]]:
    """Select utility weights on validation labels only."""

    best_weights: RankingWeights | None = None
    best_metrics: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None
    for values in product(
        action_probability_values,
        need_values,
        risk_reduction_values,
        evidence_values,
        workload_values,
        uncertainty_values,
    ):
        weights = RankingWeights(*map(float, values))
        scores = utility_scores(
            action_probability=action_probability,
            need_severity=need_severity,
            simulated_risk_reduction=simulated_risk_reduction,
            evidence_confidence=evidence_confidence,
            workload=workload,
            uncertainty=uncertainty,
            mask=mask,
            weights=weights,
        )
        metrics = ranking_metrics(scores, target, mask)
        key = (
            float(metrics["precision_at_1"]),
            float(metrics["ndcg_at_3"]),
            float(metrics["recall_at_3"]),
            float(metrics["pairwise_accuracy"]),
            -float(metrics["top_action_concentration"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics
    if best_weights is None or best_metrics is None:
        raise RuntimeError("ranking weight grid produced no candidate")
    return best_weights, best_metrics


__all__ = [
    "MinMaxNormalizer",
    "RankingWeights",
    "ranking_baselines",
    "select_ranking_weights",
    "utility_scores",
]
