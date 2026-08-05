"""Student-cluster bootstrap for repeated landmark observations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterBootstrapResult:
    estimate: float
    confidence_interval: tuple[float, float]
    bootstrap_standard_error: float
    iterations: int
    cluster_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "estimate": self.estimate,
            "confidence_interval": list(self.confidence_interval),
            "bootstrap_standard_error": self.bootstrap_standard_error,
            "iterations": self.iterations,
            "cluster_count": self.cluster_count,
        }


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 20260806,
) -> ClusterBootstrapResult:
    """Bootstrap a row-level mean while resampling complete student clusters."""

    score = np.asarray(values, dtype=np.float64).reshape(-1)
    cluster = np.asarray(groups).reshape(-1)
    if len(score) != len(cluster) or not len(score):
        raise ValueError("values and groups must be non-empty and aligned")
    if not np.isfinite(score).all():
        raise ValueError("values must be finite")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    unique_groups = np.unique(cluster)
    if len(unique_groups) < 2:
        raise ValueError("cluster bootstrap requires at least two students")
    row_by_group = {value: np.flatnonzero(cluster == value) for value in unique_groups}
    rng = np.random.default_rng(random_state)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        row_index = np.concatenate([row_by_group[value] for value in drawn])
        samples[index] = float(np.mean(score[row_index]))

    alpha = 1.0 - confidence_level
    interval = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ClusterBootstrapResult(
        estimate=float(np.mean(score)),
        confidence_interval=(float(interval[0]), float(interval[1])),
        bootstrap_standard_error=float(np.std(samples, ddof=1)),
        iterations=int(iterations),
        cluster_count=int(len(unique_groups)),
    )


__all__ = ["ClusterBootstrapResult", "cluster_bootstrap_mean"]
