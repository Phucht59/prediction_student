"""Operational definition of the STUDY_REGULARITY treatment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPSILON = 1.0e-9


def _longest_run(mask: np.ndarray, value: bool) -> int:
    best = 0
    current = 0
    for item in mask:
        if bool(item) is value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def study_regularity_components(weekly_activity: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate bounded regularity components from non-negative weekly activity."""

    values = np.asarray(weekly_activity, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("weekly_activity must have shape [N, at least 2 weeks]")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("weekly_activity must be finite and non-negative")

    active = values > 0.0
    active_ratio = np.mean(active, axis=1)
    mean = np.mean(values, axis=1)
    has_activity = mean > 0.0
    coefficient_variation = np.divide(
        np.std(values, axis=1),
        mean,
        out=np.zeros_like(mean),
        where=has_activity,
    )
    evenness = np.where(has_activity, 1.0 / (1.0 + coefficient_variation), 0.0)

    total = np.sum(values, axis=1, keepdims=True)
    probability = np.divide(values, total, out=np.zeros_like(values), where=total > 0.0)
    entropy = -np.sum(
        np.where(probability > 0.0, probability * np.log(probability + EPSILON), 0.0),
        axis=1,
    )
    entropy = entropy / np.log(values.shape[1])

    longest_active = np.asarray(
        [_longest_run(row, True) for row in active], dtype=np.float64
    )
    longest_inactive = np.asarray(
        [_longest_run(row, False) for row in active], dtype=np.float64
    )
    streak_score = longest_active / values.shape[1]
    gap_score = 1.0 - longest_inactive / values.shape[1]
    return {
        "active_week_ratio": np.clip(active_ratio, 0.0, 1.0),
        "evenness": np.clip(evenness, 0.0, 1.0),
        "entropy": np.clip(entropy, 0.0, 1.0),
        "active_streak": np.clip(streak_score, 0.0, 1.0),
        "gap_score": np.clip(gap_score, 0.0, 1.0),
        "maximum_inactive_gap": longest_inactive.astype(np.int16),
    }


def study_regularity_score(weekly_activity: np.ndarray) -> np.ndarray:
    """Return the preregistered regularity score in [0, 1]."""

    component = study_regularity_components(weekly_activity)
    score = (
        0.30 * component["active_week_ratio"]
        + 0.20 * component["evenness"]
        + 0.20 * component["entropy"]
        + 0.15 * component["active_streak"]
        + 0.15 * component["gap_score"]
    )
    return np.clip(score, 0.0, 1.0)


@dataclass(frozen=True)
class StudyRegularityTreatmentDefinition:
    minimum_score_improvement: float = 0.20
    maximum_inactive_gap_weeks: int = 2

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_score_improvement <= 1.0:
            raise ValueError("minimum_score_improvement must lie in (0, 1]")
        if self.maximum_inactive_gap_weeks < 0:
            raise ValueError("maximum_inactive_gap_weeks must be non-negative")

    def assign(
        self,
        *,
        baseline_weekly_activity: np.ndarray,
        followup_weekly_activity: np.ndarray,
        treated_reference_score: float,
    ) -> np.ndarray:
        """Assign observed treatment from post-cutoff behaviour only.

        The reference score must be estimated from the training partition and
        frozen before validation or test assignment.
        """

        reference = float(treated_reference_score)
        if not 0.0 <= reference <= 1.0:
            raise ValueError("treated_reference_score must lie in [0, 1]")
        baseline_score = study_regularity_score(baseline_weekly_activity)
        followup_score = study_regularity_score(followup_weekly_activity)
        if baseline_score.shape != followup_score.shape:
            raise ValueError("baseline and follow-up rows must align")
        component = study_regularity_components(followup_weekly_activity)
        treatment = (
            (followup_score - baseline_score >= self.minimum_score_improvement)
            & (followup_score >= reference)
            & (
                component["maximum_inactive_gap"]
                <= self.maximum_inactive_gap_weeks
            )
        )
        return treatment.astype(np.int8)


__all__ = [
    "StudyRegularityTreatmentDefinition",
    "study_regularity_components",
    "study_regularity_score",
]
