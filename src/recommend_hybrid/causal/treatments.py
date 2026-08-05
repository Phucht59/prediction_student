"""Train-fitted observational treatment definitions for canonical actions.

The treatment is not "the learner received a recommendation" because the
system has not been deployed. It is the observed post-landmark behaviour that
matches the action. All reference thresholds must be fitted on the training
partition and frozen before validation/test assignment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.recommend_hybrid.final.actions import ACTION_ORDER, canonical_action_id


@dataclass(frozen=True)
class ActionTreatmentSpec:
    action_id: str
    measure_name: str
    minimum_improvement: float
    reference_quantile: float = 0.50

    def __post_init__(self) -> None:
        canonical = canonical_action_id(self.action_id)
        if canonical != self.action_id:
            raise ValueError("action_id must use the canonical identity")
        if not self.measure_name:
            raise ValueError("measure_name must be non-empty")
        if not 0.0 < self.minimum_improvement <= 1.0:
            raise ValueError("minimum_improvement must lie in (0, 1]")
        if not 0.0 <= self.reference_quantile <= 1.0:
            raise ValueError("reference_quantile must lie in [0, 1]")


ACTION_TREATMENT_SPECS: Mapping[str, ActionTreatmentSpec] = {
    "ASSESSMENT_COMPLETION": ActionTreatmentSpec(
        "ASSESSMENT_COMPLETION",
        "assessment_completion_rate",
        0.15,
    ),
    "STUDY_REGULARITY": ActionTreatmentSpec(
        "STUDY_REGULARITY",
        "study_regularity_score",
        0.20,
    ),
    "VLE_ENGAGEMENT": ActionTreatmentSpec(
        "VLE_ENGAGEMENT",
        "vle_active_day_rate",
        0.15,
    ),
    "QUIZ_OR_RETRIEVAL_PRACTICE": ActionTreatmentSpec(
        "QUIZ_OR_RETRIEVAL_PRACTICE",
        "retrieval_practice_rate",
        0.15,
    ),
    "CONTENT_REVIEW": ActionTreatmentSpec(
        "CONTENT_REVIEW",
        "content_review_coverage",
        0.15,
    ),
}

if tuple(ACTION_TREATMENT_SPECS) != ACTION_ORDER:
    raise RuntimeError("causal treatment registry must match the final action order")


@dataclass(frozen=True)
class FittedActionTreatmentRule:
    action_id: str
    measure_name: str
    minimum_improvement: float
    minimum_followup_level: float
    fitted_on_split: str = "train"

    def __post_init__(self) -> None:
        if canonical_action_id(self.action_id) != self.action_id:
            raise ValueError("action_id must be canonical")
        for value in (self.minimum_improvement, self.minimum_followup_level):
            if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("treatment thresholds must be finite in [0, 1]")
        if self.fitted_on_split != "train":
            raise ValueError("treatment rules may only be fitted on the train split")

    def assign(
        self,
        baseline_measure: np.ndarray,
        followup_measure: np.ndarray,
    ) -> np.ndarray:
        baseline = np.asarray(baseline_measure, dtype=np.float64).reshape(-1)
        followup = np.asarray(followup_measure, dtype=np.float64).reshape(-1)
        if len(baseline) != len(followup):
            raise ValueError("baseline and follow-up measures must align")
        if not np.isfinite(baseline).all() or not np.isfinite(followup).all():
            raise ValueError("treatment measures must be finite")
        if ((baseline < 0.0) | (baseline > 1.0)).any() or (
            (followup < 0.0) | (followup > 1.0)
        ).any():
            raise ValueError("treatment measures must be normalized to [0, 1]")
        return (
            (followup - baseline >= self.minimum_improvement)
            & (followup >= self.minimum_followup_level)
        ).astype(np.int8)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "measure_name": self.measure_name,
            "minimum_improvement": self.minimum_improvement,
            "minimum_followup_level": self.minimum_followup_level,
            "fitted_on_split": self.fitted_on_split,
        }


def fit_action_treatment_rule(
    *,
    action_id: str,
    baseline_measure: np.ndarray,
    followup_measure: np.ndarray,
) -> FittedActionTreatmentRule:
    """Fit the action threshold from train rows only.

    The reference level is the configured quantile among train rows that first
    satisfy the preregistered minimum improvement. If no such row exists the
    action is not identifiable and fitting fails closed.
    """

    canonical = canonical_action_id(action_id)
    spec = ACTION_TREATMENT_SPECS[canonical]
    baseline = np.asarray(baseline_measure, dtype=np.float64).reshape(-1)
    followup = np.asarray(followup_measure, dtype=np.float64).reshape(-1)
    if len(baseline) != len(followup) or not len(baseline):
        raise ValueError("non-empty aligned train measures are required")
    if not np.isfinite(baseline).all() or not np.isfinite(followup).all():
        raise ValueError("train measures must be finite")
    if ((baseline < 0.0) | (baseline > 1.0)).any() or (
        (followup < 0.0) | (followup > 1.0)
    ).any():
        raise ValueError("train measures must be normalized to [0, 1]")
    improved = followup - baseline >= spec.minimum_improvement
    if not improved.any():
        raise ValueError("no train row satisfies the minimum treatment improvement")
    reference = float(np.quantile(followup[improved], spec.reference_quantile))
    return FittedActionTreatmentRule(
        action_id=canonical,
        measure_name=spec.measure_name,
        minimum_improvement=spec.minimum_improvement,
        minimum_followup_level=reference,
    )


__all__ = [
    "ACTION_TREATMENT_SPECS",
    "ActionTreatmentSpec",
    "FittedActionTreatmentRule",
    "fit_action_treatment_rule",
]
