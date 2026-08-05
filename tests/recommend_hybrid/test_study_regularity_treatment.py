from __future__ import annotations

import numpy as np

from src.recommend_hybrid.causal.study_regularity import (
    StudyRegularityTreatmentDefinition,
    study_regularity_score,
)


def test_no_activity_has_zero_regularity() -> None:
    score = study_regularity_score(np.zeros((1, 4), dtype=float))
    assert score.tolist() == [0.0]


def test_long_inactive_gap_blocks_regularity_treatment() -> None:
    definition = StudyRegularityTreatmentDefinition(
        minimum_score_improvement=0.05,
        maximum_inactive_gap_weeks=2,
    )
    baseline = np.array([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    followup = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    treatment = definition.assign(
        baseline_weekly_activity=baseline,
        followup_weekly_activity=followup,
        treated_reference_score=0.0,
    )
    assert treatment.tolist() == [0]
