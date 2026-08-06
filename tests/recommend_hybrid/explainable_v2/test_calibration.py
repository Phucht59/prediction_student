from __future__ import annotations

import numpy as np
import pytest

from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.explainable_v2.calibration import (
    CalibratedActionRanker,
    PerActionIsotonicCalibrator,
)
from src.recommend_hybrid.explainable_v2.contracts import (
    CanonicalAction,
    RecommendationFeatures,
)
from src.recommend_hybrid.explainable_v2.ranker import FixedActionRanker


def features() -> RecommendationFeatures:
    return RecommendationFeatures(
        student_key="s1",
        course_key="AAA:2014J",
        stage=Stage.EARLY_35,
        cutoff_day=40,
        risk_probability=0.8,
        hybrid_uncertainty=0.1,
        seed_disagreement=0.02,
        course_progress=0.35,
    )


def test_isotonic_calibration_is_bounded_and_monotonic() -> None:
    calibrator = PerActionIsotonicCalibrator(minimum_rows=10)
    scores = np.linspace(0.0, 1.0, 20)
    targets = np.repeat([0, 1, 2, 3], 5)
    calibrator.fit_action(
        CanonicalAction.ASSESSMENT_COMPLETION,
        scores,
        targets,
    )
    low = calibrator.transform(CanonicalAction.ASSESSMENT_COMPLETION, 0.1)
    high = calibrator.transform(CanonicalAction.ASSESSMENT_COMPLETION, 0.9)
    assert 0.0 <= low <= high <= 1.0


def test_missing_action_calibrator_fails_closed() -> None:
    calibrator = PerActionIsotonicCalibrator(minimum_rows=10)
    with pytest.raises(RuntimeError, match="no validation calibrator"):
        calibrator.transform(CanonicalAction.RECOVER_ENGAGEMENT, 0.5)


def test_calibrated_ranker_preserves_explanations_and_sorts() -> None:
    calibrator = PerActionIsotonicCalibrator(minimum_rows=10)
    scores = np.linspace(0.0, 1.0, 20)
    targets = np.repeat([0, 1, 2, 3], 5)
    for action in (
        CanonicalAction.ASSESSMENT_COMPLETION,
        CanonicalAction.RECOVER_ENGAGEMENT,
    ):
        calibrator.fit_action(action, scores, targets)
    base = FixedActionRanker(
        {
            CanonicalAction.ASSESSMENT_COMPLETION: 0.9,
            CanonicalAction.RECOVER_ENGAGEMENT: 0.2,
        }
    )
    ranker = CalibratedActionRanker(base, calibrator)
    ranked = ranker.score(
        features(),
        (
            CanonicalAction.ASSESSMENT_COMPLETION,
            CanonicalAction.RECOVER_ENGAGEMENT,
        ),
    )
    assert ranked[0].action is CanonicalAction.ASSESSMENT_COMPLETION
    assert ranked[0].score >= ranked[1].score
