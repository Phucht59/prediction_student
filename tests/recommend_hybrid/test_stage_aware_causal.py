from __future__ import annotations

import numpy as np
import pytest

from src.recommend_hybrid.causal import (
    AIPWConfig,
    CrossFittedAIPW,
    FittedActionTreatmentRule,
    RecommendationEvent,
    StageActionTrialData,
    StageAwareCausalEvaluator,
    cluster_bootstrap_mean,
    fit_action_treatment_rule,
    resolve_recommendation_lifecycle,
    stage_from_fraction,
    validate_temporal_columns,
)
from src.recommend_hybrid.causal.protocol import LANDMARK_STAGES, STAGE_ORDER
from src.recommend_hybrid.final.metrics import (
    ActionAwareThresholds,
    make_decisions,
)


def test_four_stage_landmark_contract() -> None:
    assert STAGE_ORDER == ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")
    assert stage_from_fraction(0.75) == "LATE_75"
    assert LANDMARK_STAGES["LATE_75"].treatment_end_fraction == 1.0


def test_late_75_requires_new_threshold_vector() -> None:
    legacy = ActionAwareThresholds(
        stage_gate_probability=(0.4, 0.4, 0.4),
        direct_action_blend=0.5,
        minimum_action_probability=0.2,
        minimum_action_margin=0.0,
    )
    with pytest.raises(ValueError, match="LATE_75"):
        make_decisions(
            direct_gate_logits=np.array([2.0]),
            action_logits=np.array([[2.0, 0.0, 0.0, 0.0, 0.0]]),
            action_mask=np.ones((1, 5), dtype=bool),
            stages=["LATE_75"],
            thresholds=legacy,
        )

    current = ActionAwareThresholds(
        stage_gate_probability=(0.4, 0.4, 0.4, 0.4),
        direct_action_blend=0.5,
        minimum_action_probability=0.2,
        minimum_action_margin=0.0,
    )
    decision = make_decisions(
        direct_gate_logits=np.array([2.0]),
        action_logits=np.array([[2.0, 0.0, 0.0, 0.0, 0.0]]),
        action_mask=np.ones((1, 5), dtype=bool),
        stages=["LATE_75"],
        thresholds=current,
    )
    assert bool(decision.issued[0])


def test_temporal_guard_rejects_future_information() -> None:
    validate_temporal_columns(
        stage="MIDDLE_50",
        maximum_baseline_progress=0.50,
        minimum_treatment_progress=0.51,
        maximum_treatment_progress=0.75,
    )
    with pytest.raises(ValueError, match="post-cutoff"):
        validate_temporal_columns(
            stage="MIDDLE_50",
            maximum_baseline_progress=0.60,
            minimum_treatment_progress=0.61,
            maximum_treatment_progress=0.75,
        )


def test_treatment_rule_is_fitted_on_train_and_replayed() -> None:
    baseline = np.array([0.10, 0.20, 0.30, 0.40])
    followup = np.array([0.35, 0.45, 0.32, 0.80])
    rule = fit_action_treatment_rule(
        action_id="STUDY_REGULARITY",
        baseline_measure=baseline,
        followup_measure=followup,
    )
    assert isinstance(rule, FittedActionTreatmentRule)
    assert rule.fitted_on_split == "train"
    assigned = rule.assign(
        np.array([0.10, 0.50]),
        np.array([0.90, 0.55]),
    )
    assert assigned.tolist() == [1, 0]


def test_cross_fitted_aipw_detects_positive_synthetic_effect() -> None:
    rng = np.random.default_rng(20260806)
    row_count = 600
    x = rng.normal(size=(row_count, 4))
    propensity = 1.0 / (1.0 + np.exp(-(0.7 * x[:, 0] - 0.5 * x[:, 1])))
    treatment = rng.binomial(1, propensity)
    outcome_probability = 1.0 / (
        1.0 + np.exp(-(-0.8 + 1.2 * treatment + 0.6 * x[:, 0] - 0.4 * x[:, 1]))
    )
    outcome = rng.binomial(1, outcome_probability)
    estimator = CrossFittedAIPW(config=AIPWConfig(n_splits=3, random_state=20260806))
    result = estimator.fit_predict(
        x,
        treatment,
        outcome,
        groups=np.arange(row_count),
    )
    assert result.ate > 0.05
    assert np.isfinite(result.cate).all()
    assert set(result.fold_id) == {0, 1, 2}


def test_cluster_bootstrap_resamples_whole_students() -> None:
    values = np.array([1.0, 1.0, 3.0, 3.0])
    groups = np.array([10, 10, 20, 20])
    result = cluster_bootstrap_mean(
        values,
        groups,
        iterations=200,
        random_state=42,
    )
    assert result.estimate == pytest.approx(2.0)
    assert result.cluster_count == 2
    assert result.confidence_interval[0] <= result.estimate <= result.confidence_interval[1]


def test_latest_valid_recommendation_wins() -> None:
    history = resolve_recommendation_lifecycle(
        [
            RecommendationEvent("EARLY_20", "VLE_ENGAGEMENT", True),
            RecommendationEvent("MIDDLE_50", "STUDY_REGULARITY", True),
            RecommendationEvent("LATE_75", None, False),
        ]
    )
    assert history[0]["status"] == "SUPERSEDED"
    assert history[1]["status"] == "SUPERSEDED"
    assert history[2]["status"] == "ABSTAINED"
    assert sum(row["status"] == "ACTIVE" for row in history) == 0


def test_pipeline_rejects_mismatched_student_groups() -> None:
    trial = StageActionTrialData(
        stage="EARLY_20",
        action_id="STUDY_REGULARITY",
        features=np.ones((6, 2)),
        treatment=np.array([0, 1, 0, 1, 0, 1]),
        outcome=np.array([0, 1, 0, 1, 0, 1]),
        groups=np.array([1, 2, 3, 4, 5, 6]),
        student_ids=np.array([1, 2, 3, 4, 5, 99]),
        maximum_baseline_progress=0.20,
        minimum_treatment_progress=0.21,
        maximum_treatment_progress=0.35,
    )
    with pytest.raises(ValueError, match="groups must equal student_ids"):
        trial.validate()


def test_evaluator_fails_closed_when_arm_too_small() -> None:
    trial = StageActionTrialData(
        stage="EARLY_20",
        action_id="STUDY_REGULARITY",
        features=np.arange(24, dtype=float).reshape(12, 2),
        treatment=np.array([0] * 10 + [1] * 2),
        outcome=np.array([0, 1] * 6),
        groups=np.arange(12),
        student_ids=np.arange(12),
        maximum_baseline_progress=0.20,
        minimum_treatment_progress=0.21,
        maximum_treatment_progress=0.35,
    )
    evaluator = StageAwareCausalEvaluator(
        aipw_config=AIPWConfig(n_splits=3, random_state=42),
        bootstrap_iterations=200,
    )
    with pytest.raises(ValueError, match="n_splits"):
        evaluator.evaluate(trial)
