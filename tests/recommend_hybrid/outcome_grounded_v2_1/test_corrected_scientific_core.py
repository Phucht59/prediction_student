from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts/recommend_hybrid/v2_1"
sys.path.insert(0, str(SCRIPT_DIR))

from scientific_core import (  # noqa: E402
    FeaturePreprocessor,
    RelevanceTransformer,
    add_baseline_scores,
    aggregate_metrics,
    fit_ranker,
    model_selection_key,
    predict_ranker,
    random_null_distribution,
)


def synthetic_frame(learners: int = 18) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    actions = ["ASSESSMENT_COMPLETION", "VLE_ENGAGEMENT", "CONTENT_REVIEW"]
    for learner in range(learners):
        for stage in ["EARLY_20", "EARLY_35"]:
            for action_index, action in enumerate(actions):
                rows.append(
                    {
                        "group_id": f"{learner}|{stage}",
                        "base_record_id": str(learner),
                        "stage": stage,
                        "outer_fold": learner % 3,
                        "course": "AAA",
                        "presentation": "2014B",
                        "action_family": action,
                        "risk_probability": rng.random(),
                        "risk_uncertainty": rng.random() / 10,
                        "active_days": rng.random() * 10,
                        "inactive_streak": rng.random() * 5,
                        "activity_trend": rng.normal(),
                        "assessment_progress": rng.random(),
                        "vle_intensity": rng.random(),
                        "opportunity_count": 1 + action_index,
                        "deficit_score": rng.random(),
                        "evidence_strength": rng.random(),
                        "workload_minutes": 30 + 30 * action_index,
                        "counterfactual_v1_delta": rng.random() / 5,
                        "action_needed": 1,
                        "action_available": 1,
                        "future_behavior_signal": rng.normal(loc=1 - action_index / 2),
                        "future_proximal_signal": rng.normal(),
                        "proximal_outcome_available": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_relevance_thresholds_are_train_only() -> None:
    frame = synthetic_frame()
    train = frame[frame.outer_fold != 2].reset_index(drop=True)
    test = frame[frame.outer_fold == 2].reset_index(drop=True)
    transformer = RelevanceTransformer(min_model_rows=5).fit(train)
    before = {key: spec.grade_thresholds for key, spec in transformer.specs.items()}
    altered_test = test.copy()
    altered_test["future_behavior_signal"] = 1_000_000.0
    transformer.transform(altered_test)
    after = {key: spec.grade_thresholds for key, spec in transformer.specs.items()}
    assert before == after


def test_test_only_category_does_not_change_train_schema() -> None:
    frame = synthetic_frame()
    train = frame[frame.action_family != "CONTENT_REVIEW"].reset_index(drop=True)
    test = frame[frame.action_family == "CONTENT_REVIEW"].reset_index(drop=True)
    transformer = RelevanceTransformer(min_model_rows=5).fit(train)
    train = transformer.transform(train)
    test = transformer.transform(test)
    preprocessor = FeaturePreprocessor().fit(train)
    train_matrix = preprocessor.transform(train)
    test_matrix = preprocessor.transform(test)
    assert train_matrix.shape[1] == test_matrix.shape[1]


def test_pairwise_ranker_has_two_sided_training_pairs() -> None:
    frame = synthetic_frame()
    transformer = RelevanceTransformer(min_model_rows=5).fit(frame)
    labelled = transformer.transform(frame).sort_values(["group_id", "action_family"]).reset_index(drop=True)
    preprocessor = FeaturePreprocessor().fit(labelled)
    matrix = preprocessor.transform(labelled)
    bundle = fit_ranker("pairwise_logistic", matrix, labelled, {"C": 1.0}, 11)
    scores = predict_ranker(bundle, matrix)
    assert scores.shape == (len(labelled),)
    assert np.isfinite(scores).all()


def test_map_at_3_uses_relevant_item_denominator() -> None:
    frame = pd.DataFrame(
        {
            "group_id": ["g"] * 3,
            "base_record_id": ["s"] * 3,
            "stage": ["x"] * 3,
            "outer_fold": [0] * 3,
            "action_family": ["a", "b", "c"],
            "graded_relevance": [1, 0, 1],
            "continuous_relevance": [1.0, 0.0, 0.5],
            "score": [3.0, 2.0, 1.0],
        }
    )
    metrics = aggregate_metrics(frame, "score")
    assert metrics["map_at_3"] == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)


def test_random_null_is_deterministic() -> None:
    frame = synthetic_frame(6)
    transformer = RelevanceTransformer(min_model_rows=3).fit(frame)
    labelled = transformer.transform(frame)
    first = random_null_distribution(labelled, 5, 99)
    second = random_null_distribution(labelled, 5, 99)
    np.testing.assert_allclose(first, second)


def test_popular_baseline_uses_training_relevance_not_frequency() -> None:
    frame = synthetic_frame(6)
    transformer = RelevanceTransformer(min_model_rows=3).fit(frame)
    labelled = transformer.transform(frame)
    train = labelled.iloc[:-3].copy()
    test = labelled.iloc[-3:].copy()
    scored = add_baseline_scores(train, test, 1)
    expected = train.groupby("action_family")["continuous_relevance"].mean()
    for row in scored.itertuples():
        assert row.popular_score == pytest.approx(expected.get(row.action_family, 0.0))


def test_model_selection_prefers_primary_then_tiebreakers() -> None:
    stronger = {
        "mean_ndcg_at_3": 0.60,
        "mean_precision_at_1": 0.40,
        "worst_ndcg_at_3": 0.50,
        "mean_action_diversity": 4,
    }
    weaker = {
        "mean_ndcg_at_3": 0.59,
        "mean_precision_at_1": 0.99,
        "worst_ndcg_at_3": 0.99,
        "mean_action_diversity": 5,
    }
    assert model_selection_key(stronger) > model_selection_key(weaker)


def test_corrected_evaluator_registers_all_model_families() -> None:
    source = (SCRIPT_DIR / "corrected_nested_evaluation.py").read_text(encoding="utf-8")
    config = json.loads(
        json.dumps(
            __import__("yaml").safe_load(
                (ROOT / "configs/recommend_hybrid/outcome_grounded_v2_1.yaml").read_text(
                    encoding="utf-8"
                )
            )
        )
    )
    for family in config["models"]["candidates"]:
        assert family in source or "candidate_grid(config)" in source
    assert "names=['interaction_logistic']" not in source.replace(" ", "")


def test_corrected_release_is_fail_closed() -> None:
    source = (SCRIPT_DIR / "corrected_release.py").read_text(encoding="utf-8")
    assert "negative_controls_retrained/SUMMARY.csv" in source
    assert "ablations_executed/SUMMARY.csv" in source
    assert "AUTHORIZED_FOR_INTEGRATION" in source
    assert "NOT_AUTHORIZED" in source


def test_historical_preliminary_artifacts_are_not_overwritten() -> None:
    source = (SCRIPT_DIR / "corrected_nested_evaluation.py").read_text(encoding="utf-8")
    assert 'FINAL = OUT / "final_oof"' in source
    assert 'OUT / "NESTED_OOF_RESULTS.json"' not in source
