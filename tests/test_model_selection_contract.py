import inspect

import numpy as np
import pytest

from scripts import run_pipeline
from scripts import optimize_model_selection
from src import model_selection


def test_optuna_objective_does_not_accept_locked_test():
    signature = inspect.signature(model_selection.objective_mean_cv_f1)
    assert "locked_test" not in signature.parameters
    assert "test" not in signature.parameters


def test_threshold_optimizer_requires_oof_validation_source():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    labels = np.asarray([0, 1, 2])
    with pytest.raises(ValueError, match="OOF validation"):
        model_selection.optimize_class_thresholds(probabilities, labels, source="locked_test")


def test_threshold_optimizer_returns_frozen_policy_from_oof():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    labels = np.asarray([0, 1, 2])
    policy = model_selection.optimize_class_thresholds(probabilities, labels, source="oof_validation", grid=[0.3, 0.5])
    assert policy["type"] == "class_thresholds"
    assert policy["source"] == "oof_validation"
    assert len(policy["thresholds"]) == 3


def test_fold_training_oversamples_only_training_fold_by_contract():
    source = inspect.getsource(model_selection.fit_fold_predict_proba)
    assert "split_model_train_and_early_stop" in source
    assert "fit_transform(train_engineered, apply_oversampling=True)" in source
    assert "early_stop_prepared = preprocessor.transform(early_stop_engineered)" in source
    assert "validation_prepared = preprocessor.transform(validation_engineered)" in source
    assert "fit_transform(validation" not in source


def test_outer_validation_is_not_used_for_early_stopping_or_class_weights():
    source = inspect.getsource(model_selection.fit_fold_predict_proba)
    assert "train_model(model, train_loader, early_stop_loader" in source
    assert "calculate_class_weights(model_train_fold" in source
    assert "calculate_class_weights(train_fold" not in source
    assert "train_model(model, train_loader, validation_loader" not in source


def test_threshold_and_calibration_reject_locked_test_source():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    labels = np.asarray([0, 1, 2])
    with pytest.raises(ValueError, match="OOF validation"):
        model_selection.optimize_class_thresholds(probabilities, labels, source="locked_test")
    with pytest.raises(ValueError, match="inner OOF"):
        model_selection.fit_temperature_policy(probabilities, labels, source="locked_test")


def test_nested_outer_fold_freezes_inner_selection_before_scoring_outer_fold():
    source = inspect.getsource(optimize_model_selection.evaluate_deep_outer_fold)
    assert "run_optuna_cv_search(" in source
    assert "collect_oof_by_seed(" in source
    assert "evaluate_ensemble_strategies(inner_oof" in source
    assert "fit_fold_predict_proba(" in source
    assert "validation_fold=outer_val" in source
    assert "apply_selected_strategy" in source


def test_run_pipeline_persists_selected_threshold_and_ensemble_metadata():
    source = inspect.getsource(run_pipeline.main)
    assert '"validation_only_selection": selection_config' in source
    assert '"selected_ensemble_method": selected_ensemble_method' in source
    assert '"selected_calibration_policy": selected_calibration_policy' in source
    assert '"selected_threshold_policy": selected_threshold_policy' in source
