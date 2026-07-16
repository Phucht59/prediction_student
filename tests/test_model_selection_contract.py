import inspect

import numpy as np
import pytest

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
    training_source = inspect.getsource(model_selection.fit_training_partition_estimator)
    scoring_source = inspect.getsource(model_selection.fit_fold_predict_proba)
    assert "split_model_train_and_early_stop" in training_source
    assert "fit_transform(train_engineered, apply_oversampling=True)" in training_source
    assert "early_stop_prepared = preprocessor.transform(early_stop_engineered)" in training_source
    assert "validation_fold" not in training_source
    assert "fit_training_partition_estimator" in scoring_source
    assert "predict_with_fitted_estimator" in scoring_source


def test_outer_validation_is_not_used_for_early_stopping_or_class_weights():
    source = inspect.getsource(model_selection.fit_training_partition_estimator)
    assert "early_stop_loader" in source
    assert "model_train_partition[spec.target_col]" in source
    assert "validation_fold" not in source
    assert "factory.create_criterion" in source


def test_threshold_and_calibration_reject_locked_test_source():
    probabilities = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    labels = np.asarray([0, 1, 2])
    with pytest.raises(ValueError, match="OOF validation"):
        model_selection.optimize_class_thresholds(probabilities, labels, source="locked_test")
    with pytest.raises(ValueError, match="inner OOF"):
        model_selection.fit_temperature_policy(probabilities, labels, source="locked_test")
