import numpy as np

from src.studies.v5.common.metrics import binary_metrics, binary_metrics_per_record_threshold, multiclass_metrics


def test_v5_multiclass_metrics_are_recomputable():
    target = np.array([0, 1, 2, 0, 1, 2])
    probability = np.eye(3)[target] * 0.8 + 0.2 / 3
    result = multiclass_metrics(target, probability, regression_target=target * 5 + 5, regression_prediction=target * 5 + 5)
    assert result["macro_f1"] == 1.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0


def test_v5_binary_probability_contract_and_metrics():
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]), 0.5)
    assert result["macro_f1"] == 1.0
    assert result["at_risk_recall"] == 1.0


def test_v5_pooled_binary_metrics_use_each_outer_fold_threshold():
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.4, 0.6, 0.9])
    thresholds = np.array([0.2, 0.5, 0.7, 0.8])
    result = binary_metrics_per_record_threshold(target, probability, thresholds)
    assert result["confusion_matrix"] == [[2, 0], [1, 1]]
    assert result["threshold"] is None
    assert result["threshold_scope"] == "per_outer_fold"
