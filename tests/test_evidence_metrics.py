import numpy as np
import pytest

from src.evidence_metrics import (
    bootstrap_confidence_intervals,
    classification_metrics,
    expected_calibration_error,
)


def test_evidence_metrics_are_recomputable_from_predictions():
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 1, 2, 1])
    probabilities = np.eye(3)[predictions] * 0.8 + 0.2 / 3

    metrics = classification_metrics(labels, predictions, probabilities)

    assert metrics["n_samples"] == 6
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 1, 1]]
    assert metrics["one_step_errors"] == 2
    assert metrics["two_step_errors"] == 0
    assert metrics["ordinal_mae"] == pytest.approx(2 / 6)
    assert 0 <= metrics["ece"] <= 1
    assert set(metrics["per_class"]) == {"Low", "Medium", "High"}


def test_probability_validation_rejects_invalid_rows():
    with pytest.raises(ValueError, match="positive mass"):
        expected_calibration_error([0], np.zeros((1, 3)))


def test_bootstrap_intervals_are_deterministic():
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 2, 1]
    first = bootstrap_confidence_intervals(labels, predictions, n_resamples=100, seed=7)
    second = bootstrap_confidence_intervals(labels, predictions, n_resamples=100, seed=7)
    assert first == second
