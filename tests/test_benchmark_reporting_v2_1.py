import numpy as np
import pytest
from src.evaluation.metrics import top_label_ece

def test_ece_includes_terminal_confidence_one():
    assert top_label_ece([0, 1], np.eye(3)[[0, 1]]) == 0.0
    assert top_label_ece([1, 1], np.eye(3)[[0, 1]]) == pytest.approx(0.5)

def test_ece_is_deterministic_and_empty_bins_are_safe():
    probabilities=np.array([[.8,.1,.1],[.8,.1,.1]])
    assert top_label_ece([0,1],probabilities)==top_label_ece([0,1],probabilities)
