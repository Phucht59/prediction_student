from argparse import Namespace

import numpy as np
import pytest

from scripts.run_fair_model_comparison import ALL_MODELS, CLASSICAL_MODELS, DEEP_MODELS, _settings


def test_requested_comparison_models_are_registered():
    assert CLASSICAL_MODELS == ("decision_tree", "random_forest", "svm_rbf", "xgboost", "gradient_boosting")
    assert DEEP_MODELS == ("cnn_lstm", "cnn_bilstm")
    assert len(ALL_MODELS) == 7


def test_official_protocol_rejects_reduced_budget():
    assert _settings(Namespace(smoke=False, outer_folds=5, inner_folds=3, n_trials=30)) == (5, 30)
    with pytest.raises(ValueError):
        _settings(Namespace(smoke=False, outer_folds=5, inner_folds=3, n_trials=29))
