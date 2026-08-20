"""Stats and metric contracts for the Hybrid scientific evaluation."""
from __future__ import annotations

import numpy as np

from experiments.validation.metrics import full_metrics
from experiments.validation.stats import bootstrap_ci, delong_roc, mcnemar_test, pr_auc


def test_full_metrics_has_required_keys():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.clip(rng.normal(y, 0.2), 0, 1)
    m = full_metrics(y, p, threshold=0.5)
    for key in ("pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "accuracy", "brier", "ece", "h2_mean", "tp", "fp", "tn", "fn"):
        assert key in m
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == 8


def test_mcnemar_symmetric_is_one():
    y = np.array([0, 1, 0, 1, 0, 1])
    pred = np.array([0, 1, 0, 1, 1, 0])
    out = mcnemar_test(y, pred, pred)
    assert out["p"] == 1.0
    assert out["b"] == 0 and out["c"] == 0


def test_delong_identical_scores_zero_delta():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    p = np.linspace(0.1, 0.9, 10)
    out = delong_roc(y, p, p)
    assert abs(out["delta"]) < 1e-9


def test_bootstrap_ci_finite():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
    p = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.9, 0.1, 0.6, 0.85, 0.2])
    ci = bootstrap_ci(y, p, pr_auc, n_boot=50, seed=0)
    assert np.isfinite(ci["mean"])
    assert ci["lo"] <= ci["hi"]


def test_production_hybrid_untouched_by_validation_module():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "src" / "prediction" / "model" / "hybrid.py").read_text(encoding="utf-8")
    assert "KernelExplainer" not in text
    assert "McNemar" not in text
