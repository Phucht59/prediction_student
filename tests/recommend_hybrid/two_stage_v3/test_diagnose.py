from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/recommend_hybrid/two_stage_v3/diagnose.py"
)
SPEC = importlib.util.spec_from_file_location("two_stage_v3_diagnose_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnose = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = diagnose
SPEC.loader.exec_module(diagnose)


def test_roc_auc_is_one_for_perfect_ordering() -> None:
    y = np.asarray([0, 0, 1, 1])
    score = np.asarray([0.1, 0.2, 0.8, 0.9])
    assert diagnose._roc_auc(y, score) == 1.0


def test_average_precision_is_one_for_perfect_ordering() -> None:
    y = np.asarray([0, 1, 0, 1])
    score = np.asarray([0.1, 0.9, 0.2, 0.8])
    assert diagnose._average_precision(y, score) == 1.0


def test_best_precision_respects_minimum_recall() -> None:
    y = np.asarray([1, 1, 0, 0])
    score = np.asarray([0.9, 0.8, 0.7, 0.6])
    result = diagnose._best_precision_at_recall(y, score, 0.5)
    assert result["precision"] == 1.0
    assert result["recall"] >= 0.5


def test_safe_ratio_handles_zero_denominator() -> None:
    assert diagnose._safe_ratio(1, 0) == 0.0
