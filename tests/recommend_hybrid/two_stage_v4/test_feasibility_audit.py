from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/recommend_hybrid/two_stage_v4/feasibility_audit.py"
)
SPEC = importlib.util.spec_from_file_location("two_stage_v4_feasibility_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_frontier_maximizes_precision_subject_to_coverage() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float64)
    positive = np.asarray([1, 0, 1, 0], dtype=bool)
    correct = np.asarray([1, 0, 1, 0], dtype=bool)
    _, best = module._frontier(
        scores,
        positive,
        correct,
        minimum_coverage=0.50,
    )
    assert best is not None
    assert best.threshold == 0.9
    assert best.precision == 1.0
    assert best.coverage == 0.5


def test_frontier_distinguishes_gate_and_ranking_precision() -> None:
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float64)
    positive = np.asarray([1, 1, 0], dtype=bool)
    correct = np.asarray([0, 1, 0], dtype=bool)
    _, best = module._frontier(
        scores,
        positive,
        correct,
        minimum_coverage=1.0,
    )
    assert best is not None
    assert best.stage_a_precision == 1.0
    assert best.conditional_precision == 0.5
    assert best.precision == 0.5


def test_learner_instability_counts_mixed_targets() -> None:
    frame = pd.DataFrame(
        {
            "base_record_id": ["a", "a", "b", "b", "c"],
            "group_has_positive": [0, 1, 1, 1, 0],
        }
    )
    result = module._learner_instability(frame)
    assert result["learners"] == 3
    assert result["learners_with_multiple_groups"] == 2
    assert result["learners_with_mixed_group_targets"] == 1


def test_ranking_correct_uses_selected_action_target() -> None:
    frame = pd.DataFrame(
        {
            "top_action_index": [1, 0],
            "action_target_0": [1, 1],
            "action_target_1": [1, 0],
            "action_target_2": [0, 0],
            "action_target_3": [0, 0],
            "action_target_4": [0, 0],
        }
    )
    assert module._ranking_correct(frame).tolist() == [True, True]
