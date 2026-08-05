from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts/recommend_hybrid/conditional_final/evaluate_release.py"
)
SPEC = importlib.util.spec_from_file_location("conditional_final_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_ranking_values_compute_precision_ndcg_and_mrr() -> None:
    scores = np.asarray([[0.1, 0.9, 0.2], [0.8, 0.7, 0.1]], dtype=float)
    targets = np.asarray([[0, 1, 0], [0, 1, 0]], dtype=np.int8)
    valid = np.ones_like(targets, dtype=bool)
    precision, ndcg, reciprocal, top = module._ranking_values(scores, targets, valid)
    assert top.tolist() == [1, 0]
    assert precision.tolist() == [1.0, 0.0]
    assert reciprocal.tolist() == [1.0, 0.5]
    assert np.all(ndcg > 0)


def test_summary_reports_action_diversity() -> None:
    scores = np.asarray(
        [
            [0.9, 0.1, 0.0, -1.0, -2.0],
            [0.0, 0.9, 0.1, -1.0, -2.0],
            [0.0, 0.1, 0.9, -1.0, -2.0],
            [0.0, 0.1, 0.2, 0.9, -2.0],
        ]
    )
    targets = np.eye(5, dtype=np.int8)[:4]
    valid = np.ones_like(scores, dtype=bool)
    result = module._summary(scores, targets, valid)
    assert result["precision_at_1"] == 1.0
    assert result["action_selection_diversity"] == 4


def test_random_control_is_reproducible() -> None:
    targets = np.asarray([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=np.int8)
    valid = np.ones_like(targets, dtype=bool)
    first = module._random_ranking_control(
        targets, valid, 1.0, repetitions=50, seed=7
    )
    second = module._random_ranking_control(
        targets, valid, 1.0, repetitions=50, seed=7
    )
    assert first == second
    assert first["p_value"] > 0


def test_protocol_keeps_end_to_end_out_of_scope() -> None:
    import yaml

    protocol = yaml.safe_load(module.PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["module_boundary"]["end_to_end_recommendability_in_scope"] is False
    assert protocol["module_boundary"]["runtime_authorized"] is False
    assert protocol["release_gates"]["ranking_only_precision_at_1_minimum"] == 0.90
