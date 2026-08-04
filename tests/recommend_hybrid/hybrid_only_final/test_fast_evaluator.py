from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts/recommend_hybrid/hybrid_only_final"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

tuning = importlib.import_module("tune_and_evaluate_fast")


def candidate_frame() -> pd.DataFrame:
    rows = []
    for group_id, learner, stage in [
        ("g1", "s1", "EARLY_20"),
        ("g2", "s2", "EARLY_35"),
    ]:
        rows.extend(
            [
                {
                    "group_id": group_id,
                    "base_record_id": learner,
                    "stage": stage,
                    "outer_fold": 0,
                    "course": "AAA",
                    "presentation": "2014J",
                    "action_family": "STUDY_REGULARITY",
                    "runtime_action_id": "STUDY_SCHEDULE",
                    "risk_reduction": 0.20,
                    "risk_uncertainty": 0.80,
                    "evidence_strength": 0.80,
                    "deficit_score": 0.8,
                    "workload_minutes": 30,
                    "action_available": 1,
                    "prerequisite_status": 1,
                    "silver_positive": 0,
                },
                {
                    "group_id": group_id,
                    "base_record_id": learner,
                    "stage": stage,
                    "outer_fold": 0,
                    "course": "AAA",
                    "presentation": "2014J",
                    "action_family": "VLE_ENGAGEMENT",
                    "runtime_action_id": "VLE_ENGAGEMENT",
                    "risk_reduction": 0.08,
                    "risk_uncertainty": 0.02,
                    "evidence_strength": 0.70,
                    "deficit_score": 0.7,
                    "workload_minutes": 90,
                    "action_available": 1,
                    "prerequisite_status": 1,
                    "silver_positive": 1,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_matrix_evaluator_filters_candidates_before_selecting_top() -> None:
    frame = candidate_frame()
    arrays = tuning._group_arrays(frame)
    scales = {
        "risk_scale": 0.2,
        "need_scale": 1.0,
        "uncertainty_scale": 0.1,
        "workload_scale_minutes": 150.0,
    }
    weights = {
        "risk_weight": 0.8,
        "evidence_weight": 0.1,
        "need_weight": 0.1,
        "certainty_weight": 0.1,
        "workload_weight": 0.05,
    }
    top = tuning._top_selection(
        arrays,
        tuning._score_matrix(tuning._components(arrays, scales), weights),
        {
            "minimum_risk_reduction": 0.01,
            "maximum_uncertainty": 0.20,
            "minimum_evidence": 0.40,
        },
    )
    selected = np.asarray(tuning.ACTION_FAMILIES, dtype=object)[top.top_index]
    assert set(selected) == {"VLE_ENGAGEMENT"}


def test_metrics_do_not_inflate_precision_by_counting_abstentions() -> None:
    arrays = tuning._group_arrays(candidate_frame())
    scores = np.tile(np.asarray([0.9, 0.0, 0.8, 0.0, 0.0]), (2, 1))
    top = tuning._top_selection(
        arrays,
        scores,
        {
            "minimum_risk_reduction": 0.01,
            "maximum_uncertainty": 0.20,
            "minimum_evidence": 0.40,
        },
    )
    metrics = tuning._metrics(
        arrays,
        top,
        {"minimum_top_margin": 1.0, "minimum_top_score": 1.0},
    )
    assert metrics["issued_groups"] == 0
    assert metrics["precision_at_1"] == 0.0
    assert metrics["actionable_coverage"] == 0.0


def test_same_learner_always_maps_to_same_inner_fold() -> None:
    first = tuning._stable_inner_fold("student-1", "salt", 3)
    second = tuning._stable_inner_fold("student-1", "salt", 3)
    assert first == second


def test_fast_evaluator_source_has_no_learned_ranker() -> None:
    source = Path(tuning.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("xgboost", "lightgbm", "logisticregression", "randomforest"):
        assert forbidden not in source
