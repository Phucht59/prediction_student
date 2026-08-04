from __future__ import annotations

import numpy as np
import pandas as pd

from src.recommend_hybrid.two_stage_v3.data import (
    STAGE_ORDER,
    apply_scaler,
    fit_scaler,
    load_two_stage_arrays,
)


def group_frame() -> pd.DataFrame:
    rows = []
    for index, stage in enumerate(STAGE_ORDER):
        row = {
            "group_id": f"g{index}",
            "base_record_id": f"learner{index}",
            "stage": stage,
            "outer_fold": index,
            "course": "AAA",
            "presentation": "2014J",
            "group_has_positive": int(index != 1),
            "candidate_count": 2,
            "maximum_risk_reduction": 0.2,
            "mean_risk_reduction": 0.1,
            "maximum_deficit": 0.75,
            "mean_evidence_strength": 0.6,
            "maximum_evidence_strength": 0.8,
            "top_counterfactual_margin": 0.1,
            "risk_probability": 0.7,
            "risk_entropy": 0.4,
            "seed_disagreement": 0.03,
            "risk_confidence": 0.7,
        }
        row.update({f"student_state_{column:03d}": float(index + column) for column in range(64)})
        row.update({f"tabular_expert_{column:03d}": float(index - column) for column in range(32)})
        rows.append(row)
    return pd.DataFrame(rows)


def action_frame() -> pd.DataFrame:
    rows = []
    positive_action = {"g0": 1, "g1": None, "g2": 2}
    for group_index in range(3):
        for action_index in (1, 2):
            rows.append(
                {
                    "group_id": f"g{group_index}",
                    "base_record_id": f"learner{group_index}",
                    "stage": STAGE_ORDER[group_index],
                    "outer_fold": group_index,
                    "course": "AAA",
                    "presentation": "2014J",
                    "action_family": "STUDY_REGULARITY" if action_index == 1 else "VLE_ENGAGEMENT",
                    "action_index": action_index,
                    "risk_reduction": 0.1 * action_index,
                    "risk_uncertainty": 0.05,
                    "evidence_strength": 0.8,
                    "deficit_score": 0.75,
                    "opportunity_count": 4,
                    "workload_minutes": 30 + action_index,
                    "action_available": 1,
                    "prerequisite_status": 1,
                    "silver_positive": int(positive_action[f"g{group_index}"] == action_index),
                    "group_has_positive": int(positive_action[f"g{group_index}"] is not None),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_arrays_preserve_targets_and_masks() -> None:
    arrays, schema = load_two_stage_arrays(group_frame(), action_frame())
    assert arrays.size == 3
    assert arrays.group_features.shape == (3, 110)
    assert arrays.action_features.shape == (3, 5, 8)
    assert arrays.action_mask.sum(axis=1).tolist() == [2, 2, 2]
    assert arrays.group_target.tolist() == [1.0, 0.0, 1.0]
    assert schema["group_stage_one_hot"] == list(STAGE_ORDER)


def test_scaler_fits_only_selected_training_rows() -> None:
    arrays, _ = load_two_stage_arrays(group_frame(), action_frame())
    train = np.array([0, 1], dtype=np.int64)
    scaler = fit_scaler(arrays, train)
    transformed = apply_scaler(arrays, scaler)
    assert np.allclose(transformed.group_features[train, :-3].mean(axis=0), 0.0, atol=1.0e-5)
    assert np.array_equal(transformed.group_features[:, -3:], arrays.group_features[:, -3:])
