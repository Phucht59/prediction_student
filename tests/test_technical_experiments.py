from pathlib import Path

import pandas as pd

from src.experiments.common import (
    IMBALANCE_STRATEGIES,
    apply_student_scenario,
    compute_required_metrics,
    scenario_sequence_columns,
)


def test_student_scenarios_remove_unavailable_grade_features():
    frame = pd.DataFrame(
        {
            "G1": [8, 12],
            "G2": [9, 13],
            "G3": [0, 1],
            "absences": [2, 4],
            "studytime": [2, 3],
        }
    )
    early = apply_student_scenario(frame, "early")
    midterm = apply_student_scenario(frame, "midterm")
    late = apply_student_scenario(frame, "late")

    assert "G1" not in early.columns
    assert "G2" not in early.columns
    assert "grade_growth" not in early.columns
    assert "grade_avg" not in early.columns
    assert "G1" in midterm.columns
    assert "G2" not in midterm.columns
    assert "grade_growth" not in midterm.columns
    assert {"G1", "G2", "grade_growth", "grade_avg"}.issubset(late.columns)
    assert scenario_sequence_columns("early") == []
    assert scenario_sequence_columns("midterm") == ["G1"]
    assert scenario_sequence_columns("late") == ["G1", "G2"]


def test_required_metrics_include_low_class_and_student_regression_mapping():
    metrics = compute_required_metrics([0, 1, 2, 0], [0, 2, 2, 1], y_reg_true=[4, 12, 18, 5], y_reg_pred=[0, 1, 2, 1])
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "recall_low", "f1_low", "rmse", "r2"):
        assert key in metrics
    assert "regression_head_rmse" in metrics
    assert metrics["rmse"] < metrics["regression_head_rmse"]


def test_allowed_imbalance_strategies_exclude_adasyn():
    assert set(IMBALANCE_STRATEGIES) == {
        "none",
        "class_weight",
        "smotenc",
        "random_oversampling",
        "focal_loss",
        "smotenc_focal_loss",
    }
    assert "adasyn" not in IMBALANCE_STRATEGIES


def test_data_pipeline_no_longer_imports_adasyn_sampler():
    source = (Path(__file__).resolve().parents[1] / "src" / "data_pipeline.py").read_text(encoding="utf-8")
    assert "ADASYN," not in source
    assert "ADASYN(" not in source
