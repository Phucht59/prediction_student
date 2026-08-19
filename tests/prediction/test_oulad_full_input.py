"""Raw OULAD tables → FIT-only preprocess → Hybrid forward. No training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.prediction import Hybrid, HybridConfig
from src.prediction.data.oulad_features import (
    OULAD_AGGREGATE_CHANNELS,
    OULAD_TEMPORAL_CHANNELS,
    build_oulad_information_state,
    fit_oulad_preprocessor,
)


def _write_mini_oulad(root: Path) -> None:
    pd.DataFrame(
        [{"code_module": "AAA", "code_presentation": "2013J", "module_presentation_length": 270}]
    ).to_csv(root / "courses.csv", index=False)
    pd.DataFrame(
        [
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 1,
                "gender": "M",
                "region": "East",
                "highest_education": "A Level",
                "imd_band": "50-60%",
                "age_band": "0-35",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": "Pass",
            },
            {
                "code_module": "AAA",
                "code_presentation": "2013J",
                "id_student": 2,
                "gender": "F",
                "region": "West",
                "highest_education": "HE Qualification",
                "imd_band": "20-30%",
                "age_band": "35-55",
                "num_of_prev_attempts": 1,
                "studied_credits": 120,
                "disability": "Y",
                "final_result": "Fail",
            },
        ]
    ).to_csv(root / "studentInfo.csv", index=False)
    pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "date_registration": 0, "date_unregistration": np.nan},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "date_registration": 0, "date_unregistration": np.nan},
        ]
    ).to_csv(root / "studentRegistration.csv", index=False)
    pd.DataFrame(
        [{"id_site": 10, "code_module": "AAA", "code_presentation": "2013J", "activity_type": "oucontent"}]
    ).to_csv(root / "vle.csv", index=False)
    pd.DataFrame(
        [
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 10, "date": 5, "sum_click": 3},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 1, "id_site": 10, "date": 40, "sum_click": 4},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "id_site": 10, "date": 8, "sum_click": 2},
            {"code_module": "AAA", "code_presentation": "2013J", "id_student": 2, "id_site": 10, "date": 80, "sum_click": 9},
        ]
    ).to_csv(root / "studentVle.csv", index=False)
    pd.DataFrame(
        [{"id_assessment": 1, "code_module": "AAA", "code_presentation": "2013J", "assessment_type": "TMA", "date": 20, "weight": 10}]
    ).to_csv(root / "assessments.csv", index=False)
    pd.DataFrame(
        [
            {"id_assessment": 1, "id_student": 1, "date_submitted": 18, "is_banked": 0, "score": 70},
            {"id_assessment": 1, "id_student": 2, "date_submitted": 19, "is_banked": 0, "score": 40},
        ]
    ).to_csv(root / "studentAssessment.csv", index=False)


def test_fit_only_oulad_reaches_hybrid_forward(tmp_path: Path):
    _write_mini_oulad(tmp_path)
    from src.prediction.data.oulad import load_oulad_static_tables

    _, _, base = load_oulad_static_tables(tmp_path)
    fit_ids = base.record_id.astype(str).tolist()
    prep = fit_oulad_preprocessor(tmp_path, fit_ids, states=("20pct", "50pct"))
    assert prep.static_dim > 0
    built = build_oulad_information_state(tmp_path, "20pct", preprocessor=prep)
    assert built.static.shape[1] == prep.static_dim
    assert built.temporal.shape[2] == len(OULAD_TEMPORAL_CHANNELS)
    assert built.aggregate.shape[1] == len(OULAD_AGGREGATE_CHANNELS)
    assert np.isfinite(built.static).all()
    assert np.isfinite(built.temporal).all()
    assert np.isfinite(built.aggregate).all()
    assert list(built.record_id) == list(base.loc[base.record_id.astype(str).isin(built.record_id.astype(str)), "record_id"].astype(str))
    assert set(built.group_id.astype(str)) == set(base.group_id.astype(str))
    cfg = HybridConfig(
        static_dim=int(built.static.shape[1]),
        temporal_dim=int(built.temporal.shape[2]),
        aggregate_dim=int(built.aggregate.shape[1]),
    )
    model = Hybrid(cfg)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.tensor(built.static),
            torch.tensor(built.temporal),
            torch.tensor(built.temporal_mask),
            torch.tensor(built.lengths),
            torch.tensor(built.aggregate),
            torch.tensor(built.aggregate_available.astype(np.int64)),
            torch.tensor(built.progress),
        )
    assert logits.shape == (len(built.record_id),)
    assert torch.isfinite(logits).all()
