"""Thesis-final Hybrid C0 contracts. No new training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.prediction import Hybrid, HybridConfig
from src.prediction.baselines import ACTIVE_BASELINES, build_baseline
from src.prediction.contracts import OULAD_STATES, UCI_STAGES, canonical_oulad_state
from src.prediction.data.uci import build_uci_stage_view
from src.prediction.registry import ACTIVE_PREDICTION_REGISTRY


ROOT = Path(__file__).resolve().parents[2]


def _availability_cases(model: Hybrid) -> list[dict]:
    model.eval()
    cfg = model.config
    rows = [
        {"temporal": 0, "aggregate": 0},
        {"temporal": 0, "aggregate": 1},
        {"temporal": 1, "aggregate": 0},
        {"temporal": 1, "aggregate": 1},
    ]
    batch = len(rows)
    timesteps = 6
    static = torch.randn(batch, cfg.static_dim)
    temporal = torch.randn(batch, timesteps, cfg.temporal_dim)
    mask = torch.zeros(batch, timesteps, dtype=torch.bool)
    aggregate = torch.randn(batch, cfg.aggregate_dim)
    agg_avail = torch.zeros(batch, dtype=torch.bool)
    progress = torch.tensor([0.0, 0.35, 0.5, 1.0])
    for i, row in enumerate(rows):
        if row["temporal"]:
            mask[i, :3] = True
        else:
            temporal[i] = 0
        agg_avail[i] = bool(row["aggregate"])
        if not row["aggregate"]:
            aggregate[i] = 0
    temporal = temporal * mask.unsqueeze(-1)
    lengths = mask.sum(1)
    with torch.no_grad():
        logits = model(static, temporal, mask, lengths, aggregate, agg_avail, progress)
    assert torch.isfinite(logits).all()
    w = model.last_diagnostics["gate_weights"]
    out = []
    for i, row in enumerate(rows):
        out.append(
            {
                **row,
                "tabular_mass": float(w[i, 0]),
                "cnn_mass": float(w[i, 1]),
                "bilstm_mass": float(w[i, 2]),
            }
        )
    return out


def test_public_identity():
    assert Hybrid.model_id == "hybrid"
    assert Hybrid.display_name == "Hybrid"
    assert Hybrid.architecture_id == "C0"
    assert ACTIVE_PREDICTION_REGISTRY["prediction_model"]["model_id"] == "hybrid"
    assert ACTIVE_PREDICTION_REGISTRY["fitted_instances"] == ["uci", "oulad"]
    assert ACTIVE_PREDICTION_REGISTRY["uci_states"] == list(UCI_STAGES)
    assert ACTIVE_PREDICTION_REGISTRY["oulad_states"] == list(OULAD_STATES)
    assert ACTIVE_PREDICTION_REGISTRY["separate_oulad_100_model"] is False
    assert ACTIVE_PREDICTION_REGISTRY["xgboost_active"] is False
    assert ACTIVE_PREDICTION_REGISTRY["outer_test_used_for_phase4_finalization"] is False


def test_availability_mapping():
    model = Hybrid(HybridConfig(static_dim=4, temporal_dim=3, aggregate_dim=5))
    cases = _availability_cases(model)
    for case in cases:
        if case["temporal"] == 0:
            assert case["cnn_mass"] < 1e-6
            assert case["bilstm_mass"] < 1e-6
            assert case["tabular_mass"] > 0.99
        else:
            assert case["cnn_mass"] + case["bilstm_mass"] > 0
    no_agg = next(c for c in cases if c["temporal"] == 1 and c["aggregate"] == 0)
    assert no_agg["bilstm_mass"] > 0 or no_agg["cnn_mass"] > 0


def test_uci_s0_has_no_grades():
    frame = pd.DataFrame(
        {
            "G1": [9.0, 12.0],
            "G2": [10.0, 13.0],
            "G3": [8.0, 14.0],
            "target": [1, 0],
            "record_id": ["r1", "r2"],
            "global_student_group": ["g1", "g2"],
        }
    )
    s0 = build_uci_stage_view(frame, "S0")
    s1 = build_uci_stage_view(frame, "S1")
    s2 = build_uci_stage_view(frame, "S2")
    assert not s0.temporal_mask.any()
    assert s0.aggregate_available.sum() == 0
    assert s1.temporal_mask[:, 0].all() and not s1.temporal_mask[:, 1].any()
    assert s2.temporal_mask.all()


def test_oulad_states_are_aliases_not_models():
    assert canonical_oulad_state("FINAL-100") == "100pct"
    assert canonical_oulad_state("100pct") == "100pct"
    assert OULAD_STATES == ("20pct", "35pct", "50pct", "75pct", "100pct")


def test_svm_active_and_xgb_absent():
    assert ACTIVE_BASELINES == ("Logistic Regression", "Decision Tree", "Random Forest", "SVM", "MLP")
    uci_svm = build_baseline("SVM", dataset="uci")
    oulad_svm = build_baseline("SVM", dataset="oulad")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = np.array([0] * 20 + [1] * 20)
    uci_svm.fit(x, y)
    oulad_svm.fit(x, y)
    assert uci_svm.predict_proba(x).shape == (40, 2)
    assert oulad_svm.predict_proba(x).shape == (40, 2)
    try:
        build_baseline("XGBoost")
    except ValueError as exc:
        assert "XGBoost" in str(exc)
    else:
        raise AssertionError("XGBoost must not be constructible")


def test_final_artifacts_exist_and_have_no_xgb_rows():
    final = ROOT / "artifacts" / "prediction" / "final"
    required = [
        "FINALIZATION_DECISION.json",
        "ONE_MODEL_CONTRACT.json",
        "LEAKAGE_AUDIT.json",
        "OVERFIT_AUDIT.json",
        "XGBOOST_REMOVAL_MANIFEST.json",
        "SVM_CONFIG.json",
        "TRAINING_CONFIG.json",
        "BASELINE_CONFIGS.json",
    ]
    for name in required:
        assert (final / name).is_file(), name
    uci = pd.read_csv(ROOT / "reports" / "prediction" / "final" / "uci_final.csv")
    oulad = pd.read_csv(ROOT / "reports" / "prediction" / "final" / "oulad_final.csv")
    assert set(uci.model) == {"Hybrid", "LR", "DT", "RF", "SVM", "MLP"}
    assert set(oulad.model) == {"Hybrid", "LR", "DT", "RF", "SVM", "MLP"}
    assert "XGB" not in set(uci.model) and "XGBoost" not in set(uci.model)
    decision = json.loads((final / "FINALIZATION_DECISION.json").read_text(encoding="utf-8"))
    assert decision["previous_phase4_gate_status"] == "NOT_READY_FOR_FINAL_EVAL"
    assert decision["outer_test_used"] is False
    assert decision["final_authority_selected"] is True
