"""Thesis-final Hybrid CNN–BiLSTM contracts. No new training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.prediction import Hybrid, HybridConfig
from src.prediction.baselines import ACTIVE_BASELINES, build_baseline
from src.prediction.contracts import OULAD_STATES, UCI_STAGES, canonical_oulad_state
from src.prediction.data.oulad_features import assert_predictor_contract, events_strictly_before_cutoff, filter_events_cutoff_safe
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


def test_one_model_contract_fields():
    contract = json.loads((ROOT / "artifacts" / "prediction" / "final" / "ONE_MODEL_CONTRACT.json").read_text(encoding="utf-8"))
    assert contract["architecture"] == "C0"
    assert contract["fitted_instances"] == ["uci", "oulad"]
    assert contract["stage_specific_models"] is False
    assert contract["separate_oulad_100_model"] is False


def test_oulad_cutoff_excludes_on_or_after_cutoff():
    frame = pd.DataFrame(
        {
            "date": [10, 20, 21, 30],
            "observation_start": [0, 0, 0, 0],
            "cutoff_day": [21, 21, 21, 21],
            "label": ["before", "before", "at_cutoff", "after"],
        }
    )
    kept = filter_events_cutoff_safe(frame, time_col="date", start_col="observation_start", cutoff_col="cutoff_day")
    assert set(kept.label) == {"before"}
    assert events_strictly_before_cutoff(20, 0, 21) is True
    assert events_strictly_before_cutoff(21, 0, 21) is False
    assert events_strictly_before_cutoff(30, 0, 21) is False


def test_forbidden_outcome_fields_fail_predictor_contract():
    try:
        assert_predictor_contract(["activity", "final_result"])
    except ValueError as exc:
        assert "final_result" in str(exc)
    else:
        raise AssertionError("forbidden outcome must fail")
    try:
        assert_predictor_contract(["date_unregistration"])
    except ValueError:
        pass
    else:
        raise AssertionError("date_unregistration must not be a predictor")


def test_current_authority_docs_are_phase4():
    forbidden = (
        "cnn_bilstm_mat",
        "cnn_bilstm_por",
        "cnn_bilstm_oulad",
        "H1_TABULAR_RESIDUAL_EXPERT",
        "Low     :",
        "3-class",
        "oulad_early_model",
        "oulad_final_model",
    )
    files = [
        ROOT / "PROJECT.md",
        ROOT / "README.md",
        ROOT / "reports" / "prediction" / "final" / "FINAL_PREDICTION_MODEL_REPORT.md",
        ROOT / "configs" / "prediction" / "hybrid_final.json",
        ROOT / "configs" / "prediction" / "registry.json",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} still contains stale authority token {token!r}"
        assert "Hybrid" in text or "hybrid" in text.lower()
    assert not (ROOT / "configs" / "prediction" / "hybrid_phase8.json").exists()
    assert (ROOT / "configs" / "prediction" / "historical" / "hybrid_phase8.json").is_file()
    env = (ROOT / "environment.yml").read_text(encoding="utf-8")
    assert "xgboost" not in env.lower()
    assert "optuna" not in env.lower()
    research = (ROOT / "environment.research.yml").read_text(encoding="utf-8")
    assert "xgboost" in research.lower()
    registry = (ROOT / "reports" / "CURRENT_REPORTS.md").read_text(encoding="utf-8")
    assert "Hybrid CNN–BiLSTM" in registry
    assert "PROJECT.md" in registry
    assert "the current Hybrid CNN–BiLSTM authority" in registry
    assert "HISTORICAL / SUPERSEDED" in registry
    assert not (ROOT / "artifacts" / "prediction" / "historical" / "phase8").exists()
    assert not (ROOT / "reports" / "prediction" / "historical" / "phase8").exists()
    assert not (ROOT / "src" / "prediction" / "data" / "final100.py").exists()


def test_overfit_audit_is_stage_independent():
    audit = json.loads((ROOT / "artifacts" / "prediction" / "final" / "OVERFIT_AUDIT.json").read_text(encoding="utf-8"))
    uci = audit["uci"]["stages"]
    assert uci["S0"]["n_runs"] == 9
    assert uci["S1"]["n_runs"] == 9
    assert uci["S2"]["n_runs"] == 9
    assert uci["S0"]["generalization_gap_mean"] != uci["S1"]["generalization_gap_mean"]
    assert uci["S0"]["generalization_gap_mean"] != uci["S2"]["generalization_gap_mean"]
    assert audit["uci"]["s0_gap_gt_s1"] is True
    assert audit["uci"]["s0_gap_gt_s2"] is True
    oulad = audit["oulad"]["stages"]
    gaps = [oulad[st]["generalization_gap_mean"] for st in ("20pct", "35pct", "50pct", "75pct", "100pct")]
    assert len(set(round(g, 8) for g in gaps)) > 1
    assert "classification_rule" in audit
