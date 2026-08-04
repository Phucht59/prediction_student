from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY = ROOT / "configs" / "recommend_hybrid" / "final_hybrid_only_authority.yaml"
RUNTIME_FILES = [
    ROOT / "src" / "recommend_hybrid" / "pipeline.py",
    ROOT / "src" / "recommend_hybrid" / "counterfactual" / "pipeline.py",
    ROOT / "src" / "recommend_hybrid" / "counterfactual" / "ranker.py",
]


def test_final_runtime_allows_only_hybrid_predictor():
    payload = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    assert payload["status"] == "FROZEN"
    assert payload["learned_model_authority"]["allowed"] == ["residual_cnn_bilstm"]
    assert payload["runtime_pipeline"]["predictor"] == "frozen_residual_cnn_bilstm"
    assert payload["runtime_pipeline"]["ranking_method"] == "deterministic_counterfactual_utility"
    assert payload["scientific_boundary"]["learned_ranker_required"] is False


def test_final_runtime_does_not_import_secondary_ml_rankers():
    forbidden_tokens = (
        "xgboost",
        "lightgbm",
        "XGBRanker",
        "XGBRegressor",
        "LGBMRanker",
        "outcome_grounded",
    )
    for path in RUNTIME_FILES:
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source, f"{path} imports or references forbidden runtime token {token}"


def test_outcome_grounded_v2_1_is_research_only():
    payload = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    experiment = payload["historical_experiments"]["outcome_grounded_v2_1_xgboost"]
    assert experiment["runtime_authority"] is False
    assert experiment["status"] == "RESEARCH_EXPERIMENT_NOT_FINAL_ARCHITECTURE"
