"""One-weight baseline fairness and rec-prediction contract."""
from __future__ import annotations

import pytest

from experiments.hybrid_superiority_v2.paths import CACHE_DIR
from src.prediction.contracts import PredictionResult
from src.recommend_hybrid.prediction_adapter import prediction_result_to_features
from src.recommend_hybrid.v3.contracts import map_prediction_state
from src.recommend_hybrid.v3.prediction_adapter import prediction_result_to_v3_fields


pytestmark_cache = pytest.mark.skipif(not (CACHE_DIR / "uci" / "manifest.json").exists(), reason="uci cache missing")


def test_prediction_result_is_rec_input_uci_and_oulad():
    for dataset, stage in (("uci_combined", "S1"), ("uci_combined", "S2"), ("oulad", "20pct"), ("oulad", "75pct")):
        result = PredictionResult(
            dataset=dataset,
            record_id="r1",
            stage_or_endpoint=stage,
            risk_probability=0.61,
            predicted_risk=1,
            threshold=0.4,
            model_id="hybrid",
            metadata={"student_key": "s", "course_key": "AAA::2013J", "cutoff_day": 30},
        )
        feats = prediction_result_to_features(result)
        assert feats["risk_probability"] == 0.61
        assert feats["model_id"] == "hybrid"
        assert feats["predicted_risk"] == 1
        if dataset == "oulad":
            v3 = prediction_result_to_v3_fields(result)
            assert v3["risk_probability"] == 0.61
            assert "seed_disagreement" not in v3
            assert map_prediction_state(stage) is v3["stage"]


def test_rec_rejects_100pct_intervention():
    with pytest.raises(ValueError):
        map_prediction_state("100pct")


def test_research_c0r_must_wrap_as_hybrid_prediction_result():
    with pytest.raises(ValueError, match="model_id"):
        PredictionResult(
            dataset="uci_combined",
            record_id="r",
            stage_or_endpoint="S2",
            risk_probability=0.5,
            predicted_risk=0,
            threshold=0.4,
            model_id="hybrid_superiority_v2",
        )


@pytestmark_cache
def test_stacked_uci_one_estimator_all_stages():
    from experiments.hybrid_superiority_v2.baselines import fit_eval_stacked, predictor_columns
    from experiments.hybrid_superiority_v2.data import inner_partitions, scale_views, stacked_baseline_frame

    fit, stop, valid = inner_partitions("uci", 0)
    prepared = scale_views("uci", fit)
    frame = stacked_baseline_frame(prepared)
    assert set(frame.stage.unique()) == {"S0", "S1", "S2"}
    s0 = frame.stage.eq("S0")
    assert "grade_g1" in frame.columns
    assert frame.loc[s0, "grade_g1"].isna().all()
    assert frame.loc[s0, "grade_g2"].isna().all()
    s1 = frame.stage.eq("S1")
    assert frame.loc[s1, "grade_g1"].notna().all()
    assert frame.loc[s1, "grade_g2"].isna().all()
    s2 = frame.stage.eq("S2")
    assert frame.loc[s2, "grade_g1"].notna().all()
    assert frame.loc[s2, "grade_g2"].notna().all()
    cols, cats = predictor_columns(frame)
    assert "stage" not in cols
    assert "G1" not in cols and "G2" not in cols
    out = fit_eval_stacked("LR", frame, cols, cats, fit, stop, valid, 42)
    assert out["n_models"] == 1
    assert out["one_weight_all_stages"] is True
    assert set(out["stages"]) == {"S0", "S1", "S2"}
    assert out["estimator_id"]  # same object scored every stage


def test_cli_lock_and_fair_dry_run():
    from experiments.hybrid_superiority_v2.cli import main

    assert main(["--dry-run", "lock-and-fair"]) == 0
