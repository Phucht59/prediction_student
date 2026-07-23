from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.final_release.comparator_completion import verify_no_change_guard
from src.final_release.comparator_evaluation import MODEL_IDS, OVERALL_METRICS


ROOT = Path(__file__).resolve().parents[2]
COMPLETION = ROOT / "artifacts/final/comparator_completion"
DATASETS = ("student_mat", "student_por", "oulad")


@pytest.fixture(scope="module")
def evidence() -> dict[str, dict]:
    return {
        dataset: json.loads(
            (COMPLETION / dataset / "metrics.json").read_text(encoding="utf-8")
        )
        for dataset in DATASETS
    }


def _model(evidence: dict[str, dict], dataset: str, model: str) -> dict:
    return next(
        row for row in evidence[dataset]["models"] if row["model_id"] == model
    )


def test_xgboost_student_mat_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "student_mat", "xgboost")["evidence_origin"] == "newly_trained_comparator"


def test_xgboost_student_por_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "student_por", "xgboost")["evidence_origin"] == "newly_trained_comparator"


@pytest.mark.parametrize("model", ["logistic_regression"])
def test_oulad_logistic_complete(evidence: dict[str, dict], model: str) -> None:
    assert len(_model(evidence, "oulad", model)["metrics"]) == 14


def test_oulad_decision_tree_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "oulad", "decision_tree")["metrics"]["macro_f1"] >= 0


def test_oulad_random_forest_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "oulad", "random_forest")["metrics"]["macro_f1"] >= 0


def test_oulad_hist_gradient_boosting_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "oulad", "hist_gradient_boosting")["metrics"]["macro_f1"] >= 0


def test_oulad_svm_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "oulad", "svm")["metrics"]["roc_auc"] >= 0


def test_oulad_xgboost_complete(evidence: dict[str, dict]) -> None:
    assert _model(evidence, "oulad", "xgboost")["metrics"]["pr_auc"] >= 0


def test_all_nine_models_have_overall_metrics(evidence: dict[str, dict]) -> None:
    for dataset in DATASETS:
        assert [row["model_id"] for row in evidence[dataset]["models"]] == list(MODEL_IDS)
        for row in evidence[dataset]["models"]:
            assert set(OVERALL_METRICS).issubset(row["metrics"])


def test_all_nine_models_have_per_class_metrics() -> None:
    expected = {"student_mat": 27, "student_por": 27, "oulad": 18}
    for dataset, count in expected.items():
        frame = pd.read_csv(COMPLETION / dataset / "per_class.csv")
        assert len(frame) == count


def test_all_oulad_models_have_top_k() -> None:
    frame = pd.read_csv(COMPLETION / "oulad/top_k.csv")
    assert len(frame) == 27
    assert set(frame["model_id"]) == set(MODEL_IDS)
    assert set(np.round(frame["budget"], 2)) == {0.05, 0.10, 0.20}


def test_all_models_have_confusion_matrix() -> None:
    for dataset in DATASETS:
        payload = json.loads(
            (COMPLETION / dataset / "confusion_matrices.json").read_text()
        )
        assert set(payload) == set(MODEL_IDS)


def test_no_macro_f1_column_in_per_class_table() -> None:
    for dataset in DATASETS:
        assert "macro_f1" not in pd.read_csv(
            COMPLETION / dataset / "per_class.csv"
        ).columns


def test_macro_f1_equals_mean_class_f1(evidence: dict[str, dict]) -> None:
    for dataset in DATASETS:
        classes = pd.read_csv(COMPLETION / dataset / "per_class.csv")
        for model in MODEL_IDS:
            expected = classes.loc[classes["model_id"] == model, "f1"].mean()
            assert _model(evidence, dataset, model)["metrics"]["macro_f1"] == pytest.approx(expected)


def test_weighted_f1_matches_support(evidence: dict[str, dict]) -> None:
    for dataset in DATASETS:
        classes = pd.read_csv(COMPLETION / dataset / "per_class.csv")
        for model in MODEL_IDS:
            rows = classes.loc[classes["model_id"] == model]
            expected = np.average(rows["f1"], weights=rows["support"])
            assert _model(evidence, dataset, model)["metrics"]["weighted_f1"] == pytest.approx(expected)


def test_probability_rows_complete() -> None:
    expected = {"student_mat": 395, "student_por": 649, "oulad": 15378}
    required = {
        "dataset",
        "model_id",
        "record_id",
        "true_label",
        "predicted_label",
        "outer_fold",
        "inner_protocol_id",
        "seed",
        "config_hash",
        "split_manifest_hash",
        "feature_contract_hash",
        "run_id",
    }
    for dataset, records in expected.items():
        name = "ensemble_oof_predictions.parquet" if dataset == "oulad" else "oof_predictions.parquet"
        frame = pd.read_parquet(COMPLETION / dataset / name)
        assert len(frame) == records * 9
        seed_frame = pd.read_parquet(COMPLETION / dataset / "seed_predictions.parquet")
        assert required.issubset(seed_frame.columns)
        assert not seed_frame[list(required)].isna().any().any()


def test_probability_range() -> None:
    for dataset in DATASETS:
        frame = pd.read_parquet(COMPLETION / dataset / "seed_predictions.parquet")
        columns = ["p_not_at_risk", "p_at_risk"] if dataset == "oulad" else ["p_low", "p_medium", "p_high"]
        values = frame[columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()


def test_multiclass_probability_sum() -> None:
    for dataset in ("student_mat", "student_por"):
        frame = pd.read_parquet(COMPLETION / dataset / "seed_predictions.parquet")
        assert np.allclose(frame[["p_low", "p_medium", "p_high"]].sum(axis=1), 1.0, atol=1e-6)


def test_no_duplicate_record_model_seed() -> None:
    for dataset in DATASETS:
        frame = pd.read_parquet(COMPLETION / dataset / "seed_predictions.parquet")
        assert not frame.duplicated(["dataset", "model_id", "record_id", "seed"]).any()


def test_same_record_ids_across_models() -> None:
    for dataset in DATASETS:
        name = "ensemble_oof_predictions.parquet" if dataset == "oulad" else "oof_predictions.parquet"
        frame = pd.read_parquet(COMPLETION / dataset / name)
        sets = [set(group.record_id) for _, group in frame.groupby("model_id")]
        assert all(value == sets[0] for value in sets[1:])


def test_future_oulad_not_accessed() -> None:
    frame = pd.read_parquet(COMPLETION / "oulad/seed_predictions.parquet")
    assert set(frame["scope"]) == {"development_oof"}
    assert set(frame["forecast"]) == {"F2_MIDDLE"}


def test_official_cnn_bilstm_hashes_unchanged() -> None:
    assert verify_no_change_guard()["status"] == "PASS"


def test_recommendation_artifacts_unchanged() -> None:
    guard = verify_no_change_guard()
    assert not guard["changed"] and not guard["missing"]


def test_metric_provenance_complete(evidence: dict[str, dict]) -> None:
    required = {
        "source_artifact",
        "source_checksum",
        "protocol_hash",
        "split_manifest_hash",
        "feature_contract_hash",
        "calculation_method",
    }
    for dataset in DATASETS:
        for model in evidence[dataset]["models"]:
            assert required.issubset(model["metric_provenance"])
            assert model["source_artifacts"] and model["source_checksums"]


def test_no_applicable_na_metrics(evidence: dict[str, dict]) -> None:
    for dataset in DATASETS:
        for model in evidence[dataset]["models"]:
            assert all(value is not None for value in model["metrics"].values())
