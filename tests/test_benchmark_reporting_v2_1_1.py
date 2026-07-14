import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.report_benchmark_v2_1_1 import _evidence
from src.evaluation.metrics import classification_metrics, top_label_ece
from src.evaluation.protocol import file_checksum, load_fold_manifest
from src.evaluation.reporting_v2_1_1 import (
    JOB_COLUMNS, REQUIRED_PAIRS, build_expected_job_contract, checksum_validation,
    compare_expected_jobs, feature_contracts, fold_metric_estimator,
    paired_comparisons, recompute_metrics, render_paired_markdown,
    validate_record_coverage,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/benchmark_v2/benchmark-v2-full-20260713c"


@pytest.fixture(scope="module")
def artifact():
    return (pd.read_csv(RUN / "predictions/outer_validation_predictions.csv"),
            pd.read_csv(RUN / "fold_metrics.csv"), load_fold_manifest())


def test_confidence_one_is_in_terminal_bin():
    assert top_label_ece([1, 1], np.eye(3)[[0, 1]], n_bins=10) == pytest.approx(.5)


def test_one_hot_ece_equals_error_rate():
    assert top_label_ece([0, 1, 2, 2], np.eye(3)[[0, 0, 2, 1]]) == pytest.approx(.5)


def test_perfect_one_hot_ece_zero():
    assert top_label_ece([0, 1, 2], np.eye(3)) == 0


def test_zero_probability_value_is_handled():
    assert np.isfinite(top_label_ece([1], [[0.0, 0.5, 0.5]]))


@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_invalid_n_bins_rejected(value):
    with pytest.raises(ValueError): top_label_ece([0], [[1, 0, 0]], n_bins=value)


def test_probability_length_mismatch_rejected():
    with pytest.raises(ValueError): top_label_ece([0, 1], [[1, 0, 0]])


@pytest.mark.parametrize("target,pred", [([3], [0]), ([0], [-1])])
def test_invalid_labels_rejected(target, pred):
    with pytest.raises(ValueError): classification_metrics(target, pred, [[1, 0, 0]])


def test_missing_model_detected(artifact):
    pred, _, folds = artifact; contract = build_expected_job_contract(folds)
    frame = pred[pred.model_name != "g2_rule"]
    _, summary = compare_expected_jobs(contract, frame)
    assert summary["missing"] == 5


def test_missing_single_job_detected(artifact):
    pred, _, folds = artifact; contract = build_expected_job_contract(folds)
    mask = ~((pred.model_name == "g2_rule") & (pred.outer_fold == 0))
    _, summary = compare_expected_jobs(contract, pred[mask])
    assert summary["missing"] == 1


def test_unexpected_job_detected(artifact):
    pred, _, folds = artifact; contract = build_expected_job_contract(folds)
    extra = pred.iloc[[0]].copy(); extra["model_name"] = "not_registered"
    _, summary = compare_expected_jobs(contract, pd.concat([pred, extra]))
    assert summary["unexpected"] == 1


def test_duplicate_metric_job_detected(artifact):
    _, metrics, _ = artifact; duplicate = pd.concat([metrics, metrics.iloc[[0]]])
    assert duplicate.duplicated(["scenario", "model_name", "outer_fold", "training_seed"]).sum() == 1


def test_duplicate_prediction_record_detected(artifact):
    pred, _, folds = artifact; contract = build_expected_job_contract(folds)
    _, summary = validate_record_coverage(contract, pd.concat([pred, pred.iloc[[0]]]), folds)
    assert summary["duplicate_prediction_rows"] == 1


def test_missing_record_detected(artifact):
    pred, _, folds = artifact; contract = build_expected_job_contract(folds)
    _, summary = validate_record_coverage(contract, pred.drop(pred.index[0]), folds)
    assert summary["invalid_coverage_jobs"] == 1


def test_outer_train_record_detected(artifact):
    pred, _, folds = artifact; contract = build_expected_job_contract(folds); changed = pred.copy()
    validation0 = {x["source_record_identity"] for x in folds["assignments"] if x["outer_fold"] == 0 and x["outer_role"] == "validation"}
    train_id = next(x["source_record_identity"] for x in folds["assignments"] if x["outer_fold"] == 0 and x["outer_role"] == "train")
    index = changed[(changed.model_name == "g2_rule") & (changed.outer_fold == 0)].index[0]
    assert changed.at[index, "record_id"] in validation0
    changed.at[index, "record_id"] = train_id
    coverage, _ = validate_record_coverage(contract, changed, folds)
    row = coverage[(coverage.model_name == "g2_rule") & (coverage.outer_fold == 0)].iloc[0]
    assert row.records_outside_outer_validation == 1


def test_one_byte_checksum_mutation_detected(tmp_path):
    path = tmp_path / "x"; path.write_bytes(b"abc"); expected = file_checksum(path); path.write_bytes(b"abd")
    result = checksum_validation(tmp_path, {"x": expected})
    assert not bool(result.iloc[0].valid)


def _single_job(artifact):
    pred, metrics, _ = artifact
    sample = pred[(pred.model_name == "g2_rule") & (pred.outer_fold == 0)]
    stored = metrics[(metrics.model_name == "g2_rule") & (metrics.outer_fold == 0)]
    return sample, stored


def test_wrong_scalar_metric_detected(artifact):
    pred, stored = _single_job(artifact); changed = stored.copy(); changed.loc[:, "macro_f1"] += .1
    _, _, _, bad, _ = recompute_metrics(pred, changed)
    assert "macro_f1" in set(bad.metric)


def test_wrong_confusion_matrix_detected(artifact):
    pred, stored = _single_job(artifact); changed = stored.copy(); changed.loc[:, "confusion_matrix"] = "[[0,0,0],[0,0,0],[0,0,0]]"
    _, cm, _, _, structured = recompute_metrics(pred, changed)
    assert not cm.match.all() and "confusion_matrix" in set(structured.metric)


def test_wrong_per_class_f1_detected(artifact):
    pred, stored = _single_job(artifact); changed = stored.copy(); pc = eval(changed.iloc[0].per_class_f1); pc["0"]["f1"] = 0
    changed.loc[:, "per_class_f1"] = str(pc)
    *_, structured = recompute_metrics(pred, changed)
    assert any(structured.metric.str.startswith("per_class_"))


def test_feature_sets_have_distinct_checksums(artifact):
    contracts = feature_contracts(artifact[2], 1)
    values = {(x["scenario"], x["feature_set_id"], x["scaler_contract"]): x["semantic_checksum"] for x in contracts}
    assert values[("late_stage", "G2", "none")] != values[("late_stage", "G1+G2", "none")]


def test_preprocessing_changes_semantic_checksum(artifact):
    contracts = feature_contracts(artifact[2], 1)
    same_features = [x for x in contracts if x["scenario"] == "late_stage" and x["feature_set_id"] == "G1+G2"]
    assert len({x["semantic_checksum"] for x in same_features}) > 1


def test_five_seed_estimator_is_not_seed_42(artifact):
    pred = artifact[0]; aggregate = fold_metric_estimator(pred, "early_warning", "small_mlp")[1]
    seed = pred[(pred.scenario == "early_warning") & (pred.model_name == "small_mlp") & (pred.outer_fold == 1) & (pred.training_seed == 42)]
    score = classification_metrics(seed.true_label, seed.predicted_label, seed[["probability_low","probability_medium","probability_high"]].to_numpy())["macro_f1"]
    assert aggregate != pytest.approx(score)


def test_paired_csv_markdown_share_keys(artifact):
    pairs = paired_comparisons(artifact[0]); markdown = render_paired_markdown(pairs)
    for row in pairs.itertuples(): assert f"{row.scenario} / {row.model_a} / {row.model_b}" in markdown


def test_all_required_late_stage_pairs_exist(artifact):
    pairs = paired_comparisons(artifact[0]); keys = set(zip(pairs.scenario, pairs.model_a, pairs.model_b))
    assert set(REQUIRED_PAIRS).issubset(keys)


def test_missing_evidence_is_not_passed():
    assert _evidence({}, "leakage_guards")[0] == "not_checked"


def test_regression_source_run_contract(artifact):
    pred, stored, folds = artifact; contract = build_expected_job_contract(folds)
    _, jobs = compare_expected_jobs(contract, pred); scalar, cm, pc, bad, structured = recompute_metrics(pred, stored)
    assert jobs == {"expected": 215, "actual": 215, "missing": 0, "unexpected": 0, "duplicate_jobs": 0}
    assert len(pred) == 13588 and len(scalar) == 1935 and len(bad) == 30
    assert set(bad.metric) == {"ece_top_label_equal_width_10"} and cm.match.all() and structured.empty
    assert len(pc) == 645
