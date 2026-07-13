import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import optimize_model_selection
from src.data_pipeline import DataPreprocessor
from src.evaluation.compare_runs import compare_predictions
from src.evaluation.protocol import (
    DEFAULT_FOLD_MANIFEST_PATH,
    LEGACY_MANIFEST_PATH,
    assert_no_legacy_records,
    build_fold_prediction_rows,
    load_fold_manifest,
    semantic_checksum,
    outer_folds_from_manifest,
    validate_fold_manifest,
    validate_scenario_features,
    hard_label_probabilities,
    validate_probability_matrix,
)
from src import model_selection


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_manifest_is_preserved_and_marks_observed_holdout():
    manifest = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["legacy_version"] == "legacy_v1"
    assert manifest["current_79_scientific_role"] == "legacy_heldout_observed"
    assert len(manifest["current_79_record_ids"]) == 79
    assert manifest["prediction_checksum"]
    assert manifest["metric_checksum"]
    assert manifest["model_checkpoint_path"] is None


def test_shared_fold_manifest_is_valid_deterministic_and_complete():
    manifest = load_fold_manifest()
    assert manifest["outer_folds"] == 5
    assert len(manifest["development_records"]) == 316
    assert manifest["manifest_checksum"] == semantic_checksum(manifest)
    clone = copy.deepcopy(manifest)
    validate_fold_manifest(clone)
    assert clone["manifest_checksum"] == manifest["manifest_checksum"]
    validation = [row["source_record_identity"] for row in manifest["assignments"] if row["outer_role"] == "validation"]
    assert len(validation) == len(set(validation)) == 316
    labels = {row["source_record_identity"]: row["true_label"] for row in manifest["development_records"]}
    overall = np.bincount(list(labels.values()), minlength=3) / len(labels)
    for fold in range(5):
        fold_labels = [labels[row["source_record_identity"]] for row in manifest["assignments"] if row["outer_fold"] == fold and row["outer_role"] == "validation"]
        assert len(fold_labels) in {63, 64}
        assert np.max(np.abs(np.bincount(fold_labels, minlength=3) / len(fold_labels) - overall)) < 0.03


def test_v2_selection_uses_shared_manifest_not_legacy_split_reconstruction():
    source = inspect.getsource(optimize_model_selection)
    assert "outer_folds_from_manifest" in source
    assert "development_frame_from_manifest" in source
    assert "reconstruct_splits_from_run" not in source
    assert "assert_no_legacy_records" in source


def test_legacy_records_are_rejected_from_model_selection():
    legacy = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="legacy_heldout_observed"):
        assert_no_legacy_records([legacy["current_79_record_ids"][0]])


def test_fold_manifest_rejects_attempt_to_tune_with_observed_legacy_row():
    manifest = load_fold_manifest()
    legacy = json.loads(LEGACY_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_rows = [row["source_row_number"] for row in manifest["development_records"]]
    legacy_row = int(legacy["current_79_record_ids"][0].rsplit(":", 1)[1])
    frame = pd.DataFrame({"__source_row_number": source_rows + [legacy_row]})
    with pytest.raises(ValueError, match="exactly match"):
        outer_folds_from_manifest(frame, manifest)


def test_outer_validation_is_not_used_for_training_or_early_stopping():
    source = inspect.getsource(model_selection.fit_fold_predict_proba)
    assert "train_model(model, train_loader, early_stop_loader" in source
    assert "train_model(model, train_loader, validation_loader" not in source
    assert "train_fixed_epochs" in source
    assert "refit_engineered = apply_feature_engineering(train_fold.copy()" in source
    assert "validation_fold=outer_val" in inspect.getsource(optimize_model_selection.evaluate_deep_outer_fold)


def test_preprocessor_fits_scaler_on_train_and_drops_g3_derived_features():
    train = pd.DataFrame({"G1": [0.0, 10.0, 5.0], "G2": [1.0, 11.0, 6.0], "G3": [0, 1, 2], "G3_raw": [1, 10, 15]})
    validation = pd.DataFrame({"G1": [100.0], "G2": [100.0], "G3": [2], "G3_raw": [20]})
    preprocessor = DataPreprocessor("G3")
    transformed_train = preprocessor.fit_transform(train, apply_oversampling=False)
    transformed_validation = preprocessor.transform(validation)
    assert "G3_raw" not in transformed_train.columns
    assert "G3_raw" not in transformed_validation.columns
    assert transformed_validation["G1"].iloc[0] > 1.0
    assert preprocessor.scalers["G1"].data_max_[0] == 10.0


def test_resampling_is_train_only_and_transform_never_resamples():
    train = pd.DataFrame({"G1": [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22], "G2": [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 22], "G3": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]})
    validation = pd.DataFrame({"G1": [99, 98], "G2": [99, 98], "G3": [0, 1]})
    preprocessor = DataPreprocessor("G3", oversample_method="smote", smote_ratio=1.0, resampling_k_neighbors=2, oversampling_feature_columns=["G1", "G2"])
    train_result = preprocessor.fit_transform(train, apply_oversampling=True)
    validation_result = preprocessor.transform(validation)
    assert len(train_result) > len(train)
    assert len(validation_result) == len(validation)


@pytest.mark.parametrize(
    ("scenario", "features", "allowed"),
    [
        ("late_stage", ["G1", "G2"], True),
        ("early_warning", ["G1"], True),
        ("early_warning", ["G1", "G2"], False),
        ("pre_assessment", ["G1"], False),
        ("late_stage", ["G3"], False),
        ("late_stage", ["G3_raw"], False),
        ("late_stage", ["absences"], False),
    ],
)
def test_scenario_allowlist_and_target_guards(scenario, features, allowed):
    if allowed:
        validate_scenario_features(features, scenario)
    else:
        with pytest.raises(ValueError):
            validate_scenario_features(features, scenario)


def test_feature_inventory_has_all_student_mat_features_and_unknowns_are_excluded():
    inventory = json.loads((ROOT / "config" / "feature_availability.yaml").read_text(encoding="utf-8"))
    names = {row["feature_name"] for row in inventory["features"]}
    assert len(names) == 33
    for feature in ("absences", "failures", "studytime", "schoolsup", "famsup", "paid", "activities", "higher", "internet", "G1", "G2", "G3"):
        assert feature in names
    assert "absences" not in json.loads((ROOT / "config" / "features_late_stage.yaml").read_text(encoding="utf-8"))["allowed_features"]


def test_record_level_output_requires_outer_validation_and_provenance():
    manifest = load_fold_manifest(DEFAULT_FOLD_MANIFEST_PATH)
    ids = [row["source_record_identity"] for row in manifest["assignments"] if row["outer_fold"] == 0 and row["outer_role"] == "validation"]
    rows = build_fold_prediction_rows(
        run_metadata={"model_name": "rule", "scenario": "late_stage", "feature_set_id": "x", "training_seed": 42, "hyperparameter_trial_id": "na", "config_checksum": "a", "fold_manifest_checksum": manifest["manifest_checksum"], "code_commit": "b", "dataset_checksum": manifest["dataset_checksum"]},
        manifest=manifest,
        outer_fold=0,
        record_ids=ids,
        y_true=[0] * len(ids),
        y_pred=[0] * len(ids),
    )
    assert len(rows) == len(ids)
    with pytest.raises(ValueError, match="exactly"):
        build_fold_prediction_rows(
            run_metadata={**rows[0], "fold_manifest_checksum": manifest["manifest_checksum"]}, manifest=manifest, outer_fold=0, record_ids=ids[:-1], y_true=[0] * (len(ids) - 1), y_pred=[0] * (len(ids) - 1)
        )


def test_paired_comparison_requires_same_outer_predictions_and_reports_ordinal_errors():
    a = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "outer_fold": [0, 0, 1, 1], "true_label": [0, 1, 2, 1], "predicted_label": [0, 1, 2, 0]})
    b = pd.DataFrame({"record_id": ["a", "b", "c", "d"], "outer_fold": [0, 0, 1, 1], "true_label": [0, 1, 2, 1], "predicted_label": [1, 1, 1, 1]})
    result = compare_predictions(a, b, metric="macro_f1", n_bootstrap=20)
    assert set(result) >= {"fold_wise", "paired_bootstrap_95_ci", "classwise_correctness_difference", "one_step_error_difference", "two_step_error_difference"}


def test_rule_probability_fallback_is_strict_one_hot_for_every_class_and_batch():
    probabilities = hard_label_probabilities([0, 1, 2, 2, 0])
    validate_probability_matrix(probabilities, [0, 1, 2, 2, 0])
    assert probabilities.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
    round_trip = np.asarray(json.loads(json.dumps(probabilities.tolist())), dtype=float)
    validate_probability_matrix(round_trip, [0, 1, 2, 2, 0])


def test_probability_validator_rejects_regression_vector_from_invalid_run():
    with pytest.raises(ValueError, match="sum to 1"):
        validate_probability_matrix(np.asarray([[0.999, 0.001, 0.001]]), [0])


def test_probability_validator_accepts_float32_softmax_level_numerical_error():
    probabilities = np.asarray([[0.2, 0.3, 0.50000006]], dtype=np.float64)
    validate_probability_matrix(probabilities, [2])
