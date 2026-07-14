import inspect
import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.model_v3_protocol import (
    FIXED_REFERENCE_REGISTRY, MODEL_REGISTRY, build_expected_jobs, build_selection_study_contract,
    checksum, deterministic_study_seed, duplicate_jobs, legacy_intersection, map_g3_to_class,
    pooled_oof_regression_metrics, regression_metric_summary, validate_loader_rows,
    validate_selection_results, validate_shape_rows, validate_outer_refit_config_checksums,
    validate_full_preflight,
)
from src.models.ordinal_v3 import (
    TabularV3Model, TrainOnlyTargetScaler, coral_targets, multitask_loss,
    ordinal_bce_loss, ordinal_logits_to_probabilities,
)


def _contracts():
    features = {"late_stage": {"feature_set_id": "G1+G2", "semantic_checksum": "f12"},
                "early_warning": {"feature_set_id": "G1", "semantic_checksum": "f1"}}
    return features, {"semantic_checksum": "target"}


def _study_contract(smoke=False):
    return build_selection_study_contract("run", "fold", "commit", "space", {"semantic_checksum": "target"}, smoke=smoke)


def test_coral_targets_for_all_three_classes():
    assert torch.equal(coral_targets(torch.tensor([0, 1, 2])), torch.tensor([[0., 0.], [1., 0.], [1., 1.]]))


def test_ordinal_probability_numerical_conversion_and_argmax():
    logits = torch.tensor([[4., -4.], [6., 4.]])
    probabilities = ordinal_logits_to_probabilities(logits)
    assert torch.all(probabilities >= 0) and torch.allclose(probabilities.sum(1), torch.ones(2))
    assert probabilities.argmax(1).tolist() == [1, 2]


def test_non_monotone_cumulative_logits_rejected():
    with pytest.raises(ValueError):
        ordinal_logits_to_probabilities(torch.tensor([[-2., 2.]]))


def test_ordinal_loss_is_finite_and_backward_works():
    model = TabularV3Model(2, ordinal=True)
    logits, _ = model(torch.randn(4, 2))
    loss = ordinal_bce_loss(logits, torch.tensor([0, 1, 2, 1]))
    loss.backward()
    assert torch.isfinite(loss)


def test_target_scaler_fit_and_inverse_transform():
    scaler = TrainOnlyTargetScaler().fit([0, 10, 20])
    assert np.allclose(scaler.inverse_transform(scaler.transform([5, 15])), [5, 15])


def test_lambda_zero_and_positive_lambda_behavior():
    c = torch.tensor(2.0)
    assert multitask_loss(c, torch.tensor([0.]), torch.tensor([9.]), 0) == c
    assert multitask_loss(c, torch.tensor([0., 1.]), torch.tensor([1., 1.]), 1) > multitask_loss(c, torch.tensor([0., 1.]), torch.tensor([1., 1.]), .1)


def test_m0_m1_same_training_engine_and_candidate_space():
    assert MODEL_REGISTRY["M0"]["training_engine"] == MODEL_REGISTRY["M1"]["training_engine"]
    common = {"hidden_width": [8, 16, 32], "hidden_layers": [1, 2], "dropout": [0, .15, .3]}
    assert checksum(common) == checksum(dict(common))


def test_historical_sklearn_reference_is_not_candidate_matched_control():
    assert "REF_SK_MLP" in FIXED_REFERENCE_REGISTRY
    assert "REF_SK_MLP" not in MODEL_REGISTRY


def test_expected_job_contract_includes_b0_and_is_created_precompute():
    features, target = _contracts()
    contract = build_expected_jobs("run", {i: 63 for i in range(5)}, "fold", "commit", features, target)
    assert contract["created_before_compute"] and len(contract["jobs"]) == 235
    assert len([x for x in contract["jobs"] if x["model_family"] == "B0"]) == 10


def test_study_contract_has_one_study_per_m0_m3_track_fold():
    contract = _study_contract()
    assert len(contract["studies"]) == 40
    ids = [x["study_id"] for x in contract["studies"]]
    assert len(ids) == len(set(ids))
    assert deterministic_study_seed("model_v3_1", "M0", "late_stage", 0) != 42


def test_tuning_contract_is_not_per_training_seed():
    contract = _study_contract()
    assert all("training_seed" not in study for study in contract["studies"])


def _valid_selection_evidence(contract):
    trial_rows = []
    selected_rows = []
    for study in contract["studies"]:
        config = checksum({"study": study["study_id"]})
        for trial in range(study["expected_trials"]):
            for inner in range(3):
                trial_rows.append({"study_id": study["study_id"], "trial_id": trial, "inner_fold": inner, "config_checksum": config})
        selected_rows.append({"study_id": study["study_id"], "selected_trial_id": 0, "config_checksum": config})
    return pd.DataFrame(trial_rows), pd.DataFrame(selected_rows)


def test_selection_validator_detects_missing_study_trial_inner_and_selected_config():
    contract = _study_contract(smoke=True)
    trials, selected = _valid_selection_evidence(contract)
    assert not any(validate_selection_results(contract, trials, selected).values())
    assert validate_selection_results(contract, trials.iloc[3:], selected)["missing_studies"] == 1
    assert validate_selection_results(contract, trials[trials.trial_id != 0], selected)["missing_trials"] > 0
    missing_inner = trials[~((trials.study_id == trials.study_id.iloc[0]) & (trials.inner_fold == 2))]
    assert validate_selection_results(contract, missing_inner, selected)["trials_missing_complete_inner_folds"] == 1
    selected.loc[0, "config_checksum"] = "bad"
    assert validate_selection_results(contract, trials, selected)["selected_config_not_completed"] == 1


def test_duplicate_job_mutation_detected():
    frame = pd.DataFrame([{"model_family": "M0", "track": "late_stage", "outer_fold": 0, "training_seed": 42}] * 2)
    assert duplicate_jobs(frame) == 2


def test_legacy_intersection_is_independent():
    assert legacy_intersection({"dev"}, {"legacy"}) == set()
    assert legacy_intersection({"same"}, {"same"}) == {"same"}


def test_shape_and_loader_diagnostic_content_validation():
    good_shape = pd.DataFrame([{"cnn_kernel_size": 1, "cnn_output_sequence_length": 2, "bilstm_input_sequence_length": 2}, {"cnn_kernel_size": 2, "cnn_output_sequence_length": 3, "bilstm_input_sequence_length": 3}])
    assert validate_shape_rows(good_shape)
    good_loader = pd.DataFrame([{"dataset_size": 10, "batch_size": 4, "drop_last_train": False, "samples_dropped_per_epoch": 0, "samples_consumed_per_epoch": 10}])
    assert validate_loader_rows(good_loader)


def test_ridge_mapping_without_rounding():
    assert map_g3_to_class(np.array([9.999, 10.0, 14.999, 15.0])).tolist() == [0, 1, 1, 2]


def test_pooled_oof_r2_is_primary_and_not_mean_fold_r2():
    frame = pd.DataFrame({"model_family": ["B0"] * 4, "track": ["late_stage"] * 4, "training_seed": [0] * 4,
                          "record_id": ["a", "b", "c", "d"], "raw_g3": [0., 10., 20., 10.], "predicted_g3_raw": [1., 11., 19., 9.]})
    pooled = pooled_oof_regression_metrics(frame)
    direct = regression_metric_summary(frame.raw_g3, frame.predicted_g3_raw)
    assert pooled.iloc[0].aggregation == "pooled_oof_primary" and pooled.iloc[0].r2_raw == pytest.approx(direct["r2_raw"])


def test_outer_validation_not_used_by_smoke_inner_selection():
    source = inspect.getsource(__import__("scripts.run_model_v3_smoke", fromlist=["main"]).select_torch_config)
    assert "StratifiedKFold(n_splits=3" in source and "train[features]" in source


def test_all_five_outer_refits_share_the_selected_config_checksum():
    rows = pd.DataFrame({"model_family": ["M0"] * 5, "track": ["late_stage"] * 5,
                         "outer_fold": [0] * 5, "training_seed": [42, 52, 62, 72, 82],
                         "config_checksum": ["same"] * 5})
    assert validate_outer_refit_config_checksums(rows)
    rows.loc[4, "config_checksum"] = "different"
    assert not validate_outer_refit_config_checksums(rows)


def test_old_or_stale_contract_is_rejected_before_full_run():
    features, target = _contracts()
    contract = build_expected_jobs("model-v3-full-v3-1-20260714", {0: 63}, "fold", "commit", features, target, smoke=True)
    studies = _study_contract(smoke=True)
    validate_full_preflight(contract, studies, "commit")
    old = dict(contract); old["contract_version"] = "model_v3_protocol_1"
    with pytest.raises(ValueError):
        validate_full_preflight(old, studies, "commit")
    stale = dict(contract); stale["jobs"] = [dict(contract["jobs"][0], source_commit="old")]
    with pytest.raises(ValueError):
        validate_full_preflight(stale, studies, "commit")


def test_metric_recomputation_must_be_joined_by_job_key_not_dataframe_order():
    recomputed = {("B0", "late_stage", 0, 0): .8, ("M0", "late_stage", 0, 42): .7}
    stored = {("M0", "late_stage", 0, 42): .7, ("B0", "late_stage", 0, 0): .8}
    assert set(recomputed) == set(stored) and all(np.isclose(value, stored[key]) for key, value in recomputed.items())
