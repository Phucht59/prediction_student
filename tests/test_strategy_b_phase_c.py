"""Mandatory pre-run gates for Strategy B Phase C."""

from __future__ import annotations

import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from scripts import derive_strategy_b_phase_c_reporting, run_strategy_b_phase_c
from src.config import DATASETS
from src.estimator_factory import (
    REQUIRED_RESOLVED_CONFIG_KEYS,
    ResolvedConfigError,
    StudentEstimatorFactory,
    resolve_phase_c_neural_config,
    validate_resolved_config,
)
from src.models import OrderedCutpointHead, count_trainable_parameters, create_phase_c_model
from src.strategy_b_phase_ab import sha256_file
from src.strategy_b_phase_c import (
    ALL_CANDIDATES,
    MAIN_NEURAL,
    PARAMETER_GUARDRAIL,
    candidate_registry,
    detailed_metrics,
    paired_deltas,
    probability_contract,
    search_spaces,
    selection_rule,
)


def _config(candidate_id: str, **updates) -> dict:
    values = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 16,
        "oversample_method": "none",
        "class_weight_mode": "none",
        "loss": "cross_entropy",
        "smote_ratio": 1.0,
        "resampling_k_neighbors": 5,
        "cnn_channels": 8,
        "cnn_kernel_size": 1,
        "lstm_hidden_dim": 8,
        "hidden_dim": 8,
        "num_layers": 1,
        "normalization": "none",
        "dropout": 0.1,
        "sequence_dropout": 0.1,
        "max_epochs": 40,
        "patience": 10,
    }
    values.update(updates)
    config = resolve_phase_c_neural_config(candidate_id, values, suggested_parameters={})
    config["parameter_count"] = count_trainable_parameters(create_phase_c_model(config))
    return config


def test_candidate_registry_and_selection_eligibility_are_exact():
    registry = candidate_registry()
    assert [row["id"] for row in registry["candidates"]] == ALL_CANDIDATES
    assert selection_rule()["eligible_overall"] == ["R0", "M1", "M2", "N0", "N1", "N2", "N3"]
    assert selection_rule()["eligible_thesis_hybrid"] == ["N0", "N1"]
    assert registry["optional_sanity_models"]["S0"].startswith("not_activated")


@pytest.mark.parametrize("candidate_id", ["N0", "N1", "N2", "N3"])
def test_main_candidates_have_no_batchnorm_and_drop_no_records(candidate_id):
    config = _config(candidate_id, normalization="layer_norm")
    model = create_phase_c_model(config)
    assert not any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())
    assert config["drop_last_train"] is False
    assert config["scheduler"]["type"] == "fixed_lr"
    assert config["swa"]["enabled"] is False


def test_parameter_count_guardrail_covers_maximum_registered_neural_space():
    configs = [
        _config("N0", cnn_channels=16, cnn_kernel_size=2, lstm_hidden_dim=16),
        _config("N1", cnn_channels=16, cnn_kernel_size=2, lstm_hidden_dim=16),
        _config("N2", hidden_dim=32, num_layers=2),
        _config("N3", hidden_dim=32, num_layers=2),
    ]
    assert all(config["parameter_count"] <= PARAMETER_GUARDRAIL for config in configs)
    assert search_spaces()["N0_N1"]["parameter_guardrail"] == 5000


@pytest.mark.parametrize("kernel,expected_length", [(1, 2), (2, 1)])
def test_cnn_output_shape_for_length_two_and_no_semantic_padding(kernel, expected_length):
    model = create_phase_c_model(_config("N0", cnn_kernel_size=kernel))
    logits = model(torch.randn(7, 2, 1))
    assert logits.shape == (7, 3)
    assert model.cnn_output_sequence_length == expected_length
    assert model.sequence_cnn.padding == (0,)


def test_layer_norm_is_bound_to_channel_dimension():
    model = create_phase_c_model(_config("N0", normalization="layer_norm", cnn_channels=8))
    assert isinstance(model.sequence_norm, nn.LayerNorm)
    assert tuple(model.sequence_norm.normalized_shape) == (8,)
    assert model(torch.randn(5, 2, 1)).shape == (5, 3)


@pytest.mark.parametrize("candidate_id", ["N1", "N3"])
def test_ordinal_threshold_probability_and_decoding_contract(candidate_id):
    model = create_phase_c_model(_config(candidate_id))
    head = model.head
    assert isinstance(head, OrderedCutpointHead)
    thresholds = head.thresholds().detach()
    assert thresholds[1] > thresholds[0]
    logits = model(torch.randn(32, 2, 1))
    cumulative = torch.sigmoid(logits)
    assert torch.all(cumulative[:, 1] <= cumulative[:, 0])
    probabilities = model.predict_proba(torch.randn(32, 2, 1)).detach().numpy()
    probability_contract(probabilities)
    decoded = probabilities.argmax(axis=1)
    assert set(decoded).issubset({0, 1, 2})
    assert "clamp" not in inspect.getsource(OrderedCutpointHead.probabilities_from_logits)


def test_probability_contract_rejects_nonfinite_range_and_sum_violations():
    with pytest.raises(ValueError, match="non-finite"):
        probability_contract(np.asarray([[np.nan, 0.5, 0.5]]))
    with pytest.raises(ValueError, match="range"):
        probability_contract(np.asarray([[-0.1, 0.5, 0.6]]))
    with pytest.raises(ValueError, match="sum"):
        probability_contract(np.asarray([[0.2, 0.2, 0.2]]))


def test_no_target_leakage_and_same_raw_feature_contract_across_neural_models():
    configs = [_config(candidate) for candidate in MAIN_NEURAL]
    assert all(config["feature_contract"]["sequence_columns"] == ["G1", "G2"] for config in configs)
    assert all(config["preprocessing"]["deterministic_transforms"] == "none" for config in configs)
    forbidden = {"G3", "G3_raw", "record_id", "dataset_version_id", "fold_id"}
    assert forbidden.isdisjoint(configs[0]["feature_contract"]["sequence_columns"])


def test_resolved_config_is_complete_and_missing_keys_fail_fast():
    config = _config("N0")
    assert REQUIRED_RESOLVED_CONFIG_KEYS <= set(config)
    assert {"candidate_id", "head_type", "normalization", "parameter_guardrail"} <= set(config)
    invalid = deepcopy(config)
    del invalid["scheduler"]
    with pytest.raises(ResolvedConfigError, match="missing required keys"):
        validate_resolved_config(invalid)


def test_inner_outer_refit_use_same_factory_and_contract():
    config = _config("N1")
    signatures = [StudentEstimatorFactory(DATASETS["student-mat"], deepcopy(config)).estimator_signature() for _ in range(3)]
    assert signatures[0] == signatures[1] == signatures[2]
    assert signatures[0]["criterion"]["loss"] == "ordered_binary_cross_entropy"
    source = inspect.getsource(run_strategy_b_phase_c)
    assert "fit_fold_predict_proba" in source
    assert "predict_with_fitted_estimator" in source


def test_ml_and_dl_oof_alignment_metrics_recomputation_and_pairs():
    rows = []
    for candidate in ["M1", "N0"]:
        for source_row, true in enumerate([0, 1, 2, 0, 1, 2]):
            prediction = true if candidate == "N0" or source_row != 5 else 1
            probs = np.eye(3)[prediction]
            rows.append({
                "candidate_id": candidate, "seed": 42, "outer_fold": source_row % 2,
                "source_row_number": source_row, "raw_g3": [8, 10, 15, 7, 14, 16][source_row],
                "true_label": true, "predicted_label": prediction,
                "prob_0": probs[0], "prob_1": probs[1], "prob_2": probs[2],
            })
    oof = pd.DataFrame(rows)
    assert set(oof[oof["candidate_id"] == "M1"]["source_row_number"]) == set(oof[oof["candidate_id"] == "N0"]["source_row_number"])
    first = detailed_metrics(oof)[0]
    second = detailed_metrics(oof.copy())[0]
    pd.testing.assert_frame_equal(first, second)
    paired = paired_deltas(oof, [("N0", "M1")], bootstrap_samples=20)
    assert paired.iloc[0]["macro_f1_delta_left_minus_right"] > 0


def test_checkpoint_state_reproduces_exact_neural_parameters(tmp_path):
    model = create_phase_c_model(_config("N0"))
    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored = create_phase_c_model(_config("N0"))
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())


def test_phase_c_runner_never_fetches_legacy_observed_and_is_atomic():
    source = inspect.getsource(run_strategy_b_phase_c.main)
    module_source = inspect.getsource(run_strategy_b_phase_c)
    assert "load_development_subset_from_postgres" in module_source
    assert "load_dataset_from_postgres" not in module_source
    assert "legacy_observed_79_fetched" in source
    assert "os.replace(artifact_tmp, artifact_final)" in source
    assert "artifact_tmp, \"failed\"" in source
    assert "failure_reason=str(exc)" in source
    recovery_source = inspect.getsource(run_strategy_b_phase_c._resume_finalize)
    assert 'len(jobs) == 2805' in recovery_source
    assert 'len(trials) == 900' in recovery_source
    assert 'len(oof) == 9 * 3 * 316' in recovery_source
    assert '"training_or_predictions_changed": False' in recovery_source


def test_conclusion_renderer_has_no_optional_tabulate_dependency():
    source = inspect.getsource(run_strategy_b_phase_c._conclusion)
    assert "to_markdown" not in source
    assert "tabulate" not in source


def test_reporting_correction_cannot_change_predictions_metrics_or_selections():
    source = inspect.getsource(derive_strategy_b_phase_c_reporting)
    assert '"predictions_metrics_selections_changed": False' in source
    assert "source_checksum_failures" in source
    assert "len(jobs) == 2805" in source
    assert "len(trials) == 900" in source
    assert "len(oof) == 9 * 3 * 316" in source
    assert "fit_fold_predict_proba" not in source


def test_artifact_checksum_detects_mutation(tmp_path):
    path = tmp_path / "artifact.txt"
    path.write_text("frozen", encoding="utf-8")
    expected = sha256_file(path)
    path.write_text("mutated", encoding="utf-8")
    assert sha256_file(path) != expected


def test_postgres_integration_skip_is_explicit_waiver_not_fake_pass(monkeypatch):
    monkeypatch.delenv("POSTGRES_TEST_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_TEST_APP_DSN", raising=False)
    report = run_strategy_b_phase_c._run_tests(skip=True, stage="smoke")
    assert report["status"] == "SKIPPED_BY_DIAGNOSTIC_FLAG"
    assert report["official"] is False
