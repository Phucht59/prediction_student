from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import torch

from src.pipelines import oulad
from src.training.phase5_mlp_gap import _selected_configs, make_model
from src.training.phase7_endpoint import (
    ENDPOINT_ID,
    ENDPOINT_STAGE,
    FINAL_SEEDS,
    HISTORICAL_H0,
    HISTORICAL_MLP,
    PARAMETER_COUNT,
    TRIALS_PER_FOLD,
    EndpointRunner,
    _architecture_identity,
    _endpoint_metrics,
    audit_endpoint_protocol,
    audit_feature_leakage,
    early_warning_checksums,
    sample_trial_config,
    search_space,
)


def test_endpoint_is_historical_single_cutoff_not_early_warning_summary() -> None:
    bundle = oulad._build_bundle()
    audit = audit_endpoint_protocol(bundle)
    assert audit["status"] == "PASS"
    assert audit["endpoint_id"] == ENDPOINT_ID
    assert ENDPOINT_STAGE == "M1_MIDDLE_FROZEN"
    assert audit["historical_forecast_id"] == "F2_MIDDLE"
    assert audit["cutoff_fraction"] == 0.5
    assert audit["is_stage_75_percent"] is False
    assert audit["is_mean_stage"] is False
    assert audit["eligible_records"] == 15_378
    assert audit["fold_counts"] == {"0": 5120, "1": 5109, "2": 5149}


def test_endpoint_comparators_have_exact_record_fold_target_identity() -> None:
    h0 = pd.read_parquet(HISTORICAL_H0)
    mlp = pd.read_parquet(HISTORICAL_MLP)
    assert len(h0) == len(mlp) == 15_378
    h0_key = h0.set_index("record_id")[["id_student", "outer_fold", "target"]]
    mlp_key = mlp.set_index("record_id")[
        ["id_student", "outer_fold", "true_label"]
    ].rename(columns={"true_label": "target"})
    pd.testing.assert_frame_equal(h0_key.sort_index(), mlp_key.sort_index())


def test_h1_endpoint_architecture_is_frozen() -> None:
    identity = _architecture_identity()
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]
    )
    assert identity["parameter_count"] == PARAMETER_COUNT == 160_492
    assert sum(parameter.numel() for parameter in model.parameters()) == 160_492
    assert identity["architecture_id"] == "H1_TABULAR_RESIDUAL_EXPERT"


def test_endpoint_search_is_training_only_and_bounded() -> None:
    space = search_space()
    assert TRIALS_PER_FOLD == 18
    assert space["frozen"]["architecture"] == "H1_TABULAR_RESIDUAL_EXPERT"
    assert space["frozen"]["max_epochs"] == 15
    assert space["batch_size"] == [128, 256]
    source = inspect.getsource(sample_trial_config)
    for forbidden in (
        "conv_channels",
        "kernel",
        "lstm_hidden",
        "lstm_layers",
        "fusion",
        "attention",
    ):
        assert forbidden not in source


def test_outer_label_firewall_api() -> None:
    for function in (
        EndpointRunner.__init__,
        EndpointRunner.evaluate,
    ):
        parameters = inspect.signature(function).parameters
        assert "outer_y_test" not in parameters
        assert "outer_labels" not in parameters
    from src.training import phase7_endpoint

    development_source = inspect.getsource(
        phase7_endpoint.run_development_supervisor
    )
    assert "run_final_supervisor" not in development_source
    final_source = inspect.getsource(phase7_endpoint.run_final_supervisor)
    assert "study.optimize" not in final_source
    assert "create_study" not in final_source


def test_feature_leakage_manifest_and_train_only_preprocessing() -> None:
    audit = audit_feature_leakage(oulad._build_bundle())
    assert audit["status"] == "PASS"
    assert audit["predictor_forbidden_intersection"] == []
    assert audit["final_outcome_as_predictor"] is False
    assert audit["date_unregistration_as_predictor"] is False
    assert audit["preprocessing_fit_scope"] == "inner_train_or_outer_train_only"
    assert audit["future_timesteps_zero_masked"] is True


def test_h1_future_mask_invariance_at_endpoint() -> None:
    torch.manual_seed(73)
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]
    ).eval()
    lengths = torch.tensor([8, 5])
    mask = (torch.arange(10).unsqueeze(0) < lengths.unsqueeze(1)).float()
    sequence = torch.randn(2, 10, 47) * mask.unsqueeze(-1)
    changed = sequence.clone()
    changed[mask.eq(0)] = 1e8
    aggregate = torch.randn(2, 165)
    static = torch.randn(2, 13)
    first = model(sequence, lengths, mask, aggregate, static)["binary_logit"]
    second = model(changed, lengths, mask, aggregate, static)["binary_logit"]
    torch.testing.assert_close(first, second)


def test_endpoint_metric_threshold_is_inner_oof_only() -> None:
    prediction = pd.DataFrame(
        {
            "target": [0, 0, 1, 1],
            "probability": [0.1, 0.4, 0.6, 0.9],
        }
    )
    result = _endpoint_metrics(prediction)
    assert result["research_threshold"]["outer_labels_used"] is False
    assert result["research_threshold"]["source"] == "pooled_inner_oof"
    assert np.isfinite(result["macro_f1"])


def test_seed_and_early_warning_freeze_contract() -> None:
    assert FINAL_SEEDS == (42, 1201, 2026, 3407, 7319)
    checksums = early_warning_checksums()
    assert len(checksums) == 5
    assert all(len(value) == 64 for value in checksums.values())
    phase6 = json.loads(
        (
            oulad.ROOT
            / "artifacts"
            / "final"
            / "h1_final"
            / "phase6_gate.json"
        ).read_text(encoding="utf-8")
    )
    assert phase6["status"] == "PASS"
