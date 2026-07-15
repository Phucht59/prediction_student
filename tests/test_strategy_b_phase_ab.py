"""Mandatory correctness gates for the approved Strategy B Phase A-B scope."""

from __future__ import annotations

import inspect
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import run_pipeline, run_strategy_b_phase_ab
from src.config import DATASETS
from src.estimator_factory import (
    REQUIRED_RESOLVED_CONFIG_KEYS,
    ResolvedConfigError,
    StudentEstimatorFactory,
    resolve_student_config,
    validate_resolved_config,
)
from src import model_selection
from src.strategy_b_phase_ab import (
    APPROVED_SEEDS,
    assert_development_only_frame,
    materialize_early_stop_ledger,
    materialize_inner_fold_ledger,
    recompute_metrics_from_oof,
    validate_oof_coverage,
)


def _resolved(*, drop_last: bool = False) -> dict:
    return resolve_student_config(
        {
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
            "dropout": 0.1,
            "sequence_dropout": 0.0,
            "max_epochs": 2,
            "patience": 1,
        },
        suggested_parameters={"learning_rate": 0.001},
        scheduler_type="fixed_lr",
        swa_enabled=False,
        drop_last_train=drop_last,
    )


def test_resolved_config_retains_suggestions_and_every_required_constant():
    config = _resolved()
    assert REQUIRED_RESOLVED_CONFIG_KEYS <= set(config)
    assert config["suggested_parameters"] == {"learning_rate": 0.001}
    assert config["fixed_constants"]["scheduler"]["type"] == "fixed_lr"
    assert config["fixed_constants"]["swa"]["enabled"] is False
    assert config["feature_contract"]["sequence_columns"] == ["G1", "G2"]
    assert config["feature_contract"]["context_columns"] == []


def test_missing_required_resolved_config_key_fails_fast():
    config = _resolved()
    del config["class_weight_mode"]
    with pytest.raises(ResolvedConfigError, match="missing required keys"):
        validate_resolved_config(config)


def test_inner_outer_final_share_factory_criterion_and_resampling_contracts():
    config = _resolved()
    spec = DATASETS["student-mat"]
    signatures = [StudentEstimatorFactory(spec, deepcopy(config)).estimator_signature() for _ in range(3)]
    assert signatures[0] == signatures[1] == signatures[2]
    assert signatures[0]["factory"].endswith("StudentEstimatorFactory")
    assert signatures[0]["criterion"] == {
        "loss": "cross_entropy", "class_weight_mode": "none", "focal_gamma": None
    }
    assert signatures[0]["resampling"]["oversample_method"] == "none"
    for function in (
        model_selection.fit_fold_predict_proba,
        model_selection.fit_final_development_estimator,
    ):
        assert "fit_training_partition_estimator" in inspect.getsource(function)


def test_fold_inner_fold_and_early_stop_ledgers_are_complete():
    size = 90
    frame = pd.DataFrame({"__source_row_number": np.arange(1, size + 1), "G3": np.tile([0, 1, 2], size // 3)})
    splitter = model_selection.make_folds(frame, "G3", n_splits=5, seed=42)
    inner = materialize_inner_fold_ledger(frame, splitter, dataset_version_id=1, target_col="G3")
    early = materialize_early_stop_ledger(
        frame, splitter, dataset_version_id=1, target_col="G3", seeds=APPROVED_SEEDS
    )
    assert set(inner["role"]) == {"inner_train", "inner_validation"}
    assert set(early["role"]) == {"model_train", "early_stop", "outer_validation"}
    assert len(early) == size * 5 * len(APPROVED_SEEDS)


def test_no_target_feature_and_no_legacy_observed_access_contracts():
    config = _resolved()
    assert config["feature_contract"]["target_or_derived_features_forbidden"] is True
    runner_source = inspect.getsource(run_strategy_b_phase_ab)
    assert "load_development_subset_from_postgres" in runner_source
    assert "load_dataset_from_postgres" not in runner_source
    pipeline_source = inspect.getsource(run_pipeline.main)
    assert "allow_legacy_observed_evaluation" in pipeline_source
    assert "legacy_heldout_observed" in pipeline_source
    manifest = {"development_records": [{"source_row_number": i} for i in range(1, 317)]}
    invalid = pd.DataFrame({"__source_row_number": list(range(1, 317)) + [999]})
    with pytest.raises(ValueError, match="development cohort"):
        assert_development_only_frame(invalid, manifest)


def test_final_estimator_refits_the_entire_development_frame(monkeypatch):
    frame = pd.DataFrame({"G1": range(12), "G2": range(12), "G3": np.tile([0, 1, 2], 4)})
    seen = {}

    class Result:
        training_diagnostics = {"full_refit_input_records": len(frame)}

    def fake_fit(**kwargs):
        seen["rows"] = len(kwargs["train_partition"])
        return Result()

    monkeypatch.setattr(model_selection, "fit_training_partition_estimator", fake_fit)
    model_selection.fit_final_development_estimator(
        development_frame=frame,
        spec=DATASETS["student-mat"],
        resolved_config=_resolved(),
        seed=42,
    )
    assert seen["rows"] == len(frame)


def test_scheduler_refit_semantics_and_swa_are_replayable():
    config = _resolved()
    assert config["scheduler"] == {"type": "fixed_lr", "parameters": {}, "replayable": True}
    assert config["swa"] == {
        "enabled": False, "batch_norm_statistics_updated": False, "replayable": True
    }
    bad = deepcopy(config)
    bad["swa"]["enabled"] = True
    with pytest.raises(ResolvedConfigError, match="SWA"):
        validate_resolved_config(bad)


def test_drop_last_false_never_omits_a_record():
    b1 = model_selection.loader_statistics(253, 32, True)
    b2 = model_selection.loader_statistics(253, 32, False)
    assert b1["samples_dropped_per_epoch"] == 29
    assert b2["samples_dropped_per_epoch"] == 0
    assert b2["samples_consumed_per_epoch"] == 253


def test_checkpoint_state_dict_roundtrip_is_exact(tmp_path):
    torch.manual_seed(42)
    config = _resolved()
    factory = StudentEstimatorFactory(DATASETS["student-mat"], config)
    model = factory.create_model(0, [], torch.device("cpu"))
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = factory.create_model(0, [], torch.device("cpu"))
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_metrics_recompute_from_saved_oof_and_exact_coverage():
    oof = pd.DataFrame(
        {
            "policy_id": ["B2"] * 6,
            "seed": [42] * 6,
            "outer_fold": [0, 0, 1, 1, 2, 2],
            "source_row_number": [1, 2, 3, 4, 5, 6],
            "true_label": [0, 1, 1, 2, 2, 0],
            "predicted_label": [0, 1, 1, 2, 0, 0],
        }
    )
    metrics = recompute_metrics_from_oof(oof)
    assert metrics["records"].sum() == len(oof)
    validate_oof_coverage(oof, development_rows=list(range(1, 7)), policy_ids=["B2"], seeds=[42])

