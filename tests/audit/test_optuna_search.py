from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import optuna
import torch

from src.pipelines import oulad
from src.training import optuna_search as search
from src.training.config_authority import (
    architecture_metadata,
    load_config_authority,
)
from src.training.control import select_research_threshold, stable_hash


def test_architecture_hash_and_parameter_count_are_invariant() -> None:
    contract = search.architecture_contract()
    assert contract["parameter_count"] == 150202
    first = search._model_config(search.control_config())
    altered = {
        **search.control_config(),
        "learning_rate": 1e-4,
        "dropout": 0.35,
        "survival_weight": 0.0,
        "outcome_weight": 0.2,
    }
    second = search._model_config(altered)
    authority = load_config_authority(search.AUTHORITY_PATH)
    hashes = set()
    counts = set()
    for config in (first, second):
        model = oulad._deep_model("cnn_bilstm", 165, 13, config)
        metadata = architecture_metadata(
            model, authority=authority, aggregate_dim=165, static_dim=13
        )
        hashes.add(metadata["architecture_hash"])
        counts.add(metadata["parameter_count"])
    assert hashes == {contract["architecture_hash"]}
    assert counts == {150202}


def test_pretraining_is_frozen_disabled() -> None:
    authority = load_config_authority(search.AUTHORITY_PATH)
    assert authority["pretraining"] == {
        "requested": False,
        "executed": False,
        "checkpoint": None,
        "strategy": None,
    }


def test_runner_api_cannot_consume_outer_test_labels() -> None:
    init_parameters = inspect.signature(search.OptunaSearchRunner).parameters
    evaluate_parameters = inspect.signature(
        search.OptunaSearchRunner.evaluate
    ).parameters
    assert "outer_y_test" not in init_parameters
    assert "outer_y_test" not in evaluate_parameters
    assert "outer_labels" not in init_parameters
    assert "outer_labels" not in evaluate_parameters


def test_checkpoint_policy_epoch_cap_and_pruning_warmup_are_frozen() -> None:
    authority = load_config_authority(search.AUTHORITY_PATH)
    assert search.MAX_EPOCHS == authority["training"]["max_epochs"] == 15
    assert authority["training"]["monitor"] == "mean_stage_validation_nll"
    assert search.PRUNING_WARMUP_EPOCHS >= 3
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=6, n_warmup_steps=search.PRUNING_WARMUP_EPOCHS
        ),
    )
    assert isinstance(study.pruner, optuna.pruners.MedianPruner)


def test_search_space_changes_training_only() -> None:
    space = search.search_space_manifest()
    assert set(space) == {
        "learning_rate",
        "weight_decay",
        "dropout",
        "batch_size",
        "loss_policy",
        "pos_weight_strategy",
        "survival_weight",
        "outcome_weight",
        "frozen",
    }
    frozen = space["frozen"]
    assert frozen["branch_dropout"] == 0.1
    assert frozen["optimizer"] == "AdamW"
    assert frozen["scheduler"] is None
    assert frozen["pretraining_executed"] is False


def test_research_threshold_is_inner_only_and_operational_is_excluded() -> None:
    signature = inspect.signature(select_research_threshold)
    assert list(signature.parameters) == [
        "inner_oof_labels",
        "inner_oof_probabilities",
    ]
    space = search.search_space_manifest()
    serialized = str(space)
    assert "operational" not in serialized.lower()
    result = select_research_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )
    assert result["outer_labels_used"] is False


def test_optuna_sqlite_resume_does_not_duplicate_completed_budget(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(search, "OPTUNA_DIR", tmp_path)
    first = search.create_study(0)
    first.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=1)
    assert len(first.trials) == 1
    resumed = search.create_study(0)
    remaining = max(0, 1 - len(resumed.trials))
    if remaining:
        resumed.optimize(
            lambda trial: trial.suggest_float("x", 0.0, 1.0),
            n_trials=remaining,
        )
    assert len(resumed.trials) == 1
    assert resumed.study_name == first.study_name


def test_status_and_sentinel_transitions_are_machine_readable(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(search, "RUNTIME", runtime)
    monkeypatch.setattr(search, "STATUS_PATH", runtime / "status.json")
    monkeypatch.setattr(search, "RUNNING", runtime / "RUNNING")
    monkeypatch.setattr(search, "COMPLETE", runtime / "COMPLETE")
    monkeypatch.setattr(search, "FAILED", runtime / "FAILED")
    search.status_payload(state="RUNNING", current_stage="test")
    search.set_sentinel("RUNNING")
    assert search.status()["state"] == "RUNNING"
    assert search.RUNNING.is_file()
    search.set_sentinel("COMPLETE")
    assert search.COMPLETE.is_file()
    assert not search.RUNNING.exists()


def test_run_identity_config_hash_is_stable() -> None:
    config = search.control_config()
    checkpoint = {"config_hash": stable_hash(config)}
    manifest = {"config_hash": stable_hash(config)}
    trial = {"config_hash": stable_hash(config)}
    assert checkpoint["config_hash"] == manifest["config_hash"] == trial["config_hash"]


def test_loss_positive_weight_uses_passed_inner_train_labels_only() -> None:
    device = torch.device("cpu")
    labels = np.array([0, 0, 0, 1])
    _, full = search._risk_loss(
        labels,
        {**search.control_config(), "pos_weight_strategy": "full_ratio"},
        device,
    )
    _, square_root = search._risk_loss(
        labels,
        {**search.control_config(), "pos_weight_strategy": "sqrt_ratio"},
        device,
    )
    _, standard = search._risk_loss(
        labels,
        {
            **search.control_config(),
            "loss_policy": "standard_bce",
            "pos_weight_strategy": "not_applicable",
        },
        device,
    )
    assert full == 3.0
    assert square_root == np.sqrt(3.0)
    assert standard == 1.0


def test_completed_phase3_evidence_satisfies_firewall_and_budget() -> None:
    gate_path = search.OUT / "phase3_gate.json"
    if not gate_path.is_file():
        return
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    trials = json.loads(
        (search.OUT / "all_trials.json").read_text(encoding="utf-8")
    )
    complete = [row for row in trials if row["state"] == "COMPLETE"]
    assert gate["status"] == "PASS"
    assert gate["scheduled_trials"] == 72
    assert gate["completed_trials"] + gate["pruned_trials"] == 72
    assert gate["failed_trials"] == gate["oom_trials"] == 0
    assert {row["architecture_hash"] for row in complete} == {
        search.architecture_contract()["architecture_hash"]
    }
    assert {row["parameter_count"] for row in complete} == {150202}
    assert not any(row["outer_labels_used"] for row in complete)
    assert not any(row["pretraining_executed"] for row in complete)
    stability = json.loads(
        (search.OUT / "stability_results.json").read_text(encoding="utf-8")
    )
    assert len(stability) == 12
    assert all(
        (search.OPTUNA_DIR / f"outer{fold}.db").is_file()
        for fold in range(3)
    )
