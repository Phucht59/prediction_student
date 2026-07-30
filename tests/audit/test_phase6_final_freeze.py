from __future__ import annotations

import inspect
import json

import torch

from scripts.phase6.build_freeze_manifest import build_manifest
from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD
from src.pipelines import oulad
from src.training.control import stable_hash
from src.training.phase5_mlp_gap import _selected_configs, make_model
from src.training.phase6_final import (
    _checkpoint_path,
    _run_id,
    status_payload,
    validate_freeze,
)


def test_freeze_manifest_scientific_hashes_and_exact_protocol() -> None:
    manifest = build_manifest()
    science = manifest["scientific_configuration"]
    protocol = science["evaluation_protocol"]
    assert manifest["freeze_status"] == "IMMUTABLE_PRE_OUTER"
    assert stable_hash(science) == manifest["final_candidate_hash"]
    assert stable_hash(science["feature_schema"]) == manifest["feature_schema_hash"]
    assert stable_hash(science["preprocessing"]) == manifest["preprocessing_hash"]
    assert stable_hash(science["training_policy"]) == manifest["training_policy_hash"]
    assert (
        stable_hash(protocol)
        == manifest["evaluation_protocol_hash"]
    )
    assert protocol["outer_folds"] == 3
    assert protocol["inner_folds"] == 2
    assert protocol["final_seeds"] == [42, 1201, 2026, 3407, 7319]
    assert list(protocol["stages"]) == list(oulad.STAGES)
    assert [
        protocol["stages"][stage]["target_progress_fraction"]
        for stage in oulad.STAGES
    ] == [0.2, 0.35, 0.5, 0.75]
    assert protocol["candidate_count_h1"] == 1
    assert protocol["optuna_trials"] == 0
    assert protocol["one_checkpoint_all_stages"] is True


def test_freeze_uses_phase5_inner_only_authority() -> None:
    manifest = build_manifest()
    authority = manifest["scientific_configuration"]["inner_authority"]
    assert set(authority) == {
        "H1_TABULAR_RESIDUAL_EXPERT",
        "H0_CURRENT_HYBRID",
        "M0_MLP",
    }
    for folds in authority.values():
        assert set(folds) == {"0", "1", "2"}
        for frozen in folds.values():
            assert frozen["training_seed"] == 42
            assert frozen["threshold_source"] == "pooled_inner_oof_only"
            assert frozen["outer_labels_used"] is False
            assert set(frozen["research_thresholds"]) == set(oulad.STAGES)


def test_h1_frozen_identity_and_parameter_count() -> None:
    manifest = build_manifest()
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]
    ).eval()
    assert isinstance(model, CNNBiLSTMTabularResidualOULAD)
    assert sum(parameter.numel() for parameter in model.parameters()) == 160_492
    assert manifest["parameter_count"] == 160_492
    assert manifest["h2_rejected"] is True
    assert manifest["architecture_frozen"] is True
    assert manifest["hyperparameters_frozen"] is True


def test_h1_smoke_forward_backward_and_future_mask() -> None:
    torch.manual_seed(31)
    model = make_model(
        "H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]
    ).eval()
    lengths = torch.tensor([6, 3])
    mask = (torch.arange(7).unsqueeze(0) < lengths.unsqueeze(1)).float()
    sequence = torch.randn(2, 7, 47) * mask.unsqueeze(-1)
    aggregate = torch.randn(2, 165)
    static = torch.randn(2, 13)
    first = model(sequence, lengths, mask, aggregate, static)["binary_logit"]
    changed = sequence.clone()
    changed[mask.eq(0)] = 1e7
    second = model(changed, lengths, mask, aggregate, static)["binary_logit"]
    torch.testing.assert_close(first, second)
    first.square().mean().backward()
    assert model.backbone.temporal.input_projection.weight.grad is not None
    assert model.tabular_risk_head.weight.grad is not None


def test_final_runner_has_no_outer_selection_inputs() -> None:
    from src.training import phase6_final

    for function in (
        phase6_final._train_deep_final,
        phase6_final._fit_mlp_final,
        phase6_final._run_id,
    ):
        parameters = inspect.signature(function).parameters
        assert "outer_y_test" not in parameters
        assert "outer_labels" not in parameters
    source = inspect.getsource(phase6_final)
    assert "create_study" not in source
    assert "study.optimize" not in source
    assert "outer_labels_used_for_threshold_selection" in source
    assert "one-shot rerun prohibited" in source


def test_run_identity_and_checkpoint_topology_are_stable() -> None:
    first = _run_id("candidate", "H1_TABULAR_RESIDUAL_EXPERT", 0, 42, "protocol")
    second = _run_id("candidate", "H1_TABULAR_RESIDUAL_EXPERT", 0, 42, "protocol")
    other = _run_id("candidate", "H1_TABULAR_RESIDUAL_EXPERT", 0, 1201, "protocol")
    assert first == second
    assert first != other
    checkpoint = _checkpoint_path("H1_TABULAR_RESIDUAL_EXPERT", 0, 42)
    assert "stage" not in checkpoint.name


def test_status_schema_and_precommit_freeze_validation(tmp_path, monkeypatch) -> None:
    from src.training import phase6_final

    monkeypatch.setattr(phase6_final, "STATUS_PATH", tmp_path / "status.json")
    payload = status_payload(state="RUNNING", completed_runs=2)
    for field in (
        "started_at",
        "finished_at",
        "current_stage",
        "completed_runs",
        "failed_runs",
        "current_candidate",
        "current_outer_fold",
        "current_seed",
        "exit_code",
    ):
        assert field in payload
    assert payload["completed_runs"] == 2

    manifest = build_manifest()
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(phase6_final, "FREEZE_PATH", freeze_path)
    assert validate_freeze(require_freeze_commit=False)["status"] == "PASS"
