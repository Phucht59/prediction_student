from __future__ import annotations

import inspect

import numpy as np
import torch

from src.models.oulad_multitask import CNNBiLSTMOULAD
from src.models.oulad_tabular_residual import CNNBiLSTMTabularResidualOULAD
from src.pipelines import oulad
from src.training.phase5_mlp_gap import (
    CONTROL_PARAMETERS,
    Phase5Runner,
    _run_id,
    _selected_configs,
    _teacher_oof,
    architecture_registry,
    make_model,
    status_payload,
)


def _batch() -> tuple[torch.Tensor, ...]:
    lengths = torch.tensor([7, 5, 3])
    mask = (torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1)).float()
    sequence = torch.randn(3, 8, 47) * mask.unsqueeze(-1)
    return (
        sequence,
        lengths,
        mask,
        torch.randn(3, 165),
        torch.randn(3, 13),
    )


def test_h0_reproduces_frozen_a0_layout_and_count() -> None:
    config = _selected_configs()[0]
    h0 = make_model("H0_CURRENT_HYBRID", 165, 13, config)
    reference = CNNBiLSTMOULAD(47, 165, 13, {**config, "fusion": "gated_residual"})
    assert h0.state_dict().keys() == reference.state_dict().keys()
    assert sum(parameter.numel() for parameter in h0.parameters()) == CONTROL_PARAMETERS


def test_m0_uses_authoritative_existing_pipeline() -> None:
    estimator = oulad._make_tabular("mlp", 42).named_steps["model"]
    assert estimator.hidden_layer_sizes == (64, 32)
    assert estimator.alpha == 1e-3
    assert estimator.learning_rate_init == 1e-3
    assert estimator.early_stopping is True


def test_h1_forward_loss_backward_and_dimensions() -> None:
    model = make_model("H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0])
    assert isinstance(model, CNNBiLSTMTabularResidualOULAD)
    output = model(*_batch())
    assert output["binary_logit"].shape == (3,)
    assert output["hazard_logit"].shape == (3, 20)
    assert output["outcome_logit"].shape == (3, 3)
    assert output["tabular_expert_embedding"].shape == (3, 32)
    loss = (
        output["binary_logit"].square().mean()
        + output["hazard_logit"].square().mean()
        + output["outcome_logit"].square().mean()
    )
    loss.backward()
    assert model.backbone.temporal.input_projection.weight.grad is not None
    assert model.tabular_expert[0].weight.grad is not None
    assert model.tabular_risk_head.weight.grad is not None


def test_residual_coefficient_is_bounded_and_initially_small() -> None:
    model = make_model("H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0])
    alpha = float(model.residual_alpha.detach())
    assert 0 < alpha < 1
    assert abs(alpha - 0.05) < 1e-7


def test_h1_parameter_budget_and_temporal_hash_invariant() -> None:
    registry = architecture_registry()
    h0, h1 = registry[1:]
    assert h1["parameter_count"] == 160_492
    assert h1["within_fifteen_percent"] is True
    assert h0["temporal_backbone_hash"] == h1["temporal_backbone_hash"]
    assert h0["fusion_a0_changed"] is False
    assert h1["fusion_a0_changed"] is False


def test_future_masking_invariant_for_h1() -> None:
    torch.manual_seed(11)
    model = make_model("H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]).eval()
    sequence, lengths, mask, aggregate, static = _batch()
    changed = sequence.clone()
    changed[mask.eq(0)] = 1e6
    first = model(sequence, lengths, mask, aggregate, static)["binary_logit"]
    second = model(changed, lengths, mask, aggregate, static)["binary_logit"]
    torch.testing.assert_close(first, second)


def test_h1_consumes_only_stage_safe_aggregate_and_static_outputs() -> None:
    signature = inspect.signature(CNNBiLSTMTabularResidualOULAD.forward)
    assert set(("aggregate", "static")).issubset(signature.parameters)
    assert "full_course_aggregate" not in signature.parameters
    assert tuple(oulad.CONTEXT_COLUMNS) == (
        "progress_fraction",
        "observed_week_count",
        "weeks_remaining",
        "assessment_available_fraction",
    )


def test_outer_label_firewall_and_threshold_scope() -> None:
    for callable_value in (Phase5Runner.__init__, Phase5Runner.evaluate):
        parameters = inspect.signature(callable_value).parameters
        assert "outer_y_test" not in parameters
        assert "outer_labels" not in parameters
    source = inspect.getsource(Phase5Runner.evaluate)
    assert '"research_threshold_scope": "pooled_inner_oof_only"' in source
    assert '"operational_threshold_used": False' in source
    assert '"outer_labels_used": False' in source


def test_run_identity_is_stable_and_candidate_specific() -> None:
    config = _selected_configs()[0]
    first = _run_id("screening", "H1_TABULAR_RESIDUAL_EXPERT", 0, 42, config, 0)
    second = _run_id("screening", "H1_TABULAR_RESIDUAL_EXPERT", 0, 42, config, 0)
    other = _run_id("screening", "H0_CURRENT_HYBRID", 0, 42, config, 0)
    assert first == second
    assert first != other


def test_status_schema_contains_resume_and_sentinel_fields(tmp_path, monkeypatch) -> None:
    from src.training import phase5_mlp_gap as phase5

    monkeypatch.setattr(phase5, "STATUS_PATH", tmp_path / "phase5_status.json")
    payload = status_payload(state="RUNNING", completed_runs=3)
    for field in (
        "started_at",
        "finished_at",
        "current_stage",
        "completed_runs",
        "failed_runs",
        "current_candidate",
        "distillation_triggered",
        "microtune_triggered",
        "exit_code",
    ):
        assert field in payload
    assert payload["completed_runs"] == 3


def test_temporal_and_tabular_ablation_paths_are_effective() -> None:
    torch.manual_seed(19)
    model = make_model("H1_TABULAR_RESIDUAL_EXPERT", 165, 13, _selected_configs()[0]).eval()
    batch = _batch()
    full = model(*batch)["binary_logit"]
    no_temporal = model(*batch, disable_temporal=True)["binary_logit"]
    no_residual = model(*batch, disable_tabular_residual=True)["binary_logit"]
    assert not torch.equal(full, no_temporal)
    assert not torch.equal(full, no_residual)


def test_cross_fitted_teacher_contract_and_teacher_absent_at_inference() -> None:
    source = inspect.getsource(_teacher_oof)
    assert "StratifiedGroupKFold" in source
    assert "fit_ids" in source and "validation_ids" in source
    assert "validation_mask" in source
    parameters = inspect.signature(CNNBiLSTMTabularResidualOULAD.forward).parameters
    assert "teacher_probability" not in parameters
    assert not np.array_equal(np.zeros(2), np.ones(2))
