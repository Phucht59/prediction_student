from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from src.models.oulad_multitask import CNNBiLSTMOULAD
from src.pipelines import oulad
from src.training.phase4_fusion import (
    CANDIDATES,
    CONTROL_PARAMETER_COUNT,
    STAGE_CONTEXT_FIELDS,
    Phase4FusionRunner,
    _model,
    _selected_configs,
    architecture_registry,
    select_stable_architecture,
)


@pytest.mark.parametrize("architecture_id", list(CANDIDATES))
def test_all_fusions_forward_loss_backward_and_aux_heads(architecture_id: str) -> None:
    model = _model(architecture_id, 165, 13, _selected_configs()[0])
    model.train()
    batch = 3
    sequence = torch.randn(batch, 8, 47)
    lengths = torch.tensor([8, 6, 4])
    mask = torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1)
    sequence = sequence * mask.unsqueeze(-1)
    output = model(
        sequence,
        lengths,
        mask.float(),
        torch.randn(batch, 165),
        torch.randn(batch, 13),
    )
    assert output["binary_logit"].shape == (batch,)
    assert output["hazard_logit"].shape == (batch, 20)
    assert output["outcome_logit"].shape == (batch, 3)
    assert output["student_state_embedding"].shape == (batch, 64)
    loss = (
        output["binary_logit"].square().mean()
        + output["hazard_logit"].square().mean()
        + output["outcome_logit"].square().mean()
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_a0_retains_authoritative_state_dict_and_parameter_count() -> None:
    config = _selected_configs()[0]
    phase4 = _model("A0_SCALAR_GATE", 165, 13, config)
    authority_config = {**config, "fusion": "gated_residual"}
    reference = CNNBiLSTMOULAD(47, 165, 13, authority_config)
    assert phase4.state_dict().keys() == reference.state_dict().keys()
    assert sum(parameter.numel() for parameter in phase4.parameters()) == CONTROL_PARAMETER_COUNT


def test_parameter_budget_and_backbone_hash_invariant() -> None:
    registry = architecture_registry()
    assert len(registry) == 4
    assert len({row["architecture_hash"] for row in registry}) == 4
    assert len({row["backbone_hash"] for row in registry}) == 1
    assert all(row["within_ten_percent"] for row in registry)
    assert all(135_000 <= row["total_parameter_count"] <= 165_000 for row in registry)


def test_film_is_exact_temporal_identity_at_initialization() -> None:
    model = _model("A3_FILM", 165, 13, _selected_configs()[0])
    model.eval()
    temporal = torch.randn(5, 64)
    aggregate = torch.randn(5, 64)
    static = torch.randn(5, 64)
    fused, diagnostics = model.backbone.fuse(temporal, aggregate, static)
    assert torch.equal(fused, temporal)
    assert torch.count_nonzero(diagnostics["gamma"]) == 0
    assert torch.count_nonzero(diagnostics["beta"]) == 0


def test_future_timesteps_do_not_change_predictions_for_all_fusions() -> None:
    for architecture_id in CANDIDATES:
        torch.manual_seed(7)
        model = _model(architecture_id, 165, 13, _selected_configs()[0]).eval()
        lengths = torch.tensor([3, 5])
        mask = (torch.arange(8).unsqueeze(0) < lengths.unsqueeze(1)).float()
        sequence = torch.randn(2, 8, 47) * mask.unsqueeze(-1)
        changed = sequence.clone()
        changed[mask.eq(0)] = 1e6
        aggregate = torch.randn(2, 165)
        static = torch.randn(2, 13)
        first = model(sequence, lengths, mask, aggregate, static)["binary_logit"]
        second = model(changed, lengths, mask, aggregate, static)["binary_logit"]
        torch.testing.assert_close(first, second)


def test_stage_context_is_legal_and_already_explicit() -> None:
    assert tuple(oulad.CONTEXT_COLUMNS) == STAGE_CONTEXT_FIELDS
    prohibited = {"target", "prevalence", "test_metric", "future_performance"}
    assert prohibited.isdisjoint(STAGE_CONTEXT_FIELDS)


def test_runner_api_has_outer_label_firewall() -> None:
    init_parameters = inspect.signature(Phase4FusionRunner.__init__).parameters
    evaluate_parameters = inspect.signature(Phase4FusionRunner.evaluate).parameters
    for name in ("outer_y_test", "outer_labels", "y_test"):
        assert name not in init_parameters
        assert name not in evaluate_parameters


def test_architecture_ids_and_hashes_are_stable() -> None:
    first = architecture_registry()
    second = architecture_registry()
    assert first == second
    assert [row["architecture_id"] for row in first] == list(CANDIDATES)


def test_vector_gate_and_film_are_numerically_finite() -> None:
    for architecture_id in ("A1_VECTOR_GATE", "A3_FILM"):
        model = _model(architecture_id, 165, 13, _selected_configs()[0]).eval()
        temporal = torch.randn(16, 64) * 10
        aggregate = torch.randn(16, 64) * 10
        static = torch.randn(16, 64) * 10
        fused, diagnostics = model.backbone.fuse(temporal, aggregate, static)
        assert torch.isfinite(fused).all()
        assert all(torch.isfinite(value).all() for value in diagnostics.values())


def test_research_threshold_implementation_is_inner_oof_only() -> None:
    source = inspect.getsource(Phase4FusionRunner.evaluate)
    assert "evaluate_oof" in source
    assert '"research_threshold_scope": "pooled_inner_oof_only"' in source
    assert '"outer_labels_used": False' in source
    assert not np.array_equal(np.zeros(3), np.ones(3))


def test_negligible_primary_gain_with_all_secondary_metrics_worse_retains_control() -> None:
    control = {
        "architecture_id": "A0_SCALAR_GATE",
        "mean_stage_macro_f1": 0.7731,
        "worst_stage_macro_f1": 0.7055,
        "mean_stage_pr_auc": 0.8292,
        "mean_stage_nll": 0.4481,
        "mean_stage_brier": 0.1470,
        "mean_stage_ece": 0.0185,
        "total_parameter_count": 150_202,
    }
    candidate = {
        "architecture_id": "A1_VECTOR_GATE",
        "mean_stage_macro_f1": 0.7736,
        "worst_stage_macro_f1": 0.7047,
        "mean_stage_pr_auc": 0.8291,
        "mean_stage_nll": 0.4488,
        "mean_stage_brier": 0.1472,
        "mean_stage_ece": 0.0213,
        "total_parameter_count": 155_080,
    }
    assert select_stable_architecture([control, candidate]) == (
        "A1_VECTOR_GATE",
        "A0_SCALAR_GATE",
    )
