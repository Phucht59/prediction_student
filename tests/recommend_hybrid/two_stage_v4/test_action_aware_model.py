from __future__ import annotations

import torch

from src.recommend_hybrid.two_stage_v4.model import (
    ActionAwareHeadConfig,
    HybridActionAwareRecommendationHeads,
    action_aware_loss,
)


def config() -> ActionAwareHeadConfig:
    return ActionAwareHeadConfig(
        group_feature_dim=6,
        action_feature_dim=4,
        group_hidden_dim=8,
        action_embedding_dim=4,
        dropout=0.0,
        recommendability_loss_weight=1.0,
        listwise_loss_weight=1.0,
        candidate_binary_loss_weight=2.0,
        consistency_loss_weight=0.5,
        focal_gamma=0.0,
    )


def test_forward_emits_direct_and_action_derived_gate() -> None:
    model = HybridActionAwareRecommendationHeads(config())
    output = model(
        torch.randn(3, 6),
        torch.randn(3, 5, 4),
        torch.arange(5).repeat(3, 1),
        torch.ones(3, 5, dtype=torch.bool),
    )
    assert output.direct_gate_logit.shape == (3,)
    assert output.action_logits.shape == (3, 5)
    assert output.action_any_probability.shape == (3,)
    assert torch.all(output.action_any_probability > 0)
    assert torch.all(output.action_any_probability < 1)


def test_negative_groups_create_candidate_head_gradient() -> None:
    model = HybridActionAwareRecommendationHeads(config())
    group = torch.randn(2, 6)
    actions = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = model(group, actions, torch.arange(5).repeat(2, 1), mask)
    loss, parts = action_aware_loss(
        output,
        group_target=torch.zeros(2),
        action_target=torch.zeros(2, 5),
        action_mask=mask,
        group_positive_weight=torch.tensor(1.0),
        action_positive_weight=torch.ones(5),
        config=config(),
    )
    loss.backward()
    final_weight = model.action_scorer[-1].weight.grad
    assert final_weight is not None
    assert torch.count_nonzero(final_weight).item() > 0
    assert parts["candidate_binary_all_groups"].item() > 0


def test_positive_group_keeps_listwise_learning() -> None:
    model = HybridActionAwareRecommendationHeads(config())
    mask = torch.ones(1, 5, dtype=torch.bool)
    output = model(
        torch.randn(1, 6),
        torch.randn(1, 5, 4),
        torch.arange(5).reshape(1, 5),
        mask,
    )
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    loss, parts = action_aware_loss(
        output,
        group_target=torch.ones(1),
        action_target=target,
        action_mask=mask,
        group_positive_weight=torch.tensor(1.0),
        action_positive_weight=torch.ones(5),
        config=config(),
    )
    assert torch.isfinite(loss)
    assert parts["listwise_positive_groups"].item() > 0
