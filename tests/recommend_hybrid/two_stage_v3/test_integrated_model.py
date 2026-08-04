from __future__ import annotations

import torch

from src.recommend_hybrid.two_stage_v3.model import (
    HybridIntegratedRecommendationHeads,
    TwoStageHeadConfig,
    two_stage_loss,
)


def config() -> TwoStageHeadConfig:
    return TwoStageHeadConfig(
        group_feature_dim=12,
        action_feature_dim=8,
        group_hidden_dim=16,
        action_embedding_dim=4,
        dropout=0.0,
    )


def test_forward_shapes_and_action_mask() -> None:
    model = HybridIntegratedRecommendationHeads(config())
    group = torch.randn(3, 12)
    actions = torch.randn(3, 5, 8)
    action_ids = torch.arange(5).repeat(3, 1)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    output = model(group, actions, action_ids, mask)
    assert output.recommendability_logit.shape == (3,)
    assert output.action_logits.shape == (3, 5)
    assert output.group_embedding.shape == (3, 16)
    assert torch.all(output.action_logits[~mask] < -1.0e8)


def test_joint_loss_is_finite_with_mixed_positive_groups() -> None:
    model = HybridIntegratedRecommendationHeads(config())
    group = torch.randn(4, 12)
    actions = torch.randn(4, 5, 8)
    action_ids = torch.arange(5).repeat(4, 1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    group_target = torch.tensor([1.0, 0.0, 1.0, 0.0])
    action_target = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    output = model(group, actions, action_ids, mask)
    loss, components = two_stage_loss(
        output,
        group_target=group_target,
        action_target=action_target,
        action_mask=mask,
        positive_weight=torch.tensor(1.0),
        config=config(),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(components["recommendability"])
    assert torch.isfinite(components["listwise"])
    assert torch.isfinite(components["candidate_binary"])


def test_action_loss_is_zero_when_batch_has_no_positive_group() -> None:
    model = HybridIntegratedRecommendationHeads(config())
    group = torch.randn(2, 12)
    actions = torch.randn(2, 5, 8)
    action_ids = torch.arange(5).repeat(2, 1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = model(group, actions, action_ids, mask)
    _, components = two_stage_loss(
        output,
        group_target=torch.zeros(2),
        action_target=torch.zeros(2, 5),
        action_mask=mask,
        positive_weight=torch.tensor(1.0),
        config=config(),
    )
    assert components["listwise"].item() == 0.0
    assert components["candidate_binary"].item() == 0.0
