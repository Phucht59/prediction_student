"""Integrated conditional action heads over a frozen student representation.

The candidate head is supervised on every valid candidate, including all-zero
action targets. A differentiable noisy-OR provides an action-derived signal
while a consistency loss aligns it with the direct gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

ACTION_COUNT = 5
MASKED_LOGIT = -1.0e9
EPSILON = 1.0e-6


@dataclass(frozen=True)
class ActionAwareHeadConfig:
    group_feature_dim: int
    action_feature_dim: int
    group_hidden_dim: int = 64
    action_embedding_dim: int = 16
    dropout: float = 0.10
    recommendability_loss_weight: float = 1.0
    listwise_loss_weight: float = 1.0
    candidate_binary_loss_weight: float = 2.0
    consistency_loss_weight: float = 0.5
    focal_gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.group_feature_dim <= 0 or self.action_feature_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.group_hidden_dim <= 0 or self.action_embedding_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in (
            "recommendability_loss_weight",
            "listwise_loss_weight",
            "candidate_binary_loss_weight",
            "consistency_loss_weight",
            "focal_gamma",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_feature_dim": self.group_feature_dim,
            "action_feature_dim": self.action_feature_dim,
            "group_hidden_dim": self.group_hidden_dim,
            "action_embedding_dim": self.action_embedding_dim,
            "dropout": self.dropout,
            "recommendability_loss_weight": self.recommendability_loss_weight,
            "listwise_loss_weight": self.listwise_loss_weight,
            "candidate_binary_loss_weight": self.candidate_binary_loss_weight,
            "consistency_loss_weight": self.consistency_loss_weight,
            "focal_gamma": self.focal_gamma,
        }


@dataclass(frozen=True)
class ActionAwareOutput:
    direct_gate_logit: torch.Tensor
    action_logits: torch.Tensor
    action_any_probability: torch.Tensor
    group_embedding: torch.Tensor


class HybridActionAwareRecommendationHeads(nn.Module):
    """Direct gate and all-group candidate heads sharing frozen hybrid state."""

    model_id = "conditional_hybrid_action_ranker"

    def __init__(self, config: ActionAwareHeadConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.group_hidden_dim
        self.group_encoder = nn.Sequential(
            nn.Linear(config.group_feature_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.direct_gate_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.action_embedding = nn.Embedding(ACTION_COUNT, config.action_embedding_dim)
        action_input = hidden + config.action_embedding_dim + config.action_feature_dim
        self.action_scorer = nn.Sequential(
            nn.Linear(action_input, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden // 2), 1),
        )

    def forward(
        self,
        group_features: torch.Tensor,
        action_features: torch.Tensor,
        action_ids: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> ActionAwareOutput:
        if group_features.ndim != 2:
            raise ValueError("group_features must be [B, G]")
        if action_features.ndim != 3:
            raise ValueError("action_features must be [B, A, F]")
        if action_ids.shape != action_features.shape[:2]:
            raise ValueError("action_ids must align with action_features")
        if action_mask.shape != action_features.shape[:2]:
            raise ValueError("action_mask must align with action_features")
        if action_features.shape[1] != ACTION_COUNT:
            raise ValueError(f"expected exactly {ACTION_COUNT} action slots")
        if group_features.shape[1] != self.config.group_feature_dim:
            raise ValueError("unexpected group feature dimension")
        if action_features.shape[2] != self.config.action_feature_dim:
            raise ValueError("unexpected action feature dimension")

        group_embedding = self.group_encoder(group_features.float())
        direct_gate_logit = self.direct_gate_head(group_embedding).squeeze(1)
        expanded_group = group_embedding.unsqueeze(1).expand(
            -1, action_features.shape[1], -1
        )
        embedded_action = self.action_embedding(action_ids.long())
        scorer_input = torch.cat(
            [expanded_group, embedded_action, action_features.float()], dim=2
        )
        raw_action_logits = self.action_scorer(scorer_input).squeeze(2)
        valid = action_mask.bool()
        action_logits = raw_action_logits.masked_fill(~valid, MASKED_LOGIT)
        candidate_probability = torch.sigmoid(raw_action_logits).masked_fill(~valid, 0.0)
        action_any_probability = 1.0 - torch.prod(
            torch.where(valid, 1.0 - candidate_probability, torch.ones_like(candidate_probability)),
            dim=1,
        )
        action_any_probability = action_any_probability.clamp(EPSILON, 1.0 - EPSILON)
        return ActionAwareOutput(
            direct_gate_logit=direct_gate_logit,
            action_logits=action_logits,
            action_any_probability=action_any_probability,
            group_embedding=group_embedding,
        )


def _weighted_focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    positive_weight: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Elementwise weighted BCE with optional focal modulation."""

    target = target.float()
    base = nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
        reduction="none",
    )
    if gamma <= 0.0:
        return base
    probability = torch.sigmoid(logits)
    probability_of_target = torch.where(target > 0.5, probability, 1.0 - probability)
    return base * torch.pow((1.0 - probability_of_target).clamp_min(0.0), gamma)


def action_aware_loss(
    output: ActionAwareOutput,
    *,
    group_target: torch.Tensor,
    action_target: torch.Tensor,
    action_mask: torch.Tensor,
    group_positive_weight: torch.Tensor,
    action_positive_weight: torch.Tensor,
    config: ActionAwareHeadConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint V4 loss with candidate supervision on all valid groups.

    The listwise term remains conditional on positive groups because ranking is
    undefined for all-zero groups.  Candidate BCE and noisy-OR group loss are
    evaluated on positive and negative groups, directly penalizing false issue
    behaviour that V3 left unsupervised.
    """

    group_target = group_target.float()
    action_target = action_target.float()
    valid = action_mask.bool()
    direct_gate_loss = nn.functional.binary_cross_entropy_with_logits(
        output.direct_gate_logit,
        group_target,
        pos_weight=group_positive_weight,
    )

    expanded_positive_weight = action_positive_weight.reshape(1, -1).expand_as(
        action_target
    )
    candidate_values = _weighted_focal_bce_with_logits(
        output.action_logits.masked_fill(~valid, 0.0),
        action_target,
        positive_weight=expanded_positive_weight,
        gamma=float(config.focal_gamma),
    )
    candidate_binary_loss = candidate_values[valid].mean() if valid.any() else (
        output.direct_gate_logit.sum() * 0.0
    )

    positive_groups = group_target > 0.5
    if positive_groups.any():
        selected_logits = output.action_logits[positive_groups]
        selected_target = action_target[positive_groups]
        selected_mask = valid[positive_groups]
        selected_logits = selected_logits.masked_fill(~selected_mask, MASKED_LOGIT)
        positive_count = selected_target.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_distribution = selected_target / positive_count
        log_probability = nn.functional.log_softmax(selected_logits, dim=1)
        listwise_loss = -(target_distribution * log_probability).sum(dim=1).mean()
    else:
        listwise_loss = output.direct_gate_logit.sum() * 0.0

    action_group_loss = nn.functional.binary_cross_entropy(
        output.action_any_probability,
        group_target,
    )
    direct_probability = torch.sigmoid(output.direct_gate_logit)
    agreement_loss = nn.functional.mse_loss(
        direct_probability,
        output.action_any_probability,
    )
    consistency_loss = 0.5 * (action_group_loss + agreement_loss)

    total = (
        config.recommendability_loss_weight * direct_gate_loss
        + config.listwise_loss_weight * listwise_loss
        + config.candidate_binary_loss_weight * candidate_binary_loss
        + config.consistency_loss_weight * consistency_loss
    )
    return total, {
        "total": total.detach(),
        "direct_recommendability": direct_gate_loss.detach(),
        "candidate_binary_all_groups": candidate_binary_loss.detach(),
        "listwise_positive_groups": listwise_loss.detach(),
        "action_group_consistency": consistency_loss.detach(),
    }


__all__ = [
    "ACTION_COUNT",
    "ActionAwareHeadConfig",
    "ActionAwareOutput",
    "HybridActionAwareRecommendationHeads",
    "MASKED_LOGIT",
    "action_aware_loss",
]
