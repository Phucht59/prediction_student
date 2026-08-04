"""Integrated neural heads over frozen residual CNN-BiLSTM representations.

The prediction backbone remains frozen authority.  These heads consume only
cutoff-safe representations and candidate evidence emitted from that backbone.
No external tree, linear, kernel, or standalone ranking model is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

ACTION_COUNT = 5
MASKED_LOGIT = -1.0e9


@dataclass(frozen=True)
class TwoStageHeadConfig:
    group_feature_dim: int
    action_feature_dim: int
    group_hidden_dim: int = 64
    action_embedding_dim: int = 16
    dropout: float = 0.10
    recommendability_loss_weight: float = 1.0
    listwise_loss_weight: float = 1.0
    candidate_binary_loss_weight: float = 0.25

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
        }


@dataclass(frozen=True)
class TwoStageOutput:
    recommendability_logit: torch.Tensor
    action_logits: torch.Tensor
    group_embedding: torch.Tensor


class HybridIntegratedRecommendationHeads(nn.Module):
    """Recommendability and conditional-action heads sharing hybrid state."""

    model_id = "hybrid_integrated_two_stage_v3"

    def __init__(self, config: TwoStageHeadConfig) -> None:
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
        self.recommendability_head = nn.Sequential(
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
    ) -> TwoStageOutput:
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
        gate_logit = self.recommendability_head(group_embedding).squeeze(1)
        expanded_group = group_embedding.unsqueeze(1).expand(
            -1, action_features.shape[1], -1
        )
        embedded_action = self.action_embedding(action_ids.long())
        scorer_input = torch.cat(
            [expanded_group, embedded_action, action_features.float()], dim=2
        )
        action_logits = self.action_scorer(scorer_input).squeeze(2)
        action_logits = action_logits.masked_fill(~action_mask.bool(), MASKED_LOGIT)
        return TwoStageOutput(
            recommendability_logit=gate_logit,
            action_logits=action_logits,
            group_embedding=group_embedding,
        )


def two_stage_loss(
    output: TwoStageOutput,
    *,
    group_target: torch.Tensor,
    action_target: torch.Tensor,
    action_mask: torch.Tensor,
    positive_weight: torch.Tensor,
    config: TwoStageHeadConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint loss with conditional action learning on positive groups only."""

    group_target = group_target.float()
    action_target = action_target.float()
    valid = action_mask.bool()
    gate_loss = nn.functional.binary_cross_entropy_with_logits(
        output.recommendability_logit,
        group_target,
        pos_weight=positive_weight,
    )

    positive_groups = group_target > 0.5
    positive_valid = valid & positive_groups.unsqueeze(1)
    if positive_valid.any():
        candidate_loss_values = nn.functional.binary_cross_entropy_with_logits(
            output.action_logits[positive_valid],
            action_target[positive_valid],
            reduction="mean",
        )
        selected_logits = output.action_logits[positive_groups]
        selected_target = action_target[positive_groups]
        selected_mask = valid[positive_groups]
        selected_logits = selected_logits.masked_fill(~selected_mask, MASKED_LOGIT)
        positive_count = selected_target.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_distribution = selected_target / positive_count
        log_probability = nn.functional.log_softmax(selected_logits, dim=1)
        listwise_loss = -(target_distribution * log_probability).sum(dim=1).mean()
    else:
        zero = output.recommendability_logit.sum() * 0.0
        candidate_loss_values = zero
        listwise_loss = zero

    total = (
        config.recommendability_loss_weight * gate_loss
        + config.listwise_loss_weight * listwise_loss
        + config.candidate_binary_loss_weight * candidate_loss_values
    )
    return total, {
        "total": total.detach(),
        "recommendability": gate_loss.detach(),
        "listwise": listwise_loss.detach(),
        "candidate_binary": candidate_loss_values.detach(),
    }


__all__ = [
    "ACTION_COUNT",
    "HybridIntegratedRecommendationHeads",
    "MASKED_LOGIT",
    "TwoStageHeadConfig",
    "TwoStageOutput",
    "two_stage_loss",
]
