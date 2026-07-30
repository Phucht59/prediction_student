"""Frozen CNN-BiLSTM OULAD architecture used by the final checkpoints."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models._oulad import _OULADCNNBiLSTMBackbone

HORIZON_WEEKS = 20


class CNNBiLSTMOULAD(nn.Module):
    """Risk main head with survival and outcome auxiliary heads."""

    model_id = "cnn_bilstm_oulad"
    display_name = "CNN-BiLSTM OULAD"

    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.backbone = _OULADCNNBiLSTMBackbone(
            sequence_channels, aggregate_dim, static_dim, config, "cnn_bilstm"
        )
        self.representation_dim = self.backbone.representation_dim
        self.survival_head = nn.Linear(self.representation_dim, HORIZON_WEEKS)
        self.outcome_head = nn.Linear(self.representation_dim, 3)

    def representation(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        temporal, _, _ = self.backbone.temporal(sequence, lengths, mask)
        temporal = self.backbone.temporal_projection(temporal)
        aggregate_embedding = self.backbone._drop_branch(self.backbone.aggregate(aggregate))
        static_embedding = self.backbone._drop_branch(self.backbone.static(static))
        if self.backbone.gates is None:
            return torch.cat([temporal, aggregate_embedding, static_embedding], dim=1)
        gate = self.backbone.gates(
            torch.cat([temporal, aggregate_embedding, static_embedding], dim=1)
        )
        return temporal + gate[:, 0:1] * aggregate_embedding + gate[:, 1:2] * static_embedding

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        representation = self.representation(sequence, lengths, mask, aggregate, static)
        return {
            "binary_logit": self.backbone.head(representation).squeeze(1),
            "hazard_logit": self.survival_head(representation),
            "outcome_logit": self.outcome_head(representation),
            "student_state_embedding": representation,
        }
