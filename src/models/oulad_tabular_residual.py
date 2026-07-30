"""Phase 5 CNN-BiLSTM with a compact direct tabular residual risk expert."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from src.models.oulad_multitask import CNNBiLSTMOULAD


class CNNBiLSTMTabularResidualOULAD(CNNBiLSTMOULAD):
    """Frozen A0 hybrid plus a bounded, initially-small tabular logit bypass."""

    model_id = "cnn_bilstm_tabular_residual_oulad"
    display_name = "CNN-BiLSTM + Tabular Residual Expert"

    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
    ) -> None:
        frozen = {**config, "fusion": "gated_residual"}
        super().__init__(sequence_channels, aggregate_dim, static_dim, frozen)
        tabular_input = aggregate_dim + static_dim
        dropout = float(config.get("dropout", 0.2))
        self.tabular_expert = nn.Sequential(
            nn.Linear(tabular_input, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.GELU(),
        )
        self.tabular_risk_head = nn.Linear(32, 1)
        initial_alpha = float(config.get("residual_alpha_initial", 0.05))
        if not 0.0 < initial_alpha < 1.0:
            raise ValueError("residual_alpha_initial must be strictly between zero and one")
        self.residual_alpha_logit = nn.Parameter(
            torch.tensor(math.log(initial_alpha / (1.0 - initial_alpha)))
        )

    @property
    def residual_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_alpha_logit)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
        *,
        disable_temporal: bool = False,
        disable_tabular_residual: bool = False,
    ) -> dict[str, torch.Tensor]:
        temporal, aggregate_embedding, static_embedding, _, _ = (
            self.backbone.encode_branches(sequence, lengths, mask, aggregate, static)
        )
        if disable_temporal:
            temporal = torch.zeros_like(temporal)
        representation, _ = self.backbone.fuse(
            temporal, aggregate_embedding, static_embedding
        )
        hybrid_logit = self.backbone.head(representation).squeeze(1)
        tabular_representation = self.tabular_expert(
            torch.cat([aggregate.float(), static.float()], dim=1)
        )
        tabular_logit = self.tabular_risk_head(tabular_representation).squeeze(1)
        residual = (
            torch.zeros_like(tabular_logit)
            if disable_tabular_residual
            else self.residual_alpha * tabular_logit
        )
        return {
            "binary_logit": hybrid_logit + residual,
            "hybrid_logit": hybrid_logit,
            "tabular_logit": tabular_logit,
            "residual_logit": residual,
            "residual_alpha": self.residual_alpha.expand_as(hybrid_logit),
            "hazard_logit": self.survival_head(representation),
            "outcome_logit": self.outcome_head(representation),
            "student_state_embedding": representation,
            "tabular_expert_embedding": tabular_representation,
        }


__all__ = ["CNNBiLSTMTabularResidualOULAD"]
