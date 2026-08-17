"""Exploratory stage-conditioned Hybrid with one shared temporal/context backbone."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .components import BiLSTMBranch, ResidualCNNBranch


@dataclass(frozen=True)
class StageConditionedConfig:
    domain: str
    temporal_dim: int
    context_dim: int
    d_model: int
    cnn_channels: int
    cnn_blocks: int
    bilstm_hidden: int
    bilstm_layers: int
    context_hidden: int
    shared_head_hidden: int
    dropout: float
    summary_residual: bool = False
    uci_wide_context: bool = True


class StageConditionedHybrid(nn.Module):
    """One jointly-trained model with shared experts and lightweight stage heads."""

    model_id = "hybrid"
    display_name = "Hybrid"

    def __init__(self, config: StageConditionedConfig):
        super().__init__()
        if config.domain not in {"uci", "oulad"}:
            raise ValueError(config.domain)
        self.config = config
        self.stage_names = ("S0", "S1", "S2") if config.domain == "uci" else ("20pct", "35pct", "50pct", "75pct")
        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, config.d_model), nn.LayerNorm(config.d_model))
        self.cnn_projection = nn.Identity() if config.d_model == config.cnn_channels else nn.Linear(config.d_model, config.cnn_channels)
        dilations = (1,) if config.cnn_blocks == 1 else (1, 2)
        self.cnn = ResidualCNNBranch(config.cnn_channels, 2, dilations, config.dropout)
        self.bilstm = BiLSTMBranch(config.d_model, config.bilstm_hidden, config.bilstm_layers)
        self.context = nn.Sequential(
            nn.Linear(config.context_dim, config.context_hidden),
            nn.LayerNorm(config.context_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        temporal_width = 2 * config.cnn_channels + 4 * config.bilstm_hidden
        if config.domain == "uci":
            self.heads = nn.ModuleDict({
                "S0": self._head(config.context_hidden),
                "S1": self._head(temporal_width + config.context_hidden + 5),
                "S2": self._head(temporal_width + config.context_hidden + 5),
            })
            self.wide_s0 = nn.Linear(config.context_dim, 1) if config.uci_wide_context else None
            self.summary_channels = ()
            self.summary_width = 0
        else:
            # Frozen small base-channel subset: clicks, activity, assessment activity,
            # submissions, recency, and inactivity streak. It derives only from the
            # already cutoff-safe tensor.
            self.summary_channels = (0, 1, 7, 8, 10, 30)
            self.summary_width = len(self.summary_channels) * 3 + 1 if config.summary_residual else 0
            width = temporal_width + config.context_hidden + self.summary_width
            self.heads = nn.ModuleDict({stage: self._head(width) for stage in self.stage_names})
            self.wide_s0 = None

    def _head(self, width: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, self.config.shared_head_hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.shared_head_hidden, 1),
        )

    def _uci_residual(self, temporal: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        g1 = temporal[:, 0, 0] if temporal.shape[1] else temporal.new_zeros(len(temporal))
        g2 = temporal[:, 1, 0] if temporal.shape[1] > 1 else temporal.new_zeros(len(temporal))
        available1 = (lengths >= 1).to(temporal.dtype)
        available2 = (lengths >= 2).to(temporal.dtype)
        g1 = g1 * available1
        g2 = g2 * available2
        return torch.stack((g1, g2, (g2 - g1) * available2, available1, available2), dim=-1)

    def _small_summary(self, temporal: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        selected = temporal[:, :, self.summary_channels]
        weights = mask.unsqueeze(-1).to(selected.dtype)
        count = weights.sum(1).clamp_min(1.0)
        mean = (selected * weights).sum(1) / count
        batch = torch.arange(len(selected), device=selected.device)
        last_index = (lengths - 1).clamp_min(0)
        previous_index = (lengths - 2).clamp_min(0)
        last = selected[batch, last_index] * (lengths > 0).unsqueeze(-1)
        previous = selected[batch, previous_index] * (lengths > 1).unsqueeze(-1)
        slope = (last - previous) * (lengths > 1).unsqueeze(-1)
        normalized_length = (lengths.to(selected.dtype) / max(1, temporal.shape[1])).unsqueeze(-1)
        return torch.cat((last, mean, slope, normalized_length), dim=-1)

    def forward(self, temporal, mask, lengths, context, stage_index):
        adapted = self.temporal_adapter(temporal) * mask.unsqueeze(-1).to(temporal.dtype)
        cnn_input = self.cnn_projection(adapted) * mask.unsqueeze(-1).to(temporal.dtype)
        cnn = self.cnn(cnn_input, mask)
        bilstm = self.bilstm(adapted, mask, lengths)
        context_rep = self.context(context)
        temporal_rep = torch.cat((cnn, bilstm), dim=-1)
        residual = self._uci_residual(temporal, lengths) if self.config.domain == "uci" else None
        summary = self._small_summary(temporal, mask, lengths) if self.summary_width else None
        # Autocast may produce FP16 stage-head logits while the input tensor stays
        # FP32. Keep the public logit contract explicitly FP32; the cast remains
        # differentiable and boolean assignment then has a stable dtype.
        output = torch.empty(len(temporal), device=temporal.device, dtype=torch.float32)
        for index, stage in enumerate(self.stage_names):
            chosen = stage_index == index
            if not chosen.any():
                continue
            if self.config.domain == "uci" and stage == "S0":
                features = context_rep[chosen]
                logits = self.heads[stage](features).squeeze(-1)
                if self.wide_s0 is not None:
                    logits = logits + self.wide_s0(context[chosen]).squeeze(-1)
            else:
                parts = [temporal_rep[chosen], context_rep[chosen]]
                if residual is not None:
                    parts.append(residual[chosen])
                if summary is not None:
                    parts.append(summary[chosen])
                logits = self.heads[stage](torch.cat(parts, dim=-1)).squeeze(-1)
            output[chosen] = logits.float()
        return output
