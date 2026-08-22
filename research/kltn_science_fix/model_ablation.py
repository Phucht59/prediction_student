"""Ablation wrapper around serving Hybrid. Topology stays C0."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.prediction.model.hybrid import Hybrid, HybridConfig

ABLATION_ARMS = (
    "full",
    "tabular_only",
    "cnn_only",
    "bilstm_only",
    "concat",
    "no_aggregate",
)
GRADE_ARMS = ("both", "temporal_only", "aggregate_only")


class AblationHybrid(Hybrid):
    def __init__(self, config: HybridConfig, ablation: str = "full"):
        if ablation not in ABLATION_ARMS:
            raise ValueError(ablation)
        super().__init__(config)
        self.ablation = ablation
        if ablation == "concat":
            d = config.d_fuse
            self.concat_proj = nn.Sequential(nn.LayerNorm(d * 3), nn.Linear(d * 3, d), nn.GELU())

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        if self.ablation == "no_aggregate":
            aggregate_available = torch.zeros_like(aggregate_available)
        h_tab, h_cnn, h_lstm, temporal_available = self.representations(
            static, temporal, temporal_mask, lengths, aggregate, aggregate_available
        )
        if self.ablation == "tabular_only":
            fused = h_tab
            weights = F.one_hot(torch.zeros(len(h_tab), dtype=torch.long, device=h_tab.device), 3).to(h_tab.dtype)
        elif self.ablation == "cnn_only":
            fused = h_cnn
            weights = F.one_hot(torch.ones(len(h_tab), dtype=torch.long, device=h_tab.device), 3).to(h_tab.dtype)
        elif self.ablation == "bilstm_only":
            fused = h_lstm
            weights = F.one_hot(torch.full((len(h_tab),), 2, dtype=torch.long, device=h_tab.device), 3).to(h_tab.dtype)
        elif self.ablation == "concat":
            fused = self.concat_proj(torch.cat((h_tab, h_cnn, h_lstm), dim=-1))
            weights = torch.full((len(h_tab), 3), 1.0 / 3.0, device=h_tab.device, dtype=h_tab.dtype)
        else:
            available = torch.stack(
                (torch.ones_like(temporal_available), temporal_available, temporal_available),
                dim=1,
            )
            logits = self.gate(
                torch.cat(
                    (h_tab, h_cnn, h_lstm, available.to(h_tab.dtype), progress.reshape(-1, 1).to(h_tab.dtype)),
                    dim=-1,
                )
            )
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1)
            fused = weights[:, :1] * h_tab + weights[:, 1:2] * h_cnn + weights[:, 2:] * h_lstm
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        self._last_gate_weights = weights
        self.last_diagnostics = {
            "gate_weights": weights.detach(),
            "gate_entropy": entropy.detach(),
            "tabular_mass": weights[:, 0].detach(),
            "cnn_mass": weights[:, 1].detach(),
            "bilstm_mass": weights[:, 2].detach(),
            "temporal_available": temporal_available.detach(),
        }
        return self.head(self.fusion_norm(fused)).squeeze(-1)
