"""Residual temporal correction Hybrid for preserving a strong tabular base."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.hybrid.models.components import BiLSTMBranch, ResidualCNNBranch
from src.hybrid.phase7.model import ResidualProjector


@dataclass(frozen=True)
class ResidualTemporalConfig:
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    d_model: int = 96
    cnn_channels: int = 128
    cnn_dilations: tuple[int, ...] = (1, 2)
    cnn_kernel_size: int = 2
    bilstm_hidden: int = 128
    bilstm_layers: int = 1
    dropout: float = .20
    sample_gate: bool = True


class ResidualTemporalHybrid(nn.Module):
    """Predict a tabular base logit plus a conservatively gated temporal delta."""
    model_id = "hybrid"
    display_name = "Unified Hybrid"

    def __init__(self, config: ResidualTemporalConfig):
        super().__init__(); self.config = config; d = config.d_model
        self.static_projector = ResidualProjector(config.static_dim, d, config.dropout)
        self.aggregate_projector = ResidualProjector(config.aggregate_dim, d, config.dropout)
        self.tabular_norm = nn.LayerNorm(d)
        self.base_head = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(128, 1))
        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, d), nn.LayerNorm(d))
        self.cnn_projection = nn.Linear(d, config.cnn_channels) if d != config.cnn_channels else nn.Identity()
        self.cnn = ResidualCNNBranch(config.cnn_channels, config.cnn_kernel_size, config.cnn_dilations, config.dropout)
        self.cnn_out = nn.Linear(config.cnn_channels * 2, d)
        self.bilstm = BiLSTMBranch(d, config.bilstm_hidden, config.bilstm_layers)
        self.lstm_out = nn.Linear(config.bilstm_hidden * 4, d)
        self.temporal_fusion = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, d), nn.GELU(), nn.Dropout(config.dropout))
        self.correction_head = nn.Linear(d, 1)
        nn.init.zeros_(self.correction_head.weight); nn.init.zeros_(self.correction_head.bias)
        self.gate = (nn.Sequential(nn.Linear(d * 2 + 2, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 1)) if config.sample_gate else None)
        if self.gate is not None:
            nn.init.constant_(self.gate[-1].bias, -2.0)  # conservative initial correction

    def representations(self, static, temporal, temporal_mask, lengths, aggregate):
        hs = self.static_projector(static); ha = self.aggregate_projector(aggregate)
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype); adapted = self.temporal_adapter(temporal) * keep
        hc = self.cnn_out(self.cnn(self.cnn_projection(adapted) * keep, temporal_mask))
        hl = self.lstm_out(self.bilstm(adapted, temporal_mask, lengths))
        return hs, hc, hl, ha

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress, *, branch_mask=None):
        hs, hc, hl, ha = self.representations(static, temporal, temporal_mask, lengths, aggregate)
        tabular = hs + ha * aggregate_available.to(hs.dtype).unsqueeze(-1)
        base = self.base_head(self.tabular_norm(tabular)).squeeze(-1)
        temporal_available = lengths.gt(0)
        if branch_mask is not None:
            mask = torch.as_tensor(branch_mask, dtype=torch.bool, device=hs.device)
            if mask.ndim == 1: mask = mask.unsqueeze(0).expand(len(hs), -1)
            temporal_available &= mask[:, 1] | mask[:, 2]
        temporal_rep = self.temporal_fusion(torch.cat((hc, hl), -1)) * temporal_available.to(hs.dtype).unsqueeze(-1)
        delta = self.correction_head(temporal_rep).squeeze(-1)
        if self.gate is None:
            alpha = temporal_available.to(hs.dtype)
        else:
            alpha = torch.sigmoid(self.gate(torch.cat((tabular, temporal_rep, progress.reshape(-1, 1).to(hs.dtype), temporal_available.to(hs.dtype).reshape(-1, 1)), -1)).squeeze(-1)) * temporal_available.to(hs.dtype)
        self.last_diagnostics = {"h_static": hs.detach(), "h_cnn": hc.detach(), "h_bilstm": hl.detach(), "h_aggregate": ha.detach(),
                                 "base_logit": base.detach(), "temporal_delta": delta.detach(), "alpha": alpha.detach()}
        return base + alpha * delta

    def fusion_regularization(self) -> torch.Tensor:
        return self.correction_head.weight.new_zeros(())
