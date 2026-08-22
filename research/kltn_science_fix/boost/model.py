"""Shared BoostHybrid for UCI and OULAD. Binary head unchanged. Not serving C0."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.prediction.model.components import BiLSTMBranch, ResidualCNNBranch, ResidualProjector


def last_valid(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.sum(dim=1)
    idx = (lengths.clamp(min=1) - 1).long()
    gathered = sequence[torch.arange(sequence.size(0), device=sequence.device), idx]
    return torch.where(mask.any(dim=1).unsqueeze(-1), gathered, torch.zeros_like(gathered))


class BoostHybrid(nn.Module):
    """Same 3-way gated hybrid; kernel 3 + last-step skip + FiLM(progress)."""

    def __init__(
        self,
        *,
        static_dim: int,
        temporal_dim: int,
        aggregate_dim: int,
        dropout: float,
        entropy_floor_coefficient: float,
        d_fuse: int = 128,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 3,
        cnn_dilations: tuple[int, ...] = (1, 2),
        bilstm_hidden: int = 128,
    ):
        super().__init__()
        self.entropy_floor_coefficient = float(entropy_floor_coefficient)
        d = d_fuse
        self.static_projector = ResidualProjector(static_dim, d, dropout)
        self.aggregate_projector = ResidualProjector(aggregate_dim, d, dropout)
        self.temporal_adapter = nn.Sequential(nn.Linear(temporal_dim, d), nn.LayerNorm(d))
        self.cnn_projection = nn.Linear(d, cnn_channels) if d != cnn_channels else nn.Identity()
        self.cnn = ResidualCNNBranch(cnn_channels, cnn_kernel_size, cnn_dilations, dropout)
        self.cnn_out = nn.Linear(cnn_channels * 2, d)
        self.bilstm = BiLSTMBranch(d, bilstm_hidden, 1)
        self.lstm_out = nn.Linear(bilstm_hidden * 4, d)
        self.last_proj = nn.Linear(d, d)
        self.film = nn.Sequential(nn.Linear(1, d), nn.GELU(), nn.Linear(d, 2 * d))
        self.gate = nn.Sequential(
            nn.Linear(d * 3 + 3 + 1, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
        )
        self.fusion_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_gate_weights = None

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        h_static = self.static_projector(static)
        keep_agg = aggregate_available.to(static.dtype).unsqueeze(-1)
        h_tab = h_static + self.aggregate_projector(aggregate) * keep_agg
        empty = h_tab.new_zeros(h_tab.shape)
        temporal_available = lengths.gt(0)
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype)
        adapted = self.temporal_adapter(temporal) * keep
        h_cnn = self.cnn_out(self.cnn(self.cnn_projection(adapted) * keep, temporal_mask))
        h_lstm = self.lstm_out(self.bilstm(adapted, temporal_mask, lengths))
        h_cnn = torch.where(temporal_available.unsqueeze(-1), h_cnn, empty)
        h_lstm = torch.where(temporal_available.unsqueeze(-1), h_lstm, empty)
        h_last = self.last_proj(last_valid(adapted, temporal_mask))
        h_last = torch.where(temporal_available.unsqueeze(-1), h_last, empty)
        available = torch.stack(
            (torch.ones_like(temporal_available), temporal_available, temporal_available),
            dim=1,
        )
        logits = self.gate(
            torch.cat((h_tab, h_cnn, h_lstm, available.to(h_tab.dtype), progress.reshape(-1, 1).to(h_tab.dtype)), dim=-1)
        )
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        fused = weights[:, :1] * h_tab + weights[:, 1:2] * h_cnn + weights[:, 2:] * h_lstm
        fused = fused + 0.5 * h_last
        film = self.film(progress.reshape(-1, 1).to(fused.dtype))
        scale, shift = film.chunk(2, dim=-1)
        fused = fused * (1.0 + torch.tanh(scale)) + shift
        self._last_gate_weights = weights
        self.last_diagnostics = {
            "gate_weights": weights.detach(),
            "tabular_mass": weights[:, 0].detach(),
            "cnn_mass": weights[:, 1].detach(),
            "bilstm_mass": weights[:, 2].detach(),
        }
        return self.head(self.fusion_norm(fused)).squeeze(-1)

    def fusion_regularization(self) -> torch.Tensor:
        coeff = self.entropy_floor_coefficient
        weights = self._last_gate_weights
        if coeff <= 0 or weights is None:
            return next(self.parameters()).new_zeros(())
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        available = (weights > 0).sum(-1).to(entropy.dtype).clamp_min(1)
        floor = available.log() * 0.35
        return weights.new_tensor(coeff) * (floor - entropy).clamp_min(0).mean()
