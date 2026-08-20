"""Thesis-final Hybrid CNN–BiLSTM, one architecture for UCI and OULAD."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .components import BiLSTMBranch, ResidualCNNBranch, ResidualProjector


@dataclass(frozen=True)
class HybridConfig:
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    d_fuse: int = 128
    cnn_channels: int = 64
    cnn_blocks: int = 2
    cnn_kernel_size: int = 2
    cnn_dilations: tuple[int, ...] = (1, 2)
    bilstm_hidden: int = 128
    bilstm_layers: int = 1
    dropout: float = 0.25
    gate_hidden: int = 64
    fusion: str = "softmax_3way"
    entropy_floor_coefficient: float = 0.002
    architecture_id: str = "C0"

    def __post_init__(self) -> None:
        if self.architecture_id != "C0":
            raise ValueError("only architecture_id='C0' is active")
        if self.cnn_blocks != len(self.cnn_dilations):
            raise ValueError("cnn_blocks must equal len(cnn_dilations)")
        if self.cnn_kernel_size < 2 or not self.cnn_dilations:
            raise ValueError("CNN needs a valid kernel and at least one block")
        if self.fusion not in {"softmax_3way", "adaptive_entropy"}:
            raise ValueError("unknown fusion")
        if self.entropy_floor_coefficient < 0:
            raise ValueError("entropy_floor_coefficient must be nonnegative")


class Hybrid(nn.Module):
    """One public binary Hybrid. Dataset differences are input dims and weights only."""

    model_id = "hybrid"
    display_name = "Hybrid"
    architecture_id = "C0"

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        d = config.d_fuse
        self.static_projector = ResidualProjector(config.static_dim, d, config.dropout)
        self.aggregate_projector = ResidualProjector(config.aggregate_dim, d, config.dropout)
        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, d), nn.LayerNorm(d))
        self.cnn_projection = nn.Linear(d, config.cnn_channels) if d != config.cnn_channels else nn.Identity()
        self.cnn = ResidualCNNBranch(config.cnn_channels, config.cnn_kernel_size, config.cnn_dilations, config.dropout)
        self.cnn_out = nn.Linear(config.cnn_channels * 2, d)
        self.bilstm = BiLSTMBranch(d, config.bilstm_hidden, config.bilstm_layers)
        self.lstm_out = nn.Linear(config.bilstm_hidden * 4, d)
        self.gate = nn.Sequential(
            nn.Linear(d * 3 + 3 + 1, config.gate_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_hidden, 3),
        )
        self.fusion_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 1),
        )
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_gate_weights = None

    def representations(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available):
        h_static = self.static_projector(static)
        keep_agg = aggregate_available.to(static.dtype).unsqueeze(-1)
        h_aggregate = self.aggregate_projector(aggregate) * keep_agg
        h_tabular = h_static + h_aggregate
        empty = h_tabular.new_zeros(h_tabular.shape)
        temporal_available = lengths.gt(0)
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype)
        adapted = self.temporal_adapter(temporal) * keep
        h_cnn = self.cnn_out(self.cnn(self.cnn_projection(adapted) * keep, temporal_mask))
        h_lstm = self.lstm_out(self.bilstm(adapted, temporal_mask, lengths))
        h_cnn = torch.where(temporal_available.unsqueeze(-1), h_cnn, empty)
        h_lstm = torch.where(temporal_available.unsqueeze(-1), h_lstm, empty)
        return h_tabular, h_cnn, h_lstm, temporal_available

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        h_tab, h_cnn, h_lstm, temporal_available = self.representations(
            static, temporal, temporal_mask, lengths, aggregate, aggregate_available
        )
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
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        self._last_gate_weights = weights
        self.last_diagnostics = {
            "gate_weights": weights.detach(),
            "gate_entropy": entropy.detach(),
            "h_tabular": h_tab.detach(),
            "h_cnn": h_cnn.detach(),
            "h_bilstm": h_lstm.detach(),
            "temporal_available": temporal_available.detach(),
            "aggregate_available": aggregate_available.detach().bool(),
            "tabular_mass": weights[:, 0].detach(),
            "cnn_mass": weights[:, 1].detach(),
            "bilstm_mass": weights[:, 2].detach(),
        }
        return self.head(self.fusion_norm(fused)).squeeze(-1)

    def fusion_regularization(self) -> torch.Tensor:
        coeff = float(self.config.entropy_floor_coefficient)
        weights = self._last_gate_weights
        if coeff <= 0 or weights is None:
            return next(self.parameters()).new_zeros(())
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        available = (weights > 0).sum(-1).to(entropy.dtype).clamp_min(1)
        floor = available.log() * 0.35
        return weights.new_tensor(coeff) * (floor - entropy).clamp_min(0).mean()


__all__ = ["Hybrid", "HybridConfig"]
