"""Approved Phase8 CNN + BiLSTM Hybrid with mask-safe adaptive fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .components import BiLSTMBranch, ResidualCNNBranch, ResidualProjector, masked_mean_max


@dataclass(frozen=True)
class HybridConfig:
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    d_fuse: int = 96
    cnn_channels: int = 128
    cnn_blocks: int = 2
    cnn_kernel_size: int = 2
    cnn_dilations: tuple[int, ...] = (1, 2)
    bilstm_hidden: int = 128
    bilstm_layers: int = 1
    dropout: float = 0.20
    gate_hidden: int = 64
    fusion: str = "adaptive_entropy"
    entropy_floor_coefficient: float = 0.002
    branch_mode: str = "full"

    def __post_init__(self) -> None:
        if self.cnn_blocks != len(self.cnn_dilations):
            raise ValueError("cnn_blocks must equal len(cnn_dilations)")
        if self.cnn_kernel_size < 2 or not self.cnn_dilations:
            raise ValueError("CNN needs a valid kernel and at least one block")
        if self.fusion not in {"equal", "global", "adaptive", "adaptive_entropy"}:
            raise ValueError("unknown fusion")
        if self.branch_mode not in {"full", "tabular", "cnn", "bilstm", "temporal"}:
            raise ValueError("unknown branch_mode")
        if self.entropy_floor_coefficient < 0:
            raise ValueError("entropy_floor_coefficient must be nonnegative")


class Hybrid(nn.Module):
    """One shared binary architecture for UCI and OULAD fitted instances."""

    model_id = "hybrid"
    display_name = "Hybrid"

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        d = config.d_fuse
        self.static_projector = ResidualProjector(config.static_dim, d, config.dropout)
        self.aggregate_projector = ResidualProjector(config.aggregate_dim, d, config.dropout)
        self.use_cnn = config.branch_mode in {"full", "cnn", "temporal"}
        self.use_bilstm = config.branch_mode in {"full", "bilstm", "temporal"}
        if self.use_cnn or self.use_bilstm:
            self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, d), nn.LayerNorm(d))
        else:
            self.temporal_adapter = None
        if self.use_cnn:
            self.cnn_projection = nn.Linear(d, config.cnn_channels) if d != config.cnn_channels else nn.Identity()
            self.cnn = ResidualCNNBranch(config.cnn_channels, config.cnn_kernel_size, config.cnn_dilations, config.dropout)
            self.cnn_out = nn.Linear(config.cnn_channels * 2, d)
        else:
            self.cnn_projection = self.cnn = self.cnn_out = None
        if self.use_bilstm:
            self.bilstm = BiLSTMBranch(d, config.bilstm_hidden, config.bilstm_layers)
            self.lstm_out = nn.Linear(config.bilstm_hidden * 4, d)
        else:
            self.bilstm = self.lstm_out = None
        self.gate = nn.Sequential(
            nn.Linear(d * 3 + 3 + 1, config.gate_hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.gate_hidden, 3),
        )
        self.global_fusion_logits = nn.Parameter(torch.zeros(3))
        self.fusion_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(128, 1)
        )

    def representations(self, static, temporal, temporal_mask, lengths, aggregate):
        static_rep, aggregate_rep = self.static_projector(static), self.aggregate_projector(aggregate)
        if self.temporal_adapter is None:
            empty = static_rep.new_zeros(static_rep.shape)
            return static_rep, empty, empty, aggregate_rep
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype)
        adapted = self.temporal_adapter(temporal) * keep
        cnn = self.cnn_out(self.cnn(self.cnn_projection(adapted) * keep, temporal_mask)) if self.cnn is not None else static_rep.new_zeros(static_rep.shape)
        lstm = self.lstm_out(self.bilstm(adapted, temporal_mask, lengths)) if self.bilstm is not None else static_rep.new_zeros(static_rep.shape)
        return static_rep, cnn, lstm, aggregate_rep

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress, *, branch_mask=None):
        hs, hc, hl, ha = self.representations(static, temporal, temporal_mask, lengths, aggregate)
        temporal_available = lengths.gt(0)
        available = torch.stack((torch.ones_like(temporal_available), temporal_available, aggregate_available.bool()), dim=1)
        tabular = hs + ha * aggregate_available.to(hs.dtype).unsqueeze(-1)
        if self.config.branch_mode == "tabular":
            available[:, 1:] = False
        elif self.config.branch_mode == "cnn":
            available[:, 0] = False; available[:, 2] = False
        elif self.config.branch_mode == "bilstm":
            available[:, 0] = False; available[:, 1] = False
        elif self.config.branch_mode == "temporal":
            available[:, 0] = False
        if branch_mask is not None:
            override = torch.as_tensor(branch_mask, device=available.device, dtype=torch.bool)
            if override.ndim == 1:
                override = override.unsqueeze(0).expand_as(available)
            if override.shape != available.shape:
                raise ValueError("branch_mask must be [3] or [batch, 3]")
            available &= override
        if not available.any(dim=1).all():
            raise ValueError("every sample needs at least one available branch")
        if self.config.fusion == "equal":
            logits = torch.zeros((len(hs), 3), dtype=hs.dtype, device=hs.device)
        elif self.config.fusion == "global":
            logits = self.global_fusion_logits.unsqueeze(0).expand(len(hs), -1)
        else:
            logits = self.gate(torch.cat((tabular, hc, hl, available.to(hs.dtype), progress.reshape(-1, 1).to(hs.dtype)), dim=-1))
        logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        fused = weights[:, :1] * tabular + weights[:, 1:2] * hc + weights[:, 2:] * hl
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        self._last_gate_weights = weights
        self.last_diagnostics = {
            "gate_weights": weights.detach(), "gate_entropy": entropy.detach(),
            "h_static": hs.detach(), "h_cnn": hc.detach(), "h_bilstm": hl.detach(), "h_aggregate": ha.detach(),
        }
        return self.head(self.fusion_norm(fused)).squeeze(-1)

    def fusion_regularization(self) -> torch.Tensor:
        if self.config.fusion != "adaptive_entropy" or not self.config.entropy_floor_coefficient:
            return self.global_fusion_logits.new_zeros(())
        weights = self._last_gate_weights
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        available = (weights > 0).sum(-1).to(entropy.dtype).clamp_min(1)
        floor = available.log() * 0.35
        return self.config.entropy_floor_coefficient * (floor - entropy).clamp_min(0).mean()


__all__ = ["Hybrid", "HybridConfig"]
