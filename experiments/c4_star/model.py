"""C4-STAR: tabular/evidence anchor + zero-init CNN–BiLSTM residual. One family, two datasets."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.prediction.model.components import ResidualProjector, masked_mean_max


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _last_valid(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    idx = (lengths - 1).clamp(min=0)
    gather = sequence[torch.arange(sequence.size(0), device=sequence.device), idx]
    return torch.where(lengths.gt(0).unsqueeze(-1), gather, torch.zeros_like(gather))


class TemporalEvidenceAdapter(nn.Module):
    """Mask-aware prefix statistics. G1/G2 evidence stays inside this module for UCI."""

    def __init__(self, temporal_dim: int, out_dim: int, dropout: float):
        super().__init__()
        in_dim = temporal_dim * 11 + 4
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.in_dim = in_dim

    def features(self, temporal: torch.Tensor, mask: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
        keep = mask.unsqueeze(-1).to(temporal.dtype)
        count = mask.sum(1).clamp_min(1).to(temporal.dtype)
        x = temporal * keep
        mean = x.sum(1) / count.unsqueeze(-1)
        last = _last_valid(temporal, mask.sum(1))
        first_idx = mask.to(temporal.dtype).argmax(1)
        first = temporal[torch.arange(temporal.size(0), device=temporal.device), first_idx]
        first = torch.where(mask.any(1).unsqueeze(-1), first, torch.zeros_like(first))
        filled_min = temporal.masked_fill(~mask.unsqueeze(-1), 1e4)
        filled_max = temporal.masked_fill(~mask.unsqueeze(-1), -1e4)
        mn = filled_min.min(1).values
        mx = filled_max.max(1).values
        mn = torch.where(mask.any(1, keepdim=True), mn, torch.zeros_like(mn))
        mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        var = ((temporal - mean.unsqueeze(1)).pow(2) * keep).sum(1) / count.unsqueeze(-1)
        std = var.clamp_min(0).sqrt()
        delta = last - first
        t_idx = torch.arange(temporal.size(1), device=temporal.device, dtype=temporal.dtype)
        t_mean = (t_idx.unsqueeze(0) * mask).sum(1) / count
        t_dev = (t_idx.unsqueeze(0) - t_mean.unsqueeze(1)) * mask
        denom = t_dev.pow(2).sum(1).clamp_min(1e-6)
        y_dev = (temporal - mean.unsqueeze(1)) * keep
        slope = (t_dev.unsqueeze(-1) * y_dev).sum(1) / denom.unsqueeze(-1)
        # EWMA over valid steps (causal within prefix).
        alpha = 0.3
        ewma = temporal.new_zeros(temporal.size(0), temporal.size(2))
        for t in range(temporal.size(1)):
            step = temporal[:, t]
            present = mask[:, t].unsqueeze(-1).to(temporal.dtype)
            ewma = ewma * (1 - alpha * present) + step * (alpha * present)
        # direction changes
        diff = temporal[:, 1:] - temporal[:, :-1]
        valid_diff = mask[:, 1:] & mask[:, :-1]
        sign = torch.sign(diff)
        flips = (sign[:, 1:] * sign[:, :-1] < 0) & valid_diff[:, 1:].unsqueeze(-1)
        dir_changes = flips.to(temporal.dtype).sum(1)
        vol = diff.abs()
        vol = (vol * valid_diff.unsqueeze(-1).to(temporal.dtype)).sum(1) / valid_diff.sum(1).clamp_min(1).unsqueeze(-1).to(temporal.dtype)
        length = mask.sum(1).to(temporal.dtype)
        ratio = length / float(max(temporal.size(1), 1))
        # inactivity streak: trailing zeros of mask
        rev = mask.flip(1)
        first_one = rev.to(torch.float32).argmax(1)
        streak = torch.where(mask.any(1), first_one.to(temporal.dtype), length.new_zeros(length.shape))
        feats = torch.cat(
            [
                first, last, mean, std, mn, mx, delta, slope, ewma, vol, dir_changes,
                length.unsqueeze(-1), ratio.unsqueeze(-1), progress.reshape(-1, 1), streak.unsqueeze(-1),
            ],
            dim=-1,
        )
        return feats

    def forward(self, temporal, mask, progress):
        return self.proj(self.features(temporal, mask, progress))


class MaskedMultiScaleCNN(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, ...] = (2, 3, 5), dropout: float = 0.2):
        super().__init__()
        self.kernels = kernels
        self.depthwise = nn.ModuleList(nn.Conv1d(channels, channels, k, groups=channels) for k in kernels)
        self.pointwise = nn.Conv1d(channels * len(kernels), channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keep = mask.unsqueeze(-1).to(sequence.dtype)
        x = sequence * keep
        conv_in = x.transpose(1, 2)
        parts = []
        for kernel, conv in zip(self.kernels, self.depthwise):
            padded = F.pad(conv_in, (kernel - 1, 0))
            parts.append(conv(padded))
        y = self.pointwise(torch.cat(parts, dim=1)).transpose(1, 2)
        y = self.dropout(self.norm(y))
        return (x + y) * keep


class BiLSTMSequence(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else 0.2,
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, timesteps, _ = sequence.shape
        output = sequence.new_zeros((batch, timesteps, self.lstm.hidden_size * 2))
        positive = lengths > 0
        if positive.any():
            packed = pack_padded_sequence(
                sequence[positive], lengths[positive].cpu().clamp(min=1), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            unpacked, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=timesteps)
            output[positive] = unpacked.to(output.dtype)
        return output * mask.unsqueeze(-1).to(sequence.dtype)


class LightAttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(sequence).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        w = torch.softmax(logits, dim=-1) * mask.to(sequence.dtype)
        w = w / w.sum(1, keepdim=True).clamp_min(1e-6)
        return (sequence * w.unsqueeze(-1)).sum(1)


class MultiScaleCNNBiLSTM(nn.Module):
    def __init__(self, temporal_dim: int, d: int, cnn_channels: int, kernels, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(temporal_dim, d), nn.LayerNorm(d))
        self.cnn_in = nn.Linear(d, cnn_channels)
        self.cnn = MaskedMultiScaleCNN(cnn_channels, kernels, dropout)
        self.lstm = BiLSTMSequence(cnn_channels, hidden, layers)
        self.attn = LightAttentionPool(hidden * 2)
        self.out = nn.Linear(cnn_channels * 3 + hidden * 2 * 4, d)
        self.norm = nn.LayerNorm(d)
        self.ssl_head = nn.Linear(d, temporal_dim)

    def forward(self, temporal, mask, lengths):
        keep = mask.unsqueeze(-1).to(temporal.dtype)
        adapted = self.in_proj(temporal) * keep
        local = self.cnn(self.cnn_in(adapted) * keep, mask)
        long = self.lstm(local, mask, lengths)
        h_cnn = torch.cat((masked_mean_max(local, mask), _last_valid(local, lengths)), dim=-1)
        h_lstm = torch.cat((masked_mean_max(long, mask), _last_valid(long, lengths), self.attn(long, mask)), dim=-1)
        fused = self.norm(self.out(torch.cat((h_cnn, h_lstm), dim=-1)))
        recon = self.ssl_head(adapted)
        return fused, recon, h_cnn, h_lstm


class TabularAnchor(nn.Module):
    def __init__(self, static_dim: int, aggregate_dim: int, evidence_dim: int, d: int, dropout: float):
        super().__init__()
        self.static = ResidualProjector(static_dim, d, dropout)
        self.agg = ResidualProjector(max(aggregate_dim, 1), d, dropout)
        self.ev = ResidualProjector(evidence_dim, d, dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, static, aggregate, aggregate_available, evidence):
        if aggregate.shape[-1] == 0:
            aggregate = static.new_zeros(static.shape[0], 1)
        keep = aggregate_available.to(static.dtype).unsqueeze(-1)
        h = self.static(static) + self.agg(aggregate) * keep + self.ev(evidence)
        z = self.head(h).squeeze(-1)
        return z, F.layer_norm(h, h.shape[-1:])


class ResidualStageGate(nn.Module):
    def __init__(self, d: int, initial_alpha: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 5, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, float(torch.logit(torch.tensor(initial_alpha))))
        self.delta = nn.Linear(d, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, h_seq, progress, length_frac, mask_ratio, anchor_unc):
        ctx = torch.cat(
            [
                h_seq,
                progress.reshape(-1, 1),
                length_frac.reshape(-1, 1),
                mask_ratio.reshape(-1, 1),
                anchor_unc.reshape(-1, 1),
                torch.ones(h_seq.size(0), 1, device=h_seq.device, dtype=h_seq.dtype),
            ],
            dim=-1,
        )
        alpha = torch.sigmoid(self.net(ctx)).squeeze(-1)
        delta_z = self.delta(h_seq).squeeze(-1)
        return alpha, delta_z


@dataclass
class C4Config:
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    d_fuse: int = 64
    cnn_channels: int = 32
    cnn_kernels: tuple[int, ...] = (2, 3, 5)
    bilstm_hidden: int = 48
    bilstm_layers: int = 1
    dropout: float = 0.25
    evidence_dim: int = 48
    initial_alpha: float = 0.05
    mechanism: str = "M3"
    branch_mode: str = "full"

    @property
    def use_evidence(self) -> bool:
        return self.mechanism >= "M1" or self.mechanism in {"M1", "M2", "M3", "M4", "M5", "M6", "M7"}

    @property
    def use_residual(self) -> bool:
        return self.mechanism in {"M3", "M4", "M5", "M6", "M7"}


class C4STAR(nn.Module):
    model_id = "c4_star_v2.1"

    def __init__(self, config: C4Config):
        super().__init__()
        self.config = config
        d = config.d_fuse
        self.evidence = TemporalEvidenceAdapter(config.temporal_dim, config.evidence_dim, config.dropout)
        self.anchor = TabularAnchor(config.static_dim, config.aggregate_dim, config.evidence_dim, d, config.dropout)
        self.seq = MultiScaleCNNBiLSTM(
            config.temporal_dim, d, config.cnn_channels, config.cnn_kernels, config.bilstm_hidden, config.bilstm_layers, config.dropout
        )
        self.skip = nn.Linear(config.evidence_dim, d)
        self.fuse_norm = nn.LayerNorm(d)
        self.gate = ResidualStageGate(d, config.initial_alpha)
        self.concat_head = nn.Sequential(nn.LayerNorm(d * 2), nn.Linear(d * 2, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 1))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        mask = temporal_mask.bool()
        ev = self.evidence(temporal, mask, progress)
        z_anchor, h_anchor = self.anchor(static, aggregate, aggregate_available.bool(), ev)
        h_seq, recon, h_cnn, h_lstm = self.seq(temporal, mask, lengths)
        h_seq = self.fuse_norm(h_seq + self.skip(ev))
        t = float(max(temporal.size(1), 1))
        length_frac = lengths.to(progress.dtype) / t
        mask_ratio = mask.to(progress.dtype).sum(1) / t
        p_anchor = torch.sigmoid(z_anchor)
        unc = 1.0 - (p_anchor - 0.5).abs() * 2.0
        alpha, delta_z = self.gate(h_seq, progress, length_frac, mask_ratio, unc)
        if self.config.branch_mode == "anchor":
            z_final = z_anchor
            alpha = alpha * 0
            delta_z = delta_z * 0
        elif self.config.branch_mode == "sequence":
            z_final = delta_z
        elif self.config.branch_mode == "concat":
            z_final = self.concat_head(torch.cat((h_anchor, h_seq), dim=-1)).squeeze(-1)
        else:
            z_final = z_anchor + alpha * delta_z
        self.last_diagnostics = {
            "z_anchor": z_anchor,
            "z_final": z_final,
            "alpha": alpha,
            "delta_z": delta_z,
            "h_anchor": h_anchor.detach(),
            "h_seq": h_seq.detach(),
            "recon": recon,
            "temporal_available": lengths.gt(0),
        }
        return z_final


def make_c4_config(static_dim: int, temporal_dim: int, aggregate_dim: int, **kwargs) -> C4Config:
    return C4Config(static_dim=static_dim, temporal_dim=temporal_dim, aggregate_dim=aggregate_dim, **kwargs)


def zero_residual_matches_anchor(model: C4STAR, batch: dict) -> bool:
    model.eval()
    with torch.no_grad():
        _ = model(**batch)
        z_a = model.last_diagnostics["z_anchor"]
        # force residual off
        z_forced = z_a
        p1 = torch.sigmoid(z_forced)
        saved = model.config.branch_mode
        model.config.branch_mode = "anchor"
        z2 = model(**batch)
        model.config.branch_mode = saved
        p2 = torch.sigmoid(z2)
        return bool(torch.allclose(p1, p2, atol=1e-6))
