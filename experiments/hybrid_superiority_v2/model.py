"""One public Hybrid topology family. No dataset if-branches in the graph."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.prediction.model.components import ResidualCNNBranch, ResidualProjector, masked_mean_max

from .protocol import CANDIDATES


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class MaskedMultiScaleCNN(nn.Module):
    """Causal, mask-safe multi-scale Conv1D. Short T degrades to a padded residual."""

    def __init__(self, channels: int, kernels: tuple[int, ...] = (2, 3, 5), dropout: float = 0.2):
        super().__init__()
        self.kernels = kernels
        self.depthwise = nn.ModuleList(
            nn.Conv1d(channels, channels, kernel, groups=channels) for kernel in kernels
        )
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


@dataclass
class SuperiorityConfig:
    candidate: str
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    d_fuse: int = 64
    cnn_channels: int = 32
    cnn_blocks: int = 2
    cnn_kernel_size: int = 2
    cnn_dilations: tuple[int, ...] = (1, 2)
    cnn_kernels: tuple[int, ...] = (2, 3, 5)
    bilstm_hidden: int = 32
    tabular_hidden: int = 64
    dropout: float = 0.30
    branch_mode: str = "full"

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES:
            raise ValueError(self.candidate)
        if self.branch_mode not in {"full", "tabular", "temporal", "cnn", "bilstm"}:
            raise ValueError(self.branch_mode)

    @property
    def serial(self) -> bool:
        return self.candidate in {"C2-S", "C3-G"}

    @property
    def gated(self) -> bool:
        return self.candidate == "C3-G"


class SuperiorityHybrid(nn.Module):
    """Unified Hybrid. Dataset differences are input dims, masks, and fitted weights."""

    model_id = "hybrid_superiority_v2"

    def __init__(self, config: SuperiorityConfig):
        super().__init__()
        self.config = config
        d = config.d_fuse
        self.static_projector = ResidualProjector(config.static_dim, d, config.dropout)
        self.aggregate_projector = ResidualProjector(max(config.aggregate_dim, 1), d, config.dropout)
        self.tabular_head = nn.Linear(d, 1)
        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, d), nn.LayerNorm(d))
        self.cnn_in = nn.Linear(d, config.cnn_channels)
        if config.serial:
            self.cnn_seq = MaskedMultiScaleCNN(config.cnn_channels, config.cnn_kernels, config.dropout)
            self.cnn_pool = nn.Linear(config.cnn_channels * 2, d)
            self.bilstm = BiLSTMSequence(config.cnn_channels, config.bilstm_hidden)
            self.lstm_pool = nn.Linear(config.bilstm_hidden * 4, d)
            self.skip_pool = nn.Linear(d * 2, d)
            self.parallel_cnn = None
            self.parallel_lstm = None
            self.cnn_out = None
            self.lstm_out = None
        else:
            self.parallel_cnn = ResidualCNNBranch(
                config.cnn_channels, config.cnn_kernel_size, config.cnn_dilations, config.dropout
            )
            self.cnn_out = nn.Linear(config.cnn_channels * 2, d)
            self.parallel_lstm = BiLSTMSequence(d, config.bilstm_hidden)
            self.lstm_out = nn.Linear(config.bilstm_hidden * 4, d)
            self.cnn_seq = self.cnn_pool = self.bilstm = self.lstm_pool = self.skip_pool = None
        self.softmax_gate = nn.Sequential(
            nn.Linear(d * 3 + 3 + 1, 32), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(32, 3)
        )
        self.residual_gate = nn.Sequential(nn.Linear(d * 2 + 2, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.constant_(self.residual_gate[-1].bias, -2.0)
        nn.init.zeros_(self.residual_gate[-1].weight)
        self.delta_head = nn.Linear(d, 1)
        self.cnn_aux = nn.Linear(d, 1)
        self.lstm_aux = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 1))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def _tabular(self, static, aggregate, aggregate_available):
        if aggregate.shape[-1] == 0:
            aggregate = static.new_zeros(static.shape[0], 1)
        keep = aggregate_available.to(static.dtype).unsqueeze(-1)
        h_static = self.static_projector(static)
        h_agg = self.aggregate_projector(aggregate) * keep
        return h_static + h_agg

    def _temporal(self, adapted, mask, lengths):
        keep = mask.unsqueeze(-1).to(adapted.dtype)
        if self.config.serial:
            local = self.cnn_seq(self.cnn_in(adapted) * keep, mask)
            long = self.bilstm(local, mask, lengths)
            h_cnn = self.cnn_pool(masked_mean_max(local, mask))
            h_lstm = self.lstm_pool(masked_mean_max(long, mask))
            h_temp = self.skip_pool(torch.cat((h_cnn, h_lstm), dim=-1))
            return h_cnn, h_lstm, h_temp
        cnn_feat = self.parallel_cnn(self.cnn_in(adapted) * keep, mask)
        h_cnn = self.cnn_out(cnn_feat)
        h_lstm = self.lstm_out(masked_mean_max(self.parallel_lstm(adapted, mask, lengths), mask))
        h_temp = 0.5 * (h_cnn + h_lstm)
        return h_cnn, h_lstm, h_temp

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        temporal_available = lengths.gt(0)
        if self.config.branch_mode == "tabular":
            temporal_available = torch.zeros_like(temporal_available)
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype)
        adapted = self.temporal_adapter(temporal) * keep
        h_tab = self._tabular(static, aggregate, aggregate_available.bool())
        empty = h_tab.new_zeros(h_tab.shape)
        if self.config.branch_mode == "tabular":
            h_cnn = h_lstm = h_temp = empty
        else:
            h_cnn, h_lstm, h_temp = self._temporal(adapted, temporal_mask, lengths)
            if self.config.branch_mode == "cnn":
                h_lstm = empty
                h_temp = h_cnn
            elif self.config.branch_mode == "bilstm":
                h_cnn = empty
                h_temp = h_lstm
            elif self.config.branch_mode == "temporal":
                h_tab = empty
            h_cnn = torch.where(temporal_available.unsqueeze(-1), h_cnn, empty)
            h_lstm = torch.where(temporal_available.unsqueeze(-1), h_lstm, empty)
            h_temp = torch.where(temporal_available.unsqueeze(-1), h_temp, empty)

        z_tab = self.tabular_head(h_tab).squeeze(-1)
        z_cnn = self.cnn_aux(h_cnn).squeeze(-1)
        z_lstm = self.lstm_aux(h_lstm).squeeze(-1)
        delta = self.delta_head(h_temp).squeeze(-1)
        avail = temporal_available.to(h_tab.dtype)

        if self.config.candidate == "C0-R":
            available = torch.stack((torch.ones_like(temporal_available), temporal_available, temporal_available), dim=1)
            logits = self.softmax_gate(
                torch.cat((h_tab, h_cnn, h_lstm, available.to(h_tab.dtype), progress.reshape(-1, 1)), dim=-1)
            )
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1)
            fused = weights[:, :1] * h_tab + weights[:, 1:2] * h_cnn + weights[:, 2:] * h_lstm
            primary = self.head(fused).squeeze(-1)
            g = (weights[:, 1] + weights[:, 2])
        elif self.config.candidate == "C1-R":
            g_raw = torch.sigmoid(
                self.residual_gate(torch.cat((h_tab, h_temp, progress.reshape(-1, 1), avail.unsqueeze(-1)), dim=-1))
            ).squeeze(-1)
            g = g_raw * avail
            fused = h_tab + g.unsqueeze(-1) * h_temp
            primary = self.head(fused).squeeze(-1)
            weights = None
        elif self.config.candidate == "C2-S":
            g = avail
            fused = h_tab + g.unsqueeze(-1) * h_temp
            primary = self.head(fused).squeeze(-1)
            weights = None
        else:
            g_raw = torch.sigmoid(
                self.residual_gate(torch.cat((h_tab, h_temp, progress.reshape(-1, 1), avail.unsqueeze(-1)), dim=-1))
            ).squeeze(-1)
            g = g_raw * avail
            primary = z_tab + g * delta
            weights = None

        self.last_diagnostics = {
            "z_tab": z_tab,
            "z_cnn": z_cnn,
            "z_lstm": z_lstm,
            "delta": delta,
            "g": g,
            "h_tabular": h_tab.detach(),
            "h_cnn": h_cnn.detach(),
            "h_bilstm": h_lstm.detach(),
            "h_temporal": h_temp.detach(),
            "temporal_available": temporal_available.detach(),
            "gate_weights": None if weights is None else weights.detach(),
        }
        return primary


def make_config(candidate: str, static_dim: int, temporal_dim: int, aggregate_dim: int, **kwargs) -> SuperiorityConfig:
    payload = dict(candidate=candidate, static_dim=static_dim, temporal_dim=temporal_dim, aggregate_dim=aggregate_dim)
    payload.update(kwargs)
    return SuperiorityConfig(**payload)


def availability_cases(model: SuperiorityHybrid) -> list[dict]:
    model.eval()
    cfg = model.config
    rows = [
        {"temporal": 0, "aggregate": 0},
        {"temporal": 0, "aggregate": 1},
        {"temporal": 1, "aggregate": 0},
        {"temporal": 1, "aggregate": 1},
    ]
    batch = len(rows)
    t = 6
    static = torch.randn(batch, cfg.static_dim)
    temporal = torch.randn(batch, t, cfg.temporal_dim)
    mask = torch.zeros(batch, t, dtype=torch.bool)
    aggregate = torch.randn(batch, max(cfg.aggregate_dim, 1))
    agg_avail = torch.zeros(batch, dtype=torch.bool)
    progress = torch.tensor([0.0, 0.35, 0.5, 1.0])
    for i, row in enumerate(rows):
        if row["temporal"]:
            mask[i, :3] = True
        else:
            temporal[i] = 0
        agg_avail[i] = bool(row["aggregate"])
        if not row["aggregate"]:
            aggregate[i] = 0
    temporal = temporal * mask.unsqueeze(-1)
    lengths = mask.sum(1)
    with torch.no_grad():
        model(static, temporal, mask, lengths, aggregate, agg_avail, progress)
    diag = model.last_diagnostics
    out = []
    for i, row in enumerate(rows):
        g = float(diag["g"][i])
        ok = True
        reasons = []
        if not row["temporal"] and g > 1e-5:
            ok = False
            reasons.append("temporal_mass_when_unavailable")
        if row["temporal"] and cfg.candidate == "C0-R":
            weights = diag["gate_weights"]
            if weights is not None and float(weights[i, 1] + weights[i, 2]) <= 0:
                ok = False
                reasons.append("temporal_zero_when_available")
        out.append({**row, "g": g, "pass": ok, "reasons": reasons})
    return out
