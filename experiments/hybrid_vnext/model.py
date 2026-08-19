"""One unified Hybrid prototype family (C0–C3). Topology does not fork by dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.prediction.model.components import ResidualCNNBranch, ResidualTemporalBlock, masked_mean_max


def _projector(in_dim: int, out_dim: int, dropout: float) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(out_dim, out_dim),
        nn.LayerNorm(out_dim),
    )


class ResidualCNNSequence(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilations: tuple[int, ...], dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(channels, kernel_size, dilation, dropout) for dilation in dilations
        )

    def forward(self, adapted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = adapted.transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        return x.transpose(1, 2) * mask.unsqueeze(-1).to(adapted.dtype)


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
                sequence[positive], lengths[positive].cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.lstm(packed)
            unpacked, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=timesteps)
            output[positive] = unpacked.to(output.dtype)
        return output * mask.unsqueeze(-1).to(sequence.dtype)


@dataclass(frozen=True)
class VNextConfig:
    architecture_id: str
    static_dim: int
    temporal_dim: int
    aggregate_dim: int
    summary_dim: int
    d_fuse: int = 64
    cnn_channels: int = 64
    cnn_blocks: int = 2
    cnn_kernel_size: int = 2
    cnn_dilations: tuple[int, ...] = (1, 2)
    bilstm_hidden: int = 64
    tabular_hidden: int = 96
    dropout: float = 0.25
    branch_mode: str = "full"
    entropy_floor_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.architecture_id not in {"C0", "C1", "C2", "C3"}:
            raise ValueError(self.architecture_id)
        if self.entropy_floor_coefficient < 0:
            raise ValueError("entropy_floor_coefficient")
        if self.cnn_blocks != len(self.cnn_dilations):
            raise ValueError("cnn_blocks must match dilations")
        if self.branch_mode not in {"full", "tabular"}:
            raise ValueError(self.branch_mode)

    @property
    def temporal_path(self) -> str:
        return "parallel" if self.architecture_id in {"C0", "C1"} else "serial"

    @property
    def fusion(self) -> str:
        return {
            "C0": "softmax_3way",
            "C1": "residual_add",
            "C2": "residual_add",
            "C3": "gated_residual",
        }[self.architecture_id]

    @property
    def tabular_mode(self) -> str:
        return "phase8" if self.architecture_id == "C0" else "parity"


def phase8_corrected_config(static_dim: int, temporal_dim: int, aggregate_dim: int, summary_dim: int) -> VNextConfig:
    return VNextConfig(
        architecture_id="C0",
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        aggregate_dim=aggregate_dim,
        summary_dim=summary_dim,
        d_fuse=96,
        cnn_channels=128,
        cnn_blocks=2,
        cnn_kernel_size=2,
        cnn_dilations=(1, 2),
        bilstm_hidden=128,
        tabular_hidden=96,
        dropout=0.20,
    )


def balanced_config(architecture_id: str, static_dim: int, temporal_dim: int, aggregate_dim: int, summary_dim: int) -> VNextConfig:
    return VNextConfig(
        architecture_id=architecture_id,
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        aggregate_dim=aggregate_dim,
        summary_dim=summary_dim,
        d_fuse=64,
        cnn_channels=64,
        cnn_blocks=2,
        cnn_kernel_size=2,
        cnn_dilations=(1, 2),
        bilstm_hidden=64,
        tabular_hidden=96,
        dropout=0.25,
    )


def make_c0_config(
    static_dim: int,
    temporal_dim: int,
    aggregate_dim: int,
    summary_dim: int,
    *,
    d_fuse: int = 96,
    cnn_channels: int = 128,
    bilstm_hidden: int = 128,
    dropout: float = 0.20,
    entropy_floor_coefficient: float = 0.002,
) -> VNextConfig:
    if not {d_fuse, cnn_channels, bilstm_hidden} <= {64, 96, 128}:
        raise ValueError("structural widths must be in {64,96,128}")
    return VNextConfig(
        architecture_id="C0",
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        aggregate_dim=aggregate_dim,
        summary_dim=summary_dim,
        d_fuse=d_fuse,
        cnn_channels=cnn_channels,
        cnn_blocks=2,
        cnn_kernel_size=2,
        cnn_dilations=(1, 2),
        bilstm_hidden=bilstm_hidden,
        tabular_hidden=96,
        dropout=dropout,
        entropy_floor_coefficient=entropy_floor_coefficient,
    )


def assert_c0_topology(config: VNextConfig) -> None:
    if config.architecture_id != "C0":
        raise RuntimeError("TOPOLOGY_UNLOCKED:architecture_id")
    if config.temporal_path != "parallel":
        raise RuntimeError("TOPOLOGY_UNLOCKED:temporal_path")
    if config.fusion != "softmax_3way":
        raise RuntimeError("TOPOLOGY_UNLOCKED:fusion")
    if config.cnn_blocks != 2 or config.cnn_kernel_size != 2 or tuple(config.cnn_dilations) != (1, 2):
        raise RuntimeError("TOPOLOGY_UNLOCKED:cnn_family")


def make_config(architecture_id: str, static_dim: int, temporal_dim: int, aggregate_dim: int, summary_dim: int, branch_mode: str = "full") -> VNextConfig:
    if architecture_id == "C0":
        cfg = phase8_corrected_config(static_dim, temporal_dim, aggregate_dim, summary_dim)
    else:
        cfg = balanced_config(architecture_id, static_dim, temporal_dim, aggregate_dim, summary_dim)
    if branch_mode != "full":
        payload = asdict(cfg)
        payload["branch_mode"] = branch_mode
        payload["cnn_dilations"] = tuple(payload["cnn_dilations"])
        cfg = VNextConfig(**payload)
    return cfg


class VNextHybrid(nn.Module):
    """One public prototype class. Dataset differences are input dims and masks only."""

    model_id = "hybrid_vnext_phase2"

    def __init__(self, config: VNextConfig):
        super().__init__()
        self.config = config
        d = config.d_fuse
        if config.tabular_mode == "phase8":
            self.static_projector = _projector(config.static_dim, d, config.dropout)
            self.aggregate_projector = _projector(config.aggregate_dim, d, config.dropout)
            self.parity_projector = None
            tabular_in = d
        else:
            parity_in = config.static_dim + config.aggregate_dim + config.summary_dim + 1
            self.static_projector = None
            self.aggregate_projector = None
            self.parity_projector = _projector(parity_in, config.tabular_hidden, config.dropout)
            self.tabular_to_fuse = nn.Linear(config.tabular_hidden, d)
            tabular_in = d

        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, d), nn.LayerNorm(d))
        if config.temporal_path == "parallel":
            self.cnn_projection = nn.Linear(d, config.cnn_channels)
            self.cnn = ResidualCNNBranch(config.cnn_channels, config.cnn_kernel_size, config.cnn_dilations, config.dropout)
            self.cnn_out = nn.Linear(config.cnn_channels * 2, d)
            self.bilstm = BiLSTMSequence(d, config.bilstm_hidden)
            self.lstm_out = nn.Linear(config.bilstm_hidden * 4, d)
            self.serial_cnn = None
            self.serial_lstm = None
            self.serial_out = None
            self.parallel_combine = nn.Linear(d * 2, d) if config.fusion != "softmax_3way" else None
        else:
            self.cnn_projection = nn.Linear(d, config.cnn_channels)
            self.serial_cnn = ResidualCNNSequence(config.cnn_channels, config.cnn_kernel_size, config.cnn_dilations, config.dropout)
            self.serial_lstm = BiLSTMSequence(config.cnn_channels, config.bilstm_hidden)
            self.serial_out = nn.Linear(config.bilstm_hidden * 4, d)
            self.cnn = self.cnn_out = self.bilstm = self.lstm_out = self.parallel_combine = None

        self.softmax_gate = nn.Sequential(nn.Linear(d * 3 + 3 + 1, 64), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(64, 3))
        self.residual_gate = nn.Sequential(nn.Linear(d * 2 + 2, 32), nn.GELU(), nn.Linear(32, 1))
        self.fusion_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(128, 1))
        self.last_diagnostics: dict[str, torch.Tensor] = {}
        self._last_gate_weights = None

    def _tabular(self, static, aggregate, summaries, aggregate_available, temporal_available, progress):
        agg_keep = aggregate_available.to(static.dtype).unsqueeze(-1)
        temp_keep = temporal_available.to(static.dtype).unsqueeze(-1)
        if self.config.tabular_mode == "phase8":
            return self.static_projector(static) + self.aggregate_projector(aggregate) * agg_keep
        parity = torch.cat((static, aggregate * agg_keep, summaries * temp_keep, progress.reshape(-1, 1)), dim=-1)
        return self.tabular_to_fuse(self.parity_projector(parity))

    def _temporal_parallel(self, adapted, mask, lengths):
        keep = mask.unsqueeze(-1).to(adapted.dtype)
        cnn = self.cnn_out(self.cnn(self.cnn_projection(adapted) * keep, mask))
        lstm = self.lstm_out(masked_mean_max(self.bilstm(adapted, mask, lengths), mask))
        return cnn, lstm

    def _temporal_serial(self, adapted, mask, lengths):
        keep = mask.unsqueeze(-1).to(adapted.dtype)
        local = self.serial_cnn(self.cnn_projection(adapted) * keep, mask)
        long = self.serial_lstm(local, mask, lengths)
        return self.serial_out(masked_mean_max(long, mask))

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress, summaries):
        temporal_available = lengths.gt(0)
        aggregate_available = aggregate_available.bool()
        if self.config.branch_mode == "tabular":
            temporal_available = torch.zeros_like(temporal_available)
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype)
        adapted = self.temporal_adapter(temporal) * keep
        h_tab = self._tabular(static, aggregate, summaries, aggregate_available, temporal_available, progress)
        empty = h_tab.new_zeros(h_tab.shape)
        if self.config.temporal_path == "parallel":
            h_cnn, h_lstm = self._temporal_parallel(adapted, temporal_mask, lengths)
            h_cnn = torch.where(temporal_available.unsqueeze(-1), h_cnn, empty)
            h_lstm = torch.where(temporal_available.unsqueeze(-1), h_lstm, empty)
            h_temp = self.parallel_combine(torch.cat((h_cnn, h_lstm), dim=-1)) if self.parallel_combine is not None else empty
        else:
            h_cnn = empty
            h_lstm = empty
            h_temp = self._temporal_serial(adapted, temporal_mask, lengths)
            h_temp = torch.where(temporal_available.unsqueeze(-1), h_temp, empty)

        if self.config.fusion == "softmax_3way":
            available = torch.stack(
                (torch.ones_like(temporal_available), temporal_available, temporal_available), dim=1
            )
            logits = self.softmax_gate(torch.cat((h_tab, h_cnn, h_lstm, available.to(h_tab.dtype), progress.reshape(-1, 1)), dim=-1))
            logits = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
            weights = F.softmax(logits, dim=-1)
            fused = weights[:, :1] * h_tab + weights[:, 1:2] * h_cnn + weights[:, 2:] * h_lstm
            g_temporal = (weights[:, 1] + weights[:, 2]).unsqueeze(-1)
            self._last_gate_weights = weights
        elif self.config.fusion == "residual_add":
            weights = None
            g_temporal = temporal_available.to(h_tab.dtype).unsqueeze(-1)
            fused = h_tab + g_temporal * h_temp
        else:
            weights = None
            raw = torch.sigmoid(self.residual_gate(torch.cat((h_tab, h_temp, progress.reshape(-1, 1), temporal_available.to(h_tab.dtype).unsqueeze(-1)), dim=-1)))
            g_temporal = raw * temporal_available.to(h_tab.dtype).unsqueeze(-1)
            fused = h_tab + g_temporal * h_temp

        self.last_diagnostics = {
            "g_temporal": g_temporal.detach().squeeze(-1),
            "h_tabular": h_tab.detach(),
            "h_temporal": (h_temp if self.config.fusion != "softmax_3way" else (h_cnn + h_lstm)).detach(),
            "h_cnn": h_cnn.detach(),
            "h_bilstm": h_lstm.detach(),
            "temporal_available": temporal_available.detach(),
            "aggregate_available": aggregate_available.detach(),
            "gate_weights": None if weights is None else weights.detach(),
        }
        return self.head(self.fusion_norm(fused)).squeeze(-1)

    def fusion_regularization(self) -> torch.Tensor:
        coeff = float(self.config.entropy_floor_coefficient)
        weights = getattr(self, "_last_gate_weights", None)
        if self.config.fusion != "softmax_3way" or coeff <= 0 or weights is None:
            device = next(self.parameters()).device
            return torch.zeros((), device=device)
        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1)
        available = (weights > 0).sum(-1).to(entropy.dtype).clamp_min(1)
        floor = available.log() * 0.35
        return weights.new_tensor(coeff) * (floor - entropy).clamp_min(0).mean()


def availability_unit_cases(model: VNextHybrid) -> list[dict]:
    """Required four-case semantic mapping test. No training."""
    model.eval()
    cfg = model.config
    rows = [
        {"temporal": 0, "aggregate": 0},
        {"temporal": 0, "aggregate": 1},
        {"temporal": 1, "aggregate": 0},
        {"temporal": 1, "aggregate": 1},
    ]
    batch = len(rows)
    timesteps = 6
    static = torch.randn(batch, cfg.static_dim)
    temporal = torch.randn(batch, timesteps, cfg.temporal_dim)
    mask = torch.zeros(batch, timesteps, dtype=torch.bool)
    aggregate = torch.randn(batch, cfg.aggregate_dim)
    summaries = torch.randn(batch, cfg.summary_dim)
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
        if not row["temporal"]:
            summaries[i] = 0
    temporal = temporal * mask.unsqueeze(-1)
    lengths = mask.sum(1)
    with torch.no_grad():
        model(static, temporal, mask, lengths, aggregate, agg_avail, progress, summaries)
    diag = model.last_diagnostics
    results = []
    for i, row in enumerate(rows):
        g = float(diag["g_temporal"][i])
        cnn_norm = float(diag["h_cnn"][i].norm())
        lstm_norm = float(diag["h_bilstm"][i].norm())
        temp_on = bool(row["temporal"])
        if diag["gate_weights"] is not None:
            cnn_mass = float(diag["gate_weights"][i, 1])
            lstm_mass = float(diag["gate_weights"][i, 2])
        else:
            cnn_mass = lstm_mass = g if temp_on else 0.0
        ok = True
        reasons = []
        if not temp_on and (cnn_mass > 1e-6 or lstm_mass > 1e-6 or g > 1e-6):
            ok = False
            reasons.append("temporal_mass_when_unavailable")
        if temp_on and cfg.fusion == "softmax_3way" and (cnn_mass + lstm_mass) <= 0:
            ok = False
            reasons.append("temporal_zero_when_available")
        if temp_on and not bool(row["aggregate"]) and cfg.fusion == "softmax_3way" and lstm_mass <= 0 and cnn_mass <= 0:
            ok = False
            reasons.append("bilstm_blocked_by_aggregate")
        results.append(
            {
                **row,
                "g_temporal": g,
                "cnn_mass": cnn_mass,
                "lstm_mass": lstm_mass,
                "cnn_norm": cnn_norm,
                "lstm_norm": lstm_norm,
                "pass": ok,
                "reasons": reasons,
            }
        )
    return results
