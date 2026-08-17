"""The single Phase 7 residual-fusion Hybrid topology for both domains."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from src.hybrid.models.components import BiLSTMBranch, ResidualCNNBranch, masked_max, masked_mean


class ResidualProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__(); self.shortcut = nn.Linear(input_dim, output_dim); self.deep = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(output_dim, output_dim)); self.norm = nn.LayerNorm(output_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.norm(self.shortcut(x) + self.deep(x))


@dataclass(frozen=True)
class UnifiedHybridConfig:
    static_dim: int; temporal_dim: int; aggregate_dim: int
    d_fuse: int = 96; cnn_channels: int = 128; cnn_blocks: int = 2; bilstm_hidden: int = 128; bilstm_layers: int = 1; interaction_hidden: int = 61; dropout: float = .20
    use_last_state: bool = True; use_progress: bool = True; use_interaction: bool = True


class UnifiedHybrid(nn.Module):
    """One shared head and topology; dataset-specific variation is input shape only."""
    model_id = "hybrid"; display_name = "Unified Hybrid"
    def __init__(self, config: UnifiedHybridConfig):
        super().__init__(); self.config = config; d = config.d_fuse
        self.static_projector = ResidualProjector(config.static_dim, d, config.dropout)
        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, d), nn.LayerNorm(d))
        self.cnn_projection = nn.Linear(d, config.cnn_channels) if d != config.cnn_channels else nn.Identity()
        self.cnn = ResidualCNNBranch(config.cnn_channels, 2, (1, 2), config.dropout)
        self.cnn_out = nn.Linear(config.cnn_channels * 2, d)
        self.bilstm = BiLSTMBranch(d, config.bilstm_hidden, config.bilstm_layers)
        self.lstm_out = nn.Linear(config.bilstm_hidden * (6 if config.use_last_state else 4), d)
        self.aggregate_projector = ResidualProjector(config.aggregate_dim, d, config.dropout)
        interaction_width = d * 4 + 4 + (1 if config.use_progress else 0)
        self.interaction = nn.Sequential(nn.Linear(interaction_width, config.interaction_hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.interaction_hidden, d)) if config.use_interaction else None
        self.fusion_norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(128, 1))

    def representations(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available):
        keep = temporal_mask.unsqueeze(-1).to(temporal.dtype); adapted = self.temporal_adapter(temporal) * keep
        cnn = self.cnn_out(self.cnn(self.cnn_projection(adapted) * keep, temporal_mask))
        lstm_raw = self.bilstm(adapted, temporal_mask, lengths)
        if self.config.use_last_state:
            safe = (lengths - 1).clamp_min(0); batch = torch.arange(len(lengths), device=lengths.device)
            # BiLSTMBranch's packed output is recovered safely by a second packed branch pass.
            # Last valid state is reconstructed from a masked recurrent output exposed here.
            recurrent = self.bilstm.lstm
            out = adapted.new_zeros((len(lengths), adapted.shape[1], recurrent.hidden_size * 2)); positive = lengths > 0
            if positive.any():
                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
                packed = pack_padded_sequence(adapted[positive], lengths[positive].cpu(), batch_first=True, enforce_sorted=False)
                unpacked, _ = pad_packed_sequence(recurrent(packed)[0], batch_first=True, total_length=adapted.shape[1]); out[positive] = unpacked
            last = out[batch, safe] * positive.unsqueeze(-1); lstm_raw = torch.cat((lstm_raw, last), -1)
        lstm = self.lstm_out(lstm_raw); static_rep = self.static_projector(static); aggregate_rep = self.aggregate_projector(aggregate)
        return static_rep, cnn, lstm, aggregate_rep

    def forward(self, static, temporal, temporal_mask, lengths, aggregate, aggregate_available, progress):
        hs, hc, hl, ha = self.representations(static, temporal, temporal_mask, lengths, aggregate, aggregate_available)
        temporal_available = (lengths > 0).to(hs.dtype); available = torch.stack((torch.ones_like(temporal_available), temporal_available, temporal_available, aggregate_available.to(hs.dtype)), -1)
        stacked = torch.stack((hs, hc, hl, ha), 1); base = (stacked * available.unsqueeze(-1)).sum(1) / available.sum(1, keepdim=True).clamp_min(1.)
        interaction_input = [hs, hc, hl, ha, available]
        if self.config.use_progress: interaction_input.append(progress.reshape(-1, 1).to(hs.dtype))
        correction = self.interaction(torch.cat(interaction_input, -1)) if self.interaction is not None else torch.zeros_like(base)
        self.last_diagnostics = {"h_static": hs.detach(), "h_cnn": hc.detach(), "h_bilstm": hl.detach(), "h_aggregate": ha.detach(), "base": base.detach(), "interaction": correction.detach()}
        return self.head(self.fusion_norm(base + correction)).squeeze(-1)
