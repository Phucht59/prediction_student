"""Frozen Hybrid V1: parallel residual CNN and BiLSTM."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .components import BiLSTMBranch, ResidualCNNBranch, temporal_summaries


@dataclass(frozen=True)
class HybridConfig:
    temporal_dim: int
    context_dim: int
    d_model: int = 64
    cnn_channels: int = 64
    cnn_blocks: int = 3
    bilstm_hidden: int = 64
    bilstm_layers: int = 1
    context_hidden: int = 64
    head_hidden: int = 128
    representation: str = "R0"
    summary_hidden: int = 64
    progress_hidden: int = 16
    dropout: float = 0.20


class Hybrid(nn.Module):
    model_id = "hybrid"
    display_name = "Hybrid"

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        self.temporal_adapter = nn.Sequential(nn.Linear(config.temporal_dim, config.d_model), nn.LayerNorm(config.d_model))
        self.cnn_projection = nn.Identity() if config.d_model == config.cnn_channels else nn.Linear(config.d_model, config.cnn_channels)
        dilations = (1, 2) if config.cnn_blocks == 2 else (1, 2, 4)
        self.cnn = ResidualCNNBranch(config.cnn_channels, 2, dilations, config.dropout)
        self.bilstm = BiLSTMBranch(config.d_model, config.bilstm_hidden, config.bilstm_layers)
        advanced=config.representation in {'R1','R2','R3','R4'}
        if advanced:
            self.context_deep=nn.Sequential(nn.Linear(config.context_dim,config.context_hidden),nn.LayerNorm(config.context_hidden),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(config.context_hidden,config.context_hidden),nn.GELU(),nn.Dropout(config.dropout));self.context_residual=nn.Linear(config.context_dim,config.context_hidden);self.context_norm=nn.LayerNorm(config.context_hidden);self.wide_context=nn.Linear(config.context_dim,1)
        else:self.context=nn.Sequential(nn.Linear(config.context_dim,config.context_hidden),nn.LayerNorm(config.context_hidden),nn.GELU(),nn.Dropout(config.dropout))
        has_summary=config.representation in {'R2','R3','R4'};has_progress=config.representation in {'R3','R4'}
        if has_summary:self.summary=nn.Sequential(nn.Linear(5*config.temporal_dim+1,config.summary_hidden),nn.LayerNorm(config.summary_hidden),nn.GELU(),nn.Dropout(config.dropout))
        if has_progress:self.progress=nn.Sequential(nn.Linear(1,config.progress_hidden),nn.GELU())
        if config.representation=='R4':self.gate=nn.Linear(config.context_hidden+config.progress_hidden,4)
        fusion_dim = 2*config.cnn_channels+4*config.bilstm_hidden+config.context_hidden+(config.summary_hidden if has_summary else 0)+(config.progress_hidden if has_progress else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, config.head_hidden), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.head_hidden, 1)
        )

    def forward(self, temporal: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor, context: torch.Tensor, progress: torch.Tensor | None = None) -> torch.Tensor:
        adapted = self.temporal_adapter(temporal)
        adapted = adapted * mask.unsqueeze(-1).to(adapted.dtype)
        cnn_input = self.cnn_projection(adapted) * mask.unsqueeze(-1).to(adapted.dtype)
        cnn = self.cnn(cnn_input, mask)
        bilstm = self.bilstm(adapted, mask, lengths)
        advanced=self.config.representation in {'R1','R2','R3','R4'};context_rep=self.context_norm(self.context_deep(context)+self.context_residual(context)) if advanced else self.context(context);parts=[cnn,bilstm,context_rep]
        summary_rep=None;progress_rep=None
        if self.config.representation in {'R2','R3','R4'}:summary_rep=self.summary(temporal_summaries(temporal,mask,lengths));parts.append(summary_rep)
        if self.config.representation in {'R3','R4'}:
            if progress is None:raise ValueError('Progress scalar required for R3/R4')
            progress_rep=self.progress(progress.reshape(-1,1));parts.append(progress_rep)
        if self.config.representation=='R4':
            weights=torch.softmax(self.gate(torch.cat((context_rep,progress_rep),-1)),-1);self.last_gate_weights=weights.detach();parts[0]=parts[0]*weights[:,0:1];parts[1]=parts[1]*weights[:,1:2];parts[2]=parts[2]*weights[:,2:3];parts[3]=parts[3]*weights[:,3:4]
        logit=self.head(torch.cat(parts,-1)).squeeze(-1)
        return logit+(self.wide_context(context).squeeze(-1) if advanced else 0.)
