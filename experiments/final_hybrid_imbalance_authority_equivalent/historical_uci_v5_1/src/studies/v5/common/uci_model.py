from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DualBranchCNNBiLSTM(nn.Module):
    """Small two-step CNN-BiLSTM with a separately preprocessed context branch."""

    def __init__(self, context_dim: int, config: dict[str, Any]):
        super().__init__()
        channels = int(config["cnn_channels"])
        kernel = int(config["kernel_size"])
        hidden = int(config["lstm_hidden"])
        context_hidden = int(config["context_hidden"])
        fusion_hidden = int(config["fusion_hidden"])
        dropout = float(config["dropout"])
        if kernel not in {1, 2}:
            raise ValueError("A two-step G1/G2 sequence permits kernel 1 or 2")
        self.conv = nn.Conv1d(1, channels, kernel_size=kernel, padding="same")
        self.conv_norm = nn.LayerNorm(channels)
        self.bilstm = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.context = nn.Sequential(
            nn.Linear(context_dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        temporal_dim = hidden * 4
        self.temporal_projection = nn.Sequential(
            nn.Linear(temporal_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
        )
        self.context_projection = nn.Linear(context_hidden, fusion_hidden)
        self.fusion = str(config.get("fusion", "gated"))
        if self.fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(fusion_hidden * 2, fusion_hidden), nn.Sigmoid())
            head_dim = fusion_hidden
        elif self.fusion == "concatenation":
            self.gate = None
            head_dim = fusion_hidden * 2
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_dim, fusion_hidden), nn.GELU())
        self.classifier = nn.Linear(fusion_hidden, 3)
        self.regressor = nn.Linear(fusion_hidden, 1)

    def encode(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (2, 1):
            raise ValueError("UCI sequence input must have shape [batch,2,1]")
        values = self.conv(sequence.float().transpose(1, 2)).transpose(1, 2)
        values = F.gelu(self.conv_norm(values))
        recurrent, _ = self.bilstm(values)
        temporal = torch.cat([recurrent.mean(dim=1), recurrent.max(dim=1).values], dim=1)
        temporal = self.temporal_projection(temporal)
        context_embedding = self.context_projection(self.context(context.float()))
        if self.gate is not None:
            gate = self.gate(torch.cat([temporal, context_embedding], dim=1))
            fused = gate * temporal + (1.0 - gate) * context_embedding
        else:
            fused = torch.cat([temporal, context_embedding], dim=1)
        return self.head(fused)

    def forward(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(sequence, context)
        return self.classifier(representation), self.regressor(representation).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = ["DualBranchCNNBiLSTM", "count_parameters"]
