from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.studies.v5_1.common.uci_model import SameLengthConv1d


def attention_entropy(weights: torch.Tensor | None, mask: torch.Tensor) -> torch.Tensor | None:
    if weights is None:
        return None
    valid = mask.float()
    entropy = -(weights.clamp_min(1e-12).log() * weights * valid).sum(dim=1)
    normalization = valid.sum(dim=1).clamp_min(2.0).log()
    return entropy / normalization


class OULADTemporalEncoderV51(nn.Module):
    def __init__(self, input_channels: int, config: dict[str, Any], variant: str = "cnn_bilstm"):
        super().__init__()
        if variant not in {"cnn_bilstm", "cnn_only", "bilstm_only"}:
            raise ValueError(f"Unknown temporal variant: {variant}")
        self.variant = variant
        projection = int(config.get("input_projection", 48))
        channels = int(config.get("conv_channels", 32))
        hidden = int(config.get("lstm_hidden", 64))
        layers = int(config.get("lstm_layers", 1))
        dropout = float(config.get("dropout", 0.2))
        dilation = int(config.get("dilation", 1))
        self.input_projection = nn.Linear(input_channels, projection)
        self.input_norm = nn.LayerNorm(projection)
        self.convolutions = nn.ModuleList()
        recurrent_input = projection
        if variant in {"cnn_bilstm", "cnn_only"}:
            kernels = [int(kernel) for kernel in config.get("kernels", [2, 3, 5])]
            if not kernels or len(kernels) > 3:
                raise ValueError("OULAD V5.1 requires one to three kernels")
            self.convolutions = nn.ModuleList(
                [
                    SameLengthConv1d(projection, channels, kernel_size=kernel, dilation=dilation)
                    for kernel in kernels
                ]
            )
            recurrent_input = channels * len(kernels)
            self.residual = nn.Linear(projection, recurrent_input)
            self.conv_norm = nn.LayerNorm(recurrent_input)
        else:
            self.residual = nn.Identity()
            self.conv_norm = nn.Identity()
        self.recurrent: nn.LSTM | None = None
        output_dim = recurrent_input
        if variant in {"cnn_bilstm", "bilstm_only"}:
            self.recurrent = nn.LSTM(
                recurrent_input,
                hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            output_dim = hidden * 2
        self.sequence_output_dim = output_dim
        self.dropout = nn.Dropout(dropout)
        self.pooling = str(config.get("pooling", "masked_mean_max"))
        pooling_projection = int(config.get("pooling_projection", 64))
        if self.pooling == "masked_attention":
            self.attention = nn.Sequential(
                nn.Linear(output_dim, pooling_projection), nn.Tanh(), nn.Linear(pooling_projection, 1)
            )
            pooled_dim = output_dim
        elif self.pooling == "masked_mean_max":
            self.attention = None
            pooled_dim = output_dim * 2
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, pooling_projection), nn.LayerNorm(pooling_projection), nn.GELU()
        )
        self.output_dim = pooling_projection

    def encode_sequence(
        self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        projected = F.gelu(self.input_norm(self.input_projection(sequence.float()))) * mask.unsqueeze(-1)
        values = projected
        if self.convolutions:
            convolved = [layer(projected.transpose(1, 2)).transpose(1, 2) for layer in self.convolutions]
            values = F.gelu(self.conv_norm(torch.cat(convolved, dim=2) + self.residual(projected)))
            values = self.dropout(values) * mask.unsqueeze(-1)
        if self.recurrent is not None:
            packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.recurrent(packed)
            values, _ = pad_packed_sequence(output, batch_first=True, total_length=sequence.shape[1])
            values *= mask.unsqueeze(-1)
        return values

    def forward(
        self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        valid = mask.bool()
        values = self.encode_sequence(sequence, lengths, mask)
        if self.attention is not None:
            scores = self.attention(values).squeeze(-1).masked_fill(~valid, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
        else:
            weights = None
            mean = (values * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(values.device).unsqueeze(1)
            maximum = values.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
            pooled = torch.cat([mean, maximum], dim=1)
        entropy = attention_entropy(weights, mask)
        return self.projection(pooled), weights, entropy


class DenseBranch(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.float())


class OULADHybridV51(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
        variant: str = "cnn_bilstm",
    ):
        super().__init__()
        dropout = float(config.get("dropout", 0.2))
        fusion_dim = int(config.get("fusion_hidden", 64))
        self.temporal = OULADTemporalEncoderV51(sequence_channels, config, variant)
        self.temporal_projection = nn.Linear(self.temporal.output_dim, fusion_dim)
        self.aggregate = DenseBranch(
            aggregate_dim, int(config.get("aggregate_hidden", 64)), fusion_dim, dropout
        )
        self.static = DenseBranch(static_dim, int(config.get("static_hidden", 32)), fusion_dim, dropout)
        self.fusion_name = str(config.get("fusion", "gated_residual"))
        self.branch_dropout = float(config.get("branch_dropout", 0.1))
        if self.fusion_name == "gated_residual":
            self.gates = nn.Sequential(nn.Linear(fusion_dim * 3, 2), nn.Sigmoid())
            head_input = fusion_dim
        elif self.fusion_name == "concatenation":
            self.gates = None
            head_input = fusion_dim * 3
        else:
            raise ValueError(f"Unknown fusion: {self.fusion_name}")
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Dropout(dropout),
            nn.Linear(head_input, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
        )

    def _drop_branch(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.branch_dropout <= 0:
            return values
        keep = torch.rand((values.shape[0], 1), device=values.device) >= self.branch_dropout
        return values * keep / (1.0 - self.branch_dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        temporal, attention, entropy = self.temporal(sequence, lengths, mask)
        temporal = self.temporal_projection(temporal)
        aggregate_embedding = self._drop_branch(self.aggregate(aggregate))
        static_embedding = self._drop_branch(self.static(static))
        gate: torch.Tensor | None = None
        if self.gates is None:
            fused = torch.cat([temporal, aggregate_embedding, static_embedding], dim=1)
        else:
            gate = self.gates(torch.cat([temporal, aggregate_embedding, static_embedding], dim=1))
            fused = temporal + gate[:, 0:1] * aggregate_embedding + gate[:, 1:2] * static_embedding
        logits = self.head(fused).squeeze(1)
        if not return_diagnostics:
            return logits
        return logits, {
            "attention": attention,
            "attention_entropy": entropy,
            "gate": gate,
            "temporal_norm": temporal.norm(dim=1),
            "aggregate_norm": aggregate_embedding.norm(dim=1),
            "static_norm": static_embedding.norm(dim=1),
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "DenseBranch",
    "OULADHybridV51",
    "OULADTemporalEncoderV51",
    "attention_entropy",
    "count_parameters",
]
