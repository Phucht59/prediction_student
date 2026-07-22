from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalV5(nn.Module):
    def __init__(self, input_channels: int, config: dict[str, Any], variant: str):
        super().__init__()
        channels = int(config["conv_channels"])
        hidden = int(config["lstm_hidden"])
        layers = int(config["lstm_layers"])
        dropout = float(config["dropout"])
        self.variant = variant
        self.convolutions = nn.ModuleList()
        recurrent_input = input_channels
        if variant in {"cnn_bilstm", "cnn_only"}:
            kernels = list(config.get("kernels", [3]))
            self.convolutions = nn.ModuleList(
                [nn.Conv1d(input_channels, channels, int(kernel), padding=int(kernel) // 2) for kernel in kernels]
            )
            recurrent_input = channels * len(kernels)
            self.conv_norm = nn.LayerNorm(recurrent_input)
            self.residual = nn.Linear(input_channels, recurrent_input)
        else:
            self.conv_norm = nn.Identity()
            self.residual = nn.Identity()
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
        self.dropout = nn.Dropout(dropout)
        self.pooling = str(config["pooling"])
        if self.pooling == "masked_attention":
            projection = int(config["pooling_projection"])
            self.attention = nn.Sequential(nn.Linear(output_dim, projection), nn.Tanh(), nn.Linear(projection, 1))
            pooled_dim = output_dim
        elif self.pooling == "masked_mean_max":
            self.attention = None
            pooled_dim = output_dim * 2
        else:
            raise ValueError(self.pooling)
        self.output_dim = int(config["pooling_projection"])
        self.projection = nn.Sequential(nn.Linear(pooled_dim, self.output_dim), nn.LayerNorm(self.output_dim), nn.GELU())

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        values = sequence
        if self.convolutions:
            convolved = [torch.relu(layer(values.transpose(1, 2))).transpose(1, 2) for layer in self.convolutions]
            values = self.conv_norm(torch.cat(convolved, dim=2) + self.residual(sequence))
            values = self.dropout(values) * mask.unsqueeze(-1)
        if self.recurrent is not None:
            packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.recurrent(packed)
            values, _ = pad_packed_sequence(output, batch_first=True, total_length=sequence.shape[1])
        valid = mask.bool()
        if self.attention is not None:
            scores = self.attention(values).squeeze(-1).masked_fill(~valid, float("-inf"))
            weights = torch.softmax(scores, dim=1)
            pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
        else:
            weights = None
            mean = (values * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1).to(values.device)
            maximum = values.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
            pooled = torch.cat([mean, maximum], dim=1)
        return self.projection(pooled), weights


class OULADCNNBiLSTMV5(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
        variant: str = "cnn_bilstm",
    ):
        super().__init__()
        dropout = float(config["dropout"])
        fusion_dim = int(config["fusion_hidden"])
        self.temporal = TemporalV5(sequence_channels, config, variant)
        self.aggregate = nn.Sequential(
            nn.Linear(aggregate_dim, int(config["aggregate_hidden"])),
            nn.LayerNorm(int(config["aggregate_hidden"])),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(config["aggregate_hidden"]), fusion_dim),
        )
        self.static = nn.Sequential(
            nn.Linear(static_dim, int(config["static_hidden"])),
            nn.LayerNorm(int(config["static_hidden"])),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(config["static_hidden"]), fusion_dim),
        )
        self.temporal_projection = nn.Linear(self.temporal.output_dim, fusion_dim)
        self.gates = nn.Sequential(nn.Linear(fusion_dim * 3, 3), nn.Softmax(dim=1))
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, sequence, lengths, mask, aggregate, static, return_attention: bool = False):
        temporal, attention = self.temporal(sequence, lengths, mask)
        branches = torch.stack(
            [self.temporal_projection(temporal), self.aggregate(aggregate), self.static(static)], dim=1
        )
        gate = self.gates(branches.reshape(branches.shape[0], -1)).unsqueeze(-1)
        logits = self.head((branches * gate).sum(dim=1)).squeeze(1)
        return (logits, attention, gate.squeeze(-1)) if return_attention else logits


def build_model(prepared, config: dict[str, Any], variant: str = "cnn_bilstm") -> OULADCNNBiLSTMV5:
    return OULADCNNBiLSTMV5(
        prepared.sequence.shape[2], prepared.aggregate.shape[1], prepared.static.shape[1], config, variant
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = ["OULADCNNBiLSTMV5", "TemporalV5", "build_model", "count_parameters"]

