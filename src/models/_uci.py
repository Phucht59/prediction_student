from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class SameLengthConv1d(nn.Conv1d):
    """Conv1D with explicit asymmetric padding for even kernels."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        total = self.dilation[0] * (self.kernel_size[0] - 1)
        left = total // 2
        right = total - left
        return F.conv1d(
            F.pad(values, (left, right)),
            self.weight,
            self.bias,
            self.stride,
            0,
            self.dilation,
            self.groups,
        )


class _UCITemporalEncoder(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, Any]):
        super().__init__()
        projection_dim = int(config.get("input_projection", 24))
        channels = int(config.get("cnn_channels", 24))
        hidden = int(config.get("lstm_hidden", 32))
        layers = int(config.get("lstm_layers", 1))
        dropout = float(config.get("dropout", 0.15))
        activation = str(config.get("activation", "gelu")).lower()
        self.variant = str(config.get("temporal_variant", "cnn_bilstm"))
        if self.variant not in {"cnn_bilstm", "cnn_only", "bilstm_only"}:
            raise ValueError(f"Unknown UCI temporal variant: {self.variant}")
        self.input_projection = nn.Linear(input_dim, projection_dim)
        self.input_norm = nn.LayerNorm(projection_dim)
        self.convolutions = nn.ModuleList()
        recurrent_input = projection_dim
        if self.variant in {"cnn_bilstm", "cnn_only"}:
            self.convolutions = nn.ModuleList(
                [SameLengthConv1d(projection_dim, channels, kernel_size=kernel) for kernel in (1, 2)]
            )
            recurrent_input = channels * len(self.convolutions)
            self.residual = nn.Linear(projection_dim, recurrent_input)
            self.conv_norm = nn.LayerNorm(recurrent_input)
        else:
            self.residual = nn.Identity()
            self.conv_norm = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu
        if activation not in {"gelu", "relu"}:
            raise ValueError(f"Unknown activation: {activation}")
        self.recurrent: nn.LSTM | None = None
        sequence_output = recurrent_input
        if self.variant in {"cnn_bilstm", "bilstm_only"}:
            self.recurrent = nn.LSTM(
                recurrent_input,
                hidden,
                num_layers=layers,
                dropout=dropout if layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            sequence_output = hidden * 2
        self.output_dim = sequence_output * 2

    def forward(
        self, temporal: torch.Tensor, availability_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if temporal.ndim != 3 or temporal.shape[1] != 2:
            raise ValueError(
                "UCI temporal input must have shape [batch,2,channels]"
            )
        if availability_mask is None:
            # Preserve the frozen official forward path byte-for-byte.
            projected = self.activation(
                self.input_norm(self.input_projection(temporal.float()))
            )
            values = projected
            if self.convolutions:
                convolved = [
                    layer(projected.transpose(1, 2)).transpose(1, 2)
                    for layer in self.convolutions
                ]
                values = self.activation(
                    self.conv_norm(
                        torch.cat(convolved, dim=2) + self.residual(projected)
                    )
                )
            if self.recurrent is not None:
                values, _ = self.recurrent(self.dropout(values))
            return torch.cat(
                [values.mean(dim=1), values.max(dim=1).values], dim=1
            )
        if availability_mask.shape != temporal.shape[:2]:
            raise ValueError("availability mask must have shape [batch,2]")
        mask = availability_mask.to(device=temporal.device, dtype=temporal.dtype)
        temporal = temporal * mask.unsqueeze(-1)
        projected = self.activation(self.input_norm(self.input_projection(temporal.float())))
        projected = projected * mask.unsqueeze(-1)
        values = projected
        if self.convolutions:
            convolved = [layer(projected.transpose(1, 2)).transpose(1, 2) for layer in self.convolutions]
            values = self.activation(self.conv_norm(torch.cat(convolved, dim=2) + self.residual(projected)))
            values = values * mask.unsqueeze(-1)
        if self.recurrent is not None:
            # A bidirectional LSTM must never inspect an unavailable future
            # timestep. Post-hoc multiplication is insufficient because its
            # backward state has already consumed padding. Pack only the
            # positive-length rows and scatter them back; zero-length S0 rows
            # bypass the recurrent network entirely. The no-mask official
            # path above remains byte-for-byte unchanged.
            lengths = mask.sum(dim=1).to(dtype=torch.long)
            recurrent_values = torch.zeros(
                values.shape[0],
                values.shape[1],
                self.recurrent.hidden_size * 2,
                dtype=values.dtype,
                device=values.device,
            )
            positive = lengths > 0
            if positive.any():
                selected = self.dropout(values[positive])
                selected_lengths = lengths[positive].cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    selected,
                    selected_lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )
                encoded, _ = self.recurrent(packed)
                unpacked, _ = nn.utils.rnn.pad_packed_sequence(
                    encoded,
                    batch_first=True,
                    total_length=temporal.shape[1],
                )
                recurrent_values[positive] = unpacked
            values = recurrent_values * mask.unsqueeze(-1)
        valid = mask.unsqueeze(-1)
        counts = valid.sum(dim=1).clamp_min(1.0)
        mean = (values * valid).sum(dim=1) / counts
        maximum = values.masked_fill(valid == 0, float("-inf")).max(dim=1).values
        all_masked = mask.sum(dim=1) == 0
        maximum = torch.where(all_masked.unsqueeze(1), torch.zeros_like(maximum), maximum)
        pooled = torch.cat([mean, maximum], dim=1)
        return torch.where(all_masked.unsqueeze(1), torch.zeros_like(pooled), pooled)


class _UCIContextEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_dim
        for _ in range(layers):
            blocks.extend([nn.Linear(current, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)])
            current = hidden
        self.network = nn.Sequential(*blocks)
        self.output_dim = hidden

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context.float())


class UCICNNBiLSTM(nn.Module):
    """Registered UCI temporal-context hybrid with auditable fusion diagnostics."""

    def __init__(self, temporal_dim: int, context_dim: int, config: dict[str, Any]):
        super().__init__()
        dropout = float(config.get("dropout", 0.15))
        fusion_dim = int(config.get("fusion_hidden", 32))
        context_hidden = int(config.get("context_hidden", 32))
        context_layers = int(config.get("context_layers", 1))
        self.temporal = _UCITemporalEncoder(temporal_dim, config)
        self.context = _UCIContextEncoder(context_dim, context_hidden, context_layers, dropout)
        self.temporal_projection = nn.Linear(self.temporal.output_dim, fusion_dim)
        self.context_projection = nn.Linear(self.context.output_dim, fusion_dim)
        self.fusion_name = str(config.get("fusion", "gated"))
        self.gate: nn.Module | None = None
        self.film: nn.Module | None = None
        if self.fusion_name == "concatenation":
            head_input = fusion_dim * 2
        elif self.fusion_name == "gated":
            self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
            head_input = fusion_dim
        elif self.fusion_name == "film_residual":
            self.film = nn.Linear(fusion_dim, fusion_dim * 2)
            self.gate = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Sigmoid())
            head_input = fusion_dim
        else:
            raise ValueError(f"Unknown fusion: {self.fusion_name}")
        self.head = nn.Sequential(
            nn.LayerNorm(head_input),
            nn.Dropout(dropout),
            nn.Linear(head_input, fusion_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(fusion_dim, 3)
        self.regressor = nn.Linear(fusion_dim, 1)
        self.ordinal = nn.Linear(fusion_dim, 2)

    def encode(
        self,
        temporal: torch.Tensor,
        context: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        temporal_embedding = self.temporal_projection(
            self.temporal(temporal, availability_mask)
        )
        if availability_mask is not None:
            has_temporal = (availability_mask.sum(dim=1, keepdim=True) > 0).to(
                temporal_embedding.dtype
            )
            temporal_embedding = temporal_embedding * has_temporal
        context_embedding = self.context_projection(self.context(context))
        diagnostics: dict[str, torch.Tensor] = {
            "temporal_norm": temporal_embedding.norm(dim=1),
            "context_norm": context_embedding.norm(dim=1),
        }
        if self.fusion_name == "concatenation":
            fused = torch.cat([temporal_embedding, context_embedding], dim=1)
        elif self.fusion_name == "gated":
            assert self.gate is not None
            gate = self.gate(torch.cat([temporal_embedding, context_embedding], dim=1))
            fused = gate * temporal_embedding + (1.0 - gate) * context_embedding
            diagnostics["gate"] = gate
        else:
            assert self.film is not None and self.gate is not None
            gamma, beta = self.film(context_embedding).chunk(2, dim=1)
            conditioned = temporal_embedding * (1.0 + 0.5 * torch.tanh(gamma)) + beta
            gate = self.gate(torch.cat([temporal_embedding, context_embedding], dim=1))
            fused = temporal_embedding + gate * (conditioned + context_embedding)
            diagnostics.update({"gate": gate, "film_gamma": gamma, "film_beta": beta})
        return self.head(fused), diagnostics

    def forward(
        self,
        temporal: torch.Tensor,
        context: torch.Tensor,
        availability_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        representation, diagnostics = self.encode(
            temporal, context, availability_mask
        )
        return {
            "classification": self.classifier(representation),
            "regression": self.regressor(representation).squeeze(1),
            "ordinal": self.ordinal(representation),
            **diagnostics,
        }


def gate_statistics(gate: torch.Tensor | None, *, saturation_threshold: float = 0.05) -> dict[str, float | bool | None]:
    if gate is None:
        return {"mean": None, "variance": None, "saturation_fraction": None, "collapsed": False}
    values = gate.detach().float()
    saturation = ((values <= saturation_threshold) | (values >= 1.0 - saturation_threshold)).float().mean()
    variance = values.var(unbiased=False)
    return {
        "mean": float(values.mean().cpu()),
        "variance": float(variance.cpu()),
        "saturation_fraction": float(saturation.cpu()),
        "collapsed": bool(float(saturation.cpu()) >= 0.95 or float(variance.cpu()) < 1e-6),
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


__all__ = [
    "SameLengthConv1d",
    "_UCIContextEncoder",
    "UCICNNBiLSTM",
    "_UCITemporalEncoder",
    "count_parameters",
    "gate_statistics",
]
