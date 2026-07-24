from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.studies.v5.common.metrics import expected_calibration_error
from src.studies.v5_1.common.uci_model import SameLengthConv1d
from src.studies.v5_1.oulad.models import DenseBranch


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    phase: str
    temporal: str
    branches: str
    conv_channels: int = 24
    kernels: tuple[int, ...] = (2, 3)
    dilations: tuple[int, ...] = (2,)
    branch_dropout: float = 0.0

    def resolved_dilations(self) -> tuple[int, ...]:
        if len(self.dilations) == 1:
            return self.dilations * len(self.kernels)
        if len(self.dilations) != len(self.kernels):
            raise ValueError(f"{self.candidate_id}: kernel/dilation length mismatch")
        return self.dilations


def candidate_specs() -> dict[str, CandidateSpec]:
    rows = [
        CandidateSpec("A0_aggregate_static_only", "A", "none", "aggregate_static"),
        CandidateSpec("A1_cnn_small_temporal", "A", "cnn", "temporal_only"),
        CandidateSpec("A2_bilstm_current_temporal", "A", "bilstm", "temporal_only"),
        CandidateSpec("A3_serial_current_temporal", "A", "serial", "temporal_only"),
        CandidateSpec("A4_serial_current_full", "A", "serial", "full"),
        CandidateSpec(
            "B2_cnn_matched_temporal",
            "B",
            "cnn",
            "temporal_only",
            conv_channels=72,
            kernels=(2, 3, 5),
            dilations=(1,),
        ),
        CandidateSpec(
            "C1_cnn_d1_temporal", "C", "cnn", "temporal_only", dilations=(1,)
        ),
        CandidateSpec(
            "C3_cnn_multidilation_temporal",
            "C",
            "cnn",
            "temporal_only",
            kernels=(2, 3, 3),
            dilations=(1, 1, 2),
        ),
        CandidateSpec("D_serial_with_cnn_skip", "D", "serial_skip", "full"),
        CandidateSpec("E_parallel_concat", "E", "parallel_concat", "full"),
    ]
    return {row.candidate_id: row for row in rows}


def _masked_pool(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask.unsqueeze(-1)).sum(dim=1) / denominator
    maximum = values.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
    return torch.cat([mean, maximum], dim=1)


class MultiScaleCNN(nn.Module):
    def __init__(
        self,
        projection: int,
        channels: int,
        kernels: Iterable[int],
        dilations: Iterable[int],
        pooling_projection: int,
        dropout: float,
        pool: bool = True,
    ):
        super().__init__()
        pairs = list(zip(kernels, dilations, strict=True))
        self.convolutions = nn.ModuleList(
            [
                SameLengthConv1d(
                    projection, channels, kernel_size=int(kernel), dilation=int(dilation)
                )
                for kernel, dilation in pairs
            ]
        )
        merged = channels * len(pairs)
        self.residual = nn.Linear(projection, merged)
        self.norm = nn.LayerNorm(merged)
        self.dropout = nn.Dropout(dropout)
        self.pool_projection: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(merged * 2, pooling_projection),
                nn.LayerNorm(pooling_projection),
                nn.GELU(),
            )
            if pool
            else None
        )
        self.sequence_dim = merged
        self.output_dim = pooling_projection

    def sequence(self, projected: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        convolved = [
            layer(projected.transpose(1, 2)).transpose(1, 2)
            for layer in self.convolutions
        ]
        values = F.gelu(self.norm(torch.cat(convolved, dim=2) + self.residual(projected)))
        return self.dropout(values) * mask.unsqueeze(-1)

    def pool(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool_projection is None:
            raise RuntimeError("CNN pooling was not registered for this serial-only encoder")
        return self.pool_projection(_masked_pool(values, mask))


class BiLSTMExpert(nn.Module):
    def __init__(
        self, input_dim: int, hidden: int, layers: int, pooling_projection: int, dropout: float
    ):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.pool_projection = nn.Sequential(
            nn.Linear(hidden * 4, pooling_projection),
            nn.LayerNorm(pooling_projection),
            nn.GELU(),
        )
        self.output_dim = pooling_projection

    def sequence(
        self, values: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        packed = pack_padded_sequence(
            values, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.recurrent(packed)
        unpacked, _ = pad_packed_sequence(
            output, batch_first=True, total_length=values.shape[1]
        )
        return unpacked * mask.unsqueeze(-1)

    def pool(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.pool_projection(_masked_pool(values, mask))


class OULADArchitectureDiagnosisNet(nn.Module):
    def __init__(
        self,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        config: dict[str, Any],
        spec: CandidateSpec,
    ):
        super().__init__()
        self.spec = spec
        projection = int(config["input_projection"])
        pooling_projection = int(config["pooling_projection"])
        fusion_dim = int(config["fusion_hidden"])
        dropout = float(config["dropout"])
        self.input_projection: nn.Linear | None = (
            None if spec.temporal == "none" else nn.Linear(sequence_channels, projection)
        )
        self.input_norm: nn.LayerNorm | None = (
            None if spec.temporal == "none" else nn.LayerNorm(projection)
        )

        uses_cnn = spec.temporal in {"cnn", "serial", "serial_skip", "parallel_concat"}
        uses_lstm = spec.temporal in {"bilstm", "serial", "serial_skip", "parallel_concat"}
        self.cnn: MultiScaleCNN | None = None
        self.lstm: BiLSTMExpert | None = None
        if uses_cnn:
            self.cnn = MultiScaleCNN(
                projection,
                spec.conv_channels,
                spec.kernels,
                spec.resolved_dilations(),
                pooling_projection,
                dropout,
                pool=spec.temporal != "serial",
            )
        if uses_lstm:
            recurrent_input = (
                self.cnn.sequence_dim if spec.temporal in {"serial", "serial_skip"} else projection
            )
            self.lstm = BiLSTMExpert(
                recurrent_input,
                int(config["lstm_hidden"]),
                int(config["lstm_layers"]),
                pooling_projection,
                dropout,
            )

        self.temporal_fusion: nn.Module | None = None
        if spec.temporal in {"serial_skip", "parallel_concat"}:
            self.temporal_fusion = nn.Sequential(
                nn.Linear(pooling_projection * 2, pooling_projection),
                nn.LayerNorm(pooling_projection),
                nn.GELU(),
            )
        self.temporal_projection: nn.Linear | None = (
            nn.Linear(pooling_projection, fusion_dim)
            if spec.branches in {"temporal_only", "full"}
            else None
        )
        uses_compact = spec.branches in {"aggregate_static", "full"}
        self.aggregate: DenseBranch | None = (
            DenseBranch(
                aggregate_dim, int(config["aggregate_hidden"]), fusion_dim, dropout
            )
            if uses_compact
            else None
        )
        self.static: DenseBranch | None = (
            DenseBranch(static_dim, int(config["static_hidden"]), fusion_dim, dropout)
            if uses_compact
            else None
        )
        self.aggregate_static_fusion: nn.Sequential | None = (
            nn.Sequential(
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
            )
            if spec.branches == "aggregate_static"
            else None
        )
        self.full_gates: nn.Sequential | None = (
            nn.Sequential(nn.Linear(fusion_dim * 3, 2), nn.Sigmoid())
            if spec.branches == "full"
            else None
        )
        self.branch_dropout = float(spec.branch_dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 1),
        )

    def _drop_branch(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.branch_dropout <= 0:
            return values
        keep = torch.rand((values.shape[0], 1), device=values.device) >= self.branch_dropout
        return values * keep / (1.0 - self.branch_dropout)

    def _temporal(
        self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor | None]]:
        if self.spec.temporal == "none":
            return None, {"cnn": None, "lstm": None}
        assert self.input_projection is not None and self.input_norm is not None
        projected = F.gelu(self.input_norm(self.input_projection(sequence.float())))
        projected = projected * mask.unsqueeze(-1)
        z_cnn: torch.Tensor | None = None
        z_lstm: torch.Tensor | None = None
        cnn_sequence: torch.Tensor | None = None
        if self.cnn is not None:
            cnn_sequence = self.cnn.sequence(projected, mask)
            if self.spec.temporal != "serial":
                z_cnn = self.cnn.pool(cnn_sequence, mask)
        if self.lstm is not None:
            recurrent_input = (
                cnn_sequence
                if self.spec.temporal in {"serial", "serial_skip"}
                else projected
            )
            assert recurrent_input is not None
            lstm_sequence = self.lstm.sequence(recurrent_input, lengths, mask)
            z_lstm = self.lstm.pool(lstm_sequence, mask)
        if self.spec.temporal == "cnn":
            temporal = z_cnn
        elif self.spec.temporal in {"bilstm", "serial"}:
            temporal = z_lstm
        else:
            assert self.temporal_fusion is not None and z_cnn is not None and z_lstm is not None
            temporal = self.temporal_fusion(torch.cat([z_cnn, z_lstm], dim=1))
        assert temporal is not None
        return temporal, {"cnn": z_cnn, "lstm": z_lstm}

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
        temporal, experts = self._temporal(sequence, lengths, mask)
        z_aggregate = self.aggregate(aggregate) if self.aggregate is not None else None
        z_static = self.static(static) if self.static is not None else None
        gates: torch.Tensor | None = None
        if self.spec.branches == "aggregate_static":
            assert (
                self.aggregate_static_fusion is not None
                and z_aggregate is not None
                and z_static is not None
            )
            fused = self.aggregate_static_fusion(
                torch.cat([z_aggregate, z_static], dim=1)
            )
        elif self.spec.branches == "temporal_only":
            assert temporal is not None and self.temporal_projection is not None
            fused = self.temporal_projection(temporal)
        elif self.spec.branches == "full":
            assert (
                temporal is not None
                and self.temporal_projection is not None
                and self.full_gates is not None
                and z_aggregate is not None
                and z_static is not None
            )
            z_temporal = self.temporal_projection(temporal)
            z_aggregate = self._drop_branch(z_aggregate)
            z_static = self._drop_branch(z_static)
            gates = self.full_gates(torch.cat([z_temporal, z_aggregate, z_static], dim=1))
            fused = z_temporal + gates[:, 0:1] * z_aggregate + gates[:, 1:2] * z_static
        else:
            raise ValueError(self.spec.branches)
        logits = self.head(fused).squeeze(1)
        if not return_diagnostics:
            return logits
        z_cnn = experts["cnn"]
        z_lstm = experts["lstm"]
        cosine = (
            F.cosine_similarity(z_cnn, z_lstm, dim=1)
            if z_cnn is not None and z_lstm is not None and z_cnn.shape == z_lstm.shape
            else None
        )
        return logits, {
            "gates": gates,
            "cnn_norm": None if z_cnn is None else z_cnn.norm(dim=1),
            "lstm_norm": None if z_lstm is None else z_lstm.norm(dim=1),
            "expert_cosine": cosine,
        }


def parameter_breakdown(model: OULADArchitectureDiagnosisNet) -> dict[str, int | float]:
    def count(module: nn.Module | None) -> int:
        return 0 if module is None else int(sum(p.numel() for p in module.parameters()))

    groups = {
        "input_projection": count(model.input_projection) + count(model.input_norm),
        "cnn": count(model.cnn),
        "bilstm": count(model.lstm),
        "temporal_fusion": count(model.temporal_fusion)
        + count(model.temporal_projection),
        "aggregate_branch": count(model.aggregate),
        "static_branch": count(model.static),
        "fusion_head": count(model.aggregate_static_fusion)
        + count(model.full_gates)
        + count(model.head),
    }
    total = int(sum(p.numel() for p in model.parameters()))
    result: dict[str, int | float] = {"total": total, **groups}
    result.update({f"{name}_ratio": value / total for name, value in groups.items()})
    return result


__all__ = [
    "CandidateSpec",
    "OULADArchitectureDiagnosisNet",
    "candidate_specs",
    "expected_calibration_error",
    "parameter_breakdown",
]
