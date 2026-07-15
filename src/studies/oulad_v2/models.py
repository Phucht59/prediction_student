from __future__ import annotations

import hashlib
import io
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .data import OULADV2Data, STATIC_COLUMNS


CATEGORICAL_STATIC = ["code_module", "presentation_season"]
INPUTS_BY_CANDIDATE = {
    "V2-H2T": {"sequence", "static"},
    "V2-A0": {"aggregate", "static"},
    "V2-T0": {"sequence"},
    "V2-H3C": {"sequence", "aggregate", "static"},
    "V2-H2P": {"sequence", "static"},
}


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _static_preprocessor() -> ColumnTransformer:
    numeric = [column for column in STATIC_COLUMNS if column not in CATEGORICAL_STATIC]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_STATIC,
            ),
        ],
        sparse_threshold=0.0,
    )


@dataclass
class V2Preprocessors:
    sequence_mean: np.ndarray | None = None
    sequence_std: np.ndarray | None = None
    aggregate: Pipeline | None = None
    static: ColumnTransformer | None = None


@dataclass
class PreparedInputs:
    sequence: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    aggregate: np.ndarray
    static: np.ndarray
    target: np.ndarray
    preprocessors: V2Preprocessors


def prepare_inputs(
    data: OULADV2Data,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    candidate_id: str,
    fitted: V2Preprocessors | None = None,
) -> PreparedInputs:
    if candidate_id not in INPUTS_BY_CANDIDATE:
        raise KeyError(candidate_id)
    uses = INPUTS_BY_CANDIDATE[candidate_id]
    fitted = fitted or V2Preprocessors()
    sequence_values = np.zeros((len(transform_indices), data.base.sequence.shape[1], data.base.sequence.shape[2]), dtype=np.float32)
    if "sequence" in uses:
        if fitted.sequence_mean is None or fitted.sequence_std is None:
            train_values = data.base.sequence[fit_indices]
            train_mask = data.base.padding_mask[fit_indices]
            denominator = int(train_mask.sum())
            mean = (train_values * train_mask[..., None]).sum(axis=(0, 1)) / max(1, denominator)
            variance = (((train_values - mean) ** 2) * train_mask[..., None]).sum(axis=(0, 1)) / max(1, denominator)
            fitted.sequence_mean = mean.astype(np.float32)
            fitted.sequence_std = np.sqrt(variance).clip(1e-6).astype(np.float32)
        sequence_values = (data.base.sequence[transform_indices] - fitted.sequence_mean) / fitted.sequence_std
        sequence_values *= data.base.padding_mask[transform_indices][..., None]

    aggregate_values = np.zeros((len(transform_indices), 1), dtype=np.float32)
    if "aggregate" in uses:
        if fitted.aggregate is None:
            fitted.aggregate = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
            fitted.aggregate.fit(data.aggregate.loc[fit_indices, list(data.aggregate_columns)])
        aggregate_values = fitted.aggregate.transform(data.aggregate.loc[transform_indices, list(data.aggregate_columns)]).astype(np.float32)

    static_values = np.zeros((len(transform_indices), 1), dtype=np.float32)
    if "static" in uses:
        if fitted.static is None:
            fitted.static = _static_preprocessor()
            fitted.static.fit(data.base.cohort.loc[fit_indices, STATIC_COLUMNS])
        static_values = fitted.static.transform(data.base.cohort.loc[transform_indices, STATIC_COLUMNS]).astype(np.float32)

    return PreparedInputs(
        sequence_values.astype(np.float32),
        data.base.valid_lengths[transform_indices].astype(np.int64),
        data.base.padding_mask[transform_indices].astype(np.float32),
        aggregate_values,
        static_values,
        data.y[transform_indices].astype(np.float32),
        fitted,
    )


class TemporalEncoder(nn.Module):
    def __init__(self, input_channels: int, config: dict[str, Any]):
        super().__init__()
        channels = int(config["conv_channels"])
        kernel = int(config["kernel_size"])
        hidden = int(config["lstm_hidden"])
        layers = int(config["lstm_layers"])
        self.conv = nn.Conv1d(input_channels, channels, kernel_size=kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(float(config["dropout"]))
        self.lstm = nn.LSTM(channels, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        self.output_dim = 2 * hidden

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.conv(sequence.transpose(1, 2)).transpose(1, 2))
        values = self.dropout(self.norm(values)) * mask.unsqueeze(-1)
        packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.output_dim = hidden

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class AggregateEncoder(nn.Module):
    def __init__(self, input_dim: int, config: dict[str, Any]):
        super().__init__()
        hidden_1 = int(config["aggregate_hidden_1"])
        hidden_2 = int(config["aggregate_hidden_2"])
        dropout = float(config["dropout"])
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_1), nn.LayerNorm(hidden_1), nn.GELU(), nn.Dropout(dropout)]
        output_dim = hidden_1
        if hidden_2 > 0:
            layers.extend([nn.Linear(hidden_1, hidden_2), nn.LayerNorm(hidden_2), nn.GELU(), nn.Dropout(dropout)])
            output_dim = hidden_2
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class OULADV2Net(nn.Module):
    def __init__(
        self,
        candidate_id: str,
        sequence_channels: int,
        aggregate_dim: int,
        static_dim: int,
        temporal_config: dict[str, Any] | None,
        aggregate_config: dict[str, Any] | None,
    ):
        super().__init__()
        self.candidate_id = candidate_id
        dropout = float((temporal_config or aggregate_config or {})["dropout"])
        if "sequence" in INPUTS_BY_CANDIDATE[candidate_id]:
            assert temporal_config is not None
            self.temporal = TemporalEncoder(sequence_channels, temporal_config)
        if "aggregate" in INPUTS_BY_CANDIDATE[candidate_id]:
            assert aggregate_config is not None
            self.aggregate = AggregateEncoder(aggregate_dim, aggregate_config)
        if "static" in INPUTS_BY_CANDIDATE[candidate_id]:
            static_hidden = int((temporal_config or {}).get("static_hidden", 32))
            self.static = StaticEncoder(static_dim, static_hidden, dropout)

        if candidate_id in {"V2-H2T", "V2-H2P"}:
            fusion_hidden = int((temporal_config or {}).get("fusion_hidden", 32))
            self.head = nn.Sequential(
                nn.Linear(self.temporal.output_dim + self.static.output_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, 1),
            )
        elif candidate_id == "V2-T0":
            self.head = nn.Linear(self.temporal.output_dim, 1)
        elif candidate_id == "V2-A0":
            self.aggregate_norm = nn.LayerNorm(self.aggregate.output_dim)
            self.static_norm = nn.LayerNorm(self.static.output_dim)
            self.head = nn.Sequential(
                nn.Linear(self.aggregate.output_dim + self.static.output_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        elif candidate_id == "V2-H3C":
            self.temporal_norm = nn.LayerNorm(self.temporal.output_dim)
            self.aggregate_norm = nn.LayerNorm(self.aggregate.output_dim)
            self.static_norm = nn.LayerNorm(self.static.output_dim)
            self.head = nn.Sequential(
                nn.Linear(self.temporal.output_dim + self.aggregate.output_dim + self.static.output_dim, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            raise KeyError(candidate_id)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
        mask: torch.Tensor,
        aggregate: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        if self.candidate_id in {"V2-H2T", "V2-H2P"}:
            embedding = torch.cat([self.temporal(sequence, lengths, mask), self.static(static)], dim=1)
        elif self.candidate_id == "V2-T0":
            embedding = self.temporal(sequence, lengths, mask)
        elif self.candidate_id == "V2-A0":
            embedding = torch.cat([self.aggregate_norm(self.aggregate(aggregate)), self.static_norm(self.static(static))], dim=1)
        elif self.candidate_id == "V2-H3C":
            embedding = torch.cat(
                [
                    self.temporal_norm(self.temporal(sequence, lengths, mask)),
                    self.aggregate_norm(self.aggregate(aggregate)),
                    self.static_norm(self.static(static)),
                ],
                dim=1,
            )
        else:
            raise KeyError(self.candidate_id)
        return self.head(embedding).squeeze(1)


def build_model(candidate_id: str, prepared: PreparedInputs, temporal_config: dict | None, aggregate_config: dict | None) -> OULADV2Net:
    return OULADV2Net(
        candidate_id,
        prepared.sequence.shape[2],
        prepared.aggregate.shape[1],
        prepared.static.shape[1],
        temporal_config,
        aggregate_config,
    )


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()
