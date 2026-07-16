from __future__ import annotations

import hashlib
import io
import os
import random
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .data import OULADV3Data, STATIC_COLUMNS

CATEGORICAL_STATIC = ["code_module", "presentation_season"]
INPUTS = {
    "V3-P0": {"base_sequence", "base_aggregate", "static"},
    "V3-D0": {"dynamic_sequence", "base_aggregate", "static"},
    "V3-A1": {"matched_vector", "static"},
}


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _static_preprocessor() -> ColumnTransformer:
    numeric = [c for c in STATIC_COLUMNS if c not in CATEGORICAL_STATIC]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_STATIC),
    ], sparse_threshold=0.0)


@dataclass
class V3Preprocessors:
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
    preprocessors: V3Preprocessors


def prepare_inputs(data: OULADV3Data, fit_indices: np.ndarray, transform_indices: np.ndarray,
                   candidate_id: str, fitted: V3Preprocessors | None = None) -> PreparedInputs:
    uses = INPUTS[candidate_id]; fitted = fitted or V3Preprocessors()
    source = data.dynamic_sequence if "dynamic_sequence" in uses else data.base.sequence
    seq = np.zeros((len(transform_indices), source.shape[1], source.shape[2]), dtype=np.float32)
    if "base_sequence" in uses or "dynamic_sequence" in uses:
        if fitted.sequence_mean is None:
            train = source[fit_indices]; mask = data.base.padding_mask[fit_indices]
            denominator = max(1, int(mask.sum()))
            mean = (train * mask[..., None]).sum((0, 1)) / denominator
            variance = (((train - mean) ** 2) * mask[..., None]).sum((0, 1)) / denominator
            fitted.sequence_mean = mean.astype(np.float32)
            fitted.sequence_std = np.sqrt(variance).clip(1e-6).astype(np.float32)
        seq = (source[transform_indices] - fitted.sequence_mean) / fitted.sequence_std
        seq *= data.base.padding_mask[transform_indices][..., None]
    aggregate = np.zeros((len(transform_indices), 1), dtype=np.float32)
    if "base_aggregate" in uses or "matched_vector" in uses:
        raw = (data.matched_vector if "matched_vector" in uses else
               data.v2.aggregate.loc[:, list(data.v2.aggregate_columns)].to_numpy(dtype=np.float32))
        if fitted.aggregate is None:
            fitted.aggregate = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
            fitted.aggregate.fit(raw[fit_indices])
        aggregate = fitted.aggregate.transform(raw[transform_indices]).astype(np.float32)
    static = np.zeros((len(transform_indices), 1), dtype=np.float32)
    if "static" in uses:
        if fitted.static is None:
            fitted.static = _static_preprocessor(); fitted.static.fit(data.base.cohort.loc[fit_indices, STATIC_COLUMNS])
        static = fitted.static.transform(data.base.cohort.loc[transform_indices, STATIC_COLUMNS]).astype(np.float32)
    return PreparedInputs(seq.astype(np.float32), data.base.valid_lengths[transform_indices].astype(np.int64),
                          data.base.padding_mask[transform_indices].astype(np.float32), aggregate, static,
                          data.y[transform_indices].astype(np.float32), fitted)


class TemporalPoolingEncoder(nn.Module):
    def __init__(self, input_channels: int, config: dict[str, Any]):
        super().__init__()
        channels, kernel = int(config["conv_channels"]), int(config["kernel_size"])
        hidden, layers = int(config["lstm_hidden"]), int(config["lstm_layers"])
        self.pooling = str(config["pooling"]); projection = int(config["pooling_projection"])
        self.conv = nn.Conv1d(input_channels, channels, kernel_size=kernel, padding=kernel // 2)
        self.norm = nn.LayerNorm(channels); self.dropout = nn.Dropout(float(config["dropout"]))
        self.lstm = nn.LSTM(channels, hidden, num_layers=layers, batch_first=True, bidirectional=True)
        recurrent = 2 * hidden
        if self.pooling == "last_mean_max":
            self.projection = nn.Sequential(nn.Linear(3 * recurrent, projection), nn.GELU(), nn.LayerNorm(projection))
            self.attention = None
        elif self.pooling == "masked_attention":
            self.attention = nn.Sequential(nn.Linear(recurrent, projection), nn.Tanh(), nn.Linear(projection, 1))
            self.projection = nn.Sequential(nn.Linear(recurrent, projection), nn.GELU(), nn.LayerNorm(projection))
        else: raise KeyError(self.pooling)
        self.output_dim = projection

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor,
                return_attention: bool = False):
        values = torch.relu(self.conv(sequence.transpose(1, 2)).transpose(1, 2))
        values = self.dropout(self.norm(values)) * mask.unsqueeze(-1)
        packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=sequence.shape[1])
        valid = mask.bool(); row = torch.arange(output.shape[0], device=output.device)
        last = output[row, lengths.to(output.device) - 1]
        attention_weights = None
        if self.pooling == "last_mean_max":
            mean = (output * mask.unsqueeze(-1)).sum(1) / lengths.to(output.device).unsqueeze(1)
            maximum = output.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(1).values
            embedding = self.projection(torch.cat([last, mean, maximum], dim=1))
        else:
            scores = self.attention(output).squeeze(-1).masked_fill(~valid, float("-inf"))
            attention_weights = torch.softmax(scores, dim=1)
            embedding = self.projection((output * attention_weights.unsqueeze(-1)).sum(1))
        return (embedding, attention_weights) if return_attention else embedding


class DenseEncoder(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, dropout: float):
        super().__init__(); layers: list[nn.Module] = [nn.Linear(input_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(dropout)]
        self.output_dim = h1
        if h2:
            layers += [nn.Linear(h1, h2), nn.LayerNorm(h2), nn.GELU(), nn.Dropout(dropout)]; self.output_dim = h2
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


class V3Net(nn.Module):
    def __init__(self, candidate_id: str, prepared: PreparedInputs, temporal_config: dict | None, aggregate_config: dict):
        super().__init__(); self.candidate_id = candidate_id
        cfg = temporal_config or aggregate_config; dropout = float(cfg["dropout"])
        self.aggregate = DenseEncoder(prepared.aggregate.shape[1], int(aggregate_config["aggregate_hidden_1"]),
                                      int(aggregate_config["aggregate_hidden_2"]), dropout)
        self.static = DenseEncoder(prepared.static.shape[1], int(cfg.get("static_hidden", 32)), 0, dropout)
        if candidate_id in {"V3-P0", "V3-D0"}:
            assert temporal_config is not None
            self.temporal = TemporalPoolingEncoder(prepared.sequence.shape[2], temporal_config)
            self.tnorm, self.anorm, self.snorm = nn.LayerNorm(self.temporal.output_dim), nn.LayerNorm(self.aggregate.output_dim), nn.LayerNorm(self.static.output_dim)
            dim = self.temporal.output_dim + self.aggregate.output_dim + self.static.output_dim
        else:
            self.anorm, self.snorm = nn.LayerNorm(self.aggregate.output_dim), nn.LayerNorm(self.static.output_dim)
            dim = self.aggregate.output_dim + self.static.output_dim
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, sequence, lengths, mask, aggregate, static, return_attention: bool = False):
        attention = None
        if self.candidate_id in {"V3-P0", "V3-D0"}:
            temporal, attention = self.temporal(sequence, lengths, mask, True)
            embedding = torch.cat([self.tnorm(temporal), self.anorm(self.aggregate(aggregate)), self.snorm(self.static(static))], 1)
        else: embedding = torch.cat([self.anorm(self.aggregate(aggregate)), self.snorm(self.static(static))], 1)
        logits = self.head(embedding).squeeze(1)
        return (logits, attention) if return_attention else logits


def build_model(candidate_id, prepared, temporal_config, aggregate_config):
    return V3Net(candidate_id, prepared, temporal_config, aggregate_config)


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO(); torch.save(state_dict, buffer); return hashlib.sha256(buffer.getvalue()).hexdigest()
