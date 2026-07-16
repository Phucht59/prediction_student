from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from src.studies.oulad.models_ml import CATEGORICAL, make_preprocessor


DEEP_CONFIGS: dict[str, dict[str, Any]] = {
    "C-M0": {"hidden": 32, "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 12, "patience": 3},
    "C-C0": {"conv_channels": 32, "kernel_size": 3, "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 12, "patience": 3},
    "C-L1": {"lstm_hidden": 32, "lstm_layers": 1, "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 12, "patience": 3},
    "C-H1": {"conv_channels": 32, "kernel_size": 3, "lstm_hidden": 32, "lstm_layers": 1, "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 12, "patience": 3},
    "C-H2": {"conv_channels": 32, "kernel_size": 3, "lstm_hidden": 32, "lstm_layers": 1, "static_hidden": 32, "fusion_hidden": 32, "dropout": 0.2, "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 12, "patience": 3},
}


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class OULADNet(nn.Module):
    def __init__(self, candidate_id: str, sequence_channels: int, tabular_dim: int, static_dim: int, config: dict[str, Any]):
        super().__init__()
        self.candidate_id = candidate_id
        self.dropout = nn.Dropout(config.get("dropout", 0.2))
        if candidate_id == "C-M0":
            self.mlp = nn.Sequential(nn.Linear(tabular_dim, config["hidden"]), nn.ReLU(), self.dropout, nn.Linear(config["hidden"], 1))
            return
        if candidate_id in {"C-C0", "C-H1", "C-H2"}:
            channels = config["conv_channels"]
            kernel = config["kernel_size"]
            self.conv = nn.Conv1d(sequence_channels, channels, kernel_size=kernel, padding=kernel // 2)
            self.conv_norm = nn.LayerNorm(channels)
            temporal_input = channels
        else:
            temporal_input = sequence_channels
        if candidate_id in {"C-L1", "C-H1", "C-H2"}:
            self.lstm = nn.LSTM(temporal_input, config["lstm_hidden"], num_layers=config.get("lstm_layers", 1), batch_first=True, bidirectional=True)
            temporal_dim = 2 * config["lstm_hidden"]
        else:
            temporal_dim = temporal_input
        if candidate_id == "C-H2":
            self.static = nn.Sequential(nn.Linear(static_dim, config["static_hidden"]), nn.ReLU(), self.dropout)
            self.head = nn.Sequential(nn.Linear(temporal_dim + config["static_hidden"], config["fusion_hidden"]), nn.ReLU(), self.dropout, nn.Linear(config["fusion_hidden"], 1))
        else:
            self.head = nn.Linear(temporal_dim, 1)

    def _temporal(self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = sequence
        if hasattr(self, "conv"):
            values = torch.relu(self.conv(values.transpose(1, 2)).transpose(1, 2))
            values = self.conv_norm(values)
            values = self.dropout(values) * mask.unsqueeze(-1)
        if hasattr(self, "lstm"):
            packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed)
            return torch.cat([hidden[-2], hidden[-1]], dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / denominator

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, mask: torch.Tensor, tabular: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if self.candidate_id == "C-M0":
            return self.mlp(tabular).squeeze(1)
        temporal = self._temporal(sequence, lengths, mask)
        if self.candidate_id == "C-H2":
            return self.head(torch.cat([temporal, self.static(static)], dim=1)).squeeze(1)
        return self.head(temporal).squeeze(1)


@dataclass
class PreparedInputs:
    sequence: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    tabular: np.ndarray
    static: np.ndarray
    y: np.ndarray
    preprocessors: dict[str, Any]


def _static_preprocessor(columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in columns if column not in CATEGORICAL and column != "record_id"]
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], sparse_threshold=0.0)


def prepare_inputs(data, fit_indices: np.ndarray, transform_indices: np.ndarray, candidate_id: str, fitted: dict[str, Any] | None = None) -> PreparedInputs:
    fitted = {} if fitted is None else fitted
    sequence = data.sequence.copy()
    if "sequence_mean" not in fitted:
        train_values = sequence[fit_indices]
        train_mask = data.padding_mask[fit_indices]
        denominator = train_mask.sum()
        mean = (train_values * train_mask[..., None]).sum(axis=(0, 1)) / max(1, denominator)
        variance = (((train_values - mean) ** 2) * train_mask[..., None]).sum(axis=(0, 1)) / max(1, denominator)
        fitted["sequence_mean"] = mean.astype(np.float32); fitted["sequence_std"] = np.sqrt(variance).clip(1e-6).astype(np.float32)
    selected_sequence = (sequence[transform_indices] - fitted["sequence_mean"]) / fitted["sequence_std"]
    selected_sequence *= data.padding_mask[transform_indices][..., None]

    tabular_values = np.zeros((len(transform_indices), 1), dtype=np.float32)
    if candidate_id == "C-M0":
        if "tabular" not in fitted:
            fitted["tabular"] = make_preprocessor(list(data.tabular.columns)).fit(data.tabular.iloc[fit_indices])
        tabular_values = fitted["tabular"].transform(data.tabular.iloc[transform_indices]).astype(np.float32)
    static_columns = ["record_id", "code_module", "presentation_season", "num_of_prev_attempts", "studied_credits", "registration_lead_time", "module_presentation_length"]
    static_values = np.zeros((len(transform_indices), 1), dtype=np.float32)
    if candidate_id == "C-H2":
        if "static" not in fitted:
            fitted["static"] = _static_preprocessor(static_columns).fit(data.cohort[static_columns].iloc[fit_indices])
        static_values = fitted["static"].transform(data.cohort[static_columns].iloc[transform_indices]).astype(np.float32)
    return PreparedInputs(selected_sequence.astype(np.float32), data.valid_lengths[transform_indices], data.padding_mask[transform_indices].astype(np.float32), tabular_values, static_values, data.y[transform_indices], fitted)


def _loader(inputs: PreparedInputs, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(inputs.sequence), torch.from_numpy(inputs.lengths.astype(np.int64)), torch.from_numpy(inputs.mask), torch.from_numpy(inputs.tabular), torch.from_numpy(inputs.static), torch.from_numpy(inputs.y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _predict(model: nn.Module, inputs: PreparedInputs, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval(); values = []
    with torch.no_grad():
        for sequence, lengths, mask, tabular, static, _ in _loader(inputs, batch_size, False):
            logits = model(sequence.to(device), lengths.to(device), mask.to(device), tabular.to(device), static.to(device))
            values.extend(torch.sigmoid(logits).cpu().numpy())
    output = np.asarray(values, dtype=float)
    if not np.isfinite(output).all(): raise RuntimeError("Non-finite deep probabilities")
    return output


def fit_deep(
    data,
    candidate_id: str,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    *,
    fixed_epochs: int | None = None,
) -> dict[str, Any]:
    config = DEEP_CONFIGS[candidate_id]
    set_seed(seed); torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    train_inputs = prepare_inputs(data, train_indices, train_indices, candidate_id)
    validation_inputs = prepare_inputs(data, train_indices, validation_indices, candidate_id, train_inputs.preprocessors)
    model = OULADNet(candidate_id, data.sequence.shape[2], train_inputs.tabular.shape[1], train_inputs.static.shape[1], config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    positive = float(train_inputs.y.sum()); negative = float(len(train_inputs.y) - positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / max(1.0, positive), device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_state = copy.deepcopy(model.state_dict()); best_loss = float("inf"); best_epoch = 1; stale = 0
    epochs = fixed_epochs or config["max_epochs"]
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for sequence, lengths, mask, tabular, static, target in _loader(train_inputs, config["batch_size"], True):
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequence.to(device), lengths.to(device), mask.to(device), tabular.to(device), static.to(device))
            loss = criterion(logits, target.to(device));
            if not torch.isfinite(loss): raise RuntimeError("Non-finite training loss")
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); train_losses.append(float(loss.detach().cpu()))
        if fixed_epochs is None:
            probabilities = _predict(model, validation_inputs, config["batch_size"], device)
            eps = 1e-7; validation_loss = float(-np.mean(validation_inputs.y * np.log(np.clip(probabilities, eps, 1 - eps)) + (1 - validation_inputs.y) * np.log(np.clip(1 - probabilities, eps, 1 - eps))))
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_nll": validation_loss})
            if validation_loss < best_loss - 1e-5:
                best_loss, best_epoch, best_state, stale = validation_loss, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= config["patience"]: break
        else:
            history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_nll": None})
            best_state, best_epoch = copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    probabilities = _predict(model, validation_inputs, config["batch_size"], device)
    return {"probabilities": probabilities, "selected_epoch": best_epoch, "state_dict": best_state, "preprocessors": train_inputs.preprocessors, "history": history, "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad), "config": config, "prediction_reproduction_max_abs_difference": float(np.max(np.abs(probabilities - _predict(model, validation_inputs, config["batch_size"], device))))}
