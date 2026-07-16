from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import OULADV2Data
from .models import PreparedInputs, V2Preprocessors, build_model, prepare_inputs, set_deterministic_seed, state_dict_sha256


@dataclass
class FitResult:
    probabilities: np.ndarray
    selected_epoch: int
    epochs_ran: int
    history: list[dict[str, Any]]
    parameter_count: int
    runtime_seconds: float
    state_dict: dict[str, torch.Tensor]
    state_dict_sha256: str
    preprocessors: V2Preprocessors
    reproduction_max_abs_difference: float
    device: str


def _dataset(inputs: PreparedInputs) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(inputs.sequence),
        torch.from_numpy(inputs.lengths),
        torch.from_numpy(inputs.mask),
        torch.from_numpy(inputs.aggregate),
        torch.from_numpy(inputs.static),
        torch.from_numpy(inputs.target),
    )


def _loader(inputs: PreparedInputs, batch_size: int, shuffle: bool, seed: int, device: torch.device) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        _dataset(inputs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def predict(model: nn.Module, inputs: PreparedInputs, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, lengths, mask, aggregate, static, _ in _loader(inputs, batch_size, False, 0, device):
            logits = model(
                sequence.to(device, non_blocking=True),
                lengths.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                aggregate.to(device, non_blocking=True),
                static.to(device, non_blocking=True),
            )
            output.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(output).astype(float)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise RuntimeError("Probability contract failed")
    return probabilities


def _nll(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))


def _positive_weight(policy: str, y: np.ndarray) -> float:
    positive = float(y.sum())
    negative = float(len(y) - positive)
    balanced = negative / max(positive, 1.0)
    if policy == "none":
        return 1.0
    if policy == "sqrt_balanced":
        return float(np.sqrt(balanced))
    if policy == "fully_balanced":
        return float(balanced)
    raise KeyError(policy)


def fit_candidate(
    data: OULADV2Data,
    candidate_id: str,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    temporal_config: dict[str, Any] | None,
    aggregate_config: dict[str, Any] | None,
    seed: int,
    fixed_epochs: int | None = None,
    device_name: str | None = None,
) -> FitResult:
    started = time.perf_counter()
    training_config = temporal_config if candidate_id != "V2-A0" else aggregate_config
    if training_config is None:
        raise RuntimeError(f"Missing training config for {candidate_id}")
    config = dict(training_config)
    max_epochs = int(config.get("max_epochs", 40))
    patience = int(config.get("patience", 6))
    batch_size = int(config["batch_size"])
    set_deterministic_seed(seed)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_inputs = prepare_inputs(data, train_indices, train_indices, candidate_id)
    evaluation_inputs = prepare_inputs(data, train_indices, evaluation_indices, candidate_id, train_inputs.preprocessors)
    model = build_model(candidate_id, train_inputs, temporal_config, aggregate_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count >= 300_000:
        raise RuntimeError(f"Parameter guardrail exceeded: {parameter_count}")
    pos_weight = _positive_weight(str(config["positive_weight"]), train_inputs.target)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = None
    if config.get("scheduler", "fixed_lr") == "deterministic_cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    elif config.get("scheduler", "fixed_lr") != "fixed_lr":
        raise RuntimeError("Non-replayable scheduler requested")

    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    history: list[dict[str, Any]] = []
    epochs_to_run = int(fixed_epochs or max_epochs)
    for epoch in range(1, epochs_to_run + 1):
        model.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        for sequence, lengths, mask, aggregate, static, target in _loader(train_inputs, batch_size, True, seed + epoch, device):
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                sequence.to(device, non_blocking=True),
                lengths.to(device, non_blocking=True),
                mask.to(device, non_blocking=True),
                aggregate.to(device, non_blocking=True),
                static.to(device, non_blocking=True),
            )
            loss = criterion(logits, target.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
        if scheduler is not None:
            scheduler.step()
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_nll": None,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm_mean": float(np.mean(gradient_norms)),
        }
        if fixed_epochs is None:
            probabilities = predict(model, evaluation_inputs, batch_size, device)
            validation_loss = _nll(evaluation_inputs.target, probabilities)
            row["validation_nll"] = validation_loss
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(row)
        if fixed_epochs is None and stale >= patience:
            break

    model.load_state_dict(best_state)
    probabilities = predict(model, evaluation_inputs, batch_size, device)
    cpu_state = {key: value.detach().cpu().clone() for key, value in best_state.items()}
    state_hash = state_dict_sha256(cpu_state)

    replay = build_model(candidate_id, train_inputs, temporal_config, aggregate_config).to(device)
    replay.load_state_dict(cpu_state)
    replay_probabilities = predict(replay, evaluation_inputs, batch_size, device)
    reproduction_difference = float(np.max(np.abs(probabilities - replay_probabilities)))
    if reproduction_difference > 1e-7:
        raise RuntimeError(f"Checkpoint reproduction failed: {reproduction_difference}")
    runtime = time.perf_counter() - started
    del replay, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return FitResult(
        probabilities,
        best_epoch,
        len(history),
        history,
        parameter_count,
        runtime,
        cpu_state,
        state_hash,
        train_inputs.preprocessors,
        reproduction_difference,
        str(device),
    )
