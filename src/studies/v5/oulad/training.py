from __future__ import annotations

import copy
import hashlib
import io
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.studies.oulad_v4.models import PreparedInputs, prepare_inputs

from .augmentation import augment_training_data
from .models import build_model, count_parameters


@dataclass
class OULADFitResult:
    probability: np.ndarray
    selected_epoch: int
    history: list[dict[str, Any]]
    parameter_count: int
    runtime_seconds: float
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    augmentation_audit: dict[str, object]
    attention_padding_max: float | None
    gate_means: list[float]


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _loader(inputs: PreparedInputs, batch_size: int, shuffle: bool, seed: int, pin: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(inputs.sequence),
        torch.from_numpy(inputs.lengths),
        torch.from_numpy(inputs.mask),
        torch.from_numpy(inputs.aggregate),
        torch.from_numpy(inputs.static),
        torch.from_numpy(inputs.target),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )


def _predict(model, inputs: PreparedInputs, batch_size: int, device: torch.device, diagnostics: bool = False):
    model.eval()
    probabilities: list[np.ndarray] = []
    padding_maxima: list[float] = []
    gates: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, lengths, mask, aggregate, static, _ in _loader(
            inputs, batch_size, False, 0, device.type == "cuda"
        ):
            sequence = sequence.to(device)
            lengths = lengths.to(device)
            mask_device = mask.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                result = model(
                    sequence,
                    lengths,
                    mask_device,
                    aggregate.to(device),
                    static.to(device),
                    return_attention=diagnostics,
                )
            if diagnostics:
                logits, attention, gate = result
                gates.append(gate.float().cpu().numpy())
                if attention is not None and (~mask_device.bool()).any():
                    padding_maxima.append(float(attention.float().masked_select(~mask_device.bool()).max().cpu()))
            else:
                logits = result
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
    probability = np.concatenate(probabilities).astype(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise RuntimeError("OULAD probability contract failed")
    return (
        probability,
        max(padding_maxima) if padding_maxima else (0.0 if diagnostics else None),
        np.concatenate(gates).mean(axis=0).tolist() if gates else [],
    )


def _loss(logits: torch.Tensor, target: torch.Tensor, config: dict[str, Any], positive_weight: float):
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=logits.device)
    base = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), pos_weight=pos_weight, reduction="none"
    )
    if config.get("loss", "weighted_bce") == "focal":
        probability = torch.sigmoid(logits.float())
        probability_target = probability * target + (1 - probability) * (1 - target)
        base = ((1 - probability_target) ** float(config.get("focal_gamma", 2.0))) * base
    return base.mean()


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_oulad_model(
    data,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
    augmentation: str,
    variant: str = "cnn_bilstm",
    fixed_epochs: int | None = None,
    device_name: str = "cuda",
) -> OULADFitResult:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    augmented, augmentation_audit = augment_training_data(data, train_indices, augmentation, seed)
    train = prepare_inputs(augmented, train_indices, train_indices, "V4-D0")
    evaluation = prepare_inputs(data, train_indices, evaluation_indices, "V4-D0", train.preprocessors)
    model = build_model(train, config, variant).to(device)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError(f"OULAD V5 parameter guard exceeded: {parameter_count}")
    positive_weight = 1.0
    if config.get("positive_weight", "none") == "balanced":
        positives = max(float(train.target.sum()), 1.0)
        positive_weight = float((len(train.target) - positives) / positives)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    epoch_limit = int(fixed_epochs or config.get("max_epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_limit)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    batch_size = int(config.get("batch_size", 128))
    patience = int(config.get("patience", 10))
    best_loss = float("inf")
    best_epoch = 1
    stale = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []
    for epoch in range(1, epoch_limit + 1):
        model.train()
        losses: list[float] = []
        norms: list[float] = []
        for sequence, lengths, mask, aggregate, static, target in _loader(
            train, batch_size, True, seed + epoch, device.type == "cuda"
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(
                    sequence.to(device),
                    lengths.to(device),
                    mask.to(device),
                    aggregate.to(device),
                    static.to(device),
                )
                loss = _loss(logits, target.to(device), config, positive_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite OULAD V5 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            norms.append(float(norm.detach().cpu()))
        scheduler.step()
        validation_nll = None
        validation_macro_f1 = None
        if fixed_epochs is None:
            probability, _, _ = _predict(model, evaluation, batch_size, device)
            clipped = probability.clip(1e-7, 1 - 1e-7)
            validation_nll = float(
                -(evaluation.target * np.log(clipped) + (1 - evaluation.target) * np.log(1 - clipped)).mean()
            )
            validation_macro_f1 = float(
                f1_score(evaluation.target, probability >= 0.5, average="macro", zero_division=0)
            )
            if validation_nll < best_loss - 1e-5:
                best_loss = validation_nll
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
        else:
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_nll": validation_nll,
                "validation_macro_f1_at_0_5": validation_macro_f1,
                "gradient_norm": float(np.mean(norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if fixed_epochs is None and stale >= patience:
            break
    model.load_state_dict(best_state)
    probability, padding_max, gates = _predict(model, evaluation, batch_size, device, diagnostics=True)
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = build_model(train, config, variant).to(device)
    replay.load_state_dict(cpu_state)
    replay_probability, _, _ = _predict(replay, evaluation, batch_size, device)
    difference = float(np.max(np.abs(probability - replay_probability)))
    if difference > 1e-6:
        raise RuntimeError(f"OULAD checkpoint replay failed: {difference}")
    if padding_max is not None and padding_max > 1e-7:
        raise RuntimeError(f"Attention leaked to padding: {padding_max}")
    return OULADFitResult(
        probability=probability,
        selected_epoch=best_epoch,
        history=history,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=difference,
        augmentation_audit=augmentation_audit,
        attention_padding_max=padding_max,
        gate_means=gates,
    )


def choose_threshold(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    rows = []
    for threshold in np.linspace(0.2, 0.8, 121):
        prediction = probability >= threshold
        rows.append(
            (
                float(f1_score(target, prediction, average="macro", zero_division=0)),
                -abs(float(threshold) - 0.5),
                float(threshold),
            )
        )
    best = max(rows)
    return {"threshold": best[2], "inner_macro_f1": best[0]}


__all__ = ["OULADFitResult", "choose_threshold", "fit_oulad_model"]

