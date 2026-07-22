from __future__ import annotations

import copy
import hashlib
import io
import random
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.studies.v5.oulad.augmentation import augment_training_data

from .data import OULADInputsV51, prepare_oulad_inputs
from .models import OULADHybridV51, count_parameters
from .pretraining import load_pretrained_temporal


@dataclass
class OULADFitResultV51:
    probability: np.ndarray
    selected_epoch: int
    history: list[dict[str, float | int | None]]
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    parameter_count: int
    runtime_seconds: float
    gpu_peak_memory_bytes: int
    augmentation_audit: dict[str, object]
    gate_statistics: dict[str, object]
    attention_entropy_mean: float | None
    attention_padding_max: float
    branch_norm_means: dict[str, float]


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def transform_temporal_order(inputs: OULADInputsV51, order: str, seed: int = 0) -> OULADInputsV51:
    if order == "original":
        return inputs
    if order not in {"reversed", "shuffled"}:
        raise ValueError(f"Unknown temporal order: {order}")
    rng = np.random.default_rng(seed)
    sequence = inputs.sequence.copy()
    for row, length in enumerate(inputs.lengths):
        valid = sequence[row, : int(length)].copy()
        if order == "reversed":
            sequence[row, : int(length)] = valid[::-1]
        else:
            sequence[row, : int(length)] = valid[rng.permutation(int(length))]
    sequence *= inputs.mask[..., None]
    return replace(inputs, sequence=sequence)


def _loader(inputs: OULADInputsV51, batch_size: int, shuffle: bool, seed: int, pin: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(inputs.sequence),
            torch.from_numpy(inputs.lengths),
            torch.from_numpy(inputs.mask),
            torch.from_numpy(inputs.aggregate),
            torch.from_numpy(inputs.static),
            torch.from_numpy(inputs.target),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )


def classification_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    positive_weight: float,
    focal_gamma: float,
) -> torch.Tensor:
    pos_weight = torch.tensor(positive_weight, dtype=torch.float32, device=logits.device)
    base = nn.functional.binary_cross_entropy_with_logits(
        logits.float(), target.float(), pos_weight=pos_weight, reduction="none"
    )
    if loss_name == "focal":
        probability = torch.sigmoid(logits.float())
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        base = ((1.0 - pt) ** focal_gamma) * base
    elif loss_name not in {"standard_bce", "weighted_bce"}:
        raise ValueError(f"Unknown OULAD loss: {loss_name}")
    return base.mean()


def _diagnostics(gates: list[np.ndarray]) -> dict[str, object]:
    if not gates:
        return {
            "mean": None,
            "variance": None,
            "saturation_fraction": None,
            "per_branch_mean": [],
            "collapsed": False,
        }
    values = np.concatenate(gates)
    saturation = float(((values <= 0.05) | (values >= 0.95)).mean())
    variance = float(values.var())
    return {
        "mean": float(values.mean()),
        "variance": variance,
        "saturation_fraction": saturation,
        "per_branch_mean": values.mean(axis=0).tolist(),
        "collapsed": bool(saturation >= 0.95 or variance < 1e-6),
    }


def _predict(
    model: OULADHybridV51,
    inputs: OULADInputsV51,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object], float | None, float, dict[str, float]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    padding_maxima: list[float] = []
    norms: dict[str, list[np.ndarray]] = {"temporal": [], "aggregate": [], "static": []}
    with torch.no_grad():
        for sequence, lengths, mask, aggregate, static, _ in _loader(
            inputs, batch_size, False, 0, device.type == "cuda"
        ):
            mask_device = mask.to(device)
            logits, diagnostic = model(
                sequence.to(device),
                lengths.to(device),
                mask_device,
                aggregate.to(device),
                static.to(device),
                return_diagnostics=True,
            )
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            gate = diagnostic["gate"]
            if gate is not None:
                gates.append(gate.float().cpu().numpy())
            entropy = diagnostic["attention_entropy"]
            if entropy is not None:
                entropies.append(entropy.float().cpu().numpy())
            attention = diagnostic["attention"]
            if attention is not None and (~mask_device.bool()).any():
                padding_maxima.append(float(attention.masked_select(~mask_device.bool()).max().cpu()))
            for name in norms:
                value = diagnostic[f"{name}_norm"]
                assert value is not None
                norms[name].append(value.float().cpu().numpy())
    probability = np.concatenate(probabilities).astype(float)
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise RuntimeError("OULAD V5.1 probability contract failed")
    padding_max = max(padding_maxima, default=0.0)
    if padding_max > 1e-7:
        raise RuntimeError(f"OULAD attention leaked to padding: {padding_max}")
    return (
        probability,
        _diagnostics(gates),
        float(np.concatenate(entropies).mean()) if entropies else None,
        padding_max,
        {name: float(np.concatenate(values).mean()) for name, values in norms.items()},
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_prepared_oulad_model(
    train: OULADInputsV51,
    evaluation: OULADInputsV51,
    *,
    config: dict[str, Any],
    seed: int,
    variant: str = "cnn_bilstm",
    fixed_epochs: int | None = None,
    device_name: str = "cuda",
    initial_temporal_state: dict[str, torch.Tensor] | None = None,
    augmentation_audit: dict[str, object] | None = None,
) -> OULADFitResultV51:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = OULADHybridV51(
        train.sequence.shape[2], train.aggregate.shape[1], train.static.shape[1], config, variant
    ).to(device)
    if initial_temporal_state is not None:
        load_pretrained_temporal(model, initial_temporal_state)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError(f"OULAD V5.1 parameter guard exceeded: {parameter_count}")
    positive_weight = 1.0
    if str(config.get("loss", "standard_bce")) == "weighted_bce":
        positives = max(float(train.target.sum()), 1.0)
        positive_weight = float((len(train.target) - positives) / positives)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 5e-4)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    epoch_limit = int(fixed_epochs or config.get("max_epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_limit)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    batch_size = int(config.get("batch_size", 256))
    patience = int(config.get("patience", 10))
    best_metric = -1.0
    best_nll = float("inf")
    best_epoch = 1
    stale = 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int | None]] = []
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
                loss = classification_loss(
                    logits,
                    target.to(device),
                    loss_name=str(config.get("loss", "standard_bce")),
                    positive_weight=positive_weight,
                    focal_gamma=float(config.get("focal_gamma", 2.0)),
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite OULAD V5.1 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            norms.append(float(norm.detach().cpu()))
        scheduler.step()
        probability, _, _, _, _ = _predict(model, evaluation, batch_size, device)
        metric = float(f1_score(evaluation.target, probability >= 0.5, average="macro", zero_division=0))
        clipped = probability.clip(1e-7, 1 - 1e-7)
        nll = float(
            -(evaluation.target * np.log(clipped) + (1 - evaluation.target) * np.log(1 - clipped)).mean()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1_at_0_5": metric,
                "validation_nll": nll,
                "gradient_norm": float(np.mean(norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        improved = metric > best_metric + 1e-6 or (abs(metric - best_metric) <= 1e-6 and nll < best_nll)
        if fixed_epochs is not None or improved:
            best_metric = metric
            best_nll = nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    model.load_state_dict(best_state)
    probability, gate_stats, entropy, padding_max, branch_norms = _predict(
        model, evaluation, batch_size, device
    )
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = OULADHybridV51(
        train.sequence.shape[2], train.aggregate.shape[1], train.static.shape[1], config, variant
    ).to(device)
    replay.load_state_dict(cpu_state)
    replay_probability, _, _, _, _ = _predict(replay, evaluation, batch_size, device)
    difference = float(np.max(np.abs(probability - replay_probability)))
    if difference > 1e-6:
        raise RuntimeError(f"OULAD V5.1 checkpoint replay failed: {difference}")
    peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return OULADFitResultV51(
        probability=probability,
        selected_epoch=best_epoch,
        history=history,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=difference,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
        gpu_peak_memory_bytes=peak_memory,
        augmentation_audit=augmentation_audit or {"strategy": "none", "changed_values": 0},
        gate_statistics=gate_stats,
        attention_entropy_mean=entropy,
        attention_padding_max=padding_max,
        branch_norm_means=branch_norms,
    )


def fit_oulad_model_v5_1(
    data,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    config: dict[str, Any],
    seed: int,
    augmentation: str = "none",
    variant: str = "cnn_bilstm",
    fixed_epochs: int | None = None,
    device_name: str = "cuda",
    initial_temporal_state: dict[str, torch.Tensor] | None = None,
) -> OULADFitResultV51:
    mapped = "channel_dropout" if augmentation == "channel_group_dropout" else augmentation
    augmented, audit = augment_training_data(data, train_indices, mapped, seed)
    audit = {**audit, "registered_strategy": augmentation}
    train = prepare_oulad_inputs(augmented, train_indices, train_indices)
    evaluation = prepare_oulad_inputs(
        data, train_indices, evaluation_indices, fitted=train.preprocessors
    )
    order = str(config.get("temporal_order", "original"))
    train = transform_temporal_order(train, order, seed)
    evaluation = transform_temporal_order(evaluation, order, seed)
    return fit_prepared_oulad_model(
        train,
        evaluation,
        config=config,
        seed=seed,
        variant=variant,
        fixed_epochs=fixed_epochs,
        device_name=device_name,
        initial_temporal_state=initial_temporal_state,
        augmentation_audit=audit,
    )


def choose_threshold(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    candidates = []
    for threshold in np.linspace(0.2, 0.8, 121):
        metric = float(f1_score(target, probability >= threshold, average="macro", zero_division=0))
        candidates.append((metric, -abs(float(threshold) - 0.5), float(threshold)))
    best = max(candidates)
    return {"threshold": best[2], "inner_macro_f1": best[0]}


__all__ = [
    "OULADFitResultV51",
    "choose_threshold",
    "classification_loss",
    "deterministic_seed",
    "fit_oulad_model_v5_1",
    "fit_prepared_oulad_model",
    "transform_temporal_order",
]

