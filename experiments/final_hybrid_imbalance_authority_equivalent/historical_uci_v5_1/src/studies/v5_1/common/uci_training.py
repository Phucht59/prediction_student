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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .uci_data import UCIDataV51, context_preprocessor
from .uci_model import UCIHybridV51, count_parameters, gate_statistics


@dataclass(frozen=True)
class UCIInputsV51:
    temporal: np.ndarray
    context: np.ndarray
    target: np.ndarray
    raw_g3: np.ndarray


@dataclass
class UCIFitResultV51:
    probability: np.ndarray
    regression: np.ndarray
    ordinal_probability: np.ndarray
    selected_epoch: int
    epochs_ran: int
    history: list[dict[str, float | int | bool | None]]
    parameter_count: int
    runtime_seconds: float
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    before_class_counts: dict[int, int]
    after_class_counts: dict[int, int]
    gate_stats: dict[str, float | bool | None]
    temporal_norm_mean: float
    context_norm_mean: float
    freeze_epochs: int


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def prepare_partition(
    data: UCIDataV51,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    *,
    fitted: ColumnTransformer | None = None,
) -> tuple[UCIInputsV51, ColumnTransformer]:
    transformer = fitted or context_preprocessor(include_absences=data.context_contract == "sensitivity_with_absences")
    if fitted is None:
        transformer.fit(data.context.iloc[fit_indices])
    context = transformer.transform(data.context.iloc[transform_indices]).astype(np.float32)
    return (
        UCIInputsV51(
            temporal=data.temporal[transform_indices].astype(np.float32),
            context=context,
            target=data.target[transform_indices].astype(np.int64),
            raw_g3=data.raw_g3[transform_indices].astype(np.float32),
        ),
        transformer,
    )


def _counts(target: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(target.astype(int), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def resample_deep_training(inputs: UCIInputsV51, strategy: str, seed: int) -> tuple[UCIInputsV51, dict[int, int], dict[int, int]]:
    before = _counts(inputs.target)
    if strategy in {"none", "class_weight", "focal"}:
        return inputs, before, before
    if strategy != "random_sample_duplication":
        raise KeyError(f"Unsupported deep UCI imbalance strategy: {strategy}")
    rng = np.random.default_rng(seed)
    maximum = max(before.values())
    selected: list[np.ndarray] = []
    for label in sorted(before):
        rows = np.flatnonzero(inputs.target == label)
        selected.append(rng.choice(rows, size=maximum, replace=len(rows) < maximum))
    indices = np.concatenate(selected)
    rng.shuffle(indices)
    sampled = UCIInputsV51(
        inputs.temporal[indices],
        inputs.context[indices],
        inputs.target[indices],
        inputs.raw_g3[indices],
    )
    return sampled, before, _counts(sampled.target)


def _loader(inputs: UCIInputsV51, batch_size: int, shuffle: bool, seed: int, pin: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(inputs.temporal),
            torch.from_numpy(inputs.context),
            torch.from_numpy(inputs.target),
            torch.from_numpy(inputs.raw_g3),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=pin,
        drop_last=False,
    )


def _class_weights(target: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(target.astype(int), minlength=3).astype(float)
    return torch.tensor(len(target) / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)


def multitask_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    raw_g3: torch.Tensor,
    *,
    config: dict[str, Any],
    class_weights: torch.Tensor | None,
    regression_mean: float,
    regression_std: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    classification = nn.functional.cross_entropy(
        output["classification"].float(),
        target,
        weight=class_weights,
        reduction="none",
        label_smoothing=float(config.get("label_smoothing", 0.0)),
    )
    if str(config.get("classification_loss", "standard")) == "focal":
        probability = torch.softmax(output["classification"].float(), dim=1)
        pt = probability.gather(1, target[:, None]).squeeze(1)
        classification = ((1.0 - pt) ** float(config.get("focal_gamma", 2.0))) * classification
    classification_loss = classification.mean()
    normalized_g3 = (raw_g3.float() - regression_mean) / regression_std
    regression_loss = nn.functional.huber_loss(output["regression"].float(), normalized_g3)
    ordinal_target = torch.stack([(target > 0).float(), (target > 1).float()], dim=1)
    ordinal_loss = nn.functional.binary_cross_entropy_with_logits(output["ordinal"].float(), ordinal_target)
    objective = str(config.get("objective", "classification_only"))
    regression_weight = float(config.get("regression_weight", 0.0))
    ordinal_weight = float(config.get("ordinal_weight", 0.0))
    if objective == "classification_only":
        regression_weight = 0.0
        ordinal_weight = 0.0
    elif objective == "classification_plus_huber_regression":
        ordinal_weight = 0.0
    elif objective != "classification_plus_huber_regression_plus_ordinal":
        raise ValueError(f"Unknown objective: {objective}")
    total = classification_loss + regression_weight * regression_loss + ordinal_weight * ordinal_loss
    return total, {
        "classification_loss": classification_loss,
        "regression_loss": regression_loss,
        "ordinal_loss": ordinal_loss,
    }


def encoder_parameter_names(model: UCIHybridV51) -> tuple[str, ...]:
    prefixes = ("temporal.", "context.", "temporal_projection.", "context_projection.", "gate.", "film.")
    return tuple(name for name, _ in model.named_parameters() if name.startswith(prefixes))


def encoder_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("temporal.", "context.", "temporal_projection.", "context_projection.", "gate.", "film.")
    return {name: value.detach().cpu().clone() for name, value in state.items() if name.startswith(prefixes)}


def set_encoder_trainable(model: UCIHybridV51, trainable: bool) -> None:
    names = set(encoder_parameter_names(model))
    for name, parameter in model.named_parameters():
        if name in names:
            parameter.requires_grad = trainable


def _load_encoder_state(model: UCIHybridV51, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    incompatible = [name for name, value in state.items() if name not in current or current[name].shape != value.shape]
    if incompatible:
        raise ValueError(f"Incompatible transferred encoder keys: {incompatible}")
    current.update(state)
    model.load_state_dict(current)


def _predict(
    model: UCIHybridV51,
    inputs: UCIInputsV51,
    batch_size: int,
    device: torch.device,
    regression_mean: float,
    regression_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | bool | None], float, float]:
    model.eval()
    probabilities: list[np.ndarray] = []
    regressions: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    gates: list[torch.Tensor] = []
    temporal_norms: list[np.ndarray] = []
    context_norms: list[np.ndarray] = []
    with torch.no_grad():
        for temporal, context, _, _ in _loader(inputs, batch_size, False, 0, device.type == "cuda"):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(temporal.to(device), context.to(device))
            probabilities.append(torch.softmax(output["classification"].float(), dim=1).cpu().numpy())
            regressions.append(
                (output["regression"].float().cpu().numpy() * regression_std + regression_mean).clip(0, 20)
            )
            ordinals.append(torch.sigmoid(output["ordinal"].float()).cpu().numpy())
            temporal_norms.append(output["temporal_norm"].float().cpu().numpy())
            context_norms.append(output["context_norm"].float().cpu().numpy())
            if "gate" in output:
                gates.append(output["gate"].float().cpu())
    probability = np.concatenate(probabilities).astype(float)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("UCI V5.1 probability contract failed")
    return (
        probability,
        np.concatenate(regressions).astype(float),
        np.concatenate(ordinals).astype(float),
        gate_statistics(torch.cat(gates) if gates else None),
        float(np.concatenate(temporal_norms).mean()),
        float(np.concatenate(context_norms).mean()),
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_uci_model_v5_1(
    train: UCIInputsV51,
    evaluation: UCIInputsV51,
    *,
    config: dict[str, Any],
    seed: int,
    imbalance_strategy: str = "none",
    fixed_epochs: int | None = None,
    device_name: str = "cuda",
    initial_encoder_state: dict[str, torch.Tensor] | None = None,
    freeze_epochs: int = 0,
) -> UCIFitResultV51:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    sampled, before, after = resample_deep_training(train, imbalance_strategy, seed)
    model = UCIHybridV51(sampled.temporal.shape[2], sampled.context.shape[1], config).to(device)
    if initial_encoder_state is not None:
        _load_encoder_state(model, initial_encoder_state)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError(f"UCI V5.1 parameter guard exceeded: {parameter_count}")
    encoder_names = set(encoder_parameter_names(model))
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if name in encoder_names]
    head_parameters = [parameter for name, parameter in model.named_parameters() if name not in encoder_names]
    learning_rate = float(config.get("learning_rate", 8e-4))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": learning_rate * float(config.get("encoder_learning_rate_fraction", 1.0)),
            },
            {"params": head_parameters, "lr": learning_rate},
        ],
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    epoch_limit = int(fixed_epochs or config.get("max_epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_limit)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    batch_size = int(config.get("batch_size", 64))
    patience = int(config.get("patience", 12))
    regression_mean = float(train.raw_g3.mean())
    regression_std = float(max(train.raw_g3.std(), 1e-6))
    class_weights = _class_weights(sampled.target, device) if imbalance_strategy == "class_weight" else None
    best_metric = -1.0
    best_nll = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int | bool | None]] = []
    stale = 0
    for epoch in range(1, epoch_limit + 1):
        frozen = epoch <= int(freeze_epochs)
        set_encoder_trainable(model, not frozen)
        model.train()
        losses: list[float] = []
        classification_losses: list[float] = []
        regression_losses: list[float] = []
        ordinal_losses: list[float] = []
        norms: list[float] = []
        for temporal, context, target, raw_g3 in _loader(
            sampled, batch_size, True, seed + epoch, device.type == "cuda"
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(temporal.to(device), context.to(device))
                loss, components = multitask_loss(
                    output,
                    target.to(device),
                    raw_g3.to(device),
                    config=config,
                    class_weights=class_weights,
                    regression_mean=regression_mean,
                    regression_std=regression_std,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite UCI V5.1 loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            classification_losses.append(float(components["classification_loss"].detach().cpu()))
            regression_losses.append(float(components["regression_loss"].detach().cpu()))
            ordinal_losses.append(float(components["ordinal_loss"].detach().cpu()))
            norms.append(float(norm.detach().cpu()))
        scheduler.step()
        probability, _, _, _, _, _ = _predict(
            model, evaluation, batch_size, device, regression_mean, regression_std
        )
        metric = float(f1_score(evaluation.target, probability.argmax(axis=1), average="macro", zero_division=0))
        validation_nll = float(-np.log(probability[np.arange(len(probability)), evaluation.target].clip(1e-7)).mean())
        history.append(
            {
                "epoch": epoch,
                "encoder_frozen": frozen,
                "train_loss": float(np.mean(losses)),
                "classification_loss": float(np.mean(classification_losses)),
                "regression_loss": float(np.mean(regression_losses)),
                "ordinal_loss": float(np.mean(ordinal_losses)),
                "validation_macro_f1": metric,
                "validation_nll": validation_nll,
                "gradient_norm": float(np.mean(norms)),
                "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "head_learning_rate": float(optimizer.param_groups[1]["lr"]),
            }
        )
        improved = metric > best_metric + 1e-6 or (abs(metric - best_metric) <= 1e-6 and validation_nll < best_nll)
        if fixed_epochs is not None or improved:
            best_metric = metric
            best_nll = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    set_encoder_trainable(model, True)
    model.load_state_dict(best_state)
    probability, regression, ordinal, gate_stats, temporal_norm, context_norm = _predict(
        model, evaluation, batch_size, device, regression_mean, regression_std
    )
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = UCIHybridV51(sampled.temporal.shape[2], sampled.context.shape[1], config).to(device)
    replay.load_state_dict(cpu_state)
    replay_probability, _, _, _, _, _ = _predict(
        replay, evaluation, batch_size, device, regression_mean, regression_std
    )
    replay_difference = float(np.max(np.abs(probability - replay_probability)))
    if replay_difference > 1e-6:
        raise RuntimeError(f"UCI V5.1 checkpoint replay failed: {replay_difference}")
    return UCIFitResultV51(
        probability=probability,
        regression=regression,
        ordinal_probability=ordinal,
        selected_epoch=best_epoch,
        epochs_ran=len(history),
        history=history,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=replay_difference,
        before_class_counts=before,
        after_class_counts=after,
        gate_stats=gate_stats,
        temporal_norm_mean=temporal_norm,
        context_norm_mean=context_norm,
        freeze_epochs=int(freeze_epochs),
    )


__all__ = [
    "UCIFitResultV51",
    "UCIInputsV51",
    "deterministic_seed",
    "encoder_parameter_names",
    "encoder_state_dict",
    "fit_uci_model_v5_1",
    "multitask_loss",
    "prepare_partition",
    "resample_deep_training",
    "set_encoder_trainable",
]

