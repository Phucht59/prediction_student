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
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .uci_model import DualBranchCNNBiLSTM, count_parameters


@dataclass
class UCIInputs:
    sequence: np.ndarray
    context: np.ndarray
    target: np.ndarray
    raw_g3: np.ndarray


@dataclass
class UCIFitResult:
    probability: np.ndarray
    regression: np.ndarray
    selected_epoch: int
    epochs_ran: int
    history: list[dict[str, float | int | None]]
    parameter_count: int
    runtime_seconds: float
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    before_class_counts: dict[int, int]
    after_class_counts: dict[int, int]


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _counts(values: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(values.astype(int), return_counts=True)
    return {int(key): int(count) for key, count in zip(unique, counts)}


def resample_training(inputs: UCIInputs, strategy: str, seed: int) -> tuple[UCIInputs, dict[int, int], dict[int, int]]:
    before = _counts(inputs.target)
    if strategy in {"none", "class_weight"}:
        return inputs, before, before
    samplers = {
        "random_oversampling": RandomOverSampler(random_state=seed),
        "smote": SMOTE(random_state=seed, k_neighbors=min(5, max(1, min(before.values()) - 1))),
        "adasyn": ADASYN(random_state=seed, n_neighbors=min(5, max(1, min(before.values()) - 1))),
    }
    if strategy not in samplers:
        raise KeyError(f"Unknown UCI imbalance strategy: {strategy}")
    matrix = np.concatenate(
        [inputs.sequence.reshape(len(inputs.target), -1), inputs.context, inputs.raw_g3[:, None]], axis=1
    )
    try:
        resampled, target = samplers[strategy].fit_resample(matrix, inputs.target)
    except ValueError as error:
        raise RuntimeError(f"{strategy} failed rather than silently becoming none: {error}") from error
    if len(resampled) <= len(matrix):
        raise RuntimeError(f"{strategy} did not add training observations")
    sequence = resampled[:, :2].reshape(-1, 2, 1).astype(np.float32)
    context = resampled[:, 2:-1].astype(np.float32)
    raw_g3 = resampled[:, -1].clip(0, 20).astype(np.float32)
    result = UCIInputs(sequence, context, target.astype(np.int64), raw_g3)
    return result, before, _counts(result.target)


def _class_weights(target: np.ndarray, strategy: str, device: torch.device) -> torch.Tensor | None:
    if strategy != "class_weight":
        return None
    counts = np.bincount(target.astype(int), minlength=3).astype(float)
    weights = len(target) / (3 * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _loader(inputs: UCIInputs, batch_size: int, shuffle: bool, seed: int, pin: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(
        torch.from_numpy(inputs.sequence),
        torch.from_numpy(inputs.context),
        torch.from_numpy(inputs.target),
        torch.from_numpy(inputs.raw_g3),
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


def _predict(
    model: DualBranchCNNBiLSTM,
    inputs: UCIInputs,
    batch_size: int,
    device: torch.device,
    regression_mean: float,
    regression_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    regression: list[np.ndarray] = []
    with torch.no_grad():
        for sequence, context, _, _ in _loader(inputs, batch_size, False, 0, device.type == "cuda"):
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, estimate = model(sequence.to(device), context.to(device))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            regression.append((estimate.float().cpu().numpy() * regression_std + regression_mean).clip(0, 20))
    probability = np.concatenate(probabilities).astype(float)
    estimate = np.concatenate(regression).astype(float)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("UCI probability contract failed")
    return probability, estimate


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_uci_model(
    train: UCIInputs,
    evaluation: UCIInputs,
    *,
    config: dict[str, Any],
    seed: int,
    imbalance_strategy: str = "none",
    fixed_epochs: int | None = None,
    device_name: str = "cuda",
    initial_state: dict[str, torch.Tensor] | None = None,
) -> UCIFitResult:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    sampled, before, after = resample_training(train, imbalance_strategy, seed)
    regression_mean = float(train.raw_g3.mean())
    regression_std = float(max(train.raw_g3.std(), 1e-6))
    model = DualBranchCNNBiLSTM(sampled.context.shape[1], config).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError(f"UCI model parameter guard exceeded: {parameter_count}")
    criterion = nn.CrossEntropyLoss(weight=_class_weights(sampled.target, imbalance_strategy, device))
    regression_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    max_epochs = int(config.get("max_epochs", 100))
    epoch_limit = int(fixed_epochs or max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_limit)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    alpha = float(config.get("multitask_alpha", 0.0))
    patience = int(config.get("patience", 12))
    batch_size = int(config.get("batch_size", 64))
    best_metric = -1.0
    best_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int | None]] = []
    stale = 0
    for epoch in range(1, epoch_limit + 1):
        model.train()
        losses: list[float] = []
        classification_losses: list[float] = []
        regression_losses: list[float] = []
        norms: list[float] = []
        for sequence, context, target, raw_g3 in _loader(
            sampled, batch_size, True, seed + epoch, device.type == "cuda"
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits, estimate = model(sequence.to(device), context.to(device))
                classification_loss = criterion(logits.float(), target.to(device))
                normalized_g3 = (raw_g3.to(device) - regression_mean) / regression_std
                regression_loss = regression_criterion(estimate.float(), normalized_g3)
                loss = classification_loss + alpha * regression_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite UCI loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            classification_losses.append(float(classification_loss.detach().cpu()))
            regression_losses.append(float(regression_loss.detach().cpu()))
            norms.append(float(norm.detach().cpu()))
        scheduler.step()
        probability, _ = _predict(model, evaluation, batch_size, device, regression_mean, regression_std)
        prediction = probability.argmax(axis=1)
        metric = float(f1_score(evaluation.target, prediction, average="macro", zero_division=0))
        validation_loss = float(-np.log(probability[np.arange(len(probability)), evaluation.target].clip(1e-7)).mean())
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "classification_loss": float(np.mean(classification_losses)),
                "regression_loss": float(np.mean(regression_losses)),
                "validation_macro_f1": metric,
                "validation_nll": validation_loss,
                "gradient_norm": float(np.mean(norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        improved = metric > best_metric + 1e-6 or (abs(metric - best_metric) <= 1e-6 and validation_loss < best_loss)
        if fixed_epochs is not None or improved:
            best_metric = metric
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if fixed_epochs is None and stale >= patience:
            break
    model.load_state_dict(best_state)
    probability, regression = _predict(model, evaluation, batch_size, device, regression_mean, regression_std)
    cpu_state = {name: tensor.detach().cpu().clone() for name, tensor in best_state.items()}
    replay = DualBranchCNNBiLSTM(sampled.context.shape[1], config).to(device)
    replay.load_state_dict(cpu_state)
    replay_probability, _ = _predict(replay, evaluation, batch_size, device, regression_mean, regression_std)
    replay_difference = float(np.max(np.abs(probability - replay_probability)))
    if replay_difference > 1e-6:
        raise RuntimeError(f"UCI checkpoint replay failed: {replay_difference}")
    return UCIFitResult(
        probability=probability,
        regression=regression,
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
    )


__all__ = ["UCIFitResult", "UCIInputs", "deterministic_seed", "fit_uci_model", "resample_training"]
