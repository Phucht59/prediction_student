from __future__ import annotations

import copy
import hashlib
import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .uci_model import UCIHybridV51, count_parameters
from .uci_training import (
    UCIFitResultV51,
    UCIInputsV51,
    deterministic_seed,
    encoder_state_dict,
    fit_uci_model_v5_1,
    multitask_loss,
)


@dataclass(frozen=True)
class SharedSubjectInputs:
    temporal: np.ndarray
    context: np.ndarray
    target: np.ndarray
    raw_g3: np.ndarray
    subject: np.ndarray


@dataclass
class SharedSubjectFitResult:
    probability: np.ndarray
    selected_epoch: int
    history: list[dict[str, float | int]]
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    parameter_count: int
    runtime_seconds: float


def overlap_safe_source_indices(source_groups: np.ndarray, target_validation_groups: np.ndarray) -> np.ndarray:
    validation = set(map(str, target_validation_groups.tolist()))
    selected = np.array([str(group) not in validation for group in source_groups], dtype=bool)
    return np.flatnonzero(selected)


def combine_subject_inputs(mat: UCIInputsV51, por: UCIInputsV51) -> SharedSubjectInputs:
    if mat.context.shape[1] != por.context.shape[1] or mat.temporal.shape[2] != por.temporal.shape[2]:
        raise ValueError("Shared-subject inputs must use the same target-fitted feature contract")
    return SharedSubjectInputs(
        temporal=np.concatenate([mat.temporal, por.temporal]),
        context=np.concatenate([mat.context, por.context]),
        target=np.concatenate([mat.target, por.target]),
        raw_g3=np.concatenate([mat.raw_g3, por.raw_g3]),
        subject=np.concatenate(
            [np.zeros(len(mat.target), dtype=np.int64), np.ones(len(por.target), dtype=np.int64)]
        ),
    )


class SharedTrunkSubjectHeadsV51(nn.Module):
    """Shared temporal/context/fusion trunk with an embedding and separate subject heads."""

    def __init__(self, temporal_dim: int, context_dim: int, config: dict[str, Any]):
        super().__init__()
        self.trunk = UCIHybridV51(temporal_dim, context_dim, config)
        self.trunk.classifier = nn.Identity()
        self.trunk.regressor = nn.Identity()
        self.trunk.ordinal = nn.Identity()
        hidden = int(config.get("fusion_hidden", 32))
        embedding_dim = int(config.get("subject_embedding_dim", 4))
        self.subject_embedding = nn.Embedding(2, embedding_dim)
        self.subject_projection = nn.Sequential(nn.Linear(hidden + embedding_dim, hidden), nn.GELU())
        self.classifiers = nn.ModuleList([nn.Linear(hidden, 3), nn.Linear(hidden, 3)])
        self.regressors = nn.ModuleList([nn.Linear(hidden, 1), nn.Linear(hidden, 1)])
        self.ordinals = nn.ModuleList([nn.Linear(hidden, 2), nn.Linear(hidden, 2)])

    @staticmethod
    def _select(heads: nn.ModuleList, representation: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        candidates = torch.stack([head(representation) for head in heads], dim=1)
        shape = [len(subject), 1, *([1] * (candidates.ndim - 2))]
        index = subject.reshape(shape).expand(-1, 1, *candidates.shape[2:])
        return candidates.gather(1, index).squeeze(1)

    def forward(self, temporal: torch.Tensor, context: torch.Tensor, subject: torch.Tensor) -> dict[str, torch.Tensor]:
        representation, diagnostics = self.trunk.encode(temporal, context)
        representation = self.subject_projection(
            torch.cat([representation, self.subject_embedding(subject.long())], dim=1)
        )
        return {
            "classification": self._select(self.classifiers, representation, subject),
            "regression": self._select(self.regressors, representation, subject).squeeze(1),
            "ordinal": self._select(self.ordinals, representation, subject),
            **diagnostics,
        }


def pretrain_then_finetune(
    source_train: UCIInputsV51,
    source_validation: UCIInputsV51,
    target_train: UCIInputsV51,
    target_validation: UCIInputsV51,
    *,
    config: dict[str, Any],
    seed: int,
    source_epochs: int,
    target_epochs: int,
    freeze_epochs: int,
    device_name: str = "cuda",
    target_imbalance: str = "none",
) -> tuple[UCIFitResultV51, UCIFitResultV51]:
    pretrained = fit_uci_model_v5_1(
        source_train,
        source_validation,
        config=config,
        seed=seed,
        fixed_epochs=source_epochs,
        device_name=device_name,
    )
    fine_config = dict(config)
    fine_config["encoder_learning_rate_fraction"] = float(config.get("unfreeze_learning_rate_fraction", 0.35))
    fine_tuned = fit_uci_model_v5_1(
        target_train,
        target_validation,
        config=fine_config,
        seed=seed,
        imbalance_strategy=target_imbalance,
        fixed_epochs=target_epochs,
        device_name=device_name,
        initial_encoder_state=encoder_state_dict(pretrained.state_dict),
        freeze_epochs=freeze_epochs,
    )
    return pretrained, fine_tuned


def _shared_loader(inputs: SharedSubjectInputs, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(inputs.temporal),
            torch.from_numpy(inputs.context),
            torch.from_numpy(inputs.target),
            torch.from_numpy(inputs.raw_g3),
            torch.from_numpy(inputs.subject),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


def _shared_predict(
    model: SharedTrunkSubjectHeadsV51,
    inputs: SharedSubjectInputs,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for temporal, context, _, _, subject in _shared_loader(inputs, batch_size, False, 0):
            output = model(temporal.to(device), context.to(device), subject.to(device))
            rows.append(torch.softmax(output["classification"].float(), dim=1).cpu().numpy())
    probability = np.concatenate(rows).astype(float)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Shared-subject probability contract failed")
    return probability


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def fit_shared_subject_model(
    train: SharedSubjectInputs,
    evaluation: SharedSubjectInputs,
    *,
    config: dict[str, Any],
    seed: int,
    fixed_epochs: int,
    device_name: str = "cuda",
) -> SharedSubjectFitResult:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model = SharedTrunkSubjectHeadsV51(train.temporal.shape[2], train.context.shape[1], config).to(device)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError(f"Shared-subject parameter guard exceeded: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 8e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    batch_size = int(config.get("batch_size", 64))
    regression_mean = float(train.raw_g3.mean())
    regression_std = float(max(train.raw_g3.std(), 1e-6))
    best_metric = -1.0
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(fixed_epochs) + 1):
        model.train()
        losses: list[float] = []
        for temporal, context, target, raw_g3, subject in _shared_loader(
            train, batch_size, True, seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model(temporal.to(device), context.to(device), subject.to(device))
            loss, _ = multitask_loss(
                output,
                target.to(device),
                raw_g3.to(device),
                config=config,
                class_weights=None,
                regression_mean=regression_mean,
                regression_std=regression_std,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        probability = _shared_predict(model, evaluation, batch_size, device)
        metric = float(f1_score(evaluation.target, probability.argmax(axis=1), average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": metric})
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    probability = _shared_predict(model, evaluation, batch_size, device)
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = SharedTrunkSubjectHeadsV51(train.temporal.shape[2], train.context.shape[1], config).to(device)
    replay.load_state_dict(cpu_state)
    replay_probability = _shared_predict(replay, evaluation, batch_size, device)
    difference = float(np.max(np.abs(probability - replay_probability)))
    if difference > 1e-6:
        raise RuntimeError(f"Shared-subject checkpoint replay failed: {difference}")
    return SharedSubjectFitResult(
        probability=probability,
        selected_epoch=best_epoch,
        history=history,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=difference,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
    )


__all__ = [
    "SharedSubjectFitResult",
    "SharedSubjectInputs",
    "SharedTrunkSubjectHeadsV51",
    "combine_subject_inputs",
    "fit_shared_subject_model",
    "overlap_safe_source_indices",
    "pretrain_then_finetune",
]

