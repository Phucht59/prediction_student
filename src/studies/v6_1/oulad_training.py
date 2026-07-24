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

from src.studies.v5_1.oulad.data import OULADInputsV51

from .oulad_architecture import CandidateSpec, OULADArchitectureDiagnosisNet


@dataclass
class DiagnosisFitResult:
    probability: np.ndarray
    selected_epoch: int
    history: list[dict[str, float | int]]
    state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    replay_max_abs_difference: float
    parameter_count: int
    runtime_seconds: float
    gpu_peak_memory_bytes: int
    diagnostics: dict[str, Any]


def deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def transform_order(
    inputs: OULADInputsV51, order: str, seed: int
) -> OULADInputsV51:
    if order == "original":
        return inputs
    if order not in {"reversed", "shuffled", "bag_of_weeks"}:
        raise ValueError(order)
    sequence = inputs.sequence.copy()
    rng = np.random.default_rng(seed)
    for row, length_value in enumerate(inputs.lengths):
        length = int(length_value)
        valid = sequence[row, :length].copy()
        if order == "reversed":
            sequence[row, :length] = valid[::-1]
        elif order == "shuffled":
            sequence[row, :length] = valid[rng.permutation(length)]
        else:
            sequence[row, :length] = valid.mean(axis=0, keepdims=True)
    sequence *= inputs.mask[..., None]
    return replace(inputs, sequence=sequence.astype(np.float32))


def _loader(
    inputs: OULADInputsV51,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
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
        pin_memory=pin_memory,
        drop_last=False,
    )


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _predict(
    model: OULADArchitectureDiagnosisNet,
    inputs: OULADInputsV51,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    model.eval()
    probability: list[np.ndarray] = []
    values: dict[str, list[np.ndarray]] = {
        "gates": [],
        "cnn_norm": [],
        "lstm_norm": [],
        "expert_cosine": [],
    }
    with torch.no_grad():
        for sequence, lengths, mask, aggregate, static, _ in _loader(
            inputs, batch_size, False, 0, device.type == "cuda"
        ):
            logits, diagnostic = model(
                sequence.to(device),
                lengths.to(device),
                mask.to(device),
                aggregate.to(device),
                static.to(device),
                return_diagnostics=True,
            )
            probability.append(torch.sigmoid(logits.float()).cpu().numpy())
            for name in values:
                item = diagnostic[name]
                if item is not None:
                    values[name].append(item.float().cpu().numpy())
    result = np.concatenate(probability).astype(float)
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise RuntimeError("Invalid diagnosis probability contract")
    diagnostics: dict[str, Any] = {}
    for name, parts in values.items():
        if not parts:
            diagnostics[name] = None
            continue
        joined = np.concatenate(parts)
        diagnostics[name] = {
            "mean": float(joined.mean()),
            "std": float(joined.std()),
            "min": float(joined.min()),
            "max": float(joined.max()),
        }
        if name == "gates":
            diagnostics[name]["per_branch_mean"] = joined.mean(axis=0).tolist()
    return result, diagnostics


def fit_diagnosis_model(
    train: OULADInputsV51,
    evaluation: OULADInputsV51,
    *,
    config: dict[str, Any],
    spec: CandidateSpec,
    seed: int,
    device_name: str = "cuda",
    fixed_epochs: int | None = None,
) -> DiagnosisFitResult:
    started = time.perf_counter()
    deterministic_seed(seed)
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = OULADArchitectureDiagnosisNet(
        train.sequence.shape[2],
        train.aggregate.shape[1],
        train.static.shape[1],
        config,
        spec,
    ).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epoch_limit = int(fixed_epochs or config["max_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_limit)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    batch_size = int(config["batch_size"])
    patience = int(config["patience"])
    best_metric = -1.0
    best_nll = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epoch_limit + 1):
        model.train()
        losses: list[float] = []
        norms: list[float] = []
        for sequence, lengths, mask, aggregate, static, target in _loader(
            train, batch_size, True, seed + epoch, device.type == "cuda"
        ):
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                logits = model(
                    sequence.to(device),
                    lengths.to(device),
                    mask.to(device),
                    aggregate.to(device),
                    static.to(device),
                )
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), target.to(device).float()
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite architecture diagnosis loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            norms.append(float(norm.detach().cpu()))
        scheduler.step()
        probability, _ = _predict(model, evaluation, batch_size, device)
        metric = float(
            f1_score(
                evaluation.target,
                probability >= 0.5,
                average="macro",
                zero_division=0,
            )
        )
        clipped = probability.clip(1e-7, 1 - 1e-7)
        nll = float(
            -(
                evaluation.target * np.log(clipped)
                + (1 - evaluation.target) * np.log(1 - clipped)
            ).mean()
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_macro_f1_at_0_5": metric,
                "validation_nll": nll,
                "gradient_norm": float(np.mean(norms)),
            }
        )
        improved = metric > best_metric + 1e-6 or (
            abs(metric - best_metric) <= 1e-6 and nll < best_nll
        )
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
    probability, diagnostics = _predict(model, evaluation, batch_size, device)
    cpu_state = {
        name: value.detach().cpu().clone() for name, value in best_state.items()
    }
    replay = OULADArchitectureDiagnosisNet(
        train.sequence.shape[2],
        train.aggregate.shape[1],
        train.static.shape[1],
        config,
        spec,
    ).to(device)
    replay.load_state_dict(cpu_state)
    replay_probability, _ = _predict(replay, evaluation, batch_size, device)
    difference = float(np.max(np.abs(probability - replay_probability)))
    if difference > 1e-6:
        raise RuntimeError(f"Architecture diagnosis replay failed: {difference}")
    return DiagnosisFitResult(
        probability=probability,
        selected_epoch=best_epoch,
        history=history,
        state_dict=cpu_state,
        checkpoint_sha256=_state_hash(cpu_state),
        replay_max_abs_difference=difference,
        parameter_count=parameter_count,
        runtime_seconds=time.perf_counter() - started,
        gpu_peak_memory_bytes=(
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        diagnostics=diagnostics,
    )


__all__ = [
    "DiagnosisFitResult",
    "deterministic_seed",
    "fit_diagnosis_model",
    "transform_order",
]

