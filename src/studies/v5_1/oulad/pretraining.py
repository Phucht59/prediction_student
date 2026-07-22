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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import OULADInputsV51
from .models import OULADTemporalEncoderV51, count_parameters


RECONSTRUCTION_CHANNELS = (
    "total_clicks",
    "active_days",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
    "submitted_assessment_count",
    "weeks_without_activity",
)


@dataclass
class OULADPretrainResult:
    state_dict: dict[str, torch.Tensor]
    temporal_state_dict: dict[str, torch.Tensor]
    checkpoint_sha256: str
    history: list[dict[str, float | int]]
    runtime_seconds: float
    parameter_count: int
    masked_values: int
    replay_max_abs_difference: float


def deterministic_week_mask(valid_mask: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("Mask fraction must be inside (0,1)")
    rng = np.random.default_rng(seed)
    valid = np.asarray(valid_mask, dtype=bool)
    selected = np.zeros_like(valid, dtype=bool)
    for row in range(len(valid)):
        positions = np.flatnonzero(valid[row])
        if len(positions) <= 1:
            continue
        count = min(len(positions) - 1, max(1, int(round(len(positions) * fraction))))
        selected[row, rng.choice(positions, size=count, replace=False)] = True
    if (selected & ~valid).any() or np.any(selected.sum(axis=1) >= valid.sum(axis=1)):
        raise RuntimeError("Invalid masked-week selection")
    return selected


class MaskedWeekReconstructorV51(nn.Module):
    def __init__(self, input_channels: int, reconstructed_channels: tuple[int, ...], config: dict[str, Any]):
        super().__init__()
        self.encoder = OULADTemporalEncoderV51(input_channels, config, "cnn_bilstm")
        self.reconstructed_channels = reconstructed_channels
        self.reconstruction_head = nn.Linear(self.encoder.sequence_output_dim, len(reconstructed_channels))

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        values = self.encoder.encode_sequence(sequence, lengths, valid_mask)
        return self.reconstruction_head(values)


def reconstruction_channel_indices(channel_order: tuple[str, ...]) -> tuple[int, ...]:
    missing = sorted(set(RECONSTRUCTION_CHANNELS) - set(channel_order))
    if missing:
        raise ValueError(f"Missing reconstruction channels: {missing}")
    return tuple(channel_order.index(name) for name in RECONSTRUCTION_CHANNELS)


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target_sequence: torch.Tensor,
    week_mask: torch.Tensor,
    channel_indices: tuple[int, ...],
) -> torch.Tensor:
    if not week_mask.bool().any():
        raise ValueError("At least one valid week must be masked")
    target = target_sequence[:, :, list(channel_indices)]
    selected = week_mask.bool().unsqueeze(-1).expand_as(target)
    return nn.functional.huber_loss(prediction.masked_select(selected), target.masked_select(selected))


def temporal_state_dict(pretrain_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "encoder."
    return {
        f"temporal.{name[len(prefix):]}": value.detach().cpu().clone()
        for name, value in pretrain_state.items()
        if name.startswith(prefix)
    }


def load_pretrained_temporal(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    incompatible = [name for name, value in state.items() if name not in current or current[name].shape != value.shape]
    if incompatible:
        raise ValueError(f"Incompatible OULAD pretrained keys: {incompatible}")
    current.update(state)
    model.load_state_dict(current)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def fit_masked_week_pretraining(
    train: OULADInputsV51,
    *,
    dynamic_channel_order: tuple[str, ...],
    config: dict[str, Any],
    seed: int,
    epochs: int,
    mask_fraction: float,
    device_name: str = "cuda",
) -> OULADPretrainResult:
    started = time.perf_counter()
    _seed(seed)
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    channel_indices = reconstruction_channel_indices(dynamic_channel_order)
    model = MaskedWeekReconstructorV51(train.sequence.shape[2], channel_indices, config).to(device)
    parameter_count = count_parameters(model)
    if parameter_count >= int(config.get("parameter_limit", 1_500_000)):
        raise RuntimeError(f"OULAD pretraining parameter guard exceeded: {parameter_count}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("pretraining_learning_rate", config.get("learning_rate", 5e-4))),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    dataset = TensorDataset(
        torch.from_numpy(train.sequence),
        torch.from_numpy(train.lengths),
        torch.from_numpy(train.mask),
    )
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    total_masked = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses: list[float] = []
        masked_values = 0
        loader = DataLoader(
            dataset,
            batch_size=int(config.get("batch_size", 256)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + epoch),
            num_workers=0,
            drop_last=False,
        )
        offset = 0
        for sequence, lengths, valid_mask in loader:
            batch_mask = deterministic_week_mask(valid_mask.numpy(), mask_fraction, seed + epoch * 1_000_003 + offset)
            offset += len(sequence)
            if not batch_mask.any():
                continue
            corrupted = sequence.clone()
            corrupted[torch.from_numpy(batch_mask)] = 0.0
            optimizer.zero_grad(set_to_none=True)
            prediction = model(corrupted.to(device), lengths.to(device), valid_mask.to(device))
            loss = masked_reconstruction_loss(
                prediction,
                sequence.to(device),
                torch.from_numpy(batch_mask).to(device),
                channel_indices,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite masked-week reconstruction loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip", 1.0)))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            masked_values += int(batch_mask.sum()) * len(channel_indices)
        if not losses:
            raise RuntimeError("No valid weeks were available for pretraining")
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "reconstruction_loss": mean_loss, "masked_values": masked_values})
        total_masked += masked_values
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    replay = MaskedWeekReconstructorV51(train.sequence.shape[2], channel_indices, config).to(device)
    replay.load_state_dict(cpu_state)
    model.eval()
    replay.eval()
    sample = torch.from_numpy(train.sequence[: min(8, len(train.sequence))]).to(device)
    lengths = torch.from_numpy(train.lengths[: len(sample)]).to(device)
    mask = torch.from_numpy(train.mask[: len(sample)]).to(device)
    with torch.no_grad():
        first = model(sample, lengths, mask)
        second = replay(sample, lengths, mask)
    difference = float((first - second).abs().max().cpu())
    if difference > 1e-6:
        raise RuntimeError(f"OULAD pretraining replay failed: {difference}")
    return OULADPretrainResult(
        state_dict=cpu_state,
        temporal_state_dict=temporal_state_dict(cpu_state),
        checkpoint_sha256=_state_hash(cpu_state),
        history=history,
        runtime_seconds=time.perf_counter() - started,
        parameter_count=parameter_count,
        masked_values=total_masked,
        replay_max_abs_difference=difference,
    )


__all__ = [
    "MaskedWeekReconstructorV51",
    "OULADPretrainResult",
    "RECONSTRUCTION_CHANNELS",
    "deterministic_week_mask",
    "fit_masked_week_pretraining",
    "load_pretrained_temporal",
    "masked_reconstruction_loss",
    "reconstruction_channel_indices",
    "temporal_state_dict",
]

