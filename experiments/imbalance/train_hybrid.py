"""Train frozen Hybrid CNN–BiLSTM numerics. Experiment-only; does not write production checkpoints."""
from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.prediction.model import Hybrid, HybridConfig

from experiments.imbalance.evaluation import metrics, select_stop_threshold
from experiments.imbalance.samplers import PackedBatch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch_tensors(batch: PackedBatch, idx: np.ndarray, device: torch.device):
    return (
        torch.tensor(batch.static[idx], dtype=torch.float32, device=device),
        torch.tensor(batch.temporal[idx], dtype=torch.float32, device=device),
        torch.tensor(batch.temporal_mask[idx], dtype=torch.bool, device=device),
        torch.tensor(batch.lengths[idx], dtype=torch.long, device=device),
        torch.tensor(batch.aggregate[idx], dtype=torch.float32, device=device),
        torch.tensor(batch.aggregate_available[idx], dtype=torch.float32, device=device),
        torch.tensor(batch.progress[idx], dtype=torch.float32, device=device),
    )


@torch.no_grad()
def predict_scores(model: Hybrid, batch: PackedBatch, *, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    out = []
    n = len(batch.target)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(n, start + batch_size))
        logits = model(*_batch_tensors(batch, idx, device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def make_config(static_dim: int, temporal_dim: int, aggregate_dim: int, hparams: dict[str, Any]) -> HybridConfig:
    return HybridConfig(
        static_dim=static_dim,
        temporal_dim=temporal_dim,
        aggregate_dim=aggregate_dim,
        d_fuse=128,
        cnn_channels=64,
        cnn_blocks=2,
        cnn_kernel_size=2,
        cnn_dilations=(1, 2),
        bilstm_hidden=128,
        dropout=float(hparams["dropout"]),
        entropy_floor_coefficient=float(hparams["entropy_floor_coefficient"]),
        fusion="softmax_3way",
    )


def _macro_pr(model: Hybrid, stages: dict[str, PackedBatch], *, batch_size: int) -> float:
    values = []
    for batch in stages.values():
        if len(np.unique(batch.target)) < 2:
            continue
        scores = predict_scores(model, batch, batch_size=batch_size)
        values.append(float(average_precision_score(batch.target, scores)))
    if not values:
        raise RuntimeError("no STOP stage with both classes")
    return float(np.mean(values))


def train_one(
    train_stages: dict[str, PackedBatch],
    stop_stages: dict[str, PackedBatch],
    valid_stages: dict[str, PackedBatch],
    hparams: dict[str, Any],
    *,
    seed: int,
    max_epochs: int = 24,
    patience: int = 8,
    original_target: np.ndarray | None = None,
    keep_model: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    device = _device()
    first = next(iter(train_stages.values()))
    config = make_config(first.static.shape[1], first.temporal.shape[2], first.aggregate.shape[1], hparams)
    model = Hybrid(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
    )
    y_ref = np.asarray(
        original_target if original_target is not None else np.concatenate([batch.target for batch in train_stages.values()]),
        dtype=np.int64,
    )
    n_pos = max(1, int(y_ref.sum()))
    base = float((len(y_ref) - int(y_ref.sum())) / n_pos)
    pos_weight = torch.tensor([base * float(hparams["pos_weight_multiplier"])], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = int(hparams["batch_size"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = -np.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = []
        rng = np.random.default_rng(seed + epoch)
        for batch in train_stages.values():
            n = len(batch.target)
            order = np.arange(n)
            rng.shuffle(order)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(*_batch_tensors(batch, idx, device))
                    labels = torch.tensor(batch.target[idx], dtype=torch.float32, device=device)
                    loss = loss_fn(logits, labels) + model.fusion_regularization()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                epoch_loss.append(float(loss.detach().float().cpu()))
        stop_pr = _macro_pr(model, stop_stages, batch_size=batch_size)
        history.append({"epoch": epoch, "loss": float(np.mean(epoch_loss)), "stop_pr_auc": stop_pr})
        if stop_pr > best:
            best = stop_pr
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    stage_metrics = {}
    valid_score_map = {}
    stop_score_map = {}
    stop_threshold_map = {}
    for stage, valid in valid_stages.items():
        stop = stop_stages[stage]
        stop_scores = predict_scores(model, stop, batch_size=batch_size)
        threshold = select_stop_threshold(stop.target, stop_scores)
        valid_scores = predict_scores(model, valid, batch_size=batch_size)
        stage_metrics[stage] = metrics(valid.target, valid_scores, threshold=threshold)
        stage_metrics[stage]["n_train"] = int(len(train_stages[stage].target))
        stage_metrics[stage]["n_valid"] = int(len(valid.target))
        valid_score_map[stage] = valid_scores
        stop_score_map[stage] = stop_scores
        stop_threshold_map[stage] = threshold
    payload = {
        "stage_metrics": stage_metrics,
        "valid_scores": valid_score_map,
        "stop_scores": stop_score_map,
        "stop_thresholds": stop_threshold_map,
        "best_stop_pr_auc": best,
        "epochs_run": len(history),
        "history": history,
        "device": str(device),
        "availability_s0": _s0_gate(model, valid_stages),
        "pos_weight": float(pos_weight.detach().cpu().item()),
        "pos_weight_from_original_fit": original_target is not None,
    }
    if keep_model:
        payload["model"] = model.cpu()
    return payload


def _s0_gate(model: Hybrid, valid_stages: dict[str, PackedBatch]) -> dict[str, float] | None:
    batch = valid_stages.get("S0")
    if batch is None or len(batch.target) == 0:
        return None
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        idx = np.arange(min(8, len(batch.target)))
        _ = model(*_batch_tensors(batch, idx, device))
        weights = model.last_diagnostics["gate_weights"].cpu().numpy()
    return {
        "tabular_mass_mean": float(weights[:, 0].mean()),
        "cnn_mass_mean": float(weights[:, 1].mean()),
        "bilstm_mass_mean": float(weights[:, 2].mean()),
    }


__all__ = ["make_config", "predict_scores", "train_one"]
