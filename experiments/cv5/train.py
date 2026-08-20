"""Hybrid CNN–BiLSTM trainer for 5-fold CV. Same public architecture; training-only improvements."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from experiments.imbalance.evaluation import metrics, select_stop_threshold
from experiments.imbalance.samplers import PackedBatch
from experiments.imbalance.train_hybrid import _batch_tensors, _device, make_config, predict_scores, seed_everything
from src.prediction.model import Hybrid


def _stage_prs(model: Hybrid, stages: dict[str, PackedBatch], *, batch_size: int) -> dict[str, float]:
    out = {}
    for name, batch in stages.items():
        if len(np.unique(batch.target)) < 2:
            continue
        scores = predict_scores(model, batch, batch_size=batch_size)
        out[name] = float(average_precision_score(batch.target, scores))
    return out


def _stop_objective(prs: dict[str, float]) -> float:
    if not prs:
        return -np.inf
    values = np.asarray(list(prs.values()), dtype=float)
    return float(values.mean() - 0.2 * values.std())


def train_cv5(
    train_stages: dict[str, PackedBatch],
    stop_stages: dict[str, PackedBatch],
    valid_stages: dict[str, PackedBatch],
    hparams: dict[str, Any],
    *,
    seed: int,
    max_epochs: int = 40,
    patience: int = 12,
    tabular_aux: float = 0.35,
    static_noise: float = 0.05,
    extra_s0: bool = True,
) -> dict[str, Any]:
    seed_everything(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = _device()
    first = next(iter(train_stages.values()))
    config = make_config(first.static.shape[1], first.temporal.shape[2], first.aggregate.shape[1], hparams)
    model = Hybrid(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hparams["lr"]), weight_decay=float(hparams["weight_decay"]))
    y_ref = np.concatenate([batch.target for batch in train_stages.values()])
    n_pos = max(1, int(y_ref.sum()))
    pos_weight = torch.tensor(
        [((len(y_ref) - int(y_ref.sum())) / n_pos) * float(hparams["pos_weight_multiplier"])],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = int(hparams["batch_size"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = -np.inf
    best_state = None
    stale = 0
    history = []
    stage_order = list(train_stages.keys())
    if extra_s0 and "S0" in train_stages:
        stage_order = ["S0", "S0"] + stage_order

    def _step(batch: PackedBatch, idx: np.ndarray, noise: bool) -> float:
        tensors = list(_batch_tensors(batch, idx, device))
        if noise and static_noise > 0:
            tensors[0] = tensors[0] + static_noise * torch.randn_like(tensors[0])
        labels = torch.tensor(batch.target[idx], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(*tensors)
            loss = loss_fn(logits, labels) + model.fusion_regularization()
            if tabular_aux > 0:
                zero_temp = tensors[1] * 0
                zero_mask = tensors[2] & False
                zero_len = torch.zeros_like(tensors[3])
                tab_logits = model(tensors[0], zero_temp, zero_mask, zero_len, tensors[4], tensors[5], tensors[6])
                loss = loss + tabular_aux * loss_fn(tab_logits, labels)
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
        return float(loss.detach().float().cpu())

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = []
        rng = np.random.default_rng(seed + epoch)
        for stage in stage_order:
            batch = train_stages[stage]
            n = len(batch.target)
            order = np.arange(n)
            rng.shuffle(order)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                epoch_loss.append(_step(batch, idx, noise=static_noise > 0))
        prs = _stage_prs(model, stop_stages, batch_size=batch_size)
        objective = _stop_objective(prs)
        history.append({"epoch": epoch, "loss": float(np.mean(epoch_loss)), "stop_objective": objective, **{f"stop_{k}": v for k, v in prs.items()}})
        if objective > best:
            best = objective
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
    valid_scores = {}
    for stage, valid in valid_stages.items():
        stop_scores = predict_scores(model, stop_stages[stage], batch_size=batch_size)
        threshold = select_stop_threshold(stop_stages[stage].target, stop_scores)
        scores = predict_scores(model, valid, batch_size=batch_size)
        stage_metrics[stage] = metrics(valid.target, scores, threshold=threshold)
        stage_metrics[stage]["n_valid"] = int(len(valid.target))
        valid_scores[stage] = scores
    return {
        "stage_metrics": stage_metrics,
        "valid_scores": valid_scores,
        "best_stop_objective": best,
        "epochs_run": len(history),
        "history": history,
        "architecture_id": model.config.architecture_id,
        "fusion": model.config.fusion,
        "model": model.cpu(),
    }


def finetune_weak_stage(
    model: Hybrid,
    train_stages: dict[str, PackedBatch],
    stop_stages: dict[str, PackedBatch],
    hparams: dict[str, Any],
    *,
    weak_stage: str,
    seed: int,
    max_epochs: int = 8,
    patience: int = 4,
) -> Hybrid:
    """Extra FIT updates on the weak information level. Checkpoint still chosen on all-stage STOP."""
    if weak_stage not in train_stages:
        return model
    device = _device()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(hparams["lr"]) * 0.5, weight_decay=float(hparams["weight_decay"]))
    y = train_stages[weak_stage].target
    n_pos = max(1, int(y.sum()))
    pos_weight = torch.tensor([((len(y) - int(y.sum())) / n_pos) * float(hparams["pos_weight_multiplier"])], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = int(hparams["batch_size"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = _stop_objective(_stage_prs(model, stop_stages, batch_size=batch_size))
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    batch = train_stages[weak_stage]
    for epoch in range(max_epochs):
        model.train()
        rng = np.random.default_rng(seed + 1000 + epoch)
        order = np.arange(len(batch.target))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            tensors = _batch_tensors(batch, idx, device)
            labels = torch.tensor(batch.target[idx], dtype=torch.float32, device=device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(*tensors)
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
        objective = _stop_objective(_stage_prs(model, stop_stages, batch_size=batch_size))
        if objective > best:
            best = objective
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model.cpu()

