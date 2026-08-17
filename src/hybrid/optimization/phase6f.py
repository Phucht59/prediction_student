"""Phase 6F AP-oriented fine-tuning on sealed UCI inner-development data."""
from __future__ import annotations

import copy
import math
import time
from typing import Iterable

import numpy as np
import torch
from libauc.losses import APLoss
from libauc.optimizers import SOAP

from src.hybrid.models import SharedHeadHybrid
from src.hybrid.optimization.phase6 import STAGES, _metrics, class_pos_weight, predict
from src.hybrid.optimization.phase6c import multistage_arrays
from src.hybrid.optimization.phase6e import shared_config
from src.hybrid.training.evaluation import binary_classification_metrics
from src.hybrid.training.trainer import seed_everything

FLOORS = {
    "S0": {"pr_auc": .4592734340, "risk_recall": .5289473684, "risk_f1": .4480721585},
    "S1": {"pr_auc": .8211933451, "risk_recall": .8622807018, "risk_f1": .6868060536},
    "S2": {"pr_auc": .8971660889, "risk_recall": .8505847953, "risk_f1": .7581128748},
}
GRID = tuple(round(x / 100, 2) for x in range(5, 96))
SEARCH = {
    "ap_lr": (1e-5, 5e-4), "ap_weight_decay": (1e-7, 1e-3),
    "ap_epochs": (5, 10, 15, 20, 30), "alpha": (.5, .75, 1.0),
    "gradient_clip": (.5, 1.0, 2.0), "gamma": (.7, .9),
}


def make_stage_ap_losses(data_len: int, device: str, gamma: float = .9):
    """Independent stateful APLoss instances; state is never shared across stages."""
    return {stage: APLoss(data_len=data_len, gamma=gamma, margin=1.0, device=device)
            for stage in STAGES["uci"]}


def recall_constrained_threshold(target, score, stage):
    """Choose an operational threshold from STOP data, with diagnostic fallback."""
    rows = [binary_classification_metrics(target, score, threshold=t) for t in GRID]
    floor = FLOORS[stage]
    feasible = [m for m in rows if m["risk_recall"] >= floor["risk_recall"]
                and m["risk_f1"] >= floor["risk_f1"]]
    if feasible:
        chosen = max(feasible, key=lambda m: (m["risk_f1"], m["risk_precision"],
                                               -abs(m["threshold"] - .5), -m["threshold"]))
        return chosen, True, None
    recall_ok = [m for m in rows if m["risk_recall"] >= floor["risk_recall"]]
    fallback = (max(recall_ok, key=lambda m: (m["risk_f1"], m["risk_precision"],
                                              -abs(m["threshold"] - .5), -m["threshold"]))
                if recall_ok else
                max(rows, key=lambda m: (m["risk_recall"], m["risk_f1"], -m["threshold"])))
    return fallback, False, "STOP_THRESHOLD_INFEASIBLE"


def positive_safe_batches(target: np.ndarray, batch_size: int, seed: int, epoch: int) -> list[np.ndarray]:
    """Use every training record once while guaranteeing an AP-valid positive per batch."""
    target = np.asarray(target, dtype=int)
    n_batches = math.ceil(len(target) / batch_size)
    positives = np.flatnonzero(target == 1); negatives = np.flatnonzero(target == 0)
    if len(positives) < n_batches:
        raise RuntimeError("APLoss cannot form positive-safe batches without resampling")
    rng = np.random.default_rng(seed + epoch * 1009)
    positives = rng.permutation(positives); negatives = rng.permutation(negatives)
    reserved = positives[:n_batches]
    remainder = rng.permutation(np.concatenate([positives[n_batches:], negatives]))
    batches, cursor = [], 0
    for reserved_positive in reserved:
        take = min(batch_size - 1, len(remainder) - cursor)
        batch = np.concatenate([[reserved_positive], remainder[cursor:cursor + take]])
        cursor += take; batches.append(rng.permutation(batch))
    if cursor != len(remainder):
        raise RuntimeError("positive-safe batching dropped training records")
    return batches


def ap_blended_stage_loss(logits, target, sample_index, ap_loss, bce_loss, alpha: float):
    ap = ap_loss(torch.sigmoid(logits.float()), target.float(), sample_index)
    bce = bce_loss(logits.float(), target.float())
    return float(alpha) * ap + (1.0 - float(alpha)) * bce


def _target(stages, stage, ids):
    data = stages[stage]
    return np.asarray([data.target[data.index[r]] for r in ids], dtype=int)


def fine_tune_development_fold(stages, fit_ids, stop_ids, valid_by_stage, contexts,
                               temporal_dim, context_dim, pretrain_params, checkpoint,
                               fine_tune_params, seed=42):
    """Load a frozen pretrain checkpoint and AP-fine-tune using FIT/STOP only."""
    if set(fit_ids) & set(stop_ids): raise RuntimeError("fit/stop leakage")
    valid_ids = set().union(*map(set, valid_by_stage.values()))
    if (set(fit_ids) | set(stop_ids)) & valid_ids: raise RuntimeError("training/evaluation leakage")
    seed_everything(seed); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    model = SharedHeadHybrid(shared_config("uci", temporal_dim, context_dim, pretrain_params)).cuda()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True); model.load_state_dict(payload["model_state"])
    nparams = sum(p.numel() for p in model.parameters())
    if nparams != 494795: raise RuntimeError(f"Phase6F frozen C2 parameter count changed: {nparams}")
    optimizer = SOAP(model.parameters(), lr=float(fine_tune_params["ap_lr"]), mode="adam",
                     clip_value=float(fine_tune_params["gradient_clip"]),
                     weight_decay=float(fine_tune_params["ap_weight_decay"]), device="cuda", verbose=False)
    target = _target(stages, "S0", fit_ids); pos_weight = class_pos_weight(target, pretrain_params["class_weight_mode"])
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device="cuda"))
    ap_losses = make_stage_ap_losses(len(fit_ids), "cuda", float(fine_tune_params["gamma"]))
    arrays = {s: tuple(torch.from_numpy(x) for x in values)
              for s, values in multistage_arrays(stages, fit_ids, contexts).items()}
    best, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(int(fine_tune_params["ap_epochs"])):
        model.train()
        for batch in positive_safe_batches(target, int(pretrain_params["batch_size"]), seed, epoch):
            index = torch.as_tensor(batch, dtype=torch.long, device="cuda")
            optimizer.zero_grad(set_to_none=True); losses = []
            for stage in STAGES["uci"]:
                temporal, mask, lengths, context, batch_target, stage_index = (
                    tensor[batch].cuda(non_blocking=True) for tensor in arrays[stage])
                logits = model(temporal, mask, lengths, context, stage_index)
                losses.append(ap_blended_stage_loss(logits, batch_target, index, ap_losses[stage], bce,
                                                     fine_tune_params["alpha"]))
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss): raise RuntimeError("non_finite_ap_loss")
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(fine_tune_params["gradient_clip"]))
            if not torch.isfinite(norm): raise RuntimeError("non_finite_ap_gradient")
            optimizer.step()
        stop_by = {s: [r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]}
        stop_metrics = _metrics(model, "uci", stages, stop_by, contexts, pretrain_params["batch_size"])
        macro = float(np.mean([stop_metrics[s]["pr_auc"] for s in STAGES["uci"]]))
        if macro > best:
            best, best_epoch = macro, epoch + 1
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is None: raise RuntimeError("no_finite_ap_epoch")
    model.load_state_dict(best_state)
    train_by = {s: [r for r in fit_ids if r in stages[s].index] for s in STAGES["uci"]}
    stop_by = {s: [r for r in stop_ids if r in stages[s].index] for s in STAGES["uci"]}
    train_metrics = _metrics(model, "uci", stages, train_by, contexts, pretrain_params["batch_size"])
    rows = []
    for stage in STAGES["uci"]:
        stop_score = predict(model, "uci", stages, stage, stop_by[stage], contexts, pretrain_params["batch_size"])
        valid_score = predict(model, "uci", stages, stage, valid_by_stage[stage], contexts, pretrain_params["batch_size"])
        selected, feasible, reason = recall_constrained_threshold(_target(stages, stage, stop_by[stage]), stop_score, stage)
        valid = binary_classification_metrics(_target(stages, stage, valid_by_stage[stage]), valid_score,
                                              threshold=selected["threshold"])
        rows.append({"stage": stage, **valid, "selected_threshold": selected["threshold"],
                     "stop_threshold_feasible": feasible, "threshold_reason": reason,
                     "stop_metrics": selected, "train_pr_auc": train_metrics[stage]["pr_auc"],
                     "validation_pr_auc": valid["pr_auc"],
                     "train_validation_gap": train_metrics[stage]["pr_auc"] - valid["pr_auc"]})
    train_macro = float(np.mean([train_metrics[s]["pr_auc"] for s in STAGES["uci"]])); valid_macro = float(np.mean([r["pr_auc"] for r in rows]))
    result = {"best_epoch": best_epoch, "parameter_count": nparams,
              "train_macro_pr_auc": train_macro, "validation_macro_pr_auc": valid_macro,
              "train_validation_gap": train_macro - valid_macro, "runtime_seconds": time.monotonic() - started,
              "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "rows": rows}
    del model, optimizer, bce, ap_losses; torch.cuda.empty_cache(); return result


def passed_primary_gates(stage_rows: Iterable[dict]) -> int:
    return sum(row[metric] >= FLOORS[row["stage"]][metric]
               for row in stage_rows for metric in ("pr_auc", "risk_recall", "risk_f1"))
