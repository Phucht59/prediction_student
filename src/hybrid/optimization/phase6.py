"""Phase 6 development-only trainer for the stage-conditioned Hybrid."""
from __future__ import annotations

import copy
import gc
import hashlib
import math
import time
from dataclasses import asdict

import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.hybrid.models import StageConditionedConfig, StageConditionedHybrid
from src.hybrid.training.data import sample_prefixes, sample_prefixes_stage_balanced
from src.hybrid.training.trainer import assemble, eligible_prefixes, seed_everything, _scheduler

STAGES = {"uci": ("S0", "S1", "S2"), "oulad": ("20pct", "35pct", "50pct", "75pct")}


def class_pos_weight(target: np.ndarray, mode: str) -> float:
    positive = max(1, int(np.asarray(target).sum()))
    ratio = (len(target) - positive) / positive
    if mode == "none":
        return 1.0
    if mode == "sqrt":
        return float(math.sqrt(ratio))
    if mode == "full":
        return float(ratio)
    raise ValueError(mode)


def deterministic_pairwise_loss(logits: torch.Tensor, target: torch.Tensor, seed: int, maximum: int = 256) -> torch.Tensor:
    """Bounded deterministic positive/negative ranking loss for one minibatch."""
    positives = torch.nonzero(target > 0.5, as_tuple=False).flatten()
    negatives = torch.nonzero(target <= 0.5, as_tuple=False).flatten()
    if not len(positives) or not len(negatives):
        return logits.sum() * 0.0
    total = min(maximum, len(positives) * len(negatives))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pi = positives[torch.randint(len(positives), (total,), generator=generator, device="cpu").to(positives.device)]
    ni = negatives[torch.randint(len(negatives), (total,), generator=generator, device="cpu").to(negatives.device)]
    return torch.nn.functional.softplus(-(logits[pi] - logits[ni])).mean()


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float | None):
        self.decay = decay
        self.shadow = None if decay is None else {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        if self.shadow is None:
            return
        for key, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[key].copy_(value)

    def state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        source = model.state_dict() if self.shadow is None else self.shadow
        return {k: v.detach().cpu().clone() for k, v in source.items()}


def model_config(domain: str, temporal_dim: int, context_dim: int, params: dict) -> StageConditionedConfig:
    return StageConditionedConfig(
        domain=domain,
        temporal_dim=temporal_dim,
        context_dim=context_dim,
        d_model=int(params["d_model"]),
        cnn_channels=int(params["cnn_channels"]),
        cnn_blocks=int(params["cnn_blocks"]),
        bilstm_hidden=int(params["bilstm_hidden"]),
        bilstm_layers=int(params.get("bilstm_layers", 1)),
        context_hidden=int(params["context_hidden"]),
        shared_head_hidden=int(params["shared_head_hidden"]),
        dropout=float(params["dropout"]),
        summary_residual=bool(params.get("summary_residual", False)),
        uci_wide_context=bool(params.get("uci_wide_context", True)),
    )


def parameter_count(domain: str, temporal_dim: int, context_dim: int, params: dict) -> int:
    return sum(p.numel() for p in StageConditionedHybrid(model_config(domain, temporal_dim, context_dim, params)).parameters())


def _arrays(stages, ids, selected, contexts, domain):
    arrays = assemble(stages, ids, selected, contexts)
    stage_map = {stage: index for index, stage in enumerate(STAGES[domain])}
    return arrays + (np.asarray([stage_map[stage] for stage in selected], np.int64),)


def _loader(arrays, batch_size: int, seed: int, shuffle: bool):
    tensors = tuple(torch.from_numpy(value) for value in arrays)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), pin_memory=True)


def _choices(ids, available, seed, epoch, mode):
    if mode == "uci_round_robin":
        selected=[]
        for record_id, choices in zip(ids,available,strict=True):
            if len(choices)!=3 or set(choices)!={"S0","S1","S2"}:
                raise ValueError("uci_round_robin requires all three UCI stages")
            offset=int.from_bytes(hashlib.sha256(f"{record_id}:{seed}".encode()).digest()[:8],"big")%3
            selected.append(choices[(offset+epoch)%3])
        return selected
    return sample_prefixes_stage_balanced(ids, available, seed, epoch) if mode == "stage_balanced" else sample_prefixes(ids, available, seed, epoch)


@torch.no_grad()
def predict(model, domain, stages, stage, ids, contexts, batch_size):
    if not ids:
        return np.asarray([], dtype=np.float32)
    model.eval()
    arrays = _arrays(stages, ids, [stage] * len(ids), contexts, domain)
    scores = []
    for temporal, mask, lengths, context, _, stage_index in _loader(arrays, batch_size, 42, False):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(temporal.cuda(non_blocking=True), mask.cuda(non_blocking=True),
                           lengths.cuda(non_blocking=True), context.cuda(non_blocking=True),
                           stage_index.cuda(non_blocking=True))
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores)


def _metrics(model, domain, stages, ids_by_stage, contexts, batch_size):
    rows = {}
    for stage, ids in ids_by_stage.items():
        if not ids:
            continue
        data = stages[stage]
        target = np.asarray([data.target[data.index[record_id]] for record_id in ids])
        if len(np.unique(target)) != 2:
            raise RuntimeError(f"single_class_evaluation:{domain}:{stage}")
        score = predict(model, domain, stages, stage, ids, contexts, batch_size)
        rows[stage] = {
            "pr_auc": float(average_precision_score(target, score)),
            "roc_auc": float(roc_auc_score(target, score)),
            "recall": float(recall_score(target, score >= 0.5, zero_division=0)),
        }
    return rows


def fit_development_fold(domain, stages, fit_ids, stop_ids, valid_by_stage, contexts,
                         temporal_dim, context_dim, params, seed=42, trial=None, step_offset=0,
                         max_epochs=80, patience=10, return_predictions=False, return_stop_predictions=False):
    """Fit/stop/evaluate with all epoch selection confined to train-only data."""
    valid_by_stage = valid_by_stage or {}
    if set(fit_ids) & set(stop_ids):
        raise RuntimeError("fit/early-stop record leakage")
    if (set(fit_ids) | set(stop_ids)) & set().union(*map(set, valid_by_stage.values())):
        raise RuntimeError("training/evaluation record leakage")
    seed_everything(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    config = model_config(domain, temporal_dim, context_dim, params)
    model = StageConditionedHybrid(config).cuda()
    nparams = sum(p.numel() for p in model.parameters())
    cap = 450_000 if domain == "uci" else 2_000_000
    if nparams > cap:
        raise optuna.TrialPruned(f"parameter_cap:{nparams}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = _scheduler(optimizer, max_epochs, params.get("warmup_epochs", 0), 1e-5, params["learning_rate"])
    target = np.asarray([next(data.target[data.index[r]] for data in stages.values() if r in data.index) for r in fit_ids])
    weight = class_pos_weight(target, params["class_weight_mode"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device="cuda"))
    scaler = torch.amp.GradScaler("cuda")
    ema = ExponentialMovingAverage(model, params.get("ema_decay"))
    available = eligible_prefixes(stages, fit_ids)
    best, best_epoch, best_state, stale = -1.0, 1, None, 0
    mode = params.get("prefix_sampling", "record_uniform")
    for epoch in range(max_epochs):
        chosen = _choices(fit_ids, available, seed, epoch, mode)
        arrays = _arrays(stages, fit_ids, chosen, contexts, domain)
        model.train()
        for batch_index, (temporal, mask, lengths, context, batch_target, stage_index) in enumerate(_loader(arrays, params["batch_size"], seed + epoch, True)):
            temporal = temporal.cuda(non_blocking=True); mask = mask.cuda(non_blocking=True)
            lengths = lengths.cuda(non_blocking=True); context = context.cuda(non_blocking=True)
            batch_target = batch_target.cuda(non_blocking=True); stage_index = stage_index.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(temporal, mask, lengths, context, stage_index)
                bce = loss_fn(logits, batch_target)
                rank = deterministic_pairwise_loss(logits.float(), batch_target, seed + epoch * 10000 + batch_index)
                loss = bce + float(params["lambda_rank"]) * rank
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise optuna.TrialPruned("non_finite_loss_or_logits")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip_norm"])
            if not torch.isfinite(norm):
                raise optuna.TrialPruned("non_finite_gradients")
            scaler.step(optimizer); scaler.update(); ema.update(model)
        scheduler.step()
        raw_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema.state_dict(model))
        stop_by = {stage: [r for r in stop_ids if r in data.index] for stage, data in stages.items()}
        stop_metrics = _metrics(model, domain, stages, stop_by, contexts, params["batch_size"])
        macro = float(np.mean([row["pr_auc"] for row in stop_metrics.values()]))
        model.load_state_dict(raw_state)
        if trial is not None:
            trial.report(macro, step_offset + epoch)
            if epoch >= 10 and trial.should_prune():
                raise optuna.TrialPruned("median_pruner")
        if macro > best:
            best, best_epoch, best_state, stale = macro, epoch + 1, ema.state_dict(model), 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("no_finite_early_stop_epoch")
    model.load_state_dict(best_state)
    train_by = {stage: [r for r in fit_ids if r in data.index] for stage, data in stages.items()}
    train_metrics = _metrics(model, domain, stages, train_by, contexts, params["batch_size"])
    stop_by = {stage: [r for r in stop_ids if r in data.index] for stage, data in stages.items()}
    stop_metrics = _metrics(model, domain, stages, stop_by, contexts, params["batch_size"])
    valid_metrics = _metrics(model, domain, stages, valid_by_stage, contexts, params["batch_size"])
    if return_stop_predictions:
        predictions={"validation":{stage:predict(model,domain,stages,stage,ids,contexts,params["batch_size"]) for stage,ids in valid_by_stage.items()},
                     "stop":{stage:predict(model,domain,stages,stage,ids,contexts,params["batch_size"]) for stage,ids in stop_by.items() if ids}}
    else:
        predictions={stage:predict(model,domain,stages,stage,ids,contexts,params["batch_size"]) for stage,ids in valid_by_stage.items()} if return_predictions else None
    train_macro = float(np.mean([row["pr_auc"] for row in train_metrics.values()]))
    valid_macro = float(np.mean([row["pr_auc"] for row in valid_metrics.values()])) if valid_metrics else None
    result = {
        "domain": domain,
        "seed": seed,
        "macro_pr_auc": valid_macro,
        "train_macro_pr_auc": train_macro,
        "early_stop_macro_pr_auc": float(np.mean([row["pr_auc"] for row in stop_metrics.values()])),
        "train_validation_gap": None if valid_macro is None else train_macro - valid_macro,
        "per_stage": valid_metrics,
        "best_epoch": best_epoch,
        "parameter_count": nparams,
        "pos_weight": weight,
        "ema_inference": params.get("ema_decay") is not None,
        "runtime_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "model_config": asdict(config),
    }
    del model, optimizer, scheduler, scaler, loss_fn, ema
    gc.collect(); torch.cuda.empty_cache()
    return result, predictions


def refit_predict(domain, stages, train_ids, valid_by_stage, contexts, temporal_dim,
                  context_dim, params, seed, epochs):
    """Fresh selected-epoch full-training refit; does not inspect evaluation labels."""
    seed_everything(seed); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    model = StageConditionedHybrid(model_config(domain, temporal_dim, context_dim, params)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"])
    scheduler = _scheduler(optimizer, max(epochs, 1), params.get("warmup_epochs", 0), 1e-5, params["learning_rate"])
    target = np.asarray([next(data.target[data.index[r]] for data in stages.values() if r in data.index) for r in train_ids])
    weight = class_pos_weight(target, params["class_weight_mode"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device="cuda"))
    scaler = torch.amp.GradScaler("cuda"); ema = ExponentialMovingAverage(model, params.get("ema_decay"))
    available = eligible_prefixes(stages, train_ids); mode = params.get("prefix_sampling", "record_uniform")
    for epoch in range(epochs):
        chosen = _choices(train_ids, available, seed, epoch, mode)
        arrays = _arrays(stages, train_ids, chosen, contexts, domain); model.train()
        for batch_index, (temporal, mask, lengths, context, batch_target, stage_index) in enumerate(_loader(arrays, params["batch_size"], seed + epoch, True)):
            temporal=temporal.cuda(non_blocking=True);mask=mask.cuda(non_blocking=True);lengths=lengths.cuda(non_blocking=True)
            context=context.cuda(non_blocking=True);batch_target=batch_target.cuda(non_blocking=True);stage_index=stage_index.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits=model(temporal,mask,lengths,context,stage_index)
                loss=loss_fn(logits,batch_target)+float(params["lambda_rank"])*deterministic_pairwise_loss(logits.float(),batch_target,seed+epoch*10000+batch_index)
            if not torch.isfinite(loss): raise RuntimeError("non_finite_refit_loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),params["gradient_clip_norm"])
            if not torch.isfinite(norm): raise RuntimeError("non_finite_refit_gradient")
            scaler.step(optimizer);scaler.update();ema.update(model)
        scheduler.step()
    model.load_state_dict(ema.state_dict(model))
    predictions={stage:predict(model,domain,stages,stage,ids,contexts,params["batch_size"]) for stage,ids in valid_by_stage.items() if ids}
    audit={"runtime_seconds":time.monotonic()-started,"peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),
           "parameter_count":sum(p.numel() for p in model.parameters()),"pos_weight":weight,"epochs":epochs}
    del model,optimizer,scheduler,scaler,loss_fn,ema;gc.collect();torch.cuda.empty_cache()
    return predictions,audit
