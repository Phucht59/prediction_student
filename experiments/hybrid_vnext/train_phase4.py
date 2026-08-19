"""Phase 4 C0 trainer: same topology, allowed training mechanisms only."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn import functional as F

from .data import PreparedDomain
from .metrics import binary_metrics, select_stop_threshold
from .model import VNextConfig, VNextHybrid
from .protocol import seed_everything
from .train import Trainer, _ids_for_stage


STAGE_ORDER = {
    "uci": ["S0", "S1", "S2"],
    "oulad": ["20pct", "35pct", "50pct", "75pct", "100pct"],
}


@dataclass
class TrainingStrategy:
    name: str
    stage_norm: bool = False
    curriculum: str = "C3"
    hard_stage_weights: bool = False
    weight_lo: float = 0.75
    weight_hi: float = 1.50
    ema: float = 0.7
    trunc_p: float = 0.0
    lambda_rank: float = 0.0
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage_norm": self.stage_norm,
            "curriculum": self.curriculum,
            "hard_stage_weights": self.hard_stage_weights,
            "weight_lo": self.weight_lo,
            "weight_hi": self.weight_hi,
            "ema": self.ema,
            "trunc_p": self.trunc_p,
            "lambda_rank": self.lambda_rank,
            "notes": self.notes,
        }


def allowed_stages(domain: str, epoch: int, max_epochs: int, curriculum: str) -> set[str]:
    order = [s for s in STAGE_ORDER[domain] if True]
    if curriculum in {"C3", "mixed", ""}:
        return set(order)
    frac = epoch / max(1, max_epochs)
    if curriculum == "C1":
        if frac < 1 / 3:
            return set(order[: max(1, len(order) // 2)])
        if frac < 2 / 3:
            return set(order[: max(2, (2 * len(order)) // 3)])
        return set(order)
    if curriculum == "C2":
        if frac < 1 / 3:
            return set(order[-max(1, len(order) // 2) :])
        if frac < 2 / 3:
            return set(order[-max(2, (2 * len(order)) // 3) :])
        return set(order)
    return set(order)


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, max_pairs: int = 32) -> torch.Tensor:
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    n = int(min(pos.numel(), neg.numel(), max_pairs))
    g = torch.Generator(device=logits.device)
    g.manual_seed(int(logits.detach().sum().item() * 1000) % (2**31 - 1) + n)
    pi = torch.randint(0, pos.numel(), (n,), device=logits.device)
    ni = torch.randint(0, neg.numel(), (n,), device=logits.device)
    return F.softplus(-(pos[pi] - neg[ni])).mean()


class StrategyTrainer(Trainer):
    def __init__(self, prepared: PreparedDomain, config: VNextConfig, strategy: TrainingStrategy, **kwargs):
        super().__init__(prepared, config, **kwargs)
        self.strategy = strategy
        self.domain = prepared.domain

    def fit(self, fit_ids: list[str], stop_ids: list[str]) -> dict[str, Any]:
        from src.hybrid.training.data import sample_prefixes_stage_balanced

        seed_everything(self.seed)
        torch.cuda.reset_peak_memory_stats()
        assert torch.cuda.is_available()
        model = VNextHybrid(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rec = self.prepared.context.record_id.astype(str).to_numpy()
        tgt = self.prepared.context["target"].to_numpy()
        id_to_y = {str(record_id): int(label) for record_id, label in zip(rec, tgt)}
        y = np.asarray([id_to_y[record_id] for record_id in fit_ids])
        base = (len(y) - y.sum()) / max(1, y.sum())
        pos_weight = torch.tensor([base * self.pos_weight_multiplier], dtype=torch.float32, device=self.device)
        bce_none = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        scaler = GradScaler("cuda", enabled=self.amp)
        indexes = {stage: {str(r): i for i, r in enumerate(view.record_id)} for stage, view in self.prepared.views.items()}
        eligible_all = {r: [s for s, lookup in indexes.items() if r in lookup] for r in fit_ids}
        order = [s for s in STAGE_ORDER[self.domain] if s in self.prepared.views]
        weights = {s: 1.0 for s in order}
        ema_loss = {s: None for s in order}
        best = -np.inf
        best_epoch = 0
        stale = 0
        best_state = None
        started = time.monotonic()
        batch_size = self.batch_size
        rng = np.random.default_rng(self.seed)
        for epoch in range(self.max_epochs):
            model.train()
            allow = allowed_stages(self.domain, epoch, self.max_epochs, self.strategy.curriculum)
            eligible = []
            for r in fit_ids:
                kept = [s for s in eligible_all[r] if s in allow]
                eligible.append(kept or eligible_all[r])
            choices = sample_prefixes_stage_balanced(fit_ids, eligible, self.seed, epoch)
            if self.strategy.trunc_p > 0:
                new_choices = []
                for record_id, stage, opts in zip(fit_ids, choices, eligible, strict=True):
                    if rng.random() < self.strategy.trunc_p and stage in order:
                        idx = order.index(stage)
                        shorter = [s for s in order[: idx + 1] if s in opts]
                        stage = shorter[int(rng.integers(0, len(shorter)))] if shorter else stage
                    new_choices.append(stage)
                choices = new_choices
            by_stage: dict[str, list[str]] = {stage: [] for stage in self.prepared.views}
            for record_id, stage in zip(fit_ids, choices, strict=True):
                by_stage[stage].append(record_id)
            epoch_losses = []
            grad_norms = []
            stage_loss_acc: dict[str, list[float]] = {s: [] for s in by_stage}
            try:
                present = [s for s, ids in by_stage.items() if ids]
                n_present = max(1, len(present))
                for stage, ids in by_stage.items():
                    if not ids:
                        continue
                    for start in range(0, len(ids), batch_size):
                        chunk = ids[start : start + batch_size]
                        optimizer.zero_grad(set_to_none=True)
                        inputs, labels = self._batch(stage, chunk)
                        yb = torch.tensor(labels, dtype=torch.float32, device=self.device)
                        with autocast("cuda", enabled=self.amp):
                            logits = model(**inputs)
                            sample_loss = bce_none(logits, yb)
                            bce = sample_loss.mean()
                            scale = weights.get(stage, 1.0)
                            if self.strategy.stage_norm:
                                scale = scale * (len(self.prepared.views) / n_present)
                            loss = scale * bce + model.fusion_regularization()
                            if self.strategy.lambda_rank > 0:
                                loss = loss + self.strategy.lambda_rank * pairwise_rank_loss(logits.float(), yb)
                        if not torch.isfinite(loss):
                            raise RuntimeError("NONFINITE_LOSS")
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                        scaler.step(optimizer)
                        scaler.update()
                        val = float(bce.detach().float().cpu())
                        epoch_losses.append(val)
                        stage_loss_acc[stage].append(val)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 32:
                    raise
                batch_size = max(32, batch_size // 2)
                self.batch_size = batch_size
                continue
            if self.strategy.hard_stage_weights:
                observed = [np.mean(v) for v in stage_loss_acc.values() if v]
                mean_obs = float(np.mean(observed)) if observed else 1.0
                for stage, vals in stage_loss_acc.items():
                    if not vals:
                        continue
                    cur = float(np.mean(vals))
                    prev = ema_loss[stage]
                    ema_loss[stage] = cur if prev is None else self.strategy.ema * prev + (1 - self.strategy.ema) * cur
                    raw = (ema_loss[stage] or cur) / max(mean_obs, 1e-6)
                    weights[stage] = float(np.clip(raw, self.strategy.weight_lo, self.strategy.weight_hi))
            train_pr = float("nan")
            stop_pr = float("nan") if self.fixed_epochs is not None or not stop_ids else self._macro_pr(model, stop_ids)
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    "train_pr_auc": train_pr,
                    "stop_pr_auc": stop_pr,
                    "grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
                    "batch_size": batch_size,
                    "lr": self.lr,
                    "stage_weights": dict(weights),
                }
            )
            if self.epoch_callback is not None and np.isfinite(stop_pr):
                self.epoch_callback(epoch + 1, stop_pr)
            if self.fixed_epochs is None and np.isfinite(stop_pr) and stop_pr > best:
                best = stop_pr
                best_epoch = epoch + 1
                stale = 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
            if self.fixed_epochs is None and stale >= self.patience:
                break
        if self.fixed_epochs is not None:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = self.fixed_epochs
            best = stop_pr
        if best_state is None:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = self.max_epochs
        model.load_state_dict(best_state, strict=True)
        self.model = model
        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_stop_macro_pr_auc": float(best),
            "history": self.history,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "runtime_seconds": time.monotonic() - started,
            "batch_size": batch_size,
            "device": "cuda:0",
            "amp": self.amp,
            "gpu_name": torch.cuda.get_device_name(0),
            "stage_weights_final": weights,
            "strategy": self.strategy.as_dict(),
        }
