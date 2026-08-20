"""CUDA Hybrid trainer. Multi-prefix, AMP, STOP-only early stop. Never sees outer labels."""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from .data import PreparedDomain, permute_temporal
from .hardware import require_cuda
from .losses import gate_live_penalty, kd_kl, pairwise_rank_loss
from .metrics import binary_metrics, select_stop_threshold
from .model import SuperiorityConfig, SuperiorityHybrid, count_parameters
from .protocol import STAGE_WEIGHTS, seed_everything, stages_for, warm_for


def _ids_for_stage(view, ids: list[str]) -> list[str]:
    present = set(map(str, view.record_id))
    return [record_id for record_id in ids if record_id in present]


class HybridTrainer:
    def __init__(
        self,
        prepared: PreparedDomain,
        config: SuperiorityConfig,
        *,
        lr: float = 2e-4,
        weight_decay: float = 2e-4,
        max_epochs: int = 24,
        patience: int = 8,
        batch_size: int = 128,
        amp: bool = True,
        seed: int = 42,
        pos_weight_multiplier: float = 1.0,
        lambda_rank: float = 0.10,
        lambda_kd: float = 0.0,
        lambda_aux: float = 0.25,
        lambda_gate: float = 0.05,
        kd_temperature: float = 2.0,
        teacher_map: dict[str, dict[str, float]] | None = None,
        use_ema: bool = True,
        ema_decay: float = 0.995,
        multiprefix: bool = True,
        warmup_epochs: int = 2,
    ):
        require_cuda()
        self.prepared = prepared
        self.config = config
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.amp = amp
        self.seed = seed
        self.pos_weight_multiplier = pos_weight_multiplier
        self.lambda_rank = lambda_rank
        self.lambda_kd = lambda_kd
        self.lambda_aux = lambda_aux
        self.lambda_gate = lambda_gate
        self.kd_temperature = kd_temperature
        self.teacher_map = teacher_map or {}
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.multiprefix = multiprefix
        self.warmup_epochs = warmup_epochs
        self.device = torch.device("cuda")
        self.history: list[dict[str, Any]] = []

    def _batch(self, stage: str, ids: list[str], temporal_mode: str = "identity"):
        view = self.prepared.views[stage]
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        idx = np.asarray([lookup[i] for i in ids])
        temporal = view.temporal[idx]
        mask = view.temporal_mask[idx]
        if temporal_mode != "identity":
            temporal = permute_temporal(temporal, mask, temporal_mode, self.seed)
        static = np.stack([self.prepared.static_map[i] for i in ids])
        return {
            "static": torch.tensor(static, dtype=torch.float32, device=self.device),
            "temporal": torch.tensor(temporal, dtype=torch.float32, device=self.device),
            "temporal_mask": torch.tensor(mask, device=self.device),
            "lengths": torch.tensor(view.lengths[idx], device=self.device),
            "aggregate": torch.tensor(view.aggregate[idx], dtype=torch.float32, device=self.device),
            "aggregate_available": torch.tensor(view.aggregate_available[idx], device=self.device),
            "progress": torch.tensor(view.progress[idx], dtype=torch.float32, device=self.device),
        }, view.target[idx].astype(np.float32)

    def _predict(self, model: SuperiorityHybrid, stage: str, ids: list[str], temporal_mode: str = "identity") -> np.ndarray:
        if not ids:
            return np.empty(0, np.float32)
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                inputs, _ = self._batch(stage, chunk, temporal_mode)
                with autocast("cuda", enabled=self.amp):
                    logits = model(**inputs)
                scores.append(torch.sigmoid(logits.float()).cpu().numpy())
        out = np.concatenate(scores).astype(np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("NONFINITE_SCORES")
        return out

    def stage_ap(self, model: SuperiorityHybrid, ids: list[str]) -> dict[str, float]:
        values = {}
        for stage, view in self.prepared.views.items():
            stage_ids = _ids_for_stage(view, ids)
            if not stage_ids:
                continue
            lookup = {str(r): i for i, r in enumerate(view.record_id)}
            y = view.target[[lookup[i] for i in stage_ids]]
            if len(np.unique(y)) < 2:
                continue
            values[stage] = binary_metrics(y, self._predict(model, stage, stage_ids))["ap"]
        return values

    def _stop_score(self, ap: dict[str, float]) -> float:
        warm = [ap[s] for s in warm_for(self.prepared.domain) if s in ap]
        if not warm:
            return float("nan")
        return float(np.mean(warm))

    def _teacher(self, stage: str, ids: list[str], device) -> torch.Tensor | None:
        table = self.teacher_map.get(stage)
        if not table:
            return None
        vals = [table.get(i) for i in ids]
        if any(v is None for v in vals):
            return None
        return torch.tensor(vals, dtype=torch.float32, device=device)

    def fit(self, fit_ids: list[str], stop_ids: list[str]) -> dict[str, Any]:
        seed_everything(self.seed)
        torch.cuda.reset_peak_memory_stats()
        model = SuperiorityHybrid(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        y_fit = []
        rec = self.prepared.context.drop_duplicates("record_id")
        id_to_y = {str(r): int(t) for r, t in zip(rec.record_id.astype(str), rec.target)}
        y_fit = np.asarray([id_to_y[i] for i in fit_ids])
        base = (len(y_fit) - y_fit.sum()) / max(1, y_fit.sum())
        pos_weight = torch.tensor([base * self.pos_weight_multiplier], dtype=torch.float32, device=self.device)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = GradScaler("cuda", enabled=self.amp)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if self.use_ema else None
        indexes = {stage: {str(r): i for i, r in enumerate(view.record_id)} for stage, view in self.prepared.views.items()}
        eligible = {r: [s for s, lookup in indexes.items() if r in lookup] for r in fit_ids}
        best = -np.inf
        best_epoch = 0
        stale = 0
        best_state = None
        started = time.monotonic()
        batch_size = self.batch_size
        n_params = count_parameters(model)
        for epoch in range(self.max_epochs):
            model.train()
            epoch_losses = []
            grad_norms = []
            if self.multiprefix:
                by_stage = {stage: [] for stage in self.prepared.views}
                for record_id, stages in eligible.items():
                    for stage in stages:
                        by_stage[stage].append(record_id)
            else:
                rng = np.random.default_rng(self.seed + epoch)
                by_stage = {stage: [] for stage in self.prepared.views}
                for record_id, stages in eligible.items():
                    by_stage[str(rng.choice(stages))].append(record_id)
            try:
                for stage, ids in by_stage.items():
                    rng = np.random.default_rng(self.seed + epoch + abs(hash(stage)) % 1000)
                    order = np.arange(len(ids))
                    rng.shuffle(order)
                    ids = [ids[i] for i in order]
                    weight = float(STAGE_WEIGHTS.get(stage, 1.0))
                    for start in range(0, len(ids), batch_size):
                        chunk = ids[start : start + batch_size]
                        if len(chunk) < 2:
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        inputs, labels = self._batch(stage, chunk)
                        y = torch.tensor(labels, dtype=torch.float32, device=self.device)
                        with autocast("cuda", enabled=self.amp):
                            logits = model(**inputs)
                            diag = model.last_diagnostics
                            loss = bce(logits, y)
                            if self.lambda_rank > 0:
                                loss = loss + self.lambda_rank * pairwise_rank_loss(logits, y)
                            if self.lambda_aux > 0:
                                aux = bce(diag["z_tab"], y)
                                if diag["temporal_available"].any():
                                    aux = aux + bce(diag["z_cnn"], y) + bce(diag["z_lstm"], y)
                                loss = loss + self.lambda_aux * aux / 3.0
                            if self.lambda_gate > 0:
                                loss = loss + self.lambda_gate * gate_live_penalty(diag["g"], diag["temporal_available"])
                            teacher = self._teacher(stage, chunk, self.device)
                            if teacher is not None and self.lambda_kd > 0:
                                loss = loss + self.lambda_kd * kd_kl(logits, teacher, self.kd_temperature)
                            loss = loss * weight
                        if not torch.isfinite(loss):
                            raise RuntimeError("NONFINITE_LOSS")
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_losses.append(float(loss.detach().float().cpu()))
                        if ema is not None:
                            with torch.no_grad():
                                for k, v in model.state_dict().items():
                                    if v.dtype.is_floating_point:
                                        ema[k].mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)
                                    else:
                                        ema[k].copy_(v)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 16:
                    raise
                batch_size = max(16, batch_size // 2)
                self.batch_size = batch_size
                continue
            eval_model = model
            backup = None
            if ema is not None:
                backup = copy.deepcopy(model.state_dict())
                eval_model.load_state_dict(ema, strict=False)
            stop_ap = self.stage_ap(eval_model, stop_ids) if stop_ids else {}
            stop_score = self._stop_score(stop_ap)
            if backup is not None:
                model.load_state_dict(backup)
            self.history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
                    "stop_ap": stop_ap,
                    "stop_score": stop_score,
                    "grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
                    "batch_size": batch_size,
                }
            )
            if np.isfinite(stop_score) and stop_score > best:
                best = stop_score
                best_epoch = epoch + 1
                stale = 0
                best_state = copy.deepcopy(ema if ema is not None else model.state_dict())
            else:
                stale += 1
                if stale >= self.patience and epoch + 1 >= self.warmup_epochs:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        runtime = time.monotonic() - started
        peak = torch.cuda.max_memory_allocated() / 1024**3
        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_stop_score": best,
            "history": self.history,
            "parameter_count": n_params,
            "peak_vram_gb": peak,
            "runtime_seconds": runtime,
            "batch_size": batch_size,
            "outer_test_used": False,
        }

    def score_split(self, model: SuperiorityHybrid, ids: list[str], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
        out = {"stages": {}, "predictions": []}
        for stage, view in self.prepared.views.items():
            stage_ids = _ids_for_stage(view, ids)
            if not stage_ids:
                continue
            lookup = {str(r): i for i, r in enumerate(view.record_id)}
            y = view.target[[lookup[i] for i in stage_ids]]
            p = self._predict(model, stage, stage_ids)
            t = 0.5 if thresholds is None else thresholds.get(stage, 0.5)
            if len(np.unique(y)) >= 2:
                out["stages"][stage] = binary_metrics(y, p, threshold=t)
            groups = view.group_id[[lookup[i] for i in stage_ids]]
            for record_id, group_id, yi, pi in zip(stage_ids, groups, y, p):
                out["predictions"].append(
                    {
                        "record_id": str(record_id),
                        "group_id": str(group_id),
                        "stage": stage,
                        "y": int(yi),
                        "p": float(pi),
                    }
                )
        return out

    def fit_thresholds(self, model: SuperiorityHybrid, stop_ids: list[str]) -> dict[str, float]:
        thresholds = {}
        for stage, view in self.prepared.views.items():
            stage_ids = _ids_for_stage(view, stop_ids)
            if not stage_ids:
                continue
            lookup = {str(r): i for i, r in enumerate(view.record_id)}
            y = view.target[[lookup[i] for i in stage_ids]]
            if len(np.unique(y)) < 2:
                continue
            p = self._predict(model, stage, stage_ids)
            thresholds[stage] = select_stop_threshold(y, p)
        return thresholds
