"""C4-STAR trainer. CUDA AMP, thermal pause at 80C, STOP-only early stop, never outer labels."""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from experiments.hybrid_superiority_v2.data import permute_temporal
from experiments.hybrid_superiority_v2.hardware import require_cuda
from experiments.hybrid_superiority_v2.metrics import binary_metrics, select_stop_threshold
from experiments.hybrid_superiority_v2.protocol import STAGE_WEIGHTS, seed_everything, warm_for
from experiments.hybrid_superiority_v2.train import _ids_for_stage

from .losses import gate_reg, kd_bce, pairwise_rank_loss, ssl_reconstruct
from .model import C4Config, C4STAR, count_parameters
from .protocol import MAX_EPOCHS, PATIENCE, WARMUP_EPOCHS
from .thermal import wait_if_hot


class C4Trainer:
    def __init__(
        self,
        prepared,
        config: C4Config,
        *,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        batch_size: int = 256,
        amp: bool = True,
        seed: int = 42,
        pos_weight_multiplier: float = 1.0,
        lambda_rank: float = 0.05,
        lambda_kd: float = 0.25,
        lambda_aux: float = 0.5,
        lambda_ssl: float = 0.0,
        lambda_gate: float = 0.02,
        kd_temperature: float = 2.0,
        teacher_map: dict | None = None,
        use_ema: bool = True,
        group_dro: bool = False,
        dro_eta: float = 0.1,
        multiprefix: bool = True,
        warmup_epochs: int = WARMUP_EPOCHS,
        freeze_anchor_epochs: int = 0,
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
        self.lambda_ssl = lambda_ssl
        self.lambda_gate = lambda_gate
        self.kd_temperature = kd_temperature
        self.teacher_map = teacher_map or {}
        self.use_ema = use_ema
        self.group_dro = group_dro
        self.dro_eta = dro_eta
        self.multiprefix = multiprefix
        self.warmup_epochs = warmup_epochs
        self.freeze_anchor_epochs = freeze_anchor_epochs
        self.device = torch.device("cuda")
        self.history: list[dict[str, Any]] = []
        self._lookups = {stage: {str(r): i for i, r in enumerate(view.record_id)} for stage, view in prepared.views.items()}
        recs = list(prepared.static_map.keys())
        self._static_index = {r: i for i, r in enumerate(recs)}
        self._static_mat = np.stack([prepared.static_map[r] for r in recs]).astype(np.float32)
        self._pin()

    def _pin(self) -> None:
        try:
            self._static_gpu = torch.as_tensor(self._static_mat, device=self.device)
            self._gpu = {}
            for stage, view in self.prepared.views.items():
                self._gpu[stage] = {
                    "temporal": torch.as_tensor(view.temporal, dtype=torch.float32, device=self.device),
                    "temporal_mask": torch.as_tensor(view.temporal_mask, device=self.device),
                    "lengths": torch.as_tensor(view.lengths, device=self.device),
                    "aggregate": torch.as_tensor(view.aggregate, dtype=torch.float32, device=self.device),
                    "aggregate_available": torch.as_tensor(view.aggregate_available, device=self.device),
                    "progress": torch.as_tensor(view.progress, dtype=torch.float32, device=self.device),
                    "target": torch.as_tensor(view.target.astype(np.float32), device=self.device),
                }
            self._views_on_gpu = True
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._gpu = {}
            self._static_gpu = None
            self._views_on_gpu = False

    def _batch(self, stage: str, ids: list[str], temporal_mode: str = "identity"):
        lookup = self._lookups[stage]
        np_idx = np.fromiter((lookup[i] for i in ids), dtype=np.int64, count=len(ids))
        view = self.prepared.views[stage]
        if temporal_mode != "identity" or not self._views_on_gpu:
            temporal = view.temporal[np_idx]
            mask = view.temporal_mask[np_idx]
            if temporal_mode != "identity":
                temporal = permute_temporal(temporal, mask, temporal_mode, self.seed)
            static = self._static_mat[[self._static_index[i] for i in ids]]
            return {
                "static": torch.as_tensor(static, dtype=torch.float32, device=self.device),
                "temporal": torch.as_tensor(temporal, dtype=torch.float32, device=self.device),
                "temporal_mask": torch.as_tensor(mask, device=self.device),
                "lengths": torch.as_tensor(view.lengths[np_idx], device=self.device),
                "aggregate": torch.as_tensor(view.aggregate[np_idx], dtype=torch.float32, device=self.device),
                "aggregate_available": torch.as_tensor(view.aggregate_available[np_idx], device=self.device),
                "progress": torch.as_tensor(view.progress[np_idx], dtype=torch.float32, device=self.device),
            }, torch.as_tensor(view.target[np_idx].astype(np.float32), device=self.device)
        idx = torch.from_numpy(np_idx).to(self.device)
        st_idx = torch.tensor([self._static_index[i] for i in ids], device=self.device, dtype=torch.long)
        g = self._gpu[stage]
        return {
            "static": self._static_gpu.index_select(0, st_idx),
            "temporal": g["temporal"].index_select(0, idx),
            "temporal_mask": g["temporal_mask"].index_select(0, idx),
            "lengths": g["lengths"].index_select(0, idx),
            "aggregate": g["aggregate"].index_select(0, idx),
            "aggregate_available": g["aggregate_available"].index_select(0, idx),
            "progress": g["progress"].index_select(0, idx),
        }, g["target"].index_select(0, idx)

    def _teacher(self, stage: str, ids: list[str]):
        table = self.teacher_map.get(stage)
        if not table:
            return None
        vals = [table.get(i) for i in ids]
        if any(v is None for v in vals):
            return None
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    def _predict(self, model: C4STAR, stage: str, ids: list[str], temporal_mode: str = "identity") -> np.ndarray:
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
        return np.concatenate(scores).astype(np.float32)

    def stage_ap(self, model: C4STAR, ids: list[str]) -> dict[str, float]:
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
        return float(np.mean(warm)) if warm else float("nan")

    def fit(self, fit_ids: list[str], stop_ids: list[str]) -> dict[str, Any]:
        seed_everything(self.seed)
        torch.cuda.reset_peak_memory_stats()
        model = C4STAR(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rec = self.prepared.context.drop_duplicates("record_id")
        id_to_y = {str(r): int(t) for r, t in zip(rec.record_id.astype(str), rec.target)}
        y_fit = np.asarray([id_to_y[i] for i in fit_ids])
        base = (len(y_fit) - y_fit.sum()) / max(1, y_fit.sum())
        pos_weight = torch.tensor([base * self.pos_weight_multiplier], dtype=torch.float32, device=self.device)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = GradScaler("cuda", enabled=self.amp)
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if self.use_ema else None
        indexes = self._lookups
        eligible = {r: [s for s, lookup in indexes.items() if r in lookup] for r in fit_ids}
        stages = list(self.prepared.views)
        dro_w = {s: 1.0 / len(stages) for s in stages}
        best = -np.inf
        best_epoch = 0
        stale = 0
        best_state = None
        started = time.monotonic()
        n_params = count_parameters(model)
        batch_size = self.batch_size
        for epoch in range(self.max_epochs):
            wait_if_hot()
            model.train()
            if self.freeze_anchor_epochs and epoch < self.freeze_anchor_epochs:
                for p in model.anchor.parameters():
                    p.requires_grad = False
            else:
                for p in model.anchor.parameters():
                    p.requires_grad = True
            if self.multiprefix:
                by_stage = {stage: [] for stage in stages}
                for record_id, st in eligible.items():
                    for stage in st:
                        by_stage[stage].append(record_id)
            else:
                rng = np.random.default_rng(self.seed + epoch)
                by_stage = {stage: [] for stage in stages}
                for record_id, st in eligible.items():
                    by_stage[str(rng.choice(st))].append(record_id)
            epoch_losses = []
            stage_losses = {s: [] for s in stages}
            try:
                for stage, ids in by_stage.items():
                    rng = np.random.default_rng(self.seed + epoch + abs(hash(stage)) % 1000)
                    order = np.arange(len(ids))
                    rng.shuffle(order)
                    ids = [ids[i] for i in order]
                    weight = float(STAGE_WEIGHTS.get(stage, 1.0))
                    if self.group_dro:
                        weight = weight * dro_w[stage] * len(stages)
                    for start in range(0, len(ids), batch_size):
                        chunk = ids[start : start + batch_size]
                        if len(chunk) < 2:
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        inputs, y = self._batch(stage, chunk)
                        with autocast("cuda", enabled=self.amp):
                            logits = model(**inputs)
                            diag = model.last_diagnostics
                        logits = logits.float()
                        y = y.float()
                        z_anchor = diag["z_anchor"].float()
                        alpha = diag["alpha"].float()
                        loss = bce(logits, y)
                        if self.lambda_aux > 0:
                            loss = loss + self.lambda_aux * bce(z_anchor, y)
                        if self.lambda_rank > 0:
                            loss = loss + self.lambda_rank * pairwise_rank_loss(logits, y)
                        if self.lambda_gate > 0:
                            loss = loss + self.lambda_gate * gate_reg(alpha)
                        teacher = self._teacher(stage, chunk)
                        if teacher is not None and self.lambda_kd > 0:
                            loss = loss + self.lambda_kd * kd_bce(z_anchor, teacher, self.kd_temperature)
                        if self.lambda_ssl > 0:
                            lengths = inputs["lengths"]
                            ssl_mask = inputs["temporal_mask"].bool() & lengths.ge(3).unsqueeze(1)
                            if ssl_mask.any():
                                recon = diag["recon"].float()
                                target = inputs["temporal"].float()
                                loss = loss + self.lambda_ssl * ssl_reconstruct(recon, target, ssl_mask)
                        loss = loss * weight
                        if not torch.isfinite(loss):
                            raise RuntimeError("NONFINITE_LOSS")
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        val = float(loss.detach().float().cpu())
                        epoch_losses.append(val)
                        stage_losses[stage].append(val)
                        if ema is not None:
                            with torch.no_grad():
                                for k, v in model.state_dict().items():
                                    if v.dtype.is_floating_point:
                                        ema[k].mul_(0.995).add_(v.detach(), alpha=0.005)
                                    else:
                                        ema[k].copy_(v)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 16:
                    raise
                batch_size = max(16, batch_size // 2)
                self.batch_size = batch_size
                continue
            if self.group_dro:
                means = {s: float(np.mean(v)) if v else 0.0 for s, v in stage_losses.items()}
                mx = max(means.values()) if means else 1.0
                for s, m in means.items():
                    dro_w[s] = dro_w[s] * np.exp(self.dro_eta * (m / max(mx, 1e-6)))
                tot = sum(dro_w.values())
                for s in dro_w:
                    dro_w[s] /= tot
                # cap cold
                cold = "S0" if "S0" in dro_w else "20pct"
                if cold in dro_w:
                    dro_w[cold] = min(dro_w[cold], 0.25)
                    tot = sum(dro_w.values())
                    for s in dro_w:
                        dro_w[s] /= tot
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
                    "batch_size": batch_size,
                    "temp": (wait_if_hot() or {}).get("temp_c"),
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
        return {
            "model": model,
            "best_epoch": best_epoch,
            "best_stop_score": best,
            "history": self.history,
            "parameter_count": n_params,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "runtime_seconds": time.monotonic() - started,
            "batch_size": batch_size,
            "outer_test_used": False,
        }

    def score_split(self, model: C4STAR, ids: list[str], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
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
                    {"record_id": str(record_id), "group_id": str(group_id), "stage": stage, "y": int(yi), "p": float(pi)}
                )
        return out

    def fit_thresholds(self, model: C4STAR, stop_ids: list[str]) -> dict[str, float]:
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
