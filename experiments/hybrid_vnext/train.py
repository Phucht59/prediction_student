"""CUDA-only Hybrid trainer. Restores best STOP checkpoint. Never sees outer labels."""
from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from .data import PreparedDomain, permute_temporal
from .metrics import binary_metrics, select_stop_threshold
from .model import VNextConfig, VNextHybrid
from .protocol import seed_everything


def _ids_for_stage(view, ids: list[str]) -> list[str]:
    present = set(map(str, view.record_id))
    return [record_id for record_id in ids if record_id in present]


class Trainer:
    def __init__(
        self,
        prepared: PreparedDomain,
        config: VNextConfig,
        *,
        lr: float = 2e-4,
        weight_decay: float = 2e-4,
        max_epochs: int = 24,
        patience: int = 8,
        batch_size: int = 256,
        amp: bool = True,
        seed: int = 42,
        pos_weight_multiplier: float = 1.0,
        epoch_callback=None,
        fixed_epochs: int | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED_FOR_HYBRID_PHASE2")
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
        self.epoch_callback = epoch_callback
        self.fixed_epochs = fixed_epochs
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
        return {
            "static": torch.tensor(np.asarray([self.prepared.static_map[i] for i in ids]), dtype=torch.float32, device=self.device),
            "temporal": torch.tensor(temporal, dtype=torch.float32, device=self.device),
            "temporal_mask": torch.tensor(mask, device=self.device),
            "lengths": torch.tensor(view.lengths[idx], device=self.device),
            "aggregate": torch.tensor(view.aggregate[idx], dtype=torch.float32, device=self.device),
            "aggregate_available": torch.tensor(view.aggregate_available[idx], device=self.device),
            "progress": torch.tensor(view.progress[idx], dtype=torch.float32, device=self.device),
            "summaries": torch.tensor(np.asarray([self.prepared.summary_map[stage][i] for i in ids]), dtype=torch.float32, device=self.device),
        }, view.target[idx].astype(np.float32)

    def _predict(self, model: VNextHybrid, stage: str, ids: list[str], temporal_mode: str = "identity") -> np.ndarray:
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

    def _predict_logits(self, model: VNextHybrid, stage: str, ids: list[str], temporal_mode: str = "identity") -> np.ndarray:
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
                scores.append(logits.float().cpu().numpy())
        out = np.concatenate(scores).astype(np.float32)
        if not np.isfinite(out).all():
            raise RuntimeError("NONFINITE_LOGITS")
        return out

    def _macro_pr(self, model: VNextHybrid, ids: list[str]) -> float:
        values = []
        for stage, view in self.prepared.views.items():
            stage_ids = _ids_for_stage(view, ids)
            if not stage_ids:
                continue
            lookup = {str(r): i for i, r in enumerate(view.record_id)}
            y = view.target[[lookup[i] for i in stage_ids]]
            if len(np.unique(y)) < 2:
                continue
            values.append(binary_metrics(y, self._predict(model, stage, stage_ids))["pr_auc"])
        if not values:
            raise RuntimeError("EMPTY_MACRO_PR")
        return float(np.mean(values))

    def fit(self, fit_ids: list[str], stop_ids: list[str]) -> dict[str, Any]:
        from src.hybrid.training.data import sample_prefixes_stage_balanced

        seed_everything(self.seed)
        torch.cuda.reset_peak_memory_stats()
        model = VNextHybrid(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rec = self.prepared.context.record_id.astype(str).to_numpy()
        tgt = self.prepared.context["target"].to_numpy()
        id_to_y = {str(record_id): int(label) for record_id, label in zip(rec, tgt)}
        y = np.asarray([id_to_y[record_id] for record_id in fit_ids])
        base = (len(y) - y.sum()) / max(1, y.sum())
        pos_weight = torch.tensor([base * self.pos_weight_multiplier], dtype=torch.float32, device=self.device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = GradScaler("cuda", enabled=self.amp)
        indexes = {stage: {str(r): i for i, r in enumerate(view.record_id)} for stage, view in self.prepared.views.items()}
        eligible = {r: [s for s, lookup in indexes.items() if r in lookup] for r in fit_ids}
        best = -np.inf
        best_epoch = 0
        stale = 0
        best_state = None
        started = time.monotonic()
        batch_size = self.batch_size
        for epoch in range(self.max_epochs):
            model.train()
            choices = sample_prefixes_stage_balanced(fit_ids, [eligible[r] for r in fit_ids], self.seed, epoch)
            by_stage: dict[str, list[str]] = {stage: [] for stage in self.prepared.views}
            for record_id, stage in zip(fit_ids, choices, strict=True):
                by_stage[stage].append(record_id)
            epoch_losses = []
            grad_norms = []
            try:
                for stage, ids in by_stage.items():
                    for start in range(0, len(ids), batch_size):
                        chunk = ids[start : start + batch_size]
                        if not chunk:
                            continue
                        optimizer.zero_grad(set_to_none=True)
                        inputs, labels = self._batch(stage, chunk)
                        with autocast("cuda", enabled=self.amp):
                            logits = model(**inputs)
                            loss = loss_fn(logits, torch.tensor(labels, dtype=torch.float32, device=self.device)) + model.fusion_regularization()
                        if not torch.isfinite(loss):
                            raise RuntimeError("NONFINITE_LOSS")
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)))
                        scaler.step(optimizer)
                        scaler.update()
                        epoch_losses.append(float(loss.detach().float().cpu()))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size <= 32:
                    raise
                batch_size = max(32, batch_size // 2)
                self.batch_size = batch_size
                continue
            # Checkpoint selection uses STOP only. Per-epoch train PR-AUC is diagnostic
            # and does not change weights; skip it to keep model semantics identical.
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
        }

    def evaluate(self, model: VNextHybrid, stop_ids: list[str], valid_ids: list[str], *, temporal_mode: str = "identity") -> dict[str, Any]:
        metrics = {}
        diagnostics = {}
        for stage, view in self.prepared.views.items():
            stop = _ids_for_stage(view, stop_ids)
            valid = _ids_for_stage(view, valid_ids)
            lookup = {str(r): i for i, r in enumerate(view.record_id)}
            stop_y = view.target[[lookup[i] for i in stop]]
            valid_y = view.target[[lookup[i] for i in valid]]
            stop_p = self._predict(model, stage, stop, temporal_mode)
            valid_p = self._predict(model, stage, valid, temporal_mode)
            if len(np.unique(stop_y)) < 2 or len(np.unique(valid_y)) < 2:
                raise ValueError("single-class partition")
            threshold = select_stop_threshold(stop_y, stop_p)
            row = binary_metrics(valid_y, valid_p, threshold=threshold)
            row["stop_pr_auc"] = binary_metrics(stop_y, stop_p)["pr_auc"]
            row["valid_pr_auc"] = row["pr_auc"]
            metrics[stage] = row
            model.eval()
            with torch.no_grad():
                if valid:
                    inputs, _ = self._batch(stage, valid[: min(len(valid), 1024)], temporal_mode)
                    model(**inputs)
                    diag = model.last_diagnostics
                    diagnostics[stage] = {
                        "g_temporal_mean": float(diag["g_temporal"].mean().cpu()),
                        "g_temporal_std": float(diag["g_temporal"].std().cpu()) if diag["g_temporal"].numel() > 1 else 0.0,
                        "tabular_norm": float(diag["h_tabular"].norm(dim=1).mean().cpu()),
                        "temporal_norm": float(diag["h_temporal"].norm(dim=1).mean().cpu()),
                        "temporal_available_rate": float(diag["temporal_available"].float().mean().cpu()),
                    }
                    if diag["gate_weights"] is not None:
                        w = diag["gate_weights"]
                        diagnostics[stage].update(
                            {
                                "tabular_mass": float(w[:, 0].mean().cpu()),
                                "cnn_mass": float(w[:, 1].mean().cpu()),
                                "bilstm_mass": float(w[:, 2].mean().cpu()),
                            }
                        )
        macro = float(np.mean([m["pr_auc"] for m in metrics.values()]))
        worst = float(np.min([m["pr_auc"] for m in metrics.values()]))
        return {"stages": metrics, "macro_pr_auc": macro, "worst_pr_auc": worst, "diagnostics": diagnostics, "temporal_mode": temporal_mode}
