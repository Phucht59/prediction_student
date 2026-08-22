"""Single-stage CUDA trainer. FIT-only pos_weight. STOP-only early-stop and threshold."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from src.prediction.model.hybrid import HybridConfig

from .data import PreparedDomain, ids_for_stage
from .metrics import binary_metrics, select_threshold
from .model_ablation import AblationHybrid
from .paths import CKPT, RUN, ensure

HP = {
    "uci": {
        "lr": 8.605034792033103e-05,
        "weight_decay": 0.0032859708169642424,
        "dropout": 0.4061978796339918,
        "batch_size": 32,
        "pos_weight_multiplier": 1.1830880728874675,
        "entropy_floor_coefficient": 0.002,
        "max_epochs": 50,
        "patience": 10,
    },
    "oulad": {
        "lr": 0.00011844319751820385,
        "weight_decay": 0.0007114476009343421,
        "dropout": 0.31959818254342154,
        "batch_size": 128,
        "pos_weight_multiplier": 0.7790418060840998,
        "entropy_floor_coefficient": 0.005,
        "max_epochs": 40,
        "patience": 8,
    },
}


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StageTrainer:
    def __init__(self, prepared: PreparedDomain, stage: str, ablation: str, *, seed: int):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED")
        self.prepared = prepared
        self.stage = stage
        self.ablation = ablation
        self.seed = seed
        self.hp = HP[prepared.domain]
        self.device = torch.device("cuda")
        self.batch_size = int(self.hp["batch_size"])
        view = prepared.views[stage]
        self.lookup = {str(r): i for i, r in enumerate(view.record_id)}
        recs = list(prepared.static_map)
        self.static_index = {r: i for i, r in enumerate(recs)}
        self.static_mat = torch.as_tensor(
            np.stack([prepared.static_map[r] for r in recs]), dtype=torch.float32, device=self.device
        )
        self.temporal = torch.as_tensor(view.temporal, dtype=torch.float32, device=self.device)
        self.mask = torch.as_tensor(view.temporal_mask, device=self.device)
        self.lengths = torch.as_tensor(view.lengths, device=self.device)
        self.aggregate = torch.as_tensor(view.aggregate, dtype=torch.float32, device=self.device)
        self.agg_avail = torch.as_tensor(view.aggregate_available, device=self.device)
        self.progress = torch.as_tensor(view.progress, dtype=torch.float32, device=self.device)
        self.target = torch.as_tensor(view.target.astype(np.float32), device=self.device)

    def _batch(self, ids: list[str]):
        np_idx = np.fromiter((self.lookup[i] for i in ids), dtype=np.int64, count=len(ids))
        idx = torch.from_numpy(np_idx).to(self.device)
        st = torch.tensor([self.static_index[i] for i in ids], device=self.device, dtype=torch.long)
        x = {
            "static": self.static_mat.index_select(0, st),
            "temporal": self.temporal.index_select(0, idx),
            "temporal_mask": self.mask.index_select(0, idx),
            "lengths": self.lengths.index_select(0, idx),
            "aggregate": self.aggregate.index_select(0, idx),
            "aggregate_available": self.agg_avail.index_select(0, idx),
            "progress": self.progress.index_select(0, idx),
        }
        y = self.target.index_select(0, idx)
        return x, y

    def _predict(self, model, ids: list[str]) -> np.ndarray:
        model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                x, _ = self._batch(chunk)
                logits = model(**x)
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.empty(0, np.float32)

    def _gate_means(self, model, ids: list[str]) -> dict[str, float]:
        model.eval()
        masses = []
        with torch.inference_mode():
            for start in range(0, min(len(ids), 2048), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                x, _ = self._batch(chunk)
                model(**x)
                masses.append(model.last_diagnostics["gate_weights"].float().cpu().numpy())
        if not masses:
            return {"tabular_mass": float("nan"), "cnn_mass": float("nan"), "bilstm_mass": float("nan")}
        w = np.concatenate(masses)
        return {
            "tabular_mass": float(w[:, 0].mean()),
            "cnn_mass": float(w[:, 1].mean()),
            "bilstm_mass": float(w[:, 2].mean()),
        }

    def fit(self, fit_ids: list[str], stop_ids: list[str], valid_ids: list[str], run_id: str) -> dict:
        ensure()
        seed_everything(self.seed)
        torch.backends.cudnn.benchmark = True
        fit_ids = ids_for_stage(self.prepared.views[self.stage], fit_ids)
        stop_ids = ids_for_stage(self.prepared.views[self.stage], stop_ids)
        valid_ids = ids_for_stage(self.prepared.views[self.stage], valid_ids)
        y_fit = np.asarray([int(self.target[self.lookup[i]].item()) for i in fit_ids])
        pos_w = (len(y_fit) - y_fit.sum()) / max(1, y_fit.sum()) * self.hp["pos_weight_multiplier"]
        cfg = HybridConfig(
            static_dim=self.prepared.static_dim,
            temporal_dim=self.prepared.temporal_dim,
            aggregate_dim=self.prepared.aggregate_dim,
            dropout=float(self.hp["dropout"]),
            entropy_floor_coefficient=float(self.hp["entropy_floor_coefficient"]),
        )
        model = AblationHybrid(cfg, self.ablation).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(self.hp["lr"]), weight_decay=float(self.hp["weight_decay"]))
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], dtype=torch.float32, device=self.device))
        scaler = GradScaler("cuda")
        best = -np.inf
        stale = 0
        best_state = None
        history = []
        t0 = time.monotonic()
        rng = np.random.default_rng(self.seed)
        for epoch in range(1, int(self.hp["max_epochs"]) + 1):
            model.train()
            order = np.asarray(fit_ids)
            rng.shuffle(order)
            losses = []
            for start in range(0, len(order), self.batch_size):
                chunk = order[start : start + self.batch_size].tolist()
                if len(chunk) < 2:
                    continue
                x, y = self._batch(chunk)
                opt.zero_grad(set_to_none=True)
                with autocast("cuda", dtype=torch.float16):
                    logits = model(**x)
                    loss = bce(logits, y)
                    if self.ablation == "full":
                        loss = loss + model.fusion_regularization()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach()))
            stop_p = self._predict(model, stop_ids)
            stop_y = np.asarray([int(self.target[self.lookup[i]].item()) for i in stop_ids])
            stop_ap = binary_metrics(stop_y, stop_p)["ap"]
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses) if losses else np.nan), "stop_ap": stop_ap})
            if stop_ap > best + 1e-6:
                best = stop_ap
                stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= int(self.hp["patience"]):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        t_star = select_threshold(
            np.asarray([int(self.target[self.lookup[i]].item()) for i in stop_ids]),
            self._predict(model, stop_ids),
        )
        valid_p = self._predict(model, valid_ids)
        valid_y = np.asarray([int(self.target[self.lookup[i]].item()) for i in valid_ids])
        metrics = binary_metrics(valid_y, valid_p, threshold=t_star)
        ckpt = CKPT / f"{run_id}.pt"
        torch.save({"model_id": "hybrid", "ablation": self.ablation, "config": cfg.__dict__, "state_dict": model.state_dict()}, ckpt)
        payload = {
            "run_id": run_id,
            "domain": self.prepared.domain,
            "stage": self.stage,
            "ablation": self.ablation,
            "seed": self.seed,
            "best_stop_ap": float(best),
            "n_epochs": len(history),
            "history": history,
            "threshold": t_star,
            "valid": metrics,
            "gate": self._gate_means(model, valid_ids),
            "n_fit": len(fit_ids),
            "n_stop": len(stop_ids),
            "n_valid": len(valid_ids),
            "pos_weight": float(pos_w),
            "seconds": time.monotonic() - t0,
            "outer_test_used": False,
            "checkpoint": str(ckpt),
        }
        (RUN / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
