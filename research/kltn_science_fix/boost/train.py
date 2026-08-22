"""Mixed-stage trainer. Binary labels unchanged. Multi-metric STOP. Safe GPU."""
from __future__ import annotations

import copy
import json
import os
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn import functional as F

from research.kltn_science_fix.data import PreparedDomain, ids_for_stage
from research.kltn_science_fix.metrics import binary_metrics, select_threshold
from research.kltn_science_fix.paths import ART, ensure

from .model import BoostHybrid

BOOST_RUN = ART / "boost" / "runs"
BOOST_CKPT = ART / "boost" / "checkpoints"

HP = {
    "uci": {
        "lr": 8.605034792033103e-05,
        "weight_decay": 0.0032859708169642424,
        "dropout": 0.4061978796339918,
        "batch_size": 32,
        "pos_weight_multiplier": 1.1830880728874675,
        "entropy_floor_coefficient": 0.002,
        "max_epochs": 40,
        "patience": 10,
        "train_stages": ("S0", "S1", "S2"),
        "stop_stages": ("S1", "S2"),
    },
    "oulad": {
        "lr": 0.00011844319751820385,
        "weight_decay": 0.0007114476009343421,
        "dropout": 0.31959818254342154,
        "batch_size": 128,
        "pos_weight_multiplier": 0.7790418060840998,
        "entropy_floor_coefficient": 0.005,
        "max_epochs": 40,
        "patience": 10,
        "train_stages": ("35pct", "50pct", "75pct"),
        "stop_stages": ("35pct", "50pct", "75pct"),
    },
}
RANK_LAMBDA = 0.05


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(4)


def pairwise_rank_loss(logits: torch.Tensor, labels: torch.Tensor, max_pairs: int = 32) -> torch.Tensor:
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    n = int(min(pos.numel(), neg.numel(), max_pairs))
    pi = torch.randint(0, pos.numel(), (n,), device=logits.device)
    ni = torch.randint(0, neg.numel(), (n,), device=logits.device)
    return F.softplus(-(pos[pi] - neg[ni])).mean()


def combo_score(rows: list[dict]) -> float:
    ap = float(np.mean([r["ap"] for r in rows]))
    f1 = float(np.mean([r["f1"] for r in rows]))
    roc = float(np.mean([r["roc_auc"] for r in rows]))
    rec = float(np.mean([r["recall"] for r in rows]))
    prec = float(np.mean([r["precision"] for r in rows]))
    ece = float(np.mean([r["ece"] for r in rows]))
    return ap + 0.3 * f1 + 0.15 * roc + 0.1 * rec + 0.1 * prec - 0.25 * ece


class MixedTrainer:
    def __init__(self, prepared: PreparedDomain, *, seed: int):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_REQUIRED")
        _setup_threads()
        self.prepared = prepared
        self.seed = seed
        self.hp = HP[prepared.domain]
        self.device = torch.device("cuda")
        self.batch_size = int(self.hp["batch_size"])
        self.train_stages = tuple(s for s in self.hp["train_stages"] if s in prepared.views)
        self.stop_stages = tuple(s for s in self.hp["stop_stages"] if s in prepared.views)
        recs = list(prepared.static_map)
        self.static_index = {r: i for i, r in enumerate(recs)}
        self.static_mat = torch.as_tensor(
            np.stack([prepared.static_map[r] for r in recs]), dtype=torch.float32, device=self.device
        )
        self.lookups = {}
        self.gpu = {}
        for stage in self.train_stages:
            view = prepared.views[stage]
            self.lookups[stage] = {str(r): i for i, r in enumerate(view.record_id)}
            self.gpu[stage] = {
                "temporal": torch.as_tensor(view.temporal, dtype=torch.float32, device=self.device),
                "mask": torch.as_tensor(view.temporal_mask, device=self.device),
                "lengths": torch.as_tensor(view.lengths, device=self.device),
                "aggregate": torch.as_tensor(view.aggregate, dtype=torch.float32, device=self.device),
                "agg": torch.as_tensor(view.aggregate_available, device=self.device),
                "progress": torch.as_tensor(view.progress, dtype=torch.float32, device=self.device),
                "target": torch.as_tensor(view.target.astype(np.float32), device=self.device),
            }

    def _batch(self, stage: str, ids: list[str]):
        lookup = self.lookups[stage]
        np_idx = np.fromiter((lookup[i] for i in ids), dtype=np.int64, count=len(ids))
        idx = torch.from_numpy(np_idx).to(self.device)
        st = torch.tensor([self.static_index[i] for i in ids], device=self.device, dtype=torch.long)
        g = self.gpu[stage]
        x = {
            "static": self.static_mat.index_select(0, st),
            "temporal": g["temporal"].index_select(0, idx),
            "temporal_mask": g["mask"].index_select(0, idx),
            "lengths": g["lengths"].index_select(0, idx),
            "aggregate": g["aggregate"].index_select(0, idx),
            "aggregate_available": g["agg"].index_select(0, idx),
            "progress": g["progress"].index_select(0, idx),
        }
        return x, g["target"].index_select(0, idx)

    def _predict(self, model, stage: str, ids: list[str]) -> np.ndarray:
        model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(ids), self.batch_size):
                chunk = ids[start : start + self.batch_size]
                x, _ = self._batch(stage, chunk)
                logits = model(**x)
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.empty(0, np.float32)

    def _stage_metrics(self, model, stage: str, ids: list[str], threshold: float | None = None) -> dict:
        ids = ids_for_stage(self.prepared.views[stage], ids)
        p = self._predict(model, stage, ids)
        y = np.asarray([int(self.gpu[stage]["target"][self.lookups[stage][i]].item()) for i in ids])
        t = select_threshold(y, p) if threshold is None else float(threshold)
        m = binary_metrics(y, p, threshold=t)
        m["stage"] = stage
        return m

    def fit(self, fit_ids: list[str], stop_ids: list[str], valid_ids: list[str], run_id: str) -> dict:
        ensure()
        BOOST_RUN.mkdir(parents=True, exist_ok=True)
        BOOST_CKPT.mkdir(parents=True, exist_ok=True)
        seed_everything(self.seed)
        fit_by = {s: ids_for_stage(self.prepared.views[s], fit_ids) for s in self.train_stages}
        stop_by = {s: ids_for_stage(self.prepared.views[s], stop_ids) for s in self.stop_stages}
        valid_by = {s: ids_for_stage(self.prepared.views[s], valid_ids) for s in self.train_stages}
        y_fit = []
        for stage, ids in fit_by.items():
            y_fit.extend(int(self.gpu[stage]["target"][self.lookups[stage][i]].item()) for i in ids)
        y_fit = np.asarray(y_fit)
        pos_w = (len(y_fit) - y_fit.sum()) / max(1, y_fit.sum()) * self.hp["pos_weight_multiplier"]
        first = self.prepared.views[self.train_stages[0]]
        model = BoostHybrid(
            static_dim=self.prepared.static_dim,
            temporal_dim=int(first.temporal.shape[2]),
            aggregate_dim=self.prepared.aggregate_dim,
            dropout=float(self.hp["dropout"]),
            entropy_floor_coefficient=float(self.hp["entropy_floor_coefficient"]),
        ).to(self.device)
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
            losses = []
            for stage in self.train_stages:
                order = np.asarray(fit_by[stage])
                rng.shuffle(order)
                for start in range(0, len(order), self.batch_size):
                    chunk = order[start : start + self.batch_size].tolist()
                    if len(chunk) < 4:
                        continue
                    x, y = self._batch(stage, chunk)
                    opt.zero_grad(set_to_none=True)
                    with autocast("cuda", dtype=torch.float16):
                        logits = model(**x)
                        loss = bce(logits, y) + RANK_LAMBDA * pairwise_rank_loss(logits.float(), y) + model.fusion_regularization()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    losses.append(float(loss.detach()))
            stop_rows = [self._stage_metrics(model, stage, stop_by[stage]) for stage in self.stop_stages]
            score = combo_score(stop_rows)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses) if losses else np.nan),
                    "stop_score": score,
                    "stop": {r["stage"]: {k: r[k] for k in ("ap", "f1", "precision", "recall", "roc_auc", "ece")} for r in stop_rows},
                }
            )
            if score > best + 1e-6:
                best = score
                stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= int(self.hp["patience"]):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        thresholds = {}
        stop_final = {}
        for stage in self.stop_stages:
            m = self._stage_metrics(model, stage, stop_by[stage])
            thresholds[stage] = m["threshold"]
            stop_final[stage] = m
        valid_final = {}
        for stage in self.train_stages:
            t = thresholds.get(stage, list(thresholds.values())[0])
            valid_final[stage] = self._stage_metrics(model, stage, valid_by[stage], threshold=t)
        ckpt = BOOST_CKPT / f"{run_id}.pt"
        torch.save({"model": "BoostHybrid", "state_dict": model.state_dict(), "run_id": run_id}, ckpt)
        payload = {
            "run_id": run_id,
            "domain": self.prepared.domain,
            "seed": self.seed,
            "binary_label_unchanged": True,
            "oulad_20pct_used": False,
            "stop_stages": list(self.stop_stages),
            "train_stages": list(self.train_stages),
            "stop_score_best": float(best),
            "n_epochs": len(history),
            "history": history,
            "thresholds": thresholds,
            "stop": {k: {kk: vv for kk, vv in v.items() if kk != "stage"} for k, v in stop_final.items()},
            "valid": {k: {kk: vv for kk, vv in v.items() if kk != "stage"} for k, v in valid_final.items()},
            "pos_weight": float(pos_w),
            "rank_lambda": RANK_LAMBDA,
            "seconds": time.monotonic() - t0,
            "outer_test_used": False,
            "checkpoint": str(ckpt),
        }
        (BOOST_RUN / f"{run_id}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
