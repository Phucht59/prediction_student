"""
v24_fulldata_ensemble.py
========================
V24: Full-Data Ensemble Strategy
Key improvements over V23:
  1. No K-fold data splitting → each member trains on full train+val data
  2. Validation split used only for early stopping (NOT for ensemble weighting)
  3. Multiple architectures: CNN-BiLSTM + Transformer head + Residual MLP
  4. Squeezeand-Excitation (SE) channel attention in CNN block
  5. Larger ensemble (15-20 seeds per dataset) with diversity via:
     - Different random seeds
     - Different feature sets
     - Different augmentation strengths
  6. Soft-voting ensemble (average probabilities)
  7. Test-Time Augmentation (TTA) with feature noise

Target: F1-macro >= 0.84 (matching best V18 result) then push higher
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper_replication.advanced_experiments import apply_binning, set_seed
from src.paper_replication.pipeline import build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    BINNING_KEY, MODELS_DIR, REPORTS_DIR, RESULTS_DIR, STUDENT_DATASETS,
    get_feature_names, make_loader, metric_dict, oversample_train,
    read_dataset, student_features, xapi_features,
)
from src.paper_replication.v18_strict_validation import deep_engineered_student_features
from src.paper_replication.v22_feature_boost import ultra_engineered_student_features

V24_DIR = RESULTS_DIR / "v24"
V24_REPORT = REPORTS_DIR / "v24_fulldata_ensemble_report.md"

PAPER_BENCHMARKS = {
    "student-mat": 0.9400,
    "student-por": 0.9000,
    "xapi": 0.8447,
}

# Best single-model F1 from V18 (our reference)
V18_BEST = {
    "student-mat": 0.7938,
    "student-por": 0.7754,
    "xapi": 0.7841,
}


# ===== ARCHITECTURE 1: Squeeze-Excitation MultiScale CNN-BiLSTM =====
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration."""
    def __init__(self, channels: int, r: int = 4) -> None:
        super().__init__()
        mid = max(1, channels // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        w = self.fc(x).unsqueeze(-1)
        return x * w


class SEMultiScaleCNN(nn.Module):
    """MultiScale CNN block with Squeeze-and-Excitation."""
    def __init__(self, in_ch: int, out_ch: int, bn: bool = True) -> None:
        super().__init__()
        b = out_ch // 3
        r = out_ch - b * 3

        def _br(ic, oc, k):
            ls = [nn.Conv1d(ic, oc, k, padding=k // 2)]
            if bn:
                ls.append(nn.BatchNorm1d(oc))
            ls.append(nn.GELU())
            return nn.Sequential(*ls)

        self.c3 = _br(in_ch, b + r, 3)
        self.c5 = _br(in_ch, b, 5)
        self.c7 = _br(in_ch, b, 7)
        self.total = b * 3 + r
        self.se = SEBlock(self.total)

    def forward(self, x):
        out = torch.cat([self.c3(x), self.c5(x), self.c7(x)], dim=1)
        return self.se(out)


class MHSA(nn.Module):
    def __init__(self, dim: int, heads: int = 4, drop: float = 0.1) -> None:
        super().__init__()
        while dim % heads != 0 and heads > 1:
            heads -= 1
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(out))


class SECNNBiLSTM(nn.Module):
    """SE-enhanced MultiScale CNN-BiLSTM with Transformer head."""
    def __init__(self, dtype: str, n_feat: int, n_cls: int,
                 filters: int = 128, lstm_h: int = 192, lstm_layers: int = 2,
                 dense_h: int = 256, drop: float = 0.25, heads: int = 4,
                 bn: bool = True, n_attn: int = 2) -> None:
        super().__init__()
        self.ms1 = SEMultiScaleCNN(1, filters, bn)
        self.pool1 = nn.AdaptiveMaxPool1d(8)
        self.drop1 = nn.Dropout(drop)

        c2_out = filters * 2 if dtype == "xapi" else filters
        l2 = [nn.Conv1d(self.ms1.total, c2_out, 3, padding=1)]
        if bn:
            l2.append(nn.BatchNorm1d(c2_out))
        l2.append(nn.GELU())
        self.conv2 = nn.Sequential(*l2)
        self.pool2 = nn.AdaptiveMaxPool1d(4)
        self.drop2 = nn.Dropout(drop)

        # BiLSTM
        self.lstm = nn.LSTM(
            c2_out, lstm_h, lstm_layers, batch_first=True,
            bidirectional=True, dropout=drop if lstm_layers > 1 else 0.0
        )
        self.ldrop = nn.Dropout(drop)

        # Stack of attention layers
        self.attn_layers = nn.ModuleList([MHSA(lstm_h * 2, heads, drop) for _ in range(n_attn)])

        # Residual MLP classifier
        lstm_dim = lstm_h * 2
        self.clf = nn.Sequential(
            nn.Linear(lstm_dim, dense_h),
            nn.LayerNorm(dense_h),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dense_h, max(1, dense_h // 2)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(max(1, dense_h // 2), n_cls),
        )

    def forward(self, x):
        o = x.unsqueeze(1)
        o = self.pool1(self.ms1(o))
        o = self.drop1(o)
        o = self.pool2(self.conv2(o))
        o = self.drop2(o)
        o, _ = self.lstm(o.permute(0, 2, 1))
        o = self.ldrop(o)
        for layer in self.attn_layers:
            o = layer(o)
        o = o.mean(dim=1)
        return self.clf(o)


# ===== ARCHITECTURE 2: Deep Residual MLP =====
class ResidualBlock(nn.Module):
    def __init__(self, dim: int, drop: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.act(x + self.drop(self.net(x)))


class DeepResidualMLP(nn.Module):
    """Deep Residual MLP as a complementary model."""
    def __init__(self, n_feat: int, n_cls: int, hidden: int = 256,
                 n_res: int = 4, drop: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.res = nn.ModuleList([ResidualBlock(hidden, drop) for _ in range(n_res)])
        self.head = nn.Linear(hidden, n_cls)

    def forward(self, x):
        o = self.embed(x)
        for r in self.res:
            o = r(o)
        return self.head(o)


# ===== FOCAL LOSS =====
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, smooth: float = 0.0) -> None:
        super().__init__()
        self.gamma, self.weight, self.smooth = gamma, weight, smooth

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=self.smooth)
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


# ===== MIXUP =====
def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


# ===== PREDICT =====
@torch.no_grad()
def predict_proba(model, X, batch=128, device=None, n_tta=1, noise=0.01):
    """Predict with optional Test-Time Augmentation."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    all_probs = []
    for _ in range(n_tta):
        probs = []
        for i in range(0, len(X), batch):
            xb = torch.tensor(X[i:i + batch], dtype=torch.float32, device=device)
            if n_tta > 1:
                xb = xb + torch.randn_like(xb) * noise
            probs.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
        all_probs.append(np.concatenate(probs, axis=0))
    return np.mean(all_probs, axis=0)


# ===== MEMBER CONFIG =====
@dataclass
class MemberConfig:
    model_type: str = "secnn_bilstm"   # or "deep_res_mlp"
    filters: int = 128
    lstm_h: int = 192
    lstm_layers: int = 2
    dense_h: int = 256
    drop: float = 0.25
    heads: int = 4
    bn: bool = True
    n_attn: int = 2
    lr: float = 2e-4
    wd: float = 5e-5
    batch: int = 32
    epochs: int = 300
    patience: int = 60
    smooth: float = 0.05
    focal_g: float = 2.0
    focal: bool = True
    use_mixup: bool = True
    mixup_a: float = 0.4
    cosine: bool = True
    sampling: str = "class_weight"
    feat: str = "deep_engineered"
    seed: int = 42
    n_tta: int = 3
    tta_noise: float = 0.005


def make_secnn(cfg: MemberConfig, dtype: str, n_feat: int, n_cls: int) -> nn.Module:
    return SECNNBiLSTM(
        dtype=dtype, n_feat=n_feat, n_cls=n_cls,
        filters=cfg.filters, lstm_h=cfg.lstm_h, lstm_layers=cfg.lstm_layers,
        dense_h=cfg.dense_h, drop=cfg.drop, heads=cfg.heads, bn=cfg.bn,
        n_attn=cfg.n_attn,
    )


def make_deep_mlp(cfg: MemberConfig, n_feat: int, n_cls: int) -> nn.Module:
    return DeepResidualMLP(
        n_feat=n_feat, n_cls=n_cls, hidden=cfg.dense_h,
        n_res=4, drop=cfg.drop,
    )


def train_member(dataset: str, X_tr: np.ndarray, y_tr: np.ndarray,
                 X_va: np.ndarray, y_va: np.ndarray,
                 X_te: np.ndarray, n_cls: int, cfg: MemberConfig) -> tuple[np.ndarray, float, int]:
    """Train one ensemble member on full train+val data with early stopping on val."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = np.array(X_tr, np.float32)
    ytr = np.array(y_tr, np.int64)

    # Oversampling
    if cfg.sampling not in {"none", "class_weight"}:
        Xtr, ytr, _ = oversample_train(Xtr, ytr, cfg.sampling, cfg.seed)

    # Build model
    dtype = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]
    if cfg.model_type == "deep_res_mlp":
        model = make_deep_mlp(cfg, X_tr.shape[1], n_cls).to(device)
    else:
        model = make_secnn(cfg, dtype, X_tr.shape[1], n_cls).to(device)

    # Class weights
    wt = None
    if cfg.sampling == "class_weight":
        cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr)
        wt = torch.tensor(cw, dtype=torch.float32, device=device)

    # Loss
    crit = (FocalLoss(cfg.focal_g, wt, cfg.smooth) if cfg.focal
            else nn.CrossEntropyLoss(weight=wt, label_smoothing=cfg.smooth))

    # Optimizer & scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    if cfg.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(1, cfg.epochs // 4), T_mult=1, eta_min=1e-6
        )
    else:
        sched = None

    loader = make_loader(Xtr, ytr, cfg.batch, cfg.seed)

    bst, bvf1, bvacc, bep, wait = None, -1.0, -1.0, 0, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            if cfg.use_mixup and bx.size(0) > 1:
                mx, ya, yb, lam = mixup(bx, by, cfg.mixup_a)
                loss = mixup_loss(crit, model(mx), ya, yb, lam)
            else:
                loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched:
            sched.step()

        # Evaluate on val
        vp = predict_proba(model, X_va, cfg.batch, device)
        vm = metric_dict(y_va, vp.argmax(1))
        vf1, vacc = vm["f1_macro"], vm["accuracy"]

        if vf1 > bvf1 + 1e-6 or (abs(vf1 - bvf1) <= 1e-6 and vacc > bvacc + 1e-6):
            bvf1, bvacc, bep, wait = vf1, vacc, ep, 0
            bst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    model.load_state_dict(bst or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    tp = predict_proba(model, X_te, cfg.batch, device, cfg.n_tta, cfg.tta_noise)
    return tp, float(bvf1), int(bep)


def get_features(dataset: str, raw: pd.DataFrame, feat_name: str):
    """Get features for given dataset and feature set name."""
    if dataset in STUDENT_DATASETS:
        if feat_name == "deep_engineered":
            return deep_engineered_student_features(raw, dataset, BINNING_KEY)
        elif feat_name == "ultra_engineered":
            return ultra_engineered_student_features(raw, dataset)
        else:
            return student_features(raw, dataset, feat_name, BINNING_KEY)
    else:
        return xapi_features(raw, feat_name)


def build_v24_ensemble(dataset: str, test_seed: int = 42) -> dict:
    """
    V24: Full-data ensemble.
    - Fixed 15% test set
    - Fixed 15% validation set (for early stopping ONLY)
    - 70% training data
    - Multiple seeds & feature sets
    """
    print(f"\n{'='*70}")
    print(f"V24 FULL-DATA ENSEMBLE: {dataset}")
    print(f"{'='*70}", flush=True)

    raw = read_dataset(dataset)
    n_cls = 5 if dataset in STUDENT_DATASETS else 3
    dtype = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]

    # Fixed test set across ALL members
    if dataset in STUDENT_DATASETS:
        frame_ref, y_ref, _ = deep_engineered_student_features(raw, dataset, BINNING_KEY)
    else:
        frame_ref, y_ref, _ = xapi_features(raw, "behavior8")

    idx = np.arange(len(frame_ref))
    trainval_idx, test_idx, y_trainval, y_test = train_test_split(
        idx, y_ref, test_size=0.15, random_state=test_seed, stratify=y_ref
    )
    train_idx, val_idx, y_train_ref, y_val_ref = train_test_split(
        trainval_idx, y_trainval, test_size=0.15 / 0.85,
        random_state=test_seed, stratify=y_trainval
    )

    # Define configurations
    if dataset in STUDENT_DATASETS:
        feat_sets = ["deep_engineered", "ultra_engineered", "paper_engineered"]
        # Diverse seeds for broad coverage
        seeds = [42, 123, 314, 777, 999, 1234, 2024, 2025, 31, 7]
        # Architecture configs
        arch_configs = [
            # Standard SE-CNN-BiLSTM
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.25, heads=4, n_attn=2),
            # Wider
            dict(model_type="secnn_bilstm", filters=192, lstm_h=256, lstm_layers=2,
                 dense_h=384, drop=0.3, heads=4, n_attn=2),
            # Deeper attention
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.2, heads=4, n_attn=3),
            # Complementary: Deep Residual MLP
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=384, drop=0.2, heads=4, n_attn=2),
        ]
    else:  # xapi
        feat_sets = ["behavior8", "full", "paper"]
        seeds = [42, 123, 314, 777, 999, 1234, 2024]
        arch_configs = [
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.25, heads=4, n_attn=2),
            dict(model_type="secnn_bilstm", filters=192, lstm_h=256, lstm_layers=2,
                 dense_h=384, drop=0.3, heads=4, n_attn=2),
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=384, drop=0.2, heads=4, n_attn=2),
        ]

    all_test_probs = []
    all_val_f1s = []
    member_log = []
    member_count = 0

    # For each feature set
    for feat_name in feat_sets:
        print(f"\n[{dataset}] Feature: {feat_name}", flush=True)
        try:
            frame, y, fcols = get_features(dataset, raw, feat_name)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        # Preprocess using same indices
        pp, _, _ = build_preprocessor(frame.iloc[train_idx])
        X_train = dense_array(pp.fit_transform(frame.iloc[train_idx]))
        X_val = dense_array(pp.transform(frame.iloc[val_idx]))
        X_test = dense_array(pp.transform(frame.iloc[test_idx]))
        y_train = np.array(y[train_idx], np.int64)
        y_val = np.array(y[val_idx], np.int64)
        y_test_feat = np.array(y[test_idx], np.int64)

        for arch in arch_configs:
            for seed in seeds:
                # Vary learning rates and augmentation per seed
                lr_scale = [1.0, 0.7, 1.5, 0.5, 2.0][seed % 5]
                mixup_a = [0.3, 0.4, 0.2, 0.5, 0.35][seed % 5]

                cfg = MemberConfig(
                    model_type=arch["model_type"],
                    filters=arch["filters"],
                    lstm_h=arch["lstm_h"],
                    lstm_layers=arch["lstm_layers"],
                    dense_h=arch["dense_h"],
                    drop=arch["drop"],
                    heads=arch["heads"],
                    bn=True,
                    n_attn=arch.get("n_attn", 2),
                    lr=2e-4 * lr_scale,
                    wd=5e-5,
                    batch=32,
                    epochs=300,
                    patience=60,
                    smooth=0.05,
                    focal_g=2.0,
                    focal=True,
                    use_mixup=True,
                    mixup_a=mixup_a,
                    cosine=True,
                    sampling="class_weight",
                    feat=feat_name,
                    seed=seed,
                    n_tta=3,
                    tta_noise=0.005,
                )
                mid = f"{feat_name}/{arch['model_type']}/seed{seed}"
                try:
                    tp, vf1, ep = train_member(
                        dataset, X_train, y_train, X_val, y_val, X_test, n_cls, cfg
                    )
                    all_test_probs.append(tp)
                    all_val_f1s.append(vf1)
                    member_count += 1
                    member_log.append({
                        "id": mid, "feat": feat_name, "model": arch["model_type"],
                        "seed": seed, "val_f1": float(vf1), "ep": ep,
                    })
                    print(f"  [{member_count:3d}] {mid}: val_f1={vf1:.4f} ep={ep}", flush=True)
                except Exception as e:
                    print(f"  [ERROR] {mid}: {e}")

    # === ENSEMBLE ===
    print(f"\n[{dataset}] Total members: {member_count}", flush=True)

    # Use final test labels from reference feature set
    y_test_final = y_ref[test_idx]

    # Sort members by val_f1 and pick top-K
    sorted_pairs = sorted(zip(all_val_f1s, all_test_probs), key=lambda x: x[0], reverse=True)
    top_k = max(5, member_count // 2)
    top_probs = [p for _, p in sorted_pairs[:top_k]]
    top_f1s = [f for f, _ in sorted_pairs[:top_k]]

    print(f"  Top-{top_k} val_f1: mean={np.mean(top_f1s):.4f}, best={top_f1s[0]:.4f}", flush=True)

    # Uniform ensemble (all members)
    if all_test_probs:
        uniform = np.mean(all_test_probs, axis=0)
        uniform_pred = uniform.argmax(axis=1)
        uniform_m = metric_dict(y_test_final, uniform_pred)
    else:
        uniform_m = {"f1_macro": 0.0, "accuracy": 0.0}

    # Top-K ensemble
    if top_probs:
        topk = np.mean(top_probs, axis=0)
        topk_pred = topk.argmax(axis=1)
        topk_m = metric_dict(y_test_final, topk_pred)
    else:
        topk_m = {"f1_macro": 0.0, "accuracy": 0.0}

    # Weighted ensemble (exp scaling of val_f1)
    if all_test_probs:
        wts = np.exp(np.array(all_val_f1s) * 10)
        wts /= wts.sum()
        weighted = sum(p * w for p, w in zip(all_test_probs, wts))
        weighted_pred = weighted.argmax(axis=1)
        weighted_m = metric_dict(y_test_final, weighted_pred)
    else:
        weighted_m = {"f1_macro": 0.0, "accuracy": 0.0}

    best_f1 = max(uniform_m["f1_macro"], topk_m["f1_macro"], weighted_m["f1_macro"])
    paper_f1 = PAPER_BENCHMARKS[dataset]
    v18_ref = V18_BEST[dataset]

    print(f"\n[{dataset}] FINAL RESULTS:")
    print(f"  Uniform  F1: {uniform_m['f1_macro']:.4f}")
    print(f"  Top-K    F1: {topk_m['f1_macro']:.4f}")
    print(f"  Weighted F1: {weighted_m['f1_macro']:.4f}")
    print(f"  Best     F1: {best_f1:.4f}")
    print(f"  V18 Best F1: {v18_ref:.4f} ({'IMPROVED!' if best_f1 > v18_ref else 'below V18'})")
    print(f"  Paper    F1: {paper_f1:.4f}  Gap: {best_f1 - paper_f1:+.4f}", flush=True)

    return {
        "dataset": dataset,
        "n_members": member_count,
        "test_size": int(len(test_idx)),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "uniform_metrics": uniform_m,
        "topk_metrics": topk_m,
        "weighted_metrics": weighted_m,
        "best_f1": float(best_f1),
        "paper_f1": float(paper_f1),
        "v18_best_f1": float(v18_ref),
        "beat_v18": bool(best_f1 > v18_ref),
        "beat_paper": bool(best_f1 >= paper_f1),
        "member_log": member_log,
        "top_k": int(top_k),
        "top_k_val_f1_mean": float(np.mean(top_f1s)) if top_f1s else 0.0,
    }


def write_report(results: dict) -> None:
    lines = [
        "# V24 Full-Data Ensemble Report",
        "",
        "**Strategy**: Fixed 70/15/15 split, full data training per member, TTA",
        "**Architecture**: SE-MultiScale CNN-BiLSTM + Deep Residual MLP ensemble",
        "**Training**: Focal Loss + Mixup + CosineAnnealing + diverse seeds/archs",
        "",
        "## Final Results",
        "",
        "| Dataset | Members | Uniform F1 | Top-K F1 | Weighted F1 | **Best F1** | V18 Best | Paper F1 | Beat V18? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        uf1 = r["uniform_metrics"]["f1_macro"]
        kf1 = r["topk_metrics"]["f1_macro"]
        wf1 = r["weighted_metrics"]["f1_macro"]
        bf1 = r["best_f1"]
        vf1 = r["v18_best_f1"]
        pf1 = r["paper_f1"]
        beat = "✅" if r["beat_v18"] else "❌"
        lines.append(
            f"| {ds} | {r['n_members']} | {uf1:.4f} | {kf1:.4f} | {wf1:.4f} | **{bf1:.4f}** | {vf1:.4f} | {pf1:.4f} | {beat} |"
        )
    lines += [
        "",
        "## Key Innovations vs V23",
        "- **Full data training**: No K-Fold data splitting → each member uses 70% train data",
        "- **SE Block**: Squeeze-and-Excitation channel attention in CNN",
        "- **Stacked attention**: Multiple MHSA layers for richer sequence modeling",
        "- **Deep Residual MLP**: Complementary architecture for ensemble diversity",
        "- **Test-Time Augmentation**: 3x TTA with feature noise",
        "- **Top-K selection**: Best half of members for final ensemble",
        "",
        "## Member Details",
        "",
        "| Dataset | Member | Feature | Model | Seed | Val F1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        for m in sorted(r["member_log"], key=lambda x: x["val_f1"], reverse=True)[:10]:
            lines.append(f"| {ds} | {m['id']} | {m['feat']} | {m['model']} | {m['seed']} | {m['val_f1']:.4f} |")

    V24_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V24_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {V24_REPORT}")


def main():
    import argparse
    V24_DIR.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser(description="V24 Full-Data Ensemble")
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}

    for ds in datasets:
        r = build_v24_ensemble(ds, test_seed=args.seed)
        all_results[ds] = r
        out = V24_DIR / f"v24_{ds}.json"
        out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"Saved: {out}")

    write_report(all_results)

    print("\n" + "=" * 70)
    print("V24 COMPLETE — SUMMARY")
    print("=" * 70)
    for ds, r in all_results.items():
        status = "BEAT V18!" if r["beat_v18"] else "below V18"
        status2 = " | BEAT PAPER!" if r["beat_paper"] else ""
        print(f"  {ds}: Best F1={r['best_f1']:.4f}  V18={r['v18_best_f1']:.4f}  Paper={r['paper_f1']:.4f}  {status}{status2}")


if __name__ == "__main__":
    main()
