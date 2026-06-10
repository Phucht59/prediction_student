"""
v21_kfold_strict_ensemble.py
K-Fold Cross-Validation Ensemble with FIXED test set.
Protocol:
  - Split data: 85% train+val, 15% test (FIXED, used only once at the end)
  - On train+val: apply Stratified K-Fold (K=5)
  - Each fold trains on 4/5 of train+val and validates on 1/5
  - Ensemble K model predictions on the FIXED test set
  - This maximizes training data while maintaining strict test isolation
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent / "Huflit" / "kltn"))

from src.paper_replication.advanced_experiments import apply_binning, set_seed
from src.paper_replication.pipeline import PROJECT_ROOT, build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    BINNING_KEY, MODELS_DIR, REPORTS_DIR, RESULTS_DIR, STUDENT_DATASETS,
    get_feature_names, make_loader, metric_dict, oversample_train,
    predict_proba, read_dataset, student_features, xapi_features,
)
from src.paper_replication.v18_strict_validation import deep_engineered_student_features
from src.paper_replication.v20_advanced_boost import (
    MultiScaleCNNBiLSTM, FocalLoss, mixup, mixup_loss, BoostConfig, PAPER_BENCHMARKS,
)

V21_DIR = RESULTS_DIR / "v21"
V21_REPORT = REPORTS_DIR / "v21_kfold_report.md"


def get_features(dataset: str, raw: pd.DataFrame, feat: str) -> tuple[pd.DataFrame, np.ndarray, list[str], int]:
    if dataset in STUDENT_DATASETS:
        if feat == "deep_engineered":
            frame, y, fcols = deep_engineered_student_features(raw, dataset, BINNING_KEY)
        else:
            frame, y, fcols = student_features(raw, dataset, feat, BINNING_KEY)
        return frame, y, fcols, 5
    frame, y, fcols = xapi_features(raw, feat)
    return frame, y, fcols, 3


def train_fold_model(
    dataset: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    cfg: BoostConfig,
) -> tuple[np.ndarray, float, int]:
    """Train one model on a fold, return test probabilities and val F1."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = np.array(X_train, np.float32)
    ytr = np.array(y_train, np.int64)
    if cfg.sampling not in {"none", "class_weight"}:
        Xtr, ytr, _ = oversample_train(Xtr, ytr, cfg.sampling, cfg.seed)
    dt = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]
    model = MultiScaleCNNBiLSTM(dt, X_train.shape[1], n_classes,
                                  cfg.filters, cfg.lstm_h, cfg.lstm_layers,
                                  cfg.dense_h, cfg.drop, cfg.heads, cfg.bn).to(device)
    wt = None
    if cfg.sampling == "class_weight":
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
        wt = torch.tensor(cw, dtype=torch.float32, device=device)
    crit = FocalLoss(cfg.focal_g, wt, cfg.smooth) if cfg.focal else nn.CrossEntropyLoss(weight=wt, label_smoothing=cfg.smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, cfg.epochs // 3), T_mult=1, eta_min=1e-6) if cfg.cosine else None
    loader = make_loader(Xtr, ytr, cfg.batch, cfg.seed)
    bst, bvf1, bvacc, bep, wait = None, -1., -1., 0, 0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            if cfg.mixup and bx.size(0) > 1:
                mx, ya, yb, lam = mixup(bx, by, cfg.mixup_a)
                loss = mixup_loss(crit, model(mx), ya, yb, lam)
            else:
                loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sched: sched.step()
        vp = predict_proba(model, X_val, cfg.batch, device)
        vm = metric_dict(y_val, vp.argmax(1))
        vf1, vacc = vm["f1_macro"], vm["accuracy"]
        if vf1 > bvf1 + 1e-6 or (abs(vf1 - bvf1) <= 1e-6 and vacc > bvacc + 1e-6):
            bvf1, bvacc, bep, wait = vf1, vacc, ep, 0
            bst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience: break
    model.load_state_dict(bst or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    test_probs = predict_proba(model, X_test, cfg.batch, device)
    return test_probs, float(bvf1), int(bep)


def run_kfold_ensemble(
    dataset: str,
    best_params: dict,
    k: int = 5,
    n_repeat: int = 3,
    test_seed: int = 42,
) -> dict:
    """
    Run K-fold ensemble with strict test isolation.
    n_repeat: number of times to repeat K-fold (for stability), total members = k * n_repeat.
    """
    raw = read_dataset(dataset)
    feat = best_params.get("feat", "deep_engineered" if dataset in STUDENT_DATASETS else "behavior8")
    frame, y, fcols, n_classes = get_features(dataset, raw, feat)

    # Fixed test set (15%) - ISOLATED from all training
    idx = np.arange(len(frame))
    trainval_idx, test_idx, ytrainval, ytest = train_test_split(
        idx, y, test_size=0.15, random_state=test_seed, stratify=y
    )

    # Preprocess based on train+val (no test leakage)
    preprocessor, _, _ = build_preprocessor(frame.iloc[trainval_idx])
    X_all = dense_array(preprocessor.fit_transform(frame.iloc[trainval_idx]))
    X_test = dense_array(preprocessor.transform(frame.iloc[test_idx]))
    y_trainval = y[trainval_idx]

    print(f"[{dataset}] train+val={len(trainval_idx)}, test={len(test_idx)}, classes={n_classes}")
    print(f"  y_test distribution: {np.bincount(ytest)}")

    all_test_probs = []
    all_val_f1s = []

    for repeat in range(n_repeat):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=test_seed + repeat * 100)
        for fold, (train_fold_idx, val_fold_idx) in enumerate(skf.split(X_all, y_trainval)):
            seed = best_params.get("seed", 42) + repeat * 1000 + fold * 17
            cfg = BoostConfig(
                filters=int(best_params.get("filters", 96)),
                lstm_h=int(best_params.get("lstm_h", 128)),
                lstm_layers=int(best_params.get("lstm_layers", 2)),
                dense_h=int(best_params.get("dense_h", 128)),
                drop=float(best_params.get("drop", 0.25)),
                heads=int(best_params.get("heads", 4)),
                bn=bool(best_params.get("bn", True)),
                lr=float(best_params.get("lr", 3e-4)),
                wd=float(best_params.get("wd", 1e-4)),
                batch=int(best_params.get("batch", 32)),
                epochs=200, patience=int(best_params.get("patience", 40)),
                smooth=float(best_params.get("smooth", 0.05)),
                focal_g=float(best_params.get("focal_g", 2.0)),
                focal=bool(best_params.get("focal", True)),
                mixup=bool(best_params.get("mixup", True)),
                mixup_a=float(best_params.get("mixup_a", 0.4)),
                cosine=bool(best_params.get("cosine", True)),
                sampling=str(best_params.get("sampling", "class_weight")),
                feat=feat, seed=seed,
            )
            X_fold_train = X_all[train_fold_idx]
            y_fold_train = y_trainval[train_fold_idx]
            X_fold_val = X_all[val_fold_idx]
            y_fold_val = y_trainval[val_fold_idx]

            test_probs, val_f1, ep = train_fold_model(
                dataset,
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                X_test, n_classes, cfg,
            )
            all_test_probs.append(test_probs)
            all_val_f1s.append(val_f1)
            print(
                f"  R{repeat+1}/F{fold+1} seed={seed}: val_f1={val_f1:.4f} ep={ep}",
                flush=True,
            )

    # Ensemble
    n_members = len(all_test_probs)
    uniform_probs = np.mean(all_test_probs, axis=0)
    uniform_pred = uniform_probs.argmax(axis=1)
    uniform_metrics = metric_dict(ytest, uniform_pred)

    wts = np.exp(np.array(all_val_f1s) * 5); wts /= wts.sum()
    weighted_probs = sum(p * w for p, w in zip(all_test_probs, wts))
    weighted_pred = weighted_probs.argmax(axis=1)
    weighted_metrics = metric_dict(ytest, weighted_pred)

    pf1 = PAPER_BENCHMARKS[dataset]["f1_macro"]
    best_f1 = max(uniform_metrics["f1_macro"], weighted_metrics["f1_macro"])
    print(f"\n[{dataset}] K-fold ({k}x{n_repeat}={n_members} members):")
    print(f"  Uniform  F1={uniform_metrics['f1_macro']:.4f}  Acc={uniform_metrics['accuracy']:.4f}")
    print(f"  Weighted F1={weighted_metrics['f1_macro']:.4f}  Acc={weighted_metrics['accuracy']:.4f}")
    print(f"  Paper F1={pf1:.4f}  Gap={best_f1-pf1:+.4f}")

    return {
        "dataset": dataset, "k": k, "n_repeat": n_repeat, "n_members": n_members,
        "feature_set": feat, "test_seed": test_seed,
        "n_trainval": int(len(trainval_idx)), "n_test": int(len(test_idx)),
        "uniform_metrics": uniform_metrics, "weighted_metrics": weighted_metrics,
        "member_val_f1s": all_val_f1s, "paper_f1": pf1,
        "best_f1": float(best_f1),
    }


def main():
    import argparse
    V21_DIR.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_res = {}
    for ds in datasets:
        opt = V21_DIR.parent / "v20" / f"optuna_{ds}.json"
        if opt.exists():
            bp = json.loads(opt.read_text())["best_params"]
            print(f"[{ds}] Using Optuna params")
        else:
            bp = {
                "filters": 96, "lstm_h": 128, "lstm_layers": 2, "dense_h": 128,
                "drop": 0.25, "heads": 4, "bn": True, "lr": 3e-4, "wd": 1e-4,
                "batch": 32, "patience": 40, "smooth": 0.05, "focal_g": 2.0,
                "focal": True, "mixup": True, "mixup_a": 0.4, "cosine": True,
                "sampling": "class_weight",
                "feat": "deep_engineered" if ds in STUDENT_DATASETS else "behavior8",
            }
            print(f"[{ds}] Using default params")
        r = run_kfold_ensemble(ds, bp, k=args.k, n_repeat=args.repeat, test_seed=args.seed)
        all_res[ds] = r
        out = V21_DIR / f"kfold_{ds}.json"
        out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"Saved: {out}")

    # Write report
    lines = ["# V21 K-Fold Cross-Validation Ensemble Report", "",
             "Fixed 15% test set | K-fold CV on remaining 85% | Strict test isolation", "",
             "| Dataset | Members | Uniform F1 | Weighted F1 | Paper F1 | Gap |",
             "| --- | --- | --- | --- | --- | --- |"]
    for ds, r in all_res.items():
        uf1 = r["uniform_metrics"]["f1_macro"]
        wf1 = r["weighted_metrics"]["f1_macro"]
        pf1 = r["paper_f1"]
        lines.append(f"| {ds} | {r['n_members']} | {uf1:.4f} | {wf1:.4f} | {pf1:.4f} | {r['best_f1']-pf1:+.4f} |")
    V21_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V21_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {V21_REPORT}")


if __name__ == "__main__":
    main()
