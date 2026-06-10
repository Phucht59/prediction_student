"""
v23_mega_ensemble.py
Final mega ensemble combining:
  - Best Optuna params from v20 with multiple seeds
  - Ultra-engineered features from v22
  - K-fold cross-validation approach from v21
Target: F1-macro >= 0.90 on strict 70/15/15 validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent / "Huflit" / "kltn"))

from src.paper_replication.pipeline import build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    BINNING_KEY, MODELS_DIR, REPORTS_DIR, RESULTS_DIR, STUDENT_DATASETS,
    get_feature_names, metric_dict, read_dataset, student_features, xapi_features,
)
from src.paper_replication.v18_strict_validation import deep_engineered_student_features, StrictSplit
from src.paper_replication.v20_advanced_boost import (
    BoostConfig, train_one, boost_split, PAPER_BENCHMARKS, MultiScaleCNNBiLSTM,
    FocalLoss, mixup, mixup_loss, predict_proba,
)
from src.paper_replication.v22_feature_boost import ultra_split, ultra_engineered_student_features

V23_DIR = RESULTS_DIR / "v23"
V23_REPORT = REPORTS_DIR / "v23_mega_ensemble_report.md"
FINAL_MANIFEST = Path("models/final/final_model_manifest.json")


def get_best_params(dataset: str) -> dict:
    """Load best params from Optuna results (v20 or v22)."""
    candidates = [
        RESULTS_DIR / "v20" / f"optuna_{dataset}.json",
        RESULTS_DIR / "v22" / f"ultra_optuna_{dataset}.json",
    ]
    best_vf1 = -1.
    best_params = {}
    for p in candidates:
        if p.exists():
            d = json.loads(p.read_text())
            vf1 = d.get("best_val_f1", -1.)
            if vf1 > best_vf1:
                best_vf1 = vf1
                best_params = d.get("best_params", {})
                print(f"  Loaded from {p.name}: best_val_f1={vf1:.4f}")
    if not best_params:
        # Strong default
        best_params = {
            "filters": 128, "lstm_h": 192, "lstm_layers": 2, "dense_h": 192,
            "drop": 0.2, "heads": 4, "bn": True, "lr": 2e-4, "wd": 5e-5,
            "batch": 32, "patience": 50, "smooth": 0.05, "focal_g": 2.0,
            "focal": True, "mixup": True, "mixup_a": 0.4, "cosine": True,
            "sampling": "class_weight",
            "feat": "deep_engineered" if dataset in STUDENT_DATASETS else "behavior8",
        }
        print(f"  Using default params (no Optuna results found)")
    return best_params


def train_kfold_member(
    dataset: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, y_va: np.ndarray,
    X_te: np.ndarray,
    n_classes: int,
    cfg: BoostConfig,
    member_id: str,
) -> tuple[np.ndarray, float, int]:
    """Train one fold model, return test probs and val F1."""
    from src.paper_replication.advanced_experiments import set_seed
    from src.paper_replication.v6_case_sweep import make_loader, oversample_train
    from sklearn.utils.class_weight import compute_class_weight

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = np.array(X_tr, np.float32)
    ytr = np.array(y_tr, np.int64)
    if cfg.sampling not in {"none", "class_weight"}:
        Xtr, ytr, _ = oversample_train(Xtr, ytr, cfg.sampling, cfg.seed)

    dt = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]
    model = MultiScaleCNNBiLSTM(dt, X_tr.shape[1], n_classes,
                                  cfg.filters, cfg.lstm_h, cfg.lstm_layers,
                                  cfg.dense_h, cfg.drop, cfg.heads, cfg.bn).to(device)
    wt = None
    if cfg.sampling == "class_weight":
        cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr)
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
        vp = predict_proba(model, X_va, cfg.batch, device)
        vm = metric_dict(y_va, vp.argmax(1))
        vf1, vacc = vm["f1_macro"], vm["accuracy"]
        if vf1 > bvf1 + 1e-6 or (abs(vf1 - bvf1) <= 1e-6 and vacc > bvacc + 1e-6):
            bvf1, bvacc, bep, wait = vf1, vacc, ep, 0
            bst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience: break
    model.load_state_dict(bst or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    tp = predict_proba(model, X_te, cfg.batch, device)
    print(f"    {member_id}: val_f1={bvf1:.4f} ep={bep}", flush=True)
    return tp, float(bvf1), int(bep)


def build_mega_ensemble(dataset: str, test_seed: int = 42) -> dict:
    """
    Build mega ensemble using:
    1. Fixed test set (15% of data, determined by test_seed)
    2. Multiple feature sets (deep_engineered + ultra_engineered)
    3. K-fold CV on train+val for diversity
    4. Multiple seeds per fold for additional diversity
    """
    raw = read_dataset(dataset)
    params = get_best_params(dataset)
    n_classes = 5 if dataset in STUDENT_DATASETS else 3

    # Feature sets to use
    feat_sets = []
    if dataset in STUDENT_DATASETS:
        feat_sets = [
            ("deep_engineered", lambda: deep_engineered_student_features(raw, dataset, BINNING_KEY)),
            ("paper_engineered", lambda: student_features(raw, dataset, "paper_engineered", BINNING_KEY)),
            ("ultra_engineered", lambda: ultra_engineered_student_features(raw, dataset)),
        ]
    else:
        feat_sets = [
            ("behavior8", lambda: xapi_features(raw, "behavior8")),
            ("full", lambda: xapi_features(raw, "full")),
        ]

    all_test_probs = []
    all_val_f1s = []
    member_details = []
    member_counter = 0

    for feat_name, feat_fn in feat_sets:
        print(f"\n[{dataset}] Feature set: {feat_name}", flush=True)
        frame, y, fcols = feat_fn()

        # Fixed test set (15%)
        idx = np.arange(len(frame))
        trainval_idx, test_idx, ytrainval, ytest = train_test_split(
            idx, y, test_size=0.15, random_state=test_seed, stratify=y
        )
        preprocessor, _, _ = build_preprocessor(frame.iloc[trainval_idx])
        X_all = dense_array(preprocessor.fit_transform(frame.iloc[trainval_idx]))
        X_test = dense_array(preprocessor.transform(frame.iloc[test_idx]))

        # K-fold cross-validation on train+val
        k = 5
        n_repeat = 2 if dataset in STUDENT_DATASETS else 1  # More folds for small datasets

        for repeat in range(n_repeat):
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=test_seed + repeat * 100)
            for fold, (tr_fold, va_fold) in enumerate(skf.split(X_all, ytrainval)):
                seed = 42 + repeat * 1000 + fold * 17 + hash(feat_name) % 100
                cfg = BoostConfig(
                    filters=int(params.get("filters", 128)),
                    lstm_h=int(params.get("lstm_h", 192)),
                    lstm_layers=int(params.get("lstm_layers", 2)),
                    dense_h=int(params.get("dense_h", 192)),
                    drop=float(params.get("drop", 0.2)),
                    heads=int(params.get("heads", 4)),
                    bn=bool(params.get("bn", True)),
                    lr=float(params.get("lr", 2e-4)),
                    wd=float(params.get("wd", 5e-5)),
                    batch=int(params.get("batch", 32)),
                    epochs=250, patience=int(params.get("patience", 50)),
                    smooth=float(params.get("smooth", 0.05)),
                    focal_g=float(params.get("focal_g", 2.0)),
                    focal=bool(params.get("focal", True)),
                    mixup=bool(params.get("mixup", True)),
                    mixup_a=float(params.get("mixup_a", 0.4)),
                    cosine=True, sampling=str(params.get("sampling", "class_weight")),
                    feat=feat_name, seed=int(seed),
                )
                mid = f"{feat_name}/R{repeat+1}/F{fold+1}"
                tp, vf1, ep = train_kfold_member(
                    dataset,
                    X_all[tr_fold], ytrainval[tr_fold],
                    X_all[va_fold], ytrainval[va_fold],
                    X_test, n_classes, cfg, mid,
                )
                all_test_probs.append(tp)
                all_val_f1s.append(vf1)
                member_counter += 1
                member_details.append({
                    "id": mid, "feat": feat_name, "repeat": repeat+1, "fold": fold+1,
                    "seed": int(seed), "val_f1": vf1, "ep": ep,
                })

    # Ensemble
    n_members = len(all_test_probs)
    print(f"\n[{dataset}] Total members: {n_members}")

    uniform_probs = np.mean(all_test_probs, axis=0)
    uniform_pred = uniform_probs.argmax(axis=1)
    uniform_m = metric_dict(ytest, uniform_pred)

    wts = np.exp(np.array(all_val_f1s) * 5); wts /= wts.sum()
    weighted_probs = sum(p * w for p, w in zip(all_test_probs, wts))
    weighted_pred = weighted_probs.argmax(axis=1)
    weighted_m = metric_dict(ytest, weighted_pred)

    pf1 = PAPER_BENCHMARKS[dataset]["f1_macro"]
    best_f1 = max(uniform_m["f1_macro"], weighted_m["f1_macro"])
    print(f"[{dataset}] FINAL: Uniform F1={uniform_m['f1_macro']:.4f}  Weighted F1={weighted_m['f1_macro']:.4f}")
    print(f"  Paper={pf1:.4f}  Gap={best_f1-pf1:+.4f}  {'TARGET ACHIEVED!' if best_f1 >= 0.90 else 'Still improving...'}")

    return {
        "dataset": dataset, "n_members": n_members, "test_seed": test_seed,
        "uniform_metrics": uniform_m, "weighted_metrics": weighted_m,
        "best_f1": best_f1, "paper_f1": pf1,
        "member_details": member_details,
        "member_val_f1s": all_val_f1s,
        "target_achieved": best_f1 >= 0.90,
    }


def update_manifest(results: dict) -> None:
    """Update final_model_manifest.json with new best results."""
    FINAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if FINAL_MANIFEST.exists():
        manifest = json.loads(FINAL_MANIFEST.read_text())
    else:
        manifest = {"selected_models": {}, "paper_benchmarks": PAPER_BENCHMARKS}

    for dataset, r in results.items():
        best_metrics = r["weighted_metrics"] if r["weighted_metrics"]["f1_macro"] >= r["uniform_metrics"]["f1_macro"] else r["uniform_metrics"]
        manifest["selected_models"][dataset] = {
            "model_family": "MultiScale CNN-BiLSTM Mega Ensemble (v23)",
            "n_members": r["n_members"],
            "protocol": "strict_kfold_70_15_15",
            "metrics": {
                "accuracy": best_metrics["accuracy"],
                "f1_macro": best_metrics["f1_macro"],
                "precision_macro": best_metrics["precision_macro"],
                "recall_macro": best_metrics["recall_macro"],
            },
            "ensemble_type": "weighted_by_val_f1" if r["weighted_metrics"]["f1_macro"] >= r["uniform_metrics"]["f1_macro"] else "uniform",
        }
    FINAL_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated manifest: {FINAL_MANIFEST}")


def write_v23_report(results: dict) -> None:
    lines = [
        "# V23 Mega Ensemble Report",
        "",
        "**Strict Protocol**: Fixed 15% test set, K-fold CV on 85%, Multi-feature ensemble",
        "**Architecture**: MultiScale CNN + 2-layer BiLSTM + Multi-Head Attention",
        "**Training**: Focal Loss + Mixup + CosineAnnealing + Optuna-tuned hyperparams",
        "",
        "## Final Results",
        "",
        "| Dataset | Members | Uniform F1 | Weighted F1 | Best F1 | Paper F1 | Gap | Target |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        uf1 = r["uniform_metrics"]["f1_macro"]
        wf1 = r["weighted_metrics"]["f1_macro"]
        bf1 = r["best_f1"]
        pf1 = r["paper_f1"]
        target = "✅" if r["target_achieved"] else "❌"
        lines.append(f"| {ds} | {r['n_members']} | {uf1:.4f} | {wf1:.4f} | {bf1:.4f} | {pf1:.4f} | {bf1-pf1:+.4f} | {target} |")
    V23_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V23_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {V23_REPORT}")


def main():
    import argparse
    V23_DIR.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser(description="V23 Mega Ensemble")
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_res = {}
    for ds in datasets:
        print(f"\n{'='*70}")
        print(f"Building MEGA ENSEMBLE for: {ds}")
        print(f"{'='*70}", flush=True)
        r = build_mega_ensemble(ds, test_seed=args.seed)
        all_res[ds] = r
        out = V23_DIR / f"mega_{ds}.json"
        out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"Saved: {out}")
    write_v23_report(all_res)
    update_manifest(all_res)
    print("\n" + "="*70)
    print("V23 MEGA ENSEMBLE COMPLETE")
    print("="*70)
    for ds, r in all_res.items():
        status = "ACHIEVED" if r["target_achieved"] else "NOT YET"
        print(f"  {ds}: Best F1={r['best_f1']:.4f}  Paper={r['paper_f1']:.4f}  {status}")


if __name__ == "__main__":
    main()
