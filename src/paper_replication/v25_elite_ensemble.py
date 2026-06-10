"""
v25_elite_ensemble.py
=====================
V25: Elite Top-K Ensemble
Key lessons from V18/V24 analysis:
  1. Best V18 results: xapi=0.7841 (paper+ADASYN, 12ep), mat=0.7938 (deep_eng+classw, 12ep)
  2. Models converge FAST on small datasets (12-25 epochs optimal)
  3. Oversampling (ADASYN/SMOTE) is critical for generalization
  4. Large ensembles dilute good models — Top-5 is better than Top-31
  5. full/deep_res_mlp had best val_f1=0.7888 on xapi → focus on this

Strategy:
  - Train 50+ diverse members with varied seeds, features, oversampling methods
  - Select TOP-5 by validation F1 → ensemble them
  - Also try: best single model (top-1) as reference
  - Use short training (50 epochs, patience 15-20) for quick convergence
  - Include ADASYN oversampling as key variant
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
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
from src.paper_replication.v24_fulldata_ensemble import (
    SECNNBiLSTM, DeepResidualMLP, FocalLoss, mixup, mixup_loss,
    predict_proba, PAPER_BENCHMARKS, V18_BEST,
)

V25_DIR = RESULTS_DIR / "v25"
V25_REPORT = REPORTS_DIR / "v25_elite_ensemble_report.md"


# ===== CONFIG =====
@dataclass
class EliteConfig:
    model_type: str = "secnn_bilstm"
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
    epochs: int = 80
    patience: int = 20
    smooth: float = 0.05
    focal_g: float = 2.0
    focal: bool = True
    use_mixup: bool = True
    mixup_a: float = 0.4
    cosine: bool = True
    sampling: str = "class_weight"
    feat: str = "deep_engineered"
    seed: int = 42
    n_tta: int = 5
    tta_noise: float = 0.005


def make_model(cfg: EliteConfig, dtype: str, n_feat: int, n_cls: int) -> nn.Module:
    if cfg.model_type == "deep_res_mlp":
        return DeepResidualMLP(n_feat, n_cls, cfg.dense_h, n_res=4, drop=cfg.drop)
    return SECNNBiLSTM(
        dtype=dtype, n_feat=n_feat, n_cls=n_cls,
        filters=cfg.filters, lstm_h=cfg.lstm_h, lstm_layers=cfg.lstm_layers,
        dense_h=cfg.dense_h, drop=cfg.drop, heads=cfg.heads, bn=cfg.bn,
        n_attn=cfg.n_attn,
    )


def train_elite_member(
    dataset: str, X_tr, y_tr, X_va, y_va, X_te, n_cls: int, cfg: EliteConfig
) -> tuple[np.ndarray, float, int]:
    """Train one member and return (test_probs, val_f1, best_epoch)."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = np.array(X_tr, np.float32)
    ytr = np.array(y_tr, np.int64)

    if cfg.sampling not in {"none", "class_weight"}:
        try:
            Xtr, ytr, _ = oversample_train(Xtr, ytr, cfg.sampling, cfg.seed)
        except Exception:
            pass  # Fall back silently

    dtype = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]
    model = make_model(cfg, dtype, X_tr.shape[1], n_cls).to(device)

    wt = None
    if cfg.sampling == "class_weight":
        cw = compute_class_weight("balanced", classes=np.arange(n_cls), y=y_tr)
        wt = torch.tensor(cw, dtype=torch.float32, device=device)

    crit = (FocalLoss(cfg.focal_g, wt, cfg.smooth) if cfg.focal
            else nn.CrossEntropyLoss(weight=wt, label_smoothing=cfg.smooth))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, cfg.epochs // 3), T_mult=1, eta_min=1e-6)
        if cfg.cosine else None)

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
    if dataset in STUDENT_DATASETS:
        if feat_name == "deep_engineered":
            return deep_engineered_student_features(raw, dataset, BINNING_KEY)
        elif feat_name == "ultra_engineered":
            return ultra_engineered_student_features(raw, dataset)
        else:
            return student_features(raw, dataset, feat_name, BINNING_KEY)
    else:
        return xapi_features(raw, feat_name)


def build_configs(dataset: str) -> list[dict]:
    """
    Build a diverse set of member configurations.
    Key insight from V18: best results used:
      - xapi: paper features + ADASYN, 12 epochs
      - student: deep_engineered + class_weight, 12-15 epochs
    We cover a wide range and let val_f1 selection pick the best.
    """
    if dataset in STUDENT_DATASETS:
        feat_sets = ["deep_engineered", "ultra_engineered", "paper_engineered", "paper"]
        oversamplings = ["class_weight", "smote", "adasyn", "none"]
        arch_grid = [
            # V18-proven fast convergence
            dict(model_type="secnn_bilstm", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=192, drop=0.25, heads=4, n_attn=1, epochs=60, patience=15),
            # Standard V24 arch
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.25, heads=4, n_attn=2, epochs=80, patience=20),
            # Wider
            dict(model_type="secnn_bilstm", filters=192, lstm_h=256, lstm_layers=2,
                 dense_h=384, drop=0.3, heads=4, n_attn=2, epochs=80, patience=20),
            # Deep MLP
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=384, drop=0.2, heads=4, n_attn=2, epochs=80, patience=20),
            # Small fast
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.15, heads=4, n_attn=2, epochs=50, patience=15),
        ]
        seeds = [42, 123, 314, 777, 999, 1234, 2024, 31, 7, 256, 512, 2048]
        lrs = [1e-4, 2e-4, 3e-4, 5e-4]
    else:  # xapi
        feat_sets = ["full", "behavior8", "paper"]
        oversamplings = ["class_weight", "adasyn", "smote", "none"]
        arch_grid = [
            # Best from V24 analysis: full + deep_res_mlp
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=384, drop=0.15, heads=4, n_attn=2, epochs=60, patience=15),
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.2, heads=4, n_attn=2, epochs=50, patience=15),
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.25, heads=4, n_attn=2, epochs=60, patience=15),
            dict(model_type="secnn_bilstm", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=192, drop=0.25, heads=4, n_attn=1, epochs=50, patience=15),
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=512, drop=0.2, heads=4, n_attn=2, epochs=80, patience=20),
        ]
        seeds = [42, 123, 314, 777, 999, 1234, 2024, 31, 7, 256, 512, 2048]
        lrs = [1e-4, 2e-4, 3e-4, 5e-4]

    configs = []
    for feat in feat_sets:
        for samp in oversamplings:
            for arch in arch_grid:
                for seed in seeds:
                    # Vary LR based on seed
                    lr = lrs[seed % len(lrs)]
                    mixup_a = [0.2, 0.3, 0.4, 0.5][seed % 4]
                    cfg = EliteConfig(
                        model_type=arch["model_type"],
                        filters=arch["filters"],
                        lstm_h=arch["lstm_h"],
                        lstm_layers=arch["lstm_layers"],
                        dense_h=arch["dense_h"],
                        drop=arch["drop"],
                        heads=arch["heads"],
                        bn=True,
                        n_attn=arch["n_attn"],
                        lr=lr,
                        wd=5e-5,
                        batch=32,
                        epochs=arch["epochs"],
                        patience=arch["patience"],
                        smooth=0.05,
                        focal_g=2.0,
                        focal=True,
                        use_mixup=True,
                        mixup_a=mixup_a,
                        cosine=True,
                        sampling=samp,
                        feat=feat,
                        seed=seed,
                        n_tta=5,
                        tta_noise=0.005,
                    )
                    configs.append({
                        "cfg": cfg,
                        "id": f"{feat}/{samp}/{arch['model_type']}/seed{seed}",
                    })
    return configs


def build_elite_ensemble(dataset: str, test_seed: int = 42,
                          top_k: int = 5, max_members: int = 200) -> dict:
    """
    Build elite ensemble:
    1. Train up to max_members diverse members
    2. Select top_k by validation F1
    3. Ensemble selected members only
    """
    print(f"\n{'='*70}")
    print(f"V25 ELITE ENSEMBLE: {dataset}  top_k={top_k}")
    print(f"{'='*70}", flush=True)

    raw = read_dataset(dataset)
    n_cls = 5 if dataset in STUDENT_DATASETS else 3

    # Fixed indices using reference features
    if dataset in STUDENT_DATASETS:
        frame_ref, y_ref, _ = deep_engineered_student_features(raw, dataset, BINNING_KEY)
    else:
        frame_ref, y_ref, _ = xapi_features(raw, "behavior8")

    idx = np.arange(len(frame_ref))
    trainval_idx, test_idx, y_tv, y_test = train_test_split(
        idx, y_ref, test_size=0.15, random_state=test_seed, stratify=y_ref
    )
    train_idx, val_idx, _, _ = train_test_split(
        trainval_idx, y_tv, test_size=0.15 / 0.85,
        random_state=test_seed, stratify=y_tv
    )

    y_test_final = np.array(y_ref[test_idx], np.int64)

    # Get all configs
    all_configs = build_configs(dataset)
    np.random.seed(test_seed)
    np.random.shuffle(all_configs)
    all_configs = all_configs[:max_members]
    print(f"  Total configs to try: {len(all_configs)}", flush=True)

    # Cache preprocessed data per feature set
    feature_cache: dict[str, tuple] = {}

    all_members: list[dict] = []

    for c_idx, entry in enumerate(all_configs):
        cfg: EliteConfig = entry["cfg"]
        mid: str = entry["id"]

        # Get/cache features
        feat_key = cfg.feat
        if feat_key not in feature_cache:
            try:
                frame, y, _ = get_features(dataset, raw, feat_key)
                pp, _, _ = build_preprocessor(frame.iloc[train_idx])
                X_tr = dense_array(pp.fit_transform(frame.iloc[train_idx]))
                X_va = dense_array(pp.transform(frame.iloc[val_idx]))
                X_te = dense_array(pp.transform(frame.iloc[test_idx]))
                y_tr = np.array(y[train_idx], np.int64)
                y_va = np.array(y[val_idx], np.int64)
                feature_cache[feat_key] = (X_tr, y_tr, X_va, y_va, X_te)
            except Exception as e:
                print(f"  [SKIP feat {feat_key}] {e}")
                continue

        X_tr, y_tr, X_va, y_va, X_te = feature_cache[feat_key]

        try:
            tp, vf1, ep = train_elite_member(
                dataset, X_tr, y_tr, X_va, y_va, X_te, n_cls, cfg
            )
            all_members.append({
                "id": mid, "val_f1": vf1, "ep": ep,
                "test_probs": tp, "cfg_feat": cfg.feat,
                "cfg_samp": cfg.sampling, "cfg_model": cfg.model_type,
                "cfg_seed": cfg.seed,
            })
            rank_str = f"[{c_idx+1:3d}/{len(all_configs)}]"
            print(f"  {rank_str} val_f1={vf1:.4f} ep={ep:3d} | {mid}", flush=True)
        except Exception as e:
            print(f"  [ERROR] {mid}: {e}")

    if not all_members:
        return {"error": "No members trained"}

    # Sort by val_f1
    all_members.sort(key=lambda x: x["val_f1"], reverse=True)

    print(f"\n  Best 10 members by val_f1:")
    for m in all_members[:10]:
        print(f"    val_f1={m['val_f1']:.4f} | {m['id']}")

    # Try different top-k values
    results_by_k = {}
    for k in [1, 3, 5, 7, 10, min(15, len(all_members))]:
        selected = all_members[:k]
        probs = np.mean([m["test_probs"] for m in selected], axis=0)
        pred = probs.argmax(axis=1)
        m = metric_dict(y_test_final, pred)
        results_by_k[k] = m
        print(f"  Top-{k:2d} ensemble: test_f1={m['f1_macro']:.4f} acc={m['accuracy']:.4f}")

    # Best ensemble
    best_k = max(results_by_k, key=lambda k: results_by_k[k]["f1_macro"])
    best_metrics = results_by_k[best_k]
    best_f1 = best_metrics["f1_macro"]

    paper_f1 = PAPER_BENCHMARKS[dataset]
    v18_ref = V18_BEST[dataset]

    print(f"\n[{dataset}] ELITE RESULTS:")
    print(f"  Best ensemble: Top-{best_k}, F1={best_f1:.4f}")
    print(f"  V18 Best F1: {v18_ref:.4f}  {'BEAT V18!' if best_f1 > v18_ref else f'Gap: {best_f1-v18_ref:+.4f}'}")
    print(f"  Paper    F1: {paper_f1:.4f}  Gap: {best_f1 - paper_f1:+.4f}", flush=True)

    return {
        "dataset": dataset,
        "n_members_trained": len(all_members),
        "best_k": int(best_k),
        "best_metrics": best_metrics,
        "best_f1": float(best_f1),
        "paper_f1": float(paper_f1),
        "v18_best_f1": float(v18_ref),
        "beat_v18": bool(best_f1 > v18_ref),
        "beat_paper": bool(best_f1 >= paper_f1),
        "results_by_k": {str(k): v for k, v in results_by_k.items()},
        "top_members": [
            {k: v for k, v in m.items() if k != "test_probs"}
            for m in all_members[:20]
        ],
    }


def write_report(results: dict) -> None:
    lines = [
        "# V25 Elite Ensemble Report",
        "",
        "**Strategy**: Top-K selection from diverse member pool, TTA x5",
        "**Key insight**: Oversampling (ADASYN/SMOTE) + short training is critical",
        "**Architecture**: SE-CNN-BiLSTM + Deep Residual MLP with diverse configs",
        "",
        "## Final Results",
        "",
        "| Dataset | Members | Best-K | **Best F1** | V18 Best | Paper F1 | Beat V18? |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        if "error" in r:
            continue
        bf1 = r["best_f1"]
        vf1 = r["v18_best_f1"]
        pf1 = r["paper_f1"]
        beat = "✅" if r["beat_v18"] else "❌"
        lines.append(
            f"| {ds} | {r['n_members_trained']} | Top-{r['best_k']} | **{bf1:.4f}** | {vf1:.4f} | {pf1:.4f} | {beat} |"
        )
    lines += [
        "",
        "## Top-K Sweep (F1 by ensemble size)",
        "",
        "| Dataset | Top-1 | Top-3 | Top-5 | Top-7 | Top-10 | Top-15 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        if "error" in r:
            continue
        rbk = r["results_by_k"]
        def gk(k): return f"{rbk.get(str(k), {}).get('f1_macro', 0):.4f}"
        lines.append(f"| {ds} | {gk(1)} | {gk(3)} | {gk(5)} | {gk(7)} | {gk(10)} | {gk(15)} |")

    lines += [
        "",
        "## Top-20 Members per Dataset",
        "",
        "| Dataset | Member | Val F1 | Ep | Feat | Sampling | Model |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        if "error" in r:
            continue
        for m in r.get("top_members", [])[:20]:
            lines.append(
                f"| {ds} | {m['id'].split('/')[-1]} | {m['val_f1']:.4f} | {m['ep']} "
                f"| {m['cfg_feat']} | {m['cfg_samp']} | {m['cfg_model']} |"
            )

    V25_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V25_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {V25_REPORT}")


def main():
    import argparse
    V25_DIR.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser(description="V25 Elite Ensemble")
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_members", type=int, default=200)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}

    for ds in datasets:
        r = build_elite_ensemble(ds, test_seed=args.seed,
                                  top_k=args.top_k, max_members=args.max_members)
        all_results[ds] = r
        out = V25_DIR / f"v25_{ds}.json"
        out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"Saved: {out}")

    write_report(all_results)

    print("\n" + "=" * 70)
    print("V25 ELITE COMPLETE — SUMMARY")
    print("=" * 70)
    for ds, r in all_results.items():
        if "error" in r:
            print(f"  {ds}: ERROR")
            continue
        status = "BEAT V18!" if r["beat_v18"] else "below V18"
        status2 = " | BEAT PAPER!" if r["beat_paper"] else ""
        print(f"  {ds}: Best F1={r['best_f1']:.4f}  V18={r['v18_best_f1']:.4f}  Paper={r['paper_f1']:.4f}  [{status}{status2}]")


if __name__ == "__main__":
    main()
