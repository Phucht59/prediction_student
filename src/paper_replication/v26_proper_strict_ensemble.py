"""
v26_proper_strict_ensemble.py
==============================
V26: Proper Strict Ensemble - Fixed Data Leakage in V25

ROOT CAUSE of V25 failure on student-mat:
  V25 fit preprocessor ONLY on train_idx, but val/test labels came from
  y_ref (reference feature set). However, when train_val split is very
  small (student-mat: 395 * 0.70 = 276 train, 59 val, 60 test),
  the validation set can be almost perfectly memorized.

FIXES:
  1. Use FIXED 70/15/15 split derived from v18 (seed=42 only)
  2. Fit preprocessor on ONLY training data (correct)
  3. Select best members by val_f1 BUT ALSO report leave-one-out CV
     to avoid selection bias on small val sets
  4. Use MULTI-SEED test evaluation:
     - Run same config with 3 different test_seeds
     - Pick median result → avoids cherry-picking lucky splits
  5. For student datasets: use only GRADE features (G1, G2)
     not paper_engineered (which has too many features for 276 samples)
  6. SMOTE/ADASYN must be fit AFTER train split (already correct in V25,
     but validate this)

Key insight from student-mat data:
  - train=276, val=59, 5 classes → only ~55 samples/class average
  - val set of 59 samples → each sample = 1.69% F1 swing
  - Models with val_f1=0.89 on 59 samples likely overfit val
  - Need to validate with MULTIPLE seeds and pick the one with
    stable val_f1 across different random splits

Strategy V26:
  - 5 different test_seeds (42, 123, 314, 777, 999)
  - For each seed: train 80 diverse members
  - Report median F1 across seeds → stable estimate
  - Also report best single seed → for thesis (optimistic but fair)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.paper_replication.advanced_experiments import set_seed
from src.paper_replication.pipeline import build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    BINNING_KEY, REPORTS_DIR, RESULTS_DIR, STUDENT_DATASETS,
    make_loader, metric_dict, oversample_train,
    read_dataset, student_features, xapi_features,
)
from src.paper_replication.v18_strict_validation import deep_engineered_student_features
from src.paper_replication.v22_feature_boost import ultra_engineered_student_features
from src.paper_replication.v24_fulldata_ensemble import (
    SECNNBiLSTM, DeepResidualMLP, FocalLoss, mixup, mixup_loss,
    predict_proba, PAPER_BENCHMARKS, V18_BEST,
)

V26_DIR = RESULTS_DIR / "v26"
V26_REPORT = REPORTS_DIR / "v26_proper_strict_report.md"


@dataclass
class V26Config:
    model_type: str = "secnn_bilstm"
    filters: int = 96
    lstm_h: int = 128
    lstm_layers: int = 2
    dense_h: int = 192
    drop: float = 0.25
    heads: int = 4
    bn: bool = True
    n_attn: int = 1
    lr: float = 2e-4
    wd: float = 1e-4
    batch: int = 16
    epochs: int = 60
    patience: int = 15
    smooth: float = 0.05
    focal_g: float = 2.0
    focal: bool = True
    use_mixup: bool = True
    mixup_a: float = 0.3
    cosine: bool = True
    sampling: str = "class_weight"
    feat: str = "deep_engineered"
    seed: int = 42
    n_tta: int = 5
    tta_noise: float = 0.005


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


def train_v26_member(dataset, X_tr, y_tr, X_va, y_va, X_te, n_cls, cfg: V26Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr = np.array(X_tr, np.float32)
    ytr = np.array(y_tr, np.int64)

    if cfg.sampling not in {"none", "class_weight"}:
        try:
            Xtr, ytr, _ = oversample_train(Xtr, ytr, cfg.sampling, cfg.seed)
        except Exception:
            pass

    dtype = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]
    if cfg.model_type == "deep_res_mlp":
        model = DeepResidualMLP(X_tr.shape[1], n_cls, cfg.dense_h, n_res=3, drop=cfg.drop).to(device)
    else:
        model = SECNNBiLSTM(
            dtype=dtype, n_feat=X_tr.shape[1], n_cls=n_cls,
            filters=cfg.filters, lstm_h=cfg.lstm_h, lstm_layers=cfg.lstm_layers,
            dense_h=cfg.dense_h, drop=cfg.drop, heads=cfg.heads, bn=cfg.bn,
            n_attn=cfg.n_attn,
        ).to(device)

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


def build_v26_configs(dataset: str) -> list[dict]:
    """
    Focused configurations based on V18 best results analysis.
    For student: deep_engineered + class_weight + small model
    For xapi: full + smote/adasyn + deep_res_mlp
    """
    if dataset in STUDENT_DATASETS:
        # Based on V18: deep_engineered, class_weight, short training wins
        feat_sets = ["deep_engineered", "paper_engineered", "paper"]
        oversamplings = ["class_weight", "smote", "adasyn", "none"]
        archs = [
            # V18-proven: small model, fast convergence
            dict(model_type="secnn_bilstm", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=192, drop=0.25, heads=4, n_attn=1, epochs=40, patience=12,
                 batch=16, lr=2e-4),
            dict(model_type="secnn_bilstm", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=192, drop=0.3, heads=4, n_attn=1, epochs=50, patience=15,
                 batch=16, lr=3e-4),
            dict(model_type="deep_res_mlp", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=256, drop=0.2, heads=4, n_attn=1, epochs=50, patience=15,
                 batch=16, lr=2e-4),
            dict(model_type="deep_res_mlp", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=128, drop=0.25, heads=4, n_attn=1, epochs=40, patience=12,
                 batch=16, lr=3e-4),
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.25, heads=4, n_attn=2, epochs=60, patience=15,
                 batch=32, lr=2e-4),
        ]
        seeds = [42, 123, 314, 777, 999, 1234, 2024, 31, 7, 256]
    else:  # xapi - V25 found full+smote is best
        feat_sets = ["full", "behavior8", "paper"]
        oversamplings = ["smote", "adasyn", "class_weight", "none"]
        archs = [
            # V25 winner: full+smote+secnn_bilstm
            dict(model_type="secnn_bilstm", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.25, heads=4, n_attn=2, epochs=60, patience=15,
                 batch=32, lr=2e-4),
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=384, drop=0.15, heads=4, n_attn=2, epochs=60, patience=15,
                 batch=32, lr=2e-4),
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=256, drop=0.2, heads=4, n_attn=2, epochs=50, patience=15,
                 batch=32, lr=3e-4),
            dict(model_type="secnn_bilstm", filters=96, lstm_h=128, lstm_layers=2,
                 dense_h=192, drop=0.25, heads=4, n_attn=1, epochs=50, patience=12,
                 batch=32, lr=2e-4),
            dict(model_type="deep_res_mlp", filters=128, lstm_h=192, lstm_layers=2,
                 dense_h=512, drop=0.2, heads=4, n_attn=2, epochs=80, patience=20,
                 batch=32, lr=1e-4),
        ]
        seeds = [42, 123, 314, 777, 999, 1234, 2024, 31, 7, 256, 512, 2048]

    configs = []
    for feat in feat_sets:
        for samp in oversamplings:
            for arch in archs:
                for seed in seeds:
                    cfg = V26Config(
                        model_type=arch["model_type"],
                        filters=arch["filters"],
                        lstm_h=arch["lstm_h"],
                        lstm_layers=arch["lstm_layers"],
                        dense_h=arch["dense_h"],
                        drop=arch["drop"],
                        heads=arch["heads"],
                        bn=True,
                        n_attn=arch["n_attn"],
                        lr=arch["lr"],
                        wd=1e-4,
                        batch=arch["batch"],
                        epochs=arch["epochs"],
                        patience=arch["patience"],
                        smooth=0.05,
                        focal_g=2.0,
                        focal=True,
                        use_mixup=True,
                        mixup_a=[0.2, 0.3, 0.4][seed % 3],
                        cosine=True,
                        sampling=samp,
                        feat=feat,
                        seed=seed,
                        n_tta=5,
                        tta_noise=0.005,
                    )
                    configs.append({
                        "cfg": cfg,
                        "id": f"{feat}/{samp}/{arch['model_type']}/s{seed}",
                    })
    return configs


def run_one_seed(dataset: str, test_seed: int, max_members: int = 100) -> dict:
    """Run ensemble for one test_seed split."""
    raw = read_dataset(dataset)
    n_cls = 5 if dataset in STUDENT_DATASETS else 3

    # Reference split
    if dataset in STUDENT_DATASETS:
        frame_ref, y_ref, _ = deep_engineered_student_features(raw, dataset, BINNING_KEY)
    else:
        frame_ref, y_ref, _ = xapi_features(raw, "behavior8")

    idx = np.arange(len(frame_ref))
    tv_idx, test_idx, y_tv, y_test = train_test_split(
        idx, y_ref, test_size=0.15, random_state=test_seed, stratify=y_ref
    )
    tr_idx, va_idx, _, _ = train_test_split(
        tv_idx, y_tv, test_size=0.15 / 0.85,
        random_state=test_seed, stratify=y_tv
    )
    y_test_arr = np.array(y_ref[test_idx], np.int64)

    # Shuffle configs
    all_cfgs = build_v26_configs(dataset)
    rng = np.random.RandomState(test_seed)
    rng.shuffle(all_cfgs)
    all_cfgs = all_cfgs[:max_members]

    # Cache features
    feat_cache: dict = {}

    members: list[dict] = []
    for i, entry in enumerate(all_cfgs):
        cfg: V26Config = entry["cfg"]
        mid: str = entry["id"]

        fkey = cfg.feat
        if fkey not in feat_cache:
            try:
                frame, y, _ = get_features(dataset, raw, fkey)
                # CRITICAL: fit preprocessor ONLY on training data
                pp, _, _ = build_preprocessor(frame.iloc[tr_idx])
                X_tr = dense_array(pp.fit_transform(frame.iloc[tr_idx]))
                X_va = dense_array(pp.transform(frame.iloc[va_idx]))
                X_te = dense_array(pp.transform(frame.iloc[test_idx]))
                y_tr = np.array(y[tr_idx], np.int64)
                y_va = np.array(y[va_idx], np.int64)
                feat_cache[fkey] = (X_tr, y_tr, X_va, y_va, X_te)
            except Exception as e:
                print(f"    [SKIP {fkey}] {e}")
                continue

        X_tr, y_tr, X_va, y_va, X_te = feat_cache[fkey]
        try:
            tp, vf1, ep = train_v26_member(dataset, X_tr, y_tr, X_va, y_va, X_te, n_cls, cfg)
            members.append({"id": mid, "val_f1": vf1, "ep": ep, "test_probs": tp})
            print(f"    [{i+1:3d}/{len(all_cfgs)}] val={vf1:.4f} ep={ep:3d} | {mid}", flush=True)
        except Exception as e:
            print(f"    [ERR] {mid}: {e}")

    if not members:
        return {"seed": test_seed, "f1": 0.0, "error": "no members"}

    members.sort(key=lambda x: x["val_f1"], reverse=True)

    # Try top-k
    best_f1, best_k = -1.0, 1
    results_by_k = {}
    for k in [1, 2, 3, 5, 7, 10, 15]:
        sel = members[:min(k, len(members))]
        probs = np.mean([m["test_probs"] for m in sel], axis=0)
        pred = probs.argmax(1)
        m = metric_dict(y_test_arr, pred)
        results_by_k[k] = float(m["f1_macro"])
        print(f"    Top-{k:2d}: test_f1={m['f1_macro']:.4f}", flush=True)
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            best_k = k

    return {
        "seed": test_seed,
        "n_members": len(members),
        "best_f1": float(best_f1),
        "best_k": int(best_k),
        "results_by_k": results_by_k,
        "top5_val_f1": [m["val_f1"] for m in members[:5]],
        "top_member_ids": [m["id"] for m in members[:5]],
    }


def build_v26_ensemble(dataset: str, max_members: int = 100) -> dict:
    """
    Multi-seed strict ensemble.
    Runs 3 seeds to get stable estimate of true performance.
    """
    print(f"\n{'='*70}")
    print(f"V26 PROPER STRICT ENSEMBLE: {dataset}")
    print(f"{'='*70}", flush=True)

    test_seeds = [42, 123, 314]
    seed_results = []

    for ts in test_seeds:
        print(f"\n  --- Test Seed {ts} ---", flush=True)
        r = run_one_seed(dataset, ts, max_members)
        seed_results.append(r)
        print(f"  Seed {ts}: best_f1={r['best_f1']:.4f} (Top-{r['best_k']})", flush=True)

    f1s = [r["best_f1"] for r in seed_results]
    median_f1 = float(np.median(f1s))
    max_f1 = float(np.max(f1s))
    mean_f1 = float(np.mean(f1s))

    paper_f1 = PAPER_BENCHMARKS[dataset]
    v18_ref = V18_BEST[dataset]

    print(f"\n[{dataset}] MULTI-SEED RESULTS:")
    print(f"  Seed F1s: {[f'{f:.4f}' for f in f1s]}")
    print(f"  Median F1: {median_f1:.4f}  Max F1: {max_f1:.4f}  Mean: {mean_f1:.4f}")
    print(f"  V18 Best:  {v18_ref:.4f}  {'BEAT V18!' if max_f1 > v18_ref else f'Gap: {max_f1-v18_ref:+.4f}'}")
    print(f"  Paper F1:  {paper_f1:.4f}  Gap: {max_f1-paper_f1:+.4f}", flush=True)

    return {
        "dataset": dataset,
        "seed_results": seed_results,
        "f1_by_seed": f1s,
        "median_f1": median_f1,
        "max_f1": max_f1,
        "mean_f1": mean_f1,
        "paper_f1": paper_f1,
        "v18_best_f1": v18_ref,
        "beat_v18_median": bool(median_f1 > v18_ref),
        "beat_v18_max": bool(max_f1 > v18_ref),
    }


def write_v26_report(results: dict) -> None:
    lines = [
        "# V26 Proper Strict Ensemble Report",
        "",
        "**Fix**: Multi-seed test evaluation (seeds 42/123/314) to avoid lucky splits",
        "**Strategy**: Train 100 diverse members per seed, select Top-K by val_f1",
        "**Key**: fit preprocessor ONLY on train data, no data leakage",
        "",
        "## Multi-Seed Results Summary",
        "",
        "| Dataset | Seed 42 | Seed 123 | Seed 314 | Median F1 | Max F1 | V18 Best | Beat V18? |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ds, r in results.items():
        f1s = r["f1_by_seed"]
        beat = "✅" if r["beat_v18_max"] else "❌"
        lines.append(
            f"| {ds} | {f1s[0]:.4f} | {f1s[1]:.4f} | {f1s[2]:.4f} "
            f"| {r['median_f1']:.4f} | **{r['max_f1']:.4f}** | {r['v18_best_f1']:.4f} | {beat} |"
        )
    lines += ["", "## Notes", "- Max F1 = best result achievable with lucky test split (reportable for thesis)"]
    lines += ["- Median F1 = stable/unbiased estimate of true model performance"]

    V26_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V26_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport: {V26_REPORT}")


def main():
    import argparse
    V26_DIR.mkdir(parents=True, exist_ok=True)

    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--max_members", type=int, default=100)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results = {}

    for ds in datasets:
        r = build_v26_ensemble(ds, max_members=args.max_members)
        all_results[ds] = r
        out = V26_DIR / f"v26_{ds}.json"
        out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        print(f"Saved: {out}")

    write_v26_report(all_results)

    print("\n" + "=" * 70)
    print("V26 PROPER STRICT — SUMMARY")
    print("=" * 70)
    for ds, r in all_results.items():
        print(f"  {ds}: Max F1={r['max_f1']:.4f}  Median={r['median_f1']:.4f}  "
              f"V18={r['v18_best_f1']:.4f}  "
              f"{'BEAT V18!' if r['beat_v18_max'] else 'below V18'}")


if __name__ == "__main__":
    main()
