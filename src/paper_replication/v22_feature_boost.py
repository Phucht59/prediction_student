"""
v22_feature_boost.py
Enhanced feature engineering + Optuna sweep targeting F1 >= 0.90.
Key additions:
  - Polynomial features (G1^2, G2^2, G1*G2)
  - Grade trajectory (acceleration, velocity)
  - Subject-specific features
  - Grade-to-class consistency features
  - Bayesian-inspired grade confidence features
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent / "Huflit" / "kltn"))

from src.paper_replication.advanced_experiments import apply_binning, set_seed
from src.paper_replication.pipeline import PROJECT_ROOT, build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    BINNING_KEY, RESULTS_DIR, REPORTS_DIR, STUDENT_DATASETS,
    get_feature_names, metric_dict, read_dataset, student_features, xapi_features,
)
from src.paper_replication.v18_strict_validation import deep_engineered_student_features
from src.paper_replication.v20_advanced_boost import BoostConfig, train_one, boost_split, PAPER_BENCHMARKS
from sklearn.model_selection import train_test_split

V22_DIR = RESULTS_DIR / "v22"


def ultra_engineered_student_features(raw: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Ultra-engineered feature set for student grade prediction.
    All features derived from G1, G2, and raw attributes.
    """
    bk = BINNING_KEY
    g1 = pd.to_numeric(raw["G1"], errors="raise").astype(float)
    g2 = pd.to_numeric(raw["G2"], errors="raise").astype(float)

    # Basic
    delta = g2 - g1                        # Grade improvement
    ratio = g2 / np.maximum(g1, 0.5)      # Improvement ratio
    g_mean = (g1 + g2) / 2.0              # Average grade
    g_max = np.maximum(g1, g2)
    g_min = np.minimum(g1, g2)

    # Polynomial features
    g1_sq = g1 ** 2 / 400.0               # Normalized G1^2
    g2_sq = g2 ** 2 / 400.0               # Normalized G2^2
    g1g2 = g1 * g2 / 400.0                # G1 * G2 interaction

    # Grade trajectory (treat G1, G2 as time series)
    # G2 = G1 + delta → G3 ≈ G2 + delta (linear extrapolation)
    g3_linear_pred = g2 + delta            # Linear extrapolation
    g3_exp_pred = g2 + delta * (g2 / np.maximum(g2 - delta, 0.1))  # Accelerating

    # Binned versions
    g1_bin = apply_binning(g1, bk, dataset).astype(float)
    g2_bin = apply_binning(g2, bk, dataset).astype(float)
    bin_delta = g2_bin - g1_bin            # Grade-class movement

    # Percentile-based features
    g1_pct = g1 / 20.0
    g2_pct = g2 / 20.0

    # Category features (trend)
    trend = pd.Series(
        np.where(delta > 1, "improving", np.where(delta < -1, "declining", "stable")),
        index=raw.index,
    )

    # Categorical from raw dataset
    cols_to_add = {}
    for col in ["failures", "studytime", "absences", "Medu", "Fedu", "higher", "schoolsup"]:
        if col in raw.columns:
            if raw[col].dtype == object:
                cols_to_add[col] = raw[col].astype(str)
            else:
                cols_to_add[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(float)

    data = pd.DataFrame({
        "G1": g1, "G2": g2,
        "G1_sq": g1_sq, "G2_sq": g2_sq, "G1xG2": g1g2,
        "G2_minus_G1": delta, "G2_div_G1": ratio,
        "G_mean": g_mean, "G_max": g_max, "G_min": g_min,
        "G1_pct": g1_pct, "G2_pct": g2_pct,
        "G1_bin": g1_bin, "G2_bin": g2_bin, "bin_delta": bin_delta,
        "G3_linear_pred": g3_linear_pred, "G3_exp_pred": g3_exp_pred,
        "G_trend": trend,
        **cols_to_add,
    }, index=raw.index)

    y = apply_binning(raw["G3"], bk, dataset).to_numpy(dtype=np.int64)
    return data, y, data.columns.tolist()


def ultra_split(dataset: str, raw: pd.DataFrame, seed: int):
    """Prepare ultra-feature strict split."""
    from src.paper_replication.v18_strict_validation import StrictSplit
    if dataset in STUDENT_DATASETS:
        frame, y, fcols = ultra_engineered_student_features(raw, dataset)
        nc = 5
    else:
        frame, y, fcols = xapi_features(raw, "full")
        nc = 3

    idx = np.arange(len(frame))
    tr, tmp, ytr, ytmp = train_test_split(idx, y, test_size=0.30, random_state=seed, stratify=y)
    va, te, yva, yte = train_test_split(tmp, ytmp, test_size=0.50, random_state=seed, stratify=ytmp)
    pp, _, _ = build_preprocessor(frame.iloc[tr])
    Xtr = dense_array(pp.fit_transform(frame.iloc[tr]))
    Xva = dense_array(pp.transform(frame.iloc[va]))
    Xte = dense_array(pp.transform(frame.iloc[te]))
    return StrictSplit(
        X_train=Xtr, y_train=np.array(ytr, np.int64),
        X_val=Xva, y_val=np.array(yva, np.int64),
        X_test=Xte, y_test=np.array(yte, np.int64),
        train_indices=np.array(tr, np.int64), val_indices=np.array(va, np.int64),
        test_indices=np.array(te, np.int64), feature_columns=fcols,
        processed_feature_names=get_feature_names(pp, fcols),
        n_features=int(Xtr.shape[1]), n_classes=int(nc),
    )


def run_quick_ultra_test(datasets):
    """Quick test ultra-engineered features with good default params."""
    print("=== V22 Ultra Feature Test ===", flush=True)
    for dataset in datasets:
        raw = read_dataset(dataset)
        sp = ultra_split(dataset, raw, seed=42)
        cfg = BoostConfig(
            epochs=120, patience=30, seed=42,
            feat="ultra_engineered",
            filters=128, lstm_h=192, lstm_layers=2, dense_h=192,
            drop=0.25, heads=4, bn=True,
            lr=3e-4, wd=1e-4, batch=32,
            focal=True, focal_g=2.0, mixup=True, mixup_a=0.4,
            cosine=True, sampling="class_weight",
            smooth=0.05,
        )
        r = train_one(dataset, sp, cfg)
        pf1 = PAPER_BENCHMARKS[dataset]["f1_macro"]
        print(
            f"[{dataset}] feat={sp.n_features} cls={sp.n_classes} "
            f"val_f1={r['val_f1_macro']:.4f} test_f1={r['test_f1_macro']:.4f} "
            f"test_acc={r['test_accuracy']:.4f} ep={r['best_epoch']} paper={pf1:.4f}",
            flush=True,
        )


def run_ultra_optuna(dataset: str, n_trials=100, seed=42, ep_per_trial=100):
    """Optuna search over ultra feature set."""
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    raw = read_dataset(dataset)
    logs = []

    def obj(trial):
        cfg = BoostConfig(
            filters=trial.suggest_categorical("filters", [64, 96, 128, 192, 256]),
            lstm_h=trial.suggest_categorical("lstm_h", [96, 128, 192, 256]),
            lstm_layers=trial.suggest_categorical("lstm_layers", [1, 2, 3]),
            dense_h=trial.suggest_categorical("dense_h", [96, 128, 192, 256]),
            drop=trial.suggest_float("drop", 0.1, 0.4, step=0.05),
            heads=trial.suggest_categorical("heads", [2, 4, 8]),
            bn=trial.suggest_categorical("bn", [True, False]),
            lr=trial.suggest_float("lr", 5e-5, 3e-3, log=True),
            wd=trial.suggest_float("wd", 1e-6, 5e-3, log=True),
            batch=trial.suggest_categorical("batch", [16, 32, 64]),
            epochs=ep_per_trial, patience=trial.suggest_categorical("patience", [20, 30, 40, 50]),
            smooth=trial.suggest_float("smooth", 0., 0.1, step=0.025),
            focal=trial.suggest_categorical("focal", [True, False]),
            focal_g=trial.suggest_float("focal_g", 1., 3., step=0.5),
            mixup=trial.suggest_categorical("mixup", [True, False]),
            mixup_a=trial.suggest_float("mixup_a", 0.2, 0.5, step=0.1),
            cosine=True,
            sampling=trial.suggest_categorical("sampling", ["class_weight", "smote", "none"]),
            feat="ultra_engineered", seed=seed,
        )
        sp = ultra_split(dataset, raw, seed)
        r = train_one(dataset, sp, cfg)
        vf1 = r["val_f1_macro"]
        logs.append({"trial": trial.number, "val_f1": vf1, "test_f1": r["test_f1_macro"],
                     **trial.params})
        print(f"  T{trial.number:3d}[{dataset}] val={vf1:.4f} test={r['test_f1_macro']:.4f} ep={r['best_epoch']}", flush=True)
        return float(vf1)

    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=15))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params; bv = study.best_value
    print(f"\n[{dataset}] Ultra Optuna: Best val_f1={bv:.4f}")
    return {"dataset": dataset, "best_val_f1": bv, "best_trial": study.best_trial.number,
            "best_params": bp, "trial_log": logs}


def main():
    import argparse
    V22_DIR.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["quick_test", "optuna"])
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.command == "quick_test":
        run_quick_ultra_test(datasets)
    elif args.command == "optuna":
        for ds in datasets:
            r = run_ultra_optuna(ds, args.trials, args.seed, args.epochs)
            out = V22_DIR / f"ultra_optuna_{ds}.json"
            out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
