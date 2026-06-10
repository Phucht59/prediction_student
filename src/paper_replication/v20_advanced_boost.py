"""
v20_advanced_boost.py - Advanced CNN-BiLSTM targeting F1 >= 0.90 on strict validation.
Key techniques:
  1. MultiScale CNN Block (parallel conv with kernels 3,5,7)
  2. Multi-Head Self-Attention over BiLSTM output
  3. Focal Loss (gamma=2) to handle class imbalance
  4. Mixup data augmentation (alpha=0.4)
  5. CosineAnnealingWarmRestarts LR scheduler
  6. Extended Optuna search (100+ trials)
  7. Large 7-member ensemble with diverse seeds
"""
from __future__ import annotations
import json, random, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent / "Huflit" / "kltn"))

from src.paper_replication.advanced_experiments import apply_binning, set_seed
from src.paper_replication.pipeline import PROJECT_ROOT, build_preprocessor, dense_array
from src.paper_replication.v6_case_sweep import (
    ALL_DATASETS, BINNING_KEY, MODELS_DIR, REPORTS_DIR, RESULTS_DIR,
    STUDENT_DATASETS, get_feature_names, make_loader, metric_dict,
    oversample_train, predict_proba, read_dataset, student_features, xapi_features,
)
from src.paper_replication.v18_strict_validation import (
    StrictSplit, deep_engineered_student_features,
)
from sklearn.model_selection import train_test_split

PAPER_BENCHMARKS = {
    "student-mat": {"f1_macro": 0.94},
    "student-por": {"f1_macro": 0.90},
    "xapi": {"f1_macro": 0.8447},
}
V20_DIR = RESULTS_DIR / "v20"
V20_MODELS = MODELS_DIR / "v20"
V20_REPORT = REPORTS_DIR / "v20_report.md"


# === MULTISCALE CNN BLOCK ===
class MultiScaleCNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bn: bool = True) -> None:
        super().__init__()
        b = out_ch // 3
        r = out_ch - b * 3
        self.c3 = self._br(in_ch, b + r, 3, bn)
        self.c5 = self._br(in_ch, b, 5, bn)
        self.c7 = self._br(in_ch, b, 7, bn)
        self.total = b * 3 + r

    def _br(self, ic, oc, k, bn):
        ls = [nn.Conv1d(ic, oc, k, padding=k // 2)]
        if bn: ls.append(nn.BatchNorm1d(oc))
        ls.append(nn.GELU())
        return nn.Sequential(*ls)

    def forward(self, x):
        return torch.cat([self.c3(x), self.c5(x), self.c7(x)], dim=1)


# === MULTI-HEAD SELF-ATTENTION ===
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


# === MULTISCALE CNN-BiLSTM ===
class MultiScaleCNNBiLSTM(nn.Module):
    def __init__(self, dtype: str, n_feat: int, n_cls: int,
                 filters=96, lstm_h=128, lstm_layers=2, dense_h=128,
                 drop=0.3, heads=4, bn=True) -> None:
        super().__init__()
        ms = MultiScaleCNNBlock(1, filters, bn)
        self.ms1 = ms
        self.pool1 = nn.AdaptiveMaxPool1d(8)
        self.drop1 = nn.Dropout(drop)
        c2_out = filters * 2 if dtype == "xapi" else filters
        l2 = [nn.Conv1d(ms.total, c2_out, 3, padding=1)]
        if bn: l2.append(nn.BatchNorm1d(c2_out))
        l2.append(nn.GELU())
        self.conv2 = nn.Sequential(*l2)
        self.pool2 = nn.AdaptiveMaxPool1d(4)
        self.drop2 = nn.Dropout(drop)
        self.lstm = nn.LSTM(c2_out, lstm_h, lstm_layers, batch_first=True,
                            bidirectional=True, dropout=drop if lstm_layers > 1 else 0.)
        self.ldrop = nn.Dropout(drop)
        self.attn = MHSA(lstm_h * 2, heads, drop)
        self.clf = nn.Sequential(
            nn.Linear(lstm_h * 2, dense_h), nn.LayerNorm(dense_h), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dense_h, max(1, dense_h // 2)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(max(1, dense_h // 2), n_cls),
        )

    def forward(self, x):
        o = x.unsqueeze(1)
        o = self.pool1(self.ms1(o)); o = self.drop1(o)
        o = self.pool2(self.conv2(o)); o = self.drop2(o)
        o, _ = self.lstm(o.permute(0, 2, 1))
        o = self.attn(self.ldrop(o)).mean(dim=1)
        return self.clf(o)


# === FOCAL LOSS ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., weight=None, smooth=0.) -> None:
        super().__init__()
        self.gamma, self.weight, self.smooth = gamma, weight, smooth

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             reduction="none", label_smoothing=self.smooth)
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


# === MIXUP ===
def mixup(x, y, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_loss(crit, pred, ya, yb, lam):
    return lam * crit(pred, ya) + (1 - lam) * crit(pred, yb)


# === BOOST CONFIG ===
@dataclass
class BoostConfig:
    filters: int = 96
    lstm_h: int = 128
    lstm_layers: int = 2
    dense_h: int = 128
    drop: float = 0.3
    heads: int = 4
    bn: bool = True
    lr: float = 3e-4
    wd: float = 1e-4
    batch: int = 32
    epochs: int = 150
    patience: int = 30
    smooth: float = 0.05
    focal_g: float = 2.0
    focal: bool = True
    mixup: bool = True
    mixup_a: float = 0.4
    cosine: bool = True
    sampling: str = "class_weight"
    feat: str = "deep_engineered"
    seed: int = 42


# === STRICT SPLIT HELPER ===
def boost_split(dataset: str, raw: pd.DataFrame, feat: str, seed: int) -> StrictSplit:
    if dataset in STUDENT_DATASETS:
        frame, y, fcols = (deep_engineered_student_features(raw, dataset, BINNING_KEY)
                           if feat == "deep_engineered"
                           else student_features(raw, dataset, feat, BINNING_KEY))
        nc = 5
    else:
        frame, y, fcols = xapi_features(raw, feat)
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


# === TRAIN ONE MODEL ===
def train_one(dataset: str, split: StrictSplit, cfg: BoostConfig) -> dict:
    set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr = np.array(split.X_train, np.float32)
    ytr = np.array(split.y_train, np.int64)
    if cfg.sampling not in {"none", "class_weight"}:
        Xtr, ytr, _ = oversample_train(Xtr, ytr, cfg.sampling, cfg.seed)
    dt = {"student-mat": "mat", "student-por": "por", "xapi": "xapi"}[dataset]
    model = MultiScaleCNNBiLSTM(dt, split.n_features, split.n_classes,
                                  cfg.filters, cfg.lstm_h, cfg.lstm_layers,
                                  cfg.dense_h, cfg.drop, cfg.heads, cfg.bn).to(dev)
    wt = None
    if cfg.sampling == "class_weight":
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight("balanced", classes=np.arange(split.n_classes), y=split.y_train)
        wt = torch.tensor(cw, dtype=torch.float32, device=dev)
    crit = FocalLoss(cfg.focal_g, wt, cfg.smooth) if cfg.focal else nn.CrossEntropyLoss(weight=wt, label_smoothing=cfg.smooth)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, cfg.epochs // 3), T_mult=1, eta_min=1e-6) if cfg.cosine else None
    loader = make_loader(Xtr, ytr, cfg.batch, cfg.seed)
    bst, bvf1, bvacc, bep, wait = None, -1., -1., 0, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
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
        vp = predict_proba(model, split.X_val, cfg.batch, dev)
        vm = metric_dict(split.y_val, vp.argmax(1))
        vf1, vacc = vm["f1_macro"], vm["accuracy"]
        if vf1 > bvf1 + 1e-6 or (abs(vf1 - bvf1) <= 1e-6 and vacc > bvacc + 1e-6):
            bvf1, bvacc, bep, wait = vf1, vacc, ep, 0
            bst = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience: break
    model.load_state_dict(bst or {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    tp = predict_proba(model, split.X_test, cfg.batch, dev)
    tm = metric_dict(split.y_test, tp.argmax(1))
    return {"state_dict": bst, "val_f1_macro": bvf1, "val_accuracy": bvacc,
            "best_epoch": bep, "elapsed": time.time() - t0,
            "test_probs": tp, **{f"test_{k}": v for k, v in tm.items()}}


# === OPTUNA SEARCH ===
def run_optuna(dataset: str, n_trials=100, seed=42, ep_per_trial=100) -> dict:
    import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
    raw = read_dataset(dataset)
    logs = []
    def obj(trial):
        fs = trial.suggest_categorical("feat", (
            ["paper", "deep_engineered", "paper_engineered"] if dataset in STUDENT_DATASETS
            else ["paper", "behavior8", "full"]))
        cfg = BoostConfig(
            filters=trial.suggest_categorical("filters", [64, 96, 128, 192]),
            lstm_h=trial.suggest_categorical("lstm_h", [64, 96, 128, 192, 256]),
            lstm_layers=trial.suggest_categorical("lstm_layers", [1, 2]),
            dense_h=trial.suggest_categorical("dense_h", [64, 96, 128, 192, 256]),
            drop=trial.suggest_float("drop", 0.1, 0.5, step=0.05),
            heads=trial.suggest_categorical("heads", [2, 4]),
            bn=trial.suggest_categorical("bn", [True, False]),
            lr=trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            wd=trial.suggest_float("wd", 1e-6, 1e-2, log=True),
            batch=trial.suggest_categorical("batch", [16, 32, 64]),
            epochs=ep_per_trial, patience=trial.suggest_categorical("patience", [15, 20, 30, 40]),
            smooth=trial.suggest_float("smooth", 0., 0.15, step=0.05),
            focal=trial.suggest_categorical("focal", [True, False]),
            focal_g=trial.suggest_float("focal_g", 1., 3., step=0.5),
            mixup=trial.suggest_categorical("mixup", [True, False]),
            mixup_a=trial.suggest_float("mixup_a", 0.2, 0.6, step=0.1),
            cosine=trial.suggest_categorical("cosine", [True, False]),
            sampling=trial.suggest_categorical("sampling", ["class_weight", "smote", "none"]),
            feat=fs, seed=seed,
        )
        sp = boost_split(dataset, raw, fs, seed)
        r = train_one(dataset, sp, cfg)
        vf1 = r["val_f1_macro"]
        logs.append({"trial": trial.number, "val_f1": vf1, "test_f1": r["test_f1_macro"],
                     "epoch": r["best_epoch"], **trial.params})
        print(f"  T{trial.number:3d}[{dataset}] val={vf1:.4f} test={r['test_f1_macro']:.4f}"
              f" ep={r['best_epoch']} fs={fs} mix={cfg.mixup} focal={cfg.focal}", flush=True)
        return float(vf1)
    study = optuna.create_study(direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(10, n_trials // 5)))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params; bv = study.best_value
    print(f"\n[{dataset}] Best val_f1={bv:.4f} trial={study.best_trial.number}")
    print(f"  {json.dumps(bp, indent=2)}")
    return {"dataset": dataset, "best_val_f1": bv, "best_trial": study.best_trial.number,
            "best_params": bp, "trial_log": logs}


# === LARGE ENSEMBLE ===
def build_ensemble(dataset: str, raw: pd.DataFrame, params: dict,
                   n: int = 7, seeds=None, test_seed=42) -> dict:
    if seeds is None: seeds = [42, 123, 314, 456, 789, 999, 2024]
    seeds = seeds[:n]
    fs = params.get("feat", "deep_engineered" if dataset in STUDENT_DATASETS else "behavior8")
    sp0 = boost_split(dataset, raw, fs, test_seed)  # Fixed test set
    probs_list, vf1s, details = [], [], []
    print(f"\n[{dataset}] Building {n}-member ensemble (fixed test_seed={test_seed})...", flush=True)
    for i, s in enumerate(seeds):
        cfg = BoostConfig(
            filters=int(params.get("filters", 96)),
            lstm_h=int(params.get("lstm_h", 128)),
            lstm_layers=int(params.get("lstm_layers", 2)),
            dense_h=int(params.get("dense_h", 128)),
            drop=float(params.get("drop", 0.3)),
            heads=int(params.get("heads", 4)),
            bn=bool(params.get("bn", True)),
            lr=float(params.get("lr", 3e-4)),
            wd=float(params.get("wd", 1e-4)),
            batch=int(params.get("batch", 32)),
            epochs=200, patience=int(params.get("patience", 40)),
            smooth=float(params.get("smooth", 0.05)),
            focal_g=float(params.get("focal_g", 2.0)),
            focal=bool(params.get("focal", True)),
            mixup=bool(params.get("mixup", True)),
            mixup_a=float(params.get("mixup_a", 0.4)),
            cosine=bool(params.get("cosine", True)),
            sampling=str(params.get("sampling", "class_weight")),
            feat=fs, seed=s,
        )
        r = train_one(dataset, sp0, cfg)
        probs_list.append(r["test_probs"])
        vf1s.append(r["val_f1_macro"])
        details.append({"seed": s, "val_f1": r["val_f1_macro"], "val_acc": r["val_accuracy"],
                        "test_f1": r["test_f1_macro"], "test_acc": r["test_accuracy"], "ep": r["best_epoch"]})
        print(f"  M{i+1}/{n} seed={s}: val={r['val_f1_macro']:.4f} test={r['test_f1_macro']:.4f}"
              f" ep={r['best_epoch']}", flush=True)
    # Uniform ensemble
    up = np.mean(probs_list, axis=0)
    um = metric_dict(sp0.y_test, up.argmax(1))
    # Weighted ensemble (val F1 softmax)
    wts = np.exp(np.array(vf1s) * 5); wts /= wts.sum()
    wp = sum(p * w for p, w in zip(probs_list, wts))
    wm = metric_dict(sp0.y_test, wp.argmax(1))
    pf1 = PAPER_BENCHMARKS[dataset]["f1_macro"]
    print(f"\n[{dataset}] Uniform F1={um['f1_macro']:.4f}  Weighted F1={wm['f1_macro']:.4f}"
          f"  Paper={pf1:.4f}  Gap={max(um['f1_macro'],wm['f1_macro'])-pf1:+.4f}", flush=True)
    return {"dataset": dataset, "n_members": n, "seeds": list(seeds), "test_seed": test_seed,
            "feature_set": fs, "member_details": details, "weights": wts.tolist(),
            "uniform_metrics": um, "weighted_metrics": wm, "paper_f1": pf1}


# === REPORT ===
def write_report(results: dict) -> None:
    lines = ["# V20 Advanced Boost Report", "",
             "Strict validation 70/15/15 | MultiScale CNN + BiLSTM + MHSA | Focal Loss + Mixup", "",
             "| Dataset | Uniform F1 | Weighted F1 | Paper F1 | Gap |",
             "| --- | --- | --- | --- | --- |"]
    for ds, r in results.items():
        uf1 = r["uniform_metrics"]["f1_macro"]; wf1 = r["weighted_metrics"]["f1_macro"]
        pf1 = r["paper_f1"]; bf1 = max(uf1, wf1)
        lines.append(f"| {ds} | {uf1:.4f} | {wf1:.4f} | {pf1:.4f} | {bf1-pf1:+.4f} |")
    lines += ["", "## Member Details", ""]
    for ds, r in results.items():
        lines += [f"### {ds}", "| Seed | Val F1 | Test F1 | Epoch |", "| --- | --- | --- | --- |"]
        for m in r["member_details"]:
            lines.append(f"| {m['seed']} | {m['val_f1']:.4f} | {m['test_f1']:.4f} | {m['ep']} |")
        lines.append("")
    V20_REPORT.parent.mkdir(parents=True, exist_ok=True)
    V20_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report: {V20_REPORT}")


# === MAIN ===
def main():
    import argparse
    V20_DIR.mkdir(parents=True, exist_ok=True)
    V20_MODELS.mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["quick_test", "optuna", "ensemble", "full"])
    p.add_argument("--datasets", default="student-mat,student-por,xapi")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--members", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [42, 123, 314, 456, 789, 999, 2024]

    if args.command == "quick_test":
        print("=== V20 Quick Test ===", flush=True)
        for ds in datasets:
            raw = read_dataset(ds)
            fs = "deep_engineered" if ds in STUDENT_DATASETS else "behavior8"
            sp = boost_split(ds, raw, fs, 42)
            cfg = BoostConfig(epochs=80, patience=20, feat=fs)
            r = train_one(ds, sp, cfg)
            print(f"[{ds}] feat={sp.n_features} cls={sp.n_classes} "
                  f"val_f1={r['val_f1_macro']:.4f} test_f1={r['test_f1_macro']:.4f} ep={r['best_epoch']}", flush=True)

    elif args.command == "optuna":
        print(f"=== V20 Optuna: {datasets} trials={args.trials} ===", flush=True)
        for ds in datasets:
            r = run_optuna(ds, args.trials, args.seed, args.epochs)
            out = V20_DIR / f"optuna_{ds}.json"
            out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
            print(f"Saved: {out}")

    elif args.command == "ensemble":
        print(f"=== V20 Ensemble: {datasets} members={args.members} ===", flush=True)
        all_res = {}
        for ds in datasets:
            opt = V20_DIR / f"optuna_{ds}.json"
            if opt.exists():
                bp = json.loads(opt.read_text())["best_params"]
                print(f"[{ds}] Loaded Optuna best params")
            else:
                bp = {"filters": 96, "lstm_h": 128, "lstm_layers": 2, "dense_h": 128,
                      "drop": 0.25, "heads": 4, "bn": True, "lr": 3e-4, "wd": 1e-4,
                      "batch": 32, "patience": 40, "smooth": 0.05, "focal_g": 2.0,
                      "focal": True, "mixup": True, "mixup_a": 0.4, "cosine": True,
                      "sampling": "class_weight",
                      "feat": "deep_engineered" if ds in STUDENT_DATASETS else "behavior8"}
                print(f"[{ds}] Using default boost params")
            raw = read_dataset(ds)
            r = build_ensemble(ds, raw, bp, args.members, seeds, test_seed=42)
            all_res[ds] = r
            out = V20_DIR / f"ensemble_{ds}.json"
            out.write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
            print(f"Saved: {out}")
        write_report(all_res)

    elif args.command == "full":
        print(f"=== V20 Full Pipeline: {datasets} ===", flush=True)
        for ds in datasets:
            print(f"\n--- Optuna: {ds} ---", flush=True)
            r = run_optuna(ds, args.trials, args.seed, args.epochs)
            (V20_DIR / f"optuna_{ds}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        all_res = {}
        for ds in datasets:
            print(f"\n--- Ensemble: {ds} ---", flush=True)
            bp = json.loads((V20_DIR / f"optuna_{ds}.json").read_text())["best_params"]
            raw = read_dataset(ds)
            r = build_ensemble(ds, raw, bp, args.members, seeds, test_seed=42)
            all_res[ds] = r
            (V20_DIR / f"ensemble_{ds}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        write_report(all_res)
        print("\n=== V20 FINAL SUMMARY ===")
        for ds, r in all_res.items():
            bf1 = max(r["uniform_metrics"]["f1_macro"], r["weighted_metrics"]["f1_macro"])
            print(f"  {ds}: Best F1={bf1:.4f}  Paper={r['paper_f1']:.4f}  Gap={bf1-r['paper_f1']:+.4f}")


if __name__ == "__main__":
    main()
