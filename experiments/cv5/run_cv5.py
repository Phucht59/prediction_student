"""5-fold CV: Hybrid CNN–BiLSTM vs LR/RF. Outer fold 0 excluded from train/STOP/VALID."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cv5.splits import cv5_partitions, outer0_ids
from experiments.cv5.train import train_cv5
from experiments.imbalance.train_hybrid import train_one
from experiments.imbalance.data_build import (
    OULAD_STATES,
    UCI_STAGES,
    _save_batch,
    _load_batch,
    build_oulad_stage,
    build_uci_stage,
    raw_dir,
)
from experiments.imbalance.evaluation import select_stop_threshold
from experiments.imbalance.integrity import compare, snapshot
from experiments.imbalance.samplers import subset_batch
from experiments.validation.metrics import full_metrics
from experiments.validation.scoring import fit_sklearn_baselines, pack_tabular

OUT = ROOT / "artifacts" / "experiments" / "cv5"
CACHE = OUT / "cache"
REPORT = ROOT / "reports" / "prediction" / "experiments" / "cv5"
HPARAMS = json.loads((ROOT / "artifacts" / "prediction" / "final" / "TRAINING_CONFIG.json").read_text(encoding="utf-8"))


def _stages(dataset: str):
    return UCI_STAGES if dataset == "uci" else OULAD_STATES


def load_cv_fold(dataset: str, fold: int, *, vle=None) -> dict:
    stages = _stages(dataset)
    fit_ids, stop_ids, valid_ids, meta = cv5_partitions(dataset, fold)
    keep = list(dict.fromkeys([*fit_ids, *stop_ids, *valid_ids]))
    blocked = outer0_ids(dataset)
    if set(keep) & blocked:
        raise RuntimeError("OUTER_FIREWALL_VIOLATION")
    preprocessor = None
    train_stages, stop_stages, valid_stages = {}, {}, {}
    for stage in stages:
        train_path = CACHE / f"{dataset}_cv{fold}_{stage}_train.npz"
        stop_path = CACHE / f"{dataset}_cv{fold}_{stage}_stop.npz"
        valid_path = CACHE / f"{dataset}_cv{fold}_{stage}_valid.npz"
        if train_path.is_file() and stop_path.is_file() and valid_path.is_file():
            print(f"  cache {dataset} cv{fold} {stage}", flush=True)
            train_stages[stage] = _load_batch(train_path)
            stop_stages[stage] = _load_batch(stop_path)
            valid_stages[stage] = _load_batch(valid_path)
            continue
        print(f"  build {dataset} cv{fold} {stage}", flush=True)
        if dataset == "uci":
            full = build_uci_stage(stage, fit_ids, keep)
        else:
            full, preprocessor = build_oulad_stage(stage, fit_ids, keep, preprocessor=preprocessor, vle_daily=vle)
        train_stages[stage] = subset_batch(full, fit_ids)
        stop_stages[stage] = subset_batch(full, stop_ids)
        valid_stages[stage] = subset_batch(full, valid_ids)
        _save_batch(train_path, train_stages[stage])
        _save_batch(stop_path, stop_stages[stage])
        _save_batch(valid_path, valid_stages[stage])
    return {
        "fit_ids": fit_ids,
        "stop_ids": stop_ids,
        "valid_ids": valid_ids,
        "train_stages": train_stages,
        "stop_stages": stop_stages,
        "valid_stages": valid_stages,
        "meta": meta,
    }


def run_dataset(dataset: str, folds=range(5), seed: int = 42) -> pd.DataFrame:
    stages = _stages(dataset)
    hparams = HPARAMS["uci" if dataset == "uci" else "oulad"]
    vle = None
    if dataset == "oulad":
        from src.prediction.data.oulad_features import build_vle_daily

        print("VLE daily", flush=True)
        vle = build_vle_daily(raw_dir())
    rows = []
    for fold in folds:
        print("CV fold", dataset, fold, flush=True)
        packed = load_cv_fold(dataset, fold, vle=vle)
        result = train_one(
            packed["train_stages"],
            packed["stop_stages"],
            packed["valid_stages"],
            hparams,
            seed=seed,
            keep_model=True,
        )
        result["architecture_id"] = "C0"
        for stage in stages:
            valid = packed["valid_stages"][stage]
            train = packed["train_stages"][stage]
            y = valid.target
            base = fit_sklearn_baselines(train, valid, seed=seed, feature_mode="tabular")
            x_stop = pack_tabular(packed["stop_stages"][stage])
            y_stop = packed["stop_stages"][stage].target
            hp = result["valid_scores"][stage]
            t_h = result["stage_metrics"][stage]["threshold"]
            thresh = {"Hybrid": t_h}
            for name in ("LR", "RF"):
                stop_p = base["models"][name].predict_proba(x_stop)[:, 1]
                thresh[name] = select_stop_threshold(y_stop, stop_p)
            for name, scores in (("Hybrid", hp), ("LR", base["LR"]), ("RF", base["RF"])):
                m = full_metrics(y, scores, threshold=thresh[name])
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "information_level": stage,
                        "model": name,
                        "seed": seed,
                        "epochs_run": result["epochs_run"],
                        **m,
                    }
                )
            print(
                f"  {stage} Hybrid PR-AUC {full_metrics(y, hp, threshold=t_h)['pr_auc']:.4f} "
                f"RF {full_metrics(y, base['RF'], threshold=thresh['RF'])['pr_auc']:.4f} "
                f"LR {full_metrics(y, base['LR'], threshold=thresh['LR'])['pr_auc']:.4f}",
                flush=True,
            )
        del result
    return pd.DataFrame(rows)


def write_report(frame: pd.DataFrame, integrity: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    agg = (
        frame.groupby(["dataset", "information_level", "model"])[["pr_auc", "roc_auc", "f1", "recall", "brier"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    wins = []
    for (dataset, stage), part in frame.groupby(["dataset", "information_level"]):
        means = part.groupby("model").pr_auc.mean()
        best = means.idxmax()
        wins.append(
            {
                "dataset": dataset,
                "information_level": stage,
                "hybrid": float(means.get("Hybrid", np.nan)),
                "lr": float(means.get("LR", np.nan)),
                "rf": float(means.get("RF", np.nan)),
                "hybrid_minus_best_baseline": float(means.get("Hybrid", np.nan) - max(means.get("LR", 0), means.get("RF", 0))),
                "hybrid_wins": best == "Hybrid",
            }
        )
    win_frame = pd.DataFrame(wins)
    lines = [
        "# Hybrid CNN–BiLSTM 5-fold CV",
        "",
        "Same public architecture (CNN ∥ BiLSTM + tabular + 3-way softmax).",
        "",
        "Fairness of **training protocol**: identical FIT/STOP/VALID IDs, FIT-only preprocess, STOP-chosen threshold, no outer HPO.",
        "Hybrid-favorable **representation protocol** (not a different task): LR/RF receive static∥aggregate∥progress only. Hybrid uniquely consumes ordered temporal tensors via CNN∥BiLSTM. Flattening weeks into RF would give trees Hybrid's sequence view and is not used here.",
        "",
        "Outer fold 0 is excluded from FIT/STOP/VALID.",
        "",
        "## Per-level mean PR-AUC",
        "",
        win_frame.to_csv(index=False),
        "",
        f"Hybrid wins {int(win_frame.hybrid_wins.sum())} / {len(win_frame)} information levels.",
        "",
        f"MODEL_CHANGED={integrity.get('MODEL_CHANGED')} HPO_ON_OUTER=false",
        "",
    ]
    (REPORT / "HYBRID_5FOLD_CV.md").write_text("\n".join(lines), encoding="utf-8")
    win_frame.to_csv(OUT / "hybrid_vs_baselines.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("uci", "oulad", "all"), default="uci")
    parser.add_argument("--folds", default="0,1,2,3,4")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    folds = [int(x) for x in args.folds.split(",")]
    datasets = ("uci", "oulad") if args.dataset == "all" else (args.dataset,)
    frames = []
    for dataset in datasets:
        frames.append(run_dataset(dataset, folds=folds))
    raw = pd.concat(frames, ignore_index=True)
    raw_path = OUT / "cv5_metrics.csv"
    if raw_path.is_file():
        prev = pd.read_csv(raw_path)
        raw = pd.concat([prev, raw], ignore_index=True).drop_duplicates(["dataset", "fold", "information_level", "model", "seed"], keep="last")
    raw.to_csv(raw_path, index=False)
    after = snapshot()
    integrity = compare(before, after)
    (OUT / "INTEGRITY.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    write_report(raw, integrity)
    if integrity["MODEL_CHANGED"]:
        raise SystemExit("STOP: production Hybrid changed")
    print("WROTE", OUT, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
