"""CONTROL / SMOTE / ADASYN on frozen Hybrid numerics. Train-only sampling."""
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

from experiments.imbalance.data_build import (
    OULAD_STATES,
    UCI_STAGES,
    load_fold,
    raw_dir,
)
from experiments.imbalance.integrity import compare, snapshot
from experiments.imbalance.samplers import fingerprint, pack_features, resample_train, unpack_features
from experiments.imbalance.train_hybrid import train_one

OUT = ROOT / "artifacts" / "experiments" / "imbalance"
REPORT = ROOT / "reports" / "prediction" / "experiments" / "imbalance"
HPARAMS = json.loads((ROOT / "artifacts" / "prediction" / "final" / "TRAINING_CONFIG.json").read_text(encoding="utf-8"))
FOLDS = (0, 1, 2)
SEEDS = (42, 1201, 2026)
SAMPLERS = ("control", "smote", "adasyn")
UCI_MASK = {
    "S0": np.array([False, False]),
    "S1": np.array([True, False]),
    "S2": np.array([True, True]),
}


def _stage_mask(dataset: str, stage: str):
    if dataset == "uci":
        return UCI_MASK[stage]
    return None


def _resample_batch(batch, sampler: str, seed: int, dataset: str, stage: str):
    packed = pack_features(batch)
    x_new, y_new, audit = resample_train(packed, batch.target, sampler, random_state=seed)
    out = unpack_features(
        x_new,
        static_dim=batch.static.shape[1],
        aggregate_dim=batch.aggregate.shape[1],
        timesteps=batch.temporal.shape[1],
        temporal_dim=batch.temporal.shape[2],
        target=y_new,
        stage_mask_template=_stage_mask(dataset, stage),
    )
    return out, audit


def _done_keys(path: Path) -> set[tuple]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    if frame.empty:
        return set()
    return set(zip(frame.dataset, frame.fold, frame.seed, frame.sampler))


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.is_file():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def run_dataset(dataset: str, *, folds=FOLDS, seeds=SEEDS, samplers=SAMPLERS) -> list[dict]:
    stages = UCI_STAGES if dataset == "uci" else OULAD_STATES
    hparams = HPARAMS["uci" if dataset == "uci" else "oulad"]
    rows = []
    vle = None
    if dataset == "oulad":
        from src.prediction.data.oulad_features import build_vle_daily

        print("building OULAD VLE daily...", flush=True)
        vle = build_vle_daily(raw_dir())
    raw_path = OUT / "results_raw.csv"
    done = _done_keys(raw_path)
    split_meta = None
    for fold in folds:
        print(f"loading {dataset} fold {fold} tensors...", flush=True)
        packed = load_fold(dataset, fold, vle_daily=vle)
        split_meta = packed["split_meta"]
        original_y = packed["train_stages"][stages[0]].target
        stop_fp = {stage: fingerprint(packed["stop_stages"][stage]) for stage in stages}
        valid_fp = {stage: fingerprint(packed["valid_stages"][stage]) for stage in stages}
        for seed in seeds:
            for sampler in samplers:
                key = (dataset, fold, seed, sampler)
                if key in done:
                    print("skip completed", dataset, fold, seed, sampler, flush=True)
                    continue
                train_stages = {}
                audits = {}
                for stage in stages:
                    train = packed["train_stages"][stage]
                    if sampler != "control":
                        train, audits[stage] = _resample_batch(train, sampler, seed, dataset, stage)
                    else:
                        audits[stage] = {
                            "sampler": "control",
                            "n_train_before": int(len(train.target)),
                            "n_train_after": int(len(train.target)),
                            "n_positive_before": int((train.target == 1).sum()),
                            "n_positive_after": int((train.target == 1).sum()),
                            "fit_on": "train_only",
                        }
                    train_stages[stage] = train
                result = train_one(
                    train_stages,
                    packed["stop_stages"],
                    packed["valid_stages"],
                    hparams,
                    seed=seed,
                    original_target=original_y,
                )
                for stage in stages:
                    after_stop = fingerprint(packed["stop_stages"][stage])
                    after_valid = fingerprint(packed["valid_stages"][stage])
                    if after_stop != stop_fp[stage] or after_valid != valid_fp[stage]:
                        raise RuntimeError(f"EVAL_SET_MUTATED:{dataset}:{fold}:{stage}:{sampler}")
                job_rows = []
                for stage, metrics in result["stage_metrics"].items():
                    job_rows.append(
                        {
                            "dataset": dataset,
                            "information_level": stage,
                            "sampler": sampler,
                            "fold": fold,
                            "seed": seed,
                            **metrics,
                            "epochs_run": result["epochs_run"],
                            "n_train_sampled": audits[stage].get("n_train_after"),
                            "n_train_original": audits[stage].get("n_train_before"),
                            "pos_weight": result["pos_weight"],
                        }
                    )
                _append_rows(raw_path, job_rows)
                rows.extend(job_rows)
                (OUT / f"audit_{dataset}_f{fold}_s{seed}_{sampler}.json").write_text(
                    json.dumps(
                        {
                            "sampler_fit_scope": "train_only",
                            "validation_resampled": False,
                            "test_resampled": False,
                            "outer_test_used_for_selection": False,
                            "preprocessor_fit_scope": "train_only",
                            "oulad_cutoff_safe": True if dataset == "oulad" else "n/a",
                            "stop_valid_fingerprint_unchanged": True,
                            "split_meta": split_meta,
                            "stage_audits": audits,
                            "availability_s0": result.get("availability_s0"),
                            "pos_weight_from_original_fit": result.get("pos_weight_from_original_fit"),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(dataset, fold, seed, sampler, "epochs", result["epochs_run"], flush=True)
                if str(result["device"]).startswith("cuda"):
                    import torch

                    torch.cuda.empty_cache()
    if split_meta is not None:
        (OUT / f"split_meta_{dataset}.json").write_text(json.dumps(split_meta, indent=2) + "\n", encoding="utf-8")
    return rows


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "information_level", "sampler"]
    metrics = ["pr_auc", "f1", "precision", "recall", "accuracy", "macro_f1", "balanced_accuracy", "minority_recall"]
    rows = []
    for key, part in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_std"] = float(part[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def deltas(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, stage), part in agg.groupby(["dataset", "information_level"]):
        by = {row.sampler: row for row in part.itertuples()}
        if "control" not in by:
            continue
        control = by["control"]
        rec = {"dataset": dataset, "information_level": stage}
        for sampler in ("smote", "adasyn"):
            if sampler not in by:
                continue
            other = by[sampler]
            rec[f"{sampler}_delta_pr_auc"] = float(other.pr_auc_mean - control.pr_auc_mean)
            rec[f"{sampler}_delta_macro_f1"] = float(other.macro_f1_mean - control.macro_f1_mean)
            rec[f"{sampler}_delta_minority_recall"] = float(other.minority_recall_mean - control.minority_recall_mean)
        rows.append(rec)
    return pd.DataFrame(rows)


def write_report(raw: pd.DataFrame, agg: pd.DataFrame, delta: pd.DataFrame, integrity: dict, leakage: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Imbalance experiment: SMOTE / ADASYN on frozen Hybrid CNN–BiLSTM",
        "",
        "## 1. Objective",
        "",
        "Under the exact frozen Hybrid configuration, does train-only SMOTE or ADASYN improve VALID metrics versus CONTROL?",
        "",
        "## 2. Why imbalance handling is required",
        "",
        "The thesis requires minority-class synthetic sampling (SMOTE, ADASYN) as an investigated method, not as a replacement of the locked Hybrid.",
        "",
        "## 3. Experimental design",
        "",
        "Three conditions: CONTROL (original FIT distribution), SMOTE, ADASYN. Same architecture, hyperparameters, folds, seeds, cutoff rule, and FIT-only preprocessing. The only independent variable is the training sampler.",
        "",
        "## 4–6. Conditions",
        "",
        "- CONTROL: no resampling.",
        "- SMOTE: `imblearn.SMOTE` on flattened Hybrid tensors of FIT rows, per information level.",
        "- ADASYN: `imblearn.ADASYN` with the same packing.",
        "",
        "Temporal sequences are **not** synthesized timestep-by-timestep. Student-level flattened vectors (static ∥ aggregate ∥ temporal-flat ∥ progress) are resampled, then unpacked. UCI masks are restored from the information level. OULAD masks are recovered from non-zero temporal support. This is defensible for UCI (T≤2). For OULAD it interpolates week vectors and is a limitation, not a new sequence-SMOTE algorithm.",
        "",
        "## 7. Data leakage prevention",
        "",
        "- Split first (frozen inner FIT/STOP/VALID, outer excluded).",
        "- Preprocessor fit on FIT only.",
        "- Sampler fit on FIT tensors only.",
        "- STOP and VALID never resampled; fingerprints checked after each job.",
        "- OULAD events remain `observation_start <= t < cutoff`.",
        f"- Split source: `{leakage.get('split_source', 'unknown')}`.",
        "",
        "## 8. Dataset / information levels",
        "",
        "UCI S0/S1/S2. OULAD 20/35/50/75/100 as views of one Hybrid, not separate models.",
        "",
        "## 9. Training configuration",
        "",
        "Copied from `artifacts/prediction/final/TRAINING_CONFIG.json` (lr, dropout, weight decay, batch size, pos_weight_multiplier, entropy floor). AdamW, grad clip 1.0, early-stop on STOP macro PR-AUC, max 24 epochs, patience 8. `pos_weight` is computed from original FIT labels and kept after resampling. No HPO.",
        "",
        "## 10. Results",
        "",
        "```",
        agg.to_csv(index=False).rstrip(),
        "```",
        "",
        "## 11. Delta vs CONTROL",
        "",
    ]
    if len(delta):
        lines += ["```", delta.to_csv(index=False).rstrip(), "```", ""]
    else:
        lines.append("_pending_")
        lines.append("")
    lines += ["## 12. Interpretation", ""]
    if len(delta) and "smote_delta_pr_auc" in delta.columns:
        smote_pr = float(delta["smote_delta_pr_auc"].mean())
        adasyn_pr = float(delta["adasyn_delta_pr_auc"].mean()) if "adasyn_delta_pr_auc" in delta.columns else 0.0
        better = []
        if (delta["smote_delta_pr_auc"] > 0).any():
            better.append("SMOTE at some levels")
        if "adasyn_delta_pr_auc" in delta.columns and (delta["adasyn_delta_pr_auc"] > 0).any():
            better.append("ADASYN at some levels")
        lines.append(
            f"Mean PR-AUC delta across reported levels: SMOTE {smote_pr:+.4f}, ADASYN {adasyn_pr:+.4f}. "
            "Positive means the sampler beat CONTROL. Neither sampler is promoted to production."
        )
        if better:
            lines.append("Observed improvement: " + ", ".join(better) + ".")
        else:
            lines.append("No information level showed a PR-AUC gain over CONTROL.")
    else:
        lines.append("Deltas not yet computed.")
    lines += [
        "",
        "## 13. Limitations",
        "",
        "- Flattened SMOTE on OULAD week tensors interpolates sequences; it is not temporally generative.",
        "- CONTROL is re-trained with the frozen numerics in this isolated trainer; small numeric drift vs the published 3×3 mean is possible because this loop trains every information level each epoch rather than production stage-balanced sampling.",
        "- pos_weight from the frozen protocol is kept even after resampling.",
        "- If the kltn Phase-1 parquet bundle is absent, inner_fold is recovered from official reconstructed OOF VALID assignments rather than the original parquet bytes (FIT/STOP still use StratifiedGroupKFold n=5, seed=42).",
        "",
        "## 14. Evidence for SMOTE/ADASYN usefulness",
        "",
        "See deltas. No sampler is selected as the new production model.",
        "",
        "## 15. Production Hybrid was NOT changed",
        "",
        f"- MODEL_CHANGED = {integrity.get('MODEL_CHANGED')}",
        f"- HPO_PERFORMED = {integrity.get('HPO_PERFORMED')}",
        f"- OUTER_OPENED = {integrity.get('OUTER_OPENED')}",
        f"- RECOMMENDATION_CHANGED = {integrity.get('RECOMMENDATION_CHANGED')}",
        f"- changed_files = {integrity.get('changed_files')}",
        "",
    ]
    (REPORT / "IMBALANCE_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize(integrity: dict) -> None:
    raw_path = OUT / "results_raw.csv"
    if not raw_path.is_file():
        raise SystemExit("no results_raw.csv")
    raw = pd.read_csv(raw_path)
    agg = aggregate(raw)
    agg.to_csv(OUT / "results_aggregate.csv", index=False)
    delta = deltas(agg)
    delta.to_csv(OUT / "results_delta.csv", index=False)
    split_source = "unknown"
    for name in ("split_meta_uci.json", "split_meta_oulad.json"):
        path = OUT / name
        if path.is_file():
            split_source = json.loads(path.read_text(encoding="utf-8")).get("split_source", split_source)
    leakage = {
        "sampler_fit_scope": "train_only",
        "validation_resampled": False,
        "test_resampled": False,
        "outer_test_used_for_selection": False,
        "preprocessor_fit_scope": "train_only",
        "oulad_cutoff_safe": True,
        "split_source": split_source,
    }
    (OUT / "LEAKAGE_AUDIT.json").write_text(json.dumps(leakage, indent=2) + "\n", encoding="utf-8")
    integrity["PREDICTION_AUTHORITY"] = "Hybrid CNN-BiLSTM (frozen production)"
    (OUT / "INTEGRITY_AUDIT.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    write_report(raw, agg, delta, integrity, leakage)
    if integrity["MODEL_CHANGED"] or integrity["RECOMMENDATION_CHANGED"]:
        raise SystemExit("STOP: production files changed")
    print("WROTE", OUT, REPORT)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("uci", "oulad", "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="one fold, one seed (smoke)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    folds = (0,) if args.quick else FOLDS
    seeds = (42,) if args.quick else SEEDS
    datasets = ("uci", "oulad") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_dataset(dataset, folds=folds, seeds=seeds)
    after = snapshot()
    integrity = compare(before, after)
    _finalize(integrity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
