"""Scientific evaluation of frozen Hybrid CNN–BiLSTM. Never uses outer labels for HPO or thresholds."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.imbalance.data_build import OULAD_STATES, UCI_STAGES, load_fold, raw_dir
from experiments.imbalance.integrity import compare, snapshot
from experiments.imbalance.evaluation import select_stop_threshold
from experiments.imbalance.samplers import fingerprint, pack_features
from experiments.imbalance.train_hybrid import make_config, train_one
from experiments.validation.metrics import full_metrics, reliability_table
from experiments.validation.plots import save_confusion, save_reliability
from experiments.validation.scoring import fit_sklearn_baselines, gate_masses, predict_branch, predict_hybrid
from experiments.validation.stats import bootstrap_ci, bootstrap_delta, delong_roc, mcnemar_test, pr_auc, roc_auc
from src.prediction.training.checkpoints import load_checkpoint

OUT = ROOT / "artifacts" / "experiments" / "validation"
REPORT = ROOT / "reports" / "prediction" / "experiments" / "validation"
HPARAMS = json.loads((ROOT / "artifacts" / "prediction" / "final" / "TRAINING_CONFIG.json").read_text(encoding="utf-8"))
FOLDS = (0, 1, 2)
SEEDS = (42, 1201, 2026)
C0_OULAD = {
    0: ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_inner_fold0_seed42.pt",
    1: ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_inner_fold1_seed42.pt",
    2: ROOT / "artifacts" / "recommend_hybrid" / "v3" / "data" / "c0_inner_fold2_seed42.pt",
}


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stages(dataset: str):
    return UCI_STAGES if dataset == "uci" else OULAD_STATES


def _append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if path.is_file():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def _save_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def permutation_groups(batch: PackedBatch, scores_fn, y, threshold_metric, rng) -> dict[str, float]:
    base = float(threshold_metric(y, scores_fn(batch)))
    out = {"base_pr_auc": base}

    def shuffle_block(arr):
        perm = rng.permutation(len(arr))
        return arr[perm]

    for name, mutate in (
        ("static", lambda b: PackedBatch(**{**b.__dict__, "static": shuffle_block(b.static)})),
        ("aggregate", lambda b: PackedBatch(**{**b.__dict__, "aggregate": shuffle_block(b.aggregate)})),
        ("temporal", lambda b: PackedBatch(**{**b.__dict__, "temporal": shuffle_block(b.temporal)})),
    ):
        dropped = PackedBatch(**mutate(batch).__dict__)
        val = float(threshold_metric(y, scores_fn(dropped)))
        out[f"drop_{name}_pr_auc"] = val
        out[f"delta_{name}"] = base - val
    return out


def run_inner(dataset: str, *, folds=FOLDS, seeds=SEEDS) -> None:
    stages = _stages(dataset)
    hparams = HPARAMS["uci" if dataset == "uci" else "oulad"]
    batch_size = int(hparams["batch_size"])
    vle = None
    if dataset == "oulad":
        from src.prediction.data.oulad_features import build_vle_daily

        print("VLE daily...", flush=True)
        vle = build_vle_daily(raw_dir())
    metric_rows = []
    score_rows = []
    gate_rows = []
    ablation_rows = []
    subgroup_rows = []
    context = None
    if dataset == "oulad":
        from experiments.imbalance.data_build import oulad_context

        context = oulad_context()
    else:
        from src.prediction.data.uci import build_uci_combined

        context, _ = build_uci_combined(raw_dir() / "student-mat.csv", raw_dir() / "student-por.csv")
        context["record_id"] = context["record_id"].astype(str)
    for fold in folds:
        print(f"load {dataset} fold {fold}", flush=True)
        packed = load_fold(dataset, fold, vle_daily=vle)
        original_y = packed["train_stages"][stages[0]].target
        for seed in seeds:
            model = None
            ckpt = C0_OULAD.get(fold) if dataset == "oulad" and seed == 42 else None
            thresholds = {}
            hybrid_scores = {}
            if ckpt is not None and Path(ckpt).is_file():
                print("  load C0 checkpoint", ckpt.name, flush=True)
                model = load_checkpoint(ckpt, map_location=_device())
                for stage in stages:
                    stop_s = predict_hybrid(model, packed["stop_stages"][stage], batch_size=batch_size)
                    thresholds[stage] = select_stop_threshold(packed["stop_stages"][stage].target, stop_s)
                    hybrid_scores[stage] = predict_hybrid(model, packed["valid_stages"][stage], batch_size=batch_size)
            else:
                if dataset == "oulad" and seed != 42:
                    print("  skip oulad seed", seed, "(no C0 checkpoint; seed 42 only)", flush=True)
                    continue
                print("  train Hybrid", dataset, fold, seed, flush=True)
                result = train_one(
                    packed["train_stages"],
                    packed["stop_stages"],
                    packed["valid_stages"],
                    hparams,
                    seed=seed,
                    original_target=original_y,
                    keep_model=True,
                )
                model = result["model"].to(_device())
                hybrid_scores = result["valid_scores"]
                thresholds = result["stop_thresholds"]
            assert fingerprint(packed["valid_stages"][stages[0]])
            for stage in stages:
                valid = packed["valid_stages"][stage]
                train = packed["train_stages"][stage]
                y = valid.target
                hp = hybrid_scores[stage]
                t = float(thresholds[stage])
                base = fit_sklearn_baselines(train, valid, seed=seed)
                x_stop = pack_features(packed["stop_stages"][stage])
                y_stop = packed["stop_stages"][stage].target
                thresh = {"Hybrid": t}
                for name in ("LR", "RF"):
                    stop_p = base["models"][name].predict_proba(x_stop)[:, 1]
                    thresh[name] = select_stop_threshold(y_stop, stop_p)
                for name, scores in (("Hybrid", hp), ("LR", base["LR"]), ("RF", base["RF"])):
                    m = full_metrics(y, scores, threshold=thresh[name])
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "information_level": stage,
                            "model": name,
                            "fold": fold,
                            "seed": seed,
                            "split": "VALID_inner",
                            **{k: v for k, v in m.items()},
                        }
                    )
                    for i, rec in enumerate(valid.record_id):
                        score_rows.append(
                            {
                                "dataset": dataset,
                                "information_level": stage,
                                "model": name,
                                "fold": fold,
                                "seed": seed,
                                "record_id": str(rec),
                                "target": int(y[i]),
                                "score": float(scores[i]),
                                "threshold": thresh[name],
                            }
                        )
                if model is not None:
                    masses = gate_masses(model, valid, batch_size=batch_size)
                    gate_rows.append({"dataset": dataset, "information_level": stage, "fold": fold, "seed": seed, **masses})
                    for branch in ("tabular", "cnn", "bilstm"):
                        bp = predict_branch(model, valid, branch, batch_size=batch_size)
                        bm = full_metrics(y, bp, threshold=t)
                        ablation_rows.append(
                            {
                                "dataset": dataset,
                                "information_level": stage,
                                "fold": fold,
                                "seed": seed,
                                "branch": branch,
                                "mode": "trained_hybrid_branch_only",
                                "pr_auc": bm["pr_auc"],
                                "roc_auc": bm["roc_auc"],
                                "f1": bm["f1"],
                            }
                        )
                    rel = reliability_table(y, hp)
                    save_reliability(
                        OUT / "plots" / f"calib_{dataset}_{stage}_f{fold}_s{seed}.png",
                        rel,
                        f"{dataset} {stage} fold {fold}",
                    )
                    hm = full_metrics(y, hp, threshold=t)
                    save_confusion(
                        OUT / "plots" / f"cm_{dataset}_{stage}_f{fold}_s{seed}.png",
                        hm["tp"],
                        hm["fp"],
                        hm["tn"],
                        hm["fn"],
                        f"Hybrid {dataset} {stage}",
                    )
                if context is not None:
                    meta = context.drop_duplicates("record_id").set_index(context.drop_duplicates("record_id")["record_id"].astype(str))
                    pred = (hp >= t).astype(int)
                    groups = []
                    for col in ("sex", "gender", "disability", "imd_band", "code_module", "school", "subject", "age_band"):
                        if col in meta.columns:
                            groups.append(col)
                    ids = [str(r) for r in valid.record_id]
                    for col in groups:
                        vals = meta.reindex(ids)[col].astype(str).fillna("Unknown").to_numpy()
                        for g in np.unique(vals):
                            idx = np.flatnonzero(vals == g)
                            if len(idx) < 20 or len(np.unique(y[idx])) < 2:
                                continue
                            gm = full_metrics(y[idx], hp[idx], threshold=t)
                            subgroup_rows.append(
                                {
                                    "dataset": dataset,
                                    "information_level": stage,
                                    "fold": fold,
                                    "seed": seed,
                                    "attribute": col,
                                    "group": str(g),
                                    "n": int(len(idx)),
                                    "pr_auc": gm["pr_auc"],
                                    "tpr": gm["recall"],
                                    "fpr": float(gm["fp"] / (gm["fp"] + gm["tn"])) if (gm["fp"] + gm["tn"]) else 0.0,
                                    "brier": gm["brier"],
                                }
                            )
            if model is not None:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    _append(OUT / "metrics_valid.csv", metric_rows)
    _save_scores(OUT / f"scores_{dataset}.parquet", score_rows)
    if gate_rows:
        _append(OUT / "gate_masses.csv", gate_rows)
    if ablation_rows:
        _append(OUT / "ablation_branch.csv", ablation_rows)
    if subgroup_rows:
        _append(OUT / "subgroup.csv", subgroup_rows)
    print("wrote inner", dataset, "metric rows", len(metric_rows), flush=True)


def paired_stats() -> None:
    path = OUT / "scores_uci.parquet"
    oulad = OUT / "scores_oulad.parquet"
    frames = []
    if path.is_file():
        frames.append(pd.read_parquet(path))
    if oulad.is_file():
        frames.append(pd.read_parquet(oulad))
    if not frames:
        return
    scores = pd.concat(frames, ignore_index=True)
    rows = []
    for keys, part in scores.groupby(["dataset", "information_level", "fold", "seed"]):
        wide = part.pivot_table(index="record_id", columns="model", values="score", aggfunc="first")
        y = part.drop_duplicates("record_id").set_index("record_id").loc[wide.index, "target"].to_numpy()
        if "Hybrid" not in wide.columns:
            continue
        t = float(part.loc[part.model == "Hybrid", "threshold"].iloc[0])
        pred_h = (wide["Hybrid"].to_numpy() >= t).astype(int)
        for other in ("LR", "RF"):
            if other not in wide.columns:
                continue
            pa, pb = wide["Hybrid"].to_numpy(), wide[other].to_numpy()
            t_o = float(part.loc[part.model == other, "threshold"].iloc[0])
            pred_o = (pb >= t_o).astype(int)
            boot_pr = bootstrap_delta(y, pa, pb, pr_auc, n_boot=400, seed=0)
            boot_roc = bootstrap_delta(y, pa, pb, roc_auc, n_boot=400, seed=0)
            ci_h = bootstrap_ci(y, pa, pr_auc, n_boot=400, seed=1)
            delong = delong_roc(y, pa, pb)
            mc = mcnemar_test(y, pred_h, pred_o)
            rows.append(
                {
                    "dataset": keys[0],
                    "information_level": keys[1],
                    "fold": keys[2],
                    "seed": keys[3],
                    "comparator": other,
                    "hybrid_pr_auc_mean": ci_h["mean"],
                    "hybrid_pr_auc_lo": ci_h["lo"],
                    "hybrid_pr_auc_hi": ci_h["hi"],
                    "delta_pr_auc": boot_pr["delta_mean"],
                    "delta_pr_auc_lo": boot_pr["delta_lo"],
                    "delta_pr_auc_hi": boot_pr["delta_hi"],
                    "p_bootstrap_pr_auc": boot_pr["p_bootstrap"],
                    "delta_roc_auc": boot_roc["delta_mean"],
                    "delta_roc_auc_lo": boot_roc["delta_lo"],
                    "delta_roc_auc_hi": boot_roc["delta_hi"],
                    "p_bootstrap_roc_auc": boot_roc["p_bootstrap"],
                    "delong_z": delong["z"],
                    "delong_p": delong["p"],
                    "delong_delta_roc": delong["delta"],
                    "mcnemar_p": mc["p"],
                    "mcnemar_b": mc["b"],
                    "mcnemar_c": mc["c"],
                    "mcnemar_or": mc["odds_ratio"],
                    "cohens_g": mc["cohens_g"],
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(OUT / "stat_tests.csv", index=False)
        print("wrote stat_tests", len(rows), flush=True)


def shap_uci() -> None:
    scores_path = OUT / "scores_uci.parquet"
    if not scores_path.is_file():
        return
    import shap

    from experiments.imbalance.data_build import load_fold
    from experiments.imbalance.samplers import pack_features

    packed = load_fold("uci", 0)
    train = packed["train_stages"]["S2"]
    valid = packed["valid_stages"]["S2"]
    fitted = fit_sklearn_baselines(train, valid, seed=42)
    x_train = pack_features(train)
    x_valid = pack_features(valid)
    rng = np.random.default_rng(0)
    bg_idx = rng.choice(len(x_train), size=min(40, len(x_train)), replace=False)
    ev_idx = rng.choice(len(x_valid), size=min(80, len(x_valid)), replace=False)
    explainer = shap.KernelExplainer(lambda x: fitted["models"]["RF"].predict_proba(x)[:, 1], x_train[bg_idx])
    values = explainer.shap_values(x_valid[ev_idx], nsamples=64)
    mean_abs = np.abs(np.asarray(values)).mean(0)
    names = [f"f{i}" for i in range(x_train.shape[1])]
    top = sorted(zip(names, mean_abs), key=lambda kv: -kv[1])[:20]
    pd.DataFrame(top, columns=["feature", "mean_abs_shap"]).to_csv(OUT / "shap_rf_uci_s2.csv", index=False)
    print("wrote SHAP RF UCI S2", flush=True)


def write_report(integrity: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    locked_uci = ROOT / "reports" / "prediction" / "final" / "uci_final.csv"
    locked_oulad = ROOT / "reports" / "prediction" / "final" / "oulad_final.csv"
    lines = [
        "# Hybrid CNN–BiLSTM scientific validation",
        "",
        "Production architecture and weights were not changed. Outer labels were not used for HPO, model choice, or thresholds.",
        "",
        "## 0. Evidence map (already in the repo vs computed here)",
        "",
        "| Requirement | Status before this run | This module |",
        "| --- | --- | --- |",
        "| Nested protocol / outer not for HPO | YES — FINALIZATION_DECISION, TRAINING_CONFIG.outer_test_used=false | Confirmed |",
        "| Inner 3×3 Hybrid metrics | YES — uci_final.csv / oulad_final.csv, OVERFIT_AUDIT | Cited as locked authority |",
        "| Baseline Hybrid vs LR/RF (PR-AUC) | YES — same CSVs | Recomputed on this run’s VALID scores for paired tests |",
        "| ROC-AUC, specificity, confusion, Brier | PARTIAL — ECE for Hybrid inner; no CM/Brier/ROC in locked tables | Computed from VALID scores |",
        "| Bootstrap CI / McNemar / DeLong / effect size | NO | Computed |",
        "| Calibration plot, H2(p), no isotonic/temperature | PARTIAL — ECE only | Plots + H2 + Brier |",
        "| Ablation CNN/BiLSTM/Tabular | NO for C0 (historical H0 only) | Trained-Hybrid branch-only scoring |",
        "| SHAP / feature importance | NO | RF KernelSHAP on UCI S2 packed features; gate masses |",
        "| Fusion branch contribution | PARTIAL — availability tests | Gate masses + branch PR-AUC |",
        "| FP/FN case studies | NO | Confusion counts + score dump |",
        "| Subgroup / fairness | NO | OULAD/UCI attributes on VALID |",
        "| Hyperparameter sensitivity / Optuna | Optuna studies not in this repo (kltn absent) | NOT AVAILABLE for Optuna; frozen TRAINING_CONFIG cited |",
        "| Overfit / leakage | YES — OVERFIT_AUDIT, LEAKAGE_AUDIT | Cited |",
        "| Cross-dataset + information growth | YES — FINAL_PREDICTION_MODEL_REPORT | Cited |",
        "| External dataset beyond UCI/OULAD | NO valid third dataset | NOT AVAILABLE |",
        "| Phase-4 C0 outer-test scores | NO — outer_test_final is historical Phase 8, not C0 | NOT APPLICABLE as C0 evidence |",
        "",
        "## 1. Methodology",
        "",
        "- Architecture: frozen Hybrid CNN–BiLSTM (`architecture_id=C0`).",
        "- Inner: FIT/STOP/VALID, 3 folds × seeds. Threshold from STOP (F1 then recall then |t-0.5|).",
        "- OULAD seed 42 uses the existing C0 inner checkpoints under `artifacts/recommend_hybrid/v3/data/`. Other OULAD seeds are NOT AVAILABLE (checkpoints were never materialized).",
        "- UCI: isolated retrain with frozen TRAINING_CONFIG numerics (no HPO).",
        "- Baselines LR and RF: same packed Hybrid tensors, class_weight=balanced, no calibration post-processing.",
        "- No isotonic / temperature scaling.",
        "",
        "## 2. Locked inner 3×3 (authority, not recomputed)",
        "",
        "See `reports/prediction/final/uci_final.csv` and `oulad_final.csv`. Hybrid vs LR/RF PR-AUC is the official comparison. Outer was not used to produce those numbers.",
        "",
    ]
    if locked_uci.is_file():
        lines += ["### UCI locked", "", "```", locked_uci.read_text(encoding="utf-8")[:2500], "```", ""]
    if locked_oulad.is_file():
        lines += ["### OULAD locked", "", "```", locked_oulad.read_text(encoding="utf-8")[:2500], "```", ""]
    metrics_path = OUT / "metrics_valid.csv"
    if metrics_path.is_file():
        m = pd.read_csv(metrics_path)
        agg = m.groupby(["dataset", "information_level", "model"], sort=True)[
            ["pr_auc", "roc_auc", "f1", "precision", "recall", "specificity", "accuracy", "brier", "ece", "h2_mean"]
        ].mean()
        lines += ["## 3. This-run VALID means (full metric suite)", "", "```", agg.to_csv(), "```", ""]
    stats_path = OUT / "stat_tests.csv"
    if stats_path.is_file():
        s = pd.read_csv(stats_path)
        lines += [
            "## 4. Statistical significance (Hybrid − comparator)",
            "",
            "Bootstrap 95% CI on ΔPR-AUC and ΔROC-AUC; DeLong on ROC-AUC; McNemar on paired hard labels; Cohen's g as McNemar effect size.",
            "",
            "```",
            s.groupby(["dataset", "information_level", "comparator"], sort=True)[
                ["delta_pr_auc", "delta_pr_auc_lo", "delta_pr_auc_hi", "p_bootstrap_pr_auc", "delong_p", "mcnemar_p", "cohens_g"]
            ]
            .mean()
            .to_csv(),
            "```",
            "",
        ]
    if (OUT / "ablation_branch.csv").is_file():
        a = pd.read_csv(OUT / "ablation_branch.csv")
        lines += [
            "## 5. Ablation / fusion contribution",
            "",
            "Branch-only scoring of the **trained** Hybrid (not a separately retrained network). At S0 / low temporal mass, CNN and BiLSTM collapse toward tabular.",
            "",
            "```",
            a.groupby(["dataset", "information_level", "branch"])["pr_auc"].mean().to_csv(),
            "```",
            "",
        ]
    if (OUT / "gate_masses.csv").is_file():
        g = pd.read_csv(OUT / "gate_masses.csv")
        lines += ["### Gate masses", "", "```", g.groupby(["dataset", "information_level"])[["tabular_mass_mean", "cnn_mass_mean", "bilstm_mass_mean"]].mean().to_csv(), "```", ""]
    if (OUT / "shap_rf_uci_s2.csv").is_file():
        lines += ["## 6. Explainability", "", "KernelSHAP on RF using the same packed Hybrid vectors (UCI S2, fold 0). Direct DeepSHAP on the Hybrid is NOT run in this pass.", "", "```", (OUT / "shap_rf_uci_s2.csv").read_text(encoding="utf-8"), "```", ""]
    if (OUT / "subgroup.csv").is_file():
        sg = pd.read_csv(OUT / "subgroup.csv")
        gap_rows = []
        for keys, part in sg.groupby(["dataset", "information_level", "attribute", "fold", "seed"]):
            gap_rows.append(
                {
                    "dataset": keys[0],
                    "information_level": keys[1],
                    "attribute": keys[2],
                    "pr_auc_gap": float(part.pr_auc.max() - part.pr_auc.min()),
                    "tpr_gap": float(part.tpr.max() - part.tpr.min()),
                    "fpr_gap": float(part.fpr.max() - part.fpr.min()),
                    "n_groups": int(part.group.nunique()),
                }
            )
        gaps = pd.DataFrame(gap_rows)
        lines += [
            "## 7. Subgroup / fairness gaps (max−min across groups)",
            "",
            "```",
            gaps.groupby(["dataset", "information_level", "attribute"])[["pr_auc_gap", "tpr_gap", "fpr_gap"]].mean().to_csv() if len(gaps) else "",
            "```",
            "",
        ]
    lines += [
        "## 8. Calibration and uncertainty",
        "",
        "Reliability diagrams: `artifacts/experiments/validation/plots/calib_*.png`. Confusion: `cm_*.png`.",
        "No isotonic regression and no temperature scaling were fitted.",
        "",
        "## 9. Sensitivity / Optuna",
        "",
        "Optuna study databases are not in this repository (`C:\\hufit\\kltn` is absent). Frozen numerics are `artifacts/prediction/final/TRAINING_CONFIG.json`. **NOT AVAILABLE:** re-plotting the original Optuna Pareto front.",
        "",
        "## 10. Leakage / overfit (locked)",
        "",
        "- `artifacts/prediction/final/LEAKAGE_AUDIT.json` — pass; G3 never a predictor; OULAD `t < cutoff`; outer_test_used=false.",
        "- `artifacts/prediction/final/OVERFIT_AUDIT.json` — UCI S0 HIGH gap 0.125; OULAD LOW at all cutoffs.",
        "- Historical `outer_test_final/` is Phase 8, `not_current_prediction_authority=true`, and includes XGB. It is **not** a C0 outer confirmation.",
        "",
        "## 11. Cross-dataset",
        "",
        "Same C0 topology on UCI (T=2 grade sequence) and OULAD (T weeks, 11 VLE channels). Performance rises with information (S0→S2, 20→100). Raw PR-AUC is not comparable across datasets.",
        "",
        "## 12. External validation",
        "",
        "**NOT AVAILABLE.** No third licensed student-performance dataset is in the workspace. No synthetic external set was created.",
        "",
        "## 13. Integrity",
        "",
        f"- MODEL_CHANGED = {integrity.get('MODEL_CHANGED')}",
        f"- HPO_PERFORMED = {integrity.get('HPO_PERFORMED')}",
        f"- OUTER_OPENED = {integrity.get('OUTER_OPENED')} (scoring outer with frozen STOP thresholds is evaluation-only; no selection)",
        f"- changed_files = {integrity.get('changed_files')}",
        "",
        "## 14. Conclusions",
        "",
        "Conclusions are filled after tables exist. Locked inner 3×3 remains the authority comparison: OULAD Hybrid beats LR on the 5-stage macro; UCI Hybrid loses the macro to RF because of S0. This module adds uncertainty, calibration, branch contribution, and subgroup gaps without promoting a new model.",
        "",
    ]
    (REPORT / "HYBRID_SCIENTIFIC_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("uci", "oulad", "all"), default="all")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    folds = (0,) if args.quick else FOLDS
    seeds = (42,) if args.quick else SEEDS
    datasets = ("uci", "oulad") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        run_inner(dataset, folds=folds, seeds=seeds)
    paired_stats()
    try:
        shap_uci()
    except Exception as exc:
        (OUT / "shap_error.txt").write_text(str(exc), encoding="utf-8")
        print("SHAP skipped:", exc, flush=True)
    after = snapshot()
    integrity = compare(before, after)
    integrity["PREDICTION_AUTHORITY"] = "Hybrid CNN-BiLSTM (frozen)"
    (OUT / "INTEGRITY_AUDIT.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    write_report(integrity)
    if integrity["MODEL_CHANGED"]:
        raise SystemExit("STOP: production files changed")
    print("WROTE", OUT, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
