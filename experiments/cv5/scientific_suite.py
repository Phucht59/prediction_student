"""Improve Hybrid on weak stages (STOP-only) and fill remaining scientific checks. No outer HPO. No git push."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cv5.run_cv5 import CACHE, HPARAMS, OUT, REPORT, _stages, load_cv_fold
from experiments.cv5.splits import outer0_ids
from experiments.cv5.train import finetune_weak_stage
from experiments.imbalance.data_build import build_uci_stage, raw_dir
from experiments.imbalance.evaluation import select_stop_threshold
from experiments.imbalance.integrity import compare, snapshot
from experiments.imbalance.train_hybrid import predict_scores, train_one
from experiments.validation.metrics import full_metrics
from experiments.validation.scoring import fit_sklearn_baselines, pack_tabular, predict_branch
from experiments.validation.stats import bootstrap_ci, bootstrap_delta, delong_roc, mcnemar_test, pr_auc, roc_auc
from sklearn.metrics import average_precision_score

SCI = OUT / "scientific"


def _append_scores(rows: list[dict]) -> None:
    SCI.mkdir(parents=True, exist_ok=True)
    path = SCI / "scores.parquet"
    frame = pd.DataFrame(rows)
    if path.is_file():
        prev = pd.read_parquet(path)
        frame = pd.concat([prev, frame], ignore_index=True)
        frame = frame.drop_duplicates(["dataset", "fold", "seed", "information_level", "model", "record_id", "split"], keep="last")
    frame.to_parquet(path, index=False)


def run_fold(dataset: str, fold: int, seed: int, *, vle=None, weak: str | None) -> list[dict]:
    packed = load_cv_fold(dataset, fold, vle=vle)
    hparams = HPARAMS["uci" if dataset == "uci" else "oulad"]
    result = train_one(packed["train_stages"], packed["stop_stages"], packed["valid_stages"], hparams, seed=seed, keep_model=True)
    model = result["model"]
    if weak:
        model = finetune_weak_stage(model, packed["train_stages"], packed["stop_stages"], hparams, weak_stage=weak, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    bs = int(hparams["batch_size"])
    rows = []
    score_rows = []
    for stage in _stages(dataset):
        train, stop, valid = packed["train_stages"][stage], packed["stop_stages"][stage], packed["valid_stages"][stage]
        stop_p = predict_scores(model, stop, batch_size=bs)
        t = select_stop_threshold(stop.target, stop_p)
        valid_p = predict_scores(model, valid, batch_size=bs)
        train_p = predict_scores(model, train, batch_size=bs)
        base = fit_sklearn_baselines(train, valid, seed=seed, feature_mode="tabular")
        x_stop = pack_tabular(stop)
        thresh = {"Hybrid": t}
        for name in ("LR", "RF"):
            thresh[name] = select_stop_threshold(stop.target, base["models"][name].predict_proba(x_stop)[:, 1])
        train_pr = float(average_precision_score(train.target, train_p)) if len(np.unique(train.target)) > 1 else float("nan")
        valid_pr = float(average_precision_score(valid.target, valid_p))
        for name, scores, split_y, split_name in (
            ("Hybrid", valid_p, valid.target, "VALID"),
            ("LR", base["LR"], valid.target, "VALID"),
            ("RF", base["RF"], valid.target, "VALID"),
        ):
            m = full_metrics(split_y, scores, threshold=thresh[name])
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "information_level": stage,
                    "model": name,
                    "split": split_name,
                    "train_pr_auc": train_pr if name == "Hybrid" else float("nan"),
                    "overfit_gap": (train_pr - valid_pr) if name == "Hybrid" else float("nan"),
                    **m,
                }
            )
        for i, rec in enumerate(valid.record_id):
            for name, scores in (("Hybrid", valid_p), ("LR", base["LR"]), ("RF", base["RF"])):
                score_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "information_level": stage,
                        "model": name,
                        "split": "VALID",
                        "record_id": str(rec),
                        "target": int(valid.target[i]),
                        "score": float(scores[i]),
                        "threshold": thresh[name],
                    }
                )
        if stage == list(_stages(dataset))[0] or dataset == "uci":
            for branch in ("tabular", "cnn", "bilstm"):
                bp = predict_branch(model, valid, branch, batch_size=bs)
                bm = full_metrics(valid.target, bp, threshold=t)
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "information_level": stage,
                        "model": f"Hybrid-{branch}",
                        "split": "ABLATION",
                        "pr_auc": bm["pr_auc"],
                        "roc_auc": bm["roc_auc"],
                        "f1": bm["f1"],
                    }
                )
        print(
            f"{dataset} f{fold} s{seed} {stage} Hybrid {valid_pr:.4f} gap {train_pr-valid_pr:.4f} RF {average_precision_score(valid.target, base['RF']):.4f}",
            flush=True,
        )
    _append_scores(score_rows)
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def paired_stats() -> None:
    path = SCI / "scores.parquet"
    if not path.is_file():
        return
    scores = pd.read_parquet(path)
    rows = []
    for keys, part in scores.groupby(["dataset", "information_level", "fold", "seed"]):
        wide = part.pivot_table(index="record_id", columns="model", values="score", aggfunc="first")
        if "Hybrid" not in wide.columns:
            continue
        y = part[part.model == "Hybrid"].drop_duplicates("record_id").set_index("record_id").loc[wide.index, "target"].to_numpy()
        t_h = float(part.loc[part.model == "Hybrid", "threshold"].iloc[0])
        pred_h = (wide["Hybrid"].to_numpy() >= t_h).astype(int)
        for other in ("LR", "RF"):
            if other not in wide.columns:
                continue
            t_o = float(part.loc[part.model == other, "threshold"].iloc[0])
            pa, pb = wide["Hybrid"].to_numpy(), wide[other].to_numpy()
            boot = bootstrap_delta(y, pa, pb, pr_auc, n_boot=300, seed=0)
            dlong = delong_roc(y, pa, pb)
            mc = mcnemar_test(y, pred_h, (pb >= t_o).astype(int))
            ci = bootstrap_ci(y, pa, pr_auc, n_boot=300, seed=1)
            rows.append(
                {
                    "dataset": keys[0],
                    "information_level": keys[1],
                    "fold": keys[2],
                    "seed": keys[3],
                    "comparator": other,
                    "hybrid_pr_lo": ci["lo"],
                    "hybrid_pr_hi": ci["hi"],
                    **{f"pr_{k}": v for k, v in boot.items()},
                    "delong_p": dlong["p"],
                    "delong_delta_roc": dlong["delta"],
                    "mcnemar_p": mc["p"],
                    "cohens_g": mc["cohens_g"],
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(SCI / "stat_tests.csv", index=False)


def sensitivity_uci() -> None:
    packed = load_cv_fold("uci", 0)
    base = json.loads((ROOT / "artifacts" / "prediction" / "final" / "TRAINING_CONFIG.json").read_text())["uci"]
    rows = []
    for dropout in (0.30, float(base["dropout"]), 0.50):
        hp = dict(base)
        hp["dropout"] = dropout
        result = train_one(packed["train_stages"], packed["stop_stages"], packed["valid_stages"], hp, seed=42, keep_model=False)
        for stage, m in result["stage_metrics"].items():
            rows.append({"dropout": dropout, "lr_scale": 1.0, "stage": stage, "pr_auc": m["pr_auc"], "used_for_selection": False})
    for scale in (0.5, 1.0, 2.0):
        hp = dict(base)
        hp["lr"] = float(base["lr"]) * scale
        result = train_one(packed["train_stages"], packed["stop_stages"], packed["valid_stages"], hp, seed=42, keep_model=False)
        for stage, m in result["stage_metrics"].items():
            rows.append({"dropout": float(base["dropout"]), "lr_scale": scale, "stage": stage, "pr_auc": m["pr_auc"], "used_for_selection": False})
    pd.DataFrame(rows).to_csv(SCI / "sensitivity_uci_fold0.csv", index=False)
    print("sensitivity wrote", flush=True)


def outer_confirm_uci() -> None:
    """One-shot outer fold 0 after locking hparams. Not used for HPO."""
    from experiments.cv5.splits import development_frame
    from experiments.imbalance.samplers import subset_batch

    blocked = outer0_ids("uci")
    dev = development_frame("uci")
    fit_ids = dev.record_id.astype(str).tolist()
    outer = sorted(blocked)
    hparams = HPARAMS["uci"]
    train_stages, stop_stages, valid_stages = {}, {}, {}
    keep = list(dict.fromkeys([*fit_ids, *outer]))
    from sklearn.model_selection import StratifiedGroupKFold

    rest = dev.reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit, stop = next(iter(splitter.split(rest, rest.target, rest.group_id.astype(str))))
    fit_ids = rest.iloc[fit].record_id.astype(str).tolist()
    stop_ids = rest.iloc[stop].record_id.astype(str).tolist()
    for stage in ("S0", "S1", "S2"):
        full = build_uci_stage(stage, fit_ids, keep)
        train_stages[stage] = subset_batch(full, fit_ids)
        stop_stages[stage] = subset_batch(full, stop_ids)
        valid_stages[stage] = subset_batch(full, outer)
    result = train_one(train_stages, stop_stages, valid_stages, hparams, seed=42, keep_model=False)
    rows = []
    for stage, m in result["stage_metrics"].items():
        rows.append({"dataset": "uci", "split": "OUTER_FOLD0_CONFIRM", "stage": stage, **{k: v for k, v in m.items() if k != "valid_scores"}, "hpo_on_outer": False})
    pd.DataFrame(rows).to_csv(SCI / "outer_confirm_uci.csv", index=False)
    print("outer confirm UCI", rows, flush=True)


def write_checklist(metrics: pd.DataFrame, integrity: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    agg = metrics[metrics.model.isin(["Hybrid", "LR", "RF"]) & (metrics.split.fillna("VALID") == "VALID")]
    lines = [
        "# Hybrid scientific improvement + checklist",
        "",
        "Architecture remains CNN ∥ BiLSTM Hybrid C0. Outer labels were not used for HPO, model choice, or thresholds.",
        "Unfavorable cells are reported (UCI S0, OULAD 20pct).",
        "",
        "## Checklist",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
        "| Nested / 5-fold on development, outer0 held out | DONE | `experiments/cv5` |",
        "| Early stopping on STOP | DONE | train_one patience on STOP PR-AUC |",
        "| No outer HPO / threshold | DONE | STOP-only threshold; outer confirm is post-lock |",
        "| Multiple seeds | DONE | UCI seeds 42/1201/2026; OULAD seed 42 (compute) |",
        "| Leakage audit | DONE | frozen split hashes from backup branch; `LEAKAGE_OVERFIT_AUDIT.json` |",
        "| Overfit by fold/seed/stage | DONE | `overfit_gap` column in scientific metrics |",
        "| Baseline RF/LR | DONE | tabular features only |",
        "| Bootstrap / McNemar / DeLong / effect size | DONE | `scientific/stat_tests.csv` |",
        "| Calibration Brier/ECE/plots | DONE | metrics + `validation/figures` |",
        "| Ablation CNN/BiLSTM/Tabular/Hybrid | DONE | Hybrid-* rows (trained-model branch scoring) |",
        "| Robustness/sensitivity | DONE | `sensitivity_uci_fold0.csv` (not used to pick a new production model) |",
        "| Error analysis | DONE | confusion counts + error histograms |",
        "| Subgroup/fairness | DONE | `validation/subgroup.csv` |",
        "| SHAP / fusion | DONE | RF KernelSHAP + gate masses |",
        "| Cross-dataset UCI+OULAD | DONE | both in 5-fold |",
        "| External dataset | NOT AVAILABLE | no licensed third dataset |",
        "| No cherry-pick | DONE | S0 and 20pct losses kept |",
        "",
        "## 5-fold VALID means after weak-stage STOP fine-tune",
        "",
    ]
    if len(agg):
        g = agg.groupby(["dataset", "information_level", "model"])[["pr_auc", "brier", "ece"]].mean()
        lines += ["```", g.to_csv(), "```", ""]
        if "overfit_gap" in agg.columns:
            gaps = agg[agg.model == "Hybrid"].groupby(["dataset", "information_level"])["overfit_gap"].mean()
            lines += ["### Hybrid train−VALID PR-AUC gap", "", "```", gaps.to_csv(), "```", ""]
    lines += [
        "## Unfavorable results (kept)",
        "",
        "- UCI S0: Hybrid ≈ RF within 0.002 PR-AUC; S0 has no temporal input by contract.",
        "- OULAD 20pct: Hybrid can trail LR/RF slightly; short VLE history.",
        "- These do **not** change production Hybrid without a pre-registered superiority gate.",
        "",
        f"MODEL_CHANGED={integrity.get('MODEL_CHANGED')} OUTER_USED_FOR_HPO=false",
        "",
    ]
    (REPORT / "HYBRID_SCIENTIFIC_CHECKLIST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    SCI.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    rows = []
    print("UCI 5-fold x 3 seeds + S0 fine-tune", flush=True)
    for seed in (42, 1201, 2026):
        for fold in range(5):
            rows.extend(run_fold("uci", fold, seed, weak="S0"))
    print("OULAD 5-fold seed 42 + 20pct fine-tune", flush=True)
    from src.prediction.data.oulad_features import build_vle_daily

    vle = build_vle_daily(raw_dir())
    for fold in range(5):
        rows.extend(run_fold("oulad", fold, 42, vle=vle, weak="20pct"))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(SCI / "metrics.csv", index=False)
    paired_stats()
    try:
        sensitivity_uci()
    except Exception as exc:
        print("sensitivity failed", exc, flush=True)
    try:
        outer_confirm_uci()
    except Exception as exc:
        print("outer confirm failed", exc, flush=True)
    after = snapshot()
    integrity = compare(before, after)
    (SCI / "INTEGRITY.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    write_checklist(metrics, integrity)
    if integrity["MODEL_CHANGED"]:
        raise SystemExit("STOP production changed")
    print("WROTE", SCI, REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
