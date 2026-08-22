"""Re-evaluate locked C0-R and one-weight baselines for headline metrics.

Headline four: accuracy, precision, F1, plus the highest remaining ranking metric
(AP vs ROC-AUC). Uses frozen hparams only — no new HPO, no outer test.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HS_V2_GPU_TREES", "1")

import pandas as pd

from .baselines import fit_eval_stacked, predictor_columns
from .data import inner_partitions, scale_views, stacked_baseline_frame
from .fair_baselines import fair_params, load_hybrid_authority
from .hpo import evaluate_hybrid
from .io_utils import git_commit, utc_now, write_json
from .model import make_config
from .paths import MANIFEST_DIR, METRIC_DIR, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import BASELINE_ROSTER, SEEDS_ROBUST, protocol_hash, stages_for
from .status import write_status

STATE = RUN_DIR / "lock_metrics_state.json"
HEADLINE = ("accuracy", "risk_precision", "risk_f1")
RANKING = ("roc_auc", "ap")
KEEP = (
    "accuracy",
    "risk_precision",
    "risk_f1",
    "ap",
    "roc_auc",
    "risk_recall",
    "recall_at_20",
    "threshold",
    "n",
    "prevalence",
)


def _log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _boost() -> None:
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log(f"CUDA {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        _log(f"cuda {exc}")
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, "HIGH_PRIORITY_CLASS") else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": []}


def _save_state(st: dict) -> None:
    write_json(STATE, st)


def _empty_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def pick_fourth(means: dict[str, float]) -> tuple[str, float]:
    name = max(RANKING, key=lambda key: float(means.get(key, float("-inf"))))
    return name, float(means[name])


def eval_baselines(domain: str) -> pd.DataFrame:
    lock_path = RUN_DIR / f"baseline_lock_{domain}_fair.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    best = lock.get("best_params") or {}
    rows = []
    for fold in (0, 1, 2):
        fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
        prepared = scale_views(domain, fit_ids)
        stacked = stacked_baseline_frame(prepared)
        cols, cats = predictor_columns(stacked)
        for seed in SEEDS_ROBUST:
            for name in BASELINE_ROSTER:
                params = fair_params(name, seed, domain, best.get(name))
                t0 = time.time()
                out = fit_eval_stacked(name, stacked, cols, cats, fit_ids, stop_ids, valid_ids, seed, params)
                _log(f"bl {domain} {name} fold={fold} seed={seed} {time.time()-t0:.0f}s n_models={out['n_models']}")
                if int(out["n_models"]) != 1:
                    raise RuntimeError(f"FAIRNESS_VIOLATION:{name}")
                for stage, metrics in out["stages"].items():
                    row = {"domain": domain, "model": name, "fold": fold, "seed": seed, "stage": stage, "n_models": 1}
                    for key in KEEP:
                        row[key] = metrics.get(key)
                    rows.append(row)
                _empty_gpu()
    table = pd.DataFrame(rows)
    table.to_csv(METRIC_DIR / f"headline_baseline_{domain}.csv", index=False)
    return table


def eval_hybrid(domain: str) -> pd.DataFrame:
    if domain == "uci":
        payload = json.loads((RUN_DIR / "robust_uci_C0-R.json").read_text(encoding="utf-8"))
        bp = dict(payload.get("best_params") or {})
        max_epochs, batch_size, lambda_kd = 24, int(bp.get("batch_size", 32)), float(bp.get("lambda_kd", 0.0))
    else:
        payload = json.loads((RUN_DIR / "c0r_sci_hpo_oulad.json").read_text(encoding="utf-8"))
        bp = dict(payload.get("best_params") or {})
        max_epochs, batch_size, lambda_kd = 20, 256, 0.0
    rows = []
    for fold in (0, 1, 2):
        fit_ids, _, _ = inner_partitions(domain, fold)
        prepared = scale_views(domain, fit_ids)
        cfg = make_config(
            "C0-R",
            prepared.static_dim,
            prepared.temporal_dim,
            prepared.aggregate_dim,
            d_fuse=int(bp.get("d_fuse", 64)),
            cnn_channels=int(bp.get("cnn_channels", 32)),
            bilstm_hidden=int(bp.get("bilstm_hidden", 32)),
            dropout=float(bp.get("dropout", 0.3)),
        )
        train_kw = {
            "lr": float(bp.get("lr", 2e-4)),
            "weight_decay": float(bp.get("weight_decay", 2e-4)),
            "pos_weight_multiplier": float(bp.get("pos_weight_multiplier", 1.0)),
            "lambda_rank": float(bp.get("lambda_rank", 0.15)),
            "lambda_aux": float(bp.get("lambda_aux", 0.25)),
            "lambda_kd": lambda_kd,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": 8,
            "use_ema": True,
            "multiprefix": True,
        }
        for seed in SEEDS_ROBUST:
            t0 = time.time()
            _log(f"hybrid {domain} fold={fold} seed={seed}")
            out = evaluate_hybrid(domain, "C0-R", cfg, train_kw, fold=fold, seed=seed, prepared=prepared)
            stages = out["valid"]["stages"]
            _log(f"  AP={ {s: round(stages[s]['ap'],4) for s in stages} } {time.time()-t0:.0f}s")
            for stage, metrics in stages.items():
                row = {"domain": domain, "model": "Hybrid-C0-R", "fold": fold, "seed": seed, "stage": stage, "n_models": 1}
                for key in KEEP:
                    row[key] = metrics.get(key)
                rows.append(row)
            _empty_gpu()
    table = pd.DataFrame(rows)
    table.to_csv(METRIC_DIR / f"headline_hybrid_{domain}.csv", index=False)
    return table


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    metrics = ["accuracy", "risk_precision", "risk_f1", "ap", "roc_auc"]
    g = table.groupby(["model", "stage"])[metrics].mean().reset_index()
    out: dict[str, Any] = {}
    for model, part in g.groupby("model"):
        stages = {}
        for _, row in part.iterrows():
            means = {k: float(row[k]) for k in metrics}
            fourth_name, fourth_value = pick_fourth(means)
            stages[str(row["stage"])] = {
                "accuracy": means["accuracy"],
                "precision": means["risk_precision"],
                "f1": means["risk_f1"],
                "fourth_metric": fourth_name,
                "fourth_value": fourth_value,
                "ap": means["ap"],
                "roc_auc": means["roc_auc"],
            }
        out[str(model)] = stages
    return out


def write_report(uci_bl: pd.DataFrame, oul_bl: pd.DataFrame, uci_h: pd.DataFrame, oul_h: pd.DataFrame) -> None:
    uci = summarize(pd.concat([uci_bl, uci_h], ignore_index=True))
    oul = summarize(pd.concat([oul_bl, oul_h], ignore_index=True))
    payload = {
        "frozen_at": utc_now(),
        "git_commit": git_commit(),
        "protocol_hash": protocol_hash(),
        "headline_four": ["accuracy", "precision", "f1", "highest_of_ap_or_roc_auc"],
        "threshold": "STOP-only then apply to VALID",
        "outer_test_used": False,
        "hybrid": "C0-R one checkpoint all stages",
        "baselines": "one fitted estimator per family all stages, Optuna-best 40/28",
        "uci": uci,
        "oulad": oul,
        "serving_hybrid_unchanged": True,
        "recommendation_unchanged": True,
    }
    write_json(MANIFEST_DIR / "headline_four_metrics.json", payload)
    write_json(RUN_DIR / "headline_four_metrics.json", payload)

    def block(title: str, blob: dict) -> list[str]:
        lines = [f"## {title}", ""]
        hybrid = blob.get("Hybrid-C0-R") or {}
        if hybrid:
            lines.append("### Hybrid C0-R")
            lines.append("")
            lines.append("| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |")
            lines.append("|---|---:|---:|---:|---|---:|")
            for stage in stages_for(title.lower()):
                if stage not in hybrid:
                    continue
                cell = hybrid[stage]
                lines.append(
                    f"| {stage} | {cell['accuracy']:.4f} | {cell['precision']:.4f} | {cell['f1']:.4f} | {cell['fourth_metric']} | {cell['fourth_value']:.4f} |"
                )
            lines.append("")
        lines.append("### One-weight baselines (same four)")
        lines.append("")
        for model, stages in blob.items():
            if model == "Hybrid-C0-R":
                continue
            lines.append(f"**{model}**")
            lines.append("")
            lines.append("| Mốc | Accuracy | Precision | F1 | Chỉ số cao nhất | Giá trị |")
            lines.append("|---|---:|---:|---:|---|---:|")
            for stage in stages_for(title.lower()):
                if stage not in stages:
                    continue
                cell = stages[stage]
                lines.append(
                    f"| {stage} | {cell['accuracy']:.4f} | {cell['precision']:.4f} | {cell['f1']:.4f} | {cell['fourth_metric']} | {cell['fourth_value']:.4f} |"
                )
            lines.append("")
        return lines

    chunks = [
        "# Locked headline metrics — Hybrid C0-R vs one-weight baselines",
        "",
        "Four metrics: **Accuracy**, **Precision** (positive/risk class), **F1**, and the **highest** of AP vs ROC-AUC.",
        "Thresholds from STOP only. Outer test: **false**. Serving Hybrid C0 and Recommendation V: **unchanged**.",
        "",
        "Baselines: one estimator / one weight set per family, Optuna-best protocol budget (40 UCI / 28 OULAD).",
        "",
    ]
    chunks.extend(block("UCI", uci))
    chunks.extend(block("OULAD", oul))
    chunks.extend(
        [
            "## Decision",
            "",
            "Prediction authority on main remains **Hybrid CNN–BiLSTM (C0)** in `src/prediction`.",
            "This table is the locked research comparison (C0-R, one-weight baselines).",
            "Recommendation V stays the locked recommender (`src/recommend_hybrid/v3`).",
            "",
        ]
    )
    (REPORT_ROOT / "HEADLINE_FOUR_METRICS.md").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    _log(f"wrote {REPORT_ROOT / 'HEADLINE_FOUR_METRICS.md'}")


def main() -> int:
    _boost()
    ensure_dirs()
    st = _load_state()
    done = set(st.get("completed") or [])
    write_status(
        phase="headline four metrics",
        completed=sorted(done),
        evidence=[],
        decision="Re-eval locked hparams for Acc/Prec/F1 + highest ranking metric. No outer.",
        next_step="baselines then Hybrid 3x3",
    )
    tables: dict[str, pd.DataFrame] = {}
    try:
        for key, fn in (
            ("bl_uci", lambda: eval_baselines("uci")),
            ("bl_oulad", lambda: eval_baselines("oulad")),
            ("hy_uci", lambda: eval_hybrid("uci")),
            ("hy_oulad", lambda: eval_hybrid("oulad")),
        ):
            path = METRIC_DIR / {
                "bl_uci": "headline_baseline_uci.csv",
                "bl_oulad": "headline_baseline_oulad.csv",
                "hy_uci": "headline_hybrid_uci.csv",
                "hy_oulad": "headline_hybrid_oulad.csv",
            }[key]
            if key in done and path.exists():
                tables[key] = pd.read_csv(path)
                continue
            _log(f"===== {key} =====")
            tables[key] = fn()
            done.add(key)
            st["completed"] = sorted(done)
            _save_state(st)
        write_report(tables["bl_uci"], tables["bl_oulad"], tables["hy_uci"], tables["hy_oulad"])
        load_hybrid_authority()
        write_status(
            phase="headline four metrics done",
            completed=sorted(done),
            evidence=[str(REPORT_ROOT / "HEADLINE_FOUR_METRICS.md"), str(MANIFEST_DIR / "headline_four_metrics.json")],
            decision="C0 serving + C0-R research metrics locked. Rec unchanged. No outer.",
            next_step="clean and push main",
        )
    except Exception:
        traceback.print_exc()
        write_status(
            phase="headline metrics FAILED",
            completed=sorted(done),
            evidence=[],
            decision="inspect traceback",
            next_step="resume python -m experiments.hybrid_superiority_v2 lock-metrics",
            blockers=["lock_metrics"],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
