"""Batch metric figures: ablation, errors, subgroups, fairness, SHAP, fusion. Agg backend."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "artifacts" / "experiments" / "validation"
FIG = SRC / "figures"
LOCKED = ROOT / "artifacts" / "prediction" / "final"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "figure.facecolor": "white", "savefig.bbox": "tight"})


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)


def _metrics() -> pd.DataFrame:
    return pd.read_csv(SRC / "metrics_valid.csv")


def bars_hybrid_vs_base(metric: str, dataset: str, fname: str) -> None:
    m = _metrics()
    part = m[m.dataset == dataset]
    stages = list(part.information_level.unique())
    order = ["S0", "S1", "S2"] if dataset == "uci" else ["20pct", "35pct", "50pct", "75pct", "100pct"]
    stages = [s for s in order if s in set(stages)]
    models = ["Hybrid", "LR", "RF"]
    x = np.arange(len(stages))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for i, model in enumerate(models):
        means = [part[(part.information_level == s) & (part.model == model)][metric].mean() for s in stages]
        ax.bar(x + (i - 1) * w, means, w, label=model)
    ax.set_xticks(x, stages)
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset.upper()} {metric}: Hybrid CNN–BiLSTM vs LR/RF")
    ax.legend()
    _save(fig, fname)


def ablation_bars() -> None:
    path = SRC / "ablation_branch.csv"
    if not path.is_file():
        return
    a = pd.read_csv(path)
    m = _metrics()
    for dataset in a.dataset.unique():
        stages = ["S0", "S1", "S2"] if dataset == "uci" else ["20pct", "35pct", "50pct", "75pct", "100pct"]
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        x = np.arange(len(stages))
        w = 0.18
        series = {
            "full": [m[(m.dataset == dataset) & (m.model == "Hybrid") & (m.information_level == s)].pr_auc.mean() for s in stages],
            "tabular": [a[(a.dataset == dataset) & (a.branch == "tabular") & (a.information_level == s)].pr_auc.mean() for s in stages],
            "cnn": [a[(a.dataset == dataset) & (a.branch == "cnn") & (a.information_level == s)].pr_auc.mean() for s in stages],
            "bilstm": [a[(a.dataset == dataset) & (a.branch == "bilstm") & (a.information_level == s)].pr_auc.mean() for s in stages],
        }
        for i, (name, vals) in enumerate(series.items()):
            ax.bar(x + (i - 1.5) * w, vals, w, label=name)
        ax.set_xticks(x, stages)
        ax.set_ylabel("PR-AUC")
        ax.set_title(f"{dataset.upper()} ablation (trained Hybrid branch-only vs full)")
        ax.legend()
        _save(fig, f"ablation_{dataset}.png")


def fusion_stacked() -> None:
    path = SRC / "gate_masses.csv"
    if not path.is_file():
        return
    g = pd.read_csv(path)
    for dataset, part in g.groupby("dataset"):
        stages = ["S0", "S1", "S2"] if dataset == "uci" else ["20pct", "35pct", "50pct", "75pct", "100pct"]
        stages = [s for s in stages if s in set(part.information_level)]
        tab = [part[part.information_level == s].tabular_mass_mean.mean() for s in stages]
        cnn = [part[part.information_level == s].cnn_mass_mean.mean() for s in stages]
        lstm = [part[part.information_level == s].bilstm_mass_mean.mean() for s in stages]
        x = np.arange(len(stages))
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.bar(x, tab, label="tabular")
        ax.bar(x, cnn, bottom=tab, label="cnn")
        ax.bar(x, lstm, bottom=np.asarray(tab) + np.asarray(cnn), label="bilstm")
        ax.set_xticks(x, stages)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean gate mass")
        ax.set_title(f"{dataset.upper()} fusion branch contribution")
        ax.legend()
        _save(fig, f"fusion_gate_{dataset}.png")


def roc_pr_curves() -> None:
    for name in ("scores_uci.parquet", "scores_oulad.parquet"):
        path = SRC / name
        if not path.is_file():
            continue
        scores = pd.read_parquet(path)
        dataset = scores.dataset.iloc[0]
        for stage, part in scores.groupby("information_level"):
            fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
            for model, sub in part.groupby("model"):
                y = sub.target.to_numpy()
                p = sub.score.to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y, p)
                prec, rec, _ = precision_recall_curve(y, p)
                axes[0].plot(fpr, tpr, label=model)
                axes[1].plot(rec, prec, label=model)
            axes[0].plot([0, 1], [0, 1], "--", color="gray")
            axes[0].set_title(f"{dataset} {stage} ROC")
            axes[0].set_xlabel("FPR")
            axes[0].set_ylabel("TPR")
            axes[1].set_title(f"{dataset} {stage} PR")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[0].legend()
            axes[1].legend()
            _save(fig, f"curves_{dataset}_{stage}.png")


def error_histograms() -> None:
    for name in ("scores_uci.parquet", "scores_oulad.parquet"):
        path = SRC / name
        if not path.is_file():
            continue
        scores = pd.read_parquet(path)
        hybrid = scores[scores.model == "Hybrid"]
        dataset = hybrid.dataset.iloc[0]
        for stage, part in hybrid.groupby("information_level"):
            y = part.target.to_numpy()
            p = part.score.to_numpy()
            t = float(part.threshold.iloc[0])
            pred = (p >= t).astype(int)
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            for label, mask, color in (
                ("TN", (y == 0) & (pred == 0), "C0"),
                ("FP", (y == 0) & (pred == 1), "C3"),
                ("FN", (y == 1) & (pred == 0), "C1"),
                ("TP", (y == 1) & (pred == 1), "C2"),
            ):
                if mask.any():
                    ax.hist(p[mask], bins=20, alpha=0.45, label=f"{label} n={int(mask.sum())}", color=color)
            ax.axvline(t, color="black", ls="--", label=f"STOP t={t:.2f}")
            ax.set_xlabel("P(risk)")
            ax.set_title(f"{dataset} {stage} Hybrid score by error type")
            ax.legend()
            _save(fig, f"error_hist_{dataset}_{stage}.png")


def confusion_summary() -> None:
    m = _metrics()
    h = m[m.model == "Hybrid"]
    for dataset, part in h.groupby("dataset"):
        stages = list(part.information_level.unique())
        fig, axes = plt.subplots(1, len(stages), figsize=(3.2 * len(stages), 3.3))
        if len(stages) == 1:
            axes = [axes]
        for ax, stage in zip(axes, stages):
            s = part[part.information_level == stage][["tn", "fp", "fn", "tp"]].mean()
            mat = np.array([[s.tn, s.fp], [s.fn, s.tp]])
            ax.imshow(mat, cmap="Blues")
            ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], ["True 0", "True 1"])
            for (i, j), val in np.ndenumerate(mat):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center")
            ax.set_title(stage)
        fig.suptitle(f"{dataset} Hybrid confusion (mean counts)")
        _save(fig, f"confusion_summary_{dataset}.png")


def fairness_charts() -> None:
    path = SRC / "subgroup.csv"
    if not path.is_file():
        return
    sg = pd.read_csv(path)
    for dataset, dpart in sg.groupby("dataset"):
        for attr, apart in dpart.groupby("attribute"):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            grouped = apart.groupby("group")[["pr_auc", "tpr", "fpr"]].mean()
            grouped = grouped.sort_values("pr_auc")
            for ax, col, title in zip(axes, ("pr_auc", "tpr", "fpr"), ("PR-AUC", "TPR", "FPR")):
                ax.barh(grouped.index.astype(str), grouped[col])
                ax.set_title(f"{dataset} {attr} {title}")
            fig.tight_layout()
            safe = str(attr).replace("/", "_")
            _save(fig, f"fairness_{dataset}_{safe}.png")
            gap = apart.groupby(["information_level", "fold", "seed"]).agg(
                pr_gap=("pr_auc", lambda s: s.max() - s.min()),
                tpr_gap=("tpr", lambda s: s.max() - s.min()),
                fpr_gap=("fpr", lambda s: s.max() - s.min()),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(7.5, 4))
            stages = list(gap.information_level.unique())
            x = np.arange(len(stages))
            for i, col in enumerate(("pr_gap", "tpr_gap", "fpr_gap")):
                ax.bar(x + (i - 1) * 0.25, [gap[gap.information_level == s][col].mean() for s in stages], 0.25, label=col)
            ax.set_xticks(x, stages)
            ax.set_ylabel("max−min gap")
            ax.set_title(f"{dataset} {attr} fairness gaps")
            ax.legend()
            _save(fig, f"fairness_gap_{dataset}_{safe}.png")


def shap_bar() -> None:
    path = SRC / "shap_rf_uci_s2.csv"
    if not path.is_file():
        return
    s = pd.read_csv(path).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.feature.astype(str), s.mean_abs_shap)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("KernelSHAP RF on packed UCI S2 Hybrid features")
    _save(fig, "shap_rf_uci_s2.png")


def overfit_locked() -> None:
    path = LOCKED / "OVERFIT_AUDIT.json"
    if not path.is_file():
        return
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for dataset in ("uci", "oulad"):
        stages = payload.get(dataset, {}).get("stages", {})
        for stage, rec in stages.items():
            rows.append({"dataset": dataset, "stage": stage, "gap": rec["generalization_gap_mean"], "valid": rec["pr_auc_mean"], "cls": rec["overfit_class"]})
    if not rows:
        return
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(frame))
    colors = [{"HIGH": "C3", "MODERATE": "C1", "LOW": "C2"}.get(c, "C0") for c in frame.cls]
    ax.bar(x, frame.gap, color=colors)
    ax.set_xticks(x, [f"{d}:{s}" for d, s in zip(frame.dataset, frame.stage)], rotation=40, ha="right")
    ax.set_ylabel("train−VALID PR-AUC gap")
    ax.set_title("Locked Phase-4 overfit audit (outer unused)")
    _save(fig, "overfit_locked_gaps.png")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if (SRC / "metrics_valid.csv").is_file():
        for metric in ("pr_auc", "roc_auc", "f1", "recall", "specificity", "brier", "ece", "h2_mean"):
            for dataset in pd.read_csv(SRC / "metrics_valid.csv").dataset.unique():
                bars_hybrid_vs_base(metric, dataset, f"metric_{dataset}_{metric}.png")
        confusion_summary()
    ablation_bars()
    fusion_stacked()
    roc_pr_curves()
    error_histograms()
    fairness_charts()
    shap_bar()
    overfit_locked()
    n = len(list(FIG.glob("*.png")))
    print("figures", n, "->", FIG)


if __name__ == "__main__":
    main()
