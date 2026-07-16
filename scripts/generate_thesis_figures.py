from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.model_display_names import get_display_name


DEFAULT_SOURCE = (
    ROOT
    / "artifacts"
    / "study_c_oulad_v3_closure"
    / "oulad-v3-fair-db-closure-20260716-v1"
    / "ensemble_metrics.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "thesis_figures"

# The mixed dynamic ML comparator selected two different estimator families across
# outer folds. It remains in the thesis table under its accurate generic display
# name, but is omitted from algorithm-labelled figures to avoid calling it a pure
# Logistic Regression or HistGradientBoosting result.
FIGURE_CANDIDATES = ["V3-MLF", "V3-A0F-ENS", "V3-P0-ENS", "V3-D0-ENS"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thesis_metrics(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source).set_index("candidate_id")
    missing = sorted(set(FIGURE_CANDIDATES) - set(frame.index))
    if missing:
        raise ValueError(f"Missing thesis figure candidates: {missing}")
    selected = frame.loc[FIGURE_CANDIDATES].reset_index()
    selected["display_name"] = selected.candidate_id.map(get_display_name)
    if selected.display_name.duplicated().any():
        duplicates = selected.loc[selected.display_name.duplicated(False), "display_name"].tolist()
        raise ValueError(f"Figure labels must be unique: {duplicates}")
    return selected


def _finish(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def macro_f1_figure(frame: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    colors = ["#2f6b9a", "#5b8f6a", "#7d6aa5", "#c06b3e"]
    bars = axis.bar(frame.display_name, frame.macro_f1, color=colors)
    axis.set_ylabel("Macro-F1")
    axis.set_title("So sánh Macro-F1 trên OULAD")
    axis.set_ylim(0.80, 0.84)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, frame.macro_f1):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0004, f"{value:.4f}", ha="center", fontsize=9)
    _finish(figure, output / "model_macro_f1_comparison.png")


def precision_recall_figure(frame: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.4))
    positions = np.arange(len(frame))
    width = 0.36
    axis.bar(positions - width / 2, frame.at_risk_precision, width, label="Risk Precision", color="#2f6b9a")
    axis.bar(positions + width / 2, frame.at_risk_recall, width, label="Risk Recall", color="#c06b3e")
    axis.set_xticks(positions, frame.display_name, rotation=18)
    axis.set_ylabel("Giá trị")
    axis.set_title("Precision và Recall cho nhóm có nguy cơ")
    axis.set_ylim(0.70, 0.87)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _finish(figure, output / "model_precision_recall_comparison.png")


def pr_auc_figure(frame: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.2))
    colors = ["#2f6b9a", "#5b8f6a", "#7d6aa5", "#c06b3e"]
    bars = axis.bar(frame.display_name, frame.pr_auc, color=colors)
    axis.set_ylabel("PR-AUC")
    axis.set_title("So sánh PR-AUC trên OULAD")
    axis.set_ylim(0.87, 0.90)
    axis.grid(axis="y", alpha=0.25)
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, frame.pr_auc):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:.4f}", ha="center", fontsize=9)
    _finish(figure, output / "model_pr_auc_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-facing model comparison figures from frozen closure evidence.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = thesis_metrics(source)
    macro_f1_figure(frame, output)
    precision_recall_figure(frame, output)
    pr_auc_figure(frame, output)

    manifest = {
        "source": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "display_labels": frame.display_name.tolist(),
        "figures": [
            "model_macro_f1_comparison.png",
            "model_precision_recall_comparison.png",
            "model_pr_auc_comparison.png",
        ],
        "excluded_mixed_estimator_comparator": {
            "candidate_id": "V3-MLD",
            "reason": "Outer folds selected Logistic Regression twice and HistGradientBoosting once; a single algorithm label would be inaccurate.",
        },
        "metrics_copied_manually": False,
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "figures": len(manifest["figures"]), "source_sha256": manifest["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
