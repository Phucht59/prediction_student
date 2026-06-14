"""Generate restrained thesis figures directly from final experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "final"
OUT = REPORTS / "figures" / "current"
DATASETS = ("student-mat", "student-por", "xapi")
LABELS = ("Low", "Medium", "High")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def recommendation_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, "Đặc trưng\nngười học"),
        (0.23, "Chuẩn hóa theo\ntrain pool"),
        (0.44, "MLP\n64 → 32 → 6"),
        (0.65, "Xếp hạng 6\nyếu tố rủi ro"),
        (0.84, "Mẫu hành động\ntheo giai đoạn"),
    ]
    for x, label in boxes:
        ax.add_patch(plt.Rectangle((x, 0.32), 0.14, 0.36, fill=False, linewidth=1.2, edgecolor="#333333"))
        ax.text(x + 0.07, 0.50, label, ha="center", va="center", fontsize=10)
    for left, right in zip(boxes, boxes[1:]):
        ax.annotate("", xy=(right[0], 0.50), xytext=(left[0] + 0.14, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.1})
    ax.text(0.50, 0.12, "Huấn luyện chỉ trên train pool; locked test chỉ dùng để đánh giá", ha="center", fontsize=9)
    save(fig, "04_recommendation_mlp.png")


def classification_figures() -> None:
    metrics = {ds: load_json(REPORTS / "metrics" / f"{ds}_3class_locked_test_metrics.json") for ds in DATASETS}
    cv = {ds: load_json(REPORTS / "metrics" / f"{ds}_3class_optuna_cv.json")["f1_macro_best"] for ds in DATASETS}
    display = ["Student-Mat", "Student-Por", "xAPI"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(3)
    width = 0.24
    for offset, key, label in [(-width, "Accuracy", "Accuracy"), (0, "Precision-Macro", "Precision-Macro"), (width, "F1-Macro", "F1-Macro")]:
        values = [metrics[ds][key] for ds in DATASETS]
        ax.bar(x + offset, values, width, label=label)
    ax.set_xticks(x, display)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Điểm số")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "07_locked_metrics.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    test = [metrics[ds]["F1-Macro"] for ds in DATASETS]
    ax.bar(x - 0.18, [cv[ds] for ds in DATASETS], 0.36, label="Optuna best CV")
    ax.bar(x + 0.18, test, 0.36, label="Locked test")
    ax.set_xticks(x, display)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-Macro")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "08_cv_test_gap.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    class_f1: dict[str, list[float]] = {}
    for ax, ds, title in zip(axes, DATASETS, display):
        pred = pd.read_csv(REPORTS / "predictions" / f"{ds}_3class_predictions.csv")
        cm = confusion_matrix(pred["True_Label"], pred["Pred_Label"], labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=LABELS, yticklabels=LABELS, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        report = classification_report(pred["True_Label"], pred["Pred_Label"], labels=[0, 1, 2], output_dict=True, zero_division=0)
        class_f1[ds] = [report[str(i)]["f1-score"] for i in range(3)]
    save(fig, "09_confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for index, ds in enumerate(DATASETS):
        ax.bar(x + (index - 1) * 0.24, class_f1[ds], 0.24, label=display[index])
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "10_per_class_f1.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for ds, title in zip(DATASETS, display):
        pred = pd.read_csv(REPORTS / "predictions" / f"{ds}_3class_predictions.csv")
        ax.hist(pred["Confidence"], bins=np.linspace(0.3, 1.0, 15), alpha=0.45, label=title)
    ax.set_xlabel("Độ tin cậy ensemble")
    ax.set_ylabel("Số quan sát")
    ax.legend(frameon=False)
    save(fig, "12_confidence_distribution.png")


def feature_importance_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, ds, title in zip(axes, DATASETS, ("Student-Mat", "Student-Por", "xAPI")):
        frame = pd.read_csv(REPORTS / "explanations" / f"{ds}_3class_feature_importance.csv").head(6).sort_values("Importance")
        ax.barh(frame["Feature"], frame["Importance"], color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("Mức giảm F1-Macro")
    save(fig, "11_feature_importance.png")


def recommendation_figures() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    counts = []
    for ds in DATASETS:
        frame = pd.read_csv(REPORTS / "recommendations" / f"{ds}_3class_learning_paths.csv")
        counts.append(frame["risk_band"].value_counts().reindex(["high", "moderate", "stable"], fill_value=0).to_numpy())
    x = np.arange(3)
    for index, label in enumerate(("High", "Moderate", "Stable")):
        ax.bar(x + (index - 1) * 0.24, [row[index] for row in counts], 0.24, label=label)
    ax.set_xticks(x, ("Student-Mat", "Student-Por", "xAPI"))
    ax.set_ylabel("Số người học")
    ax.legend(frameon=False)
    save(fig, "13_risk_band_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.24
    for index, ds in enumerate(DATASETS):
        evaluation = load_json(REPORTS / "recommendations" / f"{ds.replace('-', '_')}_evaluation.json")
        values = [evaluation["ranking"]["precision_at_3"], evaluation["ranking"]["recall_at_3"], evaluation["ranking"]["ndcg_at_3"]]
        ax.bar(np.arange(3) + (index - 1) * width, values, width, label=("Student-Mat", "Student-Por", "xAPI")[index])
    ax.set_xticks(np.arange(3), ("Precision@3", "Recall@3", "NDCG@3"))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Điểm số")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "14_recommendation_ranking.png")


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    recommendation_diagram()
    classification_figures()
    feature_importance_figure()
    recommendation_figures()


if __name__ == "__main__":
    main()
