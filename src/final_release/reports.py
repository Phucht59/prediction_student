"""Generate all human-readable release reports from canonical JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.final_release.build import FINAL_ROOT, REPORT_ROOT, ROOT
from src.final_release.catalog import OFFICIAL_MODELS


def value(item: dict[str, Any]) -> Any:
    return item.get("value")


def fmt(item: dict[str, Any], digits: int = 4) -> str:
    raw = value(item)
    return "N/A" if raw is None else f"{float(raw):.{digits}f}"


def overall_table(dataset: dict[str, Any], oulad: bool = False) -> str:
    names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]
    labels = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "Macro-F1",
        "Weighted-F1",
    ]
    if oulad:
        names += ["risk_precision", "risk_recall", "risk_f1"]
        labels += ["Risk Precision", "Risk Recall", "Risk F1"]
    names += ["pr_auc", "roc_auc", "brier", "nll", "ece"]
    labels += ["PR-AUC", "ROC-AUC", "Brier ↓", "NLL ↓", "ECE ↓"]
    lines = ["| Model | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for row in dataset["models"]:
        lines.append(
            "| "
            + row["model"]
            + " | "
            + " | ".join(fmt(row["metrics"][name]) for name in names)
            + " |"
        )
    return "\n".join(lines)


def per_class_table(dataset: dict[str, Any]) -> str:
    lines = [
        "| Model | Class | Precision | Recall | F1 | Support | Model Macro-F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset["models"]:
        for item in row["per_class"]:
            lines.append(
                f"| {row['model']} | {item['class']} | {fmt(item['precision'])} | {fmt(item['recall'])} | {fmt(item['f1'])} | {fmt(item['support'], 0)} | {fmt(item['macro_f1'])} |"
            )
    return "\n".join(lines)


def confusion_sections(dataset: dict[str, Any], selected: set[str]) -> str:
    blocks = []
    for row in dataset["models"]:
        if row["model_id"] not in selected or value(row["confusion_matrix"]) is None:
            continue
        matrix = value(row["confusion_matrix"])
        blocks.append(
            f"### {row['model']}\n\n```text\n"
            + "\n".join(" ".join(str(cell) for cell in line) for line in matrix)
            + "\n```"
        )
    return (
        "\n\n".join(blocks)
        or "No frozen final prediction artifact is available for the requested matrices."
    )


def source_notes(dataset: dict[str, Any]) -> str:
    seen = {}
    for row in dataset["models"]:
        for metric in row["metrics"].values():
            if metric.get("source_artifact"):
                seen[metric["source_artifact"]] = metric["source_checksum"]
    return "\n".join(
        f"- `{path}` — SHA-256 `{checksum}`" for path, checksum in sorted(seen.items())
    )


def dataset_report(dataset_id: str, dataset: dict[str, Any]) -> str:
    official = OFFICIAL_MODELS[dataset_id]
    is_oulad = dataset_id == "oulad"
    selected_cm = (
        {"cnn_bilstm", "decision_tree", "random_forest", "xgboost"}
        if is_oulad
        else {"cnn_bilstm", "decision_tree", "random_forest"}
    )
    text = [
        f"# {official['official_name']} — Final Results",
        "",
        "All values come from frozen final outer-OOF or final probability-ensemble evidence. N/A means no frozen final prediction artifact exists; no screening metric or estimate is substituted.",
        "",
        "Precision and Recall in the overall table are macro averages.",
        "",
        "## Overall comparison",
        "",
        overall_table(dataset, is_oulad),
        "",
        "## Per-class results",
        "",
        per_class_table(dataset),
    ]
    if dataset_id == "student_por":
        cnn = dataset["models"][0]
        dt = dataset["models"][4]
        rf = dataset["models"][5]
        low = cnn["per_class"][0]
        cm = value(cnn["confusion_matrix"])
        text += [
            "",
            "## Low-class analysis",
            "",
            f"CNN-BiLSTM Low precision/recall/F1 are {fmt(low['precision'])}/{fmt(low['recall'])}/{fmt(low['f1'])}. Its frozen confusion matrix records {cm[0][1]} Low→Medium and {cm[1][0]} Medium→Low errors. Decision Tree and Random Forest Low-class results are shown in the same per-class table ({fmt(dt['per_class'][0]['f1'])} and {fmt(rf['per_class'][0]['f1'])} F1 respectively).",
        ]
    if is_oulad:
        text += [
            "",
            "## Top-k risk ranking",
            "",
            "Tie-breaking is descending probability then ascending record ID; the budget is rounded upward.",
            "",
            "| Model | Budget | Precision@k | Recall@k | F1@k | NDCG@k |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in dataset["models"]:
            for item in row["top_k"]:
                text.append(
                    f"| {row['model']} | {int(item['budget'] * 100)}% | {fmt(item['precision'])} | {fmt(item['recall'])} | {fmt(item['f1'])} | {fmt(item['ndcg'])} |"
                )
    text += [
        "",
        "## Frozen confusion matrices",
        "",
        confusion_sections(dataset, selected_cm),
        "",
        "## Evidence sources",
        "",
        source_notes(dataset),
    ]
    return "\n".join(text) + "\n"


def recommendation_report(payload: dict[str, Any]) -> str:
    rec = payload["recommendation"]
    lines = [
        "# Student Risk-Based Recommendation System — Final Results",
        "",
        "This is a deterministic risk-based decision-support system, not a causal intervention claim.",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    for name, item in rec["metrics"].items():
        lines.append(
            f"| {name.replace('_', ' ').title()} | {fmt(item) if isinstance(value(item), (float, int)) and not isinstance(value(item), bool) else value(item)} |"
        )
    lines += [
        f"| Expert status | {value(rec['expert_status'])} |",
        f"| Causal effectiveness claimed | {value(rec['causal_effectiveness_claimed'])} |",
        "",
        "Expert-label metrics remain N/A until independent labels are supplied.",
    ]
    return "\n".join(lines) + "\n"


def generate() -> None:
    payload = json.loads(
        (FINAL_ROOT / "final_results.json").read_text(encoding="utf-8")
    )
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    names = {
        "student_mat": "STUDENT_MAT_RESULTS.md",
        "student_por": "STUDENT_POR_RESULTS.md",
        "oulad": "OULAD_RESULTS.md",
    }
    reports = {}
    for dataset_id, filename in names.items():
        reports[dataset_id] = dataset_report(
            dataset_id, payload["datasets"][dataset_id]
        )
        (REPORT_ROOT / filename).write_text(reports[dataset_id], encoding="utf-8")
    rec_report = recommendation_report(payload)
    (REPORT_ROOT / "RECOMMENDATION_RESULTS.md").write_text(rec_report, encoding="utf-8")
    imbalance = """# Imbalance Results

Only selected results with registered frozen final evidence are shown. The release does not imply that every listed imbalance method was run.

## UCI datasets

| Dataset | Model | Method | Macro-F1 | Low Precision | Low Recall | Low F1 | Balanced Accuracy |
|---|---|---|---:|---:|---:|---:|---:|
"""
    for dataset_id in ("student_mat", "student_por"):
        row = payload["datasets"][dataset_id]["models"][0]
        low = row["per_class"][0]
        imbalance += f"| {payload['datasets'][dataset_id]['dataset']} | CNN-BiLSTM | Registered selected policy | {fmt(row['metrics']['macro_f1'])} | {fmt(low['precision'])} | {fmt(low['recall'])} | {fmt(low['f1'])} | {fmt(row['metrics']['balanced_accuracy'])} |\n"
    imbalance += """
## OULAD

| Model | Method | Macro-F1 | Risk Precision | Risk Recall | Risk F1 | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
"""
    row = payload["datasets"]["oulad"]["models"][0]
    imbalance += f"| CNN-BiLSTM | Registered selected policy | {fmt(row['metrics']['macro_f1'])} | {fmt(row['metrics']['risk_precision'])} | {fmt(row['metrics']['risk_recall'])} | {fmt(row['metrics']['risk_f1'])} | {fmt(row['metrics']['pr_auc'])} |\n"
    (REPORT_ROOT / "IMBALANCE_RESULTS.md").write_text(imbalance, encoding="utf-8")
    claims = (
        "# Claim Boundaries\n\n"
        + "\n".join(f"- {item}" for item in payload["claim_boundaries"])
        + "\n\nMissing metrics remain N/A; no metric is estimated from screening evidence.\n"
    )
    (REPORT_ROOT / "CLAIM_BOUNDARIES.md").write_text(claims, encoding="utf-8")
    official = "\n".join(
        f"| {OFFICIAL_MODELS[key]['dataset']} | {OFFICIAL_MODELS[key]['official_name']} | {OFFICIAL_MODELS[key]['task']} |"
        for key in OFFICIAL_MODELS
    )
    combined = (
        "# Final Model Results\n\n## Official models\n\n| Dataset | Final model | Task |\n|---|---|---|\n"
        + official
    )
    for dataset_id, heading in (
        ("student_mat", "Student-Mat"),
        ("student_por", "Student-Por"),
        ("oulad", "OULAD"),
    ):
        combined += f"\n\n## {heading} overall comparison\n\n{overall_table(payload['datasets'][dataset_id], dataset_id == 'oulad')}\n\n## {heading} per-class comparison\n\n{per_class_table(payload['datasets'][dataset_id])}"
    combined += (
        "\n\n## OULAD Top-k\n\nSee `OULAD_RESULTS.md`; only rows backed by frozen probabilities contain values.\n\n## Imbalance\n\nSee `IMBALANCE_RESULTS.md`.\n\n## Recommendation\n\n"
        + rec_report
    )
    (REPORT_ROOT / "FINAL_MODEL_RESULTS.md").write_text(combined, encoding="utf-8")
    review = "# Final Project Review\n\nThe repository exposes three official CNN-BiLSTM models and one risk-based recommendation system. Canonical JSON, CSV, registry, dataset reports, checksums, and validation are synchronized from frozen evidence. Training is disabled in the final commands.\n\nVerdict is assigned only by `python project.py final validate`.\n"
    (REPORT_ROOT / "FINAL_PROJECT_REVIEW.md").write_text(review, encoding="utf-8")


if __name__ == "__main__":
    generate()
