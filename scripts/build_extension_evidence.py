from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.common.hashing import sha256_file
from src.studies.oulad.evaluate import binary_metrics


FORECAST_ORDER = ["F1_EARLY", "F2_MIDDLE", "F3_LATE"]
COMPARISONS = [("C-H2", "C-L0"), ("C-H1", "C-L1"), ("C-H1", "C-C0"), ("C-H2", "C-H1"), ("C-H2", "C-M0")]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def class_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["candidate_id", "forecast_id", "scope", "seed"]):
        p, r, f, support = precision_recall_fscore_support(group["true_label"], group["predicted_label"], labels=[0, 1], zero_division=0)
        for index, name in enumerate(["not_at_risk", "at_risk"]):
            rows.append(dict(zip(["candidate_id", "forecast_id", "scope", "seed"], keys)) | {"class_name": name, "precision": p[index], "recall": r[index], "f1": f[index], "support": int(support[index])})
    return pd.DataFrame(rows)


def paired_deltas(predictions: pd.DataFrame, bootstrap_samples: int = 200) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for (scope, forecast), frame in predictions.groupby(["scope", "forecast_id"]):
        truth = frame.drop_duplicates("record_id").set_index("record_id")["true_label"]
        pivot = frame.pivot_table(index="record_id", columns="candidate_id", values="predicted_label", aggfunc="first")
        for left, right in COMPARISONS:
            if left not in pivot or right not in pivot: continue
            common = pivot[[left, right]].dropna().index
            y = truth.loc[common].to_numpy(int); a = pivot.loc[common, left].to_numpy(int); b = pivot.loc[common, right].to_numpy(int)
            delta = f1_score(y, a, average="macro", zero_division=0) - f1_score(y, b, average="macro", zero_division=0)
            boot = []
            for _ in range(bootstrap_samples):
                sample = rng.integers(0, len(y), len(y))
                boot.append(f1_score(y[sample], a[sample], average="macro", zero_division=0) - f1_score(y[sample], b[sample], average="macro", zero_division=0))
            rows.append({"scope": scope, "forecast_id": forecast, "left": left, "right": right, "macro_f1_delta": delta, "bootstrap_low": float(np.quantile(boot, 0.025)), "bootstrap_high": float(np.quantile(boot, 0.975)), "records": len(common), "descriptive_only": True})
    return pd.DataFrame(rows)


def grouped_metrics(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.groupby(["candidate_id", "forecast_id", "scope", group_column]):
        metric = binary_metrics(group["true_label"].to_numpy(int), group["probability_at_risk"].to_numpy(float), "record_specific", group["predicted_label"].to_numpy(int))
        rows.append(dict(zip(["candidate_id", "forecast_id", "scope", group_column], keys)) | metric)
    return pd.DataFrame(rows)


def save_figure(fig, base: Path) -> None:
    fig.tight_layout(); fig.savefig(base.with_suffix(".png"), dpi=160); fig.savefig(base.with_suffix(".svg")); plt.close(fig)


def figures(artifact: Path, metrics: pd.DataFrame, predictions: pd.DataFrame, cohort_flow: pd.DataFrame, module_metrics: pd.DataFrame) -> None:
    figure_root = artifact / "figures"; figure_root.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(cohort_flow["forecast_id"], cohort_flow["at_risk"] / (cohort_flow["at_risk"] + cohort_flow["not_at_risk"])); ax.set_ylabel("At-risk prevalence"); ax.set_ylim(0, 1); save_figure(fig, figure_root / "target_distribution_by_forecast")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(cohort_flow["forecast_id"], cohort_flow["not_at_risk"], label="Not at risk")
    ax.bar(cohort_flow["forecast_id"], cohort_flow["at_risk"], bottom=cohort_flow["not_at_risk"], label="At risk")
    ax.set_ylabel("Learner-module records"); ax.legend(); save_figure(fig, figure_root / "cohort_flow")
    development = metrics[metrics["scope"] == "development_oof"]
    for column, name in [("macro_f1", "model_macro_f1_by_forecast"), ("at_risk_recall", "at_risk_recall_by_forecast"), ("pr_auc", "pr_auc_by_forecast")]:
        pivot = development.pivot(index="candidate_id", columns="forecast_id", values=column).reindex(columns=FORECAST_ORDER)
        fig, ax = plt.subplots(figsize=(10, 5)); pivot.plot(kind="bar", ax=ax); ax.set_ylabel(column); ax.set_ylim(0, 1); save_figure(fig, figure_root / name)
    delta = []
    for forecast in FORECAST_ORDER:
        rows = development[development.forecast_id == forecast].set_index("candidate_id")
        best_ml = rows.loc[[candidate for candidate in ["C-L0", "C-R0", "C-H0"] if candidate in rows.index], "macro_f1"].max()
        delta.append(float(rows.loc["C-H2", "macro_f1"] - best_ml))
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(FORECAST_ORDER, delta); ax.axhline(0.01, color="red", linestyle="--", label="practical margin"); ax.axhline(0, color="black", linewidth=0.8); ax.legend(); ax.set_ylabel("C-H2 minus best ML Macro-F1"); save_figure(fig, figure_root / "deep_vs_ml_delta")
    flagship = predictions[(predictions.candidate_id == "C-H2") & (predictions.forecast_id == "F2_MIDDLE") & (predictions.scope == "future_presentation")]
    matrix = confusion_matrix(flagship.true_label, flagship.predicted_label, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(matrix, cmap="Blues");
    for i in range(2):
        for j in range(2): ax.text(j, i, int(matrix[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["Not at risk", "At risk"]); ax.set_yticks([0, 1], ["Not at risk", "At risk"]); ax.set_xlabel("Predicted"); ax.set_ylabel("True"); save_figure(fig, figure_root / "confusion_matrix_flagship")
    stability = module_metrics[(module_metrics.candidate_id.isin(["C-L0", "C-H2"])) & (module_metrics.scope == "future_presentation") & (module_metrics.forecast_id == "F2_MIDDLE")]
    pivot = stability.pivot(index="code_module", columns="candidate_id", values="macro_f1")
    fig, ax = plt.subplots(figsize=(8, 4)); pivot.plot(kind="bar", ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1"); save_figure(fig, figure_root / "module_stability")
    future = metrics[metrics.scope == "future_presentation"].pivot(index="candidate_id", columns="forecast_id", values="macro_f1").reindex(columns=FORECAST_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5)); future.plot(kind="bar", ax=ax); ax.set_ylim(0, 1); ax.set_ylabel("Future-presentation Macro-F1"); save_figure(fig, figure_root / "future_presentation_comparison")
    learning_path = artifact / "learning_curves.csv"
    if learning_path.exists():
        learning = pd.read_csv(learning_path); subset = learning[(learning.candidate_id == "C-H2") & (learning.forecast_id == "F2_MIDDLE") & (learning.inner_fold.astype(str) != "refit")]
        subset = subset.dropna(subset=["validation_nll"])
        fig, ax = plt.subplots(figsize=(7, 4)); subset.groupby("epoch")["validation_nll"].mean().plot(ax=ax); ax.set_ylabel("Mean inner validation NLL"); save_figure(fig, figure_root / "learning_curves_flagship")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-b-run", required=True)
    parser.add_argument("--study-c-run", required=True)
    args = parser.parse_args()
    b = ROOT / "artifacts" / "study_b_student_por" / args.study_b_run
    c = ROOT / "artifacts" / "study_c_oulad" / args.study_c_run
    oof = pd.read_parquet(c / "oof_predictions.parquet"); future = pd.read_parquet(c / "future_predictions.parquet"); predictions = pd.concat([oof, future], ignore_index=True)
    classes = class_metrics(predictions); classes.to_csv(c / "class_metrics_by_model_forecast.csv", index=False)
    paired = paired_deltas(predictions); paired.to_csv(c / "paired_deltas.csv", index=False)
    module = grouped_metrics(predictions, "code_module"); module.to_csv(c / "module_metrics.csv", index=False)
    presentation = grouped_metrics(predictions, "code_presentation"); presentation.to_csv(c / "presentation_metrics.csv", index=False)
    selected = pd.read_csv(c / "selected_configs.csv")
    selected.to_json(c / "selected_configs.json", orient="records", indent=2)
    pd.read_csv(b / "selected_configs.csv").to_json(b / "selected_configs.json", orient="records", indent=2)
    selected.groupby("candidate_id")["parameter_count"].agg(["min", "median", "max"]).reset_index().to_csv(c / "parameter_counts.csv", index=False)
    learning_directory = c / "learning_curves"; learning_directory.mkdir(exist_ok=True)
    learning = pd.read_csv(c / "learning_curves.csv")
    for candidate_id, frame in learning.groupby("candidate_id"):
        frame.to_csv(learning_directory / f"{candidate_id}.csv", index=False)
    confusion_root = c / "confusion_matrices"; confusion_root.mkdir(exist_ok=True)
    for keys, group in predictions.groupby(["candidate_id", "forecast_id", "scope"]):
        write_json(confusion_root / f"{keys[0]}_{keys[1]}_{keys[2]}.json", {"labels": ["not_at_risk", "at_risk"], "matrix": confusion_matrix(group.true_label, group.predicted_label, labels=[0, 1]).tolist(), "records": len(group)})
    cohort_flow = pd.read_csv(c / "cohort_flow.csv"); cohort_flow.to_csv(c / "cohort_by_forecast.csv", index=False)
    cohort_flow[["forecast_id", "at_risk", "not_at_risk"]].assign(prevalence=lambda x: x.at_risk / (x.at_risk + x.not_at_risk)).to_csv(c / "class_distribution.csv", index=False)
    metrics = pd.read_csv(c / "metrics_by_model_forecast.csv")
    figures(c, metrics, predictions, cohort_flow, module)
    provenance = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "protocol_commit": "a00acaa", "study_a_mutated": False, "legacy_observed_accessed": False, "raw_manifest": "data/manifests/extension_raw_manifest.json"}
    write_json(c / "source_provenance.json", provenance); write_json(b / "source_provenance.json", provenance)
    for artifact in [b, c]:
        entries = []
        for path in sorted(
            item for item in artifact.rglob("*")
            if item.is_file() and item.name not in {"artifact_checksums.json", "validation_report.json"}
        ):
            entries.append({"path": path.relative_to(artifact).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        write_json(artifact / "artifact_checksums.json", {"entries": entries, "files": len(entries), "all_hashed": True})
    # Refresh report mirrors after all compact evidence is complete; large checkpoints/parquet stay artifact-only.
    for artifact in [b, c]:
        report = ROOT / "reports" / artifact.parent.name / artifact.name
        report.mkdir(parents=True, exist_ok=True)
        for path in artifact.rglob("*"):
            if path.is_file() and path.suffix not in {".pt", ".pkl", ".parquet"}:
                destination = report / path.relative_to(artifact); destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    print(json.dumps({"status": "PASS", "study_b": str(b), "study_c": str(c), "paired_rows": len(paired), "module_rows": len(module)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
