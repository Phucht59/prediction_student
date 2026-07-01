"""Run leakage-safe final ML baselines for thesis comparison.

This script creates additive baseline artifacts. It does not overwrite existing
final deep-model artifacts or checkpoints.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports" / "final"
SEED = 42
CLASS_LABELS = [0, 1, 2]
STUDENT_BINS = [0, 9, 14, 20]


@dataclass(frozen=True)
class BaselineTask:
    dataset: str
    scenario: str
    raw_path: Path
    sep: str
    target: str
    kind: str
    feature_policy: str


TASKS = [
    BaselineTask("student-mat", "late", RAW_DIR / "student-mat.csv", ";", "G3", "student", "G1_G2"),
    BaselineTask("student-por", "late", RAW_DIR / "student-por.csv", ";", "G3", "student", "G1_G2"),
    BaselineTask("student-por", "midterm", RAW_DIR / "student-por.csv", ";", "G3", "student", "G1_only"),
    BaselineTask("xAPI", "default", RAW_DIR / "xAPI-Edu-Data.csv", ",", "Class", "xapi", "all_non_target"),
]

DEEP_FINAL = {
    ("student-mat", "late"): ("sequence_cnn_bilstm_only", "low_f1_tuned", 0.9365, 0.9615, 0.8929),
    ("student-por", "late"): ("sequence_cnn_bilstm_only", "low_f1_tuned", 0.8783, 0.9000, 0.8182),
    ("student-por", "midterm"): ("sequence_cnn_bilstm_only", "argmax", 0.8228, 0.6500, 0.7429),
    ("xAPI", "default"): ("gated_fusion_v28", "low_f1_tuned", 0.7541, 0.8846, 0.8214),
}


def load_task_frame(task: BaselineTask) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(task.raw_path, sep=task.sep)
    if task.kind == "student":
        y = pd.cut(df[task.target], bins=STUDENT_BINS, labels=CLASS_LABELS, include_lowest=True)
        keep = ~y.isna()
        df = df.loc[keep].copy()
        y = y.loc[keep].astype(int).to_numpy()
        if task.feature_policy == "G1_G2":
            feature_cols = ["G1", "G2"]
        elif task.feature_policy == "G1_only":
            feature_cols = ["G1"]
        else:
            raise ValueError(f"Unsupported student feature policy: {task.feature_policy}")
        return df[feature_cols].copy(), y, feature_cols

    if task.kind == "xapi":
        mapping = {"L": 0, "M": 1, "H": 2}
        y = df[task.target].map(mapping).astype(int).to_numpy()
        feature_cols = [column for column in df.columns if column != task.target]
        return df[feature_cols].copy(), y, feature_cols

    raise ValueError(f"Unsupported task kind: {task.kind}")


def make_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [column for column in x_train.columns if column not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, object]:
    return {
        "LogisticRegression_balanced": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=SEED,
        ),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "ExtraTrees_balanced": ExtraTreesClassifier(
            n_estimators=500,
            random_state=SEED,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=SEED,
        ),
    }


def align_probabilities(model: Pipeline, probabilities: np.ndarray) -> np.ndarray:
    classes = model.named_steps["model"].classes_
    aligned = np.zeros((probabilities.shape[0], len(CLASS_LABELS)), dtype=float)
    for source_index, cls in enumerate(classes):
        aligned[:, int(cls)] = probabilities[:, source_index]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sum, out=np.full_like(aligned, 1.0 / len(CLASS_LABELS)), where=row_sum > 0)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, object]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_low": recall_score(y_true == 0, y_pred == 0, zero_division=0),
        "f1_low": f1_score(y_true == 0, y_pred == 0, zero_division=0),
        "confusion_matrix": json.dumps(confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist()),
    }
    if probabilities is not None:
        clipped = np.clip(probabilities, 1e-12, 1.0)
        one_hot = np.eye(len(CLASS_LABELS))[y_true]
        metrics["brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
        metrics["nll"] = float(-np.mean(np.log(clipped[np.arange(len(y_true)), y_true])))
    else:
        metrics["brier"] = math.nan
        metrics["nll"] = math.nan
    return metrics


def choose_low_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, object]]:
    best_threshold = 0.5
    best_metrics: dict[str, object] | None = None
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = np.argmax(probabilities, axis=1)
        pred[probabilities[:, 0] >= threshold] = 0
        current = metric_dict(y_true, pred, probabilities)
        current_key = (current["f1_low"], current["recall_low"], current["macro_f1"], -threshold)
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = current
            continue
        best_key = (best_metrics["f1_low"], best_metrics["recall_low"], best_metrics["macro_f1"], -best_threshold)
        if current_key > best_key:
            best_threshold = float(threshold)
            best_metrics = current
    assert best_metrics is not None
    return best_threshold, best_metrics


def evaluate_candidate(task: BaselineTask, model_name: str, estimator: object, x_train: pd.DataFrame, y_train: np.ndarray):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probabilities = np.zeros((len(y_train), len(CLASS_LABELS)), dtype=float)
    fold_rows = []
    for fold, (fit_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
        x_fit = x_train.iloc[fit_idx]
        x_val = x_train.iloc[val_idx]
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(x_fit)),
                ("model", estimator),
            ]
        )
        pipe.fit(x_fit, y_train[fit_idx])
        probabilities = align_probabilities(pipe, pipe.predict_proba(x_val))
        oof_probabilities[val_idx] = probabilities
        pred = np.argmax(probabilities, axis=1)
        metrics = metric_dict(y_train[val_idx], pred, probabilities)
        fold_rows.append(
            {
                "dataset": task.dataset,
                "scenario": task.scenario,
                "model": model_name,
                "fold": fold,
                **{key: value for key, value in metrics.items() if key != "confusion_matrix"},
            }
        )
    oof_argmax = np.argmax(oof_probabilities, axis=1)
    oof_metrics = metric_dict(y_train, oof_argmax, oof_probabilities)
    threshold, tuned_metrics = choose_low_threshold(y_train, oof_probabilities)
    return fold_rows, oof_probabilities, oof_metrics, threshold, tuned_metrics


def run_task(task: BaselineTask):
    x, y, feature_cols = load_task_frame(task)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    cv_rows = []
    candidate_summaries = []
    candidate_cache = {}
    for model_name, estimator in candidate_models().items():
        fold_rows, oof_probs, oof_metrics, threshold, tuned_metrics = evaluate_candidate(
            task,
            model_name,
            estimator,
            x_train,
            y_train,
        )
        cv_rows.extend(fold_rows)
        candidate_cache[model_name] = (oof_probs, threshold)
        candidate_summaries.append(
            {
                "dataset": task.dataset,
                "scenario": task.scenario,
                "model": model_name,
                "feature_policy": task.feature_policy,
                "features": ",".join(feature_cols),
                "n_train_pool": len(y_train),
                "n_locked_test": len(y_test),
                "cv_macro_f1": oof_metrics["macro_f1"],
                "cv_recall_low": oof_metrics["recall_low"],
                "cv_f1_low": oof_metrics["f1_low"],
                "oof_low_threshold": threshold,
                "oof_tuned_macro_f1": tuned_metrics["macro_f1"],
                "oof_tuned_recall_low": tuned_metrics["recall_low"],
                "oof_tuned_f1_low": tuned_metrics["f1_low"],
            }
        )

    summary_df = pd.DataFrame(candidate_summaries)
    selected_row = summary_df.sort_values(
        ["cv_macro_f1", "cv_f1_low", "cv_recall_low"],
        ascending=False,
    ).iloc[0]
    selected_name = selected_row["model"]
    selected_threshold = float(selected_row["oof_low_threshold"])
    estimator = candidate_models()[selected_name]
    final_pipe = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(x_train)),
            ("model", estimator),
        ]
    )
    final_pipe.fit(x_train, y_train)
    test_probs = align_probabilities(final_pipe, final_pipe.predict_proba(x_test))
    argmax_pred = np.argmax(test_probs, axis=1)
    tuned_pred = argmax_pred.copy()
    tuned_pred[test_probs[:, 0] >= selected_threshold] = 0

    locked_rows = []
    for mode, pred, threshold in [
        ("argmax", argmax_pred, math.nan),
        ("low_f1_tuned", tuned_pred, selected_threshold),
    ]:
        metrics = metric_dict(y_test, pred, test_probs)
        locked_rows.append(
            {
                "dataset": task.dataset,
                "scenario": task.scenario,
                "selected_model": selected_name,
                "feature_policy": task.feature_policy,
                "features": ",".join(feature_cols),
                "selection_metric": "cv_macro_f1",
                "prediction_mode": mode,
                "threshold": threshold,
                "n_train_pool": len(y_train),
                "n_locked_test": len(y_test),
                **metrics,
            }
        )

    return cv_rows, candidate_summaries, locked_rows


def write_report(candidate_df: pd.DataFrame, locked_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        display = frame.copy()
        for column in display.columns:
            if pd.api.types.is_float_dtype(display[column]):
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
            else:
                display[column] = display[column].astype(str)
        headers = display.columns.tolist()
        rows = display.values.tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    lines = [
        "# Final Baseline Comparison Full Report",
        "",
        "This report is generated by `scripts/run_final_baseline_comparison.py`.",
        "",
        "Protocol:",
        "",
        "- Raw datasets: `data/raw/student-mat.csv`, `data/raw/student-por.csv`, `data/raw/xAPI-Edu-Data.csv`.",
        "- Locked test: 20%, stratified, seed 42.",
        "- Baseline selection: 5-fold CV on train pool using Macro F1.",
        "- Low-class threshold: tuned only from OOF probabilities on train pool.",
        "- No ADASYN and no SMOTE are used.",
        "- Baselines are comparison-only; they are not teachers, distillation sources, pseudo-label sources, or feature-importance sources for deep models.",
        "",
        "## Selected Baselines on Locked Test",
        "",
        markdown_table(locked_df[
            [
                "dataset",
                "scenario",
                "selected_model",
                "prediction_mode",
                "macro_f1",
                "recall_low",
                "f1_low",
                "balanced_accuracy",
            ]
        ]),
        "",
        "## Deep vs Selected Baselines",
        "",
        markdown_table(comparison_df[
            [
                "dataset",
                "scenario",
                "model_type",
                "model",
                "prediction_mode",
                "macro_f1",
                "recall_low",
                "f1_low",
                "interpretation",
            ]
        ]),
        "",
        "## Interpretation Guardrails",
        "",
        "- Do not claim deep learning beats all baselines unless the row shows it.",
        "- For xAPI, the existing final artifact reports RandomForest Macro F1 0.8465, higher than gated_fusion_v28 Macro F1 0.7541.",
        "- If a new xAPI deep model is proposed later, it must be compared against both this rerun baseline and the existing final xAPI RandomForest artifact.",
        "- Student Performance remains the thesis core; xAPI is an auxiliary benchmark.",
        "",
        "## Candidate CV Summary",
        "",
        markdown_table(candidate_df[
            [
                "dataset",
                "scenario",
                "model",
                "cv_macro_f1",
                "cv_recall_low",
                "cv_f1_low",
                "oof_low_threshold",
            ]
        ]),
        "",
    ]
    (REPORT_DIR / "final_baseline_comparison_full_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cv_rows = []
    candidate_rows = []
    locked_rows = []
    for task in TASKS:
        task_cv, task_candidates, task_locked = run_task(task)
        cv_rows.extend(task_cv)
        candidate_rows.extend(task_candidates)
        locked_rows.extend(task_locked)

    cv_df = pd.DataFrame(cv_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    locked_df = pd.DataFrame(locked_rows)

    comparison_rows = []
    for (dataset, scenario), (deep_model, deep_mode, macro_f1, recall_low, f1_low) in DEEP_FINAL.items():
        comparison_rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "model_type": "deep_final",
                "model": deep_model,
                "prediction_mode": deep_mode,
                "macro_f1": macro_f1,
                "recall_low": recall_low,
                "f1_low": f1_low,
                "interpretation": "final deep artifact",
            }
        )
        baseline_rows = locked_df[
            (locked_df["dataset"].str.lower() == dataset.lower())
            & (locked_df["scenario"] == scenario)
        ]
        for _, row in baseline_rows.iterrows():
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "model_type": "baseline_rerun",
                    "model": row["selected_model"],
                    "prediction_mode": row["prediction_mode"],
                    "macro_f1": row["macro_f1"],
                    "recall_low": row["recall_low"],
                    "f1_low": row["f1_low"],
                    "interpretation": "rerun baseline selected by CV Macro F1",
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)

    cv_df.to_csv(REPORT_DIR / "final_baseline_all_cv_folds.csv", index=False)
    candidate_df.to_csv(REPORT_DIR / "final_baseline_candidate_summary.csv", index=False)
    locked_df.to_csv(REPORT_DIR / "final_baseline_locked_test_results.csv", index=False)
    comparison_df.to_csv(REPORT_DIR / "final_deep_vs_baseline_comparison_full.csv", index=False)
    write_report(candidate_df, locked_df, comparison_df)


if __name__ == "__main__":
    main()
