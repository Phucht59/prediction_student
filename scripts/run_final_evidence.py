"""Create a reproducible, prediction-backed evidence bundle for student-mat.

This runner is deliberately independent of the thesis DOCX and of historical
hand-written result tables.  Model selection uses only OOF predictions from the
80% train pool.  The deterministic 20% locked split is scored after the full
set of predeclared baseline/scenario experiments has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evidence_metrics import (
    CLASS_LABELS,
    CLASS_NAMES,
    bootstrap_confidence_intervals,
    classification_metrics,
    reliability_rows,
)
from src.recommendation import (
    build_recommendation,
    structural_validity_metrics,
    validate_recommendation_schema,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "student-mat.csv"
PROCESSED_DIR = ROOT / "data" / "processed" / "final"
ARTIFACT_ROOT = ROOT / "artifacts" / "final"
SEED = 42
TARGET_BINS = [0, 9, 14, 20]
SENSITIVE_SLICE_COLUMNS = ("sex", "school", "address")


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    scenario: str
    estimator: str
    feature_policy: str
    imbalance_policy: str = "none"


EXPERIMENTS = (
    ExperimentSpec("majority_class", "late_stage", "majority", "none"),
    ExperimentSpec("g2_threshold_rule", "late_stage", "g2_rule", "G2"),
    ExperimentSpec("logistic_regression_g2", "late_stage", "logistic", "G2"),
    ExperimentSpec("logistic_regression_g1_g2", "late_stage", "logistic", "G1_G2"),
    ExperimentSpec("logistic_regression_all", "late_stage", "logistic", "all_valid"),
    ExperimentSpec("logistic_regression_all_balanced", "late_stage", "logistic", "all_valid", "class_weight"),
    ExperimentSpec("hist_gradient_boosting_all", "late_stage", "histgb", "all_valid"),
    ExperimentSpec("logistic_regression_without_g2", "early_warning", "logistic", "without_G2"),
    ExperimentSpec("hist_gradient_boosting_without_g2", "early_warning", "histgb", "without_G2"),
    ExperimentSpec("logistic_regression_without_g1_g2", "pre_assessment", "logistic", "without_G1_G2"),
    ExperimentSpec("hist_gradient_boosting_without_g1_g2", "pre_assessment", "histgb", "without_G1_G2"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True, encoding="utf-8"
    ).stdout.strip()


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(RAW_PATH, sep=";")
    frame.insert(0, "__source_row_number", np.arange(len(frame), dtype=int))
    frame["target_class"] = pd.cut(
        frame["G3"], bins=TARGET_BINS, labels=list(CLASS_LABELS), include_lowest=True
    ).astype(int)
    return frame


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, locked = train_test_split(
        frame,
        test_size=0.20,
        random_state=SEED,
        stratify=frame["target_class"],
    )
    return train.reset_index(drop=True), locked.reset_index(drop=True)


def feature_columns(frame: pd.DataFrame, policy: str) -> list[str]:
    forbidden = {"G3", "target_class", "__source_row_number"}
    if policy == "none":
        return []
    if policy == "G2":
        return ["G2"]
    if policy == "G1_G2":
        return ["G1", "G2"]
    if policy == "all_valid":
        return [column for column in frame.columns if column not in forbidden]
    if policy == "without_G2":
        return [column for column in frame.columns if column not in forbidden | {"G2"}]
    if policy == "without_G1_G2":
        return [column for column in frame.columns if column not in forbidden | {"G1", "G2"}]
    raise ValueError(f"Unknown feature policy: {policy}")


def make_preprocessor(frame: pd.DataFrame, columns: list[str]) -> ColumnTransformer:
    numeric = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in columns if column not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("numeric", StandardScaler(), numeric), ("categorical", encoder, categorical)],
        remainder="drop",
    )


def make_pipeline(spec: ExperimentSpec, fit_frame: pd.DataFrame, columns: list[str]) -> Pipeline:
    if spec.estimator == "logistic":
        model = LogisticRegression(
            max_iter=5000,
            random_state=SEED,
            class_weight="balanced" if spec.imbalance_policy == "class_weight" else None,
        )
    elif spec.estimator == "histgb":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=0.05,
            random_state=SEED,
        )
    else:
        raise ValueError(f"No sklearn pipeline for estimator {spec.estimator}.")
    return Pipeline([("preprocess", make_preprocessor(fit_frame, columns)), ("model", model)])


def aligned_probabilities(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(frame)
    classes = model.named_steps["model"].classes_
    output = np.zeros((len(frame), len(CLASS_LABELS)), dtype=float)
    for source, label in enumerate(classes):
        output[:, int(label)] = raw[:, source]
    return output / output.sum(axis=1, keepdims=True)


def rule_probabilities(
    spec: ExperimentSpec, fit_frame: pd.DataFrame, score_frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if spec.estimator == "majority":
        counts = fit_frame["target_class"].value_counts().reindex(CLASS_LABELS, fill_value=0).to_numpy(dtype=float)
        probabilities = np.repeat((counts / counts.sum())[None, :], len(score_frame), axis=0)
        predictions = np.repeat(int(np.argmax(counts)), len(score_frame))
        return probabilities, predictions
    if spec.estimator == "g2_rule":
        values = score_frame["G2"].to_numpy()
        predictions = np.where(values <= 9, 0, np.where(values <= 14, 1, 2)).astype(int)
        probabilities = np.eye(len(CLASS_LABELS), dtype=float)[predictions]
        return probabilities, predictions
    raise ValueError(f"Unknown rule estimator: {spec.estimator}")


def fit_predict(
    spec: ExperimentSpec, fit_frame: pd.DataFrame, score_frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    if spec.estimator in {"majority", "g2_rule"}:
        return rule_probabilities(spec, fit_frame, score_frame)
    columns = feature_columns(fit_frame, spec.feature_policy)
    pipeline = make_pipeline(spec, fit_frame, columns)
    pipeline.fit(fit_frame[columns], fit_frame["target_class"].to_numpy())
    probabilities = aligned_probabilities(pipeline, score_frame[columns])
    return probabilities, probabilities.argmax(axis=1)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    scaled = np.exp(np.log(clipped) / float(temperature))
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    def objective(log_temperature: float) -> float:
        scaled = temperature_scale(probabilities, np.exp(log_temperature))
        return float(-np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
    return float(np.exp(result.x))


def flat_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"confusion_matrix", "per_class", "pr_auc_ovr"}
    }


def evaluate_oof(
    spec: ExperimentSpec,
    train_pool: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    probabilities = np.zeros((len(train_pool), len(CLASS_LABELS)), dtype=float)
    predictions = np.zeros(len(train_pool), dtype=int)
    fold_ids = np.zeros(len(train_pool), dtype=int)
    fold_rows: list[dict[str, Any]] = []
    labels = train_pool["target_class"].to_numpy(dtype=int)
    for fold_index, (fit_index, validation_index) in enumerate(folds):
        fold_probs, fold_pred = fit_predict(
            spec,
            train_pool.iloc[fit_index].copy(),
            train_pool.iloc[validation_index].copy(),
        )
        probabilities[validation_index] = fold_probs
        predictions[validation_index] = fold_pred
        fold_ids[validation_index] = fold_index
        fold_metrics = classification_metrics(labels[validation_index], fold_pred, fold_probs)
        fold_rows.append({"model": spec.name, "scenario": spec.scenario, "fold": fold_index, **flat_metrics(fold_metrics)})
    metrics = classification_metrics(labels, predictions, probabilities)
    temperature = 1.0 if spec.estimator in {"majority", "g2_rule"} else fit_temperature(probabilities, labels)
    calibrated = temperature_scale(probabilities, temperature)
    calibrated_metrics = classification_metrics(labels, calibrated.argmax(axis=1), calibrated)
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "fold_ids": fold_ids,
        "fold_rows": fold_rows,
        "metrics": metrics,
        "temperature": temperature,
        "calibrated_probabilities": calibrated,
        "calibrated_metrics": calibrated_metrics,
    }


def pr_rows(model: str, split: str, labels: np.ndarray, probabilities: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        binary = (labels == class_index).astype(int)
        precision, recall, thresholds = precision_recall_curve(binary, probabilities[:, class_index])
        padded = list(thresholds) + [None]
        rows.extend(
            {
                "model": model,
                "split": split,
                "class_label": class_index,
                "class_name": class_name,
                "precision": float(p),
                "recall": float(r),
                "threshold": None if threshold is None else float(threshold),
            }
            for p, r, threshold in zip(precision, recall, padded)
        )
    return rows


def fairness_rows(model: str, locked: pd.DataFrame, predictions: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = locked["target_class"].to_numpy(dtype=int)
    for column in SENSITIVE_SLICE_COLUMNS:
        for group, positions in locked.groupby(column).groups.items():
            index = np.asarray(list(positions), dtype=int)
            metrics = classification_metrics(labels[index], predictions[index])
            rows.append(
                {
                    "model": model,
                    "slice_column": column,
                    "slice_value": str(group),
                    "support": int(len(index)),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "ordinal_mae": metrics["ordinal_mae"],
                    "interpretation": "insufficient_support" if len(index) < 30 else "descriptive_only",
                }
            )
    return rows


def recommendation_evidence(
    locked: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    *,
    source_model: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    confidences = probabilities.max(axis=1)
    for position, record in enumerate(locked.to_dict("records")):
        payload = build_recommendation(record, int(predictions[position]), float(confidences[position]))
        payload["disclaimer"] = "Advisory support only; a teacher or advisor must review before action."
        validate_recommendation_schema(payload)
        payloads.append(payload)
        if len(cases) < 12:
            cases.append(
                {
                    "source_row_number": int(record["__source_row_number"]),
                    "source_model": source_model,
                    "predicted_class": int(predictions[position]),
                    "confidence": float(confidences[position]),
                    "recommendation": json.dumps(payload, ensure_ascii=False),
                    "expert_rating": None,
                }
            )
    metrics = structural_validity_metrics(payloads)
    metrics.update(
        {
            "source_model": source_model,
            "policy_type": "deterministic_rule_based_advisory",
            "expert_evaluation": "not_collected",
            "causal_effectiveness_claimed": False,
            "fallback_rate": float(
                np.mean(
                    [
                        any(risk.get("code") == "prediction_monitoring" for risk in payload["priority_risks"])
                        for payload in payloads
                    ]
                )
            ),
        }
    )
    return metrics, cases


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.4f}")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc)
    run_id = args.run_id or f"student-mat-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{git_output('rev-parse', '--short', 'HEAD')}"
    output_dir = ARTIFACT_ROOT / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Evidence directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame()
    train_pool, locked = split_frame(frame)
    folds = list(
        StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED).split(
            train_pool, train_pool["target_class"]
        )
    )

    all_results: dict[str, Any] = {}
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    locked_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []
    pr_curve_rows: list[dict[str, Any]] = []
    fairness: list[dict[str, Any]] = []
    confusion: list[dict[str, Any]] = []

    for spec in EXPERIMENTS:
        oof = evaluate_oof(spec, train_pool, folds)
        fold_rows.extend(oof["fold_rows"])
        test_probabilities, test_predictions = fit_predict(spec, train_pool, locked)
        calibrated_test_probabilities = temperature_scale(test_probabilities, oof["temperature"])
        calibrated_test_predictions = calibrated_test_probabilities.argmax(axis=1)
        test_metrics = classification_metrics(
            locked["target_class"].to_numpy(dtype=int), test_predictions, test_probabilities
        )
        calibrated_test_metrics = classification_metrics(
            locked["target_class"].to_numpy(dtype=int),
            calibrated_test_predictions,
            calibrated_test_probabilities,
        )
        confidence_intervals = bootstrap_confidence_intervals(
            locked["target_class"].to_numpy(dtype=int),
            test_predictions,
            n_resamples=args.bootstrap_resamples,
            seed=SEED,
        )
        all_results[spec.name] = {
            "spec": asdict(spec),
            "features": feature_columns(frame, spec.feature_policy),
            "oof_metrics": oof["metrics"],
            "oof_calibrated_metrics": oof["calibrated_metrics"],
            "temperature": oof["temperature"],
            "locked_test_metrics": test_metrics,
            "locked_test_calibrated_metrics": calibrated_test_metrics,
            "locked_test_confidence_intervals": confidence_intervals,
        }
        per_fold_macro_f1 = [row["macro_f1"] for row in oof["fold_rows"]]
        summary_rows.append(
            {
                "model": spec.name,
                "scenario": spec.scenario,
                "feature_policy": spec.feature_policy,
                "imbalance_policy": spec.imbalance_policy,
                "oof_macro_f1": oof["metrics"]["macro_f1"],
                "outer_fold_macro_f1_mean": float(np.mean(per_fold_macro_f1)),
                "outer_fold_macro_f1_std": float(np.std(per_fold_macro_f1, ddof=1)),
                "locked_accuracy": test_metrics["accuracy"],
                "locked_macro_f1": test_metrics["macro_f1"],
                "locked_weighted_f1": test_metrics["weighted_f1"],
                "locked_balanced_accuracy": test_metrics["balanced_accuracy"],
                "locked_qwk": test_metrics["quadratic_weighted_kappa"],
                "locked_ordinal_mae": test_metrics["ordinal_mae"],
                "locked_two_step_errors": test_metrics["two_step_errors"],
                "locked_pr_auc_macro": test_metrics["pr_auc_macro"],
                "locked_brier": test_metrics["multiclass_brier_score"],
                "locked_ece": test_metrics["ece"],
            }
        )
        labels_train = train_pool["target_class"].to_numpy(dtype=int)
        for position in range(len(train_pool)):
            oof_rows.append(
                {
                    "model": spec.name,
                    "scenario": spec.scenario,
                    "source_row_number": int(train_pool.iloc[position]["__source_row_number"]),
                    "outer_fold": int(oof["fold_ids"][position]),
                    "true_label": int(labels_train[position]),
                    "predicted_label": int(oof["predictions"][position]),
                    "prob_low": float(oof["probabilities"][position, 0]),
                    "prob_medium": float(oof["probabilities"][position, 1]),
                    "prob_high": float(oof["probabilities"][position, 2]),
                }
            )
        labels_test = locked["target_class"].to_numpy(dtype=int)
        for position in range(len(locked)):
            locked_rows.append(
                {
                    "model": spec.name,
                    "scenario": spec.scenario,
                    "source_row_number": int(locked.iloc[position]["__source_row_number"]),
                    "true_label": int(labels_test[position]),
                    "predicted_label": int(test_predictions[position]),
                    "confidence": float(test_probabilities[position].max()),
                    "prob_low": float(test_probabilities[position, 0]),
                    "prob_medium": float(test_probabilities[position, 1]),
                    "prob_high": float(test_probabilities[position, 2]),
                }
            )
        calibration_rows.append(
            {
                "model": spec.name,
                "temperature_fit_source": "train_pool_oof",
                "temperature": oof["temperature"],
                "oof_brier_before": oof["metrics"]["multiclass_brier_score"],
                "oof_brier_after": oof["calibrated_metrics"]["multiclass_brier_score"],
                "oof_ece_before": oof["metrics"]["ece"],
                "oof_ece_after": oof["calibrated_metrics"]["ece"],
                "locked_brier_before": test_metrics["multiclass_brier_score"],
                "locked_brier_after": calibrated_test_metrics["multiclass_brier_score"],
                "locked_ece_before": test_metrics["ece"],
                "locked_ece_after": calibrated_test_metrics["ece"],
            }
        )
        for row in reliability_rows(labels_test, test_probabilities):
            reliability.append({"model": spec.name, "calibration": "before", **row})
        for row in reliability_rows(labels_test, calibrated_test_probabilities):
            reliability.append({"model": spec.name, "calibration": "after_oof_temperature", **row})
        pr_curve_rows.extend(pr_rows(spec.name, "locked_test", labels_test, test_probabilities))
        fairness.extend(fairness_rows(spec.name, locked, test_predictions))
        for true_label, values in enumerate(test_metrics["confusion_matrix"]):
            for predicted_label, count in enumerate(values):
                confusion.append(
                    {
                        "model": spec.name,
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "count": count,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    selected_by_scenario: dict[str, str] = {}
    for scenario, rows in summary.groupby("scenario"):
        selected = rows.sort_values(
            ["oof_macro_f1", "outer_fold_macro_f1_std", "model"],
            ascending=[False, True, True],
        ).iloc[0]
        selected_by_scenario[str(scenario)] = str(selected["model"])

    late_model = selected_by_scenario["late_stage"]
    late_locked = pd.DataFrame(locked_rows)
    late_locked = late_locked[late_locked["model"] == late_model].sort_values("source_row_number")
    locked_sorted = locked.sort_values("__source_row_number").reset_index(drop=True)
    recommendation_metrics, recommendation_cases = recommendation_evidence(
        locked_sorted,
        late_locked["predicted_label"].to_numpy(dtype=int),
        late_locked[["prob_low", "prob_medium", "prob_high"]].to_numpy(dtype=float),
        source_model=late_model,
    )

    split_manifest = {
        "protocol": "stratified_80_20_locked_test",
        "seed": SEED,
        "target_definition": "Low: G3 <= 9; Medium: 10 <= G3 <= 14; High: G3 >= 15",
        "train_rows": train_pool["__source_row_number"].astype(int).tolist(),
        "locked_test_rows": locked["__source_row_number"].astype(int).tolist(),
        "train_hash": sha256_json(sorted(train_pool["__source_row_number"].astype(int).tolist())),
        "locked_test_hash": sha256_json(sorted(locked["__source_row_number"].astype(int).tolist())),
    }
    dataset_manifest = {
        "dataset": "UCI Student Performance student-mat",
        "source_file": str(RAW_PATH.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(RAW_PATH),
        "row_count": len(frame),
        "target_distribution": frame["target_class"].value_counts().sort_index().to_dict(),
        "ingestion_contract_hash": (
            json.loads((PROCESSED_DIR / "student-mat_3class_split_manifest.json").read_text(encoding="utf-8")).get("ingestion_contract_hash")
            if (PROCESSED_DIR / "student-mat_3class_split_manifest.json").exists()
            else None
        ),
    }
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    selected_config = {
        "scope": "predeclared_classical_baselines_and_information_scenarios",
        "selection_source": "train_pool_oof_only",
        "locked_test_used_for_selection": False,
        "cv_folds": args.cv_folds,
        "cv_seed": SEED,
        "experiments": [asdict(spec) for spec in EXPERIMENTS],
        "selected_by_scenario": selected_by_scenario,
        "calibration": "temperature fitted on train-pool OOF probabilities only",
        "note": "Deep-model selection is intentionally not claimed by this baseline evidence runner.",
    }

    pd.DataFrame(fold_rows).to_csv(output_dir / "outer_fold_metrics.csv", index=False)
    write_json(output_dir / "outer_fold_metrics.json", fold_rows)
    pd.DataFrame(oof_rows).to_csv(output_dir / "oof_predictions.csv", index=False)
    pd.DataFrame(locked_rows).to_csv(output_dir / "locked_test_predictions.csv", index=False)
    summary.to_csv(output_dir / "baseline_results.csv", index=False)
    summary.assign(ablation_axis="feature_information_and_class_weight").to_csv(output_dir / "ablation_results.csv", index=False)
    pd.DataFrame(confusion).to_csv(output_dir / "confusion_matrix.csv", index=False)
    write_json(output_dir / "classification_report.json", all_results)
    pd.DataFrame(pr_curve_rows).to_csv(output_dir / "pr_curve_data.csv", index=False)
    write_json(output_dir / "calibration_metrics.json", calibration_rows)
    pd.DataFrame(reliability).to_csv(output_dir / "reliability_diagram_data.csv", index=False)
    pd.DataFrame(fairness).to_csv(output_dir / "fairness_slices.csv", index=False)
    write_json(output_dir / "recommendation_evaluation.json", recommendation_metrics)
    pd.DataFrame(recommendation_cases).to_csv(output_dir / "recommendation_expert_review_cases.csv", index=False)
    write_json(output_dir / "dataset_manifest.json", dataset_manifest)
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(
        output_dir / "split_hashes.json",
        {"train_hash": split_manifest["train_hash"], "locked_test_hash": split_manifest["locked_test_hash"]},
    )
    write_json(output_dir / "selected_config.json", selected_config)

    readme = "\n".join(
        [
            f"# Evidence run `{run_id}`",
            "",
            "All result values below are generated from saved prediction rows; no value is copied from the thesis report.",
            "",
            "## Protocol",
            "",
            "- Dataset: student-mat, 395 records.",
            "- Split: stratified 80/20, seed 42; locked test is not used for model or calibration selection.",
            "- Model/scenario selection: five-fold OOF Macro-F1 on the train pool.",
            "- Probability calibration: temperature fitted on OOF probabilities only.",
            "- `late_stage` includes G2; `early_warning` excludes G2; `pre_assessment` excludes G1 and G2.",
            "",
            "## Results",
            "",
            markdown_table(
                summary,
                [
                    "model",
                    "scenario",
                    "oof_macro_f1",
                    "outer_fold_macro_f1_mean",
                    "outer_fold_macro_f1_std",
                    "locked_accuracy",
                    "locked_macro_f1",
                    "locked_ordinal_mae",
                ],
            ),
            "",
            "## Interpretation",
            "",
            "The G2 threshold rule is an explicitly required baseline. If it outperforms a trained model, that adverse result must remain visible.",
            "Fairness slices are descriptive only; groups with fewer than 30 locked-test records are marked as insufficient for conclusions.",
            "Recommendation evaluation is structural/offline and has no invented expert rating or causal-effectiveness claim.",
            "",
            "Recreate with:",
            "",
            "```powershell",
            "py -3.10 scripts/run_final_evidence.py",
            "```",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    artifact_checksums = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"run_manifest.json", "model_checksums.json"}
    }
    write_json(output_dir / "model_checksums.json", {"checkpoints": {}, "note": "Baseline evidence uses refittable sklearn estimators; no binary model is committed.", "artifact_checksums": artifact_checksums})
    run_manifest = {
        "run_id": run_id,
        "timestamp_utc": timestamp.isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "working_tree_state": "dirty" if git_output("status", "--porcelain") else "clean",
        "dataset_checksum": dataset_manifest["sha256"],
        "ingestion_contract_hash": dataset_manifest["ingestion_contract_hash"],
        "feature_sets": {spec.name: feature_columns(frame, spec.feature_policy) for spec in EXPERIMENTS},
        "target_definition": split_manifest["target_definition"],
        "seed": SEED,
        "fold_definitions": {"n_splits": args.cv_folds, "shuffle": True, "random_state": SEED},
        "selected_config": selected_config,
        "model_parameter_count": {spec.name: "not_applicable_or_refit_from_config" for spec in EXPERIMENTS},
        "package_versions": environment,
        "split_hashes": {"train": split_manifest["train_hash"], "locked_test": split_manifest["locked_test_hash"]},
        "metric_summary": summary_rows,
        "artifact_checksums": artifact_checksums,
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "LATEST_RUN.txt").write_text(run_id + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "output_dir": str(output_dir.resolve()), "selected_by_scenario": selected_by_scenario}, indent=2))


if __name__ == "__main__":
    main()
