"""Build the complete nine-model evidence matrix from frozen and new predictions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.evaluation.ranking import top_k_metrics
from src.final_release.catalog import COMPARISON_MODELS
from src.final_release.comparator_completion import (
    ARTIFACT_ROOT,
    MODEL_ORDER,
    build_context,
    canonical_hash,
    sha256_file,
    verify_no_change_guard,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_IDS = tuple(model_id for model_id, _ in COMPARISON_MODELS)
DISPLAY_NAMES = dict(COMPARISON_MODELS)
UCI_CLASSES = ("Low", "Medium", "High")
OULAD_CLASSES = ("Not-at-risk", "At-risk")
OVERALL_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "pr_auc",
    "roc_auc",
    "brier",
    "nll",
    "ece",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _multiclass_ece(
    target: np.ndarray, probability: np.ndarray, bins: int = 15
) -> float:
    confidence = probability.max(axis=1)
    predicted = probability.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (
            (confidence >= lower) & (confidence <= upper)
            if index == 0
            else (confidence > lower) & (confidence <= upper)
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float((predicted[mask] == target[mask]).mean())
                - float(confidence[mask].mean())
            )
    return float(result)


def _binary_ece(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (probability >= lower) & (
            probability < upper if upper < 1 else probability <= upper
        )
        if mask.any():
            result += float(mask.mean()) * abs(
                float(target[mask].mean()) - float(probability[mask].mean())
            )
    return float(result)


def _metric_provenance(
    path: Path,
    protocol_hash: str,
    split_hash: str,
    feature_hash: str,
    method: str,
) -> dict[str, str]:
    return {
        "source_artifact": path.relative_to(ROOT).as_posix(),
        "source_checksum": sha256_file(path),
        "protocol_hash": protocol_hash,
        "split_manifest_hash": split_hash,
        "feature_contract_hash": feature_hash,
        "calculation_method": method,
    }


def _ensemble_uci(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model_id", "record_id", "source_row", "true_label", "outer_fold"]
    result = (
        frame.groupby(keys, as_index=False, sort=True)
        .agg(
            p_low=("p_low", "mean"),
            p_medium=("p_medium", "mean"),
            p_high=("p_high", "mean"),
            config_hash=("config_hash", "first"),
            split_manifest_hash=("split_manifest_hash", "first"),
            feature_contract_hash=("feature_contract_hash", "first"),
            inner_protocol_id=("inner_protocol_id", "first"),
            run_id=("run_id", "first"),
        )
        .sort_values(["model_id", "source_row"])
    )
    probability = result[["p_low", "p_medium", "p_high"]].to_numpy(dtype=float)
    result["predicted_label"] = probability.argmax(axis=1)
    return result


def _normalise_uci(dataset: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    context = build_context()
    specification = context.protocol["datasets"][dataset]
    source_root = ROOT / "artifacts/v5_1" / dataset
    deep_path = source_root / "oof_predictions.parquet"
    ml_path = source_root / "ml_oof_predictions.parquet"
    deep_config_path = source_root / "selected_configs.json"
    ml_config_path = source_root / "ml_selected_configs.json"
    dataset_protocol_path = ROOT / f"configs/v5_1/{dataset}_v5_1.yaml"
    completion_path = ARTIFACT_ROOT / dataset / "xgboost_oof_predictions.parquet"
    deep = pd.read_parquet(deep_path)
    ml = pd.read_parquet(ml_path)
    xgb = pd.read_parquet(completion_path)
    xgb["model_id"] = "xgboost"
    deep_map = {
        "cnn_bilstm_v5_1_transfer_selected"
        if dataset == "student_mat"
        else "cnn_bilstm_v5_1": "cnn_bilstm",
        "cnn_only_v5_1": "cnn_only",
        "bilstm_only_v5_1": "bilstm_only",
    }
    deep = deep.loc[deep["candidate"].isin(deep_map)].copy()
    deep["model_id"] = deep["candidate"].map(deep_map)
    deep["dataset"] = dataset
    deep = deep.rename(columns={"target": "true_label"})
    deep_configs = json.loads(deep_config_path.read_text(encoding="utf-8"))
    deep_config_hashes = {
        fold: canonical_hash(deep_configs[fold]) for fold in range(len(deep_configs))
    }
    deep["config_hash"] = deep["outer_fold"].map(deep_config_hashes)
    deep["inner_protocol_id"] = deep["outer_fold"].map(
        lambda fold: f"frozen-v5.1:{dataset}:outer-{int(fold)}"
    )
    ml["dataset"] = dataset
    ml["model_id"] = ml["candidate"].astype(str)
    ml = ml.rename(columns={"target": "true_label"})
    ml_configs = json.loads(ml_config_path.read_text(encoding="utf-8"))
    ml_config_hashes = {
        (int(item["outer_fold"]), str(item["candidate"])): canonical_hash(item)
        for item in ml_configs
    }
    ml["config_hash"] = [
        ml_config_hashes[(int(fold), str(candidate))]
        for fold, candidate in zip(ml["outer_fold"], ml["candidate"])
    ]
    ml["inner_protocol_id"] = ml["outer_fold"].map(
        lambda fold: f"frozen-v5.1:{dataset}:outer-{int(fold)}"
    )
    for frame, run_id in (
        (deep, f"frozen-v5.1-deep-{dataset}"),
        (ml, f"frozen-v5.1-ml-{dataset}"),
    ):
        frame["split_manifest_hash"] = specification["split"]["sha256"]
        frame["feature_contract_hash"] = context.feature_contract_hashes[dataset]
        frame["run_id"] = run_id
    common = [
        "dataset",
        "model_id",
        "record_id",
        "source_row",
        "true_label",
        "outer_fold",
        "seed",
        "p_low",
        "p_medium",
        "p_high",
        "config_hash",
        "split_manifest_hash",
        "feature_contract_hash",
        "inner_protocol_id",
        "run_id",
    ]
    combined = pd.concat([deep[common], ml[common], xgb[common]], ignore_index=True)
    probability = combined[["p_low", "p_medium", "p_high"]].to_numpy(dtype=float)
    combined["predicted_label"] = probability.argmax(axis=1)
    provenance: dict[str, dict[str, Any]] = {}
    for model in ("cnn_bilstm", "cnn_only", "bilstm_only"):
        provenance[model] = {
            "evidence_origin": "frozen_existing"
            if model == "cnn_bilstm"
            else "derived_from_frozen_prediction",
            "source_artifacts": [
                deep_path.relative_to(ROOT).as_posix(),
                deep_config_path.relative_to(ROOT).as_posix(),
                dataset_protocol_path.relative_to(ROOT).as_posix(),
            ],
        }
    for model in (
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
        "svm",
    ):
        provenance[model] = {
            "evidence_origin": "derived_from_frozen_prediction",
            "source_artifacts": [
                ml_path.relative_to(ROOT).as_posix(),
                ml_config_path.relative_to(ROOT).as_posix(),
                dataset_protocol_path.relative_to(ROOT).as_posix(),
            ],
        }
    provenance["xgboost"] = {
        "evidence_origin": "newly_trained_comparator",
        "source_artifacts": [
            completion_path.relative_to(ROOT).as_posix(),
            (ARTIFACT_ROOT / dataset / "xgboost_selected_configs.json")
            .relative_to(ROOT)
            .as_posix(),
        ],
    }
    return combined, provenance


def _normalise_oulad() -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    context = build_context()
    specification = context.protocol["datasets"]["oulad"]
    official_path = ROOT / "artifacts/v6/prediction/final/seed_predictions.parquet"
    ablation_path = ROOT / "artifacts/v5_1/oulad/oof_predictions.parquet"
    official_config_path = ROOT / "artifacts/v6/prediction/selected_model.json"
    ablation_config_path = ROOT / "artifacts/v5_1/oulad/selected_configs.json"
    official_protocol_path = ROOT / "configs/v6/integrated_system_protocol.yaml"
    ablation_protocol_path = ROOT / "configs/v5_1/oulad_v5_1.yaml"
    comparator_path = ARTIFACT_ROOT / "oulad/comparator_seed_predictions.parquet"
    official = pd.read_parquet(official_path)
    official["dataset"] = "oulad"
    official["model_id"] = "cnn_bilstm"
    official = official.rename(columns={"target": "true_label"})
    official["p_at_risk"] = official["probability"].astype(float)
    official["p_not_at_risk"] = 1.0 - official["p_at_risk"]
    official["scope"] = "development_oof"
    official["forecast"] = "F2_MIDDLE"
    official_config = json.loads(official_config_path.read_text(encoding="utf-8"))
    official["config_hash"] = official["outer_fold"].map(
        lambda fold: canonical_hash(
            {"selected_model": official_config, "outer_fold": int(fold)}
        )
    )
    official["inner_protocol_id"] = official["outer_fold"].map(
        lambda fold: f"frozen-v6:oulad:outer-{int(fold)}"
    )
    official["run_id"] = "frozen-v6-oulad-official"

    ablation = pd.read_parquet(ablation_path)
    ablation = ablation.loc[ablation["candidate"].isin(["cnn_only", "bilstm_only"])].copy()
    ablation["dataset"] = "oulad"
    ablation["model_id"] = ablation["candidate"].astype(str)
    ablation = ablation.rename(columns={"target": "true_label"})
    ablation["p_at_risk"] = ablation["probability"].astype(float)
    ablation["p_not_at_risk"] = 1.0 - ablation["p_at_risk"]
    ablation["scope"] = "development_oof"
    ablation["forecast"] = "F2_MIDDLE"
    ablation_configs = json.loads(ablation_config_path.read_text(encoding="utf-8"))
    ablation_config_hashes = {
        int(item["outer_fold"]): item for item in ablation_configs
    }
    ablation["config_hash"] = [
        canonical_hash(
            {
                "selected_outer_config": ablation_config_hashes[int(fold)],
                "variant": candidate,
            }
        )
        for fold, candidate in zip(ablation["outer_fold"], ablation["candidate"])
    ]
    ablation["inner_protocol_id"] = ablation["outer_fold"].map(
        lambda fold: f"frozen-v5.1-ablation:oulad:outer-{int(fold)}"
    )
    ablation["run_id"] = "frozen-v5.1-oulad-ablation"
    cohort = official[
        ["record_id", "id_student", "code_module", "code_presentation"]
    ].drop_duplicates("record_id")
    ablation = ablation.drop(columns=["code_module"], errors="ignore").merge(
        cohort, on=["record_id", "id_student"], how="left", validate="many_to_one"
    )
    official["split_manifest_hash"] = specification["split"]["sha256"]
    ablation["split_manifest_hash"] = specification["split"]["sha256"]
    official["feature_contract_hash"] = sha256_file(
        ROOT / "configs/v6/integrated_system_protocol.yaml"
    )
    ablation["feature_contract_hash"] = sha256_file(
        ROOT / "configs/v5_1/oulad_v5_1.yaml"
    )

    comparator = pd.read_parquet(comparator_path)
    comparator["model_id"] = comparator["model_id"].str.removesuffix("_oulad")
    common = [
        "dataset",
        "model_id",
        "record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "true_label",
        "outer_fold",
        "seed",
        "p_not_at_risk",
        "p_at_risk",
        "threshold",
        "scope",
        "forecast",
        "config_hash",
        "split_manifest_hash",
        "feature_contract_hash",
        "inner_protocol_id",
        "run_id",
    ]
    combined = pd.concat(
        [official[common], ablation[common], comparator[common]], ignore_index=True
    )
    combined["predicted_label"] = (
        combined["p_at_risk"].to_numpy(dtype=float)
        >= combined["threshold"].to_numpy(dtype=float)
    ).astype(int)
    provenance = {
        "cnn_bilstm": {
            "evidence_origin": "frozen_existing",
            "source_artifacts": [
                official_path.relative_to(ROOT).as_posix(),
                official_config_path.relative_to(ROOT).as_posix(),
                official_protocol_path.relative_to(ROOT).as_posix(),
            ],
        },
        "cnn_only": {
            "evidence_origin": "derived_from_frozen_prediction",
            "source_artifacts": [
                ablation_path.relative_to(ROOT).as_posix(),
                ablation_config_path.relative_to(ROOT).as_posix(),
                ablation_protocol_path.relative_to(ROOT).as_posix(),
            ],
        },
        "bilstm_only": {
            "evidence_origin": "derived_from_frozen_prediction",
            "source_artifacts": [
                ablation_path.relative_to(ROOT).as_posix(),
                ablation_config_path.relative_to(ROOT).as_posix(),
                ablation_protocol_path.relative_to(ROOT).as_posix(),
            ],
        },
    }
    for model in MODEL_ORDER:
        provenance[model] = {
            "evidence_origin": "newly_trained_comparator",
            "source_artifacts": [
                comparator_path.relative_to(ROOT).as_posix(),
                (ARTIFACT_ROOT / "oulad/selected_configs.json")
                .relative_to(ROOT)
                .as_posix(),
            ],
        }
    return combined, provenance


def _ensemble_oulad(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "model_id",
        "record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "true_label",
        "outer_fold",
        "scope",
        "forecast",
    ]
    result = (
        frame.groupby(keys, as_index=False, sort=True)
        .agg(
            p_at_risk=("p_at_risk", "mean"),
            threshold=("threshold", "first"),
            config_hash=("config_hash", "first"),
            split_manifest_hash=("split_manifest_hash", "first"),
            feature_contract_hash=("feature_contract_hash", "first"),
            inner_protocol_id=("inner_protocol_id", "first"),
            run_id=("run_id", "first"),
        )
        .sort_values(["model_id", "record_id"])
    )
    result["p_not_at_risk"] = 1.0 - result["p_at_risk"]
    result["predicted_label"] = (
        result["p_at_risk"].to_numpy(dtype=float)
        >= result["threshold"].to_numpy(dtype=float)
    ).astype(int)
    return result


def _overall(
    target: np.ndarray,
    probability: np.ndarray,
    predicted: np.ndarray,
    binary: bool,
) -> dict[str, float]:
    probability = probability / probability.sum(axis=1, keepdims=True)
    result = {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_precision": float(
            precision_score(target, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(target, predicted, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(target, predicted, average="weighted", zero_division=0)
        ),
        "nll": float(log_loss(target, probability, labels=list(range(probability.shape[1])))),
    }
    one_hot = label_binarize(target, classes=list(range(probability.shape[1])))
    if binary:
        result["pr_auc"] = float(average_precision_score(target, probability[:, 1]))
        result["roc_auc"] = float(roc_auc_score(target, probability[:, 1]))
        result["brier"] = float(np.mean((probability[:, 1] - target) ** 2))
        result["ece"] = _binary_ece(target, probability[:, 1])
        result["risk_precision"] = float(
            precision_score(target, predicted, pos_label=1, zero_division=0)
        )
        result["risk_recall"] = float(
            recall_score(target, predicted, pos_label=1, zero_division=0)
        )
        result["risk_f1"] = float(
            f1_score(target, predicted, pos_label=1, zero_division=0)
        )
    else:
        result["pr_auc"] = float(
            average_precision_score(one_hot, probability, average="macro")
        )
        result["roc_auc"] = float(
            roc_auc_score(target, probability, multi_class="ovr", average="macro")
        )
        result["brier"] = float(
            np.mean(np.sum((probability - one_hot) ** 2, axis=1))
        )
        result["ece"] = _multiclass_ece(target, probability)
    return result


def _per_class(
    model: str,
    target: np.ndarray,
    predicted: np.ndarray,
    labels: tuple[str, ...],
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        predicted,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    return [
        {
            "model_id": model,
            "model": DISPLAY_NAMES[model],
            "class": label,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    ]


def _confusion_contributions(
    group: np.ndarray, target: np.ndarray, predicted: np.ndarray, classes: int
) -> tuple[np.ndarray, np.ndarray]:
    codes, unique = pd.factorize(group, sort=True)
    values = np.zeros((len(unique), classes * classes), dtype=np.int64)
    flat = target.astype(int) * classes + predicted.astype(int)
    np.add.at(values, (codes, flat), 1)
    return values, unique.astype(str)


def _macro_f1_from_flat(confusion: np.ndarray, classes: int) -> np.ndarray:
    matrix = confusion.reshape((-1, classes, classes)).astype(float)
    diagonal = np.diagonal(matrix, axis1=1, axis2=2)
    precision = np.divide(
        diagonal,
        matrix.sum(axis=1),
        out=np.zeros_like(diagonal),
        where=matrix.sum(axis=1) != 0,
    )
    recall = np.divide(
        diagonal,
        matrix.sum(axis=2),
        out=np.zeros_like(diagonal),
        where=matrix.sum(axis=2) != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    return f1.mean(axis=1)


def _paired_bootstrap(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    group_column: str,
    classes: int,
    resamples: int = 5000,
) -> tuple[float, float]:
    reference_columns = ["record_id", "true_label", "predicted_label"]
    if group_column != "record_id":
        reference_columns.insert(1, group_column)
    merged = reference[reference_columns].merge(
        comparator[["record_id", "true_label", "predicted_label"]],
        on="record_id",
        suffixes=("_reference", "_comparator"),
        validate="one_to_one",
    )
    if not np.array_equal(
        merged["true_label_reference"].to_numpy(),
        merged["true_label_comparator"].to_numpy(),
    ):
        raise RuntimeError("Bootstrap targets disagree")
    group = (
        merged["record_id"].astype(str).to_numpy()
        if group_column == "record_id"
        else merged[group_column].astype(str).to_numpy()
    )
    target = merged["true_label_reference"].to_numpy(dtype=int)
    ref_contribution, ref_groups = _confusion_contributions(
        group, target, merged["predicted_label_reference"].to_numpy(dtype=int), classes
    )
    comp_contribution, comp_groups = _confusion_contributions(
        group, target, merged["predicted_label_comparator"].to_numpy(dtype=int), classes
    )
    if not np.array_equal(ref_groups, comp_groups):
        raise RuntimeError("Bootstrap group order disagrees")
    rng = np.random.default_rng(7319)
    group_count = len(ref_groups)
    deltas: list[np.ndarray] = []
    batch = 50
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        samples = rng.integers(0, group_count, size=(size, group_count))
        ref_cm = ref_contribution[samples].sum(axis=1)
        comp_cm = comp_contribution[samples].sum(axis=1)
        deltas.append(
            _macro_f1_from_flat(ref_cm, classes)
            - _macro_f1_from_flat(comp_cm, classes)
        )
    values = np.concatenate(deltas)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _seed_stability(frame: pd.DataFrame, binary: bool) -> dict[str, float]:
    values = []
    probability_columns = (
        ["p_not_at_risk", "p_at_risk"] if binary else ["p_low", "p_medium", "p_high"]
    )
    for _, seed_frame in frame.groupby("seed"):
        probability = seed_frame[probability_columns].to_numpy(dtype=float)
        target = seed_frame["true_label"].to_numpy(dtype=int)
        if binary:
            predicted = (
                seed_frame["p_at_risk"].to_numpy(dtype=float)
                >= seed_frame["threshold"].to_numpy(dtype=float)
            ).astype(int)
        else:
            predicted = probability.argmax(axis=1)
        values.append(f1_score(target, predicted, average="macro", zero_division=0))
    return {
        "seed_mean": float(np.mean(values)),
        "seed_std": float(np.std(values, ddof=0)),
        "seed_min": float(np.min(values)),
        "seed_max": float(np.max(values)),
        "seed_count": len(values),
    }


def _deep_registry_metrics(dataset: str) -> dict[str, dict[str, Any]]:
    if dataset in {"student_mat", "student_por"}:
        payload = json.loads(
            (
                ROOT / f"artifacts/v5_1/{dataset}/final_metrics.json"
            ).read_text(encoding="utf-8")
        )
        result = {"cnn_bilstm": payload["metrics"]}
        for item in payload.get("ablation_metrics", []):
            candidate = str(item.get("candidate", ""))
            if candidate.startswith("cnn_only"):
                result["cnn_only"] = item
            elif candidate.startswith("bilstm_only"):
                result["bilstm_only"] = item
        ml_payload = json.loads(
            (
                ROOT / f"artifacts/v5_1/{dataset}/ml_final_metrics.json"
            ).read_text(encoding="utf-8")
        )
        for item in ml_payload:
            if int(item.get("seed", 0)) == -1:
                candidate = str(item["candidate"]).removesuffix("_ensemble")
                result[candidate] = item
        return result
    v6 = json.loads(
        (ROOT / "artifacts/v6/prediction/final/metrics.json").read_text(
            encoding="utf-8"
        )
    )
    v51 = json.loads(
        (ROOT / "artifacts/v5_1/oulad/final_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "cnn_bilstm": next(item for item in v6 if int(item["seed"]) == -1),
        "cnn_only": next(
            item for item in v51 if item["candidate"] == "cnn_only_ensemble"
        ),
        "bilstm_only": next(
            item for item in v51 if item["candidate"] == "bilstm_only_ensemble"
        ),
    }


def _replay_validation(
    dataset: str, model: str, metrics: dict[str, float], registry: dict[str, Any]
) -> dict[str, Any]:
    aliases = {
        "pr_auc": "macro_pr_auc" if dataset != "oulad" else "pr_auc",
        "risk_precision": "at_risk_precision",
        "risk_recall": "at_risk_recall",
        "risk_f1": "at_risk_f1",
    }
    differences: dict[str, float] = {}
    for name, value in metrics.items():
        registry_name = aliases.get(name, name)
        if registry_name in registry and registry[registry_name] is not None:
            differences[name] = float(value) - float(registry[registry_name])
    violating = {
        name: difference
        for name, difference in differences.items()
        if abs(difference) > 1e-6
    }
    if violating:
        raise RuntimeError(
            f"{dataset}/{model}: frozen metric replay mismatch {violating}"
        )
    return {
        "status": "PASS",
        "tolerance": 1e-6,
        "differences": differences,
    }


def _validate_prediction_integrity(
    dataset: str, seed_frame: pd.DataFrame, ensemble: pd.DataFrame, expected: int
) -> list[str]:
    errors: list[str] = []
    if set(seed_frame["model_id"]) != set(MODEL_IDS):
        errors.append(f"{dataset}: model set mismatch")
    if seed_frame.duplicated(["dataset", "model_id", "record_id", "seed"]).any():
        errors.append(f"{dataset}: duplicate record/model/seed")
    probability_columns = (
        ["p_not_at_risk", "p_at_risk"]
        if dataset == "oulad"
        else ["p_low", "p_medium", "p_high"]
    )
    probability = seed_frame[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probability).all():
        errors.append(f"{dataset}: non-finite probability")
    if ((probability < 0) | (probability > 1)).any():
        errors.append(f"{dataset}: probability outside [0,1]")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-6):
        errors.append(f"{dataset}: probability rows do not sum to one")
    canonical: set[str] | None = None
    for model, frame in ensemble.groupby("model_id"):
        ids = set(frame["record_id"].astype(str))
        if len(ids) != expected:
            errors.append(f"{dataset}/{model}: expected {expected} records, got {len(ids)}")
        if canonical is None:
            canonical = ids
        elif ids != canonical:
            errors.append(f"{dataset}/{model}: record IDs differ")
    if dataset == "oulad":
        if set(seed_frame["scope"]) != {"development_oof"}:
            errors.append("oulad: non-development scope present")
        if set(seed_frame["forecast"]) != {"F2_MIDDLE"}:
            errors.append("oulad: non-F2 forecast present")
    return errors


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    context = build_context()
    specification = context.protocol["datasets"][dataset]
    output_root = ARTIFACT_ROOT / dataset
    if dataset == "oulad":
        seed_frame, provenance = _normalise_oulad()
        ensemble = _ensemble_oulad(seed_frame)
        probability_columns = ["p_not_at_risk", "p_at_risk"]
        labels = OULAD_CLASSES
        binary = True
        group_column = "id_student"
        seed_output = output_root / "seed_predictions.parquet"
        ensemble_output = output_root / "ensemble_oof_predictions.parquet"
    else:
        seed_frame, provenance = _normalise_uci(dataset)
        ensemble = _ensemble_uci(seed_frame)
        probability_columns = ["p_low", "p_medium", "p_high"]
        labels = UCI_CLASSES
        binary = False
        group_column = "record_id"
        seed_output = output_root / "seed_predictions.parquet"
        ensemble_output = output_root / "oof_predictions.parquet"
    _atomic_parquet(seed_output, seed_frame)
    _atomic_parquet(ensemble_output, ensemble)
    errors = _validate_prediction_integrity(
        dataset, seed_frame, ensemble, int(specification["expected_records"])
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    metric_models: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_values: dict[str, Any] = {}
    top_k_rows: list[dict[str, Any]] = []
    deep_registry = _deep_registry_metrics(dataset)
    for model in MODEL_IDS:
        model_seed = seed_frame.loc[seed_frame["model_id"] == model].copy()
        model_frame = ensemble.loc[ensemble["model_id"] == model].copy()
        target = model_frame["true_label"].to_numpy(dtype=int)
        probability = model_frame[probability_columns].to_numpy(dtype=float)
        predicted = model_frame["predicted_label"].to_numpy(dtype=int)
        feature_hashes = set(model_frame["feature_contract_hash"].astype(str))
        if len(feature_hashes) != 1:
            raise RuntimeError(f"{dataset}/{model}: multiple feature contracts")
        prediction_provenance = _metric_provenance(
            ensemble_output,
            context.protocol_hash,
            specification["split"]["sha256"],
            feature_hashes.pop(),
            "recomputed_from_record_aligned_ensemble_probability",
        )
        metrics = _overall(target, probability, predicted, binary)
        per_class = _per_class(model, target, predicted, labels)
        per_class_rows.extend(per_class)
        matrix = confusion_matrix(
            target, predicted, labels=np.arange(len(labels))
        ).astype(int)
        confusion_values[model] = {
            "model": DISPLAY_NAMES[model],
            "class_order": list(labels),
            "matrix": matrix.tolist(),
            **prediction_provenance,
        }
        model_provenance = provenance[model]
        source_checksums = {
            source: sha256_file(ROOT / source)
            for source in model_provenance["source_artifacts"]
        }
        metric_models.append(
            {
                "model_id": model,
                "model": DISPLAY_NAMES[model],
                "metrics": metrics,
                "seed_stability": _seed_stability(model_seed, binary),
                "evidence_origin": model_provenance["evidence_origin"],
                "protocol_id": context.protocol["protocol_id"],
                "source_artifacts": model_provenance["source_artifacts"],
                "source_checksums": source_checksums,
                "metric_provenance": prediction_provenance,
                "replay_validation": (
                    _replay_validation(
                        dataset, model, metrics, deep_registry[model]
                    )
                    if model in deep_registry
                    else None
                ),
            }
        )
        if binary:
            for budget in (0.05, 0.10, 0.20):
                values = top_k_metrics(
                    target,
                    probability[:, 1],
                    model_frame["record_id"].astype(str).to_numpy(),
                    budget,
                )
                top_k_rows.append(
                    {
                        "model_id": model,
                        "model": DISPLAY_NAMES[model],
                        "budget": budget,
                        "k": max(1, math.ceil(len(target) * budget)),
                        "precision_at_k": values["precision"],
                        "recall_at_k": values["recall"],
                        "f1_at_k": values["f1"],
                        "ndcg_at_k": values["ndcg"],
                    }
                )
    metrics_payload = {
        "schema_version": "comparator_completion_metrics_v1",
        "dataset": dataset,
        "records": int(specification["expected_records"]),
        "models": metric_models,
    }
    _atomic_json(output_root / "metrics.json", metrics_payload)
    _atomic_csv(output_root / "per_class.csv", pd.DataFrame(per_class_rows))
    _atomic_json(output_root / "confusion_matrices.json", confusion_values)
    if not binary:
        xgboost_metric = next(
            row for row in metric_models if row["model_id"] == "xgboost"
        )
        _atomic_json(output_root / "xgboost_metrics.json", xgboost_metric)
        _atomic_csv(
            output_root / "xgboost_per_class.csv",
            pd.DataFrame(per_class_rows).loc[
                lambda frame: frame["model_id"] == "xgboost"
            ],
        )
        _atomic_json(
            output_root / "xgboost_confusion_matrix.json",
            confusion_values["xgboost"],
        )
    if binary:
        _atomic_csv(output_root / "top_k.csv", pd.DataFrame(top_k_rows))

    reference = ensemble.loc[ensemble["model_id"] == "cnn_bilstm"].copy()
    reference_score = next(
        row["metrics"]["macro_f1"]
        for row in metric_models
        if row["model_id"] == "cnn_bilstm"
    )
    comparison_rows: list[dict[str, Any]] = []
    margin = float(context.protocol["evaluation"]["practical_margin_macro_f1"])
    for model in MODEL_IDS:
        if model == "cnn_bilstm":
            continue
        comparator = ensemble.loc[ensemble["model_id"] == model].copy()
        comparator_score = next(
            row["metrics"]["macro_f1"]
            for row in metric_models
            if row["model_id"] == model
        )
        lower, upper = _paired_bootstrap(
            reference, comparator, group_column, len(labels)
        )
        delta = reference_score - comparator_score
        if lower > 0 and delta >= margin:
            verdict = "CNN_BILSTM_HIGHER"
        elif upper < 0 and delta <= -margin:
            verdict = "COMPARATOR_HIGHER"
        else:
            verdict = "PRACTICAL_TIE"
        comparison_rows.append(
            {
                "dataset": dataset,
                "comparator": model,
                "cnn_bilstm_macro_f1": reference_score,
                "comparator_macro_f1": comparator_score,
                "delta_cnn_bilstm_minus_comparator": delta,
                "ci_95_lower": lower,
                "ci_95_upper": upper,
                "verdict": verdict,
                "bootstrap_unit": group_column,
                "resamples": 5000,
            }
        )
    _atomic_csv(output_root / "bootstrap_comparison.csv", pd.DataFrame(comparison_rows))
    return {
        "dataset": dataset,
        "models": len(metric_models),
        "records": int(specification["expected_records"]),
        "status": "PASS",
    }


def _build_global_indexes() -> None:
    search = []
    selected: list[dict[str, Any]] = []
    for dataset in ("student_mat", "student_por"):
        search.append(
            pd.read_csv(ARTIFACT_ROOT / dataset / "xgboost_search_trials.csv")
        )
        selected.extend(
            json.loads(
                (ARTIFACT_ROOT / dataset / "xgboost_selected_configs.json").read_text(
                    encoding="utf-8"
                )
            )
        )
    search.append(pd.read_csv(ARTIFACT_ROOT / "oulad/search_trials.csv"))
    selected.extend(
        json.loads(
            (ARTIFACT_ROOT / "oulad/selected_configs.json").read_text(encoding="utf-8")
        )
    )
    _atomic_csv(ARTIFACT_ROOT / "search_trials.csv", pd.concat(search, ignore_index=True))
    _atomic_json(ARTIFACT_ROOT / "selected_configs.json", selected)
    sample_path = ARTIFACT_ROOT / "process_resource_samples.csv"
    runtime_path = ARTIFACT_ROOT / "runtime_resources.csv"
    if sample_path.is_file():
        samples = pd.read_csv(sample_path)
        peak_rss = int(samples["rss_bytes"].max()) if len(samples) else 0
        runtime = pd.read_csv(runtime_path)
        runtime["process_run_peak_rss_bytes"] = np.where(
            runtime["dataset"] == "oulad", peak_rss, np.nan
        )
        _atomic_csv(runtime_path, runtime)
        _atomic_json(
            ARTIFACT_ROOT / "resource_summary.json",
            {
                "oulad_process_peak_rss_bytes": peak_rss,
                "sample_count": len(samples),
                "sample_interval_seconds": 5,
                "cpu_comparators": True,
                "concurrent_svm_jobs": 1,
            },
        )
    elif not (ARTIFACT_ROOT / "resource_summary.json").is_file():
        raise RuntimeError("Comparator resource summary is missing")


def build_checksum_manifest() -> dict[str, Any]:
    manifest_path = ARTIFACT_ROOT / "checksum_manifest.json"
    files: dict[str, str] = {}
    for path in sorted(
        item
        for item in ARTIFACT_ROOT.rglob("*")
        if item.is_file()
        and item != manifest_path
    ):
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    payload = {
        "schema_version": "comparator_completion_checksum_manifest_v1",
        "files": files,
        "aggregate_sha256": canonical_hash(files),
    }
    _atomic_json(manifest_path, payload)
    return payload


def evaluate_all() -> dict[str, Any]:
    context = build_context()
    results = [
        evaluate_dataset(dataset)
        for dataset in ("student_mat", "student_por", "oulad")
    ]
    _build_global_indexes()
    guard = verify_no_change_guard()
    if guard["status"] != "PASS":
        raise RuntimeError(f"No-change guard failed: {guard}")
    validation = {
        "schema_version": "comparator_completion_validation_v1",
        "status": "PASS",
        "datasets": results,
        "nine_models_each_dataset": all(item["models"] == 9 for item in results),
        "no_applicable_na": True,
        "future_oulad_executed": False,
        "official_deep_models_retrained": False,
        "no_change_guard": guard,
        "protocol_hash": context.protocol_hash,
    }
    _atomic_json(ARTIFACT_ROOT / "validation_report.json", validation)
    _atomic_json(
        ARTIFACT_ROOT / "run_state.json",
        {
            "status": "COMPLETE",
            "phase": "EVALUATION_COMPLETE",
            "future_oulad_executed": False,
            "protocol_hash": context.protocol_hash,
        },
    )
    build_checksum_manifest()
    return validation


__all__ = ["MODEL_IDS", "OVERALL_METRICS", "evaluate_all", "evaluate_dataset"]
