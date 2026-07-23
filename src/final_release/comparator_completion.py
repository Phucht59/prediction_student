"""Preregistered, resumable completion of the final classical-ML comparators."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.studies.oulad_v4.data import build_v4_inner_manifest, load_v4_data, manifest_indices
from src.studies.v5_1.common.protocol import load_protocol
from src.studies.v5_1.common.uci_data import context_preprocessor, load_uci_v5_1
from src.studies.v5_1.oulad.data import (
    CATEGORICAL_STATIC,
    compact_aggregate_columns,
)
from src.studies.oulad_v3.data import STATIC_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs/final/comparator_completion_protocol.yaml"
ARTIFACT_ROOT = ROOT / "artifacts/final/comparator_completion"
SEEDS = (42, 1201, 2026, 3407, 7319)
MODEL_ORDER = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
    "svm",
    "xgboost",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_completion_protocol() -> tuple[dict[str, Any], str]:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "PREREGISTERED_BEFORE_COMPARATOR_TRAINING":
        raise RuntimeError("Comparator completion protocol is not preregistered")
    return protocol, sha256_file(PROTOCOL_PATH)


def _directory_guard(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            result[root.relative_to(ROOT).as_posix()] = sha256_file(root)
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            result[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return result


def create_no_change_guard() -> dict[str, Any]:
    """Snapshot immutable released evidence before any comparator fit."""

    mat_official = sorted(
        (ROOT / "artifacts/v5_1/student_mat/checkpoints").glob(
            "cnn_bilstm_v5_1_transfer_selected_*.pt"
        )
    )
    por_official = sorted(
        (ROOT / "artifacts/v5_1/student_por/checkpoints").glob("cnn_bilstm_v5_1_*.pt")
    )
    oulad_official = [ROOT / "artifacts/v6/prediction/final/checkpoints"]
    fixed_files = [
        ROOT / "artifacts/v5_1/student_mat/oof_predictions.parquet",
        ROOT / "artifacts/v5_1/student_por/oof_predictions.parquet",
        ROOT / "artifacts/v6/prediction/final/seed_predictions.parquet",
        ROOT / "artifacts/v6/protocol_snapshot.json",
        ROOT / "artifacts/v6/recommendation",
    ]
    files = _directory_guard([*mat_official, *por_official, *oulad_official, *fixed_files])
    guard = {
        "schema_version": "comparator_no_change_guard_v1",
        "created_before_comparator_training": True,
        "files": files,
        "aggregate_sha256": canonical_hash(files),
    }
    atomic_json(ARTIFACT_ROOT / "no_change_guard.json", guard)
    return guard


def verify_no_change_guard() -> dict[str, Any]:
    path = ARTIFACT_ROOT / "no_change_guard.json"
    guard = json.loads(path.read_text(encoding="utf-8"))
    current: dict[str, str] = {}
    missing: list[str] = []
    changed: list[str] = []
    for relative, expected in guard["files"].items():
        source = ROOT / relative
        if not source.is_file():
            missing.append(relative)
            continue
        actual = sha256_file(source)
        current[relative] = actual
        if actual != expected:
            changed.append(relative)
    return {
        "status": "PASS" if not missing and not changed else "FAIL",
        "missing": missing,
        "changed": changed,
        "expected_aggregate_sha256": guard["aggregate_sha256"],
        "current_aggregate_sha256": canonical_hash(current),
    }


@dataclass(frozen=True)
class CompletionContext:
    protocol: dict[str, Any]
    protocol_hash: str
    feature_contract_hashes: dict[str, str]
    run_id: str


def build_context() -> CompletionContext:
    protocol, protocol_hash = load_completion_protocol()
    feature_hashes = {
        dataset: canonical_hash(specification["feature_contract"])
        for dataset, specification in protocol["datasets"].items()
    }
    return CompletionContext(
        protocol=protocol,
        protocol_hash=protocol_hash,
        feature_contract_hashes=feature_hashes,
        run_id=f"comparator-completion-{protocol_hash[:12]}",
    )


def write_protocol_snapshots(context: CompletionContext) -> None:
    protocol = context.protocol
    amendment_paths = sorted(
        (ROOT / "configs/final").glob("comparator_completion_amendment_*.yaml")
    )
    source_manifest: dict[str, Any] = {}
    split_checksums: dict[str, str] = {}
    for dataset, specification in protocol["datasets"].items():
        split = ROOT / specification["split"]["manifest"]
        actual = sha256_file(split)
        expected = specification["split"]["sha256"]
        if actual != expected:
            raise RuntimeError(f"{dataset} split checksum mismatch")
        split_checksums[dataset] = actual
        if "source" in specification:
            source = ROOT / specification["source"]
            if sha256_file(source) != specification["source_sha256"]:
                raise RuntimeError(f"{dataset} source checksum mismatch")
            source_manifest[dataset] = {
                "path": specification["source"],
                "sha256": specification["source_sha256"],
            }
        else:
            source_manifest[dataset] = specification["sources"]
            for item in specification["sources"].values():
                if sha256_file(ROOT / item["path"]) != item["sha256"]:
                    raise RuntimeError(f"{dataset} source checksum mismatch: {item['path']}")
    atomic_json(
        ARTIFACT_ROOT / "protocol_snapshot.json",
        {
            "schema_version": "comparator_completion_protocol_snapshot_v1",
            "protocol_id": protocol["protocol_id"],
            "protocol_path": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
            "protocol_hash": context.protocol_hash,
            "amendments": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in amendment_paths
            ],
            "run_id": context.run_id,
            "future_oulad_executed": False,
        },
    )
    atomic_json(ARTIFACT_ROOT / "source_manifest.json", source_manifest)
    atomic_json(ARTIFACT_ROOT / "split_manifest_checksums.json", split_checksums)
    atomic_json(
        ARTIFACT_ROOT / "feature_contract.json",
        {
            "schema_version": "comparator_feature_contract_v1",
            "contracts": {
                dataset: {
                    "value": protocol["datasets"][dataset]["feature_contract"],
                    "sha256": context.feature_contract_hashes[dataset],
                }
                for dataset in ("student_mat", "student_por", "oulad")
            },
        },
    )


def _uci_outer_indices(specification: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    manifest = pd.read_csv(ROOT / specification["split"]["manifest"])
    result = []
    for fold in range(int(specification["split"]["outer_folds"])):
        selected = manifest.loc[manifest["outer_fold"].astype(int) == fold]
        train = selected.loc[selected["role"] == "outer_train", "source_row"].to_numpy(dtype=int)
        validation = selected.loc[
            selected["role"] == "outer_validation", "source_row"
        ].to_numpy(dtype=int)
        if set(train) & set(validation):
            raise RuntimeError(f"UCI outer fold {fold} overlaps")
        result.append((train, validation))
    return result


def _uci_inner_splits(
    data: Any, outer_train: np.ndarray, split_seed: int, outer_fold: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=split_seed + outer_fold
    )
    return [
        (outer_train[train], outer_train[validation])
        for train, validation in splitter.split(
            outer_train, data.target[outer_train], data.quasi_groups[outer_train]
        )
    ]


def _uci_matrix(
    data: Any,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    fitted: ColumnTransformer | None = None,
) -> tuple[np.ndarray, np.ndarray, ColumnTransformer]:
    transformer = fitted or context_preprocessor(include_absences=False)
    if fitted is None:
        transformer.fit(data.context.iloc[fit_indices])
    context = transformer.transform(data.context.iloc[transform_indices]).astype(np.float32)
    temporal = data.temporal[transform_indices].reshape(len(transform_indices), -1)
    return (
        np.concatenate([temporal, context], axis=1).astype(np.float32),
        data.target[transform_indices].astype(int),
        transformer,
    )


def _multiclass_selection_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = probability.argmax(axis=1)
    one_hot = label_binarize(target, classes=[0, 1, 2])
    return {
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_pr_auc": float(
            average_precision_score(one_hot, probability, average="macro")
        ),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
    }


def _uci_xgb(parameters: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=3,
        random_state=seed,
        **parameters,
    )


def run_uci_xgboost(dataset: str, context: CompletionContext) -> None:
    specification = context.protocol["datasets"][dataset]
    dataset_name = "student-mat" if dataset == "student_mat" else "student-por"
    data = load_uci_v5_1(ROOT / specification["source"], dataset_name)
    if len(data.target) != int(specification["expected_records"]):
        raise RuntimeError(f"{dataset} record count mismatch")
    output_root = ARTIFACT_ROOT / dataset
    fold_root = output_root / "folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    space = specification["xgboost"]["search_space"]
    budget = int(specification["xgboost"]["search_budget_per_outer_fold"])
    split_hash = specification["split"]["sha256"]
    feature_hash = context.feature_contract_hashes[dataset]
    search_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_validation) in enumerate(
        _uci_outer_indices(specification)
    ):
        fold_path = fold_root / f"xgboost_outer_{outer_fold}.parquet"
        state_path = fold_root / f"xgboost_outer_{outer_fold}.json"
        if fold_path.is_file() and state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if (
                state.get("status") == "PASS"
                and state.get("prediction_sha256") == sha256_file(fold_path)
                and state.get("protocol_hash") == context.protocol_hash
            ):
                search_rows.extend(state["search_rows"])
                selected_rows.append(state["selected"])
                runtime_rows.append(state["runtime"])
                continue
        started = time.perf_counter()
        inner = _uci_inner_splits(
            data,
            outer_train,
            int(specification["split"]["split_seed"]),
            outer_fold,
        )
        configurations = list(
            ParameterSampler(
                space,
                n_iter=budget,
                random_state=3407 + outer_fold,
            )
        )
        fold_search: list[dict[str, Any]] = []
        for trial, parameters in enumerate(configurations):
            targets: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            trial_started = time.perf_counter()
            for train_indices, validation_indices in inner:
                train_x, train_y, transformer = _uci_matrix(
                    data, train_indices, train_indices
                )
                validation_x, validation_y, _ = _uci_matrix(
                    data, train_indices, validation_indices, transformer
                )
                model = _uci_xgb(parameters, 3407)
                model.fit(train_x, train_y)
                targets.append(validation_y)
                probabilities.append(model.predict_proba(validation_x))
            metrics = _multiclass_selection_metrics(
                np.concatenate(targets), np.concatenate(probabilities)
            )
            fold_search.append(
                {
                    "dataset": dataset,
                    "model_id": "xgboost",
                    "outer_fold": outer_fold,
                    "trial": trial,
                    "config_hash": canonical_hash(parameters),
                    "parameters": json.dumps(parameters, sort_keys=True),
                    **metrics,
                    "runtime_seconds": time.perf_counter() - trial_started,
                    "status": "COMPLETE",
                }
            )
        ranked = sorted(
            fold_search,
            key=lambda row: (
                -row["macro_f1"],
                -row["balanced_accuracy"],
                -row["macro_pr_auc"],
                row["brier"],
                json.loads(row["parameters"])["n_estimators"],
                row["trial"],
            ),
        )
        best = ranked[0]
        parameters = json.loads(best["parameters"])
        config_hash = best["config_hash"]
        train_x, train_y, transformer = _uci_matrix(data, outer_train, outer_train)
        validation_x, validation_y, _ = _uci_matrix(
            data, outer_train, outer_validation, transformer
        )
        prediction_rows: list[dict[str, Any]] = []
        for seed in SEEDS:
            model = _uci_xgb(parameters, seed)
            model.fit(train_x, train_y)
            probability = model.predict_proba(validation_x)
            for position, source_index in enumerate(outer_validation):
                values = probability[position]
                prediction_rows.append(
                    {
                        "dataset": dataset,
                        "model_id": "xgboost",
                        "record_id": str(data.record_ids[source_index]),
                        "source_row": int(source_index),
                        "true_label": int(validation_y[position]),
                        "predicted_label": int(np.argmax(values)),
                        "outer_fold": outer_fold,
                        "inner_protocol_id": f"{context.protocol['protocol_id']}:{dataset}:outer-{outer_fold}",
                        "seed": seed,
                        "p_low": float(values[0]),
                        "p_medium": float(values[1]),
                        "p_high": float(values[2]),
                        "config_hash": config_hash,
                        "split_manifest_hash": split_hash,
                        "feature_contract_hash": feature_hash,
                        "run_id": context.run_id,
                    }
                )
        predictions = pd.DataFrame(prediction_rows)
        atomic_parquet(fold_path, predictions)
        selected = {
            "dataset": dataset,
            "model_id": "xgboost",
            "outer_fold": outer_fold,
            "parameters": parameters,
            "config_hash": config_hash,
            "inner_metrics": {
                key: best[key]
                for key in ("macro_f1", "balanced_accuracy", "macro_pr_auc", "brier")
            },
            "trial": best["trial"],
        }
        runtime = {
            "dataset": dataset,
            "model_id": "xgboost",
            "outer_fold": outer_fold,
            "runtime_seconds": time.perf_counter() - started,
            "records": len(outer_validation),
            "seeds": len(SEEDS),
            "hostname": platform.node(),
        }
        state = {
            "status": "PASS",
            "protocol_hash": context.protocol_hash,
            "prediction_sha256": sha256_file(fold_path),
            "selected": selected,
            "search_rows": fold_search,
            "runtime": runtime,
        }
        atomic_json(state_path, state)
        search_rows.extend(fold_search)
        selected_rows.append(selected)
        runtime_rows.append(runtime)
    predictions = pd.concat(
        [
            pd.read_parquet(fold_root / f"xgboost_outer_{fold}.parquet")
            for fold in range(int(specification["split"]["outer_folds"]))
        ],
        ignore_index=True,
    )
    completion_model_id = specification["xgboost"]["model_id"]
    predictions["model_id"] = completion_model_id
    for row in search_rows:
        row["model_id"] = completion_model_id
    for row in selected_rows:
        row["model_id"] = completion_model_id
    atomic_parquet(output_root / "xgboost_oof_predictions.parquet", predictions)
    atomic_csv(output_root / "xgboost_search_trials.csv", pd.DataFrame(search_rows))
    atomic_json(output_root / "xgboost_selected_configs.json", selected_rows)
    atomic_json(
        output_root / "xgboost_run_state.json",
        {
            "status": "COMPLETE",
            "protocol_hash": context.protocol_hash,
            "outer_folds_complete": int(specification["split"]["outer_folds"]),
            "prediction_sha256": sha256_file(output_root / "xgboost_oof_predictions.parquet"),
        },
    )
    _append_runtime(runtime_rows)


def _oulad_static_preprocessor() -> ColumnTransformer:
    numeric = [column for column in STATIC_COLUMNS if column not in CATEGORICAL_STATIC]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                list(CATEGORICAL_STATIC),
            ),
        ],
        sparse_threshold=0.0,
    )


@dataclass
class OULADMLPreprocessor:
    aggregate: Pipeline
    static: ColumnTransformer
    aggregate_columns: tuple[str, ...]


def _oulad_matrix(
    data: Any,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    fitted: OULADMLPreprocessor | None = None,
) -> tuple[np.ndarray, np.ndarray, OULADMLPreprocessor]:
    columns = compact_aggregate_columns(list(data.v2.aggregate_columns))
    aggregate = data.v2.aggregate.loc[:, list(columns)].to_numpy(dtype=np.float32)
    if fitted is None:
        aggregate_preprocessor = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        aggregate_preprocessor.fit(aggregate[fit_indices])
        static_preprocessor = _oulad_static_preprocessor()
        static_preprocessor.fit(data.base.cohort.loc[fit_indices, STATIC_COLUMNS])
        fitted = OULADMLPreprocessor(
            aggregate=aggregate_preprocessor,
            static=static_preprocessor,
            aggregate_columns=columns,
        )
    if tuple(fitted.aggregate_columns) != tuple(columns):
        raise RuntimeError("OULAD compact aggregate feature contract changed")
    aggregate_x = fitted.aggregate.transform(aggregate[transform_indices]).astype(np.float32)
    static_x = fitted.static.transform(
        data.base.cohort.loc[transform_indices, STATIC_COLUMNS]
    ).astype(np.float32)
    return (
        np.concatenate([aggregate_x, static_x], axis=1),
        data.y[transform_indices].astype(int),
        fitted,
    )


def _balanced_scale(y: np.ndarray) -> float:
    counts = np.bincount(y.astype(int), minlength=2)
    return float(counts[0] / max(1, counts[1]))


def _oulad_model(
    name: str, parameters: dict[str, Any], seed: int, train_y: np.ndarray
) -> Any:
    local = dict(parameters)
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, **local)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **local)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=3, **local)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **local)
    if name == "svm":
        return SVC(random_state=seed, **local)
    if name == "xgboost":
        if local.pop("scale_pos_weight") == "balanced_training_partition":
            local["scale_pos_weight"] = _balanced_scale(train_y)
        return XGBClassifier(
            random_state=seed,
            n_jobs=3,
            eval_metric="logloss",
            **local,
        )
    raise ValueError(name)


def _threshold_metrics(
    target: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = (probability >= threshold).astype(int)
    return {
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "risk_precision": float(
            precision_score(target, predicted, pos_label=1, zero_division=0)
        ),
        "risk_recall": float(
            recall_score(target, predicted, pos_label=1, zero_division=0)
        ),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier": float(np.mean((probability - target) ** 2)),
    }


def choose_inner_threshold(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    unique = np.unique(np.clip(probability.astype(float), 0.0, 1.0))
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                unique,
                (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else unique,
            ]
        )
    )
    # Evaluate every registered unique probability and midpoint in O(n log n)
    # instead of recomputing a full confusion matrix for every threshold.
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_target = target[order].astype(np.int64)
    cumulative_positive = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(sorted_target, dtype=np.int64)]
    )
    cut = np.searchsorted(sorted_probability, candidates, side="left")
    total = len(target)
    total_positive = int(sorted_target.sum())
    total_negative = total - total_positive
    tp = total_positive - cumulative_positive[cut]
    predicted_positive = total - cut
    fp = predicted_positive - tp
    fn = total_positive - tp
    tn = total_negative - fp

    risk_precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) != 0
    )
    risk_recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) != 0
    )
    risk_f1 = np.divide(
        2 * risk_precision * risk_recall,
        risk_precision + risk_recall,
        out=np.zeros_like(risk_precision),
        where=(risk_precision + risk_recall) != 0,
    )
    safe_precision = np.divide(
        tn, tn + fn, out=np.zeros_like(tn, dtype=float), where=(tn + fn) != 0
    )
    safe_recall = np.divide(
        tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0
    )
    safe_f1 = np.divide(
        2 * safe_precision * safe_recall,
        safe_precision + safe_recall,
        out=np.zeros_like(safe_precision),
        where=(safe_precision + safe_recall) != 0,
    )
    macro_f1 = (risk_f1 + safe_f1) / 2.0
    balanced = (risk_recall + safe_recall) / 2.0
    rank = np.lexsort(
        (
            np.arange(len(candidates)),
            -np.abs(candidates - 0.5),
            risk_precision,
            risk_recall,
            macro_f1,
        )
    )
    index = int(rank[-1])
    return {
        "threshold": float(candidates[index]),
        "macro_f1": float(macro_f1[index]),
        "balanced_accuracy": float(balanced[index]),
        "risk_precision": float(risk_precision[index]),
        "risk_recall": float(risk_recall[index]),
        "pr_auc": float(average_precision_score(target, probability)),
        "brier": float(np.mean((probability - target) ** 2)),
    }


def _normalise_parameter_values(parameters: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in parameters.items():
        if isinstance(value, np.integer):
            result[key] = int(value)
        elif isinstance(value, np.floating):
            result[key] = float(value)
        else:
            result[key] = value
    return result


def _oulad_search_configurations(
    model: str, specification: dict[str, Any], outer_fold: int
) -> list[dict[str, Any]]:
    space = specification["search"]["spaces"][model]
    budget = int(specification["search"]["budget_per_model_per_outer_fold"][model])
    offset = MODEL_ORDER.index(model) * 101
    return [
        _normalise_parameter_values(value)
        for value in ParameterSampler(
            space, n_iter=budget, random_state=3407 + outer_fold + offset
        )
    ]


def run_oulad_comparators(context: CompletionContext) -> None:
    specification = context.protocol["datasets"]["oulad"]
    v51_protocol = load_protocol("oulad")
    v4_protocol = yaml.safe_load(
        (ROOT / "configs/oulad_v4_protocol.yaml").read_text(encoding="utf-8")
    )
    data = load_v4_data(ROOT / "data/processed/study_c_oulad", v4_protocol)
    if len(data.v2.development_indices) != int(specification["expected_records"]):
        raise RuntimeError("OULAD development record count mismatch")
    if set(data.development_manifest["role"].astype(str)) != {"historical_development"}:
        raise RuntimeError("Future OULAD role entered comparator completion")
    output_root = ARTIFACT_ROOT / "oulad"
    fold_root = output_root / "folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    split_hash = specification["split"]["sha256"]
    feature_hash = context.feature_contract_hashes["oulad"]
    all_search: list[dict[str, Any]] = []
    all_selected: list[dict[str, Any]] = []
    all_runtime: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        for outer_fold in range(int(specification["split"]["outer_folds"])):
            fold_path = fold_root / f"{model_name}_outer_{outer_fold}.parquet"
            state_path = fold_root / f"{model_name}_outer_{outer_fold}.json"
            if fold_path.is_file() and state_path.is_file():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if (
                    state.get("status") == "PASS"
                    and state.get("prediction_sha256") == sha256_file(fold_path)
                    and state.get("protocol_hash") == context.protocol_hash
                ):
                    all_search.extend(state["search_rows"])
                    all_selected.append(state["selected"])
                    all_runtime.append(state["runtime"])
                    continue
            started = time.perf_counter()
            outer_train, outer_validation = data.v2.outer_indices(outer_fold)
            inner_manifest = build_v4_inner_manifest(data, outer_fold, v4_protocol)
            inner_splits = [
                manifest_indices(data.v2, inner_manifest, int(inner_fold))
                for inner_fold in sorted(inner_manifest["inner_fold"].unique())
            ]
            configurations = _oulad_search_configurations(
                model_name, specification, outer_fold
            )
            fold_search: list[dict[str, Any]] = []
            for trial, parameters in enumerate(configurations):
                targets: list[np.ndarray] = []
                probabilities: list[np.ndarray] = []
                trial_started = time.perf_counter()
                for train_indices, validation_indices in inner_splits:
                    train_x, train_y, processor = _oulad_matrix(
                        data, train_indices, train_indices
                    )
                    validation_x, validation_y, _ = _oulad_matrix(
                        data, train_indices, validation_indices, processor
                    )
                    estimator = _oulad_model(
                        model_name, parameters, 3407, train_y
                    )
                    estimator.fit(train_x, train_y)
                    targets.append(validation_y)
                    probabilities.append(estimator.predict_proba(validation_x)[:, 1])
                target = np.concatenate(targets)
                probability = np.concatenate(probabilities)
                threshold = choose_inner_threshold(target, probability)
                fold_search.append(
                    {
                        "dataset": "oulad",
                        "model_id": model_name,
                        "outer_fold": outer_fold,
                        "trial": trial,
                        "config_hash": canonical_hash(parameters),
                        "parameters": json.dumps(parameters, sort_keys=True),
                        **threshold,
                        "runtime_seconds": time.perf_counter() - trial_started,
                        "status": "COMPLETE",
                    }
                )
            best = sorted(
                fold_search,
                key=lambda row: (
                    -row["macro_f1"],
                    -row["balanced_accuracy"],
                    -row["pr_auc"],
                    row["brier"],
                    row["trial"],
                ),
            )[0]
            parameters = json.loads(best["parameters"])
            config_hash = best["config_hash"]

            # Recompute the selected threshold from all five registered seed
            # probabilities on the same pooled inner-OOF records.
            threshold_seed_rows: list[pd.DataFrame] = []
            for inner_fold, (train_indices, validation_indices) in enumerate(inner_splits):
                train_x, train_y, processor = _oulad_matrix(
                    data, train_indices, train_indices
                )
                validation_x, validation_y, _ = _oulad_matrix(
                    data, train_indices, validation_indices, processor
                )
                for seed in SEEDS:
                    estimator = _oulad_model(model_name, parameters, seed, train_y)
                    estimator.fit(train_x, train_y)
                    probability = estimator.predict_proba(validation_x)[:, 1]
                    threshold_seed_rows.append(
                        pd.DataFrame(
                            {
                                "record_position": validation_indices,
                                "target": validation_y,
                                "seed": seed,
                                "probability": probability,
                                "inner_fold": inner_fold,
                            }
                        )
                    )
            threshold_frame = pd.concat(threshold_seed_rows, ignore_index=True)
            threshold_ensemble = (
                threshold_frame.groupby(["record_position", "target"], as_index=False)[
                    "probability"
                ]
                .mean()
                .sort_values("record_position")
            )
            threshold_result = choose_inner_threshold(
                threshold_ensemble["target"].to_numpy(dtype=int),
                threshold_ensemble["probability"].to_numpy(dtype=float),
            )

            train_x, train_y, processor = _oulad_matrix(
                data, outer_train, outer_train
            )
            validation_x, validation_y, _ = _oulad_matrix(
                data, outer_train, outer_validation, processor
            )
            prediction_rows: list[dict[str, Any]] = []
            for seed in SEEDS:
                estimator = _oulad_model(model_name, parameters, seed, train_y)
                estimator.fit(train_x, train_y)
                probability = estimator.predict_proba(validation_x)[:, 1]
                predicted = (probability >= threshold_result["threshold"]).astype(int)
                for position, source_index in enumerate(outer_validation):
                    prediction_rows.append(
                        {
                            "dataset": "oulad",
                            "model_id": model_name,
                            "record_id": str(data.base.record_ids[source_index]),
                            "id_student": int(data.groups[source_index]),
                            "code_module": str(
                                data.base.cohort.iloc[source_index]["code_module"]
                            ),
                            "code_presentation": str(
                                data.base.cohort.iloc[source_index]["code_presentation"]
                            ),
                            "true_label": int(validation_y[position]),
                            "predicted_label": int(predicted[position]),
                            "outer_fold": outer_fold,
                            "inner_protocol_id": f"{context.protocol['protocol_id']}:oulad:outer-{outer_fold}",
                            "seed": seed,
                            "p_not_at_risk": float(1.0 - probability[position]),
                            "p_at_risk": float(probability[position]),
                            "threshold": float(threshold_result["threshold"]),
                            "config_hash": config_hash,
                            "split_manifest_hash": split_hash,
                            "feature_contract_hash": feature_hash,
                            "run_id": context.run_id,
                            "scope": "development_oof",
                            "forecast": "F2_MIDDLE",
                        }
                    )
            predictions = pd.DataFrame(prediction_rows)
            atomic_parquet(fold_path, predictions)
            selected = {
                "dataset": "oulad",
                "model_id": model_name,
                "outer_fold": outer_fold,
                "parameters": parameters,
                "config_hash": config_hash,
                "trial": best["trial"],
                "single_seed_search_metrics": {
                    key: best[key]
                    for key in (
                        "macro_f1",
                        "balanced_accuracy",
                        "pr_auc",
                        "brier",
                        "threshold",
                    )
                },
                "five_seed_inner_oof_threshold": threshold_result,
            }
            runtime = {
                "dataset": "oulad",
                "model_id": model_name,
                "outer_fold": outer_fold,
                "runtime_seconds": time.perf_counter() - started,
                "records": len(outer_validation),
                "seeds": len(SEEDS),
                "hostname": platform.node(),
            }
            state = {
                "status": "PASS",
                "protocol_hash": context.protocol_hash,
                "prediction_sha256": sha256_file(fold_path),
                "selected": selected,
                "search_rows": fold_search,
                "runtime": runtime,
            }
            atomic_json(state_path, state)
            all_search.extend(fold_search)
            all_selected.append(selected)
            all_runtime.append(runtime)
            _write_global_state(
                "OULAD_COMPARATORS",
                f"{model_name}:outer-{outer_fold}",
                completed_folds=len(all_selected),
                total_folds=len(MODEL_ORDER)
                * int(specification["split"]["outer_folds"]),
            )
    predictions = pd.concat(
        [
            pd.read_parquet(fold_root / f"{model}_outer_{fold}.parquet")
            for model in MODEL_ORDER
            for fold in range(int(specification["split"]["outer_folds"]))
        ],
        ignore_index=True,
    )
    completion_ids = {
        model: f"{model}_oulad" for model in MODEL_ORDER
    }
    predictions["model_id"] = predictions["model_id"].map(completion_ids)
    for row in all_search:
        row["model_id"] = completion_ids[row["model_id"]]
    for row in all_selected:
        row["model_id"] = completion_ids[row["model_id"]]
    atomic_parquet(output_root / "comparator_seed_predictions.parquet", predictions)
    atomic_csv(output_root / "search_trials.csv", pd.DataFrame(all_search))
    atomic_json(output_root / "selected_configs.json", all_selected)
    atomic_json(
        output_root / "run_state.json",
        {
            "status": "COMPLETE",
            "protocol_hash": context.protocol_hash,
            "models": list(MODEL_ORDER),
            "outer_folds_complete": len(all_selected),
            "prediction_sha256": sha256_file(
                output_root / "comparator_seed_predictions.parquet"
            ),
            "future_oulad_executed": False,
        },
    )
    _append_runtime(all_runtime)


def _append_runtime(rows: list[dict[str, Any]]) -> None:
    path = ARTIFACT_ROOT / "runtime_resources.csv"
    current = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    combined = pd.concat([current, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["dataset", "model_id", "outer_fold"], keep="last"
    ).sort_values(["dataset", "model_id", "outer_fold"])
    atomic_csv(path, combined)


def _write_global_state(phase: str, detail: str, **extra: Any) -> None:
    atomic_json(
        ARTIFACT_ROOT / "run_state.json",
        {
            "status": "RUNNING",
            "phase": phase,
            "detail": detail,
            "updated_unix": time.time(),
            **extra,
        },
    )


def initialise() -> CompletionContext:
    context = build_context()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    if not (ARTIFACT_ROOT / "no_change_guard.json").is_file():
        create_no_change_guard()
    write_protocol_snapshots(context)
    guard = verify_no_change_guard()
    if guard["status"] != "PASS":
        raise RuntimeError(f"Immutable evidence guard failed: {guard}")
    return context


def run_training(dataset: str) -> None:
    context = initialise()
    if dataset in {"student_mat", "all"}:
        _write_global_state("UCI_XGBOOST", "student_mat")
        run_uci_xgboost("student_mat", context)
    if dataset in {"student_por", "all"}:
        _write_global_state("UCI_XGBOOST", "student_por")
        run_uci_xgboost("student_por", context)
    if dataset in {"oulad", "all"}:
        _write_global_state("OULAD_COMPARATORS", "starting")
        run_oulad_comparators(context)
    _write_global_state("TRAINING_COMPLETE", dataset)


__all__ = [
    "ARTIFACT_ROOT",
    "MODEL_ORDER",
    "SEEDS",
    "build_context",
    "create_no_change_guard",
    "initialise",
    "run_training",
    "verify_no_change_guard",
]
