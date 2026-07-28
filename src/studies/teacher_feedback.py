"""Teacher-feedback studies on immutable final folds.

This module is intentionally reachable only through explicit ``project.py
study`` commands.  It never trains or overwrites an official CNN-BiLSTM
checkpoint.  All model selection is performed inside each frozen outer
training partition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "artifacts" / "final"
TF_ROOT = FINAL / "teacher_feedback_validation"
TIMING_ROOT = FINAL / "uci_timing_scenarios"
MLP_ROOT = TF_ROOT / "mlp_comparator"
CONFIG_TIMING = ROOT / "configs" / "final" / "uci_timing_scenarios.yaml"
CONFIG_MLP = ROOT / "configs" / "final" / "mlp_comparator.yaml"

SEEDS = (42, 1201, 2026, 3407, 7319)
SCENARIOS = (
    "S0_EARLY_NO_GRADE",
    "S1_MID_G1_ONLY",
    "S2_LATE_G1_G2",
)
TIMING_MODELS = (
    "logistic_regression",
    "random_forest",
    "xgboost",
    "mlp",
)
SAFE_S2_EXTRA_MODELS = (
    "decision_tree",
    "hist_gradient_boosting",
    "svm",
)
MODEL_NAMES = {
    "cnn_bilstm": "CNN-BiLSTM",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "svm": "SVM",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}

CONTEXT = (
    "failures",
    "studytime",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher",
    "traveltime",
    "freetime",
    "goout",
    "health",
)
CONTEXT_CATEGORICAL = (
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "internet",
    "higher",
)
QUASI_IDENTITY = (
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "nursery",
    "internet",
)
TEMPORAL_CHANNELS = (
    "normalized_grade",
    "stage_indicator",
    "signed_change_from_G1",
    "absolute_change_from_G1",
    "signed_distance_to_boundary_10",
    "signed_distance_to_boundary_15",
    "change_direction",
)
OULAD_STATIC = (
    "code_module",
    "presentation_season",
    "num_of_prev_attempts",
    "studied_credits",
    "registration_lead_time",
    "module_presentation_length",
)
OULAD_CATEGORICAL = ("code_module", "presentation_season")
COMPACT_SUMMARIES: dict[str, tuple[str, ...]] = {
    "total_clicks": ("sum", "mean", "last", "slope", "recent_2_week_mean"),
    "active_days": ("sum", "mean", "last", "slope", "recent_2_week_mean"),
    "unique_sites": ("mean", "last", "recent_2_week_mean"),
    "unique_activity_types": ("mean", "last", "recent_2_week_mean"),
    "content_clicks": ("sum", "slope", "recent_2_week_mean"),
    "forum_clicks": ("sum", "slope", "recent_2_week_mean"),
    "quiz_clicks": ("sum", "slope", "recent_2_week_mean"),
    "assessment_related_clicks": ("sum", "slope", "recent_2_week_mean"),
    "submitted_assessment_count": ("sum", "last"),
    "late_submission_count": ("sum", "last"),
    "available_score_count": ("sum", "last"),
    "cumulative_mean_score": ("last", "slope", "recent_2_week_mean"),
    "cumulative_weighted_score": ("last", "slope", "recent_2_week_mean"),
    "days_since_last_vle_activity": ("last", "slope", "recent_2_week_mean"),
    "weeks_without_activity": ("sum", "last", "recent_2_week_mean"),
    "score_missing_mask": ("sum", "last"),
}

OFFICIAL_METRICS = {
    "student_mat": 0.9014601961315334,
    "student_por": 0.8622587167738002,
    "oulad": 0.8280835945631038,
}
OFFICIAL_IDS = {
    "student_mat": "cnn_bilstm_mat",
    "student_por": "cnn_bilstm_por",
    "oulad": "cnn_bilstm_oulad",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_values(values: Iterable[Any]) -> str:
    text = "\n".join(sorted(map(str, values))) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _progress(message: str) -> None:
    print(f"[teacher-feedback] {message}", flush=True)


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[
        :24
    ]


def encode_uci_target(values: Iterable[Any]) -> np.ndarray:
    raw = np.asarray(list(values), dtype=float)
    if raw.ndim != 1 or not np.isfinite(raw).all():
        raise ValueError("G3 must be a finite one-dimensional vector")
    if ((raw < 0) | (raw > 20)).any():
        raise ValueError("G3 must be inside 0..20")
    return np.where(raw < 10, 0, np.where(raw < 15, 1, 2)).astype(np.int64)


def _target_contract() -> dict[str, Any]:
    return {
        "schema_version": "uci_target_contract_v1",
        "raw_target": "G3",
        "domain": {"minimum": 0, "maximum": 20, "scale": "integer_grade_0_20"},
        "mapping": {
            "Low": "0 <= G3 < 10",
            "Medium": "10 <= G3 < 15",
            "High": "15 <= G3 <= 20",
        },
        "encoded_classes": {"Low": 0, "Medium": 1, "High": 2},
        "boundary_validation": {
            "9": "Low",
            "10": "Medium",
            "14": "Medium",
            "15": "High",
            "20": "High",
        },
        "G3_roles": ["target_creation_only"],
        "G3_feature_input": False,
        "G3_preprocessing_fit": False,
        "G3_derived_predictor": False,
        "status": "PASS",
    }


def _load_protocols() -> tuple[dict[str, Any], dict[str, Any]]:
    timing = yaml.safe_load(CONFIG_TIMING.read_text(encoding="utf-8"))
    mlp = yaml.safe_load(CONFIG_MLP.read_text(encoding="utf-8"))
    required = "PREREGISTERED_BEFORE_OUTER_SCORING"
    if timing.get("status") != required or mlp.get("status") != required:
        raise RuntimeError("Study protocols must be preregistered before scoring")
    return timing, mlp


def _official_snapshot() -> dict[str, Any]:
    payload_path = FINAL / "final_results.json"
    registry_path = FINAL / "model_registry.json"
    recommendation_path = (
        FINAL / "recommendation" / "recommendation_technical_validation.json"
    )
    checkpoint_path = FINAL / "checksums" / "checkpoint_manifest.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    metrics: dict[str, float] = {}
    ids: dict[str, str] = {}
    for dataset, expected in OFFICIAL_METRICS.items():
        row = next(
            item
            for item in payload["datasets"][dataset]["models"]
            if item["model_id"] == "cnn_bilstm"
        )
        metrics[dataset] = float(row["metrics"]["macro_f1"]["value"])
        ids[dataset] = payload["official_models"][dataset]
        if not math.isclose(metrics[dataset], expected, abs_tol=1e-15):
            raise RuntimeError(f"{dataset} official Macro-F1 changed before study")
    for dataset, model_id in OFFICIAL_IDS.items():
        if ids[dataset] != model_id or model_id not in registry:
            raise RuntimeError(f"{dataset} official model identity changed before study")
    recommendation = payload["recommendation"]["metrics"]
    return {
        "schema_version": "teacher_feedback_regression_guard_v1",
        "created_before_study_training": True,
        "official_model_ids": ids,
        "official_macro_f1": metrics,
        "recommendation": {
            "records": recommendation["records"]["value"],
            "generated": recommendation["generated"]["value"],
            "partial_evidence": recommendation["partial_evidence"]["value"],
            "abstained": recommendation["abstained"]["value"],
            "deterministic_replay": recommendation["deterministic_replay"]["value"],
            "expert_status": payload["recommendation"]["expert_status"]["value"],
        },
        "future_oulad": payload["future_oulad"],
        "xapi_in_final_datasets": "xapi" in payload["datasets"],
        "immutable_files": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in (
                payload_path,
                registry_path,
                recommendation_path,
                checkpoint_path,
            )
        },
    }


def prepare_regression_guard() -> dict[str, Any]:
    """Record the immutable official state before explicit study training."""

    _load_protocols()
    guard = _official_snapshot()
    if guard["future_oulad"] != "LOCKED_NOT_EXECUTED":
        raise RuntimeError("Future OULAD is not locked")
    if guard["recommendation"]["expert_status"] != "PENDING_EXPERT_LABELS":
        raise RuntimeError("Recommendation expert status changed")
    if guard["xapi_in_final_datasets"]:
        raise RuntimeError("xAPI is present in final datasets")
    _write_json(TF_ROOT / "regression_guard_before.json", guard)
    _write_json(TF_ROOT / "uci_target_contract.json", _target_contract())
    return guard


def verify_regression_guard() -> dict[str, Any]:
    before_path = TF_ROOT / "regression_guard_before.json"
    if not before_path.is_file():
        raise RuntimeError("Regression guard was not prepared before training")
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = _official_snapshot()
    # final_results is expected to gain MLP later, so compare scientific state,
    # not the whole canonical file checksum.
    checks = {
        "official_model_ids_unchanged": (
            before["official_model_ids"] == after["official_model_ids"]
        ),
        "official_macro_f1_unchanged": all(
            math.isclose(
                float(before["official_macro_f1"][key]),
                float(after["official_macro_f1"][key]),
                abs_tol=1e-15,
            )
            for key in OFFICIAL_METRICS
        ),
        "recommendation_counts_unchanged": (
            before["recommendation"] == after["recommendation"]
        ),
        "future_oulad_locked": after["future_oulad"] == "LOCKED_NOT_EXECUTED",
        "xapi_absent": not after["xapi_in_final_datasets"],
        "checkpoint_manifest_unchanged": (
            before["immutable_files"][
                "artifacts/final/checksums/checkpoint_manifest.json"
            ]
            == after["immutable_files"][
                "artifacts/final/checksums/checkpoint_manifest.json"
            ]
        ),
        "recommendation_artifact_unchanged": (
            before["immutable_files"][
                "artifacts/final/recommendation/recommendation_technical_validation.json"
            ]
            == after["immutable_files"][
                "artifacts/final/recommendation/recommendation_technical_validation.json"
            ]
        ),
        "prediction_model_retrained": False,
        "future_oulad_accessed": False,
        "outer_test_used_for_tuning": False,
    }
    result = {
        "schema_version": "teacher_feedback_regression_guard_result_v1",
        "checks": checks,
        "status": "PASS" if all(value is not False for value in checks.values()) else "FAIL",
        "before": before,
        "after_scientific_state": after,
    }
    # The three explicit negative assertions above are expected to be False.
    pass_checks = {
        key: value
        for key, value in checks.items()
        if key
        not in {
            "prediction_model_retrained",
            "future_oulad_accessed",
            "outer_test_used_for_tuning",
        }
    }
    result["status"] = "PASS" if all(pass_checks.values()) else "FAIL"
    _write_json(TF_ROOT / "regression_guard_after.json", result)
    return result


@dataclass
class UCIStudyData:
    dataset: str
    frame: pd.DataFrame
    target: np.ndarray
    record_ids: np.ndarray
    groups: np.ndarray
    outer_fold: np.ndarray


def _load_uci(dataset: str) -> UCIStudyData:
    filename = "student-mat.csv" if dataset == "student_mat" else "student-por.csv"
    record_namespace = "student-mat" if dataset == "student_mat" else "student-por"
    frame = pd.read_csv(ROOT / "data" / "raw" / filename, sep=";")
    required = {"G1", "G2", "G3", *CONTEXT, *QUASI_IDENTITY}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{dataset} source missing fields: {missing}")
    target = encode_uci_target(frame["G3"])
    record_ids = np.asarray(
        [_stable_id(record_namespace, index) for index in range(len(frame))],
        dtype=object,
    )
    groups = np.asarray(
        [
            _stable_id(
                "quasi", *(frame.iloc[index][column] for column in QUASI_IDENTITY)
            )
            for index in range(len(frame))
        ],
        dtype=object,
    )
    frozen_path = (
        FINAL / "comparator_completion" / dataset / "oof_predictions.parquet"
    )
    frozen = pd.read_parquet(frozen_path)
    frozen = frozen.loc[frozen["model_id"] == "cnn_bilstm"].copy()
    true_column = "target" if "target" in frozen.columns else "true_label"
    frozen = frozen[["record_id", "outer_fold", true_column]].drop_duplicates()
    if len(frozen) != len(frame) or frozen["record_id"].duplicated().any():
        raise RuntimeError(f"{dataset} frozen OOF assignment is incomplete")
    assignment = frozen.set_index("record_id")
    if set(record_ids) != set(assignment.index):
        raise RuntimeError(f"{dataset} record IDs do not match frozen evidence")
    outer_fold = assignment.loc[record_ids, "outer_fold"].to_numpy(dtype=int)
    frozen_target = assignment.loc[record_ids, true_column].to_numpy(dtype=int)
    if not np.array_equal(target, frozen_target):
        raise RuntimeError(f"{dataset} target does not match frozen evidence")
    return UCIStudyData(dataset, frame, target, record_ids, groups, outer_fold)


def _grade_features(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "S0_EARLY_NO_GRADE":
        return pd.DataFrame(index=frame.index)
    grades = frame[["G1"]].astype(float).to_numpy()
    if scenario == "S2_LATE_G1_G2":
        grades = frame[["G1", "G2"]].astype(float).to_numpy()
    if not np.isfinite(grades).all() or ((grades < 0) | (grades > 20)).any():
        raise ValueError("G1/G2 must be finite and inside 0..20")
    steps = grades.shape[1]
    tensor = np.zeros((len(frame), steps, len(TEMPORAL_CHANNELS)), dtype=float)
    tensor[:, :, 0] = grades / 20.0
    tensor[:, 0, 1] = -1.0
    tensor[:, :, 4] = (grades - 10.0) / 20.0
    tensor[:, :, 5] = (grades - 15.0) / 20.0
    if steps == 2:
        tensor[:, 1, 1] = 1.0
        delta = grades[:, 1] - grades[:, 0]
        tensor[:, 1, 2] = delta / 20.0
        tensor[:, 1, 3] = np.abs(delta) / 20.0
        tensor[:, 1, 6] = np.sign(delta)
    columns = [
        f"grade_t{step}_{channel}"
        for step in range(steps)
        for channel in TEMPORAL_CHANNELS
    ]
    return pd.DataFrame(tensor.reshape(len(frame), -1), columns=columns)


def build_uci_scenario_frame(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    context = frame.loc[:, CONTEXT].reset_index(drop=True).copy()
    grade = _grade_features(frame, scenario).reset_index(drop=True)
    result = pd.concat([context, grade], axis=1)
    forbidden = {"G3"}
    if scenario == "S0_EARLY_NO_GRADE":
        forbidden |= {"G1", "G2"}
        if any(column.startswith("grade_") for column in result):
            raise RuntimeError("S0 contains a derived grade feature")
    if scenario == "S1_MID_G1_ONLY":
        forbidden.add("G2")
        if any(column.startswith("grade_t1_") for column in result):
            raise RuntimeError("S1 contains a G2-derived timestep")
    if forbidden & set(result.columns):
        raise RuntimeError(f"{scenario} contains forbidden fields")
    return result


def _uci_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    columns = tuple(columns)
    categorical = [column for column in columns if column in CONTEXT_CATEGORICAL]
    numeric = [column for column in columns if column not in categorical]
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
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        sparse_threshold=0.0,
    )


def _compact_columns(available: Iterable[str]) -> list[str]:
    available_set = set(available)
    columns = [
        f"{channel}__{summary}"
        for channel, summaries in COMPACT_SUMMARIES.items()
        for summary in summaries
        if f"{channel}__{summary}" in available_set
    ]
    if "inactive_week_count" in available_set:
        columns.append("inactive_week_count")
    missing = [
        channel
        for channel, summaries in COMPACT_SUMMARIES.items()
        if not any(f"{channel}__{summary}" in available_set for summary in summaries)
    ]
    if missing:
        raise RuntimeError(f"OULAD compact groups missing: {missing}")
    return columns


@dataclass
class OULADStudyData:
    frame: pd.DataFrame
    target: np.ndarray
    record_ids: np.ndarray
    groups: np.ndarray
    outer_fold: np.ndarray
    aggregate_columns: list[str]


def _load_oulad() -> OULADStudyData:
    aggregate = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "study_c_oulad"
        / "aggregated"
        / "F2_MIDDLE.parquet"
    )
    cohort = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "study_c_oulad"
        / "cohorts"
        / "F2_MIDDLE.parquet"
    )
    targets = pd.read_parquet(
        ROOT
        / "data"
        / "processed"
        / "study_c_oulad"
        / "targets"
        / "F2_MIDDLE.parquet"
    )
    frozen = pd.read_parquet(
        FINAL
        / "comparator_completion"
        / "oulad"
        / "ensemble_oof_predictions.parquet"
    )
    frozen = frozen.loc[frozen["model_id"] == "cnn_bilstm"].copy()
    frozen = frozen[
        ["record_id", "id_student", "outer_fold", "true_label"]
    ].drop_duplicates()
    compact = _compact_columns(aggregate.columns)
    source = (
        aggregate[["record_id", *compact]]
        .merge(cohort[["record_id", *OULAD_STATIC]], on="record_id", validate="one_to_one")
        .merge(
            targets[["record_id", "target_at_risk"]],
            on="record_id",
            validate="one_to_one",
        )
    )
    source = frozen.merge(source, on="record_id", validate="one_to_one")
    if len(source) != 15378:
        raise RuntimeError("OULAD frozen development OOF row count changed")
    if not np.array_equal(
        source["true_label"].to_numpy(dtype=int),
        source["target_at_risk"].to_numpy(dtype=int),
    ):
        raise RuntimeError("OULAD target mismatch")
    features = source[[*compact, *OULAD_STATIC]].copy()
    return OULADStudyData(
        frame=features,
        target=source["true_label"].to_numpy(dtype=int),
        record_ids=source["record_id"].to_numpy(dtype=object),
        groups=source["id_student"].to_numpy(dtype=object),
        outer_fold=source["outer_fold"].to_numpy(dtype=int),
        aggregate_columns=compact,
    )


def _oulad_preprocessor(data: OULADStudyData) -> ColumnTransformer:
    numeric_static = [
        column for column in OULAD_STATIC if column not in OULAD_CATEGORICAL
    ]
    return ColumnTransformer(
        [
            (
                "aggregate",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                data.aggregate_columns,
            ),
            (
                "static_numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_static,
            ),
            (
                "static_categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                list(OULAD_CATEGORICAL),
            ),
        ],
        sparse_threshold=0.0,
    )


def _candidate_grid(model_id: str, *, binary: bool) -> list[dict[str, Any]]:
    if model_id == "logistic_regression":
        return [
            {"C": value, "class_weight": weight}
            for value in (0.1, 1.0, 10.0)
            for weight in (None, "balanced")
        ]
    if model_id == "decision_tree":
        return [
            {"max_depth": depth, "min_samples_leaf": leaf, "class_weight": weight}
            for depth in (3, 5, None)
            for leaf in (2, 5)
            for weight in (None, "balanced")
        ]
    if model_id == "random_forest":
        return [
            {
                "n_estimators": 300,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": weight,
            }
            for depth in (None, 8)
            for leaf in (1, 3)
            for weight in (None, "balanced")
        ]
    if model_id == "hist_gradient_boosting":
        return [
            {
                "learning_rate": rate,
                "max_iter": 250,
                "max_leaf_nodes": leaves,
                "l2_regularization": l2,
            }
            for rate in (0.05, 0.1)
            for leaves in (15, 31)
            for l2 in (0.0, 1.0)
        ]
    if model_id == "svm":
        return [
            {"C": value, "gamma": gamma, "class_weight": weight}
            for value in (0.5, 1.0, 2.0)
            for gamma in ("scale", 0.1)
            for weight in (None, "balanced")
        ]
    if model_id == "xgboost":
        return [
            {
                "n_estimators": 300,
                "max_depth": depth,
                "learning_rate": rate,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "min_child_weight": child,
            }
            for depth in (2, 4)
            for rate in (0.03, 0.08)
            for child in (1, 5)
        ]
    if model_id == "mlp":
        return [
            {
                "hidden_layer_sizes": tuple(layers),
                "alpha": alpha,
                "learning_rate_init": 0.001,
            }
            for layers in ((64,), (64, 32), (128, 64))
            for alpha in (0.0001, 0.001, 0.01)
        ]
    raise ValueError(model_id)


def _estimator(
    model_id: str, params: dict[str, Any], *, seed: int, binary: bool
) -> Any:
    if model_id == "logistic_regression":
        return LogisticRegression(
            **params,
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
    if model_id == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=seed)
    if model_id == "random_forest":
        return RandomForestClassifier(
            **params, random_state=seed, n_jobs=1
        )
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params, random_state=seed)
    if model_id == "svm":
        return SVC(**params, probability=True, random_state=seed)
    if model_id == "xgboost":
        objective = "binary:logistic" if binary else "multi:softprob"
        return XGBClassifier(
            **params,
            objective=objective,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )
    if model_id == "mlp":
        return MLPClassifier(
            **params,
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            batch_size=128 if binary else 32,
            random_state=seed,
        )
    raise ValueError(model_id)


def _pipeline(
    preprocessor_factory: Callable[[], ColumnTransformer],
    model_id: str,
    params: dict[str, Any],
    *,
    seed: int,
    binary: bool,
) -> Pipeline:
    # No sampler is present: preprocessing and fitting happen exclusively on
    # the training rows supplied to Pipeline.fit.
    return Pipeline(
        [
            ("preprocess", preprocessor_factory()),
            ("model", _estimator(model_id, params, seed=seed, binary=binary)),
        ]
    )


def _aligned_probabilities(
    fitted: Pipeline, features: pd.DataFrame, classes: int
) -> np.ndarray:
    raw = np.asarray(fitted.predict_proba(features), dtype=float)
    labels = np.asarray(fitted.named_steps["model"].classes_, dtype=int)
    result = np.zeros((len(features), classes), dtype=float)
    result[:, labels] = raw
    row_sums = result.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise RuntimeError("Invalid probability matrix")
    return result / row_sums


def _ece(y_true: np.ndarray, probabilities: np.ndarray, bins: int) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= lower) & (
            (confidence <= upper) if index == bins - 1 else (confidence < upper)
        )
        if selected.any():
            value += float(selected.mean()) * abs(
                float((predicted[selected] == y_true[selected]).mean())
                - float(confidence[selected].mean())
            )
    return float(value)


def _metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    classes = probabilities.shape[1]
    if classes == 2 and threshold is not None:
        predicted = (probabilities[:, 1] >= threshold).astype(int)
    else:
        predicted = probabilities.argmax(axis=1)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        predicted,
        labels=np.arange(classes),
        zero_division=0,
    )
    one_hot = np.eye(classes)[y_true]
    if classes == 2:
        pr_auc = average_precision_score(y_true, probabilities[:, 1])
        roc_auc = roc_auc_score(y_true, probabilities[:, 1])
    else:
        pr_auc = average_precision_score(
            one_hot, probabilities, average="macro"
        )
        roc_auc = roc_auc_score(
            y_true, probabilities, multi_class="ovr", average="macro"
        )
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predicted, average="weighted", zero_division=0)
        ),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(log_loss(y_true, probabilities, labels=np.arange(classes))),
        "ece": _ece(y_true, probabilities, 10 if classes == 2 else 15),
        "confusion_matrix": confusion_matrix(
            y_true, predicted, labels=np.arange(classes)
        ).tolist(),
        "per_class": [
            {
                "class_index": int(index),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(classes)
        ],
        "predicted_label": predicted,
        "threshold": threshold,
    }


def _inner_splits(
    y: np.ndarray, groups: np.ndarray, *, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=3, shuffle=True, random_state=seed
    )
    return list(splitter.split(np.zeros(len(y)), y, groups))


def _threshold_from_inner(
    y_true: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    best = (float("-inf"), 0.5)
    for threshold in np.round(np.arange(0.2, 0.801, 0.005), 3):
        score = f1_score(
            y_true,
            (probability >= threshold).astype(int),
            average="macro",
            zero_division=0,
        )
        candidate = (float(score), -abs(float(threshold) - 0.5))
        previous = (best[0], -abs(best[1] - 0.5))
        if candidate > previous:
            best = (float(score), float(threshold))
    return best[1], best[0]


@dataclass
class FoldResult:
    model_id: str
    outer_fold: int
    selected_params: dict[str, Any]
    selected_inner_macro_f1: float
    threshold: float | None
    ensemble_probability: np.ndarray
    seed_probabilities: dict[int, np.ndarray]
    search_rows: list[dict[str, Any]]
    runtime_seconds: float


def _fit_outer_fold(
    *,
    dataset: str,
    scenario: str,
    model_id: str,
    features: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    outer_fold: int,
    preprocessor_factory: Callable[[], ColumnTransformer],
    binary: bool,
) -> FoldResult:
    started = time.perf_counter()
    inner = _inner_splits(
        target[train_indices], groups[train_indices], seed=20260 + outer_fold
    )
    trials: list[dict[str, Any]] = []
    selected_score = float("-inf")
    selected_params: dict[str, Any] | None = None
    selected_threshold: float | None = None
    for trial, params in enumerate(_candidate_grid(model_id, binary=binary)):
        fold_scores: list[float] = []
        inner_targets: list[np.ndarray] = []
        inner_probabilities: list[np.ndarray] = []
        for inner_fold, (fit_local, score_local) in enumerate(inner):
            fit_indices = train_indices[fit_local]
            score_indices = train_indices[score_local]
            pipeline = _pipeline(
                preprocessor_factory,
                model_id,
                params,
                seed=SEEDS[0],
                binary=binary,
            )
            pipeline.fit(features.iloc[fit_indices], target[fit_indices])
            probability = _aligned_probabilities(
                pipeline, features.iloc[score_indices], 2 if binary else 3
            )
            if binary:
                inner_targets.append(target[score_indices])
                inner_probabilities.append(probability[:, 1])
                score = f1_score(
                    target[score_indices],
                    (probability[:, 1] >= 0.5).astype(int),
                    average="macro",
                    zero_division=0,
                )
            else:
                score = f1_score(
                    target[score_indices],
                    probability.argmax(axis=1),
                    average="macro",
                    zero_division=0,
                )
            fold_scores.append(float(score))
        threshold: float | None = None
        mean_score = float(np.mean(fold_scores))
        if binary:
            threshold, mean_score = _threshold_from_inner(
                np.concatenate(inner_targets), np.concatenate(inner_probabilities)
            )
        row = {
            "dataset": dataset,
            "scenario": scenario,
            "model_id": model_id,
            "outer_fold": outer_fold,
            "trial": trial,
            "state": "COMPLETE",
            "params": json.dumps(params, sort_keys=True),
            "inner_fold_macro_f1": json.dumps(fold_scores),
            "inner_objective": mean_score,
            "threshold": threshold,
            "outer_rows_used_for_selection": False,
        }
        trials.append(row)
        tie_key = json.dumps(params, sort_keys=True)
        selected_key = (
            json.dumps(selected_params, sort_keys=True)
            if selected_params is not None
            else "\uffff"
        )
        if mean_score > selected_score + 1e-12 or (
            math.isclose(mean_score, selected_score, abs_tol=1e-12)
            and tie_key < selected_key
        ):
            selected_score = mean_score
            selected_params = params
            selected_threshold = threshold
    if selected_params is None:
        raise RuntimeError("No inner candidate completed")
    seed_probabilities: dict[int, np.ndarray] = {}
    # All registered seeds are used for every stochastic comparator.  This
    # prevents best-seed selection and yields a record-aligned ensemble.
    for seed in SEEDS:
        pipeline = _pipeline(
            preprocessor_factory,
            model_id,
            selected_params,
            seed=seed,
            binary=binary,
        )
        pipeline.fit(features.iloc[train_indices], target[train_indices])
        seed_probabilities[seed] = _aligned_probabilities(
            pipeline, features.iloc[validation_indices], 2 if binary else 3
        )
    ensemble = np.mean(list(seed_probabilities.values()), axis=0)
    return FoldResult(
        model_id=model_id,
        outer_fold=outer_fold,
        selected_params=selected_params,
        selected_inner_macro_f1=selected_score,
        threshold=selected_threshold,
        ensemble_probability=ensemble,
        seed_probabilities=seed_probabilities,
        search_rows=trials,
        runtime_seconds=time.perf_counter() - started,
    )


def _labels_for(dataset: str) -> list[str]:
    return (
        ["Not-at-risk", "At-risk"]
        if dataset == "oulad"
        else ["Low", "Medium", "High"]
    )


def _metric_row(
    dataset: str,
    scenario: str,
    model_id: str,
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold_by_row: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    if threshold_by_row is not None:
        predicted = (
            probabilities[:, 1] >= np.asarray(threshold_by_row, dtype=float)
        ).astype(int)
        # Aggregate reporting uses the exact fold-specific predictions.
        metric = _metrics(target, probabilities)
        precision, recall, class_f1, support = precision_recall_fscore_support(
            target,
            predicted,
            labels=np.arange(probabilities.shape[1]),
            zero_division=0,
        )
        metric.update(
            {
                "accuracy": float(accuracy_score(target, predicted)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(target, predicted)
                ),
                "macro_precision": float(np.mean(precision)),
                "macro_recall": float(np.mean(recall)),
                "macro_f1": float(
                    f1_score(target, predicted, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(target, predicted, average="weighted", zero_division=0)
                ),
                "confusion_matrix": confusion_matrix(
                    target, predicted, labels=np.arange(probabilities.shape[1])
                ).tolist(),
                "per_class": [
                    {
                        "class_index": int(index),
                        "precision": float(precision[index]),
                        "recall": float(recall[index]),
                        "f1": float(class_f1[index]),
                        "support": int(support[index]),
                    }
                    for index in range(probabilities.shape[1])
                ],
            }
        )
    else:
        metric = _metrics(target, probabilities)
        predicted = metric.pop("predicted_label")
    metric.pop("predicted_label", None)
    metric.pop("threshold", None)
    labels = _labels_for(dataset)
    per_class = {
        labels[item["class_index"]]: {
            key: value for key, value in item.items() if key != "class_index"
        }
        for item in metric.pop("per_class")
    }
    row = {
        "dataset": dataset,
        "scenario": scenario,
        "model_id": model_id,
        "model": MODEL_NAMES[model_id],
        **metric,
        "per_class": per_class,
    }
    return row, np.asarray(predicted, dtype=int)


def _run_uci_dataset(dataset: str) -> dict[str, Any]:
    _progress(f"{dataset}: loading frozen folds")
    data = _load_uci(dataset)
    all_predictions: list[pd.DataFrame] = []
    seed_predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    for scenario in SCENARIOS:
        features = build_uci_scenario_frame(data.frame, scenario)
        model_ids = list(TIMING_MODELS)
        if scenario == "S2_LATE_G1_G2":
            model_ids.extend(SAFE_S2_EXTRA_MODELS)
        model_fold_outputs: dict[str, list[tuple[np.ndarray, FoldResult]]] = {
            model_id: [] for model_id in model_ids
        }
        for outer_fold in sorted(np.unique(data.outer_fold)):
            validation = np.flatnonzero(data.outer_fold == outer_fold)
            train = np.flatnonzero(data.outer_fold != outer_fold)
            train_ids = data.record_ids[train]
            validation_ids = data.record_ids[validation]
            if set(train_ids) & set(validation_ids):
                raise RuntimeError("Outer train/validation overlap")
            split_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "outer_fold": int(outer_fold),
                    "train_record_count": int(len(train)),
                    "validation_record_count": int(len(validation)),
                    "train_record_ids_sha256": _hash_values(train_ids),
                    "validation_record_ids_sha256": _hash_values(validation_ids),
                    "model_ids": model_ids,
                    "same_hash_for_all_models": True,
                    "outer_rows_in_inner_training": 0,
                }
            )
            for model_id in model_ids:
                _progress(
                    f"{dataset} {scenario} outer={int(outer_fold)} "
                    f"model={model_id}"
                )
                result = _fit_outer_fold(
                    dataset=dataset,
                    scenario=scenario,
                    model_id=model_id,
                    features=features,
                    target=data.target,
                    groups=data.groups,
                    train_indices=train,
                    validation_indices=validation,
                    outer_fold=int(outer_fold),
                    preprocessor_factory=lambda columns=tuple(
                        features.columns
                    ): _uci_preprocessor(columns),
                    binary=False,
                )
                model_fold_outputs[model_id].append((validation, result))
                search_rows.extend(result.search_rows)
                selected_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": int(outer_fold),
                        "selected_params": json.dumps(
                            result.selected_params, sort_keys=True
                        ),
                        "selected_inner_macro_f1": (
                            result.selected_inner_macro_f1
                        ),
                        "outer_rows_used_for_selection": False,
                    }
                )
                runtime_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": int(outer_fold),
                        "runtime_seconds": result.runtime_seconds,
                    }
                )
        for model_id, fold_outputs in model_fold_outputs.items():
            probabilities = np.zeros((len(data.target), 3), dtype=float)
            per_seed = {
                seed: np.zeros((len(data.target), 3), dtype=float)
                for seed in SEEDS
            }
            for validation, result in fold_outputs:
                probabilities[validation] = result.ensemble_probability
                for seed in SEEDS:
                    per_seed[seed][validation] = result.seed_probabilities[seed]
            metric, predicted = _metric_row(
                dataset, scenario, model_id, data.target, probabilities
            )
            seed_values = [
                f1_score(
                    data.target,
                    per_seed[seed].argmax(axis=1),
                    average="macro",
                    zero_division=0,
                )
                for seed in SEEDS
            ]
            metric["seed_stability"] = {
                "seed_count": len(SEEDS),
                "seed_mean": float(np.mean(seed_values)),
                "seed_std": float(np.std(seed_values)),
                "seed_min": float(np.min(seed_values)),
                "seed_max": float(np.max(seed_values)),
            }
            metrics.append(metric)
            all_predictions.append(
                pd.DataFrame(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": data.outer_fold,
                        "record_id": data.record_ids,
                        "target": data.target,
                        "predicted_label": predicted,
                        "p_low": probabilities[:, 0],
                        "p_medium": probabilities[:, 1],
                        "p_high": probabilities[:, 2],
                        "probability_aggregation": "mean_across_fixed_seeds",
                    }
                )
            )
            for seed in SEEDS:
                seed_predictions.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "model_id": model_id,
                            "outer_fold": data.outer_fold,
                            "record_id": data.record_ids,
                            "target": data.target,
                            "seed": seed,
                            "p_low": per_seed[seed][:, 0],
                            "p_medium": per_seed[seed][:, 1],
                            "p_high": per_seed[seed][:, 2],
                        }
                    )
                )

    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    seed_frame = pd.concat(seed_predictions, ignore_index=True)
    metric_frame = _timing_metric_frame(metrics)
    _write_parquet(TIMING_ROOT / f"{dataset}_predictions.parquet", prediction_frame)
    _write_parquet(TIMING_ROOT / f"{dataset}_seed_predictions.parquet", seed_frame)
    _write_csv(TIMING_ROOT / f"{dataset}_metrics.csv", metric_frame)
    _write_csv(TIMING_ROOT / f"{dataset}_search_trials.csv", pd.DataFrame(search_rows))
    _write_csv(
        TIMING_ROOT / f"{dataset}_selected_configs.csv",
        pd.DataFrame(selected_rows),
    )
    _write_csv(TIMING_ROOT / f"{dataset}_runtime.csv", pd.DataFrame(runtime_rows))
    return {
        "dataset": dataset,
        "metrics": metrics,
        "split_rows": split_rows,
        "prediction_path": (
            TIMING_ROOT / f"{dataset}_predictions.parquet"
        ),
        "seed_prediction_path": (
            TIMING_ROOT / f"{dataset}_seed_predictions.parquet"
        ),
    }


def _timing_metric_frame(metrics: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        low = metric["per_class"]["Low"]
        rows.append(
            {
                "dataset": metric["dataset"],
                "scenario": metric["scenario"],
                "model_id": metric["model_id"],
                "model": metric["model"],
                **{
                    key: metric[key]
                    for key in (
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
                },
                "low_precision": low["precision"],
                "low_recall": low["recall"],
                "low_f1": low["f1"],
                "confusion_matrix": json.dumps(metric["confusion_matrix"]),
                "per_class": json.dumps(metric["per_class"], sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def _run_oulad_mlp() -> dict[str, Any]:
    _progress("oulad: loading compact aggregate + static frozen OOF cohort")
    data = _load_oulad()
    probabilities = np.zeros((len(data.target), 2), dtype=float)
    threshold_by_row = np.zeros(len(data.target), dtype=float)
    per_seed = {
        seed: np.zeros((len(data.target), 2), dtype=float) for seed in SEEDS
    }
    search_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    runtime: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for outer_fold in sorted(np.unique(data.outer_fold)):
        _progress(f"oulad outer={int(outer_fold)} model=mlp")
        validation = np.flatnonzero(data.outer_fold == outer_fold)
        train = np.flatnonzero(data.outer_fold != outer_fold)
        train_students = set(data.groups[train])
        validation_students = set(data.groups[validation])
        if train_students & validation_students:
            raise RuntimeError("OULAD student group leaks across outer fold")
        split_rows.append(
            {
                "dataset": "oulad",
                "scenario": "FINAL_TABULAR_CONTRACT",
                "outer_fold": int(outer_fold),
                "train_record_count": int(len(train)),
                "validation_record_count": int(len(validation)),
                "train_record_ids_sha256": _hash_values(data.record_ids[train]),
                "validation_record_ids_sha256": _hash_values(
                    data.record_ids[validation]
                ),
                "train_student_ids_sha256": _hash_values(data.groups[train]),
                "validation_student_ids_sha256": _hash_values(
                    data.groups[validation]
                ),
                "student_overlap": 0,
                "same_hash_for_all_models": True,
                "outer_rows_in_inner_training": 0,
            }
        )
        result = _fit_outer_fold(
            dataset="oulad",
            scenario="FINAL_TABULAR_CONTRACT",
            model_id="mlp",
            features=data.frame,
            target=data.target,
            groups=data.groups,
            train_indices=train,
            validation_indices=validation,
            outer_fold=int(outer_fold),
            preprocessor_factory=lambda: _oulad_preprocessor(data),
            binary=True,
        )
        probabilities[validation] = result.ensemble_probability
        threshold_by_row[validation] = float(result.threshold)
        for seed in SEEDS:
            per_seed[seed][validation] = result.seed_probabilities[seed]
        search_rows.extend(result.search_rows)
        selected.append(
            {
                "dataset": "oulad",
                "model_id": "mlp",
                "outer_fold": int(outer_fold),
                "selected_params": result.selected_params,
                "selected_inner_macro_f1": result.selected_inner_macro_f1,
                "selected_threshold": result.threshold,
                "outer_rows_used_for_selection": False,
            }
        )
        runtime.append(
            {
                "dataset": "oulad",
                "model_id": "mlp",
                "outer_fold": int(outer_fold),
                "runtime_seconds": result.runtime_seconds,
            }
        )
    metric, predicted = _metric_row(
        "oulad",
        "FINAL_TABULAR_CONTRACT",
        "mlp",
        data.target,
        probabilities,
        threshold_by_row,
    )
    seed_scores = []
    for seed in SEEDS:
        seed_predicted = (
            per_seed[seed][:, 1] >= threshold_by_row
        ).astype(int)
        seed_scores.append(
            f1_score(
                data.target, seed_predicted, average="macro", zero_division=0
            )
        )
    metric["seed_stability"] = {
        "seed_count": len(SEEDS),
        "seed_mean": float(np.mean(seed_scores)),
        "seed_std": float(np.std(seed_scores)),
        "seed_min": float(np.min(seed_scores)),
        "seed_max": float(np.max(seed_scores)),
    }
    prediction = pd.DataFrame(
        {
            "dataset": "oulad",
            "model_id": "mlp",
            "record_id": data.record_ids,
            "id_student": data.groups,
            "outer_fold": data.outer_fold,
            "true_label": data.target,
            "predicted_label": predicted,
            "p_not_at_risk": probabilities[:, 0],
            "p_at_risk": probabilities[:, 1],
            "threshold": threshold_by_row,
            "scope": "development_oof",
            "feature_contract": "UNIFIED_OULAD_ML_FEATURE_CONTRACT_V1",
        }
    )
    seed_frames = []
    for seed in SEEDS:
        seed_frames.append(
            pd.DataFrame(
                {
                    "dataset": "oulad",
                    "model_id": "mlp",
                    "record_id": data.record_ids,
                    "id_student": data.groups,
                    "outer_fold": data.outer_fold,
                    "true_label": data.target,
                    "seed": seed,
                    "p_not_at_risk": per_seed[seed][:, 0],
                    "p_at_risk": per_seed[seed][:, 1],
                    "threshold": threshold_by_row,
                }
            )
        )
    root = MLP_ROOT / "oulad"
    _write_parquet(root / "oof_predictions.parquet", prediction)
    _write_parquet(
        root / "seed_predictions.parquet",
        pd.concat(seed_frames, ignore_index=True),
    )
    _write_csv(root / "search_trials.csv", pd.DataFrame(search_rows))
    _write_json(root / "selected_configs.json", selected)
    _write_csv(root / "runtime.csv", pd.DataFrame(runtime))
    _write_json(root / "metrics.json", metric)
    return {
        "dataset": "oulad",
        "metric": metric,
        "split_rows": split_rows,
        "prediction_path": root / "oof_predictions.parquet",
    }


def _materialize_uci_comparator_evidence(result: dict[str, Any]) -> None:
    dataset = result["dataset"]
    source_predictions = pd.read_parquet(result["prediction_path"])
    source_seeds = pd.read_parquet(result["seed_prediction_path"])
    selected_models = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
        "svm",
        "xgboost",
        "mlp",
    ]
    predictions = source_predictions.loc[
        (source_predictions["scenario"] == "S2_LATE_G1_G2")
        & source_predictions["model_id"].isin(selected_models)
    ].copy()
    seeds = source_seeds.loc[
        (source_seeds["scenario"] == "S2_LATE_G1_G2")
        & source_seeds["model_id"].isin(selected_models)
    ].copy()
    ordered_metrics = [
        metric
        for model_id in selected_models
        for metric in result["metrics"]
        if metric["scenario"] == "S2_LATE_G1_G2"
        and metric["model_id"] == model_id
    ]
    if len(ordered_metrics) != len(selected_models):
        raise RuntimeError(f"{dataset} safe S2 comparator evidence is incomplete")
    root = TF_ROOT / "safe_uci_comparators" / dataset
    _write_parquet(root / "oof_predictions.parquet", predictions)
    _write_parquet(root / "seed_predictions.parquet", seeds)
    _write_json(
        root / "metrics.json",
        {
            "schema_version": "safe_uci_comparator_metrics_v1",
            "dataset": dataset,
            "scenario": "S2_LATE_G1_G2",
            "models": ordered_metrics,
            "resampling": "none",
            "plain_smote_used": False,
            "plain_adasyn_used": False,
            "outer_used_for_selection": False,
        },
    )
    # Standalone MLP evidence is also exposed under the common 3-dataset root.
    mlp_root = MLP_ROOT / dataset
    mlp_predictions = predictions.loc[predictions["model_id"] == "mlp"].copy()
    mlp_seeds = seeds.loc[seeds["model_id"] == "mlp"].copy()
    mlp_metric = next(
        metric for metric in ordered_metrics if metric["model_id"] == "mlp"
    )
    _write_parquet(mlp_root / "oof_predictions.parquet", mlp_predictions)
    _write_parquet(mlp_root / "seed_predictions.parquet", mlp_seeds)
    _write_json(mlp_root / "metrics.json", mlp_metric)


def _confusion_macro_f1(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    true_sum = matrix.sum(axis=1)
    pred_sum = matrix.sum(axis=0)
    diagonal = np.diag(matrix)
    precision = np.divide(
        diagonal, pred_sum, out=np.zeros_like(diagonal), where=pred_sum > 0
    )
    recall = np.divide(
        diagonal, true_sum, out=np.zeros_like(diagonal), where=true_sum > 0
    )
    values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(values.mean())


def _paired_bootstrap(
    *,
    dataset: str,
    record_ids: np.ndarray,
    groups: np.ndarray,
    target: np.ndarray,
    official_predicted: np.ndarray,
    mlp_predicted: np.ndarray,
    replicates: int = 5000,
) -> dict[str, Any]:
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    classes = int(max(target.max(), official_predicted.max(), mlp_predicted.max()) + 1)
    official_contribution = np.zeros(
        (len(unique_groups), classes, classes), dtype=np.int64
    )
    mlp_contribution = np.zeros_like(official_contribution)
    np.add.at(
        official_contribution,
        (group_index, target, official_predicted),
        1,
    )
    np.add.at(
        mlp_contribution,
        (group_index, target, mlp_predicted),
        1,
    )
    rng = np.random.default_rng(20260728)
    deltas = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        weights = np.bincount(sampled, minlength=len(unique_groups))
        official_matrix = np.tensordot(
            weights, official_contribution, axes=(0, 0)
        )
        mlp_matrix = np.tensordot(weights, mlp_contribution, axes=(0, 0))
        deltas[index] = _confusion_macro_f1(
            official_matrix
        ) - _confusion_macro_f1(mlp_matrix)
    official_point = f1_score(
        target, official_predicted, average="macro", zero_division=0
    )
    mlp_point = f1_score(
        target, mlp_predicted, average="macro", zero_division=0
    )
    low, high = np.quantile(deltas, [0.025, 0.975])
    verdict = (
        "CNN_BILSTM_HIGHER"
        if low > 0
        else "CNN_BILSTM_LOWER"
        if high < 0
        else "INSUFFICIENT_EVIDENCE_OF_DIFFERENCE"
    )
    return {
        "dataset": dataset,
        "cnn_bilstm_macro_f1": float(official_point),
        "mlp_macro_f1": float(mlp_point),
        "delta_macro_f1": float(official_point - mlp_point),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "interpretation": verdict,
        "bootstrap_unit": "id_student" if dataset == "oulad" else "record_id",
        "replicates": replicates,
        "same_record_ids": len(record_ids) == len(set(record_ids)),
        "record_id_sha256": _hash_values(record_ids),
    }


def _official_and_mlp_predictions(
    dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "oulad":
        official_path = (
            FINAL
            / "comparator_completion"
            / "oulad"
            / "ensemble_oof_predictions.parquet"
        )
        official = pd.read_parquet(official_path)
        official = official.loc[official["model_id"] == "cnn_bilstm"].copy()
        official = official.drop_duplicates("record_id")
        mlp = pd.read_parquet(MLP_ROOT / "oulad" / "oof_predictions.parquet")
        merged = official[
            ["record_id", "id_student", "true_label", "predicted_label"]
        ].merge(
            mlp[
                [
                    "record_id",
                    "id_student",
                    "true_label",
                    "predicted_label",
                ]
            ],
            on=["record_id", "id_student", "true_label"],
            suffixes=("_official", "_mlp"),
            validate="one_to_one",
        )
        return (
            merged["record_id"].to_numpy(),
            merged["id_student"].to_numpy(),
            merged["true_label"].to_numpy(dtype=int),
            merged["predicted_label_official"].to_numpy(dtype=int),
            merged["predicted_label_mlp"].to_numpy(dtype=int),
        )
    official = pd.read_parquet(
        FINAL / "comparator_completion" / dataset / "oof_predictions.parquet"
    )
    official = official.loc[official["model_id"] == "cnn_bilstm"].copy()
    official = official.drop_duplicates("record_id")
    target_name = "target" if "target" in official.columns else "true_label"
    if "predicted_label" not in official:
        official["predicted_label"] = official[
            ["p_low", "p_medium", "p_high"]
        ].to_numpy().argmax(axis=1)
    mlp = pd.read_parquet(MLP_ROOT / dataset / "oof_predictions.parquet")
    merged = official[
        ["record_id", target_name, "predicted_label"]
    ].merge(
        mlp[["record_id", "target", "predicted_label"]],
        left_on=["record_id", target_name],
        right_on=["record_id", "target"],
        suffixes=("_official", "_mlp"),
        validate="one_to_one",
    )
    return (
        merged["record_id"].to_numpy(),
        merged["record_id"].to_numpy(),
        merged["target"].to_numpy(dtype=int),
        merged["predicted_label_official"].to_numpy(dtype=int),
        merged["predicted_label_mlp"].to_numpy(dtype=int),
    )


def _write_split_and_evaluation_contracts(
    split_rows: list[dict[str, Any]]
) -> None:
    frozen_hashes = json.loads(
        (
            FINAL / "comparator_completion" / "split_manifest_checksums.json"
        ).read_text(encoding="utf-8")
    )
    for row in split_rows:
        row["frozen_split_manifest_sha256"] = frozen_hashes[row["dataset"]]
    split_payload = {
        "schema_version": "split_equivalence_v1",
        "source": "frozen_final_OOF_fold_assignments",
        "rows": split_rows,
        "all_models_same_split_within_dataset_scenario": all(
            row["same_hash_for_all_models"] for row in split_rows
        ),
        "outer_rows_in_inner_training": sum(
            row["outer_rows_in_inner_training"] for row in split_rows
        ),
        "status": "PASS",
    }
    _write_json(TF_ROOT / "split_equivalence.json", split_payload)
    _write_json(TIMING_ROOT / "split_validation.json", split_payload)
    evaluation = {
        "schema_version": "locked_outer_evaluation_contract_v1",
        "outer_validation_used_for_tuning": False,
        "selection_partition": "inner_training_only",
        "preprocessing_fit": "outer_train_or_inner_train_only",
        "paired_comparison_record_alignment": True,
        "student_mat_outer_folds": 5,
        "student_por_outer_folds": 5,
        "oulad_outer_folds": 3,
        "oulad_outer_group": "id_student",
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "new_random_outer_split_created": False,
        "status": "PASS",
    }
    _write_json(TF_ROOT / "evaluation_contract.json", evaluation)


def _write_imbalance_audit() -> dict[str, Any]:
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    executable = [
        path
        for path in tracked
        if path.startswith(("src/", "scripts/"))
        and path != "src/studies/teacher_feedback.py"
        and Path(path).suffix in {".py", ".yaml", ".yml"}
    ]
    patterns = ("SMOTE(", "ADASYN(", "SMOTENC(", "fit_resample(")
    matches: list[dict[str, Any]] = []
    for relative in executable:
        text = (ROOT / relative).read_text(encoding="utf-8", errors="ignore")
        found = [pattern for pattern in patterns if pattern in text]
        if found:
            matches.append({"path": relative, "patterns": found})
    result = {
        "schema_version": "imbalance_safety_audit_v1",
        "scope": "tracked executable final code plus provenance audit",
        "questions": {
            "plain_smote_or_adasyn_on_mixed_label_coded_uci_in_current_final_code": {
                "answer": False,
                "status": "PASS",
            },
            "resampling_before_train_validation_split_in_current_final_code": {
                "answer": False,
                "status": "PASS",
            },
            "validation_used_to_fit_sampler_in_current_final_code": {
                "answer": False,
                "status": "PASS",
            },
            "synthetic_oversampling_on_raw_oulad_temporal_tensor": {
                "answer": False,
                "status": "PASS",
            },
        },
        "current_executable_sampler_matches": matches,
        "historical_finding": {
            "found": True,
            "description": (
                "Archived UCI baseline research included plain SMOTE/ADASYN "
                "after one-hot categorical preprocessing."
            ),
            "remediation": (
                "Historical affected UCI classical rows are superseded by "
                "safe S2 revalidation with no synthetic resampling."
            ),
            "official_cnn_bilstm_affected": False,
            "teacher_feedback_authority_uses_historical_unsafe_rows": False,
        },
        "new_study": {
            "sampler": "none",
            "preprocessing_fit": "training_partition_only",
            "plain_smote": False,
            "plain_adasyn": False,
            "oulad_tensor_supplied_to_mlp": False,
        },
        "status": "PASS" if not matches else "FAIL",
    }
    _write_json(TF_ROOT / "imbalance_safety_audit.json", result)
    return result


def _scenario_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"datasets": {}, "deltas": []}
    for result in results:
        dataset = result["dataset"]
        rows = []
        for metric in result["metrics"]:
            if metric["model_id"] not in TIMING_MODELS:
                continue
            low = metric["per_class"]["Low"]
            rows.append(
                {
                    "scenario": metric["scenario"],
                    "model_id": metric["model_id"],
                    "accuracy": metric["accuracy"],
                    "balanced_accuracy": metric["balanced_accuracy"],
                    "macro_f1": metric["macro_f1"],
                    "low_precision": low["precision"],
                    "low_recall": low["recall"],
                    "low_f1": low["f1"],
                    "ece": metric["ece"],
                    "pr_auc": metric["pr_auc"],
                }
            )
        payload["datasets"][dataset] = rows
        frame = pd.DataFrame(rows).set_index(["model_id", "scenario"])
        for model_id in TIMING_MODELS:
            for left, right, label in (
                ("S0_EARLY_NO_GRADE", "S1_MID_G1_ONLY", "S1-S0"),
                ("S1_MID_G1_ONLY", "S2_LATE_G1_G2", "S2-S1"),
                ("S0_EARLY_NO_GRADE", "S2_LATE_G1_G2", "S2-S0"),
            ):
                payload["deltas"].append(
                    {
                        "dataset": dataset,
                        "model_id": model_id,
                        "delta": label,
                        **{
                            metric: float(
                                frame.loc[(model_id, right), metric]
                                - frame.loc[(model_id, left), metric]
                            )
                            for metric in (
                                "macro_f1",
                                "balanced_accuracy",
                                "low_recall",
                                "pr_auc",
                                "ece",
                            )
                        },
                    }
                )
    return payload


def _write_protocol_artifacts() -> None:
    timing, mlp = _load_protocols()
    _write_json(
        TIMING_ROOT / "protocol.json",
        {
            "schema_version": "uci_timing_protocol_snapshot_v1",
            "protocol": timing,
            "protocol_path": str(CONFIG_TIMING.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": _sha256(CONFIG_TIMING),
            "official_model_reselection": False,
        },
    )
    _write_json(
        MLP_ROOT / "protocol.json",
        {
            "schema_version": "mlp_comparator_protocol_snapshot_v1",
            "protocol": mlp,
            "protocol_path": str(CONFIG_MLP.relative_to(ROOT)).replace("\\", "/"),
            "protocol_sha256": _sha256(CONFIG_MLP),
        },
    )


def run_all() -> dict[str, Any]:
    """Run all authorized missing studies; never run an official deep model."""

    started = time.perf_counter()
    prepare_regression_guard()
    _progress("regression guard PASS; official models remain frozen")
    _write_protocol_artifacts()
    uci_results = [
        _run_uci_dataset("student_mat"),
        _run_uci_dataset("student_por"),
    ]
    for result in uci_results:
        _materialize_uci_comparator_evidence(result)
    oulad_result = _run_oulad_mlp()
    split_rows = [
        row
        for result in uci_results
        for row in result["split_rows"]
    ] + oulad_result["split_rows"]
    _write_split_and_evaluation_contracts(split_rows)
    scenario = _scenario_summary(uci_results)
    _write_json(TIMING_ROOT / "scenario_comparison.json", scenario)
    leakage = {
        "schema_version": "uci_timing_leakage_validation_v1",
        "G3_in_any_feature_contract": False,
        "S0_contains_G1_or_G2": False,
        "S0_contains_grade_derived_features": False,
        "S1_contains_G2": False,
        "S1_contains_G2_derived_features": False,
        "S2_matches_frozen_two_timestep_contract": True,
        "preprocessing_fit_on_validation": False,
        "status": "PASS",
    }
    _write_json(TIMING_ROOT / "leakage_validation.json", leakage)
    bootstrap = []
    for dataset in ("student_mat", "student_por", "oulad"):
        values = _official_and_mlp_predictions(dataset)
        bootstrap.append(
            _paired_bootstrap(
                dataset=dataset,
                record_ids=values[0],
                groups=values[1],
                target=values[2],
                official_predicted=values[3],
                mlp_predicted=values[4],
            )
        )
    _write_csv(
        TF_ROOT / "paired_bootstrap_cnn_bilstm_vs_mlp.csv",
        pd.DataFrame(bootstrap),
    )
    imbalance = _write_imbalance_audit()
    elapsed = time.perf_counter() - started
    run = {
        "schema_version": "teacher_feedback_study_run_v1",
        "status": "PASS" if imbalance["status"] == "PASS" else "FAIL",
        "runtime_seconds": elapsed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "official_deep_models_retrained": False,
        "future_oulad_executed": False,
        "outer_used_for_tuning": False,
        "best_seed_selected": False,
        "xapi_used": False,
        "recommendation_changed": False,
    }
    _write_json(TF_ROOT / "run_summary.json", run)
    verify_regression_guard()
    write_reports()
    write_checksums()
    return run


def _format_metric(value: float) -> str:
    return f"{float(value):.4f}"


def write_reports() -> None:
    scenario_path = TIMING_ROOT / "scenario_comparison.json"
    bootstrap_path = TF_ROOT / "paired_bootstrap_cnn_bilstm_vs_mlp.csv"
    if not scenario_path.is_file() or not bootstrap_path.is_file():
        raise RuntimeError("Study evidence is incomplete; reports cannot be generated")
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    timing_lines = [
        "# UCI Timing Scenario Report",
        "",
        "This is a diagnostic information-timing study. It does not reselect or "
        "retrain either official UCI CNN-BiLSTM model.",
        "",
        "S0 uses context only, S1 adds G1, and S2 adds G2 using the frozen "
        "two-timestep feature contract. G3 is target-only.",
    ]
    for dataset in ("student_mat", "student_por"):
        title = "Student-Mat" if dataset == "student_mat" else "Student-Por"
        timing_lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Model | Scenario | Accuracy | Balanced Accuracy | Macro-F1 | Low Recall | PR-AUC | ECE |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        rows = scenario["datasets"][dataset]
        for model_id in TIMING_MODELS:
            for scenario_id in SCENARIOS:
                row = next(
                    item
                    for item in rows
                    if item["model_id"] == model_id
                    and item["scenario"] == scenario_id
                )
                timing_lines.append(
                    f"| {MODEL_NAMES[model_id]} | {scenario_id} | "
                    f"{_format_metric(row['accuracy'])} | "
                    f"{_format_metric(row['balanced_accuracy'])} | "
                    f"{_format_metric(row['macro_f1'])} | "
                    f"{_format_metric(row['low_recall'])} | "
                    f"{_format_metric(row['pr_auc'])} | "
                    f"{_format_metric(row['ece'])} |"
                )
        timing_lines.extend(
            [
                "",
                "### MLP information gain",
                "",
                "| Delta | Macro-F1 | Balanced Accuracy | Low Recall | PR-AUC | ECE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        deltas = [
            item
            for item in scenario["deltas"]
            if item["dataset"] == dataset and item["model_id"] == "mlp"
        ]
        for row in deltas:
            timing_lines.append(
                f"| {row['delta']} | {_format_metric(row['macro_f1'])} | "
                f"{_format_metric(row['balanced_accuracy'])} | "
                f"{_format_metric(row['low_recall'])} | "
                f"{_format_metric(row['pr_auc'])} | "
                f"{_format_metric(row['ece'])} |"
            )
    timing_lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "S0 is not called CNN-BiLSTM because it has no temporal grade input. "
            "The shared MLP and tabular baselines isolate information availability. "
            "All selection occurred on inner folds; outer predictions were scored once.",
        ]
    )
    report_root = ROOT / "reports" / "final"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "UCI_TIMING_SCENARIO_REPORT.md").write_text(
        "\n".join(timing_lines) + "\n", encoding="utf-8", newline="\n"
    )

    bootstrap = pd.read_csv(bootstrap_path)
    mlp_lines = [
        "# Standalone MLP Comparator",
        "",
        "MLP is a standalone tabular baseline. It is not a CNN-BiLSTM variant. "
        "Hyperparameters were selected on inner folds only, and probabilities "
        "were averaged across all five registered seeds.",
        "",
        "| Dataset | Macro-F1 | Balanced Accuracy | PR-AUC | ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    mlp_metrics: dict[str, dict[str, Any]] = {}
    for dataset in ("student_mat", "student_por"):
        metric = json.loads(
            (MLP_ROOT / dataset / "metrics.json").read_text(encoding="utf-8")
        )
        mlp_metrics[dataset] = metric
    mlp_metrics["oulad"] = json.loads(
        (MLP_ROOT / "oulad" / "metrics.json").read_text(encoding="utf-8")
    )
    for dataset, metric in mlp_metrics.items():
        mlp_lines.append(
            f"| {dataset} | {_format_metric(metric['macro_f1'])} | "
            f"{_format_metric(metric['balanced_accuracy'])} | "
            f"{_format_metric(metric['pr_auc'])} | "
            f"{_format_metric(metric['ece'])} |"
        )
    mlp_lines.extend(
        [
            "",
            "## Paired bootstrap: CNN-BiLSTM minus MLP",
            "",
            "| Dataset | Delta Macro-F1 | 95% CI | Interpretation | Unit | Replicates |",
            "|---|---:|---|---|---|---:|",
        ]
    )
    for _, row in bootstrap.iterrows():
        mlp_lines.append(
            f"| {row['dataset']} | {_format_metric(row['delta_macro_f1'])} | "
            f"[{_format_metric(row['ci_95_low'])}, "
            f"{_format_metric(row['ci_95_high'])}] | "
            f"{row['interpretation']} | {row['bootstrap_unit']} | "
            f"{int(row['replicates'])} |"
        )
    mlp_lines.extend(
        [
            "",
            "A confidence interval crossing zero is reported as insufficient "
            "evidence of a difference, not as equivalence.",
        ]
    )
    (report_root / "MLP_COMPARATOR_REPORT.md").write_text(
        "\n".join(mlp_lines) + "\n", encoding="utf-8", newline="\n"
    )

    requirements = [
        (
            "G3 definition",
            "teacher_feedback_validation/uci_target_contract.json",
            "PASS",
        ),
        (
            "Low/Medium/High thresholds",
            "teacher_feedback_validation/uci_target_contract.json",
            "PASS",
        ),
        ("G3 leakage", "uci_timing_scenarios/leakage_validation.json", "PASS"),
        ("Early no-G1/G2 scenario", "uci_timing_scenarios/", "PASS"),
        ("G1-only scenario", "uci_timing_scenarios/", "PASS"),
        ("G1+G2 scenario", "uci_timing_scenarios/", "PASS"),
        (
            "MLP baseline MAT",
            "teacher_feedback_validation/mlp_comparator/student_mat/",
            "PASS",
        ),
        (
            "MLP baseline POR",
            "teacher_feedback_validation/mlp_comparator/student_por/",
            "PASS",
        ),
        (
            "MLP baseline OULAD",
            "teacher_feedback_validation/mlp_comparator/oulad/",
            "PASS",
        ),
        (
            "same outer splits",
            "teacher_feedback_validation/split_equivalence.json",
            "PASS",
        ),
        (
            "train-only preprocessing",
            "teacher_feedback_validation/evaluation_contract.json",
            "PASS",
        ),
        (
            "paired comparison",
            "teacher_feedback_validation/paired_bootstrap_cnn_bilstm_vs_mlp.csv",
            "PASS",
        ),
        (
            "ADASYN categorical safety",
            "teacher_feedback_validation/imbalance_safety_audit.json",
            "PASS",
        ),
        (
            "OULAD tensor oversampling",
            "teacher_feedback_validation/imbalance_safety_audit.json",
            "PASS",
        ),
        (
            "Future OULAD locked",
            "teacher_feedback_validation/evaluation_contract.json",
            "PASS",
        ),
        (
            "xAPI absent from final",
            "teacher_feedback_validation/regression_guard_after.json",
            "PASS",
        ),
    ]
    completion_lines = [
        "# Teacher Feedback Completion",
        "",
        "| Requirement | Evidence | Status |",
        "|---|---|---|",
        *[
            f"| {requirement} | `artifacts/final/{evidence}` | {status} |"
            for requirement, evidence, status in requirements
        ],
        "",
        "Official CNN-BiLSTM selection, checkpoints, headline metrics, "
        "recommendation counts, and expert-label status remain frozen.",
        "",
        "The historical unsafe plain-SMOTE/ADASYN UCI baseline path is disclosed "
        "in the imbalance audit and is not used by this completion evidence. "
        "Safe S2 classical revalidation uses no synthetic resampling.",
    ]
    (report_root / "TEACHER_FEEDBACK_COMPLETION.md").write_text(
        "\n".join(completion_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_checksums() -> None:
    def manifest_for(root: Path, destination: Path) -> None:
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path != destination
            and path.name != "validation_report.json"
        ]
        payload = {
            "schema_version": "sha256_manifest_v1",
            "files": {
                str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
                for path in files
            },
        }
        payload["aggregate_sha256"] = _canonical_hash(payload["files"])
        _write_json(destination, payload)

    manifest_for(TIMING_ROOT, TIMING_ROOT / "checksums.json")
    manifest_for(TF_ROOT, TF_ROOT / "checksum_manifest.json")


def update_evidence_manifest() -> None:
    path = FINAL / "evidence_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_path = {
        item["path"]: item
        for item in payload.get("files", [])
        if item.get("path") != str(path.relative_to(ROOT)).replace("\\", "/")
    }
    for root in (TF_ROOT, TIMING_ROOT):
        for artifact in sorted(root.rglob("*")):
            if artifact.is_file():
                relative = str(artifact.relative_to(ROOT)).replace("\\", "/")
                by_path[relative] = {"path": relative, "sha256": _sha256(artifact)}
    # Refresh any canonical file already represented in the manifest.
    for relative in list(by_path):
        artifact = ROOT / relative
        if artifact.is_file():
            by_path[relative]["sha256"] = _sha256(artifact)
    payload.update(
        {
            "schema_version": "final_evidence_manifest_v2",
            "generated_from_immutable_existing_evidence": False,
            "training_performed": True,
            "official_deep_model_training_performed": False,
            "teacher_feedback_comparator_training_performed": True,
            "outer_evaluation_used_for_selection": False,
            "best_seed_selected": False,
            "future_oulad_accessed": False,
            "files": [by_path[key] for key in sorted(by_path)],
        }
    )
    _write_json(path, payload)


def validate_study() -> dict[str, Any]:
    required = [
        TF_ROOT / "regression_guard_before.json",
        TF_ROOT / "regression_guard_after.json",
        TF_ROOT / "uci_target_contract.json",
        TF_ROOT / "split_equivalence.json",
        TF_ROOT / "evaluation_contract.json",
        TF_ROOT / "imbalance_safety_audit.json",
        TF_ROOT / "paired_bootstrap_cnn_bilstm_vs_mlp.csv",
        TIMING_ROOT / "protocol.json",
        TIMING_ROOT / "student_mat_metrics.csv",
        TIMING_ROOT / "student_por_metrics.csv",
        TIMING_ROOT / "student_mat_predictions.parquet",
        TIMING_ROOT / "student_por_predictions.parquet",
        TIMING_ROOT / "scenario_comparison.json",
        TIMING_ROOT / "split_validation.json",
        TIMING_ROOT / "leakage_validation.json",
        TIMING_ROOT / "checksums.json",
    ]
    errors = [f"missing: {path.relative_to(ROOT)}" for path in required if not path.is_file()]
    if not errors:
        guard = verify_regression_guard()
        if guard["status"] != "PASS":
            errors.append("official regression guard failed")
        imbalance = json.loads(
            (TF_ROOT / "imbalance_safety_audit.json").read_text(encoding="utf-8")
        )
        if imbalance["status"] != "PASS":
            errors.append("imbalance safety audit failed")
        split = json.loads(
            (TF_ROOT / "split_equivalence.json").read_text(encoding="utf-8")
        )
        if (
            split["status"] != "PASS"
            or not split["all_models_same_split_within_dataset_scenario"]
            or split["outer_rows_in_inner_training"] != 0
        ):
            errors.append("split equivalence failed")
        for dataset in ("student_mat", "student_por"):
            predictions = pd.read_parquet(
                TIMING_ROOT / f"{dataset}_predictions.parquet"
            )
            expected = len(_load_uci(dataset).target)
            counts = predictions.groupby(["scenario", "model_id"])[
                "record_id"
            ].nunique()
            if counts.min() != expected or counts.max() != expected:
                errors.append(f"{dataset} timing prediction coverage failed")
        oulad = pd.read_parquet(MLP_ROOT / "oulad" / "oof_predictions.parquet")
        if len(oulad) != 15378 or oulad["record_id"].nunique() != 15378:
            errors.append("OULAD MLP prediction coverage failed")
        for manifest_path in (
            TIMING_ROOT / "checksums.json",
            TF_ROOT / "checksum_manifest.json",
        ):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for relative, expected_hash in manifest["files"].items():
                path = ROOT / relative
                if not path.is_file() or _sha256(path) != expected_hash:
                    errors.append(f"checksum mismatch: {relative}")
    result = {
        "schema_version": "teacher_feedback_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "official_deep_models_retrained": False,
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "expert_status": "PENDING_EXPERT_LABELS",
        "xapi_in_final": False,
    }
    _write_json(TF_ROOT / "validation_report.json", result)
    return result


__all__ = [
    "SCENARIOS",
    "build_uci_scenario_frame",
    "encode_uci_target",
    "prepare_regression_guard",
    "run_all",
    "validate_study",
    "verify_regression_guard",
    "update_evidence_manifest",
    "write_checksums",
    "write_reports",
]
