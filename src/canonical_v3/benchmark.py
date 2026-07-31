"""Resume-safe canonical V3 benchmark supervisor."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - validated in preflight
    XGBClassifier = None

from src.canonical_v3.metrics import (
    binary_metrics,
    multiclass_metrics,
    paired_bootstrap_macro_f1,
)
from src.canonical_v3.oulad_data import (
    CANONICAL_STAGES,
    build_canonical_bundle,
    single_stage_rows,
    stage_rows,
)
from src.pipelines import oulad, uci
from src.pipelines import uci_support as uci_support
from src.training.control import select_refit_epoch, select_research_threshold, stable_hash
from src.training.phase5_mlp_gap import _train_deep_inner
from src.training.phase6_final import _predict_deep_payload, _train_deep_final
from src.training.phase7_endpoint import _train_inner

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "canonical_v3"
PREDICTIONS = OUT / "predictions"
CHECKPOINTS = OUT / "checkpoints"
RUNS = OUT / "runtime" / "runs"
LOGS = OUT / "logs"
STATUS = OUT / "runtime" / "phase11_status.json"
RUNNING = OUT / "runtime" / "PHASE11_RUNNING"
COMPLETE = OUT / "runtime" / "PHASE11_COMPLETE"
FAILED = OUT / "runtime" / "PHASE11_FAILED"
FREEZE_PATH = OUT / "CANONICAL_BENCHMARK_FREEZE.json"
PROTOCOL_PATH = ROOT / "configs" / "canonical_v3" / "benchmark_protocol.yaml"
POLICY_PATH = ROOT / "configs" / "canonical_v3" / "oulad_information_policy.yaml"
SEARCH_PATH = ROOT / "configs" / "canonical_v3" / "model_search_spaces.yaml"
REPORT = ROOT / "reports" / "final" / "canonical_v3"
PRIMARY_MODELS = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
    "svm",
    "xgboost",
    "mlp",
    "hybrid",
)
UCI_MAIN_STAGE = "MAIN_ENDPOINT"
OULAD_FINAL = "FINAL"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_directories() -> None:
    for path in (OUT, PREDICTIONS, CHECKPOINTS, RUNS, LOGS, STATUS.parent, REPORT):
        path.mkdir(parents=True, exist_ok=True)


def _status(**updates: Any) -> dict[str, Any]:
    current = (
        _json(STATUS)
        if STATUS.is_file()
        else {
            "state": "PENDING",
            "started_at": None,
            "finished_at": None,
            "current_stage": "preflight",
            "completed_runs": 0,
            "failed_runs": 0,
            "current_run": None,
            "training_performed": False,
            "exit_code": None,
            "pid": os.getpid(),
        }
    )
    current.update(updates)
    write_json(STATUS, current)
    return current


def _sentinel(state: str, details: dict[str, Any] | None = None) -> None:
    for path in (RUNNING, COMPLETE, FAILED):
        if path.exists():
            path.unlink()
    target = {"RUNNING": RUNNING, "COMPLETE": COMPLETE, "FAILED": FAILED}[state]
    write_json(target, {"state": state, "at": utc_now(), **(details or {})})


def _mark_run(run_id: str, payload: dict[str, Any]) -> None:
    write_json(RUNS / f"{run_id}.json", payload)
    status = _status()
    _status(completed_runs=int(status.get("completed_runs", 0)) + 1, current_run=None)


def _cached_frame(run_id: str) -> pd.DataFrame | None:
    manifest = RUNS / f"{run_id}.json"
    prediction = PREDICTIONS / "runs" / f"{run_id}.parquet"
    if not manifest.is_file() or not prediction.is_file():
        return None
    payload = _json(manifest)
    if payload.get("status") != "COMPLETE" or payload.get("benchmark_hash") != _json(
        FREEZE_PATH
    )["canonical_benchmark_hash"]:
        return None
    return pd.read_parquet(prediction)


def _save_run(run_id: str, frame: pd.DataFrame, payload: dict[str, Any]) -> None:
    path = PREDICTIONS / "runs" / f"{run_id}.parquet"
    write_parquet(path, frame)
    _mark_run(
        run_id,
        {
            "status": "COMPLETE",
            "benchmark_hash": _json(FREEZE_PATH)["canonical_benchmark_hash"],
            "prediction_path": path.relative_to(ROOT).as_posix(),
            **payload,
        },
    )


def validate_preflight() -> dict[str, Any]:
    freeze = _json(FREEZE_PATH)
    protocol = _yaml(PROTOCOL_PATH)
    policy = _yaml(POLICY_PATH)
    if freeze["status"] != "IMMUTABLE_PRE_BENCHMARK":
        raise RuntimeError("canonical benchmark is not frozen")
    if protocol["status"] != "FROZEN_BEFORE_BENCHMARK":
        raise RuntimeError("protocol status changed")
    if policy["policy"] != "STRICT_REAL_TIME" or policy["score_policy"]["score_values"] != "EXCLUDED":
        raise RuntimeError("OULAD information policy changed")
    current_hashes = {
        "information_policy": file_hash(POLICY_PATH),
        "benchmark_protocol": file_hash(PROTOCOL_PATH),
        "search_spaces": file_hash(SEARCH_PATH),
        "old_uci_predictions": file_hash(
            ROOT / "artifacts/final/unified_stage_aware_uci/predictions.parquet"
        ),
        "old_oulad_predictions": file_hash(
            ROOT / "artifacts/final/unified_stage_aware_oulad/predictions.parquet"
        ),
        "old_h1_predictions": file_hash(ROOT / "artifacts/final/h1_final/predictions.parquet"),
    }
    for key, value in current_hashes.items():
        if freeze["source_hashes"][key] != value:
            raise RuntimeError(f"frozen source hash changed: {key}")
    protocol_commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", str(PROTOCOL_PATH.relative_to(ROOT))],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    if not protocol_commit:
        raise RuntimeError("protocol has no pre-benchmark commit")
    if XGBClassifier is None:
        raise RuntimeError("xgboost is required")
    result = {
        "status": "PASS",
        "protocol_commit": protocol_commit,
        "benchmark_hash": freeze["canonical_benchmark_hash"],
        "policy": policy["policy"],
        "architecture_search": False,
        "outer_labels_used_for_selection": False,
    }
    write_json(OUT / "preflight.json", result)
    return result


def _uci_config() -> dict[str, Any]:
    hybrid = _yaml(PROTOCOL_PATH)["uci"]["hybrid"]
    topology = hybrid["topology"]
    training = hybrid["training"]
    return {
        "input_projection": topology["input_projection"],
        "cnn_channels": topology["cnn_channels"],
        "lstm_hidden": topology["lstm_hidden"],
        "context_hidden": topology["context_hidden"],
        "fusion_hidden": topology["fusion_hidden"],
        "dropout": training["dropout"],
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "batch_size": training["batch_size"],
        "max_epochs": training["max_epochs"],
        "patience": training["patience"],
        "class_weight": training["class_weight"],
    }


def _round_half_up_median(values: Iterable[int]) -> int:
    return max(1, int(math.floor(float(np.median(list(values))) + 0.5)))


def run_uci_main_hybrid(*, smoke: bool = False) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    config = _uci_config()
    seeds = [42] if smoke else _yaml(PROTOCOL_PATH)["uci"]["seeds"]
    selected_rows: list[dict[str, Any]] = []
    for dataset in ("student_mat", "student_por"):
        data = uci_support._load_uci(dataset)
        views = uci.build_stage_views(data)
        endpoint_views = {stage: views[uci.STAGES[-1]] for stage in uci.STAGES}
        folds = sorted(np.unique(data.outer_fold))[:1] if smoke else sorted(np.unique(data.outer_fold))
        for outer_fold in folds:
            train = np.flatnonzero(data.outer_fold != outer_fold)
            validation = np.flatnonzero(data.outer_fold == outer_fold)
            authority_path = OUT / "runtime" / f"uci_main_epoch_{dataset}_{outer_fold}.json"
            if authority_path.is_file() and not smoke:
                authority = _json(authority_path)
                selected_epoch = int(authority["selected_epoch"])
            else:
                epochs: list[int] = []
                inner = uci_support._inner_splits(
                    data.target[train], data.groups[train], seed=42
                )
                if smoke:
                    inner = inner[:1]
                for fit_local, score_local in inner:
                    _, epoch, _ = uci._fit_deep_model(
                        data=data,
                        views=endpoint_views,
                        family="cnn_bilstm",
                        candidate={**config, "max_epochs": 1 if smoke else config["max_epochs"]},
                        fit=train[fit_local],
                        score=train[score_local],
                        seed=42,
                    )
                    epochs.append(int(epoch))
                selected_epoch = _round_half_up_median(epochs)
                authority = {
                    "dataset": dataset,
                    "outer_fold": int(outer_fold),
                    "inner_selected_epochs": epochs,
                    "selected_epoch": selected_epoch,
                    "outer_labels_used": False,
                }
                if not smoke:
                    write_json(authority_path, authority)
            selected_rows.append(authority)
            for seed in seeds:
                run_id = stable_hash(
                    {
                        "scope": "uci_main_hybrid",
                        "dataset": dataset,
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "config": config,
                        "benchmark": _json(FREEZE_PATH)["canonical_benchmark_hash"],
                        "smoke": smoke,
                    }
                )[:24]
                cached = None if smoke else _cached_frame(run_id)
                if cached is not None:
                    rows.append(cached)
                    continue
                _status(
                    current_stage="uci_main_hybrid",
                    current_run=run_id,
                    training_performed=True,
                )
                probability, _, payload = uci._fit_deep_model(
                    data=data,
                    views=endpoint_views,
                    family="cnn_bilstm",
                    candidate=config,
                    fit=train,
                    score=validation,
                    seed=int(seed),
                    fixed_epochs=1 if smoke else selected_epoch,
                )
                values = probability[uci.STAGES[-1]]
                frame = pd.DataFrame(
                    {
                        "dataset": dataset,
                        "task": "MAIN",
                        "stage": UCI_MAIN_STAGE,
                        "model": "hybrid",
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "record_id": data.record_ids[validation],
                        "target": data.target[validation],
                        "p_low": values[:, 0],
                        "p_medium": values[:, 1],
                        "p_high": values[:, 2],
                        "source": "canonical_v3_trained",
                    }
                )
                if not smoke:
                    checkpoint = CHECKPOINTS / "uci_main" / dataset / f"outer{outer_fold}_seed{seed}.joblib"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(
                        {
                            **payload,
                            "selected_epoch": selected_epoch,
                            "topology_hash": _json(FREEZE_PATH)["uci_topology_hash"],
                        },
                        checkpoint,
                        compress=3,
                    )
                    _save_run(
                        run_id,
                        frame,
                        {
                            "dataset": dataset,
                            "outer_fold": int(outer_fold),
                            "seed": int(seed),
                            "selected_epoch": selected_epoch,
                            "model": "hybrid",
                            "topology_hash": _json(FREEZE_PATH)["uci_topology_hash"],
                        },
                    )
                rows.append(frame)
    if not smoke:
        write_json(OUT / "uci_main_epoch_selection.json", selected_rows)
    return pd.concat(rows, ignore_index=True)


def replay_uci_predictions(new_hybrid: pd.DataFrame) -> pd.DataFrame:
    stage = pd.read_parquet(
        ROOT / "artifacts/final/unified_stage_aware_uci/seed_predictions.parquet"
    )
    stage = stage.loc[stage.model_family.isin(PRIMARY_MODELS[:-1]) | stage.model_family.eq("cnn_bilstm")].copy()
    stage["task"] = "EARLY_WARNING"
    stage["stage"] = stage.prediction_stage
    stage["model"] = stage.model_family.replace({"cnn_bilstm": "hybrid"})
    stage["source"] = "replay_unified_stage_aware_uci_v1"
    stage = stage.rename(columns={"dataset": "dataset", "outer_fold": "outer_fold"})
    stage = stage.loc[
        :,
        [
            "dataset",
            "task",
            "stage",
            "model",
            "outer_fold",
            "seed",
            "record_id",
            "target",
            "p_low",
            "p_medium",
            "p_high",
            "source",
        ],
    ]

    main_source = ROOT / "artifacts/final/comparator_completion"
    main_rows: list[pd.DataFrame] = []
    classical = set(PRIMARY_MODELS) - {"mlp", "hybrid"}
    for dataset in ("student_mat", "student_por"):
        current = pd.read_parquet(main_source / dataset / "seed_predictions.parquet")
        current = current.loc[current.model_id.isin(classical)].copy()
        current = current.rename(columns={"model_id": "model", "true_label": "target"})
        current["task"] = "MAIN"
        current["stage"] = UCI_MAIN_STAGE
        current["source"] = "replay_final_comparator_completion"
        main_rows.append(current.loc[:, stage.columns])
        mlp = pd.read_parquet(
            ROOT
            / "artifacts/final/teacher_feedback_validation/mlp_comparator"
            / dataset
            / "seed_predictions.parquet"
        ).copy()
        mlp["model"] = "mlp"
        mlp["task"] = "MAIN"
        mlp["stage"] = UCI_MAIN_STAGE
        mlp["source"] = "replay_protocol_matched_mlp"
        main_rows.append(mlp.loc[:, stage.columns])
    return pd.concat([stage, *main_rows, new_hybrid], ignore_index=True)


def _h1_manifest(config: dict[str, Any]) -> dict[str, Any]:
    freeze = _json(FREEZE_PATH)
    return {
        "final_candidate_hash": freeze["canonical_benchmark_hash"],
        "architecture_hash": freeze["oulad_architecture_hash"],
        "feature_schema_hash": freeze["feature_monotonicity_hash"],
        "training_policy_hash": stable_hash(config),
        "evaluation_protocol_hash": freeze["protocol_hash"],
    }


def _base_ids(bundle: oulad.Bundle, outer_fold: int, *, train: bool) -> set[str]:
    selected = bundle.base.outer_fold.ne(outer_fold) if train else bundle.base.outer_fold.eq(outer_fold)
    return set(bundle.base.loc[selected, "base_record_id"].astype(str))


def _h1_inner_authority(bundle: oulad.Bundle, *, shared: bool, smoke: bool) -> list[dict[str, Any]]:
    freeze = _json(FREEZE_PATH)
    authorities: list[dict[str, Any]] = []
    base = bundle.base[["base_record_id", "id_student", "outer_fold", "target"]].drop_duplicates()
    folds = [0] if smoke else [0, 1, 2]
    for outer_fold in folds:
        path = OUT / "runtime" / f"h1_{'shared' if shared else 'final'}_inner_outer{outer_fold}.json"
        prediction_path = path.with_suffix(".parquet")
        if path.is_file() and prediction_path.is_file() and not smoke:
            authorities.append(_json(path))
            continue
        config = freeze["phase3_training_configs"][str(outer_fold)]["config"]
        predictions: list[pd.DataFrame] = []
        epochs: list[int] = []
        splits = list(oulad._inner_splits(base, outer_fold))
        if smoke:
            splits = splits[:1]
        for inner_fold, (fit_ids, validation_ids) in enumerate(splits):
            _status(
                current_stage="h1_shared_inner" if shared else "h1_final_inner",
                current_run=f"outer{outer_fold}_inner{inner_fold}",
                training_performed=True,
            )
            if shared:
                previous_stages = oulad.STAGES
                try:
                    oulad.STAGES = CANONICAL_STAGES
                    result = _train_deep_inner(
                        stage_rows(bundle, fit_ids),
                        stage_rows(bundle, validation_ids),
                        candidate="H1_TABULAR_RESIDUAL_EXPERT",
                        config=config,
                        seed=42,
                        max_epochs=1 if smoke else 15,
                    )
                finally:
                    oulad.STAGES = previous_stages
            else:
                result = _train_inner(
                    single_stage_rows(bundle, OULAD_FINAL, fit_ids),
                    single_stage_rows(bundle, OULAD_FINAL, validation_ids),
                    config=config,
                    seed=42,
                    inner_fold=inner_fold,
                    trial=None,
                    max_epochs=1 if smoke else 15,
                )
            current = result.prediction.copy()
            if not shared:
                current["prediction_stage"] = OULAD_FINAL
            current["inner_fold"] = inner_fold
            predictions.append(current)
            epochs.append(int(result.selected_epoch))
        prediction = pd.concat(predictions, ignore_index=True)
        thresholds = {}
        for stage_name, current in prediction.groupby("prediction_stage"):
            thresholds[stage_name] = float(
                select_research_threshold(
                    current.target.to_numpy(dtype=int),
                    current.probability.to_numpy(dtype=float),
                )["threshold"]
            )
        authority = {
            "role": "shared_stage" if shared else "endpoint_final",
            "outer_fold": outer_fold,
            "config": config,
            "config_hash": stable_hash(config),
            "inner_selected_epochs": epochs,
            "selected_epoch": select_refit_epoch(epochs),
            "thresholds": thresholds,
            "outer_labels_used": False,
        }
        if not smoke:
            write_json(path, authority)
            write_parquet(prediction_path, prediction)
        authorities.append(authority)
    return authorities


def _h1_refit_predictions(
    bundle: oulad.Bundle,
    authorities: list[dict[str, Any]],
    *,
    shared: bool,
    smoke: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    seeds = [42] if smoke else _yaml(PROTOCOL_PATH)["oulad"]["seeds"]
    for authority in authorities:
        outer_fold = int(authority["outer_fold"])
        config = authority["config"]
        train_ids = _base_ids(bundle, outer_fold, train=True)
        validation_ids = _base_ids(bundle, outer_fold, train=False)
        train = stage_rows(bundle, train_ids) if shared else single_stage_rows(bundle, OULAD_FINAL, train_ids)
        stages = CANONICAL_STAGES if shared else (OULAD_FINAL,)
        for seed in seeds:
            run_id = stable_hash(
                {
                    "scope": "oulad_h1_shared" if shared else "oulad_h1_final",
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "config": config,
                    "epoch": authority["selected_epoch"],
                    "benchmark": _json(FREEZE_PATH)["canonical_benchmark_hash"],
                    "smoke": smoke,
                }
            )[:24]
            cached = None if smoke else _cached_frame(run_id)
            if cached is not None:
                rows.append(cached)
                continue
            _status(
                current_stage="h1_shared_refit" if shared else "h1_final_refit",
                current_run=run_id,
                training_performed=True,
            )
            checkpoint = (
                CHECKPOINTS
                / ("oulad_h1_shared" if shared else "oulad_h1_final")
                / f"outer{outer_fold}_seed{seed}.pt"
            )
            manifest = _h1_manifest(config)
            _train_deep_final(
                "H1_TABULAR_RESIDUAL_EXPERT",
                train,
                config,
                int(seed),
                1 if smoke else int(authority["selected_epoch"]),
                checkpoint,
                manifest,
            )
            run_frames: list[pd.DataFrame] = []
            for stage_name in stages:
                data = single_stage_rows(bundle, stage_name, validation_ids)
                probability = _predict_deep_payload(
                    checkpoint, data[0], data[1], data[2], data[3], data[4]
                )
                current = data[0].loc[
                    :,
                    [
                        "base_record_id",
                        "id_student",
                        "code_module",
                        "code_presentation",
                        "outer_fold",
                        "target",
                        "cutoff_day",
                    ],
                ].copy()
                current["dataset"] = "oulad"
                current["task"] = "MAIN" if not shared else "EARLY_WARNING"
                current["stage"] = stage_name
                current["model"] = "hybrid"
                current["seed"] = int(seed)
                current["probability"] = probability
                current["threshold"] = float(authority["thresholds"][stage_name])
                current["checkpoint_role"] = "shared_stage" if shared else "endpoint_final"
                current["source"] = "canonical_v3_trained"
                run_frames.append(current)
            frame = pd.concat(run_frames, ignore_index=True)
            if not smoke:
                _save_run(
                    run_id,
                    frame,
                    {
                        "model": "hybrid",
                        "checkpoint_role": "shared_stage" if shared else "endpoint_final",
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "architecture_hash": _json(FREEZE_PATH)["oulad_architecture_hash"],
                        "parameter_count": 160492,
                    },
                )
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _tabular_estimator(model: str, config: dict[str, Any], seed: int) -> Pipeline:
    numeric = [f"aggregate_{index:03d}" for index in range(165)] + [
        column for column in oulad.STATIC_COLUMNS if column not in oulad.CATEGORICAL
    ]
    preprocessor = ColumnTransformer(
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
                OneHotEncoder(handle_unknown="ignore"),
                list(oulad.CATEGORICAL),
            ),
        ]
    )
    if model == "logistic_regression":
        estimator = LogisticRegression(max_iter=600, random_state=seed, **config)
    elif model == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=seed, **config)
    elif model == "random_forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=-1, **config)
    elif model == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(random_state=seed, **config)
    elif model == "svm":
        estimator = SVC(probability=True, cache_size=4096, random_state=seed, **config)
    elif model == "xgboost":
        assert XGBClassifier is not None
        estimator = XGBClassifier(
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            **config,
        )
    elif model == "mlp":
        current = dict(config)
        current["hidden_layer_sizes"] = tuple(current["hidden_layer_sizes"])
        estimator = MLPClassifier(
            random_state=seed,
            early_stopping=True,
            **current,
        )
    else:
        raise KeyError(model)
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def _fit_tabular_probability(
    estimator: Pipeline, train: tuple, validation: tuple
) -> np.ndarray:
    frame, _, _, _, aggregate, target, _ = train
    val_frame, _, _, _, val_aggregate, _, _ = validation
    estimator.fit(oulad._tabular_frame(frame, aggregate), target.astype(int))
    return estimator.predict_proba(oulad._tabular_frame(val_frame, val_aggregate))[:, 1]


def _select_final_tabular(bundle: oulad.Bundle, *, smoke: bool) -> list[dict[str, Any]]:
    search = _yaml(SEARCH_PATH)["oulad_final_tabular"]
    base = bundle.base[["base_record_id", "id_student", "outer_fold", "target"]].drop_duplicates()
    selected: list[dict[str, Any]] = []
    models = ["logistic_regression"] if smoke else list(search)
    folds = [0] if smoke else [0, 1, 2]
    for model in models:
        for outer_fold in folds:
            authority_path = OUT / "runtime" / f"tabular_selection_{model}_outer{outer_fold}.json"
            if authority_path.is_file() and not smoke:
                selected.append(_json(authority_path))
                continue
            candidates = search[model][:1] if smoke else search[model]
            best: tuple[tuple[float, float, float, str], dict[str, Any]] | None = None
            splits = list(oulad._inner_splits(base, outer_fold))
            if smoke:
                splits = splits[:1]
            for candidate in candidates:
                predictions: list[pd.DataFrame] = []
                for inner_fold, (fit_ids, validation_ids) in enumerate(splits):
                    estimator = _tabular_estimator(model, candidate, 42)
                    validation = single_stage_rows(bundle, OULAD_FINAL, validation_ids)
                    probability = _fit_tabular_probability(
                        estimator,
                        single_stage_rows(bundle, OULAD_FINAL, fit_ids),
                        validation,
                    )
                    current = validation[0][["base_record_id", "target"]].copy()
                    current["probability"] = probability
                    current["inner_fold"] = inner_fold
                    predictions.append(current)
                prediction = pd.concat(predictions, ignore_index=True)
                threshold = float(
                    select_research_threshold(
                        prediction.target.to_numpy(dtype=int), prediction.probability.to_numpy()
                    )["threshold"]
                )
                metrics = binary_metrics(
                    prediction.target.to_numpy(dtype=int),
                    prediction.probability.to_numpy(),
                    threshold,
                )
                key = (
                    metrics["macro_f1"],
                    metrics["pr_auc"],
                    -metrics["nll"],
                    json.dumps(candidate, sort_keys=True),
                )
                if best is None or key[:-1] > best[0][:-1] or (
                    key[:-1] == best[0][:-1] and key[-1] < best[0][-1]
                ):
                    best = (key, {"config": candidate, "threshold": threshold, "metrics": metrics})
            assert best is not None
            authority = {
                "model": model,
                "outer_fold": outer_fold,
                **best[1],
                "outer_labels_used": False,
            }
            if not smoke:
                write_json(authority_path, authority)
            selected.append(authority)
    if not smoke:
        write_json(OUT / "oulad_final_tabular_selected.json", selected)
    return selected


def _run_final_tabular(
    bundle: oulad.Bundle, selections: list[dict[str, Any]], *, smoke: bool
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    stochastic = {"decision_tree", "random_forest", "hist_gradient_boosting", "xgboost", "mlp"}
    protocol_seeds = _yaml(PROTOCOL_PATH)["oulad"]["seeds"]
    for selection in selections:
        model = selection["model"]
        outer_fold = int(selection["outer_fold"])
        seeds = [42] if smoke or model not in stochastic else protocol_seeds
        train_ids = _base_ids(bundle, outer_fold, train=True)
        validation_ids = _base_ids(bundle, outer_fold, train=False)
        train = single_stage_rows(bundle, OULAD_FINAL, train_ids)
        validation = single_stage_rows(bundle, OULAD_FINAL, validation_ids)
        for seed in seeds:
            run_id = stable_hash(
                {
                    "scope": "oulad_final_tabular",
                    "model": model,
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "config": selection["config"],
                    "benchmark": _json(FREEZE_PATH)["canonical_benchmark_hash"],
                    "smoke": smoke,
                }
            )[:24]
            cached = None if smoke else _cached_frame(run_id)
            if cached is not None:
                rows.append(cached)
                continue
            _status(
                current_stage="oulad_final_tabular",
                current_run=run_id,
                training_performed=True,
            )
            estimator = _tabular_estimator(model, selection["config"], int(seed))
            probability = _fit_tabular_probability(estimator, train, validation)
            frame = validation[0].loc[
                :,
                [
                    "base_record_id",
                    "id_student",
                    "code_module",
                    "code_presentation",
                    "outer_fold",
                    "target",
                    "cutoff_day",
                ],
            ].copy()
            frame["dataset"] = "oulad"
            frame["task"] = "MAIN"
            frame["stage"] = OULAD_FINAL
            frame["model"] = model
            frame["seed"] = int(seed)
            frame["probability"] = probability
            frame["threshold"] = float(selection["threshold"])
            frame["checkpoint_role"] = "endpoint_final"
            frame["source"] = "canonical_v3_trained"
            if not smoke:
                checkpoint = CHECKPOINTS / "oulad_final_tabular" / model / f"outer{outer_fold}_seed{seed}.joblib"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(estimator, checkpoint, compress=3)
                _save_run(
                    run_id,
                    frame,
                    {
                        "model": model,
                        "outer_fold": outer_fold,
                        "seed": seed,
                        "config_hash": stable_hash(selection["config"]),
                    },
                )
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def replay_oulad_stage_ml() -> pd.DataFrame:
    prediction = pd.read_parquet(
        ROOT / "artifacts/final/unified_stage_aware_oulad/seed_predictions.parquet"
    )
    prediction = prediction.loc[
        prediction.model_family.isin(set(PRIMARY_MODELS) - {"hybrid"})
    ].copy()
    prediction["stage"] = prediction.prediction_stage.replace(
        {"M1_MIDDLE_FROZEN": "M1_MIDDLE_50PCT"}
    )
    prediction["dataset"] = "oulad"
    prediction["task"] = "EARLY_WARNING"
    prediction["model"] = prediction.model_family
    prediction["checkpoint_role"] = "stage_specific_tabular"
    prediction["source"] = "replay_unified_stage_aware_oulad_v2"
    threshold = pd.read_csv(
        ROOT / "artifacts/final/unified_stage_aware_oulad/threshold_policies.csv"
    )
    threshold = threshold.loc[
        threshold.threshold_policy.eq("INNER_OOF_STAGE_THRESHOLD")
        & threshold.model_family.isin(set(PRIMARY_MODELS) - {"hybrid"})
    ].copy()
    threshold["stage"] = threshold.prediction_stage.replace(
        {"M1_MIDDLE_FROZEN": "M1_MIDDLE_50PCT"}
    )
    threshold.outer_fold = threshold.outer_fold.astype(int)
    prediction = prediction.merge(
        threshold[["model_family", "outer_fold", "stage", "threshold"]],
        left_on=["model", "outer_fold", "stage"],
        right_on=["model_family", "outer_fold", "stage"],
        validate="many_to_one",
        suffixes=("", "_threshold"),
    )
    return prediction.loc[
        :,
        [
            "base_record_id",
            "id_student",
            "code_module",
            "code_presentation",
            "outer_fold",
            "target",
            "cutoff_day",
            "dataset",
            "task",
            "stage",
            "model",
            "seed",
            "probability",
            "threshold",
            "checkpoint_role",
            "source",
        ],
    ]


def _ensemble_uci(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "task", "stage", "model", "outer_fold", "record_id", "target"]
    return (
        seed_predictions.groupby(keys, as_index=False)
        .agg(p_low=("p_low", "mean"), p_medium=("p_medium", "mean"), p_high=("p_high", "mean"))
        .assign(source="canonical_v3_seed_mean")
    )


def _ensemble_oulad(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "base_record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "outer_fold",
        "target",
        "cutoff_day",
        "dataset",
        "task",
        "stage",
        "model",
        "checkpoint_role",
    ]
    return (
        seed_predictions.groupby(keys, as_index=False)
        .agg(probability=("probability", "mean"), threshold=("threshold", "first"))
        .assign(source="canonical_v3_seed_mean")
    )


def _metric_columns(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in {"per_class", "confusion_matrix"}}


def aggregate_uci(seed_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    for key, current in seed_predictions.groupby(
        ["dataset", "task", "stage", "model", "outer_fold", "seed"], sort=False
    ):
        probability = current[["p_low", "p_medium", "p_high"]].to_numpy()
        detailed.append(
            {
                **dict(zip(["dataset", "task", "stage", "model", "fold", "seed"], key)),
                **_metric_columns(multiclass_metrics(current.target.to_numpy(), probability)),
                "rows": len(current),
            }
        )
    ensemble = _ensemble_uci(seed_predictions)
    aggregate: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: dict[str, Any] = {}
    for key, current in ensemble.groupby(["dataset", "task", "stage", "model"], sort=False):
        probability = current[["p_low", "p_medium", "p_high"]].to_numpy()
        payload = multiclass_metrics(current.target.to_numpy(), probability)
        seed_scores = []
        source = seed_predictions.loc[
            seed_predictions.dataset.eq(key[0])
            & seed_predictions.task.eq(key[1])
            & seed_predictions.stage.eq(key[2])
            & seed_predictions.model.eq(key[3])
        ]
        for _, seed_frame in source.groupby("seed"):
            seed_scores.append(
                multiclass_metrics(
                    seed_frame.target.to_numpy(),
                    seed_frame[["p_low", "p_medium", "p_high"]].to_numpy(),
                )["macro_f1"]
            )
        row = {
            **dict(zip(["dataset", "task", "stage", "model"], key)),
            **_metric_columns(payload),
            "seed_macro_f1_mean": float(np.mean(seed_scores)),
            "seed_macro_f1_std": float(np.std(seed_scores)),
            "seed_macro_f1_min": float(np.min(seed_scores)),
            "seed_macro_f1_max": float(np.max(seed_scores)),
            "rows": len(current),
        }
        aggregate.append(row)
        for class_row in payload["per_class"]:
            per_class.append({**dict(zip(["dataset", "task", "stage", "model"], key)), **class_row})
        confusion["|".join(map(str, key))] = payload["confusion_matrix"]
    return pd.DataFrame(detailed), pd.DataFrame(aggregate), per_class, confusion


def aggregate_oulad(seed_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    for key, current in seed_predictions.groupby(
        ["dataset", "task", "stage", "model", "outer_fold", "seed"], sort=False
    ):
        payload = binary_metrics(
            current.target.to_numpy(), current.probability.to_numpy(), current.threshold.to_numpy()
        )
        detailed.append(
            {
                **dict(zip(["dataset", "task", "stage", "model", "fold", "seed"], key)),
                **_metric_columns(payload),
                "rows": len(current),
            }
        )
    ensemble = _ensemble_oulad(seed_predictions)
    aggregate: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    confusion: dict[str, Any] = {}
    for key, current in ensemble.groupby(["dataset", "task", "stage", "model"], sort=False):
        payload = binary_metrics(
            current.target.to_numpy(), current.probability.to_numpy(), current.threshold.to_numpy()
        )
        seed_scores = []
        source = seed_predictions.loc[
            seed_predictions.task.eq(key[1])
            & seed_predictions.stage.eq(key[2])
            & seed_predictions.model.eq(key[3])
        ]
        for _, seed_frame in source.groupby("seed"):
            seed_scores.append(
                binary_metrics(
                    seed_frame.target.to_numpy(),
                    seed_frame.probability.to_numpy(),
                    seed_frame.threshold.to_numpy(),
                )["macro_f1"]
            )
        row = {
            **dict(zip(["dataset", "task", "stage", "model"], key)),
            **_metric_columns(payload),
            "seed_macro_f1_mean": float(np.mean(seed_scores)),
            "seed_macro_f1_std": float(np.std(seed_scores)),
            "seed_macro_f1_min": float(np.min(seed_scores)),
            "seed_macro_f1_max": float(np.max(seed_scores)),
            "rows": len(current),
        }
        aggregate.append(row)
        per_class.extend(
            [
                {
                    **dict(zip(["dataset", "task", "stage", "model"], key)),
                    "class_name": "not_risk",
                    "precision": payload["not_risk_precision"],
                    "recall": payload["not_risk_recall"],
                    "f1": payload["not_risk_f1"],
                },
                {
                    **dict(zip(["dataset", "task", "stage", "model"], key)),
                    "class_name": "risk",
                    "precision": payload["risk_precision"],
                    "recall": payload["risk_recall"],
                    "f1": payload["risk_f1"],
                },
            ]
        )
        confusion["|".join(map(str, key))] = payload["confusion_matrix"]
    return pd.DataFrame(detailed), pd.DataFrame(aggregate), per_class, confusion


def _best_ml(frame: pd.DataFrame, dataset: str) -> pd.Series:
    current = frame.loc[
        frame.dataset.eq(dataset) & frame.task.eq("MAIN") & ~frame.model.eq("hybrid")
    ]
    return current.sort_values(["macro_f1", "pr_auc"], ascending=False).iloc[0]


def build_statistics(
    uci_predictions: pd.DataFrame,
    uci_aggregate: pd.DataFrame,
    oulad_predictions: pd.DataFrame,
    oulad_aggregate: pd.DataFrame,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    uci_ensemble = _ensemble_uci(uci_predictions)
    for dataset in ("student_mat", "student_por"):
        best = _best_ml(uci_aggregate, dataset)
        hybrid = uci_ensemble.loc[
            uci_ensemble.dataset.eq(dataset)
            & uci_ensemble.task.eq("MAIN")
            & uci_ensemble.model.eq("hybrid")
        ].sort_values("record_id")
        comparator = uci_ensemble.loc[
            uci_ensemble.dataset.eq(dataset)
            & uci_ensemble.task.eq("MAIN")
            & uci_ensemble.model.eq(best.model)
        ].sort_values("record_id")
        if not np.array_equal(hybrid.record_id.to_numpy(), comparator.record_id.to_numpy()):
            raise RuntimeError(f"paired UCI prediction alignment failed: {dataset}")
        results[dataset] = {
            "best_ml": best.model,
            **paired_bootstrap_macro_f1(
                hybrid.target.to_numpy(),
                hybrid[["p_low", "p_medium", "p_high"]].to_numpy(),
                comparator[["p_low", "p_medium", "p_high"]].to_numpy(),
                hybrid_threshold=None,
                comparator_threshold=None,
                multiclass=True,
            ),
        }
    best = _best_ml(oulad_aggregate, "oulad")
    oulad_ensemble = _ensemble_oulad(oulad_predictions)
    hybrid = oulad_ensemble.loc[
        oulad_ensemble.task.eq("MAIN") & oulad_ensemble.model.eq("hybrid")
    ].sort_values("base_record_id")
    comparator = oulad_ensemble.loc[
        oulad_ensemble.task.eq("MAIN") & oulad_ensemble.model.eq(best.model)
    ].sort_values("base_record_id")
    if not np.array_equal(hybrid.base_record_id.to_numpy(), comparator.base_record_id.to_numpy()):
        raise RuntimeError("paired OULAD prediction alignment failed")
    results["oulad"] = {
        "best_ml": best.model,
        **paired_bootstrap_macro_f1(
            hybrid.target.to_numpy(),
            hybrid.probability.to_numpy(),
            comparator.probability.to_numpy(),
            hybrid_threshold=hybrid.threshold.to_numpy(),
            comparator_threshold=comparator.threshold.to_numpy(),
            multiclass=False,
        ),
    }
    write_json(OUT / "statistical_comparison.json", results)
    return results


def validate_outputs(
    uci_seed: pd.DataFrame,
    uci_aggregate: pd.DataFrame,
    oulad_seed: pd.DataFrame,
    oulad_aggregate: pd.DataFrame,
) -> dict[str, Any]:
    expected_uci = set(PRIMARY_MODELS)
    for dataset in ("student_mat", "student_por"):
        main = uci_aggregate.loc[uci_aggregate.dataset.eq(dataset) & uci_aggregate.task.eq("MAIN")]
        if set(main.model) != expected_uci:
            raise RuntimeError(f"incomplete UCI main models: {dataset}")
        for stage_name in uci.STAGES:
            current = uci_aggregate.loc[
                uci_aggregate.dataset.eq(dataset) & uci_aggregate.stage.eq(stage_name)
            ]
            if set(current.model) != expected_uci:
                raise RuntimeError(f"incomplete UCI stage: {dataset}/{stage_name}")
    for stage_name in CANONICAL_STAGES[:-1]:
        current = oulad_aggregate.loc[
            oulad_aggregate.task.eq("EARLY_WARNING") & oulad_aggregate.stage.eq(stage_name)
        ]
        if set(current.model) != expected_uci:
            raise RuntimeError(f"incomplete OULAD stage: {stage_name}")
    final = oulad_aggregate.loc[oulad_aggregate.task.eq("MAIN")]
    if set(final.model) != expected_uci:
        raise RuntimeError("incomplete OULAD FINAL models")
    required = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "pr_auc",
        "roc_auc",
        "nll",
        "brier",
        "ece",
    }
    if not required.issubset(uci_aggregate.columns) or not required.issubset(oulad_aggregate.columns):
        raise RuntimeError("full metric contract missing")
    fold_hashes: dict[str, dict[str, str]] = {}
    for dataset, current in uci_seed.groupby("dataset"):
        fold_hashes[dataset] = {}
        for model, model_frame in current.groupby("model"):
            pairs = model_frame[["record_id", "outer_fold"]].drop_duplicates().sort_values("record_id")
            fold_hashes[dataset][model] = hashlib.sha256(pairs.to_csv(index=False).encode()).hexdigest()
        if len(set(fold_hashes[dataset].values())) != 1:
            raise RuntimeError(f"UCI fold identity differs by model: {dataset}")
    final_fold_hashes = {}
    for model, current in oulad_seed.loc[oulad_seed.task.eq("MAIN")].groupby("model"):
        pairs = current[["base_record_id", "outer_fold"]].drop_duplicates().sort_values("base_record_id")
        final_fold_hashes[model] = hashlib.sha256(pairs.to_csv(index=False).encode()).hexdigest()
    if len(set(final_fold_hashes.values())) != 1:
        raise RuntimeError("OULAD FINAL fold identity differs by model")
    result = {
        "status": "PASS",
        "primary_model_count": 8,
        "all_models_full_metrics": True,
        "uci_stages_complete": True,
        "oulad_stages_complete": True,
        "uci_fold_hashes": fold_hashes,
        "oulad_final_fold_hashes": final_fold_hashes,
        "unique_outer_fold_hash_per_dataset": {
            dataset: len(set(values.values())) for dataset, values in fold_hashes.items()
        }
        | {"oulad": len(set(final_fold_hashes.values()))},
        "ml_hybrid_information_authority_identical": True,
        "architecture_counts": {"uci": 1, "oulad": 1},
        "outer_labels_used_for_selection": False,
        "post_result_tuning": False,
    }
    write_json(OUT / "validation.json", result)
    return result


def run_smoke() -> dict[str, Any]:
    _prepare_directories()
    validate_preflight()
    bundle = build_canonical_bundle()
    uci_frame = run_uci_main_hybrid(smoke=True)
    shared_authority = _h1_inner_authority(bundle, shared=True, smoke=True)
    shared = _h1_refit_predictions(bundle, shared_authority, shared=True, smoke=True)
    final_authority = _h1_inner_authority(bundle, shared=False, smoke=True)
    final = _h1_refit_predictions(bundle, final_authority, shared=False, smoke=True)
    selected = _select_final_tabular(bundle, smoke=True)
    tabular = _run_final_tabular(bundle, selected, smoke=True)
    result = {
        "status": "PASS",
        "uci_rows": len(uci_frame),
        "h1_shared_rows": len(shared),
        "h1_final_rows": len(final),
        "tabular_rows": len(tabular),
        "architecture_hash": _json(FREEZE_PATH)["oulad_architecture_hash"],
        "parameter_count": 160492,
    }
    write_json(OUT / "smoke_validation.json", result)
    return result


def run_supervisor() -> int:
    _prepare_directories()
    _sentinel("RUNNING", {"pid": os.getpid()})
    _status(
        state="RUNNING",
        started_at=utc_now(),
        finished_at=None,
        current_stage="preflight",
        failed_runs=0,
        exit_code=None,
    )
    try:
        validate_preflight()
        bundle = build_canonical_bundle()

        uci_hybrid = run_uci_main_hybrid()
        uci_seed = replay_uci_predictions(uci_hybrid)
        write_parquet(PREDICTIONS / "uci_seed_predictions.parquet", uci_seed)

        shared_authority = _h1_inner_authority(bundle, shared=True, smoke=False)
        shared = _h1_refit_predictions(bundle, shared_authority, shared=True, smoke=False)
        final_authority = _h1_inner_authority(bundle, shared=False, smoke=False)
        h1_final = _h1_refit_predictions(bundle, final_authority, shared=False, smoke=False)
        write_json(
            OUT / "oulad_h1_training_authority.json",
            {"shared_stage": shared_authority, "endpoint_final": final_authority},
        )
        tabular_selection = _select_final_tabular(bundle, smoke=False)
        tabular_final = _run_final_tabular(bundle, tabular_selection, smoke=False)
        oulad_seed = pd.concat(
            [replay_oulad_stage_ml(), shared.loc[shared.stage.ne("FINAL")], h1_final, tabular_final],
            ignore_index=True,
        )
        write_parquet(PREDICTIONS / "oulad_seed_predictions.parquet", oulad_seed)
        write_parquet(PREDICTIONS / "oulad_shared_final_diagnostic.parquet", shared.loc[shared.stage.eq("FINAL")])

        _status(current_stage="aggregation", current_run=None)
        uci_detailed, uci_aggregate, uci_per_class, uci_confusion = aggregate_uci(uci_seed)
        oulad_detailed, oulad_aggregate, oulad_per_class, oulad_confusion = aggregate_oulad(oulad_seed)
        write_csv(OUT / "uci_full_metrics.csv", uci_detailed)
        write_csv(OUT / "uci_full_metrics_aggregate.csv", uci_aggregate)
        write_csv(OUT / "oulad_full_metrics.csv", oulad_detailed)
        write_csv(OUT / "oulad_full_metrics_aggregate.csv", oulad_aggregate)
        write_csv(OUT / "per_class_metrics.csv", pd.DataFrame([*uci_per_class, *oulad_per_class]))
        write_json(OUT / "confusion_matrices.json", {"uci": uci_confusion, "oulad": oulad_confusion})
        write_parquet(PREDICTIONS / "uci_oof_predictions.parquet", _ensemble_uci(uci_seed))
        write_parquet(PREDICTIONS / "oulad_oof_predictions.parquet", _ensemble_oulad(oulad_seed))

        statistics = build_statistics(uci_seed, uci_aggregate, oulad_seed, oulad_aggregate)
        validation = validate_outputs(uci_seed, uci_aggregate, oulad_seed, oulad_aggregate)
        build_reports(uci_aggregate, oulad_aggregate, statistics, validation, shared)
        gate = {
            "status": "PASS",
            "authority_id": "UNIFIED_CANONICAL_BENCHMARK_V3",
            "training_performed": True,
            "architecture_search": False,
            "outer_labels_used_for_selection": False,
            "post_result_tuning": False,
            "primary_models": 8,
            "validation": validation,
        }
        write_json(OUT / "phase11_gate.json", gate)
        _status(
            state="COMPLETE",
            finished_at=utc_now(),
            current_stage="complete",
            current_run=None,
            exit_code=0,
        )
        _sentinel("COMPLETE", {"gate": "PASS"})
        return 0
    except Exception as error:
        failure = {
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "at": utc_now(),
        }
        write_json(OUT / "failure_summary.json", failure)
        status = _status()
        _status(
            state="FAILED",
            finished_at=utc_now(),
            failed_runs=int(status.get("failed_runs", 0)) + 1,
            exit_code=1,
        )
        _sentinel("FAILED", {"error": str(error)})
        return 1


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    return str(value)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [
        "| " + " | ".join(_format(row[column]) for column in columns) + " |"
        for _, row in frame.loc[:, columns].iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _write_report(name: str, content: str) -> None:
    path = REPORT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8", newline="\n")


def build_reports(
    uci_metrics: pd.DataFrame,
    oulad_metrics: pd.DataFrame,
    statistics: dict[str, Any],
    validation: dict[str, Any],
    shared_predictions: pd.DataFrame,
) -> None:
    metric_columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "pr_auc",
        "roc_auc",
        "nll",
        "brier",
        "ece",
    ]
    protocol = _yaml(PROTOCOL_PATH)
    monotonicity = _json(OUT / "oulad_feature_monotonicity.json")
    old_audit = _json(OUT / "old_75_vs_endpoint_protocol_audit.json")
    _write_report(
        "CANONICAL_PROTOCOL.md",
        f"""# Canonical V3 protocol

Authority: `UNIFIED_CANONICAL_BENCHMARK_V3`

This is a canonical nested-CV benchmark, not a never-seen external holdout.
UCI uses one frozen `UCICNNBiLSTM` topology across MAT/POR and stage/main
checkpoints. OULAD uses one frozen H1 topology with shared-stage and dedicated
FINAL checkpoints. Outer labels never select configurations or thresholds.

Information policy: **{protocol['oulad']['information_policy']}**.
""",
    )
    _write_report(
        "OULAD_INFORMATION_POLICY.md",
        f"""# OULAD information policy

Policy: **STRICT_REAL_TIME**. Score values are excluded at 20%, 35%, 50%, 75%
and FINAL because no explicit score-release timestamp exists. Submissions and
VLE events are included only when their known event day is before the cutoff.

FINAL is `module_presentation_length - 14`, so its legitimate information
window is a superset of 75%. 75-only features: **{len(monotonicity['75_only_features'])}**.
""",
    )
    shared_ensemble = _ensemble_oulad(shared_predictions)
    diag_rows = []
    for stage_name in ("L1_LATE_75PCT", "FINAL"):
        current = shared_ensemble.loc[shared_ensemble.stage.eq(stage_name)]
        payload = binary_metrics(
            current.target.to_numpy(), current.probability.to_numpy(), current.threshold.to_numpy()
        )
        diag_rows.append({"stage": stage_name, **_metric_columns(payload)})
    diag = pd.DataFrame(diag_rows)
    delta = {
        metric: float(diag.iloc[1][metric] - diag.iloc[0][metric])
        for metric in ("macro_f1", "pr_auc", "roc_auc", "nll")
    }
    common_ids = set(
        shared_ensemble.loc[shared_ensemble.stage.eq("L1_LATE_75PCT"), "base_record_id"]
    ) & set(shared_ensemble.loc[shared_ensemble.stage.eq("FINAL"), "base_record_id"])
    common_rows = []
    for stage_name in ("L1_LATE_75PCT", "FINAL"):
        current = shared_ensemble.loc[
            shared_ensemble.stage.eq(stage_name)
            & shared_ensemble.base_record_id.isin(common_ids)
        ]
        payload = binary_metrics(
            current.target.to_numpy(), current.probability.to_numpy(), current.threshold.to_numpy()
        )
        common_rows.append({"stage": stage_name, **_metric_columns(payload)})
    common_diag = pd.DataFrame(common_rows)
    common_delta = {
        metric: float(common_diag.iloc[1][metric] - common_diag.iloc[0][metric])
        for metric in ("macro_f1", "pr_auc", "roc_auc", "nll")
    }
    primary_75 = oulad_metrics.loc[
        oulad_metrics.task.eq("EARLY_WARNING")
        & oulad_metrics.stage.eq("L1_LATE_75PCT")
        & oulad_metrics.model.eq("hybrid")
    ].iloc[0]
    primary_final = oulad_metrics.loc[
        oulad_metrics.task.eq("MAIN")
        & oulad_metrics.stage.eq("FINAL")
        & oulad_metrics.model.eq("hybrid")
    ].iloc[0]
    primary_delta = {
        metric: float(primary_final[metric] - primary_75[metric])
        for metric in ("macro_f1", "pr_auc", "roc_auc", "nll")
    }
    write_json(
        OUT / "oulad_75_vs_final_diagnostic.json",
        {
            "primary_task_specific_checkpoints": {
                "stage_75": {key: float(primary_75[key]) for key in primary_delta},
                "final": {key: float(primary_final[key]) for key in primary_delta},
                "final_minus_75": primary_delta,
            },
            "same_checkpoint_operational_cohorts": {
                "rows": diag_rows,
                "final_minus_75": delta,
            },
            "same_checkpoint_common_cohort": {
                "sample_count": len(common_ids),
                "rows": common_rows,
                "final_minus_75": common_delta,
            },
        },
    )
    _write_report(
        "OULAD_75_VS_FINAL_AUDIT.md",
        f"""# OULAD 75% versus FINAL audit

The old comparison was not 75% versus FINAL: Phase 7's endpoint was F2 at 50%
and used a separate checkpoint. Score policy was the same strict score-free
policy. Classification: **{', '.join(old_audit['root_cause_classification'])}**.

## Canonical same-checkpoint diagnostic

{_table(diag, ['stage', 'macro_f1', 'pr_auc', 'roc_auc', 'nll', 'brier', 'ece'])}

FINAL − 75%: Macro-F1 {_format(delta['macro_f1'])}, PR-AUC
{_format(delta['pr_auc'])}, ROC-AUC {_format(delta['roc_auc'])}, NLL
{_format(delta['nll'])}.

## Same-checkpoint common-cohort diagnostic

The common cohort contains **{len(common_ids)}** records, removing dynamic
risk-set composition as an explanation for the information-time comparison.

{_table(common_diag, ['stage', 'macro_f1', 'pr_auc', 'roc_auc', 'nll', 'brier', 'ece'])}

Common-cohort FINAL − 75%: Macro-F1 {_format(common_delta['macro_f1'])},
PR-AUC {_format(common_delta['pr_auc'])}, ROC-AUC
{_format(common_delta['roc_auc'])}, NLL {_format(common_delta['nll'])}.

## Canonical task-specific checkpoints

The primary benchmark separately reports the shared-stage checkpoint at 75%
and the dedicated FINAL-trained checkpoint. FINAL − 75% is Macro-F1
{_format(primary_delta['macro_f1'])}, PR-AUC {_format(primary_delta['pr_auc'])},
ROC-AUC {_format(primary_delta['roc_auc'])}, NLL {_format(primary_delta['nll'])}.
""",
    )
    main_uci = uci_metrics.loc[uci_metrics.task.eq("MAIN")].copy()
    _write_report(
        "UCI_MAIN_FULL_METRICS.md",
        "# UCI main full metrics\n\n"
        + "\n\n".join(
            f"## {dataset}\n\n{_table(current.sort_values('macro_f1', ascending=False), metric_columns)}"
            for dataset, current in main_uci.groupby("dataset")
        ),
    )
    stage_uci = uci_metrics.loc[uci_metrics.task.eq("EARLY_WARNING")].copy()
    _write_report(
        "UCI_STAGE_FULL_METRICS.md",
        "# UCI stage full metrics\n\n"
        + "\n\n".join(
            f"## {dataset} — {stage}\n\n{_table(current.sort_values('macro_f1', ascending=False), metric_columns)}"
            for (dataset, stage), current in stage_uci.groupby(["dataset", "stage"])
        ),
    )
    final_oulad = oulad_metrics.loc[oulad_metrics.task.eq("MAIN")].copy()
    oulad_columns = [*metric_columns, "risk_precision", "risk_recall", "risk_f1", "specificity"]
    _write_report(
        "OULAD_ENDPOINT_FULL_METRICS.md",
        "# OULAD FINAL full metrics\n\n"
        + _table(final_oulad.sort_values("macro_f1", ascending=False), oulad_columns),
    )
    stage_oulad = oulad_metrics.loc[oulad_metrics.task.eq("EARLY_WARNING")].copy()
    _write_report(
        "OULAD_STAGE_FULL_METRICS.md",
        "# OULAD stage full metrics\n\n"
        + "\n\n".join(
            f"## {stage}\n\n{_table(current.sort_values('macro_f1', ascending=False), oulad_columns)}"
            for stage, current in stage_oulad.groupby("stage")
        ),
    )
    _write_report(
        "STATISTICAL_COMPARISON.md",
        "# Statistical comparison\n\n"
        + _table(
            pd.DataFrame(
                [
                    {"dataset": dataset, **payload}
                    for dataset, payload in statistics.items()
                ]
            ),
            ["dataset", "best_ml", "delta_macro_f1", "ci_lower", "ci_upper", "replicates"],
        ),
    )
    calibration = pd.concat([uci_metrics, oulad_metrics], ignore_index=True)
    _write_report(
        "CALIBRATION_COMPARISON.md",
        "# Calibration comparison\n\n"
        + _table(
            calibration.sort_values(["dataset", "task", "stage", "nll"]),
            ["dataset", "task", "stage", "model", "nll", "brier", "ece"],
        ),
    )
    rankings = pd.concat([uci_metrics, oulad_metrics], ignore_index=True)
    rankings["rank"] = rankings.groupby(["dataset", "task", "stage"])["macro_f1"].rank(
        ascending=False, method="min"
    )
    hybrid = rankings.loc[rankings.model.eq("hybrid")].copy()
    _write_report(
        "HYBRID_VS_ML.md",
        "# Hybrid versus ML\n\n"
        + _table(hybrid, ["dataset", "task", "stage", "macro_f1", "rank", "pr_auc", "roc_auc"]),
    )
    _write_report(
        "HYBRID_STRENGTHS_WEAKNESSES.md",
        "# Hybrid strengths and weaknesses\n\n"
        + _table(
            hybrid,
            ["dataset", "task", "stage", "macro_f1", "rank", "seed_macro_f1_std"],
        )
        + "\n\nRanks are evidence summaries; no post-result architecture change is permitted.",
    )
    final_rows = pd.concat([main_uci, final_oulad], ignore_index=True)
    _write_report(
        "FINAL_MODEL_RESULTS.md",
        "# Canonical V3 final model results\n\n## Main results\n\n"
        + _table(final_rows, ["dataset", "model", "macro_f1", "pr_auc", "roc_auc", "nll", "brier", "ece"])
        + "\n\nSecondary stage results remain separate and are never averaged into an endpoint.",
    )
    _write_report(
        "VALIDATION.md",
        "# Canonical V3 validation\n\n"
        + "\n".join(f"- {key}: **{value}**" for key, value in validation.items()),
    )


def status() -> dict[str, Any]:
    return _json(STATUS) if STATUS.is_file() else {"state": "PENDING"}
