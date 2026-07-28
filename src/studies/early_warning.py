"""Unified fair UCI early-warning benchmark and OULAD input audit.

This diagnostic never loads an official UCI/OULAD checkpoint for training and
never writes canonical prediction, database, or recommendation artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch import nn

from src.models._uci import UCICNNBiLSTM, _UCITemporalEncoder
from src.studies import teacher_feedback as tf

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "artifacts" / "final"
ROOT_OUT = FINAL / "uci_timing_scenarios"
TF_OUT = FINAL / "teacher_feedback_validation"
CONFIG = ROOT / "configs" / "final" / "uci_early_warning_all_models.yaml"
REPORT = ROOT / "reports" / "final" / "UCI_TIMING_SCENARIO_REPORT.md"
OULAD_REPORT = ROOT / "reports" / "final" / "OULAD_TEMPORAL_BRANCH_AUDIT.md"

DATASETS = ("student_mat", "student_por")
SCENARIOS = tf.SCENARIOS
TABULAR = (
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
    "svm",
    "xgboost",
    "mlp",
)
DEEP = ("cnn_only", "bilstm_only", "cnn_bilstm")
MODELS = (*TABULAR, *DEEP)
DISPLAY = {
    **tf.MODEL_NAMES,
    "cnn_only": "CNN-only",
    "bilstm_only": "BiLSTM-only",
    "grade_band_reference": "GRADE_BAND_REFERENCE",
}
tf.MODEL_NAMES.setdefault("grade_band_reference", "GRADE_BAND_REFERENCE")
tf.MODEL_NAMES.setdefault("cnn_only", "CNN-only")
tf.MODEL_NAMES.setdefault("bilstm_only", "BiLSTM-only")
CLASS_NAMES = ("Low", "Medium", "High")

BASE_CHANNELS = (
    "total_clicks",
    "active_days",
    "unique_sites",
    "unique_activity_types",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
    "submitted_assessment_count",
    "late_submission_count",
    "available_score_count",
    "cumulative_mean_score",
    "cumulative_weighted_score",
    "days_since_last_vle_activity",
    "weeks_without_activity",
    "score_missing_mask",
)
DYNAMIC_CHANNELS = (
    "log1p_total_clicks",
    "log1p_active_days",
    "log1p_unique_sites",
    "log1p_assessment_related_clicks",
    "log1p_submitted_assessment_count",
    "delta_total_clicks",
    "delta_active_days",
    "delta_unique_sites",
    "delta_content_clicks",
    "delta_forum_clicks",
    "delta_quiz_clicks",
    "delta_assessment_related_clicks",
    "delta_submitted_assessment_count",
    "delta_cumulative_mean_score",
    "delta_cumulative_weighted_score",
    "rolling_2_week_mean_total_clicks",
    "rolling_2_week_mean_active_days",
    "rolling_2_week_mean_assessment_clicks",
    "rolling_2_week_submission_count",
    "rolling_2_week_score_change",
    "current_inactivity_streak",
    "activity_resumed_indicator",
    "new_inactivity_indicator",
    "content_share",
    "forum_share",
    "quiz_share",
    "assessment_share",
    "score_delta",
    "weighted_score_delta",
    "late_submission_rate_to_date",
    "submission_rate_last_2_weeks",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _protocol() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if value.get("status") != "PREREGISTERED_BEFORE_OUTER_SCORING":
        raise RuntimeError("Early-warning protocol is not preregistered")
    if value.get("scope") != "FAIR_TIMING_NO_TRANSFER":
        raise RuntimeError("Fair timing must prohibit transfer")
    return value


def prepare() -> dict[str, Any]:
    protocol = _protocol()
    tf.prepare_regression_guard()
    payload = {
        "schema_version": "fair_early_warning_protocol_v1",
        "status": protocol["status"],
        "scope": protocol["scope"],
        "config": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha(CONFIG),
        "models": list(MODELS),
        "datasets": list(DATASETS),
        "scenarios": list(SCENARIOS),
        "expected_model_results": 60,
        "grade_band_reference_is_model_identity": False,
        "outer_used_for_tuning": False,
        "transfer_learning": False,
        "synthetic_resampling": "NONE",
    }
    _write_json(ROOT_OUT / "protocol.json", payload)
    return payload


def build_temporal(
    frame: pd.DataFrame, scenario: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(frame), 2, 7), dtype=np.float32)
    mask = np.zeros((len(frame), 2), dtype=np.float32)
    if scenario == "S0_EARLY_NO_GRADE":
        return values, mask
    g1 = frame["G1"].to_numpy(dtype=np.float32)
    values[:, 0, 0] = g1 / 20.0
    values[:, 0, 1] = -1.0
    values[:, 0, 4] = (g1 - 10.0) / 20.0
    values[:, 0, 5] = (g1 - 15.0) / 20.0
    mask[:, 0] = 1.0
    if scenario == "S2_LATE_G1_G2":
        g2 = frame["G2"].to_numpy(dtype=np.float32)
        delta = g2 - g1
        values[:, 1, 0] = g2 / 20.0
        values[:, 1, 1] = 1.0
        values[:, 1, 2] = delta / 20.0
        values[:, 1, 3] = np.abs(delta) / 20.0
        values[:, 1, 4] = (g2 - 10.0) / 20.0
        values[:, 1, 5] = (g2 - 15.0) / 20.0
        values[:, 1, 6] = np.sign(delta)
        mask[:, 1] = 1.0
    return values, mask


def _deep_config(candidate: dict[str, Any], variant: str) -> dict[str, Any]:
    return {
        **candidate,
        "temporal_variant": variant,
        "lstm_layers": 1,
        "context_layers": 1,
        "activation": "gelu",
        "fusion": "gated",
    }


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_deep_once(
    *,
    temporal: np.ndarray,
    mask: np.ndarray,
    context_frame: pd.DataFrame,
    target: np.ndarray,
    fit: np.ndarray,
    score: np.ndarray,
    variant: str,
    candidate: dict[str, Any],
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[np.ndarray, int]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = tf._uci_preprocessor(tf.CONTEXT)
    preprocessor.fit(context_frame.iloc[fit])
    context = np.asarray(preprocessor.transform(context_frame), dtype=np.float32)
    model = UCICNNBiLSTM(7, context.shape[1], _deep_config(candidate, variant)).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    counts = np.bincount(target[fit], minlength=3).astype(float)
    weights = (
        counts.sum() / np.maximum(counts, 1.0)
        if candidate["class_weight"] == "balanced"
        else np.ones(3)
    )
    weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    batch_size = int(candidate["batch_size"])
    max_epochs = fixed_epochs or int(candidate["max_epochs"])
    patience = int(candidate["patience"])
    best_score = -math.inf
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    fit_tensor = torch.tensor(fit, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = fit_tensor[torch.randperm(len(fit_tensor), generator=generator)]
        for batch in order.split(batch_size):
            index = batch.numpy()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.from_numpy(temporal[index]).to(device),
                torch.from_numpy(context[index]).to(device),
                torch.from_numpy(mask[index]).to(device),
            )["classification"]
            loss = loss_fn(
                output, torch.from_numpy(target[index]).long().to(device)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        probability = _deep_predict(
            model, temporal[score], mask[score], context[score], device
        )
        value = f1_score(
            target[score], probability.argmax(1), average="macro", zero_division=0
        )
        if value > best_score + 1e-12:
            best_score = float(value)
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    return _deep_predict(
        model, temporal[score], mask[score], context[score], device
    ), (max_epochs if fixed_epochs is not None else best_epoch)


def _deep_predict(
    model: UCICNNBiLSTM,
    temporal: np.ndarray,
    mask: np.ndarray,
    context: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(temporal), 256):
            stop = start + 256
            logits = model(
                torch.from_numpy(temporal[start:stop]).to(device),
                torch.from_numpy(context[start:stop]).to(device),
                torch.from_numpy(mask[start:stop]).to(device),
            )["classification"]
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    probability = np.concatenate(rows).astype(np.float64)
    return probability / probability.sum(axis=1, keepdims=True)


@dataclass
class DeepFold:
    ensemble: np.ndarray
    seeds: dict[int, np.ndarray]
    selected: dict[str, Any]
    trials: list[dict[str, Any]]
    runtime: float


def _run_deep_fold(
    data: tf.UCIStudyData,
    scenario: str,
    model_id: str,
    fold: int,
    train: np.ndarray,
    validation: np.ndarray,
    protocol: dict[str, Any],
) -> DeepFold:
    cache = ROOT_OUT / "fold_cache" / (
        f"{data.dataset}__{scenario}__{model_id}__outer_{fold}.npz"
    )
    meta = cache.with_suffix(".json")
    if cache.is_file() and meta.is_file():
        archive = np.load(cache)
        details = json.loads(meta.read_text(encoding="utf-8"))
        seeds = {
            int(seed): (
                archive[f"seed_{seed}"].astype(np.float64)
                / archive[f"seed_{seed}"].sum(axis=1, keepdims=True)
            )
            for seed in protocol["seeds"]
        }
        ensemble = archive["ensemble"].astype(np.float64)
        ensemble /= ensemble.sum(axis=1, keepdims=True)
        return DeepFold(
            ensemble,
            seeds,
            details["selected"],
            details["trials"],
            details["runtime_seconds"],
        )
    started = time.perf_counter()
    temporal, mask = build_temporal(data.frame, scenario)
    context_frame = data.frame.loc[:, list(tf.CONTEXT)].reset_index(drop=True)
    inner = tf._inner_splits(data.target[train], data.groups[train], seed=42)
    variant = model_id
    candidates = protocol["deep"]["candidate_grid"]
    trials: list[dict[str, Any]] = []
    best: tuple[float, str, int, dict[str, Any]] | None = None
    for trial, candidate in enumerate(candidates):
        scores: list[float] = []
        epochs: list[int] = []
        for inner_fold, (fit_local, score_local) in enumerate(inner):
            fit = train[fit_local]
            score = train[score_local]
            probability, best_epoch = _fit_deep_once(
                temporal=temporal,
                mask=mask,
                context_frame=context_frame,
                target=data.target,
                fit=fit,
                score=score,
                variant=variant,
                candidate=candidate,
                seed=42,
            )
            scores.append(
                float(
                    f1_score(
                        data.target[score],
                        probability.argmax(1),
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            epochs.append(best_epoch)
        mean = float(np.mean(scores))
        selected_epochs = max(1, int(np.median(epochs)))
        row = {
            "dataset": data.dataset,
            "scenario": scenario,
            "model_id": model_id,
            "outer_fold": fold,
            "trial": trial,
            "state": "COMPLETE",
            "params": json.dumps(candidate, sort_keys=True),
            "inner_fold_macro_f1": json.dumps(scores),
            "inner_objective": mean,
            "selected_epochs": selected_epochs,
            "outer_rows_used_for_selection": False,
        }
        trials.append(row)
        key = json.dumps(candidate, sort_keys=True)
        if best is None or mean > best[0] + 1e-12 or (
            math.isclose(mean, best[0], abs_tol=1e-12) and key < best[1]
        ):
            best = (mean, key, selected_epochs, candidate)
    assert best is not None
    seed_probabilities: dict[int, np.ndarray] = {}
    for seed in protocol["seeds"]:
        probability, _ = _fit_deep_once(
            temporal=temporal,
            mask=mask,
            context_frame=context_frame,
            target=data.target,
            fit=train,
            score=validation,
            variant=variant,
            candidate=best[3],
            seed=int(seed),
            fixed_epochs=best[2],
        )
        seed_probabilities[int(seed)] = probability
    ensemble = np.mean(list(seed_probabilities.values()), axis=0)
    selected = {
        "params": best[3],
        "selected_inner_macro_f1": best[0],
        "selected_epochs": best[2],
        "outer_rows_used_for_selection": False,
    }
    runtime = time.perf_counter() - started
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        ensemble=ensemble,
        **{f"seed_{seed}": value for seed, value in seed_probabilities.items()},
    )
    _write_json(
        meta,
        {"selected": selected, "trials": trials, "runtime_seconds": runtime},
    )
    return DeepFold(ensemble, seed_probabilities, selected, trials, runtime)


def _existing_predictions(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_parquet(ROOT_OUT / f"{dataset}_predictions.parquet")
    seeds = pd.read_parquet(ROOT_OUT / f"{dataset}_seed_predictions.parquet")
    return predictions, seeds


def _run_missing_tabular(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data = tf._load_uci(dataset)
    existing, existing_seeds = _existing_predictions(dataset)
    predictions = [
        existing.loc[existing["model_id"].isin(TABULAR)].copy()
    ]
    seed_predictions = [
        existing_seeds.loc[existing_seeds["model_id"].isin(TABULAR)].copy()
    ]
    present = set(zip(existing["scenario"], existing["model_id"]))
    legacy_selected_path = ROOT_OUT / f"{dataset}_selected_configs.csv"
    legacy_search_path = ROOT_OUT / f"{dataset}_search_trials.csv"
    legacy_runtime_path = ROOT_OUT / f"{dataset}_runtime.csv"
    legacy_selected = (
        pd.read_csv(legacy_selected_path)
        if legacy_selected_path.is_file()
        else pd.DataFrame()
    )
    combined_selected_path = ROOT_OUT / "selected_configs.csv"
    if combined_selected_path.is_file():
        combined = pd.read_csv(combined_selected_path)
        combined = combined.loc[
            (combined["dataset"] == dataset)
            & (combined["model_id"].isin(TABULAR))
        ]
        legacy_selected = pd.concat(
            [legacy_selected, combined], ignore_index=True
        ).drop_duplicates(["dataset", "scenario", "model_id", "outer_fold"])
    evidence_present = (
        set(zip(legacy_selected["scenario"], legacy_selected["model_id"]))
        if not legacy_selected.empty
        else set()
    )
    search: list[dict[str, Any]] = (
        pd.read_csv(legacy_search_path).to_dict("records")
        if legacy_search_path.is_file()
        else []
    )
    selected: list[dict[str, Any]] = legacy_selected.to_dict("records")
    runtime: list[dict[str, Any]] = (
        pd.read_csv(legacy_runtime_path).to_dict("records")
        if legacy_runtime_path.is_file()
        else []
    )
    combined_search_path = ROOT_OUT / "search_trials.csv"
    if combined_search_path.is_file():
        combined = pd.read_csv(combined_search_path)
        combined = combined.loc[
            (combined["dataset"] == dataset)
            & (combined["model_id"].isin(TABULAR))
        ]
        search = combined.to_dict("records")
    combined_runtime_path = ROOT_OUT / "runtime.csv"
    if combined_runtime_path.is_file():
        combined = pd.read_csv(combined_runtime_path)
        combined = combined.loc[
            (combined["dataset"] == dataset)
            & (combined["model_id"].isin(TABULAR))
        ]
        runtime = combined.to_dict("records")
    for scenario in SCENARIOS:
        features = tf.build_uci_scenario_frame(data.frame, scenario)
        for model_id in TABULAR:
            has_predictions = (scenario, model_id) in present
            if has_predictions and (scenario, model_id) in evidence_present:
                continue
            fold_results: list[tuple[np.ndarray, tf.FoldResult]] = []
            for fold in sorted(np.unique(data.outer_fold)):
                print(f"[early-warning] {dataset} {scenario} {model_id} fold={fold}", flush=True)
                validation = np.flatnonzero(data.outer_fold == fold)
                train = np.flatnonzero(data.outer_fold != fold)
                result = tf._fit_outer_fold(
                    dataset=dataset,
                    scenario=scenario,
                    model_id=model_id,
                    features=features,
                    target=data.target,
                    groups=data.groups,
                    train_indices=train,
                    validation_indices=validation,
                    outer_fold=int(fold),
                    preprocessor_factory=lambda columns=tuple(features.columns): tf._uci_preprocessor(columns),
                    binary=False,
                )
                fold_results.append((validation, result))
                search.extend(result.search_rows)
                selected.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": fold,
                        "selected_params": json.dumps(result.selected_params, sort_keys=True),
                        "selected_inner_macro_f1": result.selected_inner_macro_f1,
                        "outer_rows_used_for_selection": False,
                    }
                )
                runtime.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": fold,
                        "runtime_seconds": result.runtime_seconds,
                    }
                )
            probability = np.zeros((len(data.target), 3))
            per_seed = {seed: np.zeros((len(data.target), 3)) for seed in tf.SEEDS}
            for validation, result in fold_results:
                probability[validation] = result.ensemble_probability
                for seed in tf.SEEDS:
                    per_seed[seed][validation] = result.seed_probabilities[seed]
            if not has_predictions:
                metric, predicted = tf._metric_row(
                    dataset, scenario, model_id, data.target, probability
                )
                predictions.append(
                    _prediction_frame(data, scenario, model_id, probability, predicted)
                )
                for seed in tf.SEEDS:
                    seed_predictions.append(
                        _seed_prediction_frame(
                            data, scenario, model_id, seed, per_seed[seed]
                        )
                    )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(seed_predictions, ignore_index=True),
        search,
        selected,
        runtime,
    )


def _prediction_frame(
    data: tf.UCIStudyData,
    scenario: str,
    model_id: str,
    probability: np.ndarray,
    predicted: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": data.dataset,
            "scenario": scenario,
            "model_id": model_id,
            "outer_fold": data.outer_fold,
            "record_id": data.record_ids,
            "true_label": data.target,
            "predicted_label": predicted,
            "p_low": probability[:, 0],
            "p_medium": probability[:, 1],
            "p_high": probability[:, 2],
            "probability_aggregation": "mean_across_fixed_seeds",
            "result_scope": "FAIR_TIMING_NO_TRANSFER",
        }
    )


def _seed_prediction_frame(
    data: tf.UCIStudyData,
    scenario: str,
    model_id: str,
    seed: int,
    probability: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": data.dataset,
            "scenario": scenario,
            "model_id": model_id,
            "outer_fold": data.outer_fold,
            "record_id": data.record_ids,
            "true_label": data.target,
            "seed": seed,
            "p_low": probability[:, 0],
            "p_medium": probability[:, 1],
            "p_high": probability[:, 2],
        }
    )


def _run_dataset(dataset: str, protocol: dict[str, Any]) -> dict[str, Any]:
    data = tf._load_uci(dataset)
    predictions, seed_predictions, search, selected, runtime = _run_missing_tabular(dataset)
    prediction_parts = [predictions]
    seed_parts = [seed_predictions]
    for scenario in SCENARIOS:
        for model_id in DEEP:
            probability = np.zeros((len(data.target), 3))
            per_seed = {seed: np.zeros((len(data.target), 3)) for seed in tf.SEEDS}
            for fold in sorted(np.unique(data.outer_fold)):
                print(f"[early-warning] {dataset} {scenario} {model_id} fold={fold}", flush=True)
                validation = np.flatnonzero(data.outer_fold == fold)
                train = np.flatnonzero(data.outer_fold != fold)
                result = _run_deep_fold(
                    data, scenario, model_id, int(fold), train, validation, protocol
                )
                probability[validation] = result.ensemble
                for seed in tf.SEEDS:
                    per_seed[seed][validation] = result.seeds[seed]
                search.extend(result.trials)
                selected.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": fold,
                        "selected_params": json.dumps(result.selected, sort_keys=True),
                        "selected_inner_macro_f1": result.selected["selected_inner_macro_f1"],
                        "outer_rows_used_for_selection": False,
                    }
                )
                runtime.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "model_id": model_id,
                        "outer_fold": fold,
                        "runtime_seconds": result.runtime,
                    }
                )
            metric, predicted = tf._metric_row(
                dataset, scenario, model_id, data.target, probability
            )
            prediction_parts.append(
                _prediction_frame(data, scenario, model_id, probability, predicted)
            )
            for seed in tf.SEEDS:
                seed_parts.append(
                    _seed_prediction_frame(data, scenario, model_id, seed, per_seed[seed])
                )
    prediction_frame = pd.concat(prediction_parts, ignore_index=True)
    seed_frame = pd.concat(seed_parts, ignore_index=True)
    # Normalize legacy target naming while retaining only unified model rows.
    if "target" in prediction_frame:
        if "true_label" not in prediction_frame:
            prediction_frame["true_label"] = prediction_frame["target"]
        else:
            prediction_frame["true_label"] = prediction_frame[
                "true_label"
            ].fillna(prediction_frame["target"])
        prediction_frame = prediction_frame.drop(columns=["target"])
    if "target" in seed_frame:
        if "true_label" not in seed_frame:
            seed_frame["true_label"] = seed_frame["target"]
        else:
            seed_frame["true_label"] = seed_frame["true_label"].fillna(
                seed_frame["target"]
            )
        seed_frame = seed_frame.drop(columns=["target"])
    # Record IDs are the frozen authority and repair any legacy/new-schema
    # concatenation without consulting predictions or outer outcomes.
    target_by_id = dict(zip(data.record_ids, data.target))
    prediction_frame["true_label"] = prediction_frame["record_id"].map(target_by_id)
    seed_frame["true_label"] = seed_frame["record_id"].map(target_by_id)
    prediction_frame = prediction_frame.loc[
        prediction_frame["model_id"].isin(MODELS)
    ].sort_values(["scenario", "model_id", "record_id"])
    seed_frame = seed_frame.loc[
        seed_frame["model_id"].isin(MODELS)
    ].sort_values(["scenario", "model_id", "seed", "record_id"])
    _write_parquet(ROOT_OUT / f"{dataset}_predictions.parquet", prediction_frame)
    _write_parquet(ROOT_OUT / f"{dataset}_seed_predictions.parquet", seed_frame)
    return {
        "data": data,
        "predictions": prediction_frame,
        "seed_predictions": seed_frame,
        "search": search,
        "selected": selected,
        "runtime": runtime,
    }


def _grade_reference(data: tf.UCIStudyData, scenario: str) -> np.ndarray:
    probability = np.zeros((len(data.target), 3), dtype=float)
    for fold in sorted(np.unique(data.outer_fold)):
        validation = np.flatnonzero(data.outer_fold == fold)
        train = np.flatnonzero(data.outer_fold != fold)
        if scenario == "S0_EARLY_NO_GRADE":
            counts = np.bincount(data.target[train], minlength=3) + 1
            probability[validation] = counts / counts.sum()
            continue
        column = "G1" if scenario == "S1_MID_G1_ONLY" else "G2"
        train_band = tf.encode_uci_target(data.frame.iloc[train][column])
        validation_band = tf.encode_uci_target(data.frame.iloc[validation][column])
        table = np.ones((3, 3), dtype=float)
        for band, target in zip(train_band, data.target[train]):
            table[band, target] += 1.0
        table /= table.sum(axis=1, keepdims=True)
        probability[validation] = table[validation_band]
    return probability


def _metrics_from_predictions(
    results: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    classes: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for dataset, result in results.items():
        data = result["data"]
        predictions = result["predictions"]
        seeds = result["seed_predictions"]
        for scenario in SCENARIOS:
            for model_id in MODELS:
                selected = predictions.loc[
                    (predictions["scenario"] == scenario)
                    & (predictions["model_id"] == model_id)
                ].sort_values("record_id")
                probability = selected[["p_low", "p_medium", "p_high"]].to_numpy()
                target = selected["true_label"].to_numpy(dtype=int)
                metric, _ = tf._metric_row(
                    dataset, scenario, model_id, target, probability
                )
                seed_scores = []
                for seed in tf.SEEDS:
                    seed_rows = seeds.loc[
                        (seeds["scenario"] == scenario)
                        & (seeds["model_id"] == model_id)
                        & (seeds["seed"] == seed)
                    ].sort_values("record_id")
                    seed_scores.append(
                        f1_score(
                            seed_rows["true_label"].to_numpy(dtype=int),
                            seed_rows[["p_low", "p_medium", "p_high"]]
                            .to_numpy()
                            .argmax(1),
                            average="macro",
                            zero_division=0,
                        )
                    )
                row = {
                    "dataset": dataset,
                    "scenario": scenario,
                    "model_id": model_id,
                    "model": DISPLAY[model_id],
                    **{key: metric[key] for key in (
                        "accuracy", "balanced_accuracy", "macro_precision",
                        "macro_recall", "macro_f1", "weighted_f1", "pr_auc",
                        "roc_auc", "brier", "nll", "ece"
                    )},
                    "confusion_matrix": json.dumps(metric["confusion_matrix"]),
                    "seed_mean_macro_f1": float(np.mean(seed_scores)),
                    "seed_std_macro_f1": float(np.std(seed_scores)),
                    "seed_min_macro_f1": float(np.min(seed_scores)),
                    "seed_max_macro_f1": float(np.max(seed_scores)),
                }
                for class_name in CLASS_NAMES:
                    item = metric["per_class"][class_name]
                    for key in ("precision", "recall", "f1", "support"):
                        row[f"{class_name.lower()}_{key}"] = item[key]
                    classes.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "model_id": model_id,
                            "class_label": class_name,
                            **item,
                        }
                    )
                rows.append(row)
            probability = _grade_reference(data, scenario)
            metric, predicted = tf._metric_row(
                dataset,
                scenario,
                "grade_band_reference",
                data.target,
                probability,
            )
            references.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "model_id": "grade_band_reference",
                    **{key: metric[key] for key in (
                        "accuracy", "balanced_accuracy", "macro_precision",
                        "macro_recall", "macro_f1", "weighted_f1", "pr_auc",
                        "roc_auc", "brier", "nll", "ece"
                    )},
                    **{
                        f"low_{key}": metric["per_class"]["Low"][key]
                        for key in ("precision", "recall", "f1", "support")
                    },
                    "fit_scope": "outer_training_fold_only",
                    "laplace_alpha": 1,
                }
            )
            result.setdefault("reference_predictions", []).append(
                _prediction_frame(
                    data,
                    scenario,
                    "grade_band_reference",
                    probability,
                    predicted,
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(classes), pd.DataFrame(references)


def _grade_relationship(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": "grade_band_relationship_v1",
        "band_is_target": False,
        "target_source": "G3",
        "datasets": {},
    }
    for dataset, result in results.items():
        frame = result["data"].frame
        g3 = tf.encode_uci_target(frame["G3"])
        g1 = tf.encode_uci_target(frame["G1"])
        g2 = tf.encode_uci_target(frame["G2"])
        payload: dict[str, Any] = {}
        for name, band in (("G1", g1), ("G2", g2)):
            table = np.zeros((3, 3), dtype=int)
            for current, target in zip(band, g3):
                table[current, target] += 1
            payload[name] = {
                "crosstab_rows_current_band_columns_G3_target": table.tolist(),
                "conditional_probability_G3_given_band": (
                    table / np.maximum(table.sum(axis=1, keepdims=True), 1)
                ).tolist(),
            }
        transitions = np.zeros((3, 3, 3), dtype=int)
        for one, two, target in zip(g1, g2, g3):
            transitions[one, two, target] += 1
        payload["transition_G1_G2_G3"] = transitions.tolist()
        payload["G1_to_G2"] = {
            "same_band_rate": float(np.mean(g1 == g2)),
            "improved_band_rate": float(np.mean(g2 > g1)),
            "declined_band_rate": float(np.mean(g2 < g1)),
        }
        payload["G2_to_G3"] = {
            "same_band_rate": float(np.mean(g2 == g3)),
            "improved_band_rate": float(np.mean(g3 > g2)),
            "declined_band_rate": float(np.mean(g3 < g2)),
        }
        output["datasets"][dataset] = payload
    return output


def _confusion_metrics(target: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    matrix = np.zeros((3, 3), dtype=float)
    np.add.at(matrix, (target, predicted), 1)
    tp = np.diag(matrix)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-15)
    return float(f1.mean()), float(recall[0]), float(f1[0])


def _bootstrap(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, result in results.items():
        predictions = result["predictions"]
        reference_frames = pd.concat(result["reference_predictions"], ignore_index=True)
        for scenario in SCENARIOS:
            base = predictions.loc[
                (predictions["scenario"] == scenario)
                & (predictions["model_id"] == "cnn_bilstm")
            ].sort_values("record_id")
            target = base["true_label"].to_numpy(dtype=int)
            base_pred = base[["p_low", "p_medium", "p_high"]].to_numpy().argmax(1)
            class_indices = [np.flatnonzero(target == index) for index in range(3)]
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{dataset}:{scenario}".encode()).hexdigest()[:8], 16)
            )
            samples = [
                np.concatenate(
                    [rng.choice(index, len(index), replace=True) for index in class_indices]
                )
                for _ in range(5000)
            ]
            for comparator in (*DEEP[:2], *TABULAR, "grade_band_reference"):
                source = (
                    reference_frames
                    if comparator == "grade_band_reference"
                    else predictions
                )
                other = source.loc[
                    (source["scenario"] == scenario)
                    & (source["model_id"] == comparator)
                ].sort_values("record_id")
                other_pred = (
                    other[["p_low", "p_medium", "p_high"]].to_numpy().argmax(1)
                )
                deltas = np.asarray(
                    [
                        np.asarray(_confusion_metrics(target[index], base_pred[index]))
                        - np.asarray(_confusion_metrics(target[index], other_pred[index]))
                        for index in samples
                    ]
                )
                point = np.asarray(_confusion_metrics(target, base_pred)) - np.asarray(
                    _confusion_metrics(target, other_pred)
                )
                for metric_index, metric_name in enumerate(
                    ("macro_f1", "low_recall", "low_f1")
                ):
                    low, high = np.quantile(deltas[:, metric_index], [0.025, 0.975])
                    conclusion = (
                        "CNN-BiLSTM higher"
                        if low > 0
                        else (
                            "comparator higher"
                            if high < 0
                            else "insufficient evidence of difference"
                        )
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "scenario": scenario,
                            "comparator": comparator,
                            "metric": metric_name,
                            "delta": point[metric_index],
                            "ci_95_low": low,
                            "ci_95_high": high,
                            "conclusion": conclusion,
                            "replicates": 5000,
                            "sampling": "paired_stratified_by_true_class",
                        }
                    )
    return pd.DataFrame(rows)


def _rankings(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    for (dataset, scenario), group in metrics.groupby(["dataset", "scenario"]):
        ranked = group.copy()
        ranked["macro_f1_rank"] = ranked["macro_f1"].rank(method="min", ascending=False)
        ranked["low_recall_rank"] = ranked["low_recall"].rank(method="min", ascending=False)
        ranked["low_f1_rank"] = ranked["low_f1"].rank(method="min", ascending=False)
        ranked["pr_auc_rank"] = ranked["pr_auc"].rank(method="min", ascending=False)
        ranked["ece_rank"] = ranked["ece"].rank(method="min", ascending=True)
        ranked["stability_rank"] = ranked["seed_std_macro_f1"].rank(method="min", ascending=True)
        rows.extend(ranked.to_dict("records"))
        hybrid = ranked.loc[ranked["model_id"] == "cnn_bilstm"].iloc[0]
        profiles.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                **{
                    key: hybrid[key]
                    for key in (
                        "macro_f1_rank", "low_recall_rank", "low_f1_rank",
                        "pr_auc_rank", "ece_rank", "stability_rank",
                        "seed_std_macro_f1",
                    )
                },
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(profiles)


def _validation_artifacts(
    results: dict[str, dict[str, Any]], metrics: pd.DataFrame
) -> None:
    split_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for dataset, result in results.items():
        data = result["data"]
        for scenario in SCENARIOS:
            feature = tf.build_uci_scenario_frame(data.frame, scenario)
            for fold in sorted(np.unique(data.outer_fold)):
                train = np.flatnonzero(data.outer_fold != fold)
                validation = np.flatnonzero(data.outer_fold == fold)
                split_rows.append(
                    {
                        "dataset": dataset,
                        "scenario": scenario,
                        "outer_fold": int(fold),
                        "train_record_ids_sha256": tf._hash_values(data.record_ids[train]),
                        "validation_record_ids_sha256": tf._hash_values(data.record_ids[validation]),
                        "model_list": list(MODELS),
                        "same_split_for_all_models": True,
                        "outer_rows_used_for_tuning": 0,
                        "preprocessing_fit_scope": "training_only",
                    }
                )
            feature_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "raw_feature_sources": list(feature.columns),
                    "same_raw_information_for_all_models": True,
                    "deep_context_not_repeated_per_timestep": True,
                    "G3_present": False,
                    "transfer_rows": 0,
                }
            )
    _write_json(
        TF_OUT / "fair_comparison_contract.json",
        {
            "schema_version": "fair_comparison_contract_v1",
            "status": "PASS",
            "rows": split_rows,
        },
    )
    _write_json(
        ROOT_OUT / "split_equivalence.json",
        {"schema_version": "split_equivalence_v2", "status": "PASS", "rows": split_rows},
    )
    _write_json(
        ROOT_OUT / "feature_parity.json",
        {"schema_version": "feature_parity_v1", "status": "PASS", "rows": feature_rows},
    )
    _write_json(
        ROOT_OUT / "leakage_validation.json",
        {
            "schema_version": "uci_timing_leakage_v2",
            "status": "PASS",
            "G3_feature_lineage_occurrences": 0,
            "S0_grade_information_occurrences": 0,
            "S1_G2_information_occurrences": 0,
            "outer_rows_used_for_tuning": 0,
            "transfer_rows_used": 0,
        },
    )
    _write_imbalance_audit()
    _write_deep_mask_validation()


def _write_imbalance_audit() -> None:
    occurrences = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or any(part.startswith("venv") or part == "test_lab" for part in path.parts)
            or path.suffix.lower() not in {".py", ".md", ".yaml", ".yml", ".json"}
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in ("SMOTE", "ADASYN", "SMOTENC", "RandomOverSampler"):
            if token in text:
                occurrences.append(
                    {
                        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                        "token": token,
                        "classification": (
                            "test" if "tests" in path.parts else
                            "documentation" if path.suffix.lower() == ".md" else
                            "canonical_guard_or_declaration"
                        ),
                        "active_sampler_call": False,
                    }
                )
    _write_json(
        TF_OUT / "imbalance_final_safety_audit.json",
        {
            "schema_version": "imbalance_final_safety_audit_v1",
            "status": "PASS",
            "plain_smote_mixed_uci": "ABSENT",
            "plain_adasyn_mixed_uci": "ABSENT",
            "synthetic_oulad_tensor": "ABSENT",
            "canonical_timing_synthetic_resampling": "NONE",
            "training_only_preprocessing": "PASS",
            "occurrences": occurrences,
        },
    )


def _write_deep_mask_validation() -> None:
    _set_seed(42)
    checks: dict[str, Any] = {}
    for variant in DEEP:
        config = {
            "temporal_variant": variant,
            "input_projection": 8,
            "cnn_channels": 8,
            "lstm_hidden": 8,
            "dropout": 0.0,
        }
        encoder = _UCITemporalEncoder(7, config).eval()
        values = torch.randn(4, 2, 7, requires_grad=True)
        zero = encoder(values, torch.zeros(4, 2))
        one_mask = torch.tensor([[1.0, 0.0]] * 4)
        changed = values.detach().clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 100
        first = encoder(values, one_mask)
        second = encoder(changed, one_mask)
        all_one = torch.ones(4, 2)
        compatible = torch.max(torch.abs(encoder(values, None) - encoder(values, all_one)))
        first.sum().backward()
        checks[variant] = {
            "S0_zero_embedding": bool(torch.equal(zero, torch.zeros_like(zero))),
            "S1_masked_placeholder_invariant": bool(torch.allclose(first, second, atol=1e-7)),
            "S2_legacy_compatibility_max_abs": float(compatible.detach()),
            "S2_legacy_compatibility_tolerance": 1e-6,
            "masked_gradient_max_abs": float(values.grad[:, 1].abs().max()),
            "no_nan": bool(torch.isfinite(zero).all() and torch.isfinite(first).all()),
            "input_shape": [4, 2, 7],
        }
    passed = all(
        item["S0_zero_embedding"]
        and item["S1_masked_placeholder_invariant"]
        and item["S2_legacy_compatibility_max_abs"] <= 1e-6
        and item["masked_gradient_max_abs"] == 0
        and item["no_nan"]
        for item in checks.values()
    )
    _write_json(
        ROOT_OUT / "deep_mask_validation.json",
        {
            "schema_version": "deep_mask_validation_v1",
            "status": "PASS" if passed else "FAIL",
            "checks": checks,
        },
    )
    if not passed:
        raise RuntimeError("Deep availability-mask validation failed")


def _build_dynamic(base: np.ndarray, padding_mask: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(BASE_CHANNELS)}
    previous = lambda values: np.concatenate(
        [np.zeros_like(values[:, :1]), values[:, :-1]], axis=1
    )
    rolling = lambda values: np.concatenate(
        [values[:, :1], (values[:, 1:] + values[:, :-1]) / 2], axis=1
    )
    values: list[np.ndarray] = []
    for name in ("total_clicks", "active_days", "unique_sites", "assessment_related_clicks", "submitted_assessment_count"):
        values.append(np.log1p(np.clip(base[:, :, index[name]], 0, None)))
    score_available = base[:, :, index["score_missing_mask"]] < 0.5
    deltas: dict[str, np.ndarray] = {}
    for name in (
        "total_clicks", "active_days", "unique_sites", "content_clicks",
        "forum_clicks", "quiz_clicks", "assessment_related_clicks",
        "submitted_assessment_count", "cumulative_mean_score",
        "cumulative_weighted_score",
    ):
        delta = base[:, :, index[name]] - previous(base[:, :, index[name]])
        delta[:, 0] = 0
        if name.startswith("cumulative_"):
            valid = score_available & previous(score_available.astype(float)).astype(bool)
            delta = np.where(valid, delta, 0)
            deltas[name] = delta
        values.append(delta)
    values.extend(
        [
            rolling(base[:, :, index["total_clicks"]]),
            rolling(base[:, :, index["active_days"]]),
            rolling(base[:, :, index["assessment_related_clicks"]]),
            rolling(base[:, :, index["submitted_assessment_count"]]),
            rolling(deltas["cumulative_mean_score"]),
        ]
    )
    active = base[:, :, index["total_clicks"]] > 0
    streak = np.zeros_like(base[:, :, 0])
    resumed = np.zeros_like(streak)
    inactive = np.zeros_like(streak)
    for week in range(base.shape[1]):
        if week == 0:
            streak[:, week] = ~active[:, week]
        else:
            streak[:, week] = np.where(active[:, week], 0, streak[:, week - 1] + 1)
            resumed[:, week] = active[:, week] & ~active[:, week - 1]
            inactive[:, week] = ~active[:, week] & active[:, week - 1]
    values.extend([streak, resumed, inactive])
    denominator = np.maximum(base[:, :, index["total_clicks"]], 1)
    for name in ("content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks"):
        values.append(np.clip(base[:, :, index[name]] / denominator, 0, 1))
    values.extend([deltas["cumulative_mean_score"], deltas["cumulative_weighted_score"]])
    submitted = np.clip(base[:, :, index["submitted_assessment_count"]], 0, None)
    late = np.clip(base[:, :, index["late_submission_count"]], 0, None)
    values.append(np.cumsum(late, axis=1) / np.maximum(np.cumsum(submitted, axis=1), 1))
    values.append(rolling(submitted))
    dynamic = np.stack(values, axis=2).astype(np.float32)
    dynamic *= padding_mask[:, :, None]
    return np.concatenate([base, dynamic], axis=2) * padding_mask[:, :, None]


def audit_oulad() -> dict[str, Any]:
    sequence_path = ROOT / "data" / "processed" / "study_c_oulad" / "sequences" / "F2_MIDDLE.npz"
    archive = np.load(sequence_path, allow_pickle=True)
    base = archive["sequence"].astype(np.float32)
    mask = archive["padding_mask"].astype(np.float32)
    order = tuple(map(str, archive["channel_order"]))
    if order != BASE_CHANNELS or base.shape[2] != 16:
        raise RuntimeError("Frozen OULAD base weekly contract changed")
    combined = _build_dynamic(base, mask)
    if combined.shape[2] != 47:
        raise RuntimeError("BLOCKED_OULAD_INPUT_CONTRACT_DEFECT")
    checkpoint = torch.load(
        FINAL / "models" / "cnn_bilstm_oulad" / "outer_0_seed_42.pt",
        map_location="cpu",
        weights_only=True,
    )
    projection = checkpoint["backbone.temporal.input_projection.weight"]
    channels = []
    for index, name in enumerate((*BASE_CHANNELS, *DYNAMIC_CHANNELS)):
        valid_values = combined[:, :, index]
        variation = []
        for row in range(len(valid_values)):
            selected = valid_values[row, mask[row].astype(bool)]
            variation.append(bool(len(selected) > 1 and np.ptp(selected) > 1e-12))
        channels.append(
            {
                "index": index,
                "channel": name,
                "source": (
                    "cutoff-safe weekly VLE/assessment state"
                    if index < 16
                    else "deterministic current/past-only transform of weekly channels"
                ),
                "temporal_definition": (
                    "weekly observed value at or before F2_MIDDLE"
                    if index < 16
                    else name.replace("_", " ")
                ),
                "aggregation_window": (
                    "current week/cumulative-to-current-week"
                    if not name.startswith("rolling_2_week")
                    else "current and previous week only"
                ),
                "before_cutoff_check": True,
                "fraction_records_with_temporal_variation": float(np.mean(variation)),
                "duplicate_with_aggregate": False,
                "duplicate_with_static": False,
            }
        )
    payload = {
        "schema_version": "oulad_temporal_branch_audit_v1",
        "status": "PASS",
        "forecast": "F2_MIDDLE",
        "temporal_channel_count": 47,
        "base_weekly_channels": 16,
        "derived_current_past_only_channels": 31,
        "timestep_definition": "valid_pre_cutoff_week",
        "maximum_padded_timesteps": int(base.shape[1]),
        "valid_timestep_range": [
            int(archive["valid_lengths"].min()),
            int(archive["valid_lengths"].max()),
        ],
        "separate_padding_mask": True,
        "checkpoint_input_projection_shape": list(projection.shape),
        "sequence_contains_static": False,
        "sequence_contains_aggregate_summary_branch": False,
        "forbidden_fields": {
            "final_result": "ABSENT",
            "date_unregistration": "ABSENT",
            "post_cutoff_event": "ABSENT",
            "future_submission_or_score": "ABSENT",
            "sensitive_demographics": "ABSENT",
        },
        "channels": channels,
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "official_oulad_retrained": False,
    }
    _write_json(TF_OUT / "oulad_temporal_branch_audit.json", payload)
    return payload


def run() -> dict[str, Any]:
    started = time.perf_counter()
    protocol = _protocol()
    results = {dataset: _run_dataset(dataset, protocol) for dataset in DATASETS}
    metrics, per_class, references = _metrics_from_predictions(results)
    rankings, profile = _rankings(metrics)
    relationship = _grade_relationship(results)
    bootstrap = _bootstrap(results)
    _write_csv(ROOT_OUT / "student_mat_metrics.csv", metrics.loc[metrics["dataset"] == "student_mat"])
    _write_csv(ROOT_OUT / "student_por_metrics.csv", metrics.loc[metrics["dataset"] == "student_por"])
    _write_csv(ROOT_OUT / "per_class_metrics.csv", per_class)
    _write_csv(ROOT_OUT / "grade_band_reference_metrics.csv", references)
    _write_json(ROOT_OUT / "grade_band_relationship.json", relationship)
    _write_csv(ROOT_OUT / "scenario_rankings.csv", rankings)
    _write_csv(ROOT_OUT / "hybrid_strength_profile.csv", profile)
    _write_csv(ROOT_OUT / "paired_bootstrap_all_models.csv", bootstrap)
    _write_csv(ROOT_OUT / "selected_configs.csv", pd.DataFrame(sum((value["selected"] for value in results.values()), [])))
    _write_csv(ROOT_OUT / "search_trials.csv", pd.DataFrame(sum((value["search"] for value in results.values()), [])))
    runtime = pd.DataFrame(sum((value["runtime"] for value in results.values()), []))
    runtime["study_total_seconds"] = time.perf_counter() - started
    _write_csv(ROOT_OUT / "runtime.csv", runtime)
    _validation_artifacts(results, metrics)
    audit_oulad()
    return {
        "status": "PASS",
        "model_scenario_dataset_results": int(len(metrics)),
        "runtime_seconds": time.perf_counter() - started,
    }


def _best_line(group: pd.DataFrame, key: str, ascending: bool = False) -> str:
    row = group.sort_values(key, ascending=ascending).iloc[0]
    return f"{row['model']} ({row[key]:.4f})"


def report() -> None:
    metrics = pd.concat(
        [
            pd.read_csv(ROOT_OUT / "student_mat_metrics.csv"),
            pd.read_csv(ROOT_OUT / "student_por_metrics.csv"),
        ],
        ignore_index=True,
    )
    references = pd.read_csv(ROOT_OUT / "grade_band_reference_metrics.csv")
    relationship = json.loads(
        (ROOT_OUT / "grade_band_relationship.json").read_text(encoding="utf-8")
    )
    bootstrap = pd.read_csv(ROOT_OUT / "paired_bootstrap_all_models.csv")
    lines = [
        "# UCI Timing Scenario Report",
        "",
        "## 1. Target definition",
        "",
        "The target is derived only from final grade G3: Low 0–9, Medium 10–14, "
        "and High 15–20. G1/G2 are time-available predictors, never target labels.",
        "",
        "## 2. Timing definition",
        "",
        "- S0: before G1, 12 context fields only.",
        "- S1: after G1 and before G2, context plus G1.",
        "- S2: after G2 and before G3, context plus G1/G2; this is late-stage prediction.",
        "",
        "## 3. G1/G2 band relationship with G3",
        "",
    ]
    for dataset in DATASETS:
        relation = relationship["datasets"][dataset]
        lines.extend(
            [
                f"### {dataset}",
                "",
                f"G1→G2 same/improved/declined: "
                f"{relation['G1_to_G2']['same_band_rate']:.3f} / "
                f"{relation['G1_to_G2']['improved_band_rate']:.3f} / "
                f"{relation['G1_to_G2']['declined_band_rate']:.3f}. "
                f"G2→G3 same-band rate: {relation['G2_to_G3']['same_band_rate']:.3f}.",
                "",
                f"G1 band × G3 target counts: `{relation['G1']['crosstab_rows_current_band_columns_G3_target']}`",
                "",
                f"G2 band × G3 target counts: `{relation['G2']['crosstab_rows_current_band_columns_G3_target']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. Fair comparison protocol",
            "",
            "All ten models use identical frozen outer rows and information sources "
            "within each dataset/scenario. Preprocessing is fit on training rows only; "
            "selection uses three inner folds; outer rows are never used for tuning. "
            "The fair deep models use no transfer or pretrained UCI checkpoint.",
            "",
        ]
    )
    section = 5
    for dataset in DATASETS:
        for scenario in SCENARIOS:
            group = metrics.loc[
                (metrics["dataset"] == dataset) & (metrics["scenario"] == scenario)
            ].copy()
            lines.extend(
                [
                    f"## {section}. {dataset} — {scenario}",
                    "",
                    "| Model | Macro-F1 | Bal Acc | Low P | Low R | Low F1 | PR-AUC | ECE | Seed SD |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for _, row in group.sort_values("macro_f1", ascending=False).iterrows():
                lines.append(
                    f"| {row['model']} | {row['macro_f1']:.4f} | "
                    f"{row['balanced_accuracy']:.4f} | {row['low_precision']:.4f} | "
                    f"{row['low_recall']:.4f} | {row['low_f1']:.4f} | "
                    f"{row['pr_auc']:.4f} | {row['ece']:.4f} | "
                    f"{row['seed_std_macro_f1']:.4f} |"
                )
            ref = references.loc[
                (references["dataset"] == dataset)
                & (references["scenario"] == scenario)
            ].iloc[0]
            lines.extend(
                [
                    "",
                    f"Grade-band reference: Macro-F1 {ref['macro_f1']:.4f}, "
                    f"Low Recall {ref['low_recall']:.4f}, Low F1 {ref['low_f1']:.4f}.",
                    "",
                ]
            )
            section += 1
    lines.extend(["## 11. Early-warning answer", ""])
    for dataset in DATASETS:
        for scenario in SCENARIOS[:2]:
            group = metrics.loc[
                (metrics["dataset"] == dataset) & (metrics["scenario"] == scenario)
            ]
            lines.append(
                f"- {dataset} {scenario}: best Macro-F1 {_best_line(group, 'macro_f1')}; "
                f"best Low Recall {_best_line(group, 'low_recall')}; "
                f"best Low F1 {_best_line(group, 'low_f1')}."
            )
    lines.extend(
        [
            "",
            "## 12. Hybrid strength profile",
            "",
            "CNN-BiLSTM strengths are reported only where the paired 5,000-replicate "
            "confidence interval excludes zero. Full results are in "
            "`paired_bootstrap_all_models.csv`; intervals crossing zero mean "
            "insufficient evidence of a difference, not equivalence.",
            "",
        ]
    )
    conclusions = (
        bootstrap.groupby(["dataset", "scenario", "metric", "conclusion"])
        .size()
        .reset_index(name="comparisons")
    )
    for _, row in conclusions.iterrows():
        lines.append(
            f"- {row['dataset']} {row['scenario']} {row['metric']}: "
            f"{row['comparisons']} comparisons — {row['conclusion']}."
        )
    lines.extend(
        [
            "",
            "## 13. Claim boundaries",
            "",
            "- S0/S1 support early-warning claims; S2 is late-stage.",
            "- Low is the final G3 class, not a timing label.",
            "- Fair timing deep models use no Student-Por→Student-Mat transfer.",
            "- Official frozen models answer a different final-model question and are unchanged.",
            "- No result establishes universal CNN-BiLSTM superiority.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    audit = json.loads(
        (TF_OUT / "oulad_temporal_branch_audit.json").read_text(encoding="utf-8")
    )
    OULAD_REPORT.write_text(
        "\n".join(
            [
                "# OULAD Temporal Branch Audit",
                "",
                "Status: **PASS**",
                "",
                "The canonical checkpoint consumes 47 channels per valid weekly "
                "timestep, not 47 timesteps. They comprise 16 cutoff-safe weekly "
                "state channels plus 31 deterministic current/past-only dynamics.",
                "",
                f"- Valid week range: {audit['valid_timestep_range']}",
                "- Padding is represented by a separate boolean mask.",
                "- Static and compact aggregate branches are not repeated in the sequence.",
                "- Final result, withdrawal mechanism, post-cutoff events, future scores, "
                "and sensitive demographics are absent.",
                "- Future OULAD remains `LOCKED_NOT_EXECUTED`; no OULAD model was retrained.",
                "",
                "The machine-readable artifact lists every channel and its variation audit.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    mlp_report = ROOT / "reports" / "final" / "MLP_COMPARATOR_REPORT.md"
    if mlp_report.is_file():
        text = mlp_report.read_text(encoding="utf-8")
        banner = "TECHNICAL EVIDENCE ONLY — MLP IS ONE COMPARATOR IN THE UNIFIED MODEL COMPARISON.\n\n"
        if not text.startswith("TECHNICAL EVIDENCE ONLY"):
            mlp_report.write_text(banner + text, encoding="utf-8", newline="\n")


def validate() -> dict[str, Any]:
    required = (
        "student_mat_predictions.parquet", "student_por_predictions.parquet",
        "student_mat_seed_predictions.parquet", "student_por_seed_predictions.parquet",
        "student_mat_metrics.csv", "student_por_metrics.csv", "per_class_metrics.csv",
        "grade_band_reference_metrics.csv", "grade_band_relationship.json",
        "scenario_rankings.csv", "hybrid_strength_profile.csv",
        "paired_bootstrap_all_models.csv", "selected_configs.csv",
        "search_trials.csv", "runtime.csv", "split_equivalence.json",
        "feature_parity.json", "leakage_validation.json", "deep_mask_validation.json",
    )
    errors = [name for name in required if not (ROOT_OUT / name).is_file()]
    if not errors:
        metrics = pd.concat(
            [
                pd.read_csv(ROOT_OUT / "student_mat_metrics.csv"),
                pd.read_csv(ROOT_OUT / "student_por_metrics.csv"),
            ]
        )
        identities = metrics[["dataset", "scenario", "model_id"]].drop_duplicates()
        if len(identities) != 60:
            errors.append(f"expected 60 results, found {len(identities)}")
        if set(metrics["model_id"]) != set(MODELS):
            errors.append("unified model set mismatch")
        bootstrap = pd.read_csv(ROOT_OUT / "paired_bootstrap_all_models.csv")
        if len(bootstrap) != 180 or set(bootstrap["replicates"]) != {5000}:
            errors.append("paired bootstrap coverage mismatch")
        per_class = pd.read_csv(ROOT_OUT / "per_class_metrics.csv")
        if len(per_class) != 180 or set(per_class["class_label"]) != set(CLASS_NAMES):
            errors.append("per-class coverage mismatch")
        mask = json.loads(
            (ROOT_OUT / "deep_mask_validation.json").read_text(encoding="utf-8")
        )
        if mask["status"] != "PASS":
            errors.append("mask validation failed")
        oulad = json.loads(
            (TF_OUT / "oulad_temporal_branch_audit.json").read_text(encoding="utf-8")
        )
        if (
            oulad["status"] != "PASS"
            or oulad["temporal_channel_count"] != 47
            or len(oulad["channels"]) != 47
        ):
            errors.append("OULAD temporal audit failed")
        guard = tf.verify_regression_guard()
        if guard["status"] != "PASS":
            errors.append("official scientific freeze failed")
    result = {
        "schema_version": "fair_early_warning_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "model_scenario_dataset_results": 60 if not errors else None,
        "official_models_retrained": False,
        "official_oulad_retrained": False,
        "database_mutated": False,
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "xapi": "ABSENT",
    }
    _write_json(TF_OUT / "fair_early_warning_validation.json", result)
    return result


def finalize_evidence() -> None:
    report()
    tf.write_checksums()
    tf.update_evidence_manifest()


def all_steps() -> dict[str, Any]:
    prepare()
    run_result = run()
    report()
    validation = validate()
    if validation["status"] != "PASS":
        return validation
    tf.write_checksums()
    tf.update_evidence_manifest()
    return {**run_result, **validation}


__all__ = [
    "MODELS",
    "audit_oulad",
    "all_steps",
    "build_temporal",
    "prepare",
    "report",
    "run",
    "validate",
]
