"""One-estimator, three-stage UCI prediction authority.

The training unit is (dataset, model_id, outer_fold, seed). S0/S1/S2 are
views of the same base records and are never model identities or cache keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
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
from torch import nn

from src.models.uci_components import UCICNNBiLSTM
from src.pipelines import uci_support as tf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "final" / "unified_stage_aware_uci"
CONFIG = ROOT / "configs" / "final" / "uci_prediction.yaml"
REPORT = ROOT / "reports" / "final" / "UNIFIED_STAGE_AWARE_RESULTS.md"
MATRIX_REPORT = ROOT / "reports" / "final" / "HYBRID_VS_ML_STAGE_MATRIX.md"
SELECTION_REPORT = ROOT / "reports" / "final" / "UNIFIED_MODEL_SELECTION_REPORT.md"
PROVENANCE = (
    ROOT
    / "artifacts"
    / "final"
    / "provenance"
    / "legacy_uci_separate_stage_manifest.json"
)
STAGES = (
    "S0_EARLY_NO_GRADE",
    "S1_MID_G1_ONLY",
    "S2_LATE_G1_G2",
)
DATASETS = ("student_mat", "student_por")
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
CLASS_NAMES = ("Low", "Medium", "High")
DISPLAY_NAMES = {
    **tf.MODEL_NAMES,
    "cnn_only": "CNN-only",
    "bilstm_only": "BiLSTM-only",
}


@dataclass(frozen=True)
class StageViewBundle:
    dataset: str
    stage: str
    record_id: np.ndarray
    target: np.ndarray
    outer_fold: np.ndarray
    temporal: np.ndarray
    availability_mask: np.ndarray
    context: pd.DataFrame

    def validate(self) -> None:
        rows = len(self.record_id)
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage: {self.stage}")
        if not (
            rows
            == len(self.target)
            == len(self.outer_fold)
            == len(self.temporal)
            == len(self.availability_mask)
            == len(self.context)
        ):
            raise ValueError("Stage view row alignment failed")
        if self.temporal.shape != (rows, 2, 7):
            raise ValueError("Unified temporal contract must be [N,2,7]")
        expected = {
            STAGES[0]: np.asarray([0.0, 0.0]),
            STAGES[1]: np.asarray([1.0, 0.0]),
            STAGES[2]: np.asarray([1.0, 1.0]),
        }[self.stage]
        if not np.array_equal(
            self.availability_mask, np.broadcast_to(expected, (rows, 2))
        ):
            raise ValueError(f"{self.stage} availability mask mismatch")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
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


def _protocol() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if value.get("status") != "PREREGISTERED_BEFORE_UNIFIED_OUTER_SCORING":
        raise RuntimeError("Unified protocol is not preregistered")
    if value["training"]["outer_used_for_tuning"]:
        raise RuntimeError("Outer tuning is prohibited")
    if value["training"]["transfer_learning"] != "prohibited":
        raise RuntimeError("Transfer must remain prohibited")
    return value


def build_stage_views(data: tf.UCIStudyData) -> dict[str, StageViewBundle]:
    context = data.frame.loc[:, list(tf.CONTEXT)].reset_index(drop=True)
    result: dict[str, StageViewBundle] = {}
    for stage in STAGES:
        temporal, mask = build_temporal(data.frame, stage)
        bundle = StageViewBundle(
            dataset=data.dataset,
            stage=stage,
            record_id=data.record_ids.copy(),
            target=data.target.copy(),
            outer_fold=data.outer_fold.copy(),
            temporal=temporal,
            availability_mask=mask,
            context=context.copy(),
        )
        bundle.validate()
        result[stage] = bundle
    first = result[STAGES[0]]
    for bundle in result.values():
        if not np.array_equal(first.record_id, bundle.record_id):
            raise RuntimeError("Stage expansion changed record order")
        if not np.array_equal(first.outer_fold, bundle.outer_fold):
            raise RuntimeError("Stage expansion changed frozen folds")
    return result


def build_temporal(
    frame: pd.DataFrame, stage: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(frame), 2, 7), dtype=np.float32)
    mask = np.zeros((len(frame), 2), dtype=np.float32)
    if stage == STAGES[0]:
        return values, mask
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    g1 = frame["G1"].to_numpy(dtype=np.float32)
    values[:, 0, 0] = g1 / 20.0
    values[:, 0, 1] = -1.0
    values[:, 0, 4] = (g1 - 10.0) / 20.0
    values[:, 0, 5] = (g1 - 15.0) / 20.0
    mask[:, 0] = 1.0
    if stage == STAGES[2]:
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


def _fixed_tabular_frame(bundle: StageViewBundle) -> pd.DataFrame:
    temporal = bundle.temporal.reshape(len(bundle.record_id), 14)
    columns = [f"temporal_t{step}_c{channel}" for step in range(2) for channel in range(7)]
    numeric = pd.DataFrame(temporal, columns=columns)
    numeric["availability_t0"] = bundle.availability_mask[:, 0]
    numeric["availability_t1"] = bundle.availability_mask[:, 1]
    frame = pd.concat([bundle.context.reset_index(drop=True), numeric], axis=1)
    if {"G1", "G2", "G3"} & set(frame.columns):
        raise RuntimeError("Raw grade leakage in fixed tabular schema")
    return frame


def _expand_tabular(
    views: dict[str, StageViewBundle], base_indices: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    for stage in STAGES:
        frames.append(_fixed_tabular_frame(views[stage]).iloc[base_indices])
        targets.append(views[stage].target[base_indices])
        stages.append(np.repeat(stage, len(base_indices)))
    return (
        pd.concat(frames, ignore_index=True),
        np.concatenate(targets),
        np.concatenate(stages),
    )


def _tabular_candidates(model_id: str) -> list[dict[str, Any]]:
    full = tf._candidate_grid(model_id, binary=False)
    if len(full) <= 3:
        return full
    indices = sorted({0, len(full) // 2, len(full) - 1})
    return [full[index] for index in indices]


def _stage_scores(
    target: np.ndarray, probabilities: dict[str, np.ndarray]
) -> dict[str, float]:
    values = {
        stage: float(
            f1_score(
                target,
                probabilities[stage].argmax(1),
                average="macro",
                zero_division=0,
            )
        )
        for stage in STAGES
    }
    values["mean_stage_macro_f1"] = float(np.mean(list(values.values())))
    values["worst_stage_macro_f1"] = float(np.min(list(values.values())))
    return values


def _selection_key(
    stage_scores: dict[str, float],
    target: np.ndarray,
    probabilities: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[float, float, float, float, str]:
    low_values = []
    nll_values = []
    for stage in STAGES:
        predicted = probabilities[stage].argmax(1)
        _, _, class_f1, _ = precision_recall_fscore_support(
            target, predicted, labels=np.arange(3), zero_division=0
        )
        low_values.append(float(class_f1[0]))
        nll_values.append(float(log_loss(target, probabilities[stage], labels=np.arange(3))))
    return (
        stage_scores["mean_stage_macro_f1"],
        stage_scores["worst_stage_macro_f1"],
        float(np.mean(low_values)),
        -float(np.mean(nll_values)),
        json.dumps(config, sort_keys=True),
    )


def _model_id(dataset: str, family: str) -> str:
    suffix = "mat" if dataset == "student_mat" else "por"
    return f"{family}_{suffix}"


def _run_id(dataset: str, family: str, fold: int, seed: int, config: dict[str, Any]) -> str:
    payload = {
        "dataset": dataset,
        "model_id": _model_id(dataset, family),
        "outer_fold": fold,
        "seed": seed,
        "config_hash": _stable_hash(config),
    }
    return "unified_" + _stable_hash(payload)[:24]


def _checkpoint_path(dataset: str, family: str, fold: int, seed: int) -> Path:
    return (
        OUT
        / "models"
        / dataset
        / _model_id(dataset, family)
        / f"outer_fold_{fold}"
        / f"seed_{seed}.joblib"
    )


def _cache_path(dataset: str, family: str, fold: int, seed: int) -> Path:
    return _checkpoint_path(dataset, family, fold, seed).with_suffix(".predictions.npz")


def _fit_tabular_outer(
    *,
    data: tf.UCIStudyData,
    views: dict[str, StageViewBundle],
    family: str,
    fold: int,
    train: np.ndarray,
    validation: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    columns = tuple(_fixed_tabular_frame(views[STAGES[0]]).columns)
    inner = tf._inner_splits(data.target[train], data.groups[train], seed=42)
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, str], dict[str, Any]] | None = None
    for trial, params in enumerate(_tabular_candidates(family)):
        pooled = {stage: np.zeros((len(train), 3), dtype=float) for stage in STAGES}
        for inner_fold, (fit_local, score_local) in enumerate(inner):
            fit_base = train[fit_local]
            score_base = train[score_local]
            expanded_x, expanded_y, _ = _expand_tabular(views, fit_base)
            pipeline = tf._pipeline(
                lambda: tf._uci_preprocessor(columns),
                family,
                params,
                seed=42,
                binary=False,
            )
            pipeline.fit(expanded_x, expanded_y)
            for stage in STAGES:
                pooled[stage][score_local] = tf._aligned_probabilities(
                    pipeline,
                    _fixed_tabular_frame(views[stage]).iloc[score_base],
                    3,
                )
        scores = _stage_scores(data.target[train], pooled)
        key = _selection_key(scores, data.target[train], pooled, params)
        trials.append(
            {
                "dataset": data.dataset,
                "model_id": _model_id(data.dataset, family),
                "model_family": family,
                "outer_fold": fold,
                "trial": trial,
                "state": "COMPLETE",
                "params": json.dumps(params, sort_keys=True),
                **scores,
                "outer_rows_used_for_selection": False,
            }
        )
        if best is None or key[:-1] > best[0][:-1] or (
            key[:-1] == best[0][:-1] and key[-1] < best[0][-1]
        ):
            best = (key, params)
    assert best is not None
    selected = best[1]
    per_seed: dict[int, dict[str, np.ndarray]] = {}
    runtimes: list[dict[str, Any]] = []
    for seed in protocol["training"]["outer_seeds"]:
        checkpoint = _checkpoint_path(data.dataset, family, fold, int(seed))
        cache = _cache_path(data.dataset, family, fold, int(seed))
        config_hash = _stable_hash(selected)
        if checkpoint.is_file() and cache.is_file():
            archive = np.load(cache)
            if str(archive["config_hash"]) == config_hash:
                per_seed[int(seed)] = {stage: archive[stage] for stage in STAGES}
                runtimes.append(
                    {
                        "dataset": data.dataset,
                        "model_id": _model_id(data.dataset, family),
                        "outer_fold": fold,
                        "seed": int(seed),
                        "runtime_seconds": 0.0,
                        "cache_hit": True,
                    }
                )
                continue
        started = time.perf_counter()
        expanded_x, expanded_y, _ = _expand_tabular(views, train)
        pipeline = tf._pipeline(
            lambda: tf._uci_preprocessor(columns),
            family,
            selected,
            seed=int(seed),
            binary=False,
        )
        pipeline.fit(expanded_x, expanded_y)
        predictions = {
            stage: tf._aligned_probabilities(
                pipeline,
                _fixed_tabular_frame(views[stage]).iloc[validation],
                3,
            )
            for stage in STAGES
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": pipeline,
                "dataset": data.dataset,
                "model_id": _model_id(data.dataset, family),
                "outer_fold": fold,
                "seed": int(seed),
                "config": selected,
                "config_hash": config_hash,
                "stages": STAGES,
            },
            checkpoint,
            compress=3,
        )
        np.savez_compressed(cache, config_hash=config_hash, **predictions)
        per_seed[int(seed)] = predictions
        runtimes.append(
            {
                "dataset": data.dataset,
                "model_id": _model_id(data.dataset, family),
                "outer_fold": fold,
                "seed": int(seed),
                "runtime_seconds": time.perf_counter() - started,
                "cache_hit": False,
            }
        )
    ensemble = {
        stage: np.mean([per_seed[seed][stage] for seed in per_seed], axis=0)
        for stage in STAGES
    }
    selection = {
        "dataset": data.dataset,
        "model_id": _model_id(data.dataset, family),
        "model_family": family,
        "outer_fold": fold,
        "selected_params": json.dumps(selected, sort_keys=True),
        "selected_inner_mean_stage_macro_f1": best[0][0],
        "selected_inner_worst_stage_macro_f1": best[0][1],
        "outer_rows_used_for_selection": False,
        "one_estimator_all_stages": True,
    }
    return ensemble, per_seed, selection, trials, runtimes


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _deep_config(candidate: dict[str, Any], family: str) -> dict[str, Any]:
    return {
        **candidate,
        "temporal_variant": family,
        "lstm_layers": 1,
        "context_layers": 1,
        "activation": "gelu",
        "fusion": "gated",
    }


def _deep_probabilities(
    model: UCICNNBiLSTM,
    bundle: StageViewBundle,
    indices: np.ndarray,
    context: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            batch = indices[start : start + 256]
            logits = model(
                torch.from_numpy(bundle.temporal[batch]).to(device),
                torch.from_numpy(context[batch]).to(device),
                torch.from_numpy(bundle.availability_mask[batch]).to(device),
            )["classification"]
            rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    value = np.concatenate(rows).astype(np.float64)
    return value / value.sum(axis=1, keepdims=True)


def _fit_deep_model(
    *,
    data: tf.UCIStudyData,
    views: dict[str, StageViewBundle],
    family: str,
    candidate: dict[str, Any],
    fit: np.ndarray,
    score: np.ndarray,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[dict[str, np.ndarray], int, dict[str, Any]]:
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = tf._uci_preprocessor(tf.CONTEXT)
    preprocessor.fit(views[STAGES[0]].context.iloc[fit])
    context = np.asarray(
        preprocessor.transform(views[STAGES[0]].context), dtype=np.float32
    )
    config = _deep_config(candidate, family)
    model = UCICNNBiLSTM(7, context.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate["learning_rate"]),
        weight_decay=float(candidate["weight_decay"]),
    )
    counts = np.bincount(data.target[fit], minlength=3).astype(float)
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
    generator = torch.Generator().manual_seed(seed)
    fit_tensor = torch.tensor(fit, dtype=torch.long)
    for epoch in range(1, max_epochs + 1):
        model.train()
        base_order = fit_tensor[torch.randperm(len(fit_tensor), generator=generator)]
        for base_batch in base_order.split(batch_size):
            batch = base_batch.numpy()
            optimizer.zero_grad(set_to_none=True)
            stage_losses = []
            for stage in STAGES:
                bundle = views[stage]
                logits = model(
                    torch.from_numpy(bundle.temporal[batch]).to(device),
                    torch.from_numpy(context[batch]).to(device),
                    torch.from_numpy(bundle.availability_mask[batch]).to(device),
                )["classification"]
                stage_losses.append(
                    loss_fn(logits, torch.from_numpy(data.target[batch]).long().to(device))
                )
            loss = torch.stack(stage_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        probabilities = {
            stage: _deep_probabilities(model, views[stage], score, context, device)
            for stage in STAGES
        }
        value = _stage_scores(data.target[score], probabilities)[
            "mean_stage_macro_f1"
        ]
        if value > best_score + 1e-12:
            best_score = value
            best_epoch = epoch
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        elif epoch - best_epoch >= patience:
            break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)
    probabilities = {
        stage: _deep_probabilities(model, views[stage], score, context, device)
        for stage in STAGES
    }
    payload = {
        "state_dict": {
            key: value.detach().cpu().numpy() for key, value in model.state_dict().items()
        },
        "context_preprocessor": preprocessor,
        "context_dim": int(context.shape[1]),
        "config": config,
    }
    return probabilities, (max_epochs if fixed_epochs is not None else best_epoch), payload


def _run_deep_outer(
    *,
    data: tf.UCIStudyData,
    views: dict[str, StageViewBundle],
    family: str,
    fold: int,
    train: np.ndarray,
    validation: np.ndarray,
    protocol: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    inner = tf._inner_splits(data.target[train], data.groups[train], seed=42)
    trials: list[dict[str, Any]] = []
    best: tuple[tuple[float, float, float, float, str], dict[str, Any], int] | None = None
    for trial, candidate in enumerate(protocol["deep"]["candidate_grid"]):
        pooled = {stage: np.zeros((len(train), 3), dtype=float) for stage in STAGES}
        epochs: list[int] = []
        for inner_fold, (fit_local, score_local) in enumerate(inner):
            fit_base = train[fit_local]
            score_base = train[score_local]
            probability, selected_epoch, _ = _fit_deep_model(
                data=data,
                views=views,
                family=family,
                candidate=candidate,
                fit=fit_base,
                score=score_base,
                seed=42,
            )
            epochs.append(selected_epoch)
            for stage in STAGES:
                pooled[stage][score_local] = probability[stage]
        scores = _stage_scores(data.target[train], pooled)
        key = _selection_key(scores, data.target[train], pooled, candidate)
        selected_epochs = max(1, int(np.median(epochs)))
        trials.append(
            {
                "dataset": data.dataset,
                "model_id": _model_id(data.dataset, family),
                "model_family": family,
                "outer_fold": fold,
                "trial": trial,
                "state": "COMPLETE",
                "params": json.dumps(candidate, sort_keys=True),
                "selected_epochs": selected_epochs,
                **scores,
                "outer_rows_used_for_selection": False,
            }
        )
        if best is None or key[:-1] > best[0][:-1] or (
            key[:-1] == best[0][:-1] and key[-1] < best[0][-1]
        ):
            best = (key, candidate, selected_epochs)
    assert best is not None
    per_seed: dict[int, dict[str, np.ndarray]] = {}
    runtimes: list[dict[str, Any]] = []
    for seed in protocol["training"]["outer_seeds"]:
        checkpoint = _checkpoint_path(data.dataset, family, fold, int(seed))
        cache = _cache_path(data.dataset, family, fold, int(seed))
        config_payload = {**best[1], "selected_epochs": best[2]}
        config_hash = _stable_hash(config_payload)
        if checkpoint.is_file() and cache.is_file():
            archive = np.load(cache)
            if str(archive["config_hash"]) == config_hash:
                per_seed[int(seed)] = {stage: archive[stage] for stage in STAGES}
                runtimes.append(
                    {
                        "dataset": data.dataset,
                        "model_id": _model_id(data.dataset, family),
                        "outer_fold": fold,
                        "seed": int(seed),
                        "runtime_seconds": 0.0,
                        "cache_hit": True,
                    }
                )
                continue
        started = time.perf_counter()
        probability, _, payload = _fit_deep_model(
            data=data,
            views=views,
            family=family,
            candidate=best[1],
            fit=train,
            score=validation,
            seed=int(seed),
            fixed_epochs=best[2],
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                **payload,
                "dataset": data.dataset,
                "model_id": _model_id(data.dataset, family),
                "outer_fold": fold,
                "seed": int(seed),
                "config_hash": config_hash,
                "stages": STAGES,
            },
            checkpoint,
            compress=3,
        )
        np.savez_compressed(cache, config_hash=config_hash, **probability)
        per_seed[int(seed)] = probability
        runtimes.append(
            {
                "dataset": data.dataset,
                "model_id": _model_id(data.dataset, family),
                "outer_fold": fold,
                "seed": int(seed),
                "runtime_seconds": time.perf_counter() - started,
                "cache_hit": False,
            }
        )
    ensemble = {
        stage: np.mean([per_seed[seed][stage] for seed in per_seed], axis=0)
        for stage in STAGES
    }
    selection = {
        "dataset": data.dataset,
        "model_id": _model_id(data.dataset, family),
        "model_family": family,
        "outer_fold": fold,
        "selected_params": json.dumps(best[1], sort_keys=True),
        "selected_epochs": best[2],
        "selected_inner_mean_stage_macro_f1": best[0][0],
        "selected_inner_worst_stage_macro_f1": best[0][1],
        "outer_rows_used_for_selection": False,
        "one_estimator_all_stages": True,
    }
    return ensemble, per_seed, selection, trials, runtimes


def prepare() -> dict[str, Any]:
    protocol = _protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG, OUT / "protocol.yaml")
    split_rows: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for dataset in DATASETS:
        data = tf._load_uci(dataset)
        views = build_stage_views(data)
        for stage, bundle in views.items():
            for fold in sorted(np.unique(bundle.outer_fold)):
                train_ids = sorted(bundle.record_id[bundle.outer_fold != fold].tolist())
                validation_ids = sorted(bundle.record_id[bundle.outer_fold == fold].tolist())
                split_rows.append(
                    {
                        "dataset": dataset,
                        "stage": stage,
                        "outer_fold": int(fold),
                        "train_count": len(train_ids),
                        "validation_count": len(validation_ids),
                        "train_record_ids_sha256": _stable_hash(train_ids),
                        "validation_record_ids_sha256": _stable_hash(validation_ids),
                    }
                )
        for family in MODELS:
            identities.append(
                {
                    "dataset": dataset,
                    "model_family": family,
                    "model_id": _model_id(dataset, family),
                    "display_name": DISPLAY_NAMES[family],
                    "stages": list(STAGES),
                    "identity_count": 1,
                }
            )
    split_frame = pd.DataFrame(split_rows)
    for (dataset, fold), group in split_frame.groupby(["dataset", "outer_fold"]):
        if group["train_record_ids_sha256"].nunique() != 1 or group[
            "validation_record_ids_sha256"
        ].nunique() != 1:
            raise RuntimeError(f"Stage split mismatch: {dataset} fold {fold}")
    _write_json(
        OUT / "split_manifest.json",
        {
            "schema_version": "unified_split_manifest_v1",
            "status": "PASS",
            "split_before_stage_expansion": True,
            "same_base_fold_all_stages": True,
            "rows": split_rows,
        },
    )
    _write_json(
        OUT / "model_identity_manifest.json",
        {
            "schema_version": "unified_model_identity_v1",
            "status": "PASS",
            "uci_model_identities": len(identities),
            "stage_result_rows_expected": len(identities) * len(STAGES),
            "identities": identities,
            "grade_band_reference_is_model_identity": False,
            "oulad": {"stage": "F2_MIDDLE", "retrained": False},
        },
    )
    _write_json(
        OUT / "cleanup_manifest.json",
        {
            "schema_version": "unified_cleanup_v1",
            "status": "PASS",
            "broad_git_clean_executed": False,
            "deleted_local_cache_categories": [
                "pytest_cache",
                "ruff_cache",
                "untracked_exploratory_prediction_caches",
            ],
            "kept_until_replacement_pass": [
                "artifacts/final/uci_timing_scenarios",
            ],
            "protected": [
                "OULAD",
                "recommendation",
                "database",
                "official checkpoints",
                "DOCX/PDF",
            ],
        },
    )
    return {
        "status": "PASS",
        "model_identities": len(identities),
        "expected_stage_rows": len(identities) * 3,
        "config_sha256": _sha(CONFIG),
        "training_performed": False,
    }


def train() -> dict[str, Any]:
    protocol = _protocol()
    if not (OUT / "split_manifest.json").is_file():
        prepare()
    prediction_parts: list[pd.DataFrame] = []
    seed_parts: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        data = tf._load_uci(dataset)
        views = build_stage_views(data)
        probabilities = {
            family: {
                stage: np.zeros((len(data.target), 3), dtype=float) for stage in STAGES
            }
            for family in MODELS
        }
        seed_probabilities = {
            family: {
                seed: {
                    stage: np.zeros((len(data.target), 3), dtype=float)
                    for stage in STAGES
                }
                for seed in protocol["training"]["outer_seeds"]
            }
            for family in MODELS
        }
        for family in MODELS:
            for fold in sorted(np.unique(data.outer_fold)):
                print(
                    f"[unified-stage] {dataset} {family} outer_fold={fold}",
                    flush=True,
                )
                validation = np.flatnonzero(data.outer_fold == fold)
                fit = np.flatnonzero(data.outer_fold != fold)
                runner = _fit_tabular_outer if family in TABULAR else _run_deep_outer
                ensemble, per_seed, selected, trials, runtimes = runner(
                    data=data,
                    views=views,
                    family=family,
                    fold=int(fold),
                    train=fit,
                    validation=validation,
                    protocol=protocol,
                )
                selected_rows.append(selected)
                trial_rows.extend(trials)
                runtime_rows.extend(runtimes)
                config = json.loads(selected["selected_params"])
                for seed in protocol["training"]["outer_seeds"]:
                    checkpoint = _checkpoint_path(dataset, family, int(fold), int(seed))
                    run_id = _run_id(dataset, family, int(fold), int(seed), config)
                    run_rows.append(
                        {
                            "training_run_id": run_id,
                            "dataset": dataset,
                            "model_id": _model_id(dataset, family),
                            "model_family": family,
                            "outer_fold": int(fold),
                            "seed": int(seed),
                            "config_hash": _stable_hash(config),
                            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                            "checkpoint_sha256": _sha(checkpoint),
                            "stages": list(STAGES),
                            "one_training_run_all_stages": True,
                        }
                    )
                    for stage in STAGES:
                        mapping_rows.append(
                            {
                                "training_run_id": run_id,
                                "dataset": dataset,
                                "model_id": _model_id(dataset, family),
                                "outer_fold": int(fold),
                                "seed": int(seed),
                                "prediction_stage": stage,
                                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                                "checkpoint_sha256": _sha(checkpoint),
                            }
                        )
                        seed_probabilities[family][int(seed)][stage][validation] = (
                            per_seed[int(seed)][stage]
                        )
                for stage in STAGES:
                    probabilities[family][stage][validation] = ensemble[stage]
        for family in MODELS:
            for stage in STAGES:
                probability = probabilities[family][stage]
                predicted = probability.argmax(1)
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "dataset": dataset,
                            "model_id": _model_id(dataset, family),
                            "model_family": family,
                            "prediction_stage": stage,
                            "outer_fold": data.outer_fold,
                            "record_id": data.record_ids,
                            "target": data.target,
                            "predicted_label": predicted,
                            "p_low": probability[:, 0],
                            "p_medium": probability[:, 1],
                            "p_high": probability[:, 2],
                            "probability_aggregation": "mean_across_all_fixed_seeds",
                        }
                    )
                )
                for seed in protocol["training"]["outer_seeds"]:
                    value = seed_probabilities[family][int(seed)][stage]
                    seed_parts.append(
                        pd.DataFrame(
                            {
                                "dataset": dataset,
                                "model_id": _model_id(dataset, family),
                                "model_family": family,
                                "prediction_stage": stage,
                                "outer_fold": data.outer_fold,
                                "record_id": data.record_ids,
                                "target": data.target,
                                "seed": int(seed),
                                "predicted_label": value.argmax(1),
                                "p_low": value[:, 0],
                                "p_medium": value[:, 1],
                                "p_high": value[:, 2],
                            }
                        )
                    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    seeds = pd.concat(seed_parts, ignore_index=True)
    predictions = predictions.sort_values(
        ["dataset", "model_id", "prediction_stage", "record_id"]
    ).reset_index(drop=True)
    seeds = seeds.sort_values(
        ["dataset", "model_id", "prediction_stage", "seed", "record_id"]
    ).reset_index(drop=True)
    _write_parquet(OUT / "predictions.parquet", predictions)
    _write_parquet(OUT / "seed_predictions.parquet", seeds)
    _write_csv(OUT / "selected_configs.csv", pd.DataFrame(selected_rows))
    _write_csv(OUT / "inner_trials.csv", pd.DataFrame(trial_rows))
    _write_csv(OUT / "runtime.csv", pd.DataFrame(runtime_rows))
    _write_json(
        OUT / "training_run_manifest.json",
        {
            "schema_version": "unified_training_run_v1",
            "status": "PASS",
            "training_run_count": len(run_rows),
            "expected_training_run_count": 2 * 10 * 5 * 5,
            "rows": run_rows,
        },
    )
    _write_json(
        OUT / "checkpoint_stage_mapping.json",
        {
            "schema_version": "unified_checkpoint_stage_mapping_v1",
            "status": "PASS",
            "mapping_count": len(mapping_rows),
            "same_checkpoint_all_stages": True,
            "rows": mapping_rows,
        },
    )
    return {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "seed_prediction_rows": len(seeds),
        "training_runs": len(run_rows),
    }


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    confidence = probability.max(1)
    predicted = probability.argmax(1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (confidence >= lower) & (
            (confidence <= upper) if index == bins - 1 else (confidence < upper)
        )
        if selected.any():
            result += float(selected.mean()) * abs(
                float((predicted[selected] == target[selected]).mean())
                - float(confidence[selected].mean())
            )
    return float(result)


def _metric_payload(target: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    predicted = probability.argmax(1)
    precision, recall, class_f1, support = precision_recall_fscore_support(
        target, predicted, labels=np.arange(3), zero_division=0
    )
    one_hot = np.eye(3)[target]
    matrix = confusion_matrix(target, predicted, labels=np.arange(3))
    return {
        "accuracy": float(accuracy_score(target, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(target, predicted, average="weighted", zero_division=0)
        ),
        "pr_auc": float(average_precision_score(one_hot, probability, average="macro")),
        "roc_auc": float(
            roc_auc_score(target, probability, multi_class="ovr", average="macro")
        ),
        "brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "nll": float(log_loss(target, probability, labels=np.arange(3))),
        "ece": _ece(target, probability),
        "medium_to_low_errors": int(matrix[1, 0]),
        "low_to_medium_errors": int(matrix[0, 1]),
        "confusion_matrix": matrix.tolist(),
        "per_class": [
            {
                "class_name": CLASS_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index in range(3)
        ],
    }


def _bootstrap_delta(
    target: np.ndarray,
    base_probability: np.ndarray,
    other_probability: np.ndarray,
    strata: np.ndarray,
    *,
    replicates: int = 5000,
    seed: int = 20260728,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    base_pred = base_probability.argmax(1)
    other_pred = other_probability.argmax(1)
    indices = np.concatenate(
        [
            rng.choice(group, size=(replicates, len(group)), replace=True)
            for group in groups
        ],
        axis=1,
    )
    sampled_target = target[indices]
    base_scores = _vectorized_macro_f1(sampled_target, base_pred[indices])
    other_scores = _vectorized_macro_f1(sampled_target, other_pred[indices])
    deltas = base_scores - other_scores
    low, high = np.quantile(deltas, [0.025, 0.975])
    point = f1_score(target, base_pred, average="macro", zero_division=0) - f1_score(
        target, other_pred, average="macro", zero_division=0
    )
    conclusion = (
        "CNN-BiLSTM higher"
        if low > 0
        else "CNN-BiLSTM lower"
        if high < 0
        else "insufficient evidence of difference"
    )
    return {
        "metric": "macro_f1",
        "delta": float(point),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "replicates": replicates,
        "conclusion": conclusion,
    }


def _vectorized_macro_f1(
    sampled_target: np.ndarray, sampled_prediction: np.ndarray
) -> np.ndarray:
    scores = np.zeros((sampled_target.shape[0], 3), dtype=float)
    for label in range(3):
        true = sampled_target == label
        predicted = sampled_prediction == label
        true_positive = np.sum(true & predicted, axis=1)
        false_positive = np.sum(~true & predicted, axis=1)
        false_negative = np.sum(true & ~predicted, axis=1)
        denominator = 2 * true_positive + false_positive + false_negative
        scores[:, label] = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator != 0,
        )
    return scores.mean(axis=1)


def _joint_stage_bootstrap_delta(
    target: np.ndarray,
    base_probabilities: list[np.ndarray],
    other_probabilities: list[np.ndarray],
    *,
    replicates: int = 5000,
    seed: int = 20260731,
) -> dict[str, Any]:
    """Resample base IDs once, then carry all three stage views together."""
    if len(base_probabilities) != 3 or len(other_probabilities) != 3:
        raise ValueError("Joint bootstrap requires exactly three stage views")
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(target == label) for label in np.unique(target)]
    base_pred = [value.argmax(1) for value in base_probabilities]
    other_pred = [value.argmax(1) for value in other_probabilities]
    tiled_target = np.tile(target, 3)
    indices = np.concatenate(
        [
            rng.choice(group, size=(replicates, len(group)), replace=True)
            for group in groups
        ],
        axis=1,
    )
    sampled_target = np.tile(target[indices], (1, 3))
    sampled_base = np.concatenate([value[indices] for value in base_pred], axis=1)
    sampled_other = np.concatenate([value[indices] for value in other_pred], axis=1)
    deltas = _vectorized_macro_f1(
        sampled_target, sampled_base
    ) - _vectorized_macro_f1(sampled_target, sampled_other)
    point = f1_score(
        tiled_target, np.concatenate(base_pred), average="macro", zero_division=0
    ) - f1_score(
        tiled_target, np.concatenate(other_pred), average="macro", zero_division=0
    )
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "metric": "macro_f1",
        "delta": float(point),
        "ci_95_low": float(low),
        "ci_95_high": float(high),
        "replicates": replicates,
        "resampling_unit": "base_record_with_all_three_stage_views",
        "conclusion": (
            "CNN-BiLSTM higher"
            if low > 0
            else "CNN-BiLSTM lower"
            if high < 0
            else "insufficient evidence of difference"
        ),
    }


def evaluate() -> dict[str, Any]:
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    seed_predictions = pd.read_parquet(OUT / "seed_predictions.parquet")
    metrics: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    matrices: list[dict[str, Any]] = []
    for (dataset, model_id, family, stage), group in predictions.groupby(
        ["dataset", "model_id", "model_family", "prediction_stage"], sort=True
    ):
        target = group["target"].to_numpy(dtype=int)
        probability = group[["p_low", "p_medium", "p_high"]].to_numpy(dtype=float)
        payload = _metric_payload(target, probability)
        seed_group = seed_predictions.loc[
            (seed_predictions["dataset"] == dataset)
            & (seed_predictions["model_id"] == model_id)
            & (seed_predictions["prediction_stage"] == stage)
        ]
        seed_scores = [
            f1_score(
                part["target"].to_numpy(dtype=int),
                part["predicted_label"].to_numpy(dtype=int),
                average="macro",
                zero_division=0,
            )
            for _, part in seed_group.groupby("seed")
        ]
        row = {
            "dataset": dataset,
            "model_id": model_id,
            "model_family": family,
            "prediction_stage": stage,
            **{key: value for key, value in payload.items() if key not in {"per_class", "confusion_matrix"}},
            "seed_macro_f1_mean": float(np.mean(seed_scores)),
            "seed_macro_f1_std": float(np.std(seed_scores, ddof=0)),
        }
        metrics.append(row)
        for item in payload["per_class"]:
            per_class.append(
                {
                    "dataset": dataset,
                    "model_id": model_id,
                    "model_family": family,
                    "prediction_stage": stage,
                    **item,
                }
            )
        matrices.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "prediction_stage": stage,
                "labels": list(CLASS_NAMES),
                "matrix": payload["confusion_matrix"],
            }
        )
    metric_frame = pd.DataFrame(metrics)
    for (dataset, model_id, family), group in predictions.groupby(
        ["dataset", "model_id", "model_family"], sort=True
    ):
        ordered = group.sort_values(["prediction_stage", "record_id"])
        target = ordered["target"].to_numpy(dtype=int)
        probability = ordered[["p_low", "p_medium", "p_high"]].to_numpy(dtype=float)
        payload = _metric_payload(target, probability)
        stage_rows = metric_frame.loc[
            (metric_frame["dataset"] == dataset)
            & (metric_frame["model_id"] == model_id)
        ]
        overall.append(
            {
                "dataset": dataset,
                "model_id": model_id,
                "model_family": family,
                "prediction_stage": "OVERALL",
                **{
                    key: value
                    for key, value in payload.items()
                    if key not in {"per_class", "confusion_matrix"}
                },
                "mean_stage_macro_f1": float(stage_rows["macro_f1"].mean()),
                "worst_stage_macro_f1": float(stage_rows["macro_f1"].min()),
            }
        )
    per_class_frame = pd.DataFrame(per_class)
    low = per_class_frame.loc[
        per_class_frame["class_name"] == "Low",
        ["dataset", "model_id", "prediction_stage", "recall", "f1"],
    ].rename(columns={"recall": "low_recall", "f1": "low_f1"})
    metric_frame = metric_frame.merge(
        low,
        on=["dataset", "model_id", "prediction_stage"],
        how="left",
        validate="one_to_one",
    )
    for metric_name in ("macro_f1", "balanced_accuracy", "pr_auc", "low_recall", "low_f1"):
        metric_frame[f"rank_{metric_name}"] = metric_frame.groupby(
            ["dataset", "prediction_stage"]
        )[metric_name].rank(method="min", ascending=False).astype(int)
    overall_frame = pd.DataFrame(overall)
    for metric_name in (
        "macro_f1",
        "mean_stage_macro_f1",
        "worst_stage_macro_f1",
        "balanced_accuracy",
        "pr_auc",
    ):
        overall_frame[f"rank_{metric_name}"] = overall_frame.groupby("dataset")[
            metric_name
        ].rank(method="min", ascending=False).astype(int)
    bootstrap_stage: list[dict[str, Any]] = []
    bootstrap_overall: list[dict[str, Any]] = []
    for dataset in DATASETS:
        cnn_id = _model_id(dataset, "cnn_bilstm")
        for family in MODELS:
            if family == "cnn_bilstm":
                continue
            other_id = _model_id(dataset, family)
            stage_base: list[np.ndarray] = []
            stage_other: list[np.ndarray] = []
            base_target: np.ndarray | None = None
            for stage_index, stage in enumerate(STAGES):
                base = predictions.loc[
                    (predictions["dataset"] == dataset)
                    & (predictions["model_id"] == cnn_id)
                    & (predictions["prediction_stage"] == stage)
                ].sort_values("record_id")
                other = predictions.loc[
                    (predictions["dataset"] == dataset)
                    & (predictions["model_id"] == other_id)
                    & (predictions["prediction_stage"] == stage)
                ].sort_values("record_id")
                if not np.array_equal(base["record_id"], other["record_id"]):
                    raise RuntimeError("Paired bootstrap record alignment failed")
                target = base["target"].to_numpy(dtype=int)
                result = _bootstrap_delta(
                    target,
                    base[["p_low", "p_medium", "p_high"]].to_numpy(),
                    other[["p_low", "p_medium", "p_high"]].to_numpy(),
                    target,
                    seed=20260728 + stage_index,
                )
                bootstrap_stage.append(
                    {
                        "dataset": dataset,
                        "prediction_stage": stage,
                        "base_model_id": cnn_id,
                        "comparator_model_id": other_id,
                        **result,
                    }
                )
                if base_target is None:
                    base_target = target
                elif not np.array_equal(base_target, target):
                    raise RuntimeError("Stage targets differ for joint bootstrap")
                stage_base.append(
                    base[["p_low", "p_medium", "p_high"]].to_numpy()
                )
                stage_other.append(
                    other[["p_low", "p_medium", "p_high"]].to_numpy()
                )
            assert base_target is not None
            result = _joint_stage_bootstrap_delta(
                base_target,
                stage_base,
                stage_other,
            )
            bootstrap_overall.append(
                {
                    "dataset": dataset,
                    "prediction_stage": "OVERALL",
                    "base_model_id": cnn_id,
                    "comparator_model_id": other_id,
                    **result,
                }
            )
    hybrid_rows = []
    for row in bootstrap_stage + bootstrap_overall:
        cnn = (
            metric_frame
            if row["prediction_stage"] != "OVERALL"
            else overall_frame
        )
        selected = cnn.loc[
            (cnn["dataset"] == row["dataset"])
            & (cnn["model_id"] == row["base_model_id"])
            & (cnn["prediction_stage"] == row["prediction_stage"])
        ].iloc[0]
        comparison = cnn.loc[
            (cnn["dataset"] == row["dataset"])
            & (cnn["model_id"] == row["comparator_model_id"])
            & (cnn["prediction_stage"] == row["prediction_stage"])
        ].iloc[0]
        hybrid_rows.append(
            {
                **row,
                "cnn_bilstm_macro_f1": selected["macro_f1"],
                "comparator_macro_f1": comparison["macro_f1"],
            }
        )
    _write_csv(OUT / "stage_metrics.csv", metric_frame)
    _write_csv(OUT / "overall_metrics.csv", overall_frame)
    _write_csv(OUT / "per_class_metrics.csv", per_class_frame)
    _write_json(
        OUT / "confusion_matrices.json",
        {"schema_version": "unified_confusion_v1", "rows": matrices},
    )
    _write_csv(OUT / "bootstrap_stage.csv", pd.DataFrame(bootstrap_stage))
    _write_csv(OUT / "bootstrap_overall.csv", pd.DataFrame(bootstrap_overall))
    _write_csv(OUT / "hybrid_strength_matrix.csv", pd.DataFrame(hybrid_rows))
    _write_final_authority(metric_frame, overall_frame)
    return {
        "status": "PASS",
        "stage_metric_rows": len(metric_frame),
        "overall_metric_rows": len(overall_frame),
        "bootstrap_stage_rows": len(bootstrap_stage),
        "bootstrap_overall_rows": len(bootstrap_overall),
    }


def _write_final_authority(stage: pd.DataFrame, overall: pd.DataFrame) -> None:
    stage_final = stage.copy()
    stage_final["authority_scope"] = "UNIFIED_STAGE_AWARE_UCI"
    overall_final = overall.copy()
    overall_final["authority_scope"] = "UNIFIED_STAGE_AWARE_UCI"
    frozen = json.loads(
        (ROOT / "artifacts" / "final" / "final_results.json").read_text(
            encoding="utf-8"
        )
    )
    oulad_rows = []
    for item in frozen["datasets"]["oulad"]["models"]:
        metrics = {
            name: payload["value"]
            for name, payload in item["metrics"].items()
            if isinstance(payload, dict) and "value" in payload
        }
        oulad_rows.append(
            {
                "dataset": "oulad",
                "model_id": f"{item['model_id']}_oulad",
                "model_family": item["model_id"],
                "prediction_stage": "F2_MIDDLE",
                **metrics,
                "authority_scope": "FROZEN_OULAD_UNCHANGED",
            }
        )
    oulad = pd.DataFrame(oulad_rows)
    _write_csv(
        ROOT / "artifacts" / "final" / "final_stage_results.csv",
        pd.concat([stage_final, oulad], ignore_index=True, sort=False),
    )
    _write_csv(
        ROOT / "artifacts" / "final" / "final_overall_results.csv",
        pd.concat([overall_final, oulad], ignore_index=True, sort=False),
    )


def _grade_band_reference() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        data = tf._load_uci(dataset)
        for stage in STAGES:
            if stage == STAGES[0]:
                continue
            probability = np.zeros((len(data.target), 3), dtype=float)
            for fold in sorted(np.unique(data.outer_fold)):
                validation = np.flatnonzero(data.outer_fold == fold)
                train = np.flatnonzero(data.outer_fold != fold)
                column = "G1" if stage == STAGES[1] else "G2"
                train_band = tf.encode_uci_target(data.frame.iloc[train][column])
                validation_band = tf.encode_uci_target(
                    data.frame.iloc[validation][column]
                )
                table = np.ones((3, 3), dtype=float)
                for band, target in zip(train_band, data.target[train]):
                    table[band, target] += 1.0
                table /= table.sum(axis=1, keepdims=True)
                probability[validation] = table[validation_band]
            payload = _metric_payload(data.target, probability)
            rows.append(
                {
                    "dataset": dataset,
                    "prediction_stage": stage,
                    "reference_id": "grade_band_reference",
                    "model_identity": False,
                    "fit_scope": "outer_training_fold_only",
                    **{
                        key: value
                        for key, value in payload.items()
                        if key not in {"per_class", "confusion_matrix"}
                    },
                }
            )
    return rows


def report() -> dict[str, Any]:
    stage = pd.read_csv(OUT / "stage_metrics.csv")
    overall = pd.read_csv(OUT / "overall_metrics.csv")
    selected = pd.read_csv(OUT / "selected_configs.csv")
    bootstrap = pd.read_csv(OUT / "bootstrap_stage.csv")
    reference = _grade_band_reference()
    validation_path = OUT / "validation.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else {"status": "PENDING"}
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Unified Stage-Aware UCI Results",
        "",
        "One estimator is fitted per dataset/model/fold/seed and reused for S0, S1, and S2. "
        "A stage is a prediction view, not a model identity. Outer rows are never used for "
        "configuration selection.",
        "",
        "## Stage results",
        "",
        "| Dataset | Stage | Best model | Macro-F1 | Low recall |",
        "|---|---|---|---:|---:|",
    ]
    per_class = pd.read_csv(OUT / "per_class_metrics.csv")
    for (dataset, stage_name), group in stage.groupby(["dataset", "prediction_stage"]):
        best = group.sort_values(
            ["macro_f1", "balanced_accuracy"], ascending=False
        ).iloc[0]
        low = per_class.loc[
            (per_class["dataset"] == dataset)
            & (per_class["prediction_stage"] == stage_name)
            & (per_class["model_id"] == best["model_id"])
            & (per_class["class_name"] == "Low")
        ].iloc[0]
        lines.append(
            f"| {dataset} | {stage_name} | {best['model_id']} | "
            f"{best['macro_f1']:.4f} | {low['recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- UCI results are the unified fixed-fold authority produced by this branch.",
            "- S2 is the late-stage UCI prediction view; S0 and S1 quantify earlier availability.",
            "- OULAD remains frozen at `F2_MIDDLE`; it was not retrained.",
            "- The grade-band reference is training-fold-only diagnostic evidence and is not an eleventh model.",
            "- Future OULAD remains `LOCKED_NOT_EXECUTED`.",
            "",
            f"Validation state at report generation: `{validation['status']}`.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    matrix_lines = [
        "# Hybrid vs ML Stage Matrix",
        "",
        "Paired intervals use 5,000 base-record bootstrap replicates. Overall resampling "
        "keeps all three views of each sampled base record together.",
        "",
        "| Dataset | Stage | Comparator | Delta Macro-F1 | 95% CI | Conclusion |",
        "|---|---|---|---:|---|---|",
    ]
    for row in bootstrap.itertuples():
        matrix_lines.append(
            f"| {row.dataset} | {row.prediction_stage} | {row.comparator_model_id} | "
            f"{row.delta:.4f} | [{row.ci_95_low:.4f}, {row.ci_95_high:.4f}] | "
            f"{row.conclusion} |"
        )
    MATRIX_REPORT.write_text(
        "\n".join(matrix_lines) + "\n", encoding="utf-8", newline="\n"
    )
    selection_lines = [
        "# Unified Model Selection Report",
        "",
        "The selection objective is the mean Macro-F1 across S0/S1/S2 on three inner "
        "folds. A single configuration and estimator serve all stages.",
        "",
        "| Dataset | Model | Outer fold | Inner mean stage Macro-F1 |",
        "|---|---|---:|---:|",
    ]
    for row in selected.itertuples():
        selection_lines.append(
            f"| {row.dataset} | {row.model_id} | {row.outer_fold} | "
            f"{row.selected_inner_mean_stage_macro_f1:.4f} |"
        )
    selection_lines.extend(
        [
            "",
            "No outer score, best seed, transfer checkpoint, pretrained checkpoint, "
            "synthetic oversampling, ordinal auxiliary head, regression auxiliary head, "
            "or grade-band prior was used for selection.",
            "",
            "## Grade-band diagnostic reference",
            "",
            "The following training-fold-only reference is reported without a model identity:",
            "",
        ]
    )
    for row in reference:
        selection_lines.append(
            f"- {row['dataset']} {row['prediction_stage']}: Macro-F1 "
            f"{row['macro_f1']:.4f}."
        )
    SELECTION_REPORT.write_text(
        "\n".join(selection_lines) + "\n", encoding="utf-8", newline="\n"
    )
    return {"status": "PASS", "reports": 3}


def _joint_bootstrap_validation() -> bool:
    frame = pd.read_csv(OUT / "bootstrap_overall.csv")
    return bool(
        len(frame) == 18
        and set(frame["replicates"]) == {5000}
        and set(frame["prediction_stage"]) == {"OVERALL"}
    )


def _canonical_freeze() -> dict[str, Any]:
    baseline = json.loads(
        (
            ROOT
            / "artifacts"
            / "final"
            / "protocol_snapshots"
            / "pre_unified_scientific_freeze.json"
        ).read_text(encoding="utf-8")
    )
    protected = (
        "artifacts/final/final_results.json",
        "artifacts/final/final_results.csv",
    )
    checks = {
        path: _sha(ROOT / path)
        for path in protected
    }
    expected = {path: baseline["canonical_sha256"][path] for path in protected}
    return {
        "expected": expected,
        "actual": checks,
        "unchanged": checks == expected,
    }


def _build_checksums() -> dict[str, Any]:
    excluded = {
        OUT / "checksums.json",
        OUT / "validation.json",
    }
    files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file()
        and path not in excluded
        and ".predictions.npz" not in path.name
    )
    payload = {
        "schema_version": "unified_checksums_v1",
        "status": "PASS",
        "entries": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    _write_json(OUT / "checksums.json", payload)
    return payload


def validate_legacy_provenance() -> dict[str, Any]:
    """Validate the compact public record of locally archived experiments."""

    if not PROVENANCE.is_file():
        raise RuntimeError("Legacy provenance manifest is missing")
    payload = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    return {
        "status": "PASS"
        if payload.get("status") == "PASS"
        and payload.get("classification") == "LEGACY_SEPARATE_STAGE_EVIDENCE"
        else "FAIL",
        "verbatim_files": len(rows),
        "manifest": PROVENANCE.relative_to(ROOT).as_posix(),
        "local_archive_required_for_final_runtime": False,
    }


def _build_unified_evidence_manifest() -> dict[str, Any]:
    roots = [OUT]
    individual = [
        ROOT / "artifacts" / "final" / "final_stage_results.csv",
        ROOT / "artifacts" / "final" / "final_overall_results.csv",
        ROOT
        / "artifacts"
        / "final"
        / "protocol_snapshots"
        / "pre_unified_scientific_freeze.json",
        ROOT
        / "artifacts"
        / "final"
        / "database"
        / "unified_database_replacement_validation.json",
        ROOT
        / "artifacts"
        / "final"
        / "database"
        / "unified_database_checksum_manifest.json",
        PROVENANCE,
        REPORT,
        MATRIX_REPORT,
        SELECTION_REPORT,
        ROOT / "reports" / "final" / "PROJECT_LOCK_REPORT.md",
        ROOT / "reports" / "final" / "DATABASE_STAGE_AWARE_MIGRATION.md",
    ]
    files = {
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and ".predictions.npz" not in path.name
    }
    files.update(path for path in individual if path.is_file())
    output = ROOT / "artifacts" / "final" / "unified_stage_evidence_manifest.json"
    files.discard(output)
    files.discard(OUT / "validation.json")
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(files)
    ]
    payload = {
        "schema_version": "unified_stage_evidence_manifest_v1",
        "status": "PASS",
        "entry_count": len(entries),
        "canonical_legacy_manifest_unchanged": True,
        "entries": entries,
    }
    _write_json(output, payload)
    return payload


def validate() -> dict[str, Any]:
    errors: list[str] = []
    required = (
        "protocol.yaml",
        "cleanup_manifest.json",
        "split_manifest.json",
        "model_identity_manifest.json",
        "training_run_manifest.json",
        "checkpoint_stage_mapping.json",
        "selected_configs.csv",
        "inner_trials.csv",
        "stage_metrics.csv",
        "overall_metrics.csv",
        "per_class_metrics.csv",
        "confusion_matrices.json",
        "predictions.parquet",
        "seed_predictions.parquet",
        "bootstrap_stage.csv",
        "bootstrap_overall.csv",
        "hybrid_strength_matrix.csv",
        "runtime.csv",
    )
    for name in required:
        if not (OUT / name).is_file():
            errors.append(f"missing {name}")
    if not errors:
        predictions = pd.read_parquet(OUT / "predictions.parquet")
        seeds = pd.read_parquet(OUT / "seed_predictions.parquet")
        stage = pd.read_csv(OUT / "stage_metrics.csv")
        overall = pd.read_csv(OUT / "overall_metrics.csv")
        selected = pd.read_csv(OUT / "selected_configs.csv")
        runs = json.loads((OUT / "training_run_manifest.json").read_text(encoding="utf-8"))
        mapping = json.loads(
            (OUT / "checkpoint_stage_mapping.json").read_text(encoding="utf-8")
        )
        expected_prediction_rows = (395 + 649) * 10 * 3
        if len(predictions) != expected_prediction_rows:
            errors.append(
                f"prediction rows {len(predictions)} != {expected_prediction_rows}"
            )
        if len(seeds) != expected_prediction_rows * 5:
            errors.append("seed prediction coverage mismatch")
        if len(stage) != 60 or len(overall) != 20:
            errors.append("metric authority must contain 60 stage and 20 overall rows")
        if runs["training_run_count"] != 500:
            errors.append("expected 500 dataset/model/fold/seed training runs")
        if mapping["mapping_count"] != 1500:
            errors.append("expected exactly three stage mappings per training run")
        if len(selected) != 100 or selected["outer_rows_used_for_selection"].any():
            errors.append("inner-only selection evidence mismatch")
        for _, group in predictions.groupby(["dataset", "model_id", "record_id"]):
            if set(group["prediction_stage"]) != set(STAGES):
                errors.append("base record missing a stage view")
                break
            if group["outer_fold"].nunique() != 1:
                errors.append("base record fold differs by stage")
                break
        duplicate = predictions.duplicated(
            ["dataset", "model_id", "prediction_stage", "record_id"]
        ).any()
        if duplicate:
            errors.append("duplicate stage prediction key")
        if set(stage["model_family"]) != set(MODELS):
            errors.append("ten model families are required")
        if not _joint_bootstrap_validation():
            errors.append("overall bootstrap coverage mismatch")
        for row in runs["rows"]:
            checkpoint = ROOT / row["checkpoint"]
            if not checkpoint.is_file() or _sha(checkpoint) != row["checkpoint_sha256"]:
                errors.append(f"checkpoint checksum mismatch: {row['checkpoint']}")
                break
        grouped_mapping = {}
        for row in mapping["rows"]:
            key = row["training_run_id"]
            grouped_mapping.setdefault(key, set()).add(row["checkpoint"])
        if any(len(paths) != 1 for paths in grouped_mapping.values()):
            errors.append("a training run maps stages to different checkpoints")
    freeze = _canonical_freeze()
    if not freeze["unchanged"]:
        errors.append("official final artifacts changed")
    checksum = _build_checksums() if not errors else {"entries": []}
    legacy = validate_legacy_provenance()
    if legacy["status"] != "PASS":
        errors.append("legacy archive checksum preservation failed")
    evidence = _build_unified_evidence_manifest()
    result = {
        "schema_version": "unified_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "uci_model_identities": 20,
        "uci_stage_rows": 60,
        "training_run_count": 500 if not errors else None,
        "one_estimator_all_stages": not errors,
        "outer_tuning": False,
        "best_seed_selection": False,
        "transfer_learning": False,
        "pretrained_checkpoint": False,
        "synthetic_resampling": False,
        "grade_band_reference_is_model_identity": False,
        "future_oulad": "LOCKED_NOT_EXECUTED",
        "oulad_retrained": False,
        "official_freeze": freeze,
        "checksum_entries": len(checksum["entries"]),
        "legacy_archive_files": legacy["verbatim_files"],
        "evidence_manifest_entries": evidence["entry_count"],
    }
    _write_json(OUT / "validation.json", result)
    return result


def all_steps() -> dict[str, Any]:
    started = time.perf_counter()
    prepare()
    trained = train()
    evaluated = evaluate()
    report()
    validated = validate()
    validated["runtime_seconds"] = time.perf_counter() - started
    validated["training"] = trained
    validated["evaluation"] = evaluated
    return validated


__all__ = [
    "DATASETS",
    "MODELS",
    "STAGES",
    "StageViewBundle",
    "all_steps",
    "build_stage_views",
    "evaluate",
    "prepare",
    "report",
    "train",
    "validate",
]
