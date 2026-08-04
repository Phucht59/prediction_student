"""Nested grouped execution for action-aware integrated V4 heads.

The residual CNN-BiLSTM prediction backbone and V3 embedding caches remain
frozen.  Only integrated neural heads are trained.  Candidate binary labels are
supervised on every valid candidate so negative groups directly penalize false
issue behaviour.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/two_stage_v4"
V3_CACHE = ROOT / "artifacts/recommend_hybrid/two_stage_v3/cache"
MODEL_SELECTION = OUT / "model_selection"
FINAL_OOF = OUT / "final_oof"
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/two_stage_v4_protocol.yaml"
sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.two_stage_v3.data import (  # noqa: E402
    TwoStageArrays,
    apply_scaler,
    fit_scaler,
    load_two_stage_arrays,
)
from src.recommend_hybrid.two_stage_v4.metrics import (  # noqa: E402
    ACTION_COUNT,
    STAGE_ORDER,
    ActionAwareThresholds,
    blended_gate_probability,
    evaluate_action_aware,
    make_decisions,
    ranking_metrics,
)
from src.recommend_hybrid.two_stage_v4.model import (  # noqa: E402
    ActionAwareHeadConfig,
    HybridActionAwareRecommendationHeads,
    action_aware_loss,
)
from src.recommend_hybrid.two_stage_v4.selection import (  # noqa: E402
    select_action_aware_thresholds,
)

ACTION_ORDER = (
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
)
INNER_SEED = 42


@dataclass(frozen=True)
class TrainedHead:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_validation_loss: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _registered_configs(
    protocol: dict[str, Any],
    group_dim: int,
    action_dim: int,
) -> list[dict[str, Any]]:
    rows = []
    for index, registered in enumerate(protocol["head_training"]["registered_configs"]):
        payload = {
            "group_feature_dim": int(group_dim),
            "action_feature_dim": int(action_dim),
            "group_hidden_dim": int(registered["hidden"]),
            "action_embedding_dim": int(registered["action_embedding"]),
            "dropout": float(registered["dropout"]),
            "learning_rate": float(registered["learning_rate"]),
            "weight_decay": 1.0e-4,
            "recommendability_loss_weight": float(registered["gate_weight"]),
            "listwise_loss_weight": float(
                protocol["head_training"]["listwise_loss_weight"]
            ),
            "candidate_binary_loss_weight": float(
                registered["candidate_weight"]
            ),
            "consistency_loss_weight": float(
                registered["consistency_weight"]
            ),
            "focal_gamma": float(registered["focal_gamma"]),
            "trial_index": int(index),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["config_id"] = hashlib.sha256(encoded).hexdigest()[:16]
        rows.append(payload)
    return rows


def _head_config(payload: dict[str, Any]) -> ActionAwareHeadConfig:
    return ActionAwareHeadConfig(
        group_feature_dim=int(payload["group_feature_dim"]),
        action_feature_dim=int(payload["action_feature_dim"]),
        group_hidden_dim=int(payload["group_hidden_dim"]),
        action_embedding_dim=int(payload["action_embedding_dim"]),
        dropout=float(payload["dropout"]),
        recommendability_loss_weight=float(
            payload["recommendability_loss_weight"]
        ),
        listwise_loss_weight=float(payload["listwise_loss_weight"]),
        candidate_binary_loss_weight=float(
            payload["candidate_binary_loss_weight"]
        ),
        consistency_loss_weight=float(payload["consistency_loss_weight"]),
        focal_gamma=float(payload["focal_gamma"]),
    )


def _dataset(arrays: TwoStageArrays, indexes: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(arrays.group_features[indexes]),
        torch.from_numpy(arrays.action_features[indexes]),
        torch.from_numpy(arrays.action_ids[indexes]),
        torch.from_numpy(arrays.action_mask[indexes]),
        torch.from_numpy(arrays.group_target[indexes]),
        torch.from_numpy(arrays.action_target[indexes]),
    )


def _group_positive_weight(
    arrays: TwoStageArrays,
    indexes: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    target = arrays.group_target[indexes]
    positive = float((target > 0.5).sum())
    negative = float((target <= 0.5).sum())
    return torch.tensor(
        negative / max(positive, 1.0),
        dtype=torch.float32,
        device=device,
    )


def _action_positive_weight(
    arrays: TwoStageArrays,
    indexes: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    target = arrays.action_target[indexes]
    valid = arrays.action_mask[indexes]
    weights = []
    for action_id in range(ACTION_COUNT):
        action_valid = valid[:, action_id]
        action_target = target[:, action_id][action_valid]
        positive = float((action_target > 0.5).sum())
        negative = float((action_target <= 0.5).sum())
        weight = negative / positive if positive > 0 else 1.0
        weights.append(min(max(weight, 1.0), 100.0))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _validation_loss(
    model: nn.Module,
    arrays: TwoStageArrays,
    indexes: np.ndarray,
    config: ActionAwareHeadConfig,
    group_positive_weight: torch.Tensor,
    action_positive_weight: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    values: list[float] = []
    weights: list[int] = []
    loader = DataLoader(_dataset(arrays, indexes), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            group, action, action_ids, mask, group_target, action_target = (
                item.to(device) for item in batch
            )
            output = model(group, action, action_ids, mask)
            loss, _ = action_aware_loss(
                output,
                group_target=group_target,
                action_target=action_target,
                action_mask=mask,
                group_positive_weight=group_positive_weight,
                action_positive_weight=action_positive_weight,
                config=config,
            )
            values.append(float(loss.detach().cpu()))
            weights.append(len(group))
    return float(np.average(values, weights=weights))


def _train_head(
    arrays: TwoStageArrays,
    train_indexes: np.ndarray,
    config_payload: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
    validation_indexes: np.ndarray | None,
    fixed_epochs: int | None,
    max_epochs: int,
    patience: int,
    batch_size: int,
) -> TrainedHead:
    _seed_everything(seed)
    config = _head_config(config_payload)
    model = HybridActionAwareRecommendationHeads(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config_payload["learning_rate"]),
        weight_decay=float(config_payload["weight_decay"]),
    )
    group_weight = _group_positive_weight(arrays, train_indexes, device)
    action_weight = _action_positive_weight(arrays, train_indexes, device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _dataset(arrays, train_indexes),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    epoch_limit = int(fixed_epochs or max_epochs)
    best_loss = float("inf")
    best_epoch = epoch_limit if fixed_epochs is not None else 1
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0
    for epoch in range(1, epoch_limit + 1):
        model.train()
        for batch in loader:
            group, action, action_ids, mask, group_target, action_target = (
                item.to(device) for item in batch
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(group, action, action_ids, mask)
            loss, _ = action_aware_loss(
                output,
                group_target=group_target,
                action_target=action_target,
                action_mask=mask,
                group_positive_weight=group_weight,
                action_positive_weight=action_weight,
                config=config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if fixed_epochs is not None:
            continue
        assert validation_indexes is not None
        current = _validation_loss(
            model,
            arrays,
            validation_indexes,
            config,
            group_weight,
            action_weight,
            device,
            batch_size,
        )
        if current < best_loss - 1.0e-5:
            best_loss = current
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if fixed_epochs is not None:
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        best_loss = float("nan")
        best_epoch = int(fixed_epochs)
    if best_state is None:
        raise RuntimeError("V4 head training did not produce a state dictionary")
    return TrainedHead(best_state, best_epoch, best_loss)


def _predict_head(
    state_dict: dict[str, torch.Tensor],
    arrays: TwoStageArrays,
    indexes: np.ndarray,
    config_payload: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model = HybridActionAwareRecommendationHeads(_head_config(config_payload)).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    direct_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    loader = DataLoader(_dataset(arrays, indexes), batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            group, action, action_ids, mask, _, _ = (item.to(device) for item in batch)
            output = model(group, action, action_ids, mask)
            direct_rows.append(output.direct_gate_logit.cpu().numpy())
            action_rows.append(output.action_logits.cpu().numpy())
    return np.concatenate(direct_rows), np.concatenate(action_rows)


def _inner_splits(
    arrays: TwoStageArrays,
    outer_fold: int,
    inner_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    outer_train = np.where(arrays.outer_folds != outer_fold)[0]
    splitter = StratifiedGroupKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=20260804 + outer_fold,
    )
    result = []
    for fit_relative, validation_relative in splitter.split(
        outer_train,
        arrays.group_target[outer_train],
        arrays.learner_ids[outer_train],
    ):
        result.append((outer_train[fit_relative], outer_train[validation_relative]))
    return result


def _thresholds_from_dict(payload: dict[str, Any]) -> ActionAwareThresholds:
    return ActionAwareThresholds(
        stage_gate_probability=tuple(
            float(value) for value in payload["stage_gate_probability"]
        ),
        direct_action_blend=float(payload["direct_action_blend"]),
        minimum_action_probability=float(payload["minimum_action_probability"]),
        minimum_action_margin=float(payload["minimum_action_margin"]),
        action_probability_by_id=tuple(
            float(value) for value in payload["action_probability_by_id"]
        ),
    )


def _worst_stage(metrics: dict[str, object]) -> float:
    supported = [
        float(row["end_to_end_precision_at_1"])
        for row in metrics.get("per_stage", [])
        if int(row["issued_groups"]) >= 50
    ]
    return min(supported) if supported else 0.0


def _select_trial(
    trials: pd.DataFrame,
    minimum_coverage: float,
    target: float,
) -> pd.Series:
    covered = trials[trials["positive_group_coverage"] >= minimum_coverage].copy()
    passed = covered[covered["end_to_end_precision_at_1"] >= target].copy()
    pool = passed if len(passed) else (covered if len(covered) else trials.copy())
    if len(passed):
        columns = [
            "positive_group_coverage",
            "worst_stage_precision",
            "stage_a_precision",
            "stage_b_conditional_precision_at_1",
            "config_id",
        ]
        ascending = [False, False, False, False, True]
    else:
        columns = [
            "end_to_end_precision_at_1",
            "stage_a_precision",
            "worst_stage_precision",
            "stage_b_conditional_precision_at_1",
            "positive_group_coverage",
            "config_id",
        ]
        ascending = [False, False, False, False, False, True]
    return pool.sort_values(columns, ascending=ascending, kind="stable").iloc[0]


def _run_inner_selection(
    arrays: TwoStageArrays,
    outer_fold: int,
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    splits = _inner_splits(
        arrays,
        outer_fold,
        int(protocol["evaluation"]["inner_group_folds"]),
    )
    outer_train = np.where(arrays.outer_folds != outer_fold)[0]
    configs = _registered_configs(
        protocol,
        arrays.group_features.shape[1],
        arrays.action_features.shape[2],
    )
    trial_rows: list[dict[str, Any]] = []
    payload_by_config: dict[str, dict[str, Any]] = {}
    for config in configs:
        direct_oof = np.full(arrays.size, np.nan, dtype=np.float32)
        action_oof = np.full((arrays.size, ACTION_COUNT), np.nan, dtype=np.float32)
        epochs: list[int] = []
        validation_losses: list[float] = []
        for inner_fold, (fit_indexes, validation_indexes) in enumerate(splits):
            scaler = fit_scaler(arrays, fit_indexes)
            scaled = apply_scaler(arrays, scaler)
            trained = _train_head(
                scaled,
                fit_indexes,
                config,
                seed=INNER_SEED + outer_fold * 100 + inner_fold,
                device=device,
                validation_indexes=validation_indexes,
                fixed_epochs=None,
                max_epochs=int(protocol["head_training"]["max_epochs"]),
                patience=int(protocol["head_training"]["patience"]),
                batch_size=int(protocol["head_training"]["batch_size"]),
            )
            direct, action = _predict_head(
                trained.state_dict,
                scaled,
                validation_indexes,
                config,
                device=device,
                batch_size=int(protocol["head_training"]["batch_size"]),
            )
            direct_oof[validation_indexes] = direct
            action_oof[validation_indexes] = action
            epochs.append(trained.best_epoch)
            validation_losses.append(trained.best_validation_loss)
        if np.isnan(direct_oof[outer_train]).any() or np.isnan(
            action_oof[outer_train]
        ).any():
            raise RuntimeError("V4 inner OOF predictions are incomplete")
        thresholds, metrics, audit = select_action_aware_thresholds(
            direct_gate_logits=direct_oof[outer_train],
            action_logits=action_oof[outer_train],
            action_mask=arrays.action_mask[outer_train],
            group_target=arrays.group_target[outer_train],
            action_target=arrays.action_target[outer_train],
            stages=arrays.stages[outer_train],
            blend_weights=protocol["selection"]["direct_action_blend_weights"],
            action_probability_grid=protocol["selection"][
                "action_probability_threshold_grid"
            ],
            margin_grid=protocol["selection"]["action_margin_threshold_grid"],
            stage_coverage_floor=protocol["selection"][
                "required_stage_coverage_for_calibration"
            ],
            minimum_global_coverage=float(
                protocol["selection"]["required_global_positive_group_coverage"]
            ),
            target_precision=float(
                protocol["release_gates"]["end_to_end_precision_at_1_minimum"]
            ),
            action_specific_minimum_support=int(
                protocol["selection"][
                    "action_specific_threshold_minimum_support"
                ]
            ),
            stage_quantile_count=61,
        )
        fixed_epoch = int(round(float(np.median(epochs))))
        row = {
            **config,
            "outer_fold": int(outer_fold),
            "fixed_epoch": fixed_epoch,
            "mean_best_epoch": float(np.mean(epochs)),
            "mean_validation_loss": float(np.nanmean(validation_losses)),
            "end_to_end_precision_at_1": float(
                metrics["end_to_end_precision_at_1"]
            ),
            "positive_group_coverage": float(metrics["positive_group_coverage"]),
            "stage_a_precision": float(metrics["stage_a_precision"]),
            "stage_a_recall": float(metrics["stage_a_recall"]),
            "stage_b_conditional_precision_at_1": float(
                metrics["stage_b_conditional_precision_at_1"]
            ),
            "conditional_precision_at_1_all_positive": float(
                metrics["conditional_precision_at_1_all_positive"]
            ),
            "ndcg_at_3": float(metrics["ndcg_at_3"]),
            "mrr": float(metrics["mrr"]),
            "abstention_rate": float(metrics["abstention_rate"]),
            "action_diversity": int(metrics["action_diversity"]),
            "top_action_concentration": float(metrics["top_action_concentration"]),
            "worst_stage_precision": _worst_stage(metrics),
            "selection_target_met": bool(metrics["selection_target_met"]),
        }
        trial_rows.append(row)
        payload_by_config[str(config["config_id"])] = {
            "config": config,
            "fixed_epoch": fixed_epoch,
            "thresholds": thresholds.to_dict(),
            "metrics": metrics,
            "audit": audit,
        }
    trials = pd.DataFrame(trial_rows)
    selected_row = _select_trial(
        trials,
        float(protocol["selection"]["required_global_positive_group_coverage"]),
        float(protocol["release_gates"]["end_to_end_precision_at_1_minimum"]),
    )
    return payload_by_config[str(selected_row["config_id"])], trials


def _classification_metrics_from_probability(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    target_values = np.asarray(target, dtype=np.int8)
    probability_values = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    if len(np.unique(target_values)) < 2:
        roc_auc = 0.5
        average_precision = float(target_values.mean())
    else:
        roc_auc = float(roc_auc_score(target_values, probability_values))
        average_precision = float(
            average_precision_score(target_values, probability_values)
        )
    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "brier_score": float(brier_score_loss(target_values, probability_values)),
    }


def _per_action_metrics(
    arrays: TwoStageArrays,
    indexes: np.ndarray,
    direct_logits: np.ndarray,
    action_logits: np.ndarray,
    thresholds: ActionAwareThresholds,
) -> list[dict[str, Any]]:
    stages = arrays.stages[indexes]
    decision = make_decisions(
        direct_logits,
        action_logits,
        arrays.action_mask[indexes],
        stages,
        thresholds,
    )
    row = np.arange(len(indexes))
    action_target = arrays.action_target[indexes]
    group_target = arrays.group_target[indexes] > 0.5
    result = []
    for action_id, action_name in enumerate(ACTION_ORDER):
        selected = decision.issued & (decision.top_action == action_id)
        selected_positive = selected & group_target
        correct = selected & (action_target[row, decision.top_action] > 0)
        result.append(
            {
                "action_id": action_id,
                "action_family": action_name,
                "issued": int(selected.sum()),
                "issued_positive_groups": int(selected_positive.sum()),
                "correct": int(correct.sum()),
                "precision": float(correct.sum() / selected.sum())
                if selected.sum()
                else 0.0,
                "conditional_precision": float(
                    correct.sum() / selected_positive.sum()
                )
                if selected_positive.sum()
                else 0.0,
            }
        )
    return result


def _prediction_frame(
    arrays: TwoStageArrays,
    indexes: np.ndarray,
    direct_logits: np.ndarray,
    action_logits: np.ndarray,
    thresholds: ActionAwareThresholds,
    outer_fold: int,
) -> pd.DataFrame:
    stages = arrays.stages[indexes]
    decision = make_decisions(
        direct_logits,
        action_logits,
        arrays.action_mask[indexes],
        stages,
        thresholds,
    )
    row = np.arange(len(indexes))
    correct = decision.issued & (
        arrays.action_target[indexes][row, decision.top_action] > 0
    )
    frame = pd.DataFrame(
        {
            "group_id": arrays.group_ids[indexes],
            "base_record_id": arrays.learner_ids[indexes],
            "stage": stages,
            "outer_fold": int(outer_fold),
            "group_has_positive": arrays.group_target[indexes].astype(int),
            "direct_gate_logit": direct_logits,
            "direct_gate_probability": decision.direct_gate_probability,
            "action_any_probability": decision.action_any_probability,
            "joint_gate_probability": decision.joint_gate_probability,
            "top_action_index": decision.top_action,
            "top_action_family": [ACTION_ORDER[value] for value in decision.top_action],
            "top_action_probability": decision.top_probability,
            "top_action_margin": decision.top_margin,
            "issued": decision.issued.astype(int),
            "correct_top1": correct.astype(int),
        }
    )
    for action_id in range(ACTION_COUNT):
        frame[f"action_logit_{action_id}"] = action_logits[:, action_id]
        frame[f"action_mask_{action_id}"] = arrays.action_mask[
            indexes, action_id
        ].astype(int)
        frame[f"action_target_{action_id}"] = arrays.action_target[
            indexes, action_id
        ].astype(int)
    return frame


def _evaluate_outer_fold(
    outer_fold: int,
    selected: dict[str, Any],
    protocol: dict[str, Any],
    action_candidates: pd.DataFrame,
    feature_schema: dict[str, list[str]],
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    group_path = V3_CACHE / f"outer_{outer_fold}/GROUP_FEATURES.parquet"
    group_features = pd.read_parquet(group_path)
    arrays, schema = load_two_stage_arrays(group_features, action_candidates)
    if schema != feature_schema:
        raise RuntimeError("V4 outer-specific feature schema drift")
    train_indexes = np.where(arrays.outer_folds != outer_fold)[0]
    test_indexes = np.where(arrays.outer_folds == outer_fold)[0]
    scaler = fit_scaler(arrays, train_indexes)
    scaled = apply_scaler(arrays, scaler)
    direct_predictions = []
    action_predictions = []
    checkpoint_rows = []
    fold_dir = FINAL_OOF / f"fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for seed in protocol["evaluation"]["final_seeds"]:
        trained = _train_head(
            scaled,
            train_indexes,
            selected["config"],
            seed=int(seed),
            device=device,
            validation_indexes=None,
            fixed_epochs=int(selected["fixed_epoch"]),
            max_epochs=int(protocol["head_training"]["max_epochs"]),
            patience=int(protocol["head_training"]["patience"]),
            batch_size=int(protocol["head_training"]["batch_size"]),
        )
        direct, action = _predict_head(
            trained.state_dict,
            scaled,
            test_indexes,
            selected["config"],
            device=device,
            batch_size=int(protocol["head_training"]["batch_size"]),
        )
        direct_predictions.append(direct)
        action_predictions.append(action)
        checkpoint_path = fold_dir / f"head_seed{seed}.pt"
        torch.save(
            {
                "schema_version": "two_stage_v4_action_aware_head_v1",
                "outer_fold": int(outer_fold),
                "seed": int(seed),
                "state_dict": trained.state_dict,
                "head_config": selected["config"],
                "fixed_epoch": int(selected["fixed_epoch"]),
                "feature_scaler": scaler.to_dict(),
                "feature_schema": feature_schema,
                "thresholds": selected["thresholds"],
                "frozen_backbone_trainable": False,
                "candidate_binary_population": "ALL_VALID_CANDIDATES",
                "claim_boundary": protocol["claim_boundary"],
            },
            checkpoint_path,
        )
        checkpoint_rows.append(
            {
                "seed": int(seed),
                "path": str(checkpoint_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _sha256(checkpoint_path),
            }
        )
    direct_logits = np.mean(direct_predictions, axis=0)
    action_logits = np.mean(action_predictions, axis=0)
    thresholds = _thresholds_from_dict(selected["thresholds"])
    metrics = evaluate_action_aware(
        direct_gate_logits=direct_logits,
        action_logits=action_logits,
        action_mask=arrays.action_mask[test_indexes],
        group_target=arrays.group_target[test_indexes],
        action_target=arrays.action_target[test_indexes],
        stages=arrays.stages[test_indexes],
        thresholds=thresholds,
    )
    direct_probability, action_any, joint = blended_gate_probability(
        direct_logits,
        action_logits,
        arrays.action_mask[test_indexes],
        thresholds.direct_action_blend,
    )
    target = arrays.group_target[test_indexes]
    metrics["stage_a_discrimination"] = {
        "direct": _classification_metrics_from_probability(target, direct_probability),
        "action_derived": _classification_metrics_from_probability(target, action_any),
        "joint": _classification_metrics_from_probability(target, joint),
    }
    metrics["per_action"] = _per_action_metrics(
        arrays,
        test_indexes,
        direct_logits,
        action_logits,
        thresholds,
    )
    metrics["outer_fold"] = int(outer_fold)
    metrics["selected_config_id"] = selected["config"]["config_id"]
    metrics["thresholds"] = thresholds.to_dict()
    metrics["checkpoints"] = checkpoint_rows
    predictions = _prediction_frame(
        arrays,
        test_indexes,
        direct_logits,
        action_logits,
        thresholds,
        outer_fold,
    )
    _atomic_parquet(fold_dir / "predictions.parquet", predictions)
    _atomic_json(fold_dir / "metrics.json", metrics)
    _atomic_json(
        fold_dir / "selected.json",
        {
            **selected,
            "feature_scaler": scaler.to_dict(),
            "feature_schema": feature_schema,
            "checkpoints": checkpoint_rows,
        },
    )
    return predictions, metrics


def _aggregate_per_stage(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for stage in STAGE_ORDER:
        frame = oof[oof["stage"] == stage]
        issued = frame["issued"].to_numpy(dtype=bool)
        positive = frame["group_has_positive"].to_numpy(dtype=bool)
        correct = frame["correct_top1"].to_numpy(dtype=bool)
        issued_positive = issued & positive
        rows.append(
            {
                "stage": stage,
                "groups": int(len(frame)),
                "positive_groups": int(positive.sum()),
                "issued_groups": int(issued.sum()),
                "issued_positive_groups": int(issued_positive.sum()),
                "correct": int(correct.sum()),
                "end_to_end_precision_at_1": float(correct.sum() / issued.sum())
                if issued.sum()
                else 0.0,
                "positive_group_coverage": float(
                    issued_positive.sum() / positive.sum()
                )
                if positive.sum()
                else 0.0,
                "stage_a_precision": float(issued_positive.sum() / issued.sum())
                if issued.sum()
                else 0.0,
                "conditional_precision_at_1": float(
                    correct.sum() / issued_positive.sum()
                )
                if issued_positive.sum()
                else 0.0,
            }
        )
    return rows


def _aggregate_per_action(oof: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for action_id, action_name in enumerate(ACTION_ORDER):
        frame = oof[(oof["issued"] == 1) & (oof["top_action_index"] == action_id)]
        positive = frame["group_has_positive"].to_numpy(dtype=bool)
        correct = frame["correct_top1"].to_numpy(dtype=bool)
        rows.append(
            {
                "action_id": action_id,
                "action_family": action_name,
                "issued": int(len(frame)),
                "issued_positive_groups": int(positive.sum()),
                "correct": int(correct.sum()),
                "precision": float(correct.mean()) if len(frame) else 0.0,
                "conditional_precision": float(correct.sum() / positive.sum())
                if positive.sum()
                else 0.0,
            }
        )
    return rows


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_V4_HEAD_TRAINING":
        raise RuntimeError("two-stage V4 protocol is not frozen")
    registry_path = V3_CACHE / "CACHE_REGISTRY.json"
    if not registry_path.exists():
        raise RuntimeError("V3 embedding cache authority is missing")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("status") != "COMPLETE" or registry.get("backbone_trainable"):
        raise RuntimeError("V3 embedding cache authority is invalid")
    action_candidates = pd.read_parquet(V3_CACHE / "ACTION_CANDIDATES.parquet")
    cross_features = pd.read_parquet(
        V3_CACHE / "cross_fitted/GROUP_FEATURES.parquet"
    )
    cross_arrays, feature_schema = load_two_stage_arrays(
        cross_features,
        action_candidates,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SELECTION.mkdir(parents=True, exist_ok=True)
    FINAL_OOF.mkdir(parents=True, exist_ok=True)

    selected_by_fold: dict[str, Any] = {}
    for outer_fold in protocol["evaluation"]["outer_folds"]:
        selected, trials = _run_inner_selection(
            cross_arrays,
            int(outer_fold),
            protocol,
            device,
        )
        trials.to_csv(
            MODEL_SELECTION / f"fold_{outer_fold}_trials.csv",
            index=False,
        )
        _atomic_json(
            MODEL_SELECTION / f"fold_{outer_fold}_selected.json",
            selected,
        )
        selected_by_fold[str(outer_fold)] = selected

    oof_rows = []
    fold_metrics = []
    for outer_fold in protocol["evaluation"]["outer_folds"]:
        predictions, metrics = _evaluate_outer_fold(
            int(outer_fold),
            selected_by_fold[str(outer_fold)],
            protocol,
            action_candidates,
            feature_schema,
            device,
        )
        oof_rows.append(predictions)
        fold_metrics.append(metrics)
    oof = pd.concat(oof_rows, ignore_index=True).sort_values(
        "group_id",
        kind="stable",
    )
    _atomic_parquet(FINAL_OOF / "OOF_PREDICTIONS.parquet", oof)

    issued = oof["issued"].to_numpy(dtype=bool)
    correct = oof["correct_top1"].to_numpy(dtype=bool)
    positive = oof["group_has_positive"].to_numpy(dtype=bool)
    issued_positive = issued & positive
    action_logits = oof[
        [f"action_logit_{index}" for index in range(ACTION_COUNT)]
    ].to_numpy(dtype=np.float32)
    action_mask = oof[
        [f"action_mask_{index}" for index in range(ACTION_COUNT)]
    ].to_numpy(dtype=bool)
    action_target = oof[
        [f"action_target_{index}" for index in range(ACTION_COUNT)]
    ].to_numpy(dtype=np.float32)
    ranking = ranking_metrics(action_logits, action_target, action_mask, positive)
    overall = {
        "groups": int(len(oof)),
        "learners": int(oof["base_record_id"].nunique()),
        "positive_groups": int(positive.sum()),
        "issued_groups": int(issued.sum()),
        "issued_positive_groups": int(issued_positive.sum()),
        "correct_issued_actions": int(correct.sum()),
        "false_issue_groups": int((issued & ~positive).sum()),
        "stage_a_precision": float(issued_positive.sum() / issued.sum())
        if issued.sum()
        else 0.0,
        "stage_a_recall": float(issued_positive.sum() / positive.sum())
        if positive.sum()
        else 0.0,
        "stage_b_conditional_precision_at_1": float(
            correct.sum() / issued_positive.sum()
        )
        if issued_positive.sum()
        else 0.0,
        "end_to_end_precision_at_1": float(correct.sum() / issued.sum())
        if issued.sum()
        else 0.0,
        "positive_group_coverage": float(issued_positive.sum() / positive.sum())
        if positive.sum()
        else 0.0,
        "abstention_rate": 1.0 - float(issued.sum() / len(oof)),
        "action_diversity": int(oof.loc[issued, "top_action_family"].nunique())
        if issued.sum()
        else 0,
        "top_action_concentration": float(
            oof.loc[issued, "top_action_family"].value_counts(normalize=True).max()
        )
        if issued.sum()
        else 1.0,
        **ranking,
        "stage_a_discrimination": {
            "direct": _classification_metrics_from_probability(
                positive,
                oof["direct_gate_probability"].to_numpy(dtype=np.float64),
            ),
            "action_derived": _classification_metrics_from_probability(
                positive,
                oof["action_any_probability"].to_numpy(dtype=np.float64),
            ),
            "joint": _classification_metrics_from_probability(
                positive,
                oof["joint_gate_probability"].to_numpy(dtype=np.float64),
            ),
        },
        "per_stage": _aggregate_per_stage(oof),
        "per_action": _aggregate_per_action(oof),
    }
    results = {
        "schema_version": "two_stage_v4_nested_oof_v1",
        "status": "COMPLETE",
        "architecture": "frozen_residual_cnn_bilstm_plus_action_aware_integrated_heads",
        "external_ml_ranker": False,
        "frozen_backbone_trainable": False,
        "candidate_binary_population": "ALL_VALID_CANDIDATES",
        "registered_config_count": len(
            protocol["head_training"]["registered_configs"]
        ),
        "final_seeds": list(protocol["evaluation"]["final_seeds"]),
        "overall": overall,
        "folds": fold_metrics,
        "selected_by_fold": selected_by_fold,
        "feature_schema": feature_schema,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "cache_registry_sha256": _sha256(registry_path),
        "v3_release_sha256": _sha256(
            ROOT
            / "artifacts/recommend_hybrid/two_stage_v3/TWO_STAGE_V3_RELEASE.json"
        ),
        "claim_boundary": protocol["claim_boundary"],
        "device": str(device),
    }
    _atomic_json(FINAL_OOF / "NESTED_OOF_RESULTS.json", results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
