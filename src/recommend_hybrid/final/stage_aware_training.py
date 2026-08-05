"""Four-stage training utilities for the integrated conditional action head.

The frozen Hybrid representation is never updated.  Scientific silver labels
are merged with cutoff-safe OULAD landmark rows, grouped by real learner, and
used to train only ``HybridActionAwareRecommendationHeads``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

from src.recommend_hybrid.weak_supervision.registry import load_action_mappings

from .actions import ACTION_COUNT, ACTION_INDEX, ACTION_ORDER, canonical_action_id
from .metrics import (
    ActionAwareThresholds,
    STAGE_ORDER,
    evaluate_action_aware,
    ranking_metrics,
)
from .model import (
    ActionAwareHeadConfig,
    HybridActionAwareRecommendationHeads,
    action_aware_loss,
)

ACTION_SOURCE_ID: Mapping[str, str] = {
    "ASSESSMENT_COMPLETION": "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY": "STUDY_SCHEDULE",
    "VLE_ENGAGEMENT": "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE": "RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW": "LEARNING_CONSOLIDATION",
}
PRIORITY_SCORE = {"LOW": 0.25, "MEDIUM": 0.50, "HIGH": 0.75, "CRITICAL": 1.0}


@dataclass(frozen=True)
class FourStageActionData:
    group_features: np.ndarray
    action_features: np.ndarray
    action_ids: np.ndarray
    action_mask: np.ndarray
    group_target: np.ndarray
    action_target: np.ndarray
    student_ids: np.ndarray
    record_ids: np.ndarray
    stages: np.ndarray
    group_feature_names: tuple[str, ...]
    action_feature_names: tuple[str, ...]

    def validate(self) -> None:
        group = np.asarray(self.group_features)
        action = np.asarray(self.action_features)
        mask = np.asarray(self.action_mask)
        group_target = np.asarray(self.group_target).reshape(-1)
        action_target = np.asarray(self.action_target)
        row_count = len(group_target)
        if group.ndim != 2 or len(group) != row_count:
            raise ValueError("group features must be [N, G]")
        if action.ndim != 3 or action.shape[:2] != (row_count, ACTION_COUNT):
            raise ValueError("action features must be [N, 5, F]")
        if np.asarray(self.action_ids).shape != (row_count, ACTION_COUNT):
            raise ValueError("action IDs must be [N, 5]")
        if mask.shape != (row_count, ACTION_COUNT):
            raise ValueError("action mask must be [N, 5]")
        if action_target.shape != (row_count, ACTION_COUNT):
            raise ValueError("action target must be [N, 5]")
        if not np.isfinite(group).all() or not np.isfinite(action).all():
            raise ValueError("ranker features must be finite")
        if not np.isin(group_target, [0, 1]).all():
            raise ValueError("group target must be binary")
        if not np.isin(action_target, [0, 1]).all():
            raise ValueError("action target must be binary")
        if not np.all(action_target[~mask.astype(bool)] == 0):
            raise ValueError("masked actions cannot carry a positive target")
        if not np.array_equal(group_target, (action_target.sum(axis=1) > 0).astype(np.int8)):
            raise ValueError("group target must equal any positive action")
        for values, name in (
            (self.student_ids, "student_ids"),
            (self.record_ids, "record_ids"),
            (self.stages, "stages"),
        ):
            if len(np.asarray(values).reshape(-1)) != row_count:
                raise ValueError(f"{name} must align with rows")
        if sorted(set(map(str, self.stages))) != sorted(STAGE_ORDER):
            raise ValueError("all four stages must be represented")
        if len(np.unique(np.asarray(self.record_ids).astype(str) + "::" + np.asarray(self.stages).astype(str))) != row_count:
            raise ValueError("record-stage groups must be unique")

    def subset(self, index: np.ndarray) -> "FourStageActionData":
        selected = np.asarray(index, dtype=np.int64)
        return FourStageActionData(
            group_features=self.group_features[selected],
            action_features=self.action_features[selected],
            action_ids=self.action_ids[selected],
            action_mask=self.action_mask[selected],
            group_target=self.group_target[selected],
            action_target=self.action_target[selected],
            student_ids=self.student_ids[selected],
            record_ids=self.record_ids[selected],
            stages=self.stages[selected],
            group_feature_names=self.group_feature_names,
            action_feature_names=self.action_feature_names,
        )


@dataclass(frozen=True)
class FeatureStandardizer:
    group_mean: np.ndarray
    group_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray

    @classmethod
    def fit(cls, data: FourStageActionData) -> "FeatureStandardizer":
        group_mean = data.group_features.mean(axis=0, dtype=np.float64)
        group_scale = data.group_features.std(axis=0, dtype=np.float64)
        group_scale = np.where(group_scale < 1.0e-6, 1.0, group_scale)
        valid = data.action_mask.astype(bool)
        values = data.action_features[valid]
        if not len(values):
            raise ValueError("at least one valid training action is required")
        action_mean = values.mean(axis=0, dtype=np.float64)
        action_scale = values.std(axis=0, dtype=np.float64)
        action_scale = np.where(action_scale < 1.0e-6, 1.0, action_scale)
        return cls(group_mean, group_scale, action_mean, action_scale)

    def transform(self, data: FourStageActionData) -> FourStageActionData:
        group = ((data.group_features - self.group_mean) / self.group_scale).astype(np.float32)
        action = ((data.action_features - self.action_mean) / self.action_scale).astype(np.float32)
        action[~data.action_mask.astype(bool)] = 0.0
        return FourStageActionData(
            group_features=group,
            action_features=action,
            action_ids=data.action_ids,
            action_mask=data.action_mask,
            group_target=data.group_target,
            action_target=data.action_target,
            student_ids=data.student_ids,
            record_ids=data.record_ids,
            stages=data.stages,
            group_feature_names=data.group_feature_names,
            action_feature_names=data.action_feature_names,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "group_mean": self.group_mean.tolist(),
            "group_scale": self.group_scale.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_scale": self.action_scale.tolist(),
        }


def _canonical_silver_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "dataset",
        "student_key",
        "stage",
        "action_id",
        "silver_label",
        "silver_status",
        "silver_confidence",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"silver-label artifact is missing columns: {missing}")
    selected = frame.loc[
        frame["dataset"].eq("oulad")
        & frame["stage"].isin(STAGE_ORDER)
        & frame["silver_status"].eq("RETAINED")
    ].copy()
    canonical: list[str | None] = []
    for action_id in selected["action_id"]:
        try:
            canonical.append(canonical_action_id(action_id))
        except ValueError:
            canonical.append(None)
    selected["canonical_action_id"] = canonical
    selected = selected.loc[selected["canonical_action_id"].notna()].copy()
    selected["student_key"] = selected["student_key"].astype(str)
    selected["silver_label"] = selected["silver_label"].astype(int)
    selected["silver_confidence"] = pd.to_numeric(
        selected["silver_confidence"], errors="coerce"
    ).fillna(0.0)
    selected = selected.sort_values(
        ["student_key", "stage", "canonical_action_id", "silver_confidence", "action_id"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).drop_duplicates(["student_key", "stage", "canonical_action_id"], keep="first")
    return selected


def build_four_stage_action_data(
    *,
    landmark_path: Path,
    silver_label_path: Path,
    action_map_path: Path,
) -> FourStageActionData:
    landmark = pd.read_parquet(landmark_path)
    silver = _canonical_silver_rows(pd.read_parquet(silver_label_path))
    required = {
        "record_id",
        "student_id",
        "stage",
        "action_id",
        "baseline_measure",
    }
    missing = sorted(required.difference(landmark.columns))
    if missing:
        raise KeyError(f"landmark artifact is missing columns: {missing}")
    landmark = landmark.loc[
        landmark["stage"].isin(STAGE_ORDER)
        & landmark["action_id"].isin(ACTION_ORDER)
    ].copy()
    landmark["record_id"] = landmark["record_id"].astype(str)
    landmark["student_id"] = landmark["student_id"].astype(str)
    if landmark.duplicated(["record_id", "stage", "action_id"]).any():
        raise ValueError("landmark actions must be unique by record-stage-action")

    embedding_columns = sorted(
        column for column in landmark.columns if column.startswith("embedding__")
    )
    scalar_columns = sorted(
        column
        for column in landmark.columns
        if column.startswith("feature__")
        and not column.startswith("feature__student_")
        and not column.startswith("feature__tabular_")
    )
    if not embedding_columns:
        raise ValueError("frozen Hybrid embedding columns are required")
    group_columns = embedding_columns + scalar_columns
    group_frame = landmark.drop_duplicates(["record_id", "stage"]).copy()
    if group_frame.groupby("student_id")["record_id"].nunique().max() < 1:
        raise ValueError("student identity is unavailable")

    mappings = {
        row.action_id: row
        for row in load_action_mappings(action_map_path)
    }
    metadata: dict[str, tuple[float, float, float]] = {}
    for action_id, source_id in ACTION_SOURCE_ID.items():
        source = mappings[source_id]
        metadata[action_id] = (
            float(source.estimated_minutes) / 180.0,
            PRIORITY_SCORE.get(str(source.default_priority), 0.5),
            float(source.human_review_required),
        )

    landmark_lookup = {
        (str(row.record_id), str(row.stage), str(row.action_id)): row
        for row in landmark.itertuples(index=False)
    }
    silver_lookup = {
        (str(row.student_key), str(row.stage), str(row.canonical_action_id)): row
        for row in silver.itertuples(index=False)
    }
    group_features: list[np.ndarray] = []
    action_features: list[np.ndarray] = []
    action_masks: list[np.ndarray] = []
    action_targets: list[np.ndarray] = []
    students: list[str] = []
    records: list[str] = []
    stages: list[str] = []
    stage_index = {stage: index for index, stage in enumerate(STAGE_ORDER)}

    for row in group_frame.itertuples(index=False):
        record_id = str(row.record_id)
        stage = str(row.stage)
        fixed = np.asarray([float(getattr(row, column)) for column in group_columns], dtype=np.float32)
        stage_one_hot = np.zeros(len(STAGE_ORDER), dtype=np.float32)
        stage_one_hot[stage_index[stage]] = 1.0
        slots = np.zeros((ACTION_COUNT, 6), dtype=np.float32)
        mask = np.zeros(ACTION_COUNT, dtype=bool)
        target = np.zeros(ACTION_COUNT, dtype=np.int8)
        for action_id in ACTION_ORDER:
            index = ACTION_INDEX[action_id]
            landmark_row = landmark_lookup.get((record_id, stage, action_id))
            label_row = silver_lookup.get((record_id, stage, action_id))
            if landmark_row is None or label_row is None:
                continue
            baseline = float(landmark_row.baseline_measure)
            minutes, priority, human = metadata[action_id]
            slots[index] = np.asarray(
                [baseline, 1.0 - baseline, minutes, priority, human, 1.0],
                dtype=np.float32,
            )
            mask[index] = True
            target[index] = int(int(label_row.silver_label) >= 1)
        if not mask.any():
            continue
        group_features.append(np.concatenate([fixed, stage_one_hot]))
        action_features.append(slots)
        action_masks.append(mask)
        action_targets.append(target)
        students.append(str(row.student_id))
        records.append(record_id)
        stages.append(stage)

    if not group_features:
        raise ValueError("no retained four-stage action groups were constructed")
    action_target_array = np.asarray(action_targets, dtype=np.int8)
    data = FourStageActionData(
        group_features=np.asarray(group_features, dtype=np.float32),
        action_features=np.asarray(action_features, dtype=np.float32),
        action_ids=np.tile(np.arange(ACTION_COUNT, dtype=np.int64), (len(group_features), 1)),
        action_mask=np.asarray(action_masks, dtype=bool),
        group_target=(action_target_array.sum(axis=1) > 0).astype(np.int8),
        action_target=action_target_array,
        student_ids=np.asarray(students, dtype=str),
        record_ids=np.asarray(records, dtype=str),
        stages=np.asarray(stages, dtype=str),
        group_feature_names=tuple(group_columns) + tuple(
            f"stage__{stage}" for stage in STAGE_ORDER
        ),
        action_feature_names=(
            "baseline_measure",
            "issue_severity",
            "estimated_minutes_fraction",
            "default_priority",
            "human_review_required",
            "evidence_available",
        ),
    )
    data.validate()
    return data


@dataclass(frozen=True)
class RankerSplit:
    outer_fold: int
    train_index: np.ndarray
    validation_index: np.ndarray
    test_index: np.ndarray


def grouped_outer_splits(
    data: FourStageActionData,
    *,
    n_splits: int = 3,
    random_state: int = 20260806,
) -> list[RankerSplit]:
    outer = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    result: list[RankerSplit] = []
    for fold, (train_validation, test) in enumerate(
        outer.split(data.group_features, data.group_target, groups=data.student_ids)
    ):
        inner = StratifiedGroupKFold(
            n_splits=4,
            shuffle=True,
            random_state=random_state + fold + 1,
        )
        inner_train, inner_validation = next(
            inner.split(
                data.group_features[train_validation],
                data.group_target[train_validation],
                groups=data.student_ids[train_validation],
            )
        )
        train = train_validation[inner_train]
        validation = train_validation[inner_validation]
        for left, right, name in (
            (train, validation, "train/validation"),
            (train, test, "train/test"),
            (validation, test, "validation/test"),
        ):
            overlap = set(data.student_ids[left]).intersection(data.student_ids[right])
            if overlap:
                raise RuntimeError(f"student leakage detected in {name}")
        result.append(RankerSplit(fold, train, validation, test))
    return result


def positive_weights(data: FourStageActionData) -> tuple[float, np.ndarray]:
    group_positive = int(data.group_target.sum())
    group_negative = int(len(data.group_target) - group_positive)
    group_weight = float(group_negative / max(group_positive, 1))
    action_weight = np.ones(ACTION_COUNT, dtype=np.float32)
    for action_id, index in ACTION_INDEX.items():
        valid = data.action_mask[:, index]
        positive = int(data.action_target[valid, index].sum())
        negative = int(valid.sum() - positive)
        action_weight[index] = float(negative / max(positive, 1))
    return max(group_weight, 1.0e-3), np.maximum(action_weight, 1.0e-3)


def _tensor_dataset(data: FourStageActionData) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(data.group_features.astype(np.float32)),
        torch.from_numpy(data.action_features.astype(np.float32)),
        torch.from_numpy(data.action_ids.astype(np.int64)),
        torch.from_numpy(data.action_mask.astype(bool)),
        torch.from_numpy(data.group_target.astype(np.float32)),
        torch.from_numpy(data.action_target.astype(np.float32)),
    )


def predict_action_head(
    model: HybridActionAwareRecommendationHeads,
    data: FourStageActionData,
    *,
    batch_size: int = 2048,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    model.eval()
    direct: list[np.ndarray] = []
    action: list[np.ndarray] = []
    loader = DataLoader(_tensor_dataset(data), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for group, action_features, action_ids, mask, _, _ in loader:
            output = model(
                group.to(device),
                action_features.to(device),
                action_ids.to(device),
                mask.to(device),
            )
            direct.append(output.direct_gate_logit.cpu().numpy())
            action.append(output.action_logits.cpu().numpy())
    return np.concatenate(direct), np.concatenate(action)


def _validation_score(
    direct_logits: np.ndarray,
    action_logits: np.ndarray,
    data: FourStageActionData,
) -> float:
    ranking = ranking_metrics(
        action_logits,
        data.action_target,
        data.action_mask,
        data.group_target,
    )
    if len(np.unique(data.group_target)) == 2:
        gate_auc = float(roc_auc_score(data.group_target, direct_logits))
    else:
        gate_auc = 0.5
    return float(
        0.40 * ranking["conditional_precision_at_1_all_positive"]
        + 0.25 * ranking["ndcg_at_3"]
        + 0.20 * ranking["mrr"]
        + 0.15 * gate_auc
    )


@dataclass(frozen=True)
class FittedActionHead:
    model: HybridActionAwareRecommendationHeads
    best_epoch: int
    validation_score: float
    history: tuple[dict[str, float], ...]


def train_action_head(
    train_data: FourStageActionData,
    validation_data: FourStageActionData,
    *,
    config: ActionAwareHeadConfig,
    seed: int,
    epochs: int = 50,
    patience: int = 8,
    batch_size: int = 512,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    device: str | torch.device = "cpu",
) -> FittedActionHead:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = HybridActionAwareRecommendationHeads(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    group_weight, action_weight = positive_weights(train_data)
    group_weight_tensor = torch.tensor(group_weight, dtype=torch.float32, device=device)
    action_weight_tensor = torch.tensor(action_weight, dtype=torch.float32, device=device)
    loader = DataLoader(
        _tensor_dataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -np.inf
    best_epoch = -1
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for group, action_features, action_ids, mask, group_target, action_target in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                group.to(device),
                action_features.to(device),
                action_ids.to(device),
                mask.to(device),
            )
            loss, _ = action_aware_loss(
                output,
                group_target=group_target.to(device),
                action_target=action_target.to(device),
                action_mask=mask.to(device),
                group_positive_weight=group_weight_tensor,
                action_positive_weight=action_weight_tensor,
                config=config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        direct, action = predict_action_head(
            model,
            validation_data,
            batch_size=max(batch_size, 1024),
            device=device,
        )
        score = _validation_score(direct, action, validation_data)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(losses)),
                "validation_score": score,
            }
        )
        if score > best_score + 1.0e-6:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("action-head training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu()
    return FittedActionHead(
        model=model,
        best_epoch=best_epoch,
        validation_score=float(best_score),
        history=tuple(history),
    )


def _decision_objective(metrics: Mapping[str, object]) -> tuple[float, ...]:
    issued = int(metrics["issued_groups"])
    positive = int(metrics["positive_groups"])
    false_issue_rate = float(metrics["false_issue_groups"]) / max(issued, 1)
    coverage = float(metrics["positive_group_coverage"])
    precision = float(metrics["end_to_end_precision_at_1"])
    stage_precision = float(metrics["stage_a_precision"])
    diversity = float(metrics["action_diversity"]) / ACTION_COUNT
    support_gate = float(issued >= max(10, int(0.05 * max(positive, 1))))
    return (
        support_gate,
        precision - 0.10 * false_issue_rate,
        stage_precision,
        coverage,
        diversity,
        -float(metrics["top_action_concentration"]),
    )


def calibrate_action_thresholds(
    *,
    direct_logits: np.ndarray,
    action_logits: np.ndarray,
    data: FourStageActionData,
) -> ActionAwareThresholds:
    blend_values = (0.25, 0.50, 0.75)
    action_probability_values = (0.30, 0.40, 0.50, 0.60)
    margin_values = (0.00, 0.05, 0.10, 0.15)
    gate_values = (0.30, 0.40, 0.50, 0.60, 0.70)
    best_thresholds: ActionAwareThresholds | None = None
    best_objective: tuple[float, ...] | None = None
    for blend in blend_values:
        for minimum_action_probability in action_probability_values:
            for margin in margin_values:
                stage_thresholds: list[float] = []
                for stage in STAGE_ORDER:
                    selected = data.stages == stage
                    stage_best = gate_values[0]
                    stage_objective: tuple[float, ...] | None = None
                    for gate in gate_values:
                        thresholds = ActionAwareThresholds(
                            stage_gate_probability=tuple(
                                gate if name == stage else 0.50 for name in STAGE_ORDER
                            ),
                            direct_action_blend=blend,
                            minimum_action_probability=minimum_action_probability,
                            minimum_action_margin=margin,
                        )
                        metrics = evaluate_action_aware(
                            direct_gate_logits=direct_logits[selected],
                            action_logits=action_logits[selected],
                            action_mask=data.action_mask[selected],
                            group_target=data.group_target[selected],
                            action_target=data.action_target[selected],
                            stages=data.stages[selected],
                            thresholds=thresholds,
                            include_breakdown=False,
                        )
                        objective = _decision_objective(metrics)
                        if stage_objective is None or objective > stage_objective:
                            stage_objective = objective
                            stage_best = gate
                    stage_thresholds.append(stage_best)
                thresholds = ActionAwareThresholds(
                    stage_gate_probability=tuple(stage_thresholds),
                    direct_action_blend=blend,
                    minimum_action_probability=minimum_action_probability,
                    minimum_action_margin=margin,
                )
                metrics = evaluate_action_aware(
                    direct_gate_logits=direct_logits,
                    action_logits=action_logits,
                    action_mask=data.action_mask,
                    group_target=data.group_target,
                    action_target=data.action_target,
                    stages=data.stages,
                    thresholds=thresholds,
                )
                per_stage_precision = [
                    float(row["end_to_end_precision_at_1"])
                    for row in metrics.get("per_stage", [])
                ]
                objective = (
                    min(per_stage_precision) if per_stage_precision else 0.0,
                    *_decision_objective(metrics),
                )
                if best_objective is None or objective > best_objective:
                    best_objective = objective
                    best_thresholds = thresholds
    if best_thresholds is None:
        raise RuntimeError("threshold calibration produced no candidate")

    action_thresholds = list(best_thresholds.action_probability_by_id)
    for action_id, action_index in ACTION_INDEX.items():
        selected_best = action_thresholds[action_index]
        selected_objective: tuple[float, ...] | None = None
        for candidate in (0.0, 0.30, 0.40, 0.50, 0.60, 0.70):
            values = list(action_thresholds)
            values[action_index] = candidate
            thresholds = ActionAwareThresholds(
                stage_gate_probability=best_thresholds.stage_gate_probability,
                direct_action_blend=best_thresholds.direct_action_blend,
                minimum_action_probability=best_thresholds.minimum_action_probability,
                minimum_action_margin=best_thresholds.minimum_action_margin,
                action_probability_by_id=tuple(values),
            )
            metrics = evaluate_action_aware(
                direct_gate_logits=direct_logits,
                action_logits=action_logits,
                action_mask=data.action_mask,
                group_target=data.group_target,
                action_target=data.action_target,
                stages=data.stages,
                thresholds=thresholds,
            )
            objective = _decision_objective(metrics)
            if selected_objective is None or objective > selected_objective:
                selected_objective = objective
                selected_best = candidate
        action_thresholds[action_index] = selected_best
    return ActionAwareThresholds(
        stage_gate_probability=best_thresholds.stage_gate_probability,
        direct_action_blend=best_thresholds.direct_action_blend,
        minimum_action_probability=best_thresholds.minimum_action_probability,
        minimum_action_margin=best_thresholds.minimum_action_margin,
        action_probability_by_id=tuple(action_thresholds),
    )


def average_logits(logit_pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pairs = list(logit_pairs)
    if not pairs:
        raise ValueError("at least one model prediction is required")
    direct = np.mean(np.stack([pair[0] for pair in pairs], axis=0), axis=0)
    action = np.mean(np.stack([pair[1] for pair in pairs], axis=0), axis=0)
    return direct.astype(np.float64), action.astype(np.float64)


__all__ = [
    "ACTION_SOURCE_ID",
    "FeatureStandardizer",
    "FittedActionHead",
    "FourStageActionData",
    "RankerSplit",
    "average_logits",
    "build_four_stage_action_data",
    "calibrate_action_thresholds",
    "grouped_outer_splits",
    "positive_weights",
    "predict_action_head",
    "train_action_head",
]
