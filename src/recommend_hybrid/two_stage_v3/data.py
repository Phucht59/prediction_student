"""Grouped tensor assembly for integrated Two-Stage V3 heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model import ACTION_COUNT

STAGE_ORDER = ("EARLY_20", "EARLY_35", "MIDDLE_50")
STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}
GROUP_SCALAR_COLUMNS = (
    "candidate_count",
    "maximum_risk_reduction",
    "mean_risk_reduction",
    "maximum_deficit",
    "mean_evidence_strength",
    "maximum_evidence_strength",
    "top_counterfactual_margin",
    "risk_probability",
    "risk_entropy",
    "seed_disagreement",
    "risk_confidence",
)
ACTION_FEATURE_COLUMNS = (
    "risk_reduction",
    "risk_uncertainty",
    "evidence_strength",
    "deficit_score",
    "opportunity_count",
    "workload_minutes",
    "action_available",
    "prerequisite_status",
)


@dataclass(frozen=True)
class FeatureScaler:
    group_mean: np.ndarray
    group_scale: np.ndarray
    action_mean: np.ndarray
    action_scale: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_mean": self.group_mean.tolist(),
            "group_scale": self.group_scale.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_scale": self.action_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeatureScaler":
        return cls(
            group_mean=np.asarray(payload["group_mean"], dtype=np.float32),
            group_scale=np.asarray(payload["group_scale"], dtype=np.float32),
            action_mean=np.asarray(payload["action_mean"], dtype=np.float32),
            action_scale=np.asarray(payload["action_scale"], dtype=np.float32),
        )


@dataclass(frozen=True)
class TwoStageArrays:
    group_ids: np.ndarray
    learner_ids: np.ndarray
    stages: np.ndarray
    outer_folds: np.ndarray
    group_features: np.ndarray
    action_features: np.ndarray
    action_ids: np.ndarray
    action_mask: np.ndarray
    group_target: np.ndarray
    action_target: np.ndarray

    def subset(self, indexes: np.ndarray) -> "TwoStageArrays":
        return TwoStageArrays(
            group_ids=self.group_ids[indexes],
            learner_ids=self.learner_ids[indexes],
            stages=self.stages[indexes],
            outer_folds=self.outer_folds[indexes],
            group_features=self.group_features[indexes],
            action_features=self.action_features[indexes],
            action_ids=self.action_ids[indexes],
            action_mask=self.action_mask[indexes],
            group_target=self.group_target[indexes],
            action_target=self.action_target[indexes],
        )

    @property
    def size(self) -> int:
        return len(self.group_ids)


def _finite(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def load_two_stage_arrays(
    group_features: pd.DataFrame,
    action_candidates: pd.DataFrame,
) -> tuple[TwoStageArrays, dict[str, list[str]]]:
    groups = group_features.sort_values("group_id", kind="stable").reset_index(drop=True)
    if groups["group_id"].duplicated().any():
        raise RuntimeError("group feature cache contains duplicate group_id")
    student_columns = sorted(
        column for column in groups.columns if column.startswith("student_state_")
    )
    tabular_columns = sorted(
        column for column in groups.columns if column.startswith("tabular_expert_")
    )
    if len(student_columns) != 64 or len(tabular_columns) != 32:
        raise RuntimeError("frozen hybrid embedding dimensions are not 64 + 32")
    missing_group = set(GROUP_SCALAR_COLUMNS) - set(groups.columns)
    if missing_group:
        raise RuntimeError(f"group cache missing columns: {sorted(missing_group)}")
    continuous_columns = [
        *student_columns,
        *tabular_columns,
        *GROUP_SCALAR_COLUMNS,
    ]
    continuous = _finite(groups, continuous_columns)
    stage_one_hot = np.zeros((len(groups), len(STAGE_ORDER)), dtype=np.float32)
    for row, stage in enumerate(groups["stage"].astype(str)):
        if stage not in STAGE_INDEX:
            raise RuntimeError(f"unknown stage {stage}")
        stage_one_hot[row, STAGE_INDEX[stage]] = 1.0
    group_matrix = np.concatenate([continuous, stage_one_hot], axis=1)

    group_map = {
        value: index for index, value in enumerate(groups["group_id"].astype(str))
    }
    action_matrix = np.zeros(
        (len(groups), ACTION_COUNT, len(ACTION_FEATURE_COLUMNS)), dtype=np.float32
    )
    action_ids = np.tile(np.arange(ACTION_COUNT, dtype=np.int64), (len(groups), 1))
    action_mask = np.zeros((len(groups), ACTION_COUNT), dtype=bool)
    action_target = np.zeros((len(groups), ACTION_COUNT), dtype=np.float32)
    duplicate = action_candidates.duplicated(["group_id", "action_index"])
    if duplicate.any():
        raise RuntimeError("action candidates contain duplicate group/action rows")
    missing_action = set(ACTION_FEATURE_COLUMNS) - set(action_candidates.columns)
    if missing_action:
        raise RuntimeError(f"action candidates missing columns: {sorted(missing_action)}")
    for row in action_candidates.itertuples(index=False):
        group_index = group_map.get(str(row.group_id))
        if group_index is None:
            raise RuntimeError(f"action candidate references unknown group {row.group_id}")
        action_index = int(row.action_index)
        if not 0 <= action_index < ACTION_COUNT:
            raise RuntimeError(f"invalid action index {action_index}")
        values = [float(getattr(row, column)) for column in ACTION_FEATURE_COLUMNS]
        action_matrix[group_index, action_index] = np.nan_to_num(
            np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        action_mask[group_index, action_index] = bool(
            int(row.action_available) == 1 and int(row.prerequisite_status) == 1
        )
        action_target[group_index, action_index] = float(row.silver_positive)

    group_target = groups["group_has_positive"].to_numpy(dtype=np.float32)
    if not np.array_equal(group_target, (action_target.max(axis=1) > 0).astype(np.float32)):
        raise RuntimeError("group and action targets are inconsistent")
    if np.any(action_mask.sum(axis=1) < 2):
        raise RuntimeError("rankable groups must contain at least two candidates")
    arrays = TwoStageArrays(
        group_ids=groups["group_id"].astype(str).to_numpy(dtype=object),
        learner_ids=groups["base_record_id"].astype(str).to_numpy(dtype=object),
        stages=groups["stage"].astype(str).to_numpy(dtype=object),
        outer_folds=groups["outer_fold"].to_numpy(dtype=np.int64),
        group_features=group_matrix.astype(np.float32),
        action_features=action_matrix,
        action_ids=action_ids,
        action_mask=action_mask,
        group_target=group_target,
        action_target=action_target,
    )
    schema = {
        "group_continuous": continuous_columns,
        "group_stage_one_hot": list(STAGE_ORDER),
        "action_continuous": list(ACTION_FEATURE_COLUMNS),
    }
    return arrays, schema


def fit_scaler(arrays: TwoStageArrays, train_indexes: np.ndarray) -> FeatureScaler:
    group = arrays.group_features[train_indexes]
    group_mean = group.mean(axis=0)
    group_scale = group.std(axis=0)
    # Stage one-hot columns should remain interpretable and unscaled.
    group_mean[-len(STAGE_ORDER) :] = 0.0
    group_scale[-len(STAGE_ORDER) :] = 1.0
    group_scale[group_scale < 1.0e-6] = 1.0

    actions = arrays.action_features[train_indexes]
    mask = arrays.action_mask[train_indexes]
    flattened = actions[mask]
    action_mean = flattened.mean(axis=0)
    action_scale = flattened.std(axis=0)
    action_scale[action_scale < 1.0e-6] = 1.0
    return FeatureScaler(
        group_mean=group_mean.astype(np.float32),
        group_scale=group_scale.astype(np.float32),
        action_mean=action_mean.astype(np.float32),
        action_scale=action_scale.astype(np.float32),
    )


def apply_scaler(arrays: TwoStageArrays, scaler: FeatureScaler) -> TwoStageArrays:
    group = (arrays.group_features - scaler.group_mean) / scaler.group_scale
    actions = (arrays.action_features - scaler.action_mean) / scaler.action_scale
    actions = np.where(arrays.action_mask[:, :, None], actions, 0.0)
    return TwoStageArrays(
        group_ids=arrays.group_ids,
        learner_ids=arrays.learner_ids,
        stages=arrays.stages,
        outer_folds=arrays.outer_folds,
        group_features=np.nan_to_num(group).astype(np.float32),
        action_features=np.nan_to_num(actions).astype(np.float32),
        action_ids=arrays.action_ids,
        action_mask=arrays.action_mask,
        group_target=arrays.group_target,
        action_target=arrays.action_target,
    )


__all__ = [
    "ACTION_FEATURE_COLUMNS",
    "FeatureScaler",
    "GROUP_SCALAR_COLUMNS",
    "STAGE_ORDER",
    "TwoStageArrays",
    "apply_scaler",
    "fit_scaler",
    "load_two_stage_arrays",
]
