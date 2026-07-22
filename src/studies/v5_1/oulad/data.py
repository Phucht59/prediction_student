from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.studies.oulad_v4.data import OULADV4Data, STATIC_COLUMNS


CATEGORICAL_STATIC = ("code_module", "presentation_season")


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


@dataclass
class OULADPreprocessorsV51:
    sequence_mean: np.ndarray | None = None
    sequence_std: np.ndarray | None = None
    aggregate: Pipeline | None = None
    static: ColumnTransformer | None = None


@dataclass(frozen=True)
class OULADInputsV51:
    sequence: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    aggregate: np.ndarray
    static: np.ndarray
    target: np.ndarray
    preprocessors: OULADPreprocessorsV51
    aggregate_columns: tuple[str, ...]


def compact_aggregate_columns(available: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    available_set = set(available)
    selected = [
        f"{channel}__{summary}"
        for channel, summaries in COMPACT_SUMMARIES.items()
        for summary in summaries
        if f"{channel}__{summary}" in available_set
    ]
    if "inactive_week_count" in available_set:
        selected.append("inactive_week_count")
    missing_groups = [
        channel
        for channel, summaries in COMPACT_SUMMARIES.items()
        if not any(f"{channel}__{summary}" in available_set for summary in summaries)
    ]
    if missing_groups:
        raise ValueError(f"Compact aggregate groups missing: {missing_groups}")
    if len(selected) >= len(available):
        raise ValueError("Compact aggregate contract did not reduce the full aggregate")
    return tuple(selected)


def _static_preprocessor() -> ColumnTransformer:
    numeric = [column for column in STATIC_COLUMNS if column not in CATEGORICAL_STATIC]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_STATIC,
            ),
        ],
        sparse_threshold=0.0,
    )


def prepare_oulad_inputs(
    data: OULADV4Data,
    fit_indices: np.ndarray,
    transform_indices: np.ndarray,
    *,
    fitted: OULADPreprocessorsV51 | None = None,
) -> OULADInputsV51:
    fitted = fitted or OULADPreprocessorsV51()
    source = data.dynamic_sequence
    if source.shape[2] != 47:
        raise ValueError(f"Expected frozen 47-channel sequence, got {source.shape[2]}")
    if fitted.sequence_mean is None or fitted.sequence_std is None:
        train = source[fit_indices]
        train_mask = data.base.padding_mask[fit_indices].astype(np.float32)
        count = max(1, int(train_mask.sum()))
        fitted.sequence_mean = ((train * train_mask[..., None]).sum((0, 1)) / count).astype(np.float32)
        variance = (((train - fitted.sequence_mean) ** 2) * train_mask[..., None]).sum((0, 1)) / count
        fitted.sequence_std = np.sqrt(np.maximum(variance, 1e-12)).clip(1e-6).astype(np.float32)
    sequence = (source[transform_indices] - fitted.sequence_mean) / fitted.sequence_std
    mask = data.base.padding_mask[transform_indices].astype(np.float32)
    sequence *= mask[..., None]

    aggregate_columns = compact_aggregate_columns(list(data.v2.aggregate_columns))
    raw_aggregate = data.v2.aggregate.loc[:, list(aggregate_columns)].to_numpy(dtype=np.float32)
    if fitted.aggregate is None:
        fitted.aggregate = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        fitted.aggregate.fit(raw_aggregate[fit_indices])
    aggregate = fitted.aggregate.transform(raw_aggregate[transform_indices]).astype(np.float32)

    if fitted.static is None:
        fitted.static = _static_preprocessor()
        fitted.static.fit(data.base.cohort.loc[fit_indices, STATIC_COLUMNS])
    static = fitted.static.transform(data.base.cohort.loc[transform_indices, STATIC_COLUMNS]).astype(np.float32)
    return OULADInputsV51(
        sequence=sequence.astype(np.float32),
        lengths=data.base.valid_lengths[transform_indices].astype(np.int64),
        mask=mask,
        aggregate=aggregate,
        static=static,
        target=data.y[transform_indices].astype(np.float32),
        preprocessors=fitted,
        aggregate_columns=aggregate_columns,
    )


__all__ = [
    "COMPACT_SUMMARIES",
    "OULADInputsV51",
    "OULADPreprocessorsV51",
    "compact_aggregate_columns",
    "prepare_oulad_inputs",
]
