from __future__ import annotations

import hashlib
import json
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.studies.oulad_v2.data import OULADV2Data, STATIC_COLUMNS, build_inner_manifest, load_v2_data, manifest_indices


BASE_CHANNELS = [
    "total_clicks", "active_days", "unique_sites", "unique_activity_types", "content_clicks", "forum_clicks",
    "quiz_clicks", "assessment_related_clicks", "submitted_assessment_count", "late_submission_count",
    "available_score_count", "cumulative_mean_score", "cumulative_weighted_score", "days_since_last_vle_activity",
    "weeks_without_activity", "score_missing_mask",
]
DYNAMIC_CHANNELS = [
    "log1p_total_clicks", "log1p_active_days", "log1p_unique_sites", "log1p_assessment_related_clicks",
    "log1p_submitted_assessment_count", "delta_total_clicks", "delta_active_days", "delta_unique_sites",
    "delta_content_clicks", "delta_forum_clicks", "delta_quiz_clicks", "delta_assessment_related_clicks",
    "delta_submitted_assessment_count", "delta_cumulative_mean_score", "delta_cumulative_weighted_score",
    "rolling_2_week_mean_total_clicks", "rolling_2_week_mean_active_days",
    "rolling_2_week_mean_assessment_clicks", "rolling_2_week_submission_count", "rolling_2_week_score_change",
    "current_inactivity_streak", "activity_resumed_indicator", "new_inactivity_indicator", "content_share",
    "forum_share", "quiz_share", "assessment_share", "score_delta", "weighted_score_delta",
    "late_submission_rate_to_date", "submission_rate_last_2_weeks",
]
SUMMARY_NAMES = ["mean", "std", "min", "max", "last", "slope", "recent_2_week_mean", "first_half_mean", "second_half_mean"]


def semantic_hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _index() -> dict[str, int]:
    return {name: index for index, name in enumerate(BASE_CHANNELS)}


def _previous(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values)
    result[:, 1:] = values[:, :-1]
    return result


def _rolling_two(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    result[:, 1:] = (values[:, 1:] + values[:, :-1]) / 2.0
    return result


def build_dynamic_representation(base_sequence: np.ndarray, padding_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build strictly current/past-only dynamics; the first difference is always zero."""
    if base_sequence.shape[2] != len(BASE_CHANNELS):
        raise RuntimeError("Unexpected V1 sequence channel count")
    idx = _index()
    x = base_sequence.astype(np.float32, copy=True)
    mask = padding_mask.astype(bool)
    values: list[np.ndarray] = []

    for name in ["total_clicks", "active_days", "unique_sites", "assessment_related_clicks", "submitted_assessment_count"]:
        values.append(np.log1p(np.clip(x[:, :, idx[name]], 0, None)))

    delta_names = [
        "total_clicks", "active_days", "unique_sites", "content_clicks", "forum_clicks", "quiz_clicks",
        "assessment_related_clicks", "submitted_assessment_count", "cumulative_mean_score", "cumulative_weighted_score",
    ]
    score_available = x[:, :, idx["score_missing_mask"]] < 0.5
    score_deltas: dict[str, np.ndarray] = {}
    for name in delta_names:
        channel = x[:, :, idx[name]]
        delta = channel - _previous(channel)
        delta[:, 0] = 0.0
        if name in {"cumulative_mean_score", "cumulative_weighted_score"}:
            valid_pair = score_available & _previous(score_available.astype(np.float32)).astype(bool)
            delta = np.where(valid_pair, delta, 0.0)
            score_deltas[name] = delta
        values.append(delta)

    values.extend(
        [
            _rolling_two(x[:, :, idx["total_clicks"]]),
            _rolling_two(x[:, :, idx["active_days"]]),
            _rolling_two(x[:, :, idx["assessment_related_clicks"]]),
            _rolling_two(x[:, :, idx["submitted_assessment_count"]]),
            _rolling_two(score_deltas["cumulative_mean_score"]),
        ]
    )

    active = x[:, :, idx["total_clicks"]] > 0
    streak = np.zeros_like(x[:, :, 0])
    resumed = np.zeros_like(streak)
    newly_inactive = np.zeros_like(streak)
    for week in range(x.shape[1]):
        if week == 0:
            streak[:, week] = (~active[:, week]).astype(np.float32)
        else:
            streak[:, week] = np.where(active[:, week], 0.0, streak[:, week - 1] + 1.0)
            resumed[:, week] = (active[:, week] & ~active[:, week - 1]).astype(np.float32)
            newly_inactive[:, week] = (~active[:, week] & active[:, week - 1]).astype(np.float32)
    values.extend([streak, resumed, newly_inactive])

    denominator = np.maximum(x[:, :, idx["total_clicks"]], 1.0)
    for name in ["content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks"]:
        values.append(np.clip(x[:, :, idx[name]] / denominator, 0.0, 1.0))

    # Score deltas never reinterpret an unavailable score as a score of zero.
    values.append(score_deltas["cumulative_mean_score"])
    values.append(score_deltas["cumulative_weighted_score"])
    submitted = np.clip(x[:, :, idx["submitted_assessment_count"]], 0, None)
    late = np.clip(x[:, :, idx["late_submission_count"]], 0, None)
    cumulative_submitted = np.cumsum(submitted, axis=1)
    cumulative_late = np.cumsum(late, axis=1)
    values.append(np.divide(cumulative_late, np.maximum(cumulative_submitted, 1.0)))
    values.append(_rolling_two(submitted))

    dynamic = np.stack(values, axis=2).astype(np.float32)
    dynamic *= mask[:, :, None]
    combined = np.concatenate([x, dynamic], axis=2).astype(np.float32)
    combined *= mask[:, :, None]
    if combined.shape[2] != 47 or len(values) != len(DYNAMIC_CHANNELS):
        raise RuntimeError(f"Dynamic channel contract mismatch: {combined.shape}, {len(values)}")
    if not np.isfinite(combined).all():
        raise RuntimeError("Dynamic representation contains NaN/inf")
    return combined, BASE_CHANNELS + DYNAMIC_CHANNELS


def aggregate_dynamic_channels(dynamic: np.ndarray, valid_lengths: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Vectorized summaries of only the 31 dynamic channels."""
    values = dynamic[:, :, len(BASE_CHANNELS):].astype(np.float64)
    n, weeks, channels = values.shape
    time = np.arange(weeks, dtype=np.float64)[None, :, None]
    mask = (np.arange(weeks)[None, :] < valid_lengths[:, None])[:, :, None]
    counts = valid_lengths.astype(np.float64)[:, None]
    total = (values * mask).sum(axis=1)
    mean = total / counts
    variance = (((values - mean[:, None, :]) ** 2) * mask).sum(axis=1) / counts
    std = np.sqrt(np.maximum(variance, 0.0))
    minimum = np.where(mask, values, np.inf).min(axis=1)
    maximum = np.where(mask, values, -np.inf).max(axis=1)
    row = np.arange(n)
    last = values[row, valid_lengths - 1]
    previous = values[row, np.maximum(valid_lengths - 2, 0)]
    recent = (last + previous) / 2.0
    sum_x = (time * mask).sum(axis=1)
    sum_x2 = ((time ** 2) * mask).sum(axis=1)
    sum_xy = (time * values * mask).sum(axis=1)
    denominator = counts * sum_x2 - sum_x ** 2
    slope = np.divide(counts * sum_xy - sum_x * total, denominator, out=np.zeros_like(total), where=np.abs(denominator) > 1e-12)
    first = np.empty((n, channels), dtype=np.float64)
    second = np.empty((n, channels), dtype=np.float64)
    for length in np.unique(valid_lengths):
        selected = valid_lengths == length
        half = max(1, int(length) // 2)
        first[selected] = values[selected, :half].mean(axis=1)
        second[selected] = values[selected, half:int(length)].mean(axis=1) if half < length else values[selected, int(length) - 1]
    blocks = [mean, std, minimum, maximum, last, slope, recent, first, second]
    output = np.stack(blocks, axis=2).reshape(n, channels * len(SUMMARY_NAMES)).astype(np.float32)
    # np.stack is channel x summary after reshape; names follow the same channel-major layout.
    names = [f"{channel}__{summary}" for channel in DYNAMIC_CHANNELS for summary in SUMMARY_NAMES]
    if output.shape[1] != 279 or not np.isfinite(output).all():
        raise RuntimeError("Dynamic aggregate contract failed")
    return output, names


@dataclass(frozen=True)
class OULADV3Data:
    v2: OULADV2Data
    dynamic_sequence: np.ndarray
    dynamic_channel_order: tuple[str, ...]
    dynamic_aggregate: np.ndarray
    dynamic_aggregate_columns: tuple[str, ...]
    matched_vector: np.ndarray
    matched_vector_columns: tuple[str, ...]

    @property
    def base(self):
        return self.v2.base

    @property
    def y(self) -> np.ndarray:
        return self.v2.y

    @property
    def groups(self) -> np.ndarray:
        return self.v2.groups

    @property
    def development_indices(self) -> np.ndarray:
        return self.v2.development_indices

    @property
    def development_manifest(self) -> pd.DataFrame:
        return self.v2.development_manifest

    def outer_indices(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        return self.v2.outer_indices(fold)


def load_v3_data(processed_root: str | Path, protocol: dict) -> OULADV3Data:
    # V2 validates the frozen 161-column aggregate contract. V3 keeps the
    # count under its parity section, so adapt an in-memory protocol copy.
    v2_protocol = copy.deepcopy(protocol)
    v2_protocol["data"]["aggregate_feature_count"] = int(
        protocol["dynamic_features"]["v2_aggregate_count"]
    )
    v2_protocol["data"]["forbidden_roles_during_selection"] = list(protocol["data"]["forbidden_roles"])
    v2 = load_v2_data(processed_root, v2_protocol)
    if list(v2.base.channel_order) != BASE_CHANNELS:
        raise RuntimeError("Base channel order changed from frozen V1/V2")
    dynamic, order = build_dynamic_representation(v2.base.sequence, v2.base.padding_mask)
    dynamic_aggregate, aggregate_names = aggregate_dynamic_channels(dynamic, v2.base.valid_lengths)
    base_aggregate = v2.aggregate.loc[:, list(v2.aggregate_columns)].to_numpy(dtype=np.float32)
    matched = np.concatenate([base_aggregate, dynamic_aggregate], axis=1).astype(np.float32)
    matched_names = tuple(list(v2.aggregate_columns) + aggregate_names)
    if semantic_hash(order) != protocol["dynamic_features"]["combined_channel_hash"]:
        raise RuntimeError("Dynamic channel hash mismatch")
    if semantic_hash(aggregate_names) != protocol["dynamic_features"]["dynamic_aggregate_hash"]:
        raise RuntimeError("Dynamic aggregate hash mismatch")
    if matched.shape[1] != protocol["dynamic_features"]["matched_vector_numeric_count"]:
        raise RuntimeError("Matched vector width mismatch")
    return OULADV3Data(v2, dynamic, tuple(order), dynamic_aggregate, tuple(aggregate_names), matched, matched_names)


__all__ = [
    "BASE_CHANNELS", "DYNAMIC_CHANNELS", "SUMMARY_NAMES", "OULADV3Data", "STATIC_COLUMNS",
    "aggregate_dynamic_channels", "build_dynamic_representation", "build_inner_manifest", "load_v3_data", "manifest_indices",
]
