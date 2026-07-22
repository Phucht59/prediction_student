from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.studies.oulad.materialize import CHANNELS
from src.studies.oulad_v3.data import build_dynamic_representation
from src.studies.oulad_v4.data import OULADV4Data


CLICK_CHANNELS = [
    "total_clicks",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
]
BEHAVIOR_CHANNELS = [
    "total_clicks",
    "active_days",
    "unique_sites",
    "unique_activity_types",
    "content_clicks",
    "forum_clicks",
    "quiz_clicks",
    "assessment_related_clicks",
]


def _channel(name: str) -> int:
    return CHANNELS.index(name)


def _aggregate_fast(sequence: np.ndarray, valid_lengths: np.ndarray) -> pd.DataFrame:
    values = sequence.astype(np.float64)
    records, weeks, channels = values.shape
    mask = np.arange(weeks)[None, :] < valid_lengths[:, None]
    count = valid_lengths.astype(np.float64)[:, None]
    masked = values * mask[:, :, None]
    total = masked.sum(axis=1)
    mean = total / count
    variance = (((values - mean[:, None, :]) ** 2) * mask[:, :, None]).sum(axis=1) / count
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    minimum = np.where(mask[:, :, None], values, np.inf).min(axis=1)
    maximum = np.where(mask[:, :, None], values, -np.inf).max(axis=1)
    row = np.arange(records)
    last = values[row, valid_lengths - 1]
    previous = values[row, np.maximum(valid_lengths - 2, 0)]
    recent = (last + previous) / 2.0
    time = np.arange(weeks, dtype=np.float64)[None, :, None]
    sum_x = (time * mask[:, :, None]).sum(axis=1)
    sum_x2 = ((time**2) * mask[:, :, None]).sum(axis=1)
    sum_xy = (time * values * mask[:, :, None]).sum(axis=1)
    denominator = count * sum_x2 - sum_x**2
    slope = np.divide(count * sum_xy - sum_x * total, denominator, out=np.zeros_like(total), where=np.abs(denominator) > 1e-12)
    first = np.zeros_like(mean)
    second = np.zeros_like(mean)
    for length in np.unique(valid_lengths):
        selected = valid_lengths == length
        half = max(1, int(length) // 2)
        first[selected] = values[selected, :half].mean(axis=1)
        second[selected] = values[selected, half:int(length)].mean(axis=1) if half < length else values[selected, int(length) - 1]
    blocks = [total, mean, standard_deviation, minimum, maximum, last, slope, recent, first, second]
    summaries = ["sum", "mean", "std", "min", "max", "last", "slope", "recent_2_week_mean", "first_half_mean", "second_half_mean"]
    output: dict[str, np.ndarray] = {}
    for channel_index, channel in enumerate(CHANNELS):
        for block, summary in zip(blocks, summaries):
            output[f"{channel}__{summary}"] = block[:, channel_index]
    output["inactive_week_count"] = ((values[:, :, _channel("total_clicks")] == 0) & mask).sum(axis=1)
    return pd.DataFrame(output)


def augment_training_data(
    data: OULADV4Data,
    train_indices: np.ndarray,
    strategy: str,
    seed: int,
) -> tuple[OULADV4Data, dict[str, object]]:
    """Apply one safe training-only transform and rebuild every dependent feature."""

    if strategy == "none":
        return data, {"strategy": "none", "changed_values": 0, "training_records": int(len(train_indices))}
    if strategy not in {"event_thinning", "short_span_masking", "channel_dropout"}:
        raise KeyError(strategy)
    rng = np.random.default_rng(seed)
    base = data.base.sequence.copy()
    before = base[train_indices].copy()
    lengths = data.base.valid_lengths
    if strategy == "event_thinning":
        keep_probability = float(rng.uniform(0.85, 0.95))
        for row in train_indices:
            length = int(lengths[row])
            for name in CLICK_CHANNELS:
                index = _channel(name)
                values = np.rint(np.clip(base[row, :length, index], 0, None)).astype(np.int64)
                base[row, :length, index] = rng.binomial(values, keep_probability)
            total = base[row, :length, _channel("total_clicks")]
            for name in ["active_days", "unique_sites", "unique_activity_types"]:
                index = _channel(name)
                base[row, :length, index] = np.minimum(base[row, :length, index], total)
    elif strategy == "short_span_masking":
        for row in train_indices:
            length = int(lengths[row])
            if length <= 1:
                continue
            width = int(rng.integers(1, min(2, length - 1) + 1))
            start = int(rng.integers(0, length - width + 1))
            base[row, start : start + width, [_channel(name) for name in BEHAVIOR_CHANNELS]] = 0
    else:
        droppable = ["content_clicks", "forum_clicks", "quiz_clicks", "assessment_related_clicks"]
        for row in train_indices:
            length = int(lengths[row])
            name = str(rng.choice(droppable))
            index = _channel(name)
            removed = base[row, :length, index].copy()
            base[row, :length, index] = 0
            total_index = _channel("total_clicks")
            base[row, :length, total_index] = np.maximum(0, base[row, :length, total_index] - removed)
    base *= data.base.padding_mask[..., None]
    if (base < 0).any() or not np.isfinite(base).all():
        raise RuntimeError("Unsafe OULAD augmentation output")
    dynamic, channel_order = build_dynamic_representation(base, data.base.padding_mask)
    aggregate = data.v2.aggregate.copy()
    rebuilt = _aggregate_fast(base[train_indices], lengths[train_indices])
    rebuilt.index = train_indices
    aggregate.loc[train_indices, list(data.v2.aggregate_columns)] = rebuilt.loc[:, list(data.v2.aggregate_columns)].to_numpy()
    base_forecast = replace(data.base, sequence=base)
    v2 = replace(data.v2, base=base_forecast, aggregate=aggregate)
    augmented = replace(data, v2=v2, dynamic_sequence=dynamic, dynamic_channel_order=tuple(channel_order))
    changed = int(np.count_nonzero(before != base[train_indices]))
    if changed == 0:
        raise RuntimeError(f"{strategy} was a no-op")
    return augmented, {
        "strategy": strategy,
        "changed_values": changed,
        "training_records": int(len(train_indices)),
        "target_unchanged": bool(np.array_equal(data.y, augmented.y)),
        "valid_lengths_unchanged": bool(np.array_equal(data.base.valid_lengths, augmented.base.valid_lengths)),
        "padding_mask_unchanged": bool(np.array_equal(data.base.padding_mask, augmented.base.padding_mask)),
        "dependent_dynamic_features_rebuilt": True,
        "dependent_aggregate_features_rebuilt": True,
    }


__all__ = ["augment_training_data"]
