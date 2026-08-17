"""Exposure-safe Phase8 OULAD data variants; Phase7 builder remains untouched."""
from __future__ import annotations

import numpy as np

from src.hybrid.phase7.contracts import UnifiedHybridData
from src.hybrid.phase7.data import OULAD_PHASE7_AGGREGATE_CHANNELS, OULAD_PHASE7_TEMPORAL_CHANNELS

TEMPORAL_INDEX = {name: i for i, name in enumerate(OULAD_PHASE7_TEMPORAL_CHANNELS)}
AGGREGATE_INDEX = {name: i for i, name in enumerate(OULAD_PHASE7_AGGREGATE_CHANNELS)}
COUNT_CHANNELS = ("activity_intensity_log1p", "content_activity", "forum_activity", "quiz_activity", "assessment_related_activity")
UNIQUE_CHANNELS = ("unique_sites", "unique_activity_types")
VARIANTS = ("D0_raw", "D1_temporal_safe", "D2_aggregate_safe", "D3_both_safe")


def _raw_counts(temporal: np.ndarray) -> np.ndarray:
    """Return semantic raw activity counts without double-decoding channels.

    Phase7 stores only ``activity_intensity_log1p`` in log space.  The other
    activity and uniqueness channels are already raw counts; applying
    ``expm1`` to them would overflow on perfectly valid values (and silently
    corrupt the representation before scaling).
    """
    raw = temporal.copy()
    i = TEMPORAL_INDEX["activity_intensity_log1p"]
    raw[..., i] = np.expm1(np.clip(temporal[..., i], 0.0, 30.0))
    return raw


def exposure_safe_temporal(temporal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Convert semantic count channels to per-observed-day rates.

    Submission/event counts are intentionally retained raw. Exposure remains
    explicit and padded/unobserved positions stay exactly zero.
    """
    result = temporal.copy().astype(np.float32)
    raw = _raw_counts(temporal); exposure = temporal[..., TEMPORAL_INDEX["week_exposure_fraction"]]
    observed_days = exposure * 7.0
    valid = mask & (observed_days > 0)
    for name in COUNT_CHANNELS:
        i = TEMPORAL_INDEX[name]; rate = np.divide(raw[..., i], observed_days, out=np.zeros_like(raw[..., i]), where=valid)
        result[..., i] = np.log1p(np.maximum(rate, 0.0))
    i = TEMPORAL_INDEX["active_days"]
    result[..., i] = np.divide(raw[..., i], observed_days, out=np.zeros_like(raw[..., i]), where=valid)
    for name in UNIQUE_CHANNELS:
        i = TEMPORAL_INDEX[name]; rate = np.divide(raw[..., i], observed_days, out=np.zeros_like(raw[..., i]), where=valid)
        result[..., i] = np.log1p(np.maximum(rate, 0.0))
    result[~mask] = 0.0
    return result


def exposure_safe_aggregate(temporal: np.ndarray, mask: np.ndarray, aggregate: np.ndarray) -> np.ndarray:
    """Correct activity aggregates while retaining raw cumulative quantity."""
    result = aggregate.copy().astype(np.float32); raw = _raw_counts(temporal)
    activity = raw[..., TEMPORAL_INDEX["activity_intensity_log1p"]]; exposure = temporal[..., TEMPORAL_INDEX["week_exposure_fraction"]]
    valid = mask & (exposure > 0); observed_days = exposure * 7.0
    total_days = observed_days * valid; total_activity = (activity * valid).sum(1); days = total_days.sum(1)
    daily_rate = np.divide(total_activity, days, out=np.zeros_like(total_activity), where=days > 0)
    result[:, AGGREGATE_INDEX["mean_weekly_activity"]] = daily_rate * 7.0
    last = np.maximum(mask.sum(1) - 1, 0); rows = np.arange(len(temporal)); last_days = observed_days[rows, last]
    last_rate = np.divide(activity[rows, last], last_days, out=np.zeros(len(rows), np.float32), where=last_days > 0)
    result[:, AGGREGATE_INDEX["recent_activity"]] = last_rate * 7.0
    result[:, AGGREGATE_INDEX["recent_historical_activity_ratio"]] = np.divide(last_rate, daily_rate, out=np.zeros_like(last_rate), where=daily_rate > 1e-6)
    # Weighted trend: partial weeks carry their observed-day weight.
    for row in rows:
        idx = np.flatnonzero(valid[row]); rates = np.divide(activity[row, idx], observed_days[row, idx], out=np.zeros(len(idx), np.float32), where=observed_days[row, idx] > 0)
        if len(idx) >= 2:
            result[row, AGGREGATE_INDEX["activity_trend"]] = np.polyfit(idx.astype(np.float32), rates, 1, w=np.sqrt(observed_days[row, idx]))[0]
        inactive = (activity[row, idx] <= 0)
        result[row, AGGREGATE_INDEX["cumulative_inactive_weeks"]] = float(exposure[row, idx][inactive].sum())
        # Partial inactive intervals contribute duration, not a full week.
        streak = 0.0
        for pos in idx[::-1]:
            if activity[row, pos] <= 0: streak += float(exposure[row, pos])
            else: break
        result[row, AGGREGATE_INDEX["current_inactivity_streak"]] = streak
    return result


def apply_data_variant(view: UnifiedHybridData, variant: str) -> UnifiedHybridData:
    if variant not in VARIANTS: raise ValueError(f"unknown data variant: {variant}")
    temporal = view.temporal.copy(); aggregate = view.aggregate.copy()
    if variant in {"D1_temporal_safe", "D3_both_safe"}: temporal = exposure_safe_temporal(temporal, view.temporal_mask)
    if variant in {"D2_aggregate_safe", "D3_both_safe"}: aggregate = exposure_safe_aggregate(view.temporal, view.temporal_mask, aggregate)
    metadata = dict(view.metadata); metadata.update({"data_variant": variant, "temporal_variant_formula": "log1p(raw_count/observed_days), active_days/observed_days, unique/day; events raw" if variant in {"D1_temporal_safe", "D3_both_safe"} else "current_raw", "aggregate_variant_formula": "exposure-weighted daily rates and weighted trend; cumulative raw retained" if variant in {"D2_aggregate_safe", "D3_both_safe"} else "current_raw"})
    result = UnifiedHybridData(static=view.static.copy(), temporal=temporal, temporal_mask=view.temporal_mask.copy(), lengths=view.lengths.copy(), aggregate=aggregate, aggregate_available=view.aggregate_available.copy(), progress=view.progress.copy(), target=view.target.copy(), record_id=view.record_id.copy(), group_id=view.group_id.copy(), metadata=metadata, temporal_exposure_normalized=None)
    result.validate(); return result


def variant_manifest(variant: str) -> dict:
    if variant not in VARIANTS: raise ValueError(variant)
    temporal_formula = "current Phase7 raw channels" if variant == "D0_raw" else "semantic exposure-safe rates; submission events raw; exposure explicit"
    aggregate_formula = "current Phase7 aggregate" if variant in {"D0_raw", "D1_temporal_safe"} else "raw cumulative retained; exposure-weighted daily mean/recent/ratio/trend; duration-weighted inactivity"
    return {"variant_id": variant, "temporal_channels": list(OULAD_PHASE7_TEMPORAL_CHANNELS), "aggregate_channels": list(OULAD_PHASE7_AGGREGATE_CHANNELS), "summary_channels": "unchanged baseline summary frame", "formula": {"temporal": temporal_formula, "aggregate": aggregate_formula}, "feature_dimension": {"temporal": len(OULAD_PHASE7_TEMPORAL_CHANNELS), "aggregate": len(OULAD_PHASE7_AGGREGATE_CHANNELS)}, "forbidden_fields_checked": ["final_result", "score", "date_unregistration"], "cutoff_semantics": "registration-aware and strict event cutoff"}
