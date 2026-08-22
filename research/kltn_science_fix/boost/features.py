"""Mask-safe extra temporal channels. Does not change the binary label."""
from __future__ import annotations

import numpy as np

from src.prediction.data.common import UnifiedHybridData


def add_temporal_deltas(views: dict[str, UnifiedHybridData]) -> dict[str, UnifiedHybridData]:
    out = {}
    for stage, view in views.items():
        temporal = view.temporal.astype(np.float32)
        delta = np.zeros_like(temporal)
        if temporal.shape[1] >= 2:
            delta[:, 1:] = temporal[:, 1:] - temporal[:, :-1]
        delta[~view.temporal_mask] = 0.0
        stacked = np.concatenate((temporal, delta), axis=-1)
        out[stage] = UnifiedHybridData(
            static=view.static,
            temporal=stacked,
            temporal_mask=view.temporal_mask,
            lengths=view.lengths,
            aggregate=view.aggregate,
            aggregate_available=view.aggregate_available,
            progress=view.progress,
            target=view.target,
            record_id=view.record_id,
            group_id=view.group_id,
            metadata={**dict(view.metadata), "temporal_deltas": True},
        )
        out[stage].validate()
    return out
