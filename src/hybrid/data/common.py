"""Common utilities, hashing helpers, and sequence manipulation for Hybrid datasets."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.hybrid.contracts import HybridDataView


def sha256_hash_str(value: str) -> str:
    """Return SHA-256 hex digest for a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_deterministic_id(*parts: Any, length: int = 24) -> str:
    """Create a stable, deterministic identifier from component strings."""
    joined = "|".join(str(p).strip() for p in parts)
    return sha256_hash_str(joined)[:length]


def truncate_history(view: HybridDataView, history_window_weeks: int) -> HybridDataView:
    """Extract per-record the most recent L observed valid weeks ending at the same cutoff.
    
    Holding cohort, target, split, and prediction time fixed while varying
    only historical window length.
    
    Semantics:
    For each record i:
        valid_indices = positions where mask[i] == True
        selected = last min(L, len(valid_indices)) valid timesteps in chronological order
        Place selected valid observations left-aligned: output[i, 0:k], mask[i, 0:k] = True
        Padded positions k..L are 0.0 with mask = False
    """
    if history_window_weeks <= 0:
        raise ValueError(f"history_window_weeks must be positive, got {history_window_weeks}")

    view.validate()
    n_records, total_weeks, n_channels = view.temporal.shape
    out_t = int(history_window_weeks)

    new_temporal = np.zeros((n_records, out_t, n_channels), dtype=np.float32)
    new_mask = np.zeros((n_records, out_t), dtype=bool)
    new_lengths = np.zeros(n_records, dtype=np.int64)

    for i in range(n_records):
        valid_indices = np.where(view.mask[i])[0]
        n_valid = len(valid_indices)
        k = min(n_valid, out_t)

        if k > 0:
            selected_indices = valid_indices[-k:]
            new_temporal[i, :k, :] = view.temporal[i, selected_indices, :]
            new_mask[i, :k] = True
            new_lengths[i] = k
        else:
            new_lengths[i] = 0

    truncated_metadata = dict(view.metadata)
    truncated_metadata["history_window_weeks"] = history_window_weeks
    truncated_metadata["history_selection"] = "most_recent_valid_timesteps"
    truncated_metadata["fixed_landmark"] = True
    truncated_metadata["is_truncated_view"] = True

    new_view = HybridDataView(
        record_id=view.record_id.copy(),
        group_id=view.group_id.copy(),
        target=view.target.copy(),
        temporal=new_temporal,
        mask=new_mask,
        lengths=new_lengths,
        context=view.context.copy() if view.context is not None else None,
        metadata=truncated_metadata,
    )
    new_view.validate()
    return new_view
