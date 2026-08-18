"""Train-only, reversible structured input adapter for frozen FINAL H1.

SMOTE/ADASYN interpolates only continuous model tensors. Sequence length, mask, and
one-hot static positions are structural: synthetic records inherit them from the
nearest original outer-train record in continuous flattened space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import pairwise_distances_argmin


@dataclass(frozen=True)
class H1Inputs:
    sequence: np.ndarray
    lengths: np.ndarray
    mask: np.ndarray
    aggregate: np.ndarray
    static: np.ndarray


@dataclass(frozen=True)
class H1Layout:
    weeks: int
    channels: int
    aggregate_dim: int
    static_dim: int
    structural_static: tuple[int, ...]
    structural_sequence_channels: tuple[int, ...]


def flatten_h1_sample(inputs: H1Inputs, layout: H1Layout) -> np.ndarray:
    """Flatten only the numeric tensors consumed by H1, preserving row order."""
    if inputs.sequence.shape[1:] != (layout.weeks, layout.channels):
        raise ValueError("H1 sequence shape mismatch")
    return np.concatenate([inputs.sequence.reshape(len(inputs.sequence), -1), inputs.aggregate, inputs.static], axis=1).astype(np.float32)


def reconstruct_h1_sample(flat: np.ndarray, originals: H1Inputs, layout: H1Layout) -> H1Inputs:
    """Reconstruct H1 tensors; derive discrete structure from nearest train row."""
    sequence_size = layout.weeks * layout.channels
    expected = sequence_size + layout.aggregate_dim + layout.static_dim
    if flat.shape[1] != expected:
        raise ValueError("H1 flat feature width mismatch")
    # Deterministic nearest parent uses continuous values only; ties resolve to the
    # first training row (numpy argmin), and never consults validation/test data.
    original_flat = flatten_h1_sample(originals, layout)
    # Parent selection uses the continuous aggregate plus numeric-static subspace,
    # rather than allocating distances over every padded temporal element.
    aggregate_start = sequence_size
    static_start = aggregate_start + layout.aggregate_dim
    numeric_static = [index for index in range(layout.static_dim) if index not in layout.structural_static]
    parent_features = list(range(aggregate_start, static_start)) + [static_start + index for index in numeric_static]
    if flat.shape == original_flat.shape and np.array_equal(flat, original_flat):
        parent = np.arange(len(flat))
    else:
        parent = pairwise_distances_argmin(flat[:, parent_features], original_flat[:, parent_features], metric="euclidean")
    sequence = flat[:, :sequence_size].reshape(-1, layout.weeks, layout.channels).astype(np.float32)
    aggregate = flat[:, sequence_size:sequence_size + layout.aggregate_dim].astype(np.float32)
    static = flat[:, sequence_size + layout.aggregate_dim:].astype(np.float32)
    # Structural fields are copied rather than interpolated.
    if layout.structural_static:
        static[:, layout.structural_static] = originals.static[parent][:, layout.structural_static]
    if layout.structural_sequence_channels:
        sequence[:, :, layout.structural_sequence_channels] = originals.sequence[parent][:, :, layout.structural_sequence_channels]
    lengths = originals.lengths[parent].copy()
    mask = originals.mask[parent].copy()
    sequence[~mask] = 0.0
    if not np.isfinite(sequence).all() or not np.isfinite(aggregate).all() or not np.isfinite(static).all():
        raise FloatingPointError("non-finite reconstructed H1 input")
    return H1Inputs(sequence, lengths, mask, aggregate, static)
