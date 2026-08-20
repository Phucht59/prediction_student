"""Train-only SMOTE/ADASYN on flattened Hybrid tensors. Never fit on STOP/VALID."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE


@dataclass(frozen=True)
class PackedBatch:
    static: np.ndarray
    temporal: np.ndarray
    temporal_mask: np.ndarray
    lengths: np.ndarray
    aggregate: np.ndarray
    aggregate_available: np.ndarray
    progress: np.ndarray
    target: np.ndarray
    record_id: np.ndarray


def pack_features(batch: PackedBatch) -> np.ndarray:
    n = len(batch.target)
    tflat = batch.temporal.reshape(n, -1)
    return np.concatenate(
        [
            batch.static.astype(np.float32),
            batch.aggregate.astype(np.float32),
            tflat.astype(np.float32),
            batch.progress.reshape(n, 1).astype(np.float32),
        ],
        axis=1,
    )


def unpack_features(
    packed: np.ndarray,
    *,
    static_dim: int,
    aggregate_dim: int,
    timesteps: int,
    temporal_dim: int,
    target: np.ndarray,
    stage_mask_template: np.ndarray | None,
) -> PackedBatch:
    n = packed.shape[0]
    i0 = static_dim
    i1 = i0 + aggregate_dim
    i2 = i1 + timesteps * temporal_dim
    static = packed[:, :i0].astype(np.float32)
    aggregate = packed[:, i0:i1].astype(np.float32)
    temporal = packed[:, i1:i2].reshape(n, timesteps, temporal_dim).astype(np.float32)
    progress = packed[:, i2:].reshape(n).astype(np.float32)
    progress = np.clip(progress, 0.0, 1.0)
    if stage_mask_template is not None:
        mask = np.broadcast_to(stage_mask_template.astype(bool), (n, timesteps)).copy()
    else:
        mask = np.abs(temporal).sum(axis=-1) > 1e-6
    temporal = temporal * mask[..., None]
    lengths = mask.sum(axis=1).astype(np.int64)
    available = (np.abs(aggregate).sum(axis=1) > 1e-8).astype(np.int8)
    ids = np.asarray([f"synth:{i}" for i in range(n)], dtype=object)
    return PackedBatch(
        static=static,
        temporal=temporal,
        temporal_mask=mask,
        lengths=lengths,
        aggregate=aggregate,
        aggregate_available=available,
        progress=progress,
        target=target.astype(np.int64),
        record_id=ids,
    )


def _k_neighbors(y: np.ndarray) -> int:
    minority = int(min((y == 0).sum(), (y == 1).sum()))
    return max(1, min(5, minority - 1))


def resample_train(
    packed: np.ndarray,
    target: np.ndarray,
    sampler: str,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    y = np.asarray(target, dtype=np.int64)
    x = np.asarray(packed, dtype=np.float32)
    audit = {
        "sampler": sampler,
        "n_train_before": int(len(y)),
        "n_positive_before": int((y == 1).sum()),
        "n_negative_before": int((y == 0).sum()),
        "fit_on": "train_only",
    }
    if sampler in {"none", "control"}:
        audit.update({"n_train_after": int(len(y)), "n_positive_after": int((y == 1).sum())})
        return x, y, audit
    k = _k_neighbors(y)
    if k < 1:
        audit["skipped"] = "minority_too_small"
        audit.update({"n_train_after": int(len(y)), "n_positive_after": int((y == 1).sum())})
        return x, y, audit
    if sampler == "smote":
        engine = SMOTE(random_state=random_state, k_neighbors=k)
    elif sampler == "adasyn":
        engine = ADASYN(random_state=random_state, n_neighbors=k)
    else:
        raise ValueError(f"unknown sampler: {sampler}")
    try:
        x_new, y_new = engine.fit_resample(x, y)
    except (ValueError, RuntimeError) as exc:
        audit["skipped"] = str(exc)
        audit.update({"n_train_after": int(len(y)), "n_positive_after": int((y == 1).sum())})
        return x, y, audit
    audit.update(
        {
            "n_train_after": int(len(y_new)),
            "n_positive_after": int((y_new == 1).sum()),
            "n_negative_after": int((y_new == 0).sum()),
            "k_neighbors": k,
        }
    )
    return np.asarray(x_new, np.float32), np.asarray(y_new, np.int64), audit


def subset_batch(batch: PackedBatch, ids: list[str]) -> PackedBatch:
    lookup = {str(record): i for i, record in enumerate(batch.record_id)}
    idx = np.asarray([lookup[i] for i in ids if i in lookup], dtype=np.int64)
    return PackedBatch(
        static=batch.static[idx],
        temporal=batch.temporal[idx],
        temporal_mask=batch.temporal_mask[idx],
        lengths=batch.lengths[idx],
        aggregate=batch.aggregate[idx],
        aggregate_available=batch.aggregate_available[idx],
        progress=batch.progress[idx],
        target=batch.target[idx],
        record_id=np.asarray(batch.record_id)[idx],
    )


def fingerprint(batch: PackedBatch) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(batch.static).tobytes())
    digest.update(np.ascontiguousarray(batch.temporal).tobytes())
    digest.update(np.ascontiguousarray(batch.target).tobytes())
    return digest.hexdigest()


__all__ = [
    "PackedBatch",
    "fingerprint",
    "pack_features",
    "resample_train",
    "subset_batch",
    "unpack_features",
]
