"""Dataset-specific Snorkel LabelModel fitting and deterministic inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from snorkel.labeling.model import LabelModel

from .lf_registry import registry


def vote_matrix(frame) -> np.ndarray:
    lfs = registry()
    return np.asarray([[lf(row) for lf in lfs] for row in frame.to_dict("records")], dtype=int)


def fit_dataset(frame, *, seed: int, epochs: int = 100) -> LabelModel:
    matrix = vote_matrix(frame)
    if not len(matrix): raise ValueError("cannot fit label model without training rows")
    model = LabelModel(cardinality=3, verbose=False)
    model.fit(L_train=matrix, n_epochs=epochs, seed=seed, log_freq=1000)
    return model


def model_payload(model: LabelModel, *, dataset: str, seed: int, train_rows: int) -> dict:
    weights = model.get_weights().tolist()
    raw = json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()
    return {"dataset": dataset, "seed": seed, "train_rows": train_rows, "cardinality": 3, "lf_ids": [lf.lf_id for lf in registry()], "weights": weights, "weights_sha256": hashlib.sha256(raw).hexdigest(), "fit_scope": "train_only"}


__all__ = ["fit_dataset", "model_payload", "vote_matrix"]
