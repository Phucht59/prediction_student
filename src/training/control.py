"""Deterministic training identity, epoch, threshold, and provenance controls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return a stable JSON representation for scientific identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingRunIdentity:
    dataset: str
    model_family: str
    outer_fold: int
    seed: int
    protocol_id: str
    stage_policy_version: str
    config_hash: str
    training_mode: str

    @property
    def fields(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model_family": self.model_family,
            "outer_fold": int(self.outer_fold),
            "seed": int(self.seed),
            "protocol_id": self.protocol_id,
            "stage_policy_version": self.stage_policy_version,
            "config_hash": self.config_hash,
            "training_mode": self.training_mode,
        }

    @property
    def run_id(self) -> str:
        return stable_hash(self.fields)[:24]

    def checkpoint_id(self, checkpoint_epoch: int | None) -> str:
        return stable_hash(
            {"training_run_id": self.run_id, "checkpoint_epoch": checkpoint_epoch}
        )[:24]


def fixed_refit_metadata(fixed_epochs: int) -> dict[str, Any]:
    """Describe a completed fixed-epoch refit without early-stop ambiguity."""
    if fixed_epochs < 1:
        raise ValueError("fixed_epochs must be positive")
    return {
        "training_mode": "fixed_epoch_refit",
        "epochs_trained": int(fixed_epochs),
        "selected_epoch": int(fixed_epochs),
        "checkpoint_epoch": int(fixed_epochs),
        "checkpoint_selection": "final_fixed_epoch",
        "early_stopping_applied": False,
    }


def early_stop_metadata(
    *, epochs_trained: int, selected_epoch: int, monitor: str
) -> dict[str, Any]:
    if not 1 <= selected_epoch <= epochs_trained:
        raise ValueError("selected_epoch must be within the trained trajectory")
    return {
        "training_mode": "inner_early_stopping",
        "epochs_trained": int(epochs_trained),
        "selected_epoch": int(selected_epoch),
        "checkpoint_epoch": int(selected_epoch),
        "checkpoint_selection": monitor,
        "early_stopping_applied": True,
    }


def finalize_training_metadata(
    *,
    fixed_epochs: int | None,
    epochs_trained: int,
    selected_epoch: int,
    monitor: str,
) -> dict[str, Any]:
    """Validate execution counts before emitting checkpoint metadata."""
    if fixed_epochs is not None:
        if epochs_trained != fixed_epochs:
            raise RuntimeError(
                "fixed refit execution count does not match requested epoch count"
            )
        return fixed_refit_metadata(fixed_epochs)
    return early_stop_metadata(
        epochs_trained=epochs_trained,
        selected_epoch=selected_epoch,
        monitor=monitor,
    )


def select_refit_epoch(
    inner_selected_epochs: Iterable[int], policy: str = "median"
) -> int:
    """Aggregate inner-only epoch choices; outer outcomes are intentionally absent."""
    values = np.asarray(list(inner_selected_epochs), dtype=int)
    if values.size == 0 or np.any(values < 1):
        raise ValueError("inner_selected_epochs must contain positive epochs")
    if policy != "median":
        raise ValueError(f"unsupported refit epoch policy: {policy}")
    return int(np.floor(np.median(values) + 0.5))


def select_research_threshold(
    inner_oof_labels: np.ndarray, inner_oof_probabilities: np.ndarray
) -> dict[str, Any]:
    """Maximize Macro-F1 on pooled inner OOF predictions."""
    y = np.asarray(inner_oof_labels, dtype=int)
    p = np.asarray(inner_oof_probabilities, dtype=float)
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 181), p])
    scored = [(float(f1_score(y, p >= t, average="macro")), float(t)) for t in candidates]
    score, threshold = max(scored, key=lambda row: (row[0], -abs(row[1] - 0.5), -row[1]))
    return {
        "policy": "INNER_OOF_RESEARCH_MACRO_F1",
        "threshold": threshold,
        "macro_f1": score,
        "source": "pooled_inner_oof",
        "outer_labels_used": False,
    }


def select_operational_threshold(
    inner_oof_labels: np.ndarray,
    inner_oof_probabilities: np.ndarray,
    *,
    minimum_precision: float = 0.75,
) -> dict[str, Any]:
    """Maximize risk recall subject to an inner-OOF precision constraint."""
    y = np.asarray(inner_oof_labels, dtype=int)
    p = np.asarray(inner_oof_probabilities, dtype=float)
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 181), p])
    rows: list[tuple[float, float, float]] = []
    for threshold in candidates:
        pred = p >= threshold
        precision, recall = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )[:2]
        rows.append((float(threshold), float(precision), float(recall)))
    qualified = [row for row in rows if row[1] >= minimum_precision]
    if qualified:
        threshold, precision, recall = max(
            qualified, key=lambda row: (row[2], row[1], -row[0])
        )
        status = "PRECISION_CONSTRAINT_MET"
    else:
        threshold, precision, recall = min(
            rows, key=lambda row: (abs(row[1] - minimum_precision), -row[2], row[0])
        )
        status = "CONSTRAINT_NOT_REACHED"
    return {
        "policy": "INNER_OOF_OPERATIONAL_RECALL_AT_PRECISION",
        "threshold": threshold,
        "minimum_precision": float(minimum_precision),
        "precision": precision,
        "recall": recall,
        "status": status,
        "source": "pooled_inner_oof",
        "outer_labels_used": False,
        "eligible_for_checkpoint_selection": False,
    }


def pretraining_provenance(
    *, requested: bool, executed: bool, checkpoint: str | None, strategy: str | None
) -> dict[str, Any]:
    if executed and not requested:
        raise ValueError("pretraining cannot execute when it was not requested")
    if not executed and checkpoint is not None:
        raise ValueError("a pretraining checkpoint cannot be consumed when not executed")
    return {
        "requested": bool(requested),
        "executed": bool(executed),
        "checkpoint": checkpoint,
        "strategy": strategy,
    }
