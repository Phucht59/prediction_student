"""Train-only aggregation of ordinal weak labels for each canonical action."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import numpy as np
import pandas as pd

ABSTAIN = -1
CARDINALITY = 4


@dataclass(frozen=True)
class WeakLabelSource:
    name: str
    family: str

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("weak label source name and family are required")


def validate_vote_matrix(
    votes: np.ndarray,
    sources: tuple[WeakLabelSource, ...],
) -> np.ndarray:
    matrix = np.asarray(votes, dtype=int)
    if matrix.ndim != 2:
        raise ValueError("weak label votes must be a two-dimensional matrix")
    if matrix.shape[1] != len(sources):
        raise ValueError("vote column count must match source manifest")
    allowed = {ABSTAIN, 0, 1, 2, 3}
    if not set(np.unique(matrix)).issubset(allowed):
        raise ValueError("weak labels must be ABSTAIN or ordinal values 0..3")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("weak label source names must be unique")
    return matrix


def fit_label_model(
    train_votes: np.ndarray,
    sources: tuple[WeakLabelSource, ...],
    *,
    seed: int,
    epochs: int = 1000,
):
    """Fit a Snorkel LabelModel on inner-training votes only."""

    matrix = validate_vote_matrix(train_votes, sources)
    if len(matrix) < 30:
        raise ValueError("at least 30 train rows are required for label aggregation")
    try:
        label_model_class = import_module(
            "snorkel.labeling.model"
        ).LabelModel
    except ModuleNotFoundError as exc:
        raise RuntimeError("snorkel is required for weak label aggregation") from exc

    model = label_model_class(cardinality=CARDINALITY, verbose=False)
    model.fit(
        L_train=matrix,
        n_epochs=epochs,
        seed=seed,
        log_freq=0,
    )
    return model


def _source_family_count(
    matrix: np.ndarray,
    sources: tuple[WeakLabelSource, ...],
) -> np.ndarray:
    families = np.empty(len(matrix), dtype=int)
    for row_index, row in enumerate(matrix):
        active = {
            sources[column_index].family
            for column_index, vote in enumerate(row)
            if vote != ABSTAIN
        }
        families[row_index] = len(active)
    return families


def aggregate_votes(
    model,
    votes: np.ndarray,
    sources: tuple[WeakLabelSource, ...],
    *,
    minimum_confidence: float,
    minimum_source_families: int,
) -> pd.DataFrame:
    """Return soft ordinal targets and auditable retention metadata."""

    if not 0.0 < minimum_confidence <= 1.0:
        raise ValueError("minimum_confidence must be in (0, 1]")
    if minimum_source_families < 1:
        raise ValueError("minimum_source_families must be positive")

    matrix = validate_vote_matrix(votes, sources)
    probabilities = np.asarray(model.predict_proba(L=matrix), dtype=float)
    if probabilities.shape != (len(matrix), CARDINALITY):
        raise RuntimeError("label model returned an unexpected probability shape")
    confidence = probabilities.max(axis=1)
    expected_relevance = probabilities @ np.arange(CARDINALITY, dtype=float)
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
        axis=1,
    )
    family_count = _source_family_count(matrix, sources)
    retained = (confidence >= minimum_confidence) & (
        family_count >= minimum_source_families
    )

    result = pd.DataFrame(
        {
            "expected_relevance": expected_relevance,
            "hard_relevance": probabilities.argmax(axis=1).astype(int),
            "label_confidence": confidence,
            "label_entropy": entropy,
            "independent_source_families": family_count,
            "label_status": np.where(retained, "RETAINED", "ABSTAINED"),
        }
    )
    for class_index in range(CARDINALITY):
        result[f"probability_relevance_{class_index}"] = probabilities[:, class_index]
    return result


def source_correlation_audit(
    votes: np.ndarray,
    sources: tuple[WeakLabelSource, ...],
) -> pd.DataFrame:
    """Pairwise agreement audit; highly redundant sources require review."""

    matrix = validate_vote_matrix(votes, sources)
    rows: list[dict[str, str | float | int]] = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            jointly_active = (matrix[:, left] != ABSTAIN) & (matrix[:, right] != ABSTAIN)
            count = int(jointly_active.sum())
            agreement = (
                float(np.mean(matrix[jointly_active, left] == matrix[jointly_active, right]))
                if count
                else float("nan")
            )
            rows.append(
                {
                    "left_source": sources[left].name,
                    "left_family": sources[left].family,
                    "right_source": sources[right].name,
                    "right_family": sources[right].family,
                    "joint_vote_count": count,
                    "exact_agreement": agreement,
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "ABSTAIN",
    "CARDINALITY",
    "WeakLabelSource",
    "aggregate_votes",
    "fit_label_model",
    "source_correlation_audit",
    "validate_vote_matrix",
]
