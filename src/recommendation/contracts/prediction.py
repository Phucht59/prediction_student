"""Read-only adapter for the frozen prediction artifact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "record_id", "group_id", "score", "model", "domain", "stage", "outer_fold", "seed",
}


@dataclass(frozen=True)
class PredictionRecord:
    """One record-stage prediction after the frozen seed ensemble is reduced."""

    dataset: str
    student_id: str
    record_id: str
    stage: str
    outer_fold: int
    risk_probability: float
    prediction_threshold: float | None
    prediction_source_version: str
    prediction_seed_count: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PredictionArtifactAdapter:
    """Expose only frozen Hybrid prediction rows to downstream consumers.

    The adapter deliberately ignores ``target``, ``prediction`` and all other
    evaluation columns. ``score`` is the persisted Hybrid probability. If
    multiple frozen seeds exist, their mean is the prediction authority.
    """

    def __init__(self, records: pd.DataFrame, source_version: str) -> None:
        self.records = records.copy()
        self.source_version = source_version

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        dataset: str,
        stages: tuple[str, ...],
        model: str = "Hybrid",
        expected_seeds: tuple[int, ...] = (42, 1201, 2026),
    ) -> "PredictionArtifactAdapter":
        artifact = Path(path)
        try:
            source_label = artifact.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            source_label = artifact.name
        source_version = f"{source_label}#sha256={file_sha256(artifact)}"
        frame = pd.read_parquet(artifact)
        missing = REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"prediction artifact missing columns: {sorted(missing)}")
        frame = frame[
            (frame["model"].astype(str).str.casefold() == model.casefold())
            & (frame["domain"].astype(str).str.casefold() == dataset.casefold())
            & frame["stage"].isin(stages)
        ].copy()
        if frame.empty:
            raise ValueError("no frozen prediction rows match adapter scope")
        if frame["record_id"].isna().any() or frame["group_id"].isna().any():
            raise ValueError("prediction identity contains nulls")
        if not np.isfinite(frame["score"]).all() or ((frame["score"] < 0) | (frame["score"] > 1)).any():
            raise ValueError("prediction score is outside [0, 1]")
        group_keys = ["record_id", "group_id", "domain", "stage", "outer_fold"]
        grouped = frame.groupby(group_keys, sort=True, dropna=False)
        seed_counts = grouped["seed"].nunique()
        if not (seed_counts == len(expected_seeds)).all():
            bad = seed_counts[seed_counts != len(expected_seeds)].head().to_dict()
            raise ValueError(f"frozen seed coverage is incomplete: {bad}")
        seed_sets = grouped["seed"].agg(lambda values: tuple(sorted(int(x) for x in values)))
        if not (seed_sets == tuple(sorted(expected_seeds))).all():
            raise ValueError("prediction artifact contains an unexpected seed set")
        records = grouped["score"].mean().rename("risk_probability").reset_index()
        if "threshold" in frame:
            thresholds = grouped["threshold"].agg(
                lambda values: float(values.iloc[0]) if values.nunique(dropna=False) == 1 else np.nan
            ).rename("prediction_threshold").reset_index()
            records = records.merge(thresholds, on=group_keys, how="left", validate="one_to_one")
        else:
            records["prediction_threshold"] = np.nan
        records["dataset"] = dataset
        records["prediction_source_version"] = source_version
        records["prediction_seed_count"] = len(expected_seeds)
        records["student_id"] = records["group_id"].astype(str)
        records["record_id"] = records["record_id"].astype(str)
        records["outer_fold"] = records["outer_fold"].astype(int)
        records = records[
            ["dataset", "student_id", "record_id", "stage", "outer_fold",
             "risk_probability", "prediction_threshold", "prediction_source_version",
             "prediction_seed_count"]
        ]
        if records.duplicated(["record_id", "stage"]).any():
            raise ValueError("prediction adapter produced duplicate record-stage rows")
        return cls(records, source_version)

    def to_records(self) -> tuple[PredictionRecord, ...]:
        return tuple(PredictionRecord(**row) for row in self.records.to_dict("records"))
