from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.studies.oulad.data import ForecastData, load_forecast


STATIC_COLUMNS = [
    "code_module",
    "presentation_season",
    "num_of_prev_attempts",
    "studied_credits",
    "registration_lead_time",
    "module_presentation_length",
]
FORBIDDEN_COLUMNS = {
    "final_result",
    "date_unregistration",
    "original_final_result",
    "target_at_risk",
    "code_presentation",
    "is_banked",
}


@dataclass(frozen=True)
class OULADV2Data:
    base: ForecastData
    aggregate: pd.DataFrame
    aggregate_columns: tuple[str, ...]
    development_indices: np.ndarray
    development_manifest: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        return self.base.y

    @property
    def groups(self) -> np.ndarray:
        return self.base.cohort["id_student"].to_numpy(dtype=np.int64)

    def outer_indices(self, outer_fold: int) -> tuple[np.ndarray, np.ndarray]:
        manifest = self.development_manifest
        validation_ids = set(manifest.loc[manifest["outer_fold"].astype(int) == outer_fold, "record_id"].astype(str))
        record_ids = self.base.record_ids
        validation = np.flatnonzero(np.isin(record_ids, list(validation_ids)))
        train = np.setdiff1d(self.development_indices, validation, assume_unique=True)
        if set(self.groups[train]) & set(self.groups[validation]):
            raise RuntimeError("Outer split has student overlap")
        return train, validation


def load_v2_data(processed_root: str | Path, protocol: dict) -> OULADV2Data:
    root = Path(processed_root)
    forecast_id = protocol["data"]["forecast_id"]
    if forecast_id != "F2_MIDDLE":
        raise RuntimeError("V2 mandatory pilot is frozen to F2_MIDDLE")
    base = load_forecast(root, forecast_id)
    aggregate_raw = pd.read_parquet(root / "aggregated" / f"{forecast_id}.parquet")
    aggregate = base.cohort[["record_id"]].merge(aggregate_raw, on="record_id", validate="one_to_one")
    aggregate_columns = tuple(column for column in aggregate.columns if column != "record_id")
    if len(aggregate_columns) != protocol["data"]["aggregate_feature_count"]:
        raise RuntimeError(f"Expected 161 aggregate features, got {len(aggregate_columns)}")
    if any(re.search(r"(?:^|__)week_\d+$", column.lower()) for column in aggregate_columns):
        raise RuntimeError("Flattened week feature entered the aggregate-only contract")
    if FORBIDDEN_COLUMNS.intersection(aggregate_columns):
        raise RuntimeError("Forbidden target/future field entered aggregate inputs")
    manifest = base.split.loc[base.split["role"] == protocol["data"]["development_role"]].copy()
    manifest["record_id"] = manifest["record_id"].astype(str)
    record_ids = base.record_ids
    development = np.flatnonzero(np.isin(record_ids, manifest["record_id"]))
    if len(development) != len(manifest):
        raise RuntimeError("Development manifest and loaded cohort are not aligned")
    forbidden_roles = set(protocol["data"]["forbidden_roles_during_selection"])
    if forbidden_roles.intersection(set(manifest["role"])):
        raise RuntimeError("Future role entered V2 selection manifest")
    return OULADV2Data(base, aggregate, aggregate_columns, development, manifest)


def build_inner_manifest(data: OULADV2Data, outer_fold: int, seed: int, n_splits: int = 2) -> pd.DataFrame:
    outer_train, _ = data.outer_indices(outer_fold)
    y = data.y[outer_train]
    groups = data.groups[outer_train]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for inner_fold, (train_rel, validation_rel) in enumerate(splitter.split(outer_train, y, groups)):
        train_indices = outer_train[train_rel]
        validation_indices = outer_train[validation_rel]
        if set(data.groups[train_indices]) & set(data.groups[validation_indices]):
            raise RuntimeError("Inner split has student overlap")
        for role, indices in (("inner_train", train_indices), ("inner_validation", validation_indices)):
            for index in indices:
                rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "role": role,
                        "record_id": str(data.base.record_ids[index]),
                        "id_student": int(data.groups[index]),
                        "target_at_risk": int(data.y[index]),
                    }
                )
    return pd.DataFrame(rows)


def manifest_indices(data: OULADV2Data, manifest: pd.DataFrame, inner_fold: int) -> tuple[np.ndarray, np.ndarray]:
    ids_to_position = {record_id: index for index, record_id in enumerate(data.base.record_ids)}
    fold = manifest.loc[manifest["inner_fold"] == inner_fold]
    train = np.asarray([ids_to_position[value] for value in fold.loc[fold["role"] == "inner_train", "record_id"]], dtype=int)
    validation = np.asarray([ids_to_position[value] for value in fold.loc[fold["role"] == "inner_validation", "record_id"]], dtype=int)
    return train, validation
