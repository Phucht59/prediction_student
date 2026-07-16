from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ForecastData:
    forecast_id: str
    cohort: pd.DataFrame
    targets: pd.DataFrame
    tabular: pd.DataFrame
    sequence: np.ndarray
    valid_lengths: np.ndarray
    padding_mask: np.ndarray
    channel_order: list[str]
    split: pd.DataFrame

    @property
    def y(self) -> np.ndarray:
        return self.targets["target_at_risk"].to_numpy(dtype=int)

    @property
    def record_ids(self) -> np.ndarray:
        return self.cohort["record_id"].astype(str).to_numpy()


def load_forecast(processed_root: str | Path, forecast_id: str) -> ForecastData:
    root = Path(processed_root)
    cohort = pd.read_parquet(root / "cohorts" / f"{forecast_id}.parquet")
    targets = pd.read_parquet(root / "targets" / f"{forecast_id}.parquet")
    aggregate = pd.read_parquet(root / "aggregated" / f"{forecast_id}.parquet")
    flat = pd.read_parquet(root / "flat" / f"{forecast_id}.parquet")
    archive = np.load(root / "sequences" / f"{forecast_id}.npz", allow_pickle=True)
    ids = archive["record_ids"].astype(str)
    if list(cohort["record_id"].astype(str)) != list(ids):
        raise RuntimeError("Cohort/sequence record order mismatch")
    targets = cohort[["record_id"]].merge(targets, on="record_id", validate="one_to_one")
    tabular = cohort[["record_id", "code_module", "presentation_season", "num_of_prev_attempts", "studied_credits", "registration_lead_time", "module_presentation_length", "valid_sequence_length"]].merge(aggregate, on="record_id", validate="one_to_one").merge(flat, on="record_id", validate="one_to_one")
    forbidden = {"final_result", "date_unregistration", "original_final_result", "target_at_risk", "code_presentation"}
    if forbidden.intersection(tabular.columns):
        raise RuntimeError(f"Target/future field in tabular features: {forbidden.intersection(tabular.columns)}")
    split = pd.read_csv(root / "manifests" / "split_manifest.csv")
    split = split[split["forecast_id"] == forecast_id].copy()
    if set(split["record_id"]) != set(cohort["record_id"]):
        raise RuntimeError("Split manifest does not cover forecast cohort")
    return ForecastData(forecast_id, cohort, targets, tabular, archive["sequence"].astype(np.float32), archive["valid_lengths"].astype(int), archive["padding_mask"].astype(bool), [str(value) for value in archive["channel_order"]], split)


def record_positions(data: ForecastData, record_ids: set[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(data.record_ids, list(record_ids)))
