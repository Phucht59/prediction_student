"""Build Hybrid tensors with production feature code and frozen inner splits.

If the kltn Phase-1 parquet bundle is present, use it and hash-verify.
Otherwise recover the same inner_fold identity from official reconstructed
OOF VALID assignments (one fold per student) and rebuild FIT/STOP with the
same StratifiedGroupKFold(n_splits=5, seed=42) rule. Outer-test IDs from
artifacts/prediction/final/outer_test_final are excluded.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.prediction.contracts import OULAD_STATES, UCI_STAGES
from src.prediction.data.oulad import load_oulad_static_tables
from src.prediction.data.oulad_features import (
    build_oulad_information_state,
    build_vle_daily,
    context_frame_from_base,
    fit_oulad_preprocessor,
)
from src.prediction.data.preprocessing import ContextPreprocessor
from src.prediction.data.uci import (
    UCI_CATEGORICAL_CONTEXT,
    UCI_NUMERIC_CONTEXT,
    build_uci_combined,
    build_uci_stage_view,
)

from experiments.hybrid_vnext.protocol import assert_disjoint, split_paths, verify_split_hashes
from experiments.imbalance.samplers import PackedBatch, subset_batch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = Path(r"C:\hufit\kltn\data\raw")
if not (DEFAULT_RAW / "student-mat.csv").is_file():
    DEFAULT_RAW = ROOT / "data" / "raw"
RECON = ROOT / "artifacts" / "prediction" / "reconstructed"
OUTER_PRED = ROOT / "artifacts" / "prediction" / "final" / "outer_test_final" / "predictions.parquet"
CACHE = ROOT / "artifacts" / "experiments" / "imbalance" / "cache"


def raw_dir() -> Path:
    if (DEFAULT_RAW / "student-mat.csv").is_file() or (DEFAULT_RAW / "studentInfo.csv").is_file():
        return DEFAULT_RAW
    raise FileNotFoundError("raw UCI/OULAD CSVs not found")


def frozen_split_bundle_available() -> bool:
    paths = split_paths()
    return paths["uci_inner"].is_file() and paths["oulad_inner"].is_file()


def uci_context() -> pd.DataFrame:
    frame, _ = build_uci_combined(raw_dir() / "student-mat.csv", raw_dir() / "student-por.csv")
    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    frame["group_id"] = frame["global_student_group"].astype(str)
    return frame


def oulad_context() -> pd.DataFrame:
    _, _, base = load_oulad_static_tables(raw_dir())
    frame = context_frame_from_base(base)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["group_id"] = frame["group_id"].astype(str)
    return frame


@lru_cache(maxsize=4)
def _outer_test_ids(dataset: str) -> set[str]:
    if not OUTER_PRED.is_file():
        return set()
    domain = "uci" if dataset == "uci" else "oulad"
    frame = pd.read_parquet(OUTER_PRED, columns=["record_id", "domain", "outer_fold"])
    keep = (frame["domain"].astype(str) == domain) & (frame["outer_fold"] == 0)
    return set(frame.loc[keep, "record_id"].astype(str))


def recovered_inner_table(dataset: str) -> pd.DataFrame:
    """Official 3×3 VALID fold identity. Not a newly sampled split."""
    if dataset == "uci":
        path = RECON / "uci" / "oof_predictions.parquet"
    else:
        path = RECON / "oulad_final" / "oof_predictions.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"cannot recover inner folds: missing {path}")
    frame = pd.read_parquet(path, columns=["record_id", "fold"]).drop_duplicates("record_id")
    frame = frame.rename(columns={"fold": "inner_fold"})
    frame["record_id"] = frame["record_id"].astype(str)
    dup = frame.groupby("record_id")["inner_fold"].nunique()
    if int((dup > 1).sum()) != 0:
        raise RuntimeError("RECOVERED_INNER_FOLD_COLLISION")
    return frame.reset_index(drop=True)


def _fit_stop_valid(dataset: str, context: pd.DataFrame, inner_fold: int) -> tuple[list[str], list[str], list[str]]:
    from sklearn.model_selection import StratifiedGroupKFold

    inner = recovered_inner_table(dataset)
    valid_ids = set(inner.loc[inner.inner_fold == inner_fold, "record_id"])
    train_ids = set(inner.loc[inner.inner_fold != inner_fold, "record_id"])
    available = set(context.record_id.astype(str))
    valid_ids &= available
    train_ids &= available
    blocked = (valid_ids | train_ids) & _outer_test_ids(dataset)
    if blocked:
        raise RuntimeError(f"OUTER_FIREWALL_VIOLATION:{dataset}:{len(blocked)}")
    assert_disjoint(train_ids, valid_ids)
    frame = context[context.record_id.astype(str).isin(train_ids)].drop_duplicates("record_id").reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    y = frame.target.to_numpy()
    groups = frame.group_id.astype(str).to_numpy()
    fit_ids = stop_ids = None
    for fit, stop in splitter.split(frame, y, groups):
        if len(np.unique(y[fit])) == 2 and len(np.unique(y[stop])) == 2 and not (set(groups[fit]) & set(groups[stop])):
            fit_ids = frame.iloc[fit].record_id.astype(str).tolist()
            stop_ids = frame.iloc[stop].record_id.astype(str).tolist()
            break
    if fit_ids is None:
        raise RuntimeError("NO_FEASIBLE_FIT_STOP")
    valid = sorted(valid_ids)
    assert_disjoint(fit_ids, stop_ids, valid)
    return fit_ids, stop_ids, valid


def partitions(dataset: str, inner_fold: int) -> tuple[list[str], list[str], list[str], dict]:
    context = uci_context() if dataset == "uci" else oulad_context()
    if frozen_split_bundle_available():
        verify_split_hashes()
        from experiments.hybrid_vnext.data import inner_partitions

        fit_ids, stop_ids, valid_ids = inner_partitions(dataset, context, inner_fold)
        meta = {"split_source": "frozen_kltn_parquet", "frozen_parquet_available": True}
    else:
        fit_ids, stop_ids, valid_ids = _fit_stop_valid(dataset, context, inner_fold)
        meta = {
            "split_source": "recovered_from_official_oof_valid",
            "frozen_parquet_available": False,
            "note": "C:\\hufit\\kltn Phase-1 parquet bundle is absent; inner_fold taken from reconstructed OOF VALID (one fold/student).",
        }
    if set(fit_ids) & set(stop_ids) or set(fit_ids) & set(valid_ids) or set(stop_ids) & set(valid_ids):
        raise RuntimeError("PARTITION_OVERLAP")
    return fit_ids, stop_ids, valid_ids, meta


def _view_to_batch(view, ids: list[str]) -> PackedBatch:
    idx = []
    lookup = {str(r): i for i, r in enumerate(np.asarray(view.record_id).astype(str))}
    for record in ids:
        if record in lookup:
            idx.append(lookup[record])
    idx = np.asarray(idx, dtype=np.int64)
    return PackedBatch(
        static=view.static[idx],
        temporal=view.temporal[idx],
        temporal_mask=view.temporal_mask[idx],
        lengths=view.lengths[idx],
        aggregate=view.aggregate[idx],
        aggregate_available=np.asarray(view.aggregate_available[idx]).astype(np.int8),
        progress=view.progress[idx],
        target=view.target[idx],
        record_id=np.asarray(view.record_id)[idx].astype(str),
    )


def build_uci_stage(stage: str, fit_ids: list[str], keep_ids: list[str]) -> PackedBatch:
    frame, _ = build_uci_combined(raw_dir() / "student-mat.csv", raw_dir() / "student-por.csv")
    frame["record_id"] = frame["record_id"].astype(str)
    fit_frame = frame[frame["record_id"].isin(set(fit_ids))].copy()
    ctx = ContextPreprocessor(UCI_NUMERIC_CONTEXT, UCI_CATEGORICAL_CONTEXT).fit(fit_frame)
    keep_frame = frame[frame["record_id"].isin(set(keep_ids))].copy()
    static = ctx.transform(keep_frame)
    view = build_uci_stage_view(keep_frame.reset_index(drop=True), stage, static=static)
    return _view_to_batch(view, keep_ids)


def build_oulad_stage(
    stage: str,
    fit_ids: list[str],
    keep_ids: list[str],
    *,
    preprocessor=None,
    vle_daily: pd.DataFrame | None = None,
):
    daily = vle_daily if vle_daily is not None else build_vle_daily(raw_dir())
    prep = preprocessor or fit_oulad_preprocessor(raw_dir(), fit_ids, vle_daily=daily, states=OULAD_STATES)
    view = build_oulad_information_state(raw_dir(), stage, vle_daily=daily, preprocessor=prep)
    return _view_to_batch(view, keep_ids), prep


def _save_batch(path: Path, batch: PackedBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        static=batch.static,
        temporal=batch.temporal,
        temporal_mask=batch.temporal_mask,
        lengths=batch.lengths,
        aggregate=batch.aggregate,
        aggregate_available=batch.aggregate_available,
        progress=batch.progress,
        target=batch.target,
        record_id=np.asarray(batch.record_id).astype("U"),
    )


def _load_batch(path: Path) -> PackedBatch:
    payload = np.load(path, allow_pickle=False)
    return PackedBatch(
        static=payload["static"],
        temporal=payload["temporal"],
        temporal_mask=payload["temporal_mask"],
        lengths=payload["lengths"],
        aggregate=payload["aggregate"],
        aggregate_available=payload["aggregate_available"],
        progress=payload["progress"],
        target=payload["target"],
        record_id=payload["record_id"].astype(str),
    )


def load_fold(dataset: str, inner_fold: int, *, vle_daily=None) -> dict:
    stages = UCI_STAGES if dataset == "uci" else OULAD_STATES
    fit_ids, stop_ids, valid_ids, split_meta = partitions(dataset, inner_fold)
    keep = list(dict.fromkeys([*fit_ids, *stop_ids, *valid_ids]))
    preprocessor = None
    train_stages: dict[str, PackedBatch] = {}
    stop_stages: dict[str, PackedBatch] = {}
    valid_stages: dict[str, PackedBatch] = {}
    for stage in stages:
        train_path = CACHE / f"{dataset}_f{inner_fold}_{stage}_train.npz"
        stop_path = CACHE / f"{dataset}_f{inner_fold}_{stage}_stop.npz"
        valid_path = CACHE / f"{dataset}_f{inner_fold}_{stage}_valid.npz"
        if train_path.is_file() and stop_path.is_file() and valid_path.is_file():
            print(f"  cache hit {dataset} fold {inner_fold} {stage}", flush=True)
            train_stages[stage] = _load_batch(train_path)
            stop_stages[stage] = _load_batch(stop_path)
            valid_stages[stage] = _load_batch(valid_path)
            continue
        print(f"  building {dataset} fold {inner_fold} {stage} ...", flush=True)
        if dataset == "uci":
            full = build_uci_stage(stage, fit_ids, keep)
        else:
            full, preprocessor = build_oulad_stage(
                stage, fit_ids, keep, preprocessor=preprocessor, vle_daily=vle_daily
            )
        train_stages[stage] = subset_batch(full, fit_ids)
        stop_stages[stage] = subset_batch(full, stop_ids)
        valid_stages[stage] = subset_batch(full, valid_ids)
        _save_batch(train_path, train_stages[stage])
        _save_batch(stop_path, stop_stages[stage])
        _save_batch(valid_path, valid_stages[stage])
    return {
        "fit_ids": fit_ids,
        "stop_ids": stop_ids,
        "valid_ids": valid_ids,
        "train_stages": train_stages,
        "stop_stages": stop_stages,
        "valid_stages": valid_stages,
        "split_meta": split_meta,
    }


__all__ = [
    "OULAD_STATES",
    "UCI_STAGES",
    "build_oulad_stage",
    "build_uci_stage",
    "frozen_split_bundle_available",
    "load_fold",
    "partitions",
    "raw_dir",
    "recovered_inner_table",
]
