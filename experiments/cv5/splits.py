"""5-fold grouped CV on the development cohort. Official outer fold 0 is a firewall."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from experiments.hybrid_vnext.protocol import DEVELOPMENT_OUTER_FOLD, assert_disjoint, split_paths, verify_split_hashes
from experiments.imbalance.data_build import oulad_context, uci_context


def outer0_ids(dataset: str) -> set[str]:
    verify_split_hashes()
    frame = pd.read_parquet(split_paths()[f"{dataset}_outer"])
    col = "record_id"
    return set(frame.loc[frame.outer_fold == DEVELOPMENT_OUTER_FOLD, col].astype(str))


def development_frame(dataset: str) -> pd.DataFrame:
    context = uci_context() if dataset == "uci" else oulad_context()
    blocked = outer0_ids(dataset)
    frame = context[~context.record_id.astype(str).isin(blocked)].drop_duplicates("record_id").reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("empty development cohort")
    return frame


def cv5_partitions(dataset: str, fold: int) -> tuple[list[str], list[str], list[str], dict]:
    """VALID = CV fold. FIT/STOP split from the other 4 folds. Outer-0 excluded."""
    if fold not in range(5):
        raise ValueError("fold must be 0..4")
    frame = development_frame(dataset)
    blocked = outer0_ids(dataset)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
    y = frame.target.to_numpy()
    groups = frame.group_id.astype(str).to_numpy()
    valid_ids = train_ids = None
    for i, (train, valid) in enumerate(splitter.split(frame, y, groups)):
        if i != fold:
            continue
        train_ids = frame.iloc[train].record_id.astype(str).tolist()
        valid_ids = frame.iloc[valid].record_id.astype(str).tolist()
        break
    if train_ids is None:
        raise RuntimeError("CV5_SPLIT_FAILED")
    if set(train_ids) & blocked or set(valid_ids) & blocked:
        raise RuntimeError("OUTER_FIREWALL_VIOLATION")
    rest = frame[frame.record_id.astype(str).isin(train_ids)].reset_index(drop=True)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fit_ids = stop_ids = None
    ry = rest.target.to_numpy()
    rg = rest.group_id.astype(str).to_numpy()
    for fit, stop in inner.split(rest, ry, rg):
        if len(np.unique(ry[fit])) == 2 and len(np.unique(ry[stop])) == 2 and not (set(rg[fit]) & set(rg[stop])):
            fit_ids = rest.iloc[fit].record_id.astype(str).tolist()
            stop_ids = rest.iloc[stop].record_id.astype(str).tolist()
            break
    if fit_ids is None:
        raise RuntimeError("NO_FEASIBLE_FIT_STOP")
    assert_disjoint(fit_ids, stop_ids, valid_ids)
    meta = {
        "protocol": "5fold_cv_development_outer0_heldout",
        "outer0_excluded": True,
        "n_fit": len(fit_ids),
        "n_stop": len(stop_ids),
        "n_valid": len(valid_ids),
    }
    return fit_ids, stop_ids, valid_ids, meta
