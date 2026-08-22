"""Cross-fitted GBDT teacher on FIT only. Soft labels never see STOP/VALID/outer."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .baselines import default_params, fit_eval, make_estimator, predictor_columns, preprocessor


def crossfit_xgb_teacher(
    frame: pd.DataFrame,
    fit_ids: list[str],
    *,
    seed: int = 42,
    n_splits: int = 3,
) -> dict[str, float]:
    """OOF probabilities for FIT records. Empty dict if the split is infeasible."""
    cols, cats = predictor_columns(frame)
    subset = frame[frame.record_id.astype(str).isin(set(fit_ids))].drop_duplicates("record_id").reset_index(drop=True)
    if len(subset) < 30 or subset.target.nunique() < 2:
        return {}
    y = subset.target.to_numpy()
    groups = subset.group_id.astype(str).to_numpy()
    rec = subset.record_id.astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(subset), np.nan, dtype=np.float32)
    params = default_params("XGB", seed)
    params["n_estimators"] = 200
    params["max_depth"] = 4
    params["early_stopping"] = False
    try:
        splits = list(splitter.split(subset, y, groups))
    except Exception:
        return {}
    for train_idx, test_idx in splits:
        if len(np.unique(y[train_idx])) < 2 or len(test_idx) == 0:
            continue
        train = subset.iloc[train_idx]
        test = subset.iloc[test_idx]
        prep = preprocessor(train, cols, cats)
        x_train = prep.fit_transform(train)
        x_test = prep.transform(test)
        model = make_estimator("XGB", params, seed, train.target.to_numpy())
        try:
            model.fit(x_train, train.target.to_numpy())
        except TypeError:
            model.fit(x_train, train.target.to_numpy())
        if hasattr(model, "predict_proba"):
            oof[test_idx] = model.predict_proba(x_test)[:, 1]
    ok = np.isfinite(oof)
    return {str(r): float(p) for r, p, keep in zip(rec, oof, ok) if keep}


def teachers_for_prepared(prepared, fit_ids: list[str], seed: int = 42) -> dict[str, dict[str, float]]:
    from .data import baseline_frame

    out: dict[str, dict[str, float]] = {}
    for stage in prepared.views:
        frame = baseline_frame(prepared, stage)
        out[stage] = crossfit_xgb_teacher(frame, fit_ids, seed=seed)
    return out
