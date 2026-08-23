"""Four-tier evaluation. final_result is eval-only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score

from .contracts import PersistLabel
from .feasibility import invalid_action
from .model import CLASSES, macro_average_precision
from .policy import attach_worklist


def targeting_table(frame: pd.DataFrame, *, k_values: tuple[float, ...] = (0.05, 0.10, 0.15)) -> pd.DataFrame:
    if "y" not in frame.columns:
        raise ValueError("targeting requires y")
    rows = []
    for stage, group in frame.groupby("stage", sort=True):
        n = len(group)
        n_pos = int(group["y"].sum())
        flagged = group["risk_probability"].astype(float) >= group["prediction_threshold"].astype(float)
        k_flag = int(flagged.sum())
        rec_flag = float(group.loc[flagged, "y"].sum() / n_pos) if n_pos else 0.0
        prec_flag = float(group.loc[flagged, "y"].mean()) if k_flag else 0.0
        ceiling = min(1.0, 0.10 / (n_pos / n)) if n_pos and n else 0.0
        row = {
            "stage": stage,
            "n": n,
            "positives": n_pos,
            "flag_frac": float(flagged.mean()),
            "recall_flag": rec_flag,
            "precision_flag": prec_flag,
            "recall_ceiling_10": ceiling,
        }
        order = np.argsort(-group["risk_probability"].to_numpy())
        y = group["y"].to_numpy()
        for frac in k_values:
            k = max(1, int(round(n * frac)))
            picked = y[order[:k]]
            rec = float(picked.sum() / n_pos) if n_pos else 0.0
            prec = float(picked.mean())
            key = int(frac * 100)
            row[f"recall@{key}"] = rec
            row[f"precision@{key}"] = prec
            row[f"k@{key}"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def model_scores(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None = None) -> dict:
    out = {
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(CLASSES), zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=list(CLASSES))),
        "accuracy": float(np.mean(y_true == y_pred)),
    }
    if proba is not None:
        out["macro_ap"] = macro_average_precision(y_true, proba, CLASSES)
        for i, name in enumerate(CLASSES):
            truth = (y_true == name).astype(int)
            if truth.sum() and truth.sum() < len(truth):
                out[f"ap_{name}"] = float(average_precision_score(truth, proba[:, i]))
    for name in CLASSES:
        out[f"support_{name}"] = int((y_true == name).sum())
    return out


def feasibility_audit(frame: pd.DataFrame, action_col: str = "pred_action") -> dict:
    invalid = 0
    for _, row in frame.iterrows():
        label = PersistLabel(str(row[action_col]))
        if invalid_action(label, row.to_dict()):
            invalid += 1
    work = frame.loc[frame["in_worklist"] == True] if "in_worklist" in frame.columns else frame
    dead = 0
    if not work.empty:
        dead = int((work[action_col].astype(str) == "").sum())
    return {
        "n": int(len(frame)),
        "invalid_action_rate": float(invalid / max(len(frame), 1)),
        "worklist_n": int(len(work)),
        "dead_end_rate": float(dead / max(len(work), 1)),
        "counsel_share_worklist": float((work[action_col].astype(str) == PersistLabel.COUNSEL.value).mean())
        if len(work)
        else 0.0,
    }


def persistence_oracle(frame: pd.DataFrame) -> np.ndarray:
    """Oracle uses 14-day stuck flags, not Pass/Fail."""
    return frame["persist_label"].astype(str).to_numpy()


def bootstrap_beta1(
    y: np.ndarray,
    resolved: np.ndarray,
    p: np.ndarray,
    *,
    iterations: int = 400,
    seed: int = 2026,
) -> dict:
    rng = np.random.default_rng(seed)
    coefs = []
    n = len(y)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        X = np.column_stack([resolved[idx], p[idx]])
        clf = LogisticRegression(max_iter=200)
        clf.fit(X, yy)
        coefs.append(float(clf.coef_[0, 0]))
    arr = np.asarray(coefs, dtype=np.float64)
    if arr.size == 0:
        return {"beta1_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_positive": 0.0}
    return {
        "beta1_mean": float(arr.mean()),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "p_positive": float((arr > 0).mean()),
        "n_boot": int(arr.size),
    }


def tier4_block(frame: pd.DataFrame, action_col: str = "pred_action") -> dict:
    work = attach_worklist(frame) if "in_worklist" not in frame.columns else frame
    work = work.loc[work["in_worklist"] == True].copy()
    if work.empty or "y" not in work.columns:
        return {"n": 0}
    resolved = np.zeros(len(work), dtype=np.int64)
    mismatch = np.zeros(len(work), dtype=np.int64)
    for i, rec in enumerate(work.itertuples(index=False)):
        action = str(getattr(rec, action_col))
        if action == PersistLabel.ASSESS.value:
            resolved[i] = int(bool(getattr(rec, "assess_resolved", False)))
            mismatch[i] = int(bool(getattr(rec, "vle_returned", False)))
        elif action == PersistLabel.ENGAGE.value:
            resolved[i] = int(bool(getattr(rec, "engage_resolved", False)))
            mismatch[i] = int(bool(getattr(rec, "assess_resolved", False)))
    fail = work["y"].to_numpy(dtype=np.int64)
    passed = 1 - fail
    p = work["risk_probability"].to_numpy(dtype=np.float64)
    matched = bootstrap_beta1(passed, resolved, p)
    mismatched = bootstrap_beta1(passed, mismatch, p)
    return {
        "n_worklist": int(len(work)),
        "resolved_rate": float(resolved.mean()),
        "mismatch_rate": float(mismatch.mean()),
        "outcome": "Pass_or_Distinction",
        "matched": matched,
        "mismatched": mismatched,
        "specificity_holds": bool(matched["beta1_mean"] > mismatched["beta1_mean"]),
    }


__all__ = [
    "feasibility_audit",
    "model_scores",
    "persistence_oracle",
    "targeting_table",
    "tier4_block",
]
