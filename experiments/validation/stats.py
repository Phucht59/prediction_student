"""Bootstrap CI, McNemar, DeLong. No hyperparameter search."""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_ci(y: np.ndarray, p: np.ndarray, metric, *, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        stats.append(float(metric(yy, pp)))
    if len(stats) < 20:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n_boot": len(stats)}
    arr = np.asarray(stats)
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(arr.mean()), "lo": float(lo), "hi": float(hi), "n_boot": int(len(arr))}


def bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric, *, n_boot: int = 1000, seed: int = 42):
    y = np.asarray(y, dtype=int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        diffs.append(float(metric(yy, p_a[idx]) - metric(yy, p_b[idx])))
    arr = np.asarray(diffs)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    mean = float(arr.mean())
    # two-sided bootstrap p: twice the tail beyond 0
    p_val = 2 * min((arr <= 0).mean(), (arr >= 0).mean())
    p_val = float(min(1.0, p_val))
    return {"delta_mean": mean, "delta_lo": float(lo), "delta_hi": float(hi), "p_bootstrap": p_val, "n_boot": int(len(arr))}


def mcnemar_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    a_ok = np.asarray(pred_a, dtype=int) == y
    b_ok = np.asarray(pred_b, dtype=int) == y
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p": 1.0, "odds_ratio": 1.0, "cohens_g": 0.0}
    chi = (abs(b - c) - 1) ** 2 / n_disc
    p = float(chi2.sf(chi, 1))
    oratio = (b / c) if c else float("inf")
    g = (b / n_disc) - 0.5
    return {"b": b, "c": c, "chi2": float(chi), "p": p, "odds_ratio": float(oratio), "cohens_g": float(g)}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks


def delong_roc(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> dict[str, float]:
    """DeLong 1988 test for paired ROC-AUC. Returns AUC_a, AUC_b, z, p, delta."""
    y = np.asarray(y, dtype=int)
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    pos = y == 1
    neg = ~pos
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"), "z": float("nan"), "p": float("nan"), "delta": float("nan")}

    def auc_and_v(scores):
        tx = scores[pos]
        ty = scores[neg]
        r = _midrank(np.concatenate([tx, ty]))
        rx = _midrank(tx)
        ry = _midrank(ty)
        auc = (r[:m].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (r[:m] - rx) / n
        v01 = 1 - (r[m:] - ry) / m
        return float(auc), v10, v01

    auc_a, v10a, v01a = auc_and_v(p_a)
    auc_b, v10b, v01b = auc_and_v(p_b)
    sx = np.cov(np.vstack([v10a, v10b]))
    sy = np.cov(np.vstack([v01a, v01b]))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    delta = auc_a - auc_b
    z = delta / np.sqrt(var) if var > 0 else 0.0
    p = float(2 * norm.sf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "delta": float(delta), "z": float(z), "p": p, "var": float(var)}


def pr_auc(y, p):
    return average_precision_score(y, p)


def roc_auc(y, p):
    return roc_auc_score(y, p)
