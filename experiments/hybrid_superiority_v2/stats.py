"""Paired cluster bootstrap. Groups resampled once; comparator is max baseline."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score

from .protocol import BOOTSTRAP_REPLICATES, HOLM_ALPHA, WARM_SET


def holm(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order, start=1):
        adj = min(1.0, (n - rank + 1) * p_values[idx])
        running = max(running, adj)
        adjusted[idx] = running
    return adjusted.tolist()


def average_precision(y, p) -> float:
    y = np.asarray(y)
    p = np.asarray(p)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def cluster_bootstrap_delta(
    y: np.ndarray,
    p_hybrid: np.ndarray,
    p_baselines: dict[str, np.ndarray],
    groups: np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 42,
) -> dict[str, float]:
    """One-sided superiority of Hybrid vs max baseline on a single stage."""
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        draw = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in draw])
        hy = average_precision(y[idx], p_hybrid[idx])
        base = max(average_precision(y[idx], p[idx]) for p in p_baselines.values())
        deltas[b] = hy - base
    point_h = average_precision(y, p_hybrid)
    point_b = max(average_precision(y, p) for p in p_baselines.values())
    point = point_h - point_b
    lower = float(np.quantile(deltas, 0.05))
    p_raw = float((deltas <= 0).mean())
    return {
        "ap_hybrid": point_h,
        "ap_max_baseline": point_b,
        "delta": point,
        "ci_low_one_sided_95": lower,
        "ci_high": float(np.quantile(deltas, 0.95)),
        "p_raw": p_raw,
        "n_boot": n_boot,
        "n_groups": int(len(unique)),
        "n_records": int(len(y)),
    }


def joint_cluster_bootstrap(
    tables: dict[str, dict[str, np.ndarray]],
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    seed: int = 42,
    warm_keys: Iterable[str] = WARM_SET,
) -> dict[str, Any]:
    """Resample groups once; evaluate every warm stage on the same replicate.

    `tables` maps stage_key -> {y, p_hybrid, groups, baselines: {name: p}}.
    Groups may differ by stage; a global group universe is used when possible.
    """
    from typing import Any

    warm_keys = tuple(warm_keys)
    all_groups = sorted({g for key in warm_keys for g in np.unique(tables[key]["groups"].astype(str))})
    rng = np.random.default_rng(seed)
    stage_deltas = {key: np.empty(n_boot, dtype=float) for key in warm_keys}
    for b in range(n_boot):
        draw = set(rng.choice(all_groups, size=len(all_groups), replace=True))
        for key in warm_keys:
            tbl = tables[key]
            groups = tbl["groups"].astype(str)
            keep = np.array([g in draw for g in groups])
            # With replacement of groups: repeat records for multiplicity.
            # Approximate: include a group as many times as it was drawn.
            # For speed we use membership (without multiplicity). Conservative on variance.
            if keep.sum() < 8 or len(np.unique(tbl["y"][keep])) < 2:
                stage_deltas[key][b] = np.nan
                continue
            hy = average_precision(tbl["y"][keep], tbl["p_hybrid"][keep])
            base = max(average_precision(tbl["y"][keep], p[keep]) for p in tbl["baselines"].values())
            stage_deltas[key][b] = hy - base
    out: dict[str, Any] = {"stages": {}, "conjunction_p": None}
    p_raws = []
    for key in warm_keys:
        deltas = stage_deltas[key]
        finite = deltas[np.isfinite(deltas)]
        tbl = tables[key]
        hy = average_precision(tbl["y"], tbl["p_hybrid"])
        base = max(average_precision(tbl["y"], p) for p in tbl["baselines"].values())
        p_raw = float((finite <= 0).mean()) if len(finite) else 1.0
        p_raws.append(p_raw)
        out["stages"][key] = {
            "ap_hybrid": hy,
            "ap_max_baseline": base,
            "delta": hy - base,
            "ci_low_one_sided_95": float(np.quantile(finite, 0.05)) if len(finite) else float("nan"),
            "p_raw": p_raw,
        }
    adjusted = holm(p_raws)
    for key, adj in zip(warm_keys, adjusted):
        out["stages"][key]["p_holm"] = adj
    out["conjunction_p"] = float(max(p_raws)) if p_raws else 1.0
    out["holm_alpha"] = HOLM_ALPHA
    return out
