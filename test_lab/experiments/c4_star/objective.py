"""Constrained Optuna objective. Warm-stage losses are heavily penalized."""
from __future__ import annotations

import numpy as np

from .protocol import material_margin, warm_for, cold_for
from experiments.hybrid_superiority_v2.protocol import COLD_GUARDRAIL


def constrained_J(
    hybrid_ap: dict[str, float],
    ceiling_ap: dict[str, float],
    domain: str,
    *,
    seed_std: float = 0.0,
    ece: float = 0.0,
    n_params: int = 0,
) -> dict[str, float]:
    warm = warm_for(domain)
    r = []
    n_loss = 0
    loss_mag = 0.0
    for stage in warm:
        b = float(ceiling_ap[stage])
        h = float(hybrid_ap[stage])
        delta = h - b
        mm = material_margin(b)
        r.append(delta / mm)
        if delta < 0:
            n_loss += 1
            loss_mag += -delta
    r_arr = np.asarray(r, dtype=float)
    softmin = float(-np.logaddexp.reduce(-r_arr) + np.log(len(r_arr))) if len(r_arr) else float("nan")
    cold_pen = 0.0
    for stage in cold_for(domain):
        allow = COLD_GUARDRAIL.get(f"{domain}:{stage}", 0.05)
        b = float(ceiling_ap[stage])
        h = float(hybrid_ap.get(stage, float("nan")))
        drop = b - h
        if np.isfinite(drop) and drop > allow:
            cold_pen += 8.0 * (drop - allow)
    hard = 25.0 * n_loss + 40.0 * loss_mag
    j = (
        softmin
        + 0.25 * float(r_arr.mean())
        - 0.15 * float(seed_std)
        - 0.05 * float(max(ece, 0.0))
        - cold_pen
        - hard
        - 0.02 * max(0.0, np.log10(max(n_params, 1) / 150_000))
    )
    return {
        "J": float(j),
        "n_warm_loss": int(n_loss),
        "loss_magnitude": float(loss_mag),
        "softmin_r": softmin,
        "mean_r": float(r_arr.mean()) if len(r_arr) else float("nan"),
        "min_r": float(r_arr.min()) if len(r_arr) else float("nan"),
        "cold_penalty": float(cold_pen),
        "r": {s: float(v) for s, v in zip(warm, r)},
    }
