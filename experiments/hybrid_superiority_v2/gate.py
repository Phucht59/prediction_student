"""Development-gate evaluator. Never opens confirmation on a failed gate."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import write_json
from .paths import METRIC_DIR, RUN_DIR
from .protocol import (
    ABLATION_MARGIN,
    COLD_GUARDRAIL,
    material_margin,
    protocol_hash,
    stages_for,
    warm_for,
)


def evaluate_development_gate(domain: str, hybrid_rows: pd.DataFrame, ceiling: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """hybrid_rows columns: stage, seed, fold, ap, recall_at_20."""
    mean_ap = hybrid_rows.groupby("stage")["ap"].mean().to_dict()
    warm = warm_for(domain)
    cold = [s for s in stages_for(domain) if s not in warm]
    checks = []
    fail = 0
    for stage in warm:
        b = float(ceiling[stage]["ap"])
        h = float(mean_ap[stage])
        delta = h - b
        mm = material_margin(b)
        ok_pos = delta > 0
        ok_mat = delta >= mm
        checks.append({"stage": stage, "ap_hybrid": h, "ap_baseline": b, "delta": delta, "material_margin": mm, "pass_positive": ok_pos, "pass_material": ok_mat})
        if not ok_pos or not ok_mat:
            fail += 1
    cold_ok = True
    for stage in cold:
        key = f"{domain}:{stage}"
        allow = COLD_GUARDRAIL.get(key, 0.05)
        b = float(ceiling[stage]["ap"])
        h = float(mean_ap.get(stage, float("nan")))
        ok = (b - h) <= allow
        checks.append({"stage": stage, "ap_hybrid": h, "ap_baseline": b, "delta": h - b, "guardrail": allow, "pass_cold": ok})
        cold_ok = cold_ok and ok
    payload = {
        "domain": domain,
        "protocol_hash": protocol_hash(),
        "pass": fail == 0 and cold_ok,
        "n_warm_fail": fail,
        "cold_ok": cold_ok,
        "checks": checks,
        "outer_test_used": False,
        "ablation_margin_required": ABLATION_MARGIN,
    }
    write_json(RUN_DIR / f"development_gate_{domain}.json", payload)
    return payload
