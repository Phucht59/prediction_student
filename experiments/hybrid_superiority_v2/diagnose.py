"""C0-R diagnostics: shuffle gap, length shortcut, gate mass, capacity."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import inner_partitions, load_cached, permute_temporal, scale_views
from .io_utils import write_json
from .metrics import binary_metrics
from .model import availability_cases, count_parameters, make_config
from .paths import RUN_DIR, ensure_dirs
from .protocol import SCREEN_FOLD, SCREEN_SEED
from .train import HybridTrainer


def length_shortcut_oulad() -> dict[str, Any]:
    from src.prediction.data.oulad import load_oulad_static_tables
    from .paths import DATA_ROOT

    _, _, base = load_oulad_static_tables(DATA_ROOT)
    course_end = base.module_presentation_length.to_numpy(np.int64)
    unreg = base.date_unregistration.to_numpy(dtype=float)
    unreg = np.where(np.isfinite(unreg), unreg, course_end)
    length = np.clip(np.minimum(course_end, unreg).astype(np.int64), 0, course_end)
    weeks = np.maximum(0, np.ceil(length / 7.0)).astype(np.int64)
    withdrawn = (base.final_result.astype(str) == "Withdrawn").to_numpy()
    risk = base.target.to_numpy()
    fail = (base.final_result.astype(str) == "Fail").to_numpy()
    completed = base.final_result.astype(str).isin(["Pass", "Distinction", "Fail"]).to_numpy()
    short = weeks <= 20
    score = -weeks.astype(np.float64)
    return {
        "n": int(len(base)),
        "withdrawn_rate": float(withdrawn.mean()),
        "mean_weeks_by_label": {
            str(label): float(weeks[base.final_result.astype(str) == label].mean())
            for label in sorted(base.final_result.astype(str).unique())
        },
        "short_history_withdrawn_rate": float(withdrawn[short].mean()) if short.any() else None,
        "length_ap_withdrawn": float(average_precision_score(withdrawn, score)),
        "length_ap_risk": float(average_precision_score(risk, score)),
        "fail_vs_success_length_ap": float(average_precision_score(fail[completed], score[completed])) if completed.any() else None,
        "flagged_shortcut_risk": bool(short.any() and withdrawn[short].mean() >= 0.95),
        "used_for_architecture_selection": False,
        "outer_test_used": False,
    }


def diagnose_candidate(domain: str, candidate: str = "C0-R") -> dict[str, Any]:
    ensure_dirs()
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    cfg = make_config(candidate, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim)
    from .model import SuperiorityHybrid

    probe = SuperiorityHybrid(cfg)
    avail = availability_cases(probe)
    trainer = HybridTrainer(prepared, cfg, seed=SCREEN_SEED, max_epochs=12, patience=5, batch_size=64 if domain == "uci" else 128)
    result = trainer.fit(fit_ids, stop_ids)
    model = result["model"]
    thresholds = trainer.fit_thresholds(model, stop_ids)
    valid = trainer.score_split(model, valid_ids, thresholds)
    shuffle_ap = {}
    identity_ap = {stage: row["ap"] for stage, row in valid["stages"].items()}
    for stage, view in prepared.views.items():
        present = [i for i in valid_ids if i in set(map(str, view.record_id))]
        if len(present) < 8:
            continue
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        y = view.target[[lookup[i] for i in present]]
        if len(np.unique(y)) < 2:
            continue
        p_shuf = trainer._predict(model, stage, present, temporal_mode="shuffle")
        shuffle_ap[stage] = float(binary_metrics(y, p_shuf)["ap"])
    gaps = {stage: identity_ap.get(stage, float("nan")) - shuffle_ap.get(stage, float("nan")) for stage in identity_ap}
    payload = {
        "domain": domain,
        "candidate": candidate,
        "parameter_count": count_parameters(model),
        "availability_cases": avail,
        "availability_pass": all(row["pass"] for row in avail),
        "valid_ap": identity_ap,
        "shuffle_ap": shuffle_ap,
        "full_minus_shuffle": gaps,
        "order_used": any(abs(v) >= 0.003 for v in gaps.values() if np.isfinite(v)),
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "peak_vram_gb": result["peak_vram_gb"],
        "outer_test_used": False,
    }
    if domain == "oulad":
        payload["length_shortcut"] = length_shortcut_oulad()
    write_json(RUN_DIR / f"diagnose_{domain}_{candidate}.json", payload)
    return payload
