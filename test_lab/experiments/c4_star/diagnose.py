"""Temporal-order diagnostics after convergence. Inner VALID only."""
from __future__ import annotations

from typing import Any

import numpy as np

from experiments.hybrid_superiority_v2.data import inner_partitions, scale_views
from experiments.hybrid_superiority_v2.metrics import binary_metrics
from experiments.hybrid_superiority_v2.model import make_config
from experiments.hybrid_superiority_v2.io_utils import write_json
from experiments.hybrid_superiority_v2.train import HybridTrainer

from .paths import RUN_DIR, ensure_dirs
from .thermal import wait_if_hot


def diagnose_backbone(domain: str, candidate: str, seed: int = 42, fold: int = 0, max_epochs: int = 24) -> dict[str, Any]:
    ensure_dirs()
    wait_if_hot()
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
    prepared = scale_views(domain, fit_ids)
    cfg = make_config(candidate, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim)
    train_kw = {
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "batch_size": 64 if domain == "uci" else 256,
        "lambda_kd": 0.0,
        "max_epochs": max_epochs,
        "patience": 8,
        "use_ema": True,
        "multiprefix": True,
    }
    trainer = HybridTrainer(
        prepared,
        cfg,
        seed=seed,
        lr=train_kw["lr"],
        weight_decay=train_kw["weight_decay"],
        batch_size=train_kw["batch_size"],
        max_epochs=max_epochs,
        patience=8,
        lambda_kd=0.0,
        use_ema=True,
        multiprefix=True,
    )
    result = trainer.fit(fit_ids, stop_ids)
    model = result["model"]
    identity = {}
    shuffle = {}
    reverse = {}
    for stage, view in prepared.views.items():
        present = [i for i in valid_ids if i in set(map(str, view.record_id))]
        if len(present) < 8:
            continue
        lookup = {str(r): i for i, r in enumerate(view.record_id)}
        y = view.target[[lookup[i] for i in present]]
        if len(np.unique(y)) < 2:
            continue
        identity[stage] = float(binary_metrics(y, trainer._predict(model, stage, present, "identity"))["ap"])
        shuffle[stage] = float(binary_metrics(y, trainer._predict(model, stage, present, "shuffle"))["ap"])
        reverse[stage] = float(binary_metrics(y, trainer._predict(model, stage, present, "reverse"))["ap"])
    payload = {
        "domain": domain,
        "candidate": candidate,
        "seed": seed,
        "fold": fold,
        "identity": identity,
        "shuffle": shuffle,
        "reverse": reverse,
        "shuffle_gap": {s: identity[s] - shuffle[s] for s in identity},
        "reverse_gap": {s: identity[s] - reverse.get(s, float("nan")) for s in identity},
        "n_params": result["parameter_count"],
        "outer_test_used": False,
        "best_epoch": result["best_epoch"],
    }
    write_json(RUN_DIR / f"diagnose_{domain}_{candidate}_s{seed}.json", payload)
    return payload
