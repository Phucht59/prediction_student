"""Evaluate C0–C3 (parent Hybrid) and C4-STAR on inner splits only."""
from __future__ import annotations

from typing import Any

from experiments.hybrid_superiority_v2.data import inner_partitions, scale_views
from experiments.hybrid_superiority_v2.hpo import evaluate_hybrid
from experiments.hybrid_superiority_v2.model import make_config as make_v2_config

from .model import make_c4_config
from .objective import constrained_J
from .train import C4Trainer


def eval_v2_hybrid(domain: str, candidate: str, fold: int, seed: int, *, max_epochs: int = 24, patience: int = 8, batch_size: int = 256) -> dict[str, Any]:
    fit_ids, _, _ = inner_partitions(domain, fold)
    prepared = scale_views(domain, fit_ids)
    cfg = make_v2_config(candidate, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim)
    train_kw = {
        "lr": 2e-4,
        "weight_decay": 2e-4,
        "batch_size": 64 if domain == "uci" else batch_size,
        "pos_weight_multiplier": 1.0,
        "lambda_rank": 0.15,
        "lambda_aux": 0.25,
        "lambda_kd": 0.0,
        "max_epochs": max_epochs,
        "patience": patience,
        "use_ema": True,
        "multiprefix": True,
    }
    return evaluate_hybrid(domain, candidate, cfg, train_kw, fold=fold, seed=seed, prepared=prepared)


def eval_c4(
    domain: str,
    fold: int,
    seed: int,
    *,
    mechanism: str = "M3",
    teacher_map: dict | None = None,
    max_epochs: int = 32,
    patience: int = 10,
    batch_size: int = 256,
    **train_kw,
) -> dict[str, Any]:
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
    prepared = scale_views(domain, fit_ids)
    cfg = make_c4_config(
        prepared.static_dim,
        prepared.temporal_dim,
        prepared.aggregate_dim,
        mechanism=mechanism,
        d_fuse=int(train_kw.pop("d_fuse", 64)),
        cnn_channels=int(train_kw.pop("cnn_channels", 32)),
        bilstm_hidden=int(train_kw.pop("bilstm_hidden", 48)),
        dropout=float(train_kw.pop("dropout", 0.25)),
        initial_alpha=float(train_kw.pop("initial_alpha", 0.05)),
        branch_mode=train_kw.pop("branch_mode", "full"),
    )
    allowed = {
        "lr",
        "weight_decay",
        "max_epochs",
        "patience",
        "batch_size",
        "pos_weight_multiplier",
        "lambda_rank",
        "lambda_kd",
        "lambda_aux",
        "lambda_ssl",
        "lambda_gate",
        "kd_temperature",
        "use_ema",
        "group_dro",
        "multiprefix",
        "freeze_anchor_epochs",
    }
    trainer = C4Trainer(
        prepared,
        cfg,
        seed=seed,
        teacher_map=teacher_map,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=64 if domain == "uci" else batch_size,
        **{k: v for k, v in train_kw.items() if k in allowed},
    )
    result = trainer.fit(fit_ids, stop_ids)
    model = result["model"]
    thresholds = trainer.fit_thresholds(model, stop_ids)
    scored = trainer.score_split(model, valid_ids, thresholds)
    return {
        "valid": scored,
        "parameter_count": result["parameter_count"],
        "peak_vram_gb": result["peak_vram_gb"],
        "runtime_seconds": result["runtime_seconds"],
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "outer_test_used": False,
        "mechanism": mechanism,
    }


def score_against_ceiling(domain: str, out: dict[str, Any], ceiling: dict[str, float]) -> dict[str, Any]:
    ap = {stage: row["ap"] for stage, row in out["valid"]["stages"].items()}
    sel = constrained_J(ap, ceiling, domain, n_params=int(out.get("parameter_count") or 0))
    return {"ap": ap, "selection": sel}
