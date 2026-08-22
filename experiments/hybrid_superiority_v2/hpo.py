"""Optuna HPO with PostgreSQL storage. Baseline lock happens before Hybrid search."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import default_params, fit_eval, predictor_columns, sample_space
from .data import PreparedDomain, baseline_frame, inner_partitions, scale_views
from .io_utils import git_commit, sha256_json, utc_now, write_json
from .metrics import selection_objective
from .model import SuperiorityConfig, count_parameters, make_config
from .paths import METRIC_DIR, OOF_DIR, RUN_DIR, ensure_dirs
from .protocol import (
    BASELINE_ROSTER,
    HPO_BUDGET,
    PROTOCOL_ID,
    SCREEN_FOLD,
    SCREEN_SEED,
    SEEDS_ROBUST,
    protocol_hash,
    stages_for,
    warm_for,
)
from .train import HybridTrainer


def _study_name(kind: str, domain: str, extra: str) -> str:
    return f"hs_v2_{kind}_{domain}_{extra}_{protocol_hash()[:12]}"


def _optuna_storage():
    from .db import optuna_storage_url

    return optuna_storage_url()


def _make_study(name: str, direction: str = "maximize"):
    import optuna

    storage = None
    try:
        storage = _optuna_storage()
        return optuna.create_study(
            study_name=name,
            storage=storage,
            load_if_exists=True,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
    except Exception:
        return optuna.create_study(
            study_name=name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )


def run_baselines_domain(domain: str, *, n_trials: int | None = None, folds: tuple[int, ...] = (0, 1, 2), seeds: tuple[int, ...] = SEEDS_ROBUST) -> dict[str, Any]:
    ensure_dirs()
    n_trials = n_trials or (HPO_BUDGET["baseline_trials_uci"] if domain == "uci" else HPO_BUDGET["baseline_trials_oulad"])
    lock_path = RUN_DIR / f"baseline_lock_{domain}.json"
    if lock_path.exists():
        return json.loads(lock_path.read_text(encoding="utf-8"))
    import optuna

    best_params: dict[str, dict[str, Any]] = {}
    # HPO on inner fold 0 / seed 42, mean warm AP on VALID.
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    for name in BASELINE_ROSTER:
        study = _make_study(_study_name("baseline", domain, name))

        def objective(trial, name=name):
            params = sample_space(name, trial, domain=domain)
            aps = []
            for stage in stages_for(domain):
                frame = baseline_frame(prepared, stage)
                cols, cats = predictor_columns(frame)
                stage_fit = [i for i in fit_ids if i in set(frame.record_id.astype(str))]
                stage_stop = [i for i in stop_ids if i in set(frame.record_id.astype(str))]
                stage_valid = [i for i in valid_ids if i in set(frame.record_id.astype(str))]
                if len(stage_fit) < 20 or len(stage_valid) < 10:
                    continue
                try:
                    metrics = fit_eval(name, frame, cols, cats, stage_fit, stage_stop, stage_valid, SCREEN_SEED, params)
                    aps.append((stage, metrics["ap"]))
                except Exception as exc:
                    raise optuna.TrialPruned(f"{type(exc).__name__}") from exc
            warm = [ap for stage, ap in aps if stage in warm_for(domain)]
            if not warm:
                raise optuna.TrialPruned("no_warm_ap")
            return float(np.mean(warm))

        n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
        remaining = max(0, n_trials - n_done)
        if remaining:
            study.optimize(objective, n_trials=remaining, catch=(Exception,))
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if completed:
            best_params[name] = study.best_params
        else:
            best_params[name] = default_params(name, SCREEN_SEED)

    rows = []
    oof = []
    for fold in folds:
        fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
        prepared = scale_views(domain, fit_ids)
        for seed in seeds:
            for name in BASELINE_ROSTER:
                params = best_params[name]
                for stage in stages_for(domain):
                    frame = baseline_frame(prepared, stage)
                    cols, cats = predictor_columns(frame)
                    stage_fit = [i for i in fit_ids if i in set(frame.record_id.astype(str))]
                    stage_stop = [i for i in stop_ids if i in set(frame.record_id.astype(str))]
                    stage_valid = [i for i in valid_ids if i in set(frame.record_id.astype(str))]
                    if len(stage_valid) < 8:
                        continue
                    metrics = fit_eval(name, frame, cols, cats, stage_fit, stage_stop, stage_valid, seed, params)
                    rows.append(
                        {
                            "domain": domain,
                            "model": name,
                            "fold": fold,
                            "seed": seed,
                            "stage": stage,
                            "ap": metrics["ap"],
                            "roc_auc": metrics["roc_auc"],
                            "risk_f1": metrics["risk_f1"],
                            "risk_recall": metrics["risk_recall"],
                            "recall_at_20": metrics["recall_at_20"],
                            "n": metrics["n"],
                            "prevalence": metrics["prevalence"],
                        }
                    )
                    for record_id, group_id, y, p in zip(metrics["valid_record_id"], metrics["valid_group"], metrics["valid_y"], metrics["valid_p"]):
                        oof.append(
                            {
                                "domain": domain,
                                "model": name,
                                "fold": fold,
                                "seed": seed,
                                "stage": stage,
                                "record_id": str(record_id),
                                "group_id": str(group_id),
                                "y": int(y),
                                "p": float(p),
                            }
                        )
    table = pd.DataFrame(rows)
    oof_df = pd.DataFrame(oof)
    table.to_csv(METRIC_DIR / f"baseline_stage_metrics_{domain}.csv", index=False)
    oof_path = OOF_DIR / f"baseline_oof_{domain}.parquet"
    oof_df.to_parquet(oof_path, index=False)
    ceiling = (
        table.groupby(["stage", "model"])["ap"].mean().reset_index()
        .sort_values(["stage", "ap"], ascending=[True, False])
    )
    best_by_stage = ceiling.loc[ceiling.groupby("stage")["ap"].idxmax()]
    lock = {
        "domain": domain,
        "protocol_id": PROTOCOL_ID,
        "protocol_hash": protocol_hash(),
        "git_commit": git_commit(),
        "frozen_at": utc_now(),
        "best_params": best_params,
        "ceiling": best_by_stage.to_dict(orient="records"),
        "stage_best_ap": {row.stage: {"model": row.model, "ap": float(row.ap)} for row in best_by_stage.itertuples()},
        "n_trials": n_trials,
        "folds": list(folds),
        "seeds": list(seeds),
        "oof_path": str(oof_path),
        "outer_test_used": False,
        "roster": list(BASELINE_ROSTER),
    }
    write_json(lock_path, lock)
    return lock


def load_baseline_ceiling(domain: str) -> dict[str, float]:
    lock = json.loads((RUN_DIR / f"baseline_lock_{domain}.json").read_text(encoding="utf-8"))
    return {k: float(v["ap"]) for k, v in lock["stage_best_ap"].items()}


def sample_hybrid(trial, candidate: str, static_dim: int, temporal_dim: int, aggregate_dim: int) -> tuple[SuperiorityConfig, dict[str, Any]]:
    cfg = make_config(
        candidate,
        static_dim,
        temporal_dim,
        aggregate_dim,
        d_fuse=trial.suggest_categorical("d_fuse", [48, 64, 96]),
        cnn_channels=trial.suggest_categorical("cnn_channels", [24, 32, 48]),
        bilstm_hidden=trial.suggest_categorical("bilstm_hidden", [24, 32, 48]),
        dropout=trial.suggest_float("dropout", 0.15, 0.45),
    )
    train_kw = {
        "lr": trial.suggest_float("lr", 3e-5, 8e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 8e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128] if static_dim < 80 else [128, 256, 512]),
        "pos_weight_multiplier": trial.suggest_float("pos_weight_multiplier", 0.6, 1.6),
        "lambda_rank": trial.suggest_float("lambda_rank", 0.0, 0.3),
        "lambda_aux": trial.suggest_float("lambda_aux", 0.05, 0.4),
        "lambda_kd": trial.suggest_float("lambda_kd", 0.0, 0.5),
        "max_epochs": 24,
        "patience": 8,
        "use_ema": True,
        "multiprefix": True,
    }
    return cfg, train_kw


def evaluate_hybrid(
    domain: str,
    candidate: str,
    cfg: SuperiorityConfig,
    train_kw: dict[str, Any],
    *,
    fold: int,
    seed: int,
    prepared: PreparedDomain | None = None,
) -> dict[str, Any]:
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
    if prepared is None:
        prepared = scale_views(domain, fit_ids)
    cfg = copy.deepcopy(cfg)
    cfg.static_dim = prepared.static_dim
    cfg.temporal_dim = prepared.temporal_dim
    cfg.aggregate_dim = prepared.aggregate_dim
    from .teacher import teachers_for_prepared

    teacher_map = teachers_for_prepared(prepared, fit_ids, seed=seed) if float(train_kw.get("lambda_kd", 0)) > 0 else None
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
        "lambda_gate",
        "kd_temperature",
        "use_ema",
        "multiprefix",
    }
    trainer = HybridTrainer(
        prepared,
        cfg,
        seed=seed,
        teacher_map=teacher_map,
        **{k: v for k, v in train_kw.items() if k in allowed},
    )
    result = trainer.fit(fit_ids, stop_ids)
    model = result["model"]
    thresholds = trainer.fit_thresholds(model, stop_ids)
    scored = trainer.score_split(model, valid_ids, thresholds)
    stop_scored = trainer.score_split(model, stop_ids, thresholds)
    return {
        "valid": scored,
        "stop": stop_scored,
        "parameter_count": result["parameter_count"],
        "peak_vram_gb": result["peak_vram_gb"],
        "runtime_seconds": result["runtime_seconds"],
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "outer_test_used": False,
    }


def run_hybrid_screen(domain: str, candidate: str, n_trials: int | None = None) -> dict[str, Any]:
    import optuna

    ensure_dirs()
    n_trials = n_trials or HPO_BUDGET["screen_trials_per_candidate"]
    ceiling = load_baseline_ceiling(domain)
    fit_ids, _, _ = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    study = _make_study(_study_name("hybrid", domain, candidate))

    def objective(trial):
        cfg, train_kw = sample_hybrid(trial, candidate, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim)
        out = evaluate_hybrid(domain, candidate, cfg, train_kw, fold=SCREEN_FOLD, seed=SCREEN_SEED)
        ap = {stage: row["ap"] for stage, row in out["valid"]["stages"].items()}
        sel = selection_objective(ap, ceiling, domain, out["parameter_count"])
        trial.set_user_attr("ap", ap)
        trial.set_user_attr("selection", sel)
        trial.set_user_attr("params_train", train_kw)
        trial.set_user_attr("n_params", out["parameter_count"])
        return float(sel["J"])

    n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    if n_done == 0:
        try:
            study.enqueue_trial(
                {
                    "d_fuse": 64,
                    "cnn_channels": 32,
                    "bilstm_hidden": 32,
                    "dropout": 0.30,
                    "lr": 2e-4,
                    "weight_decay": 2e-4,
                    "batch_size": 64,
                    "pos_weight_multiplier": 1.0,
                    "lambda_rank": 0.15,
                    "lambda_aux": 0.25,
                    "lambda_kd": 0.40,
                }
            )
        except Exception:
            pass
    remaining = max(0, n_trials - n_done)
    if remaining:
        study.optimize(objective, n_trials=remaining, catch=(Exception,))
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    summary = {
        "domain": domain,
        "candidate": candidate,
        "n_complete": len(completed),
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else None,
        "best_user_attrs": dict(study.best_trial.user_attrs) if completed else None,
        "protocol_hash": protocol_hash(),
        "outer_test_used": False,
    }
    write_json(RUN_DIR / f"screen_{domain}_{candidate}.json", summary)
    return summary
