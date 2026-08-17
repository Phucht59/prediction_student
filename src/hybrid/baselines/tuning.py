"""Deterministic, SQLite-resumable study orchestration."""
from __future__ import annotations
import hashlib
import optuna
from pathlib import Path
from .objective import run_inner_oof_trial

def study_name(domain, stage, outer_fold, family): return f"baseline__{domain}__{stage}__outer{outer_fold}__{family}__p4"
def seed_for(*parts): return int(hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:8],16)
def open_study(runtime, domain, stage, outer_fold, family):
    name=study_name(domain,stage,outer_fold,family); seed=seed_for("hybrid_binary_risk_phase2",domain,stage,outer_fold,family)
    return optuna.create_study(study_name=name,storage=f"sqlite:///{Path(runtime) / 'optuna' / 'baselines.db'}",load_if_exists=True,direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed),pruner=optuna.pruners.MedianPruner(n_startup_trials=5,n_warmup_steps=2)), seed
