"""Phase 3: numeric HPO + inner gate + optional one-shot outer. Topology C0 frozen."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hybrid_vnext.baselines import fit_eval_baseline
from experiments.hybrid_vnext.data import (
    baseline_frame,
    feature_groups,
    final100_length_diagnostic,
    inner_partitions,
    load_domain,
    outer_holdout_ids,
    scale_views,
)
from experiments.hybrid_vnext.metrics import binary_metrics, expected_calibration_error, select_stop_threshold
from experiments.hybrid_vnext.model import VNextHybrid, assert_c0_topology, availability_unit_cases, make_c0_config
from experiments.hybrid_vnext.phase3_common import (
    INNER_FOLDS,
    OULAD_OUTER_FOLDS,
    PHASE2_OULAD_REF,
    PHASE2_UCI_REF,
    PHASE3,
    REPORTS3,
    RUNS3,
    SCREEN_FOLD,
    SCREEN_SEED,
    SEEDS,
    TOPOLOGY_SPEC,
    UCI_1FOLD_FLOOR,
    UCI_OUTER_FOLDS,
    UCI_ROBUST_FLOOR,
    UCI_STD_CEILING,
    topology_hash,
    verify_phase2_locks,
)
from experiments.hybrid_vnext.protocol import git_commit, require_cuda, run_metadata, sha256_file, utc_now, write_json
from experiments.hybrid_vnext.train import Trainer, _ids_for_stage


HYBRID_FIELDS = [
    "run_id", "trial_id", "timestamp", "stage_name", "dataset", "stage", "architecture_id",
    "d_fuse", "cnn_channels", "bilstm_hidden", "lr", "weight_decay", "dropout",
    "batch_size", "pos_weight_multiplier", "entropy_floor_coefficient", "inner_fold", "seed",
    "pr_auc", "risk_f1", "risk_recall", "risk_precision", "accuracy", "balanced_accuracy",
    "ece", "brier", "nll", "stop_pr_auc", "train_pr_auc", "generalization_gap",
    "best_epoch", "parameter_count", "device", "amp", "outer_test_used", "status",
]


def _csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _append_cuda(job: dict) -> None:
    path = PHASE3 / "CUDA_EXECUTION_AUDIT.json"
    cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"jobs": []}
    cur.setdefault("jobs", []).append(job)
    write_json(path, cur)


def prepare(domain: str, inner_fold: int):
    views, context, numeric, categorical = load_domain(domain)
    fit, stop, valid = inner_partitions(domain, context, inner_fold)
    prepared = scale_views(views, context, numeric, categorical, fit, domain)
    return prepared, fit, stop, valid


def hybrid_run(
    *,
    domain: str,
    fold: int,
    seed: int,
    d_fuse: int,
    cnn_channels: int,
    bilstm_hidden: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    pos_weight_multiplier: float,
    entropy_floor: float,
    max_epochs: int,
    patience: int,
    stage_name: str,
    trial_id: str = "",
    epoch_callback=None,
    fixed_epochs: int | None = None,
    train_ids=None,
    stop_ids=None,
    valid_ids=None,
    prepared=None,
) -> dict:
    run_id = (
        f"{stage_name}_{domain}_d{d_fuse}_c{cnn_channels}_h{bilstm_hidden}_"
        f"lr{lr}_wd{weight_decay}_do{dropout}_b{batch_size}_pw{pos_weight_multiplier}_"
        f"ent{entropy_floor}_f{fold}_s{seed}"
    )
    existing = RUNS3 / f"{run_id}.json"
    if existing.exists() and fixed_epochs is None:
        return json.loads(existing.read_text(encoding="utf-8"))
    if prepared is None:
        prepared, fit, stop, valid = prepare(domain, fold)
    else:
        fit, stop, valid = train_ids, stop_ids, valid_ids
    if train_ids is not None:
        fit = train_ids
    if stop_ids is not None:
        stop = stop_ids
    if valid_ids is not None:
        valid = valid_ids
    cfg = make_c0_config(
        prepared.static_dim,
        prepared.temporal_dim,
        prepared.aggregate_dim,
        prepared.summary_dim,
        d_fuse=d_fuse,
        cnn_channels=cnn_channels,
        bilstm_hidden=bilstm_hidden,
        dropout=dropout,
        entropy_floor_coefficient=entropy_floor,
    )
    assert_c0_topology(cfg)
    trainer = Trainer(
        prepared,
        cfg,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        seed=seed,
        pos_weight_multiplier=pos_weight_multiplier,
        epoch_callback=epoch_callback,
        fixed_epochs=fixed_epochs,
    )
    try:
        fitted = trainer.fit(fit, stop)
        model = fitted["model"]
        evaluation = trainer.evaluate(model, stop, valid) if valid else {"stages": {}, "macro_pr_auc": float("nan"), "worst_pr_auc": float("nan"), "diagnostics": {}}
    except (RuntimeError, ValueError) as exc:
        if "NONFINITE" not in str(exc) and "nonfinite" not in str(exc).lower():
            raise
        torch.cuda.empty_cache()
        trainer = Trainer(
            prepared, cfg, lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, patience=patience,
            batch_size=batch_size, seed=seed, pos_weight_multiplier=pos_weight_multiplier,
            epoch_callback=epoch_callback, fixed_epochs=fixed_epochs, amp=False,
        )
        fitted = trainer.fit(fit, stop)
        model = fitted["model"]
        evaluation = trainer.evaluate(model, stop, valid) if valid else {"stages": {}, "macro_pr_auc": float("nan"), "worst_pr_auc": float("nan"), "diagnostics": {}}
        fitted["amp"] = False
    rows = []
    for stage, metrics in evaluation.get("stages", {}).items():
        train_ids_s = _ids_for_stage(prepared.views[stage], fit)
        lookup = {str(r): i for i, r in enumerate(prepared.views[stage].record_id)}
        train_y = prepared.views[stage].target[[lookup[i] for i in train_ids_s]]
        train_p = trainer._predict(model, stage, train_ids_s)
        train_pr = binary_metrics(train_y, train_p)["pr_auc"] if len(np.unique(train_y)) == 2 else float("nan")
        rows.append(
            {
                "run_id": run_id,
                "trial_id": trial_id,
                "timestamp": utc_now(),
                "stage_name": stage_name,
                "dataset": domain,
                "stage": stage,
                "architecture_id": "C0",
                "d_fuse": d_fuse,
                "cnn_channels": cnn_channels,
                "bilstm_hidden": bilstm_hidden,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "batch_size": fitted["batch_size"],
                "pos_weight_multiplier": pos_weight_multiplier,
                "entropy_floor_coefficient": entropy_floor,
                "inner_fold": fold,
                "seed": seed,
                "pr_auc": metrics["pr_auc"],
                "risk_f1": metrics["risk_f1"],
                "risk_recall": metrics["risk_recall"],
                "risk_precision": metrics["risk_precision"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "ece": metrics["ece"],
                "brier": metrics.get("brier"),
                "nll": metrics.get("nll"),
                "stop_pr_auc": metrics.get("stop_pr_auc"),
                "train_pr_auc": train_pr,
                "generalization_gap": float(train_pr - metrics["pr_auc"]) if np.isfinite(train_pr) else None,
                "best_epoch": fitted["best_epoch"],
                "parameter_count": fitted["parameter_count"],
                "device": "cuda:0",
                "amp": fitted.get("amp", True),
                "outer_test_used": False,
                "status": "ok",
            }
        )
    if rows:
        _csv(PHASE3 / "ALL_HYBRID_RUNS.csv", rows, HYBRID_FIELDS)
    payload = run_metadata(
        run_id=run_id,
        trial_id=trial_id,
        architecture_id="C0",
        topology_hash=topology_hash(),
        domain=domain,
        fold=fold,
        seed=seed,
        structural={"d_fuse": d_fuse, "cnn_channels": cnn_channels, "bilstm_hidden": bilstm_hidden},
        training={
            "lr": lr, "weight_decay": weight_decay, "dropout": dropout, "batch_size": fitted["batch_size"],
            "pos_weight_multiplier": pos_weight_multiplier, "entropy_floor_coefficient": entropy_floor,
            "max_epochs": max_epochs, "patience": patience,
        },
        fitted={k: fitted[k] for k in ("best_epoch", "best_stop_macro_pr_auc", "parameter_count", "peak_vram_bytes", "runtime_seconds", "batch_size", "device", "amp", "gpu_name", "history") if k in fitted},
        evaluation=evaluation,
        rows=rows,
        outer_test_used=False,
    )
    write_json(RUNS3 / f"{run_id}.json", payload)
    _append_cuda(
        {
            "run_id": run_id,
            "device": "cuda:0",
            "gpu_name": fitted.get("gpu_name"),
            "amp": fitted.get("amp", True),
            "peak_vram_bytes": fitted.get("peak_vram_bytes"),
            "batch_size": fitted.get("batch_size"),
            "parameter_count": fitted.get("parameter_count"),
            "outer_test_used": False,
        }
    )
    del model
    torch.cuda.empty_cache()
    return payload


def summarize(payload: dict) -> dict:
    ev = payload["evaluation"]
    return {
        "macro_pr_auc": ev.get("macro_pr_auc"),
        "worst_pr_auc": ev.get("worst_pr_auc"),
        "stages": {k: v["pr_auc"] for k, v in ev.get("stages", {}).items()},
        "f1": {k: v["risk_f1"] for k, v in ev.get("stages", {}).items()},
        "recall": {k: v["risk_recall"] for k, v in ev.get("stages", {}).items()},
        "accuracy": {k: v["accuracy"] for k, v in ev.get("stages", {}).items()},
        "diagnostics": ev.get("diagnostics"),
        "best_epoch": payload["fitted"]["best_epoch"],
        "parameter_count": payload["fitted"]["parameter_count"],
        "gap": float(np.nanmean([r.get("generalization_gap") for r in payload.get("rows", [])])),
    }


def stage0_integrity() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_HYBRID_PHASE3")
    cuda = require_cuda()
    hashes = verify_phase2_locks()
    cfg = make_c0_config(8, 4, 5, 12)
    assert_c0_topology(cfg)
    model = VNextHybrid(cfg)
    cases = availability_unit_cases(model)
    if not all(c["pass"] for c in cases):
        raise RuntimeError(f"AVAILABILITY_REGRESSION:{cases}")
    payload = {
        "cuda": cuda,
        "phase2_hashes": hashes,
        "topology_spec": TOPOLOGY_SPEC,
        "topology_hash": topology_hash(),
        "availability": cases,
        "uci_1fold_floor": UCI_1FOLD_FLOOR,
        "uci_robust_floor": UCI_ROBUST_FLOOR,
        "uci_std_ceiling": UCI_STD_CEILING,
        "variance_rule": "materially_worsen if robust UCI std > Phase2 std + 0.010",
        "outer_test_used": False,
        "git_commit": git_commit(),
        "timestamp": utc_now(),
    }
    write_json(PHASE3 / "PHASE3_PROTOCOL.json", payload)
    write_json(
        PHASE3 / "HPO_SEARCH_SPACE.json",
        {
            "structural": {"d_fuse": [64, 96, 128], "cnn_channels": [64, 96, 128], "bilstm_hidden": [64, 96, 128]},
            "oulad_training": {
                "lr": "log 5e-5..5e-4",
                "weight_decay": "log 1e-6..1e-3",
                "dropout": "0.10..0.40",
                "batch_size": [128, 256, 512],
                "pos_weight_multiplier": "0.75..1.25",
                "entropy_floor_coefficient": [0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            },
            "uci_training": {
                "lr": "log 3e-5..5e-4",
                "weight_decay": "log 1e-6..5e-3",
                "dropout": "0.15..0.50",
                "batch_size": [32, 64, 128, 256],
                "pos_weight_multiplier": "0.75..1.25",
                "entropy_floor_coefficient": [0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            },
            "forbidden": ["serial", "residual_fusion", "attention", "dataset_specific_width"],
        },
    )
    write_json(PHASE3 / "CUDA_EXECUTION_AUDIT.json", {"gate0": cuda, "jobs": [], "silent_cpu_fallback": False})
    return payload


def stage_structural() -> dict:
    tuples = list(itertools.product([64, 96, 128], [64, 96, 128], [64, 96, 128]))
    screen = []
    for d_fuse, cnn, lstm in tuples:
        rec = {"d_fuse": d_fuse, "cnn_channels": cnn, "bilstm_hidden": lstm}
        for domain in ("uci", "oulad"):
            print(f"STRUCT {domain} {d_fuse}/{cnn}/{lstm}", flush=True)
            payload = hybrid_run(
                domain=domain, fold=SCREEN_FOLD, seed=SCREEN_SEED,
                d_fuse=d_fuse, cnn_channels=cnn, bilstm_hidden=lstm,
                lr=2e-4, weight_decay=2e-4, dropout=0.20, batch_size=256,
                pos_weight_multiplier=1.0, entropy_floor=0.002,
                max_epochs=20, patience=7, stage_name="SA",
            )
            rec[domain] = summarize(payload)
            _csv(PHASE3 / "STRUCTURAL_HPO_TRIALS.csv", payload["rows"], HYBRID_FIELDS)
        rec["eligible_1fold"] = rec["uci"]["macro_pr_auc"] >= UCI_1FOLD_FLOOR
        screen.append(rec)
    screen.sort(key=lambda r: (r["oulad"]["macro_pr_auc"], r["oulad"]["worst_pr_auc"], r["uci"]["macro_pr_auc"]), reverse=True)
    eligible = [r for r in screen if r["eligible_1fold"]]
    promote = (eligible or screen)[:5]
    robust3 = []
    for rec in promote:
        item = {**{k: rec[k] for k in ("d_fuse", "cnn_channels", "bilstm_hidden")}, "folds": {}}
        for domain in ("uci", "oulad"):
            item["folds"][domain] = []
            for fold in INNER_FOLDS:
                payload = hybrid_run(
                    domain=domain, fold=fold, seed=SCREEN_SEED,
                    d_fuse=rec["d_fuse"], cnn_channels=rec["cnn_channels"], bilstm_hidden=rec["bilstm_hidden"],
                    lr=2e-4, weight_decay=2e-4, dropout=0.20, batch_size=256,
                    pos_weight_multiplier=1.0, entropy_floor=0.002,
                    max_epochs=20, patience=7, stage_name="SA3",
                )
                item["folds"][domain].append(summarize(payload))
                _csv(PHASE3 / "STRUCTURAL_HPO_TRIALS.csv", payload["rows"], HYBRID_FIELDS)
            macros = [x["macro_pr_auc"] for x in item["folds"][domain]]
            item[f"{domain}_mean"] = float(np.mean(macros))
            item[f"{domain}_std"] = float(np.std(macros))
            item[f"{domain}_worst"] = float(np.mean([x["worst_pr_auc"] for x in item["folds"][domain]]))
        item["eligible"] = item["uci_mean"] >= UCI_ROBUST_FLOOR - 0.02  # 3x1 seed only; full std later
        robust3.append(item)
    robust3.sort(key=lambda r: (r.get("eligible", False), r["oulad_mean"], r["oulad_worst"], r["uci_mean"]), reverse=True)
    top2 = robust3[:2]
    final = []
    for rec in top2:
        item = {k: rec[k] for k in ("d_fuse", "cnn_channels", "bilstm_hidden")}
        for domain in ("uci", "oulad"):
            vals = []
            for fold in INNER_FOLDS:
                for seed in SEEDS:
                    payload = hybrid_run(
                        domain=domain, fold=fold, seed=seed,
                        d_fuse=rec["d_fuse"], cnn_channels=rec["cnn_channels"], bilstm_hidden=rec["bilstm_hidden"],
                        lr=2e-4, weight_decay=2e-4, dropout=0.20, batch_size=256,
                        pos_weight_multiplier=1.0, entropy_floor=0.002,
                        max_epochs=20, patience=7, stage_name="SA33",
                    )
                    vals.append(summarize(payload))
                    _csv(PHASE3 / "STRUCTURAL_HPO_TRIALS.csv", payload["rows"], HYBRID_FIELDS)
            macros = [x["macro_pr_auc"] for x in vals]
            item[f"{domain}_mean"] = float(np.mean(macros))
            item[f"{domain}_std"] = float(np.std(macros))
            item[f"{domain}_worst"] = float(np.mean([x["worst_pr_auc"] for x in vals]))
            item[f"{domain}_min"] = float(np.min(macros))
            item[f"{domain}_params"] = vals[0]["parameter_count"]
        item["eligible"] = item["uci_mean"] >= UCI_ROBUST_FLOOR and item["uci_std"] <= UCI_STD_CEILING
        final.append(item)
    final.sort(
        key=lambda r: (
            r["eligible"],
            r["oulad_mean"],
            r["oulad_worst"],
            -r["oulad_std"],
            r["uci_mean"],
            -r.get("uci_params", 0),
        ),
        reverse=True,
    )
    if not any(r["eligible"] for r in final):
        # fall back to Phase2 default if nothing meets the tight floor
        winner = next((r for r in final if r["d_fuse"] == 96 and r["cnn_channels"] == 128 and r["bilstm_hidden"] == 128), final[0])
        winner["selected_reason"] = "no_tuple_met_strict_floor_keep_phase2_or_best"
    else:
        winner = next(r for r in final if r["eligible"])
        winner["selected_reason"] = "lexicographic_oulad_then_uci_guardrail"
    shared = {
        "d_fuse": winner["d_fuse"],
        "cnn_channels": winner["cnn_channels"],
        "bilstm_hidden": winner["bilstm_hidden"],
        "shared_across_uci_and_oulad": True,
        "selection": winner,
        "screen": [{k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "diagnostics"}) for k, v in r.items() if k != "folds"} for r in screen[:10]],
        "robust3": [{k: v for k, v in r.items() if k != "folds"} for r in robust3],
        "robust33": final,
        "outer_test_used": False,
        "topology_hash": topology_hash(),
    }
    write_json(PHASE3 / "SHARED_STRUCTURAL_CONFIG.json", shared)
    return shared


def _optuna_objective(domain: str, shared: dict, trial: optuna.Trial) -> float:
    if domain == "oulad":
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.10, 0.40)
        batch = trial.suggest_categorical("batch_size", [128, 256, 512])
        pw = trial.suggest_float("pos_weight_multiplier", 0.75, 1.25)
        ent = trial.suggest_categorical("entropy_floor_coefficient", [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
        max_epochs, patience = 60, 10
    else:
        lr = trial.suggest_float("lr", 3e-5, 5e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.15, 0.50)
        batch = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        pw = trial.suggest_float("pos_weight_multiplier", 0.75, 1.25)
        ent = trial.suggest_categorical("entropy_floor_coefficient", [0.0, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
        max_epochs, patience = 50, 10

    def cb(epoch, stop_pr):
        trial.report(stop_pr, epoch)
        if epoch >= 8 and trial.should_prune():
            raise optuna.TrialPruned(f"pruned_at_epoch_{epoch}")

    try:
        payload = hybrid_run(
            domain=domain, fold=SCREEN_FOLD, seed=SCREEN_SEED,
            d_fuse=shared["d_fuse"], cnn_channels=shared["cnn_channels"], bilstm_hidden=shared["bilstm_hidden"],
            lr=lr, weight_decay=wd, dropout=dropout, batch_size=batch,
            pos_weight_multiplier=pw, entropy_floor=ent,
            max_epochs=max_epochs, patience=patience, stage_name=f"HPO_{domain}",
            trial_id=str(trial.number), epoch_callback=cb,
        )
    except optuna.TrialPruned:
        _csv(PHASE3 / "HPO_PRUNED_FAILED_TRIALS.csv", [{
            "trial_id": trial.number, "dataset": domain, "status": "pruned", "timestamp": utc_now(),
            "params": json.dumps(trial.params), "outer_test_used": False,
        }], ["trial_id", "dataset", "status", "timestamp", "params", "outer_test_used"])
        raise
    except Exception as exc:
        _csv(PHASE3 / "HPO_PRUNED_FAILED_TRIALS.csv", [{
            "trial_id": trial.number, "dataset": domain, "status": f"fail:{exc}", "timestamp": utc_now(),
            "params": json.dumps(trial.params), "outer_test_used": False,
        }], ["trial_id", "dataset", "status", "timestamp", "params", "outer_test_used"])
        raise
    ev = payload["evaluation"]
    trial.set_user_attr("macro_pr_auc", ev["macro_pr_auc"])
    trial.set_user_attr("worst_pr_auc", ev["worst_pr_auc"])
    trial.set_user_attr("stages", ev["stages"])
    trial.set_user_attr("best_epoch", payload["fitted"]["best_epoch"])
    _csv(PHASE3 / f"{domain.upper()}_HPO_TRIALS.csv", payload["rows"], HYBRID_FIELDS)
    # scalar: macro - 0.15 * (macro-worst) to protect early stages without changing lex ranking later
    return float(ev["macro_pr_auc"]) - 0.05 * (ev["macro_pr_auc"] - ev["worst_pr_auc"])


def stage_training_hpo(domain: str, shared: dict, n_trials: int) -> dict:
    storage = f"sqlite:///{PHASE3 / 'optuna' / f'{domain}.db'}"
    (PHASE3 / "optuna").mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=8)
    study = optuna.create_study(
        study_name=f"phase3_{domain}",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    for stale in [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]:
        study.tell(stale.number, state=optuna.trial.TrialState.FAIL)
    remaining = max(0, n_trials - len([t for t in study.trials if t.state.is_finished()]))
    if remaining:
        study.optimize(lambda trial: _optuna_objective(domain, shared, trial), n_trials=remaining, catch=(RuntimeError, ValueError))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    complete.sort(key=lambda t: (-t.user_attrs.get("macro_pr_auc", t.value or -1), -t.user_attrs.get("worst_pr_auc", -1)))
    best = complete[0] if complete else study.best_trial
    result = {
        "domain": domain,
        "n_complete": len(complete),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_params": best.params,
        "best_macro": best.user_attrs.get("macro_pr_auc", best.value),
        "best_worst": best.user_attrs.get("worst_pr_auc"),
        "best_epoch": best.user_attrs.get("best_epoch"),
        "outer_test_used": False,
    }
    write_json(PHASE3 / f"{domain}_hpo_best.json", result)
    return result


def stage_robust(shared: dict, oulad_hp: dict, uci_hp: dict) -> dict:
    out = {}
    for domain, hp in (("oulad", oulad_hp), ("uci", uci_hp)):
        rows = []
        for fold in INNER_FOLDS:
            for seed in SEEDS:
                print(f"ROBUST {domain} f{fold} s{seed}", flush=True)
                payload = hybrid_run(
                    domain=domain, fold=fold, seed=seed,
                    d_fuse=shared["d_fuse"], cnn_channels=shared["cnn_channels"], bilstm_hidden=shared["bilstm_hidden"],
                    lr=hp["lr"], weight_decay=hp["weight_decay"], dropout=hp["dropout"], batch_size=int(hp["batch_size"]),
                    pos_weight_multiplier=hp["pos_weight_multiplier"], entropy_floor=hp["entropy_floor_coefficient"],
                    max_epochs=60 if domain == "oulad" else 50, patience=10, stage_name="ROB",
                )
                summ = summarize(payload)
                summ.update({"fold": fold, "seed": seed})
                rows.append(summ)
                _csv(PHASE3 / "ROBUST_CONFIRMATION.csv", payload["rows"], HYBRID_FIELDS)
                for stage, diag in (payload["evaluation"].get("diagnostics") or {}).items():
                    _csv(
                        PHASE3 / "GATE_DIAGNOSTICS.csv",
                        [{
                            "dataset": domain, "stage": stage, "fold": fold, "seed": seed,
                            **{k: diag.get(k) for k in ("g_temporal_mean", "tabular_mass", "cnn_mass", "bilstm_mass", "tabular_norm", "temporal_norm", "temporal_available_rate")},
                            "outer_test_used": False,
                        }],
                        ["dataset", "stage", "fold", "seed", "g_temporal_mean", "tabular_mass", "cnn_mass", "bilstm_mass", "tabular_norm", "temporal_norm", "temporal_available_rate", "outer_test_used"],
                    )
                for hist in payload["fitted"]["history"]:
                    _csv(
                        PHASE3 / "LEARNING_CURVES.csv",
                        [{"dataset": domain, "fold": fold, "seed": seed, **hist, "outer_test_used": False}],
                        ["dataset", "fold", "seed", "epoch", "train_loss", "train_pr_auc", "stop_pr_auc", "grad_norm", "batch_size", "lr", "outer_test_used"],
                    )
        macros = [r["macro_pr_auc"] for r in rows]
        out[domain] = {
            "mean": float(np.mean(macros)),
            "std": float(np.std(macros)),
            "min": float(np.min(macros)),
            "max": float(np.max(macros)),
            "worst_mean": float(np.mean([r["worst_pr_auc"] for r in rows])),
            "gap_mean": float(np.mean([r["gap"] for r in rows])),
            "best_epoch_median": float(np.median([r["best_epoch"] for r in rows])),
            "stage_means": {
                st: float(np.mean([r["stages"][st] for r in rows]))
                for st in rows[0]["stages"]
            },
            "f1_means": {st: float(np.mean([r["f1"][st] for r in rows])) for st in rows[0]["f1"]},
            "recall_means": {st: float(np.mean([r["recall"][st] for r in rows])) for st in rows[0]["recall"]},
            "acc_means": {st: float(np.mean([r["accuracy"][st] for r in rows])) for st in rows[0]["accuracy"]},
            "rows": rows,
        }
    write_json(PHASE3 / "robust_summary.json", {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in out.items()})
    return out


def stage_baselines() -> dict:
    results = {}
    xgb_grid = list(itertools.product([200, 400], [4, 5, 6], [0.03, 0.05, 0.08]))
    families_robust = ("LR", "DT", "RF", "XGB", "MLP")
    for domain in ("uci", "oulad"):
        prepared, fit, stop, valid = prepare(domain, SCREEN_FOLD)
        domain_out = {}
        for stage in prepared.stages:
            frame = baseline_frame(prepared, stage)
            groups = feature_groups(frame)
            cols = groups["full"]
            domain_out[stage] = {}
            for family in ("LR", "DT", "RF", "XGB", "MLP") + (("SVM",) if domain == "uci" else ()):
                metrics = fit_eval_baseline(family, frame, cols, prepared.categorical, fit, stop, valid, SCREEN_SEED)
                domain_out[stage][family] = metrics
                _csv(
                    PHASE3 / "BASELINE_INNER_RESULTS.csv",
                    [{
                        "dataset": domain, "stage": stage, "family": family, "feature_set": "full",
                        "search": "fixed", "inner_fold": 0, "seed": 42, **{k: metrics.get(k) for k in ("pr_auc", "risk_f1", "risk_recall", "accuracy", "ece")},
                        "outer_test_used": False,
                    }],
                    ["dataset", "stage", "family", "feature_set", "search", "inner_fold", "seed", "pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "outer_test_used"],
                )
            # bounded XGB search
            best_xgb = domain_out[stage]["XGB"]
            best_cfg = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}
            from xgboost import XGBClassifier
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            ids = frame.record_id.astype(str)
            train, st, va = frame[ids.isin(fit)], frame[ids.isin(stop)], frame[ids.isin(valid)]
            cats = [c for c in prepared.categorical if c in cols]
            nums = [c for c in cols if c not in cats]
            prep = ColumnTransformer([
                ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), nums),
                ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("e", OneHotEncoder(handle_unknown="ignore"))]), cats),
            ])
            xtr, xst, xva = prep.fit_transform(train), prep.transform(st), prep.transform(va)
            ytr, yst, yva = train.target.to_numpy(), st.target.to_numpy(), va.target.to_numpy()
            pos = max(1, int(ytr.sum())); neg = max(1, len(ytr) - pos)
            for n_est, depth, lrate in xgb_grid:
                clf = XGBClassifier(
                    n_estimators=n_est, max_depth=depth, learning_rate=lrate, subsample=0.8,
                    colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
                    n_jobs=-1, random_state=42, scale_pos_weight=neg / pos,
                )
                clf.fit(xtr, ytr)
                stop_p = clf.predict_proba(xst)[:, 1]
                valid_p = clf.predict_proba(xva)[:, 1]
                thr = select_stop_threshold(yst, stop_p)
                met = binary_metrics(yva, valid_p, threshold=thr)
                if met["pr_auc"] > best_xgb["pr_auc"] + 1e-4:
                    best_xgb = met
                    best_cfg = {"n_estimators": n_est, "max_depth": depth, "learning_rate": lrate}
            domain_out[stage]["XGB_tuned"] = {**best_xgb, "config": best_cfg}
            _csv(
                PHASE3 / "BASELINE_INNER_RESULTS.csv",
                [{
                    "dataset": domain, "stage": stage, "family": "XGB_tuned", "feature_set": "full",
                    "search": "bounded", "inner_fold": 0, "seed": 42, **{k: best_xgb.get(k) for k in ("pr_auc", "risk_f1", "risk_recall", "accuracy", "ece")},
                    "outer_test_used": False,
                }],
                ["dataset", "stage", "family", "feature_set", "search", "inner_fold", "seed", "pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "outer_test_used"],
            )
        macros = {}
        for family in ("LR", "DT", "RF", "XGB", "XGB_tuned", "MLP"):
            vals = [domain_out[st][family]["pr_auc"] for st in domain_out if family in domain_out[st]]
            macros[family] = float(np.mean(vals)) if vals else None
        robust_macros = {fam: [] for fam in ("LR", "RF", "XGB", "MLP")}
        robust_stage = {fam: {st: [] for st in domain_out} for fam in robust_macros}
        for fold in INNER_FOLDS:
            for seed in SEEDS:
                print(f"BASELINE_ROBUST {domain} f{fold} s{seed}", flush=True)
                prep_f, fit_f, stop_f, valid_f = prepare(domain, fold)
                for stage in prep_f.stages:
                    frame = baseline_frame(prep_f, stage)
                    cols = feature_groups(frame)["full"]
                    for fam in robust_macros:
                        met = fit_eval_baseline(fam, frame, cols, prep_f.categorical, fit_f, stop_f, valid_f, seed)
                        robust_stage[fam][stage].append(met["pr_auc"])
                        _csv(
                            PHASE3 / "BASELINE_INNER_RESULTS.csv",
                            [{
                                "dataset": domain, "stage": stage, "family": fam, "feature_set": "full",
                                "search": "robust3x3", "inner_fold": fold, "seed": seed,
                                **{k: met.get(k) for k in ("pr_auc", "risk_f1", "risk_recall", "accuracy", "ece")},
                                "outer_test_used": False,
                            }],
                            ["dataset", "stage", "family", "feature_set", "search", "inner_fold", "seed", "pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "outer_test_used"],
                        )
                for fam in robust_macros:
                    robust_macros[fam].append(float(np.mean([robust_stage[fam][st][-1] for st in prep_f.stages])))
        for fam, series in robust_macros.items():
            macros[f"{fam}_robust"] = float(np.mean(series))
            macros[f"{fam}_robust_std"] = float(np.std(series))
        for fam, stages in robust_stage.items():
            domain_out[f"{fam}_robust_stage"] = {st: {"pr_auc": float(np.mean(vs)), "risk_f1": None} for st, vs in stages.items()}
        domain_out["macro"] = macros
        results[domain] = domain_out
    write_json(PHASE3 / "BASELINE_CONFIGS_FINAL.json", {
        "feature_set": "static+aggregate+temporal_last_mean_max+progress",
        "fixed": {"LR": "balanced C=1", "DT": "depth8 leaf20", "RF": "200 trees leaf2 balanced", "XGB": "200/5/0.05", "MLP": "(128,64)"},
        "xgb_tuned_per_stage": {d: {st: results[d][st].get("XGB_tuned", {}).get("config") for st in results[d] if st != "macro"} for d in results},
        "outer_test_used": False,
    })
    write_json(PHASE3 / "baseline_inner_macros.json", {d: results[d]["macro"] for d in results})
    slim = {}
    for domain, payload in results.items():
        slim[domain] = {"macro": payload["macro"]}
        for stage, fams in payload.items():
            if stage == "macro":
                continue
            slim[domain][stage] = {
                name: {k: v.get(k) if isinstance(v, dict) else v for k in ("pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "config") if not isinstance(v, dict) or k in v}
                for name, v in fams.items()
            }
    write_json(PHASE3 / "baselines_full.json", slim)
    return results


def collect_thresholds(shared, oulad_hp, uci_hp) -> dict:
    """Stage-specific thresholds from inner STOP of fold0 seed42 robust run."""
    out = {}
    for domain, hp in (("oulad", oulad_hp), ("uci", uci_hp)):
        prepared, fit, stop, valid = prepare(domain, 0)
        payload = hybrid_run(
            domain=domain, fold=0, seed=42,
            d_fuse=shared["d_fuse"], cnn_channels=shared["cnn_channels"], bilstm_hidden=shared["bilstm_hidden"],
            lr=hp["lr"], weight_decay=hp["weight_decay"], dropout=hp["dropout"], batch_size=int(hp["batch_size"]),
            pos_weight_multiplier=hp["pos_weight_multiplier"], entropy_floor=hp["entropy_floor_coefficient"],
            max_epochs=60 if domain == "oulad" else 50, patience=10, stage_name="ROB",
        )
        out[domain] = {st: ev["threshold"] for st, ev in payload["evaluation"]["stages"].items()}
    write_json(PHASE3 / "THRESHOLD_SELECTION.json", {"policy": "STOP-only F1 then recall then |t-0.5|", "thresholds": out, "outer_test_used": False})
    return out


def temperature_report(shared, oulad_hp, uci_hp) -> dict:
    report = {"used": False, "reason": "monotonic_temperature_does_not_change_PR_AUC; ECE already moderate in Phase2", "outer_test_used": False}
    write_json(PHASE3 / "CALIBRATION_REPORT.json", report)
    return report


def inner_acceptance(robust: dict, baselines: dict) -> dict:
    def _macro_pool(macro: dict) -> dict[str, float]:
        preferred = {k: v for k, v in macro.items() if v is not None and k.endswith("_robust") and not k.endswith("_std")}
        if preferred:
            return preferred
        return {k: v for k, v in macro.items() if v is not None and not k.endswith("_std")}

    oulad_base = _macro_pool(baselines["oulad"]["macro"])
    uci_base = _macro_pool(baselines["uci"]["macro"])
    best_oulad_name = max(oulad_base, key=oulad_base.get)
    best_uci_name = max(uci_base, key=uci_base.get)
    best_oulad = oulad_base[best_oulad_name]
    best_uci = uci_base[best_uci_name]
    hy_o = robust["oulad"]["mean"]
    hy_u = robust["uci"]["mean"]
    fam_key = best_oulad_name.replace("_robust", "_robust_stage") if best_oulad_name.endswith("_robust") else best_oulad_name
    stage_delta = {}
    for st in robust["oulad"]["stage_means"]:
        if fam_key in baselines["oulad"] and st in baselines["oulad"][fam_key]:
            base_pr = baselines["oulad"][fam_key][st]["pr_auc"]
        elif st in baselines["oulad"] and best_oulad_name.replace("_robust", "") in baselines["oulad"][st]:
            base_pr = baselines["oulad"][st][best_oulad_name.replace("_robust", "")]["pr_auc"]
        else:
            base_pr = best_oulad
        stage_delta[st] = robust["oulad"]["stage_means"][st] - base_pr
    pos_stages = sum(1 for v in stage_delta.values() if v > 0)
    oulad_delta = hy_o - best_oulad
    uci_delta = hy_u - best_uci
    oulad_ok = oulad_delta > 0 and pos_stages >= 3
    fam = best_oulad_name.replace("_robust", "")
    if 0 < oulad_delta < 0.003:
        hybrid_f1 = float(np.mean(list(robust["oulad"]["f1_means"].values())))
        base_f1s = []
        for stage in robust["oulad"]["f1_means"]:
            if stage in baselines["oulad"] and fam in baselines["oulad"][stage]:
                base_f1s.append(baselines["oulad"][stage][fam]["risk_f1"])
        oulad_ok = oulad_ok and (not base_f1s or hybrid_f1 >= float(np.mean(base_f1s)))
    if oulad_delta <= 0:
        oulad_ok = False
    uci_ok = hy_u >= best_uci - 0.005
    if best_uci - 0.005 <= hy_u < best_uci:
        uci_s2 = robust["uci"]["stage_means"].get("S2")
        base_s2 = None
        if "S2" in baselines["uci"] and best_uci_name.replace("_robust", "") in baselines["uci"]["S2"]:
            base_s2 = baselines["uci"]["S2"][best_uci_name.replace("_robust", "")]["pr_auc"]
        if uci_s2 is not None and base_s2 is not None and uci_s2 < base_s2 - 0.01:
            uci_ok = False
    integrity = True
    ready = oulad_ok and uci_ok and integrity
    decision = {
        "ready": ready,
        "status": "READY_FOR_FINAL_EVAL" if ready else "NOT_READY_FOR_FINAL_EVAL",
        "oulad": {
            "hybrid": hy_o,
            "best_baseline": best_oulad,
            "best_baseline_name": best_oulad_name,
            "delta": oulad_delta,
            "positive_stages": pos_stages,
            "stage_delta": stage_delta,
            "ok": oulad_ok,
        },
        "uci": {
            "hybrid": hy_u,
            "best_baseline": best_uci,
            "best_baseline_name": best_uci_name,
            "delta": uci_delta,
            "std": robust["uci"]["std"],
            "ok": uci_ok,
        },
        "outer_test_used": False,
    }
    write_json(PHASE3 / "INNER_ACCEPTANCE.json", decision)
    return decision


def write_lock(shared, oulad_hp, uci_hp, robust, thresholds) -> Path:
    lock = {
        "lock_status": "LOCKED",
        "public_model_class": "Hybrid",
        "architecture_id": "C0",
        "topology_spec": TOPOLOGY_SPEC,
        "topology_hash": topology_hash(),
        "phase2_topology_file_hash": verify_phase2_locks(),
        "shared_structural_config": {k: shared[k] for k in ("d_fuse", "cnn_channels", "bilstm_hidden")},
        "uci_training_config": uci_hp,
        "oulad_early_training_config": oulad_hp,
        "oulad_final_training_config": oulad_hp,
        "feature_contract_hashes": {
            "uci": json.loads((PHASE2 := Path(ROOT / "artifacts/hybrid_vnext/phase2/feature_contract_uci.json")).read_text())["hash"] if (ROOT / "artifacts/hybrid_vnext/phase2/feature_contract_uci.json").exists() else None,
            "oulad": json.loads((ROOT / "artifacts/hybrid_vnext/phase2/feature_contract_oulad.json").read_text())["hash"] if (ROOT / "artifacts/hybrid_vnext/phase2/feature_contract_oulad.json").exists() else None,
        },
        "split_hashes": json.loads((ROOT / "artifacts/hybrid_vnext/phase2/PROTOCOL_LOCK.json").read_text())["split_hashes"],
        "seeds": list(SEEDS),
        "epoch_refit_policy": "median of inner robust best epochs",
        "oulad_refit_epochs": int(round(robust["oulad"]["best_epoch_median"])),
        "uci_refit_epochs": int(round(robust["uci"]["best_epoch_median"])),
        "threshold_policy": "STOP-only F1 then recall then |t-0.5|",
        "thresholds": thresholds,
        "calibration_policy": "none_temperature_not_retained",
        "git_commit": git_commit(),
        "code_hashes": {
            "model.py": sha256_file(ROOT / "experiments/hybrid_vnext/model.py"),
            "train.py": sha256_file(ROOT / "experiments/hybrid_vnext/train.py"),
            "run_phase3.py": sha256_file(Path(__file__)),
        },
        "outer_test_used": False,
        "timestamp": utc_now(),
    }
    path = PHASE3 / "FINAL_MODEL_LOCK.json"
    write_json(path, lock)
    digest = sha256_file(path)
    (PHASE3 / "FINAL_MODEL_LOCK.sha256").write_text(digest + "\n", encoding="utf-8")
    return path


def one_shot_outer(shared, oulad_hp, uci_hp, robust, thresholds) -> dict:
    lock_path = PHASE3 / "FINAL_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("OUTER_BLOCKED_NO_LOCK")
    expected = (PHASE3 / "FINAL_MODEL_LOCK.sha256").read_text(encoding="utf-8").strip()
    if sha256_file(lock_path) != expected:
        raise RuntimeError("LOCK_TAMPERED")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["lock_status"] != "LOCKED":
        raise RuntimeError("LOCK_NOT_LOCKED")

    pred_rows = []
    metrics_rows = []
    for domain, hp, folds, epochs in (
        ("uci", uci_hp, UCI_OUTER_FOLDS, lock["uci_refit_epochs"]),
        ("oulad", oulad_hp, OULAD_OUTER_FOLDS, lock["oulad_refit_epochs"]),
    ):
        views, context, numeric, categorical = load_domain(domain)
        for outer_fold in folds:
            train_ids, test_ids = outer_holdout_ids(domain, outer_fold)
            train_ids = [i for i in train_ids if i in set(context.record_id.astype(str))]
            test_ids = [i for i in test_ids if i in set(context.record_id.astype(str))]
            prepared = scale_views(views, context, numeric, categorical, train_ids, domain)
            seed_probs = {stage: [] for stage in prepared.views}
            for seed in SEEDS:
                print(f"OUTER {domain} fold={outer_fold} seed={seed}", flush=True)
                cfg = make_c0_config(
                    prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim, prepared.summary_dim,
                    d_fuse=shared["d_fuse"], cnn_channels=shared["cnn_channels"], bilstm_hidden=shared["bilstm_hidden"],
                    dropout=hp["dropout"], entropy_floor_coefficient=hp["entropy_floor_coefficient"],
                )
                trainer = Trainer(
                    prepared, cfg, lr=hp["lr"], weight_decay=hp["weight_decay"],
                    max_epochs=epochs, patience=10**9, batch_size=int(hp["batch_size"]), seed=seed,
                    pos_weight_multiplier=hp["pos_weight_multiplier"], fixed_epochs=epochs,
                )
                fitted = trainer.fit(train_ids, [])
                model = fitted["model"]
                for stage, view in prepared.views.items():
                    ids = _ids_for_stage(view, test_ids)
                    seed_probs[stage].append((ids, trainer._predict(model, stage, ids)))
                del model
                torch.cuda.empty_cache()
            for stage, view in prepared.views.items():
                # average aligned by id
                acc = {}
                for ids, probs in seed_probs[stage]:
                    for i, p in zip(ids, probs):
                        acc.setdefault(i, []).append(float(p))
                ids = sorted(acc)
                probs = np.array([float(np.mean(acc[i])) for i in ids])
                lookup = {str(r): i for i, r in enumerate(view.record_id)}
                y = view.target[[lookup[i] for i in ids]]
                groups = view.group_id[[lookup[i] for i in ids]]
                thr = thresholds[domain][stage]
                met = binary_metrics(y, probs, threshold=thr)
                metrics_rows.append({"dataset": domain, "outer_fold": outer_fold, "stage": stage, "model": "Hybrid", **met})
                frame = baseline_frame(prepared, stage)
                cols = feature_groups(frame)["full"]
                test_stage = [i for i in test_ids if i in set(frame.record_id.astype(str))]
                train_stage = [i for i in train_ids if i in set(frame.record_id.astype(str))]
                baseline_p = {family: {} for family in ("LR", "DT", "RF", "XGB", "MLP")}
                for family in ("LR", "DT", "RF", "XGB", "MLP"):
                    try:
                        scored = fit_eval_baseline(
                            family, frame, cols, prepared.categorical,
                            train_stage, train_stage[: max(1, len(train_stage)//5)], test_stage, 42,
                            return_scores=True,
                        )
                    except Exception:
                        continue
                    metrics_rows.append({"dataset": domain, "outer_fold": outer_fold, "stage": stage, "model": family, **{k: v for k, v in scored.items() if k not in ("valid_record_id", "valid_p", "valid_y")}})
                    for rec, p in zip(scored["valid_record_id"], scored["valid_p"]):
                        baseline_p[family][str(rec)] = float(p)
                for i, p, t, g in zip(ids, probs, y, groups):
                    row = {
                        "dataset": domain, "outer_fold": outer_fold, "stage": stage, "record_id": i,
                        "group_id": str(g), "target": int(t), "hybrid_prob": float(p),
                        "threshold": thr, "predicted": int(p >= thr),
                    }
                    for family, mapping in baseline_p.items():
                        row[f"{family.lower()}_prob"] = mapping.get(str(i))
                    pred_rows.append(row)
    pred = pd.DataFrame(pred_rows)
    pred.to_parquet(PHASE3 / "FINAL_OUTER_PREDICTIONS.parquet", index=False)
    met = pd.DataFrame(metrics_rows)
    met.to_csv(PHASE3 / "FINAL_OUTER_METRICS.csv", index=False)
    return {"predictions": pred, "metrics": met}


def paired_bootstrap(pred: pd.DataFrame, metrics: pd.DataFrame, n_resamples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for dataset in pred.dataset.unique():
        sub = pred[pred.dataset == dataset]
        # strongest baseline from outer metrics
        hy = metrics[(metrics.dataset == dataset) & (metrics.model == "Hybrid")]
        bases = metrics[(metrics.dataset == dataset) & (metrics.model != "Hybrid")]
        if bases.empty:
            continue
        # pick baseline with best mean pr_auc
        best_name = bases.groupby("model")["pr_auc"].mean().idxmax()
        # we don't have baseline probs in pred - compute Hybrid-only bootstrap of metrics vs observed baseline means is invalid
        # Store Hybrid bootstrap CI; comparison uses paired if we have both scores.
        # Recompute baseline not stored: skip pairing if no baseline probs.
        groups = sub.group_id.astype(str).unique()
        base_col = f"{str(best_name).lower()}_prob"
        paired = base_col in sub.columns and sub[base_col].notna().any()
        for metric_name in ("pr_auc", "risk_f1", "risk_recall"):
            vals = []
            deltas = []
            for _ in range(n_resamples):
                draw = rng.choice(groups, size=len(groups), replace=True)
                # sample rows whose group in draw with multiplicity
                parts = [sub[sub.group_id.astype(str) == g] for g in draw]
                boot = pd.concat(parts, ignore_index=True) if parts else sub.iloc[0:0]
                if boot.empty or boot.target.nunique() < 2:
                    continue
                # macro over stages
                stage_scores = []
                base_scores = []
                for stage, gdf in boot.groupby("stage"):
                    if gdf.target.nunique() < 2:
                        continue
                    if metric_name == "pr_auc":
                        from sklearn.metrics import average_precision_score
                        stage_scores.append(average_precision_score(gdf.target, gdf.hybrid_prob))
                        if paired:
                            base_scores.append(average_precision_score(gdf.target, gdf[base_col]))
                    else:
                        m = binary_metrics(gdf.target, gdf.hybrid_prob, threshold=float(gdf.threshold.iloc[0]))
                        stage_scores.append(m[metric_name])
                        if paired:
                            bm = binary_metrics(gdf.target, gdf[base_col], threshold=float(gdf.threshold.iloc[0]))
                            base_scores.append(bm[metric_name])
                if stage_scores:
                    hy_val = float(np.mean(stage_scores))
                    vals.append(hy_val)
                    if base_scores:
                        deltas.append(hy_val - float(np.mean(base_scores)))
            if not vals:
                continue
            row = {
                "dataset": dataset,
                "metric": metric_name,
                "hybrid_mean": float(np.mean(vals)),
                "ci95_low": float(np.quantile(vals, 0.025)),
                "ci95_high": float(np.quantile(vals, 0.975)),
                "n_resamples": len(vals),
                "bootstrap_seed": 2026,
                "grouping": "group_id",
                "best_baseline_name": best_name,
                "best_baseline_outer_mean": float(bases[bases.model == best_name].groupby("stage")["pr_auc"].mean().mean()) if metric_name == "pr_auc" else None,
                "paired": paired,
            }
            if deltas:
                row.update({
                    "mean_delta": float(np.mean(deltas)),
                    "delta_ci95_low": float(np.quantile(deltas, 0.025)),
                    "delta_ci95_high": float(np.quantile(deltas, 0.975)),
                    "p_delta_gt_0": float(np.mean(np.asarray(deltas) > 0)),
                })
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(PHASE3 / "PAIRED_BOOTSTRAP.csv", index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", default="integrity", choices=["integrity", "structural", "hpo", "robust", "baselines", "gate", "outer"])
    parser.add_argument("--oulad-trials", type=int, default=36)
    parser.add_argument("--uci-trials", type=int, default=28)
    args = parser.parse_args()
    PHASE3.mkdir(parents=True, exist_ok=True)
    RUNS3.mkdir(parents=True, exist_ok=True)
    REPORTS3.mkdir(parents=True, exist_ok=True)
    order = ["integrity", "structural", "hpo", "robust", "baselines", "gate", "outer"]
    start = order.index(args.from_stage)
    try:
        if start <= 0:
            print("INTEGRITY", flush=True)
            stage0_integrity()
        if start <= 1:
            print("STRUCTURAL", flush=True)
            if args.from_stage == "structural" or not (PHASE3 / "SHARED_STRUCTURAL_CONFIG.json").exists():
                stage_structural()
        shared = json.loads((PHASE3 / "SHARED_STRUCTURAL_CONFIG.json").read_text(encoding="utf-8"))
        print("SHARED", shared["d_fuse"], shared["cnn_channels"], shared["bilstm_hidden"], flush=True)
        if start <= 2:
            print("HPO_OULAD", flush=True)
            oulad = stage_training_hpo("oulad", shared, args.oulad_trials)
            print("HPO_UCI", flush=True)
            uci = stage_training_hpo("uci", shared, args.uci_trials)
            write_json(PHASE3 / "HPO_SELECTION.json", {"oulad": oulad, "uci": uci, "shared": {k: shared[k] for k in ("d_fuse", "cnn_channels", "bilstm_hidden")}, "outer_test_used": False})
        hpo = json.loads((PHASE3 / "HPO_SELECTION.json").read_text(encoding="utf-8"))
        oulad_hp = hpo["oulad"]["best_params"]
        uci_hp = hpo["uci"]["best_params"]
        if start <= 3:
            print("ROBUST", flush=True)
            robust = stage_robust(shared, oulad_hp, uci_hp)
        else:
            robust = json.loads((PHASE3 / "robust_summary.json").read_text(encoding="utf-8"))
        if start <= 4:
            print("BASELINES", flush=True)
            baselines = stage_baselines()
        else:
            baselines = json.loads((PHASE3 / "baselines_full.json").read_text(encoding="utf-8"))
        if start <= 5:
            print("THRESH", flush=True)
            thresholds = collect_thresholds(shared, oulad_hp, uci_hp)
            temperature_report(shared, oulad_hp, uci_hp)
            print("GATE", flush=True)
            decision = inner_acceptance(robust, baselines)
            write_json(PHASE3 / "phase3_status.json", {"status": decision["status"], "outer_test_used": False})
            if not decision["ready"]:
                print("NOT_READY_FOR_FINAL_EVAL", flush=True)
                return 0
            print("LOCK", flush=True)
            write_lock(shared, oulad_hp, uci_hp, robust, thresholds)
        if start <= 6 and (PHASE3 / "FINAL_MODEL_LOCK.json").exists():
            print("OUTER", flush=True)
            outer = one_shot_outer(shared, oulad_hp, uci_hp, robust, json.loads((PHASE3 / "THRESHOLD_SELECTION.json").read_text())["thresholds"])
            paired_bootstrap(outer["predictions"], outer["metrics"])
        print("PHASE3_CORE_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        write_json(PHASE3 / "phase3_failure.json", {"error": str(exc), "trace": traceback.format_exc(), "outer_test_used": False})
        print("PHASE3_FAIL", exc, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
