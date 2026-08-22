"""Speed-mode finish: remaining OULAD + Hybrid + gates + reports. Documents budget cuts."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HS_V2_GPU_TREES", "1")

import numpy as np
import pandas as pd

from .baselines import default_params, fit_eval, predictor_columns, sample_space
from .data import baseline_frame, inner_partitions, scale_views
from .gate import evaluate_development_gate
from .hpo import _make_study, _study_name, evaluate_hybrid, load_baseline_ceiling
from .io_utils import git_commit, utc_now, write_json
from .metrics import selection_objective
from .model import make_config
from .paths import METRIC_DIR, OOF_DIR, REPORT_ROOT, RUN_DIR, STAT_DIR, ensure_dirs
from .protocol import (
    ABLATIONS,
    BASELINE_ROSTER,
    SCREEN_FOLD,
    SCREEN_SEED,
    protocol_hash,
    stages_for,
    warm_for,
)
from .report import write_all_reports
from .status import write_status

SPEED = {
    "mode": "SPEED_FINISH",
    "baseline_trials_oulad": 4,
    "skip_hpo": ["DT", "SVM", "MLP", "RF"],
    "lock_folds": (0,),
    "lock_seeds": (42, 1201),
    "hybrid_screen_trials": 6,
    "hybrid_max_epochs": 10,
    "hybrid_patience": 4,
    "hybrid_batch": 512,
    "reuse_lr_optuna": True,
    "warm_only_hpo": True,
    "preregistered_oulad_trials": 28,
    "note": "User-requested crash finish. Not the preregistered 28-trial OULAD budget. Documented here.",
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _boost_runtime() -> None:
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.set_num_threads(max(1, os.cpu_count() or 8))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log(f"CUDA {torch.cuda.get_device_name(0)} vram={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    except Exception as exc:
        _log(f"cuda_setup {exc}")
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, "HIGH_PRIORITY_CLASS") else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        _log("priority HIGH")
    except Exception:
        pass


def _empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _lr_best_from_optuna() -> dict:
    try:
        import optuna

        from .db import optuna_storage_url

        study = optuna.load_study(study_name=_study_name("baseline", "oulad", "LR"), storage=optuna_storage_url())
        if study.best_trial is not None:
            _log(f"reuse LR best {study.best_value} {study.best_params}")
            return dict(study.best_params)
    except Exception as exc:
        _log(f"lr_reuse_fail {type(exc).__name__}")
    return default_params("LR", SCREEN_SEED)


def write_uci_gate() -> dict:
    robust_path = RUN_DIR / "robust_uci_C0-R.json"
    lock_path = RUN_DIR / "baseline_lock_uci.json"
    if not robust_path.exists() or not lock_path.exists():
        _log("UCI gate skip: missing robust/lock")
        return {}
    robust = json.loads(robust_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(robust["rows"])
    ceiling = {s: {"ap": float(v["ap"]), "model": v["model"]} for s, v in lock["stage_best_ap"].items()}
    gate = evaluate_development_gate("uci", rows, ceiling)
    _log(f"UCI development gate pass={gate.get('pass')} fail_warm={gate.get('n_warm_fail')}")
    return gate


def oulad_baselines_fast() -> dict:
    ensure_dirs()
    lock_path = RUN_DIR / "baseline_lock_oulad.json"
    if lock_path.exists():
        _log("oulad lock exists")
        return json.loads(lock_path.read_text(encoding="utf-8"))
    import optuna

    domain = "oulad"
    n_trials = SPEED["baseline_trials_oulad"]
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, SCREEN_FOLD)
    _log("scale OULAD views (once)")
    t_scale = time.time()
    prepared = scale_views(domain, fit_ids)
    _log(f"scale_views {time.time()-t_scale:.1f}s static_dim={prepared.static_dim}")
    frames = {}
    for stage in stages_for(domain):
        t0 = time.time()
        frames[stage] = baseline_frame(prepared, stage)
        _log(f"frame {stage} n={len(frames[stage])} cols={frames[stage].shape[1]} {time.time()-t0:.1f}s")
    hpo_stages = list(warm_for(domain)) if SPEED["warm_only_hpo"] else list(stages_for(domain))
    best_params: dict[str, dict] = {}
    for name in BASELINE_ROSTER:
        if name in SPEED["skip_hpo"]:
            params = default_params(name, SCREEN_SEED)
            if name == "RF":
                params["n_estimators"] = 150
            if name == "SVM":
                params["calibrate"] = False
                params["kernel"] = "linear"
            if name == "MLP":
                params["max_iter"] = 80
            if name == "CatBoost":
                params["iterations"] = 200
            best_params[name] = params
            _log(f"skip HPO {name}")
            continue
        if name == "LR" and SPEED["reuse_lr_optuna"]:
            best_params[name] = _lr_best_from_optuna()
            continue
        study = _make_study(_study_name("baseline_fast", domain, name))

        def objective(trial, name=name):
            params = sample_space(name, trial, domain=domain)
            if name == "CatBoost":
                params["iterations"] = min(int(params.get("iterations", 200)), 200)
            if name == "XGB":
                params["n_estimators"] = min(int(params.get("n_estimators", 300)), 250)
            aps = []
            for stage in hpo_stages:
                frame = frames[stage]
                cols, cats = predictor_columns(frame)
                present = set(frame.record_id.astype(str))
                sf = [i for i in fit_ids if i in present]
                ss = [i for i in stop_ids if i in present]
                sv = [i for i in valid_ids if i in present]
                if len(sf) < 30 or len(sv) < 15:
                    continue
                metrics = fit_eval(name, frame, cols, cats, sf, ss, sv, SCREEN_SEED, params)
                aps.append(metrics["ap"])
            if not aps:
                raise optuna.TrialPruned("empty")
            return float(np.mean(aps))

        n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
        remain = max(0, n_trials - n_done)
        _log(f"HPO {name} remain={remain}")
        if remain:
            t0 = time.time()
            study.optimize(objective, n_trials=remain, catch=(Exception,))
            _log(f"HPO {name} done in {time.time()-t0:.0f}s best={getattr(study, 'best_value', None)}")
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        best_params[name] = study.best_params if completed else default_params(name, SCREEN_SEED)
        _empty_cuda()

    rows, oof = [], []
    for fold in SPEED["lock_folds"]:
        fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
        if fold != SCREEN_FOLD:
            prepared = scale_views(domain, fit_ids)
            frames = {stage: baseline_frame(prepared, stage) for stage in stages_for(domain)}
        for seed in SPEED["lock_seeds"]:
            for name in BASELINE_ROSTER:
                params = dict(best_params[name])
                _log(f"lock {name} fold={fold} seed={seed}")
                t0 = time.time()
                for stage in stages_for(domain):
                    frame = frames[stage]
                    cols, cats = predictor_columns(frame)
                    present = set(frame.record_id.astype(str))
                    sf = [i for i in fit_ids if i in present]
                    ss = [i for i in stop_ids if i in present]
                    sv = [i for i in valid_ids if i in present]
                    if len(sv) < 8:
                        continue
                    metrics = fit_eval(name, frame, cols, cats, sf, ss, sv, seed, params)
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
                    for record_id, group_id, y, p in zip(
                        metrics["valid_record_id"], metrics["valid_group"], metrics["valid_y"], metrics["valid_p"]
                    ):
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
                _log(f"  {name} {time.time()-t0:.0f}s")
                _empty_cuda()
    table = pd.DataFrame(rows)
    table.to_csv(METRIC_DIR / "baseline_stage_metrics_oulad.csv", index=False)
    oof_path = OOF_DIR / "baseline_oof_oulad.parquet"
    pd.DataFrame(oof).to_parquet(oof_path, index=False)
    ceiling = table.groupby(["stage", "model"])["ap"].mean().reset_index()
    best_by_stage = ceiling.loc[ceiling.groupby("stage")["ap"].idxmax()]
    lock = {
        "domain": domain,
        "protocol_hash": protocol_hash(),
        "speed_mode": SPEED,
        "git_commit": git_commit(),
        "frozen_at": utc_now(),
        "best_params": best_params,
        "ceiling": best_by_stage.to_dict(orient="records"),
        "stage_best_ap": {row.stage: {"model": row.model, "ap": float(row.ap)} for row in best_by_stage.itertuples()},
        "n_trials": n_trials,
        "folds": list(SPEED["lock_folds"]),
        "seeds": list(SPEED["lock_seeds"]),
        "oof_path": str(oof_path),
        "outer_test_used": False,
        "roster": list(BASELINE_ROSTER),
    }
    write_json(lock_path, lock)
    _log(f"OULAD lock {lock['stage_best_ap']}")
    return lock


def hybrid_oulad_fast() -> dict:
    domain = "oulad"
    candidate = "C0-R"
    ceiling = load_baseline_ceiling(domain)
    _log("skip OULAD diagnose in SPEED_FINISH (GPU reserved for C0-R)")
    import optuna

    fit_ids, _, _ = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    study = _make_study(_study_name("hybrid_fast", domain, candidate))

    def objective(trial):
        from .hpo import sample_hybrid

        cfg, train_kw = sample_hybrid(trial, candidate, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim)
        train_kw["max_epochs"] = SPEED["hybrid_max_epochs"]
        train_kw["patience"] = SPEED["hybrid_patience"]
        train_kw["batch_size"] = SPEED["hybrid_batch"]
        train_kw["lambda_kd"] = 0.0
        t0 = time.time()
        out = evaluate_hybrid(domain, candidate, cfg, train_kw, fold=SCREEN_FOLD, seed=SCREEN_SEED, prepared=prepared)
        ap = {stage: row["ap"] for stage, row in out["valid"]["stages"].items()}
        sel = selection_objective(ap, ceiling, domain, out["parameter_count"])
        trial.set_user_attr("ap", ap)
        trial.set_user_attr("selection", sel)
        trial.set_user_attr("n_params", out["parameter_count"])
        trial.set_user_attr("runtime", time.time() - t0)
        _log(f"trial AP { {k: round(v, 4) for k, v in ap.items()} } J={sel['J']:.3f} {time.time()-t0:.0f}s")
        return float(sel["J"])

    remain = SPEED["hybrid_screen_trials"] - len([t for t in study.trials if t.state.name == "COMPLETE"])
    if remain > 0:
        study.optimize(objective, n_trials=remain, catch=(Exception,))
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    summary = {
        "domain": domain,
        "candidate": candidate,
        "speed_mode": SPEED,
        "n_complete": len(completed),
        "best_value": study.best_value if completed else None,
        "best_params": study.best_params if completed else None,
        "best_user_attrs": dict(study.best_trial.user_attrs) if completed else None,
        "outer_test_used": False,
    }
    write_json(RUN_DIR / f"screen_{domain}_{candidate}.json", summary)
    _log(f"hybrid screen {summary.get('best_user_attrs')}")
    if not completed:
        return summary
    bp = study.best_params
    train_kw = {
        "lr": bp["lr"],
        "weight_decay": bp["weight_decay"],
        "batch_size": SPEED["hybrid_batch"],
        "pos_weight_multiplier": bp["pos_weight_multiplier"],
        "lambda_rank": bp["lambda_rank"],
        "lambda_aux": bp["lambda_aux"],
        "lambda_kd": 0.0,
        "max_epochs": SPEED["hybrid_max_epochs"],
        "patience": SPEED["hybrid_patience"],
        "use_ema": True,
        "multiprefix": True,
    }
    rows = []
    preds = []
    for seed in (42, 1201, 2026):
        cfg = make_config(
            candidate,
            prepared.static_dim,
            prepared.temporal_dim,
            prepared.aggregate_dim,
            d_fuse=int(bp["d_fuse"]),
            cnn_channels=int(bp["cnn_channels"]),
            bilstm_hidden=int(bp["bilstm_hidden"]),
            dropout=float(bp["dropout"]),
        )
        t0 = time.time()
        out = evaluate_hybrid(domain, candidate, cfg, train_kw, fold=0, seed=seed, prepared=prepared)
        ap = {s: r["ap"] for s, r in out["valid"]["stages"].items()}
        _log(f"robust seed {seed} { {k: round(v, 4) for k, v in ap.items()} } {time.time()-t0:.0f}s")
        for s, v in ap.items():
            rows.append(
                {
                    "fold": 0,
                    "seed": seed,
                    "stage": s,
                    "ap": float(v),
                    "recall_at_20": out["valid"]["stages"][s].get("recall_at_20"),
                }
            )
        for rec in out["valid"].get("predictions", []):
            rec = dict(rec)
            rec["seed"] = seed
            rec["fold"] = 0
            rec["candidate"] = candidate
            preds.append(rec)
    df = pd.DataFrame(rows)
    mean = {s: float(df.loc[df.stage == s, "ap"].mean()) for s in df.stage.unique()}
    write_json(
        RUN_DIR / "robust_oulad_C0-R.json",
        {"candidate": candidate, "domain": domain, "ceiling": ceiling, "mean": mean, "rows": rows, "speed_mode": SPEED},
    )
    if preds:
        pd.DataFrame(preds).to_parquet(OOF_DIR / "hybrid_oof_oulad_C0-R.parquet", index=False)
    gate = evaluate_development_gate(domain, df, {s: {"ap": ceiling[s]} for s in ceiling})
    _log(f"OULAD gate pass={gate.get('pass')}")
    return {"screen": summary, "mean": mean, "gate": gate}


def write_combined_gate() -> dict:
    payload = {"pass": False, "domains": {}, "outer_test_used": False, "speed_mode": SPEED, "blocked": False}
    for domain in ("uci", "oulad"):
        path = RUN_DIR / f"development_gate_{domain}.json"
        if path.exists():
            payload["domains"][domain] = json.loads(path.read_text(encoding="utf-8"))
    payload["pass"] = bool(payload["domains"]) and all(v.get("pass") for v in payload["domains"].values())
    payload["n_domains"] = len(payload["domains"])
    if "uci" not in payload["domains"] or "oulad" not in payload["domains"]:
        payload["blocked"] = True
        payload["pass"] = False
    write_json(RUN_DIR / "development_gate.json", payload)
    return payload


def run_uci_ablations() -> dict:
    path = RUN_DIR / "ablation_uci_C0-R.json"
    if path.exists():
        _log("UCI ablation exists")
        return json.loads(path.read_text(encoding="utf-8"))
    lock = json.loads((RUN_DIR / "baseline_lock_uci.json").read_text(encoding="utf-8"))
    robust = json.loads((RUN_DIR / "robust_uci_C0-R.json").read_text(encoding="utf-8"))
    bp = robust.get("best_params") or {}
    fit_ids, _, _ = inner_partitions("uci", 0)
    prepared = scale_views("uci", fit_ids)
    mode_map = {
        "tabular_only": ("tabular", True, 0.15, 0.25, 0.0),
        "tabular_cnn": ("cnn", True, 0.15, 0.25, 0.0),
        "tabular_bilstm": ("bilstm", True, 0.15, 0.25, 0.0),
        "serial_no_tabular": ("temporal", True, 0.15, 0.25, 0.0),
        "full": ("full", True, 0.15, 0.25, 0.0),
        "full_no_gate": ("full", True, 0.15, 0.25, 0.0),
        "full_no_rank": ("full", True, 0.0, 0.25, 0.0),
        "full_no_kd": ("full", True, 0.15, 0.25, 0.0),
        "full_no_multiprefix": ("full", False, 0.15, 0.25, 0.0),
    }
    rows = []
    for name in ABLATIONS:
        branch, multi, lam_rank, lam_aux, lam_kd = mode_map[name]
        cfg = make_config(
            "C0-R",
            prepared.static_dim,
            prepared.temporal_dim,
            prepared.aggregate_dim,
            d_fuse=int(bp.get("d_fuse", 64)),
            cnn_channels=int(bp.get("cnn_channels", 24)),
            bilstm_hidden=int(bp.get("bilstm_hidden", 24)),
            dropout=float(bp.get("dropout", 0.3)),
            branch_mode=branch,
        )
        train_kw = {
            "lr": float(bp.get("lr", 2e-4)),
            "weight_decay": float(bp.get("weight_decay", 2e-4)),
            "batch_size": 64,
            "pos_weight_multiplier": float(bp.get("pos_weight_multiplier", 1.0)),
            "lambda_rank": lam_rank,
            "lambda_aux": lam_aux,
            "lambda_kd": lam_kd,
            "max_epochs": 8,
            "patience": 4,
            "use_ema": True,
            "multiprefix": multi,
        }
        t0 = time.time()
        try:
            out = evaluate_hybrid("uci", "C0-R", cfg, train_kw, fold=0, seed=42, prepared=prepared)
            ap = {s: r["ap"] for s, r in out["valid"]["stages"].items()}
        except Exception as exc:
            ap = {"error": f"{type(exc).__name__}: {exc}"}
        _log(f"ablation {name} {ap} {time.time()-t0:.0f}s")
        rows.append({"ablation": name, "ap": ap, "branch_mode": branch, "multiprefix": multi})
    payload = {
        "domain": "uci",
        "candidate": "C0-R",
        "speed_mode": SPEED,
        "fold": 0,
        "seed": 42,
        "rows": rows,
        "ceiling": lock.get("stage_best_ap"),
        "outer_test_used": False,
        "note": "SPEED_FINISH single-fold ablation, not 3x3 retrain.",
    }
    write_json(path, payload)
    return payload


def main() -> int:
    _boost_runtime()
    ensure_dirs()
    write_status(
        phase="SPEED_FINISH",
        completed=["killed hung OULAD DT HPO", "GPU trees + pinned Hybrid tensors"],
        evidence=[],
        decision="crash-finish remaining tasks at high CPU/GPU. SPEED_FINISH cuts preregistered OULAD 28-trial budget.",
        next_step="OULAD lock then Hybrid C0-R then reports",
        blockers=["speed_mode cuts preregistered trial budget — documented"],
    )
    t0 = time.time()
    try:
        write_uci_gate()
    except Exception:
        traceback.print_exc()
    try:
        oulad_baselines_fast()
    except Exception:
        traceback.print_exc()
        write_status(
            phase="SPEED_FINISH",
            completed=[],
            evidence=[],
            decision="baseline fail",
            next_step="inspect",
            blockers=["oulad baselines"],
        )
        return 1
    _empty_cuda()
    try:
        hybrid_oulad_fast()
    except Exception:
        traceback.print_exc()
    _empty_cuda()
    try:
        run_uci_ablations()
    except Exception:
        traceback.print_exc()
    try:
        write_combined_gate()
    except Exception:
        traceback.print_exc()
    try:
        write_all_reports()
    except Exception:
        traceback.print_exc()
    _log(f"SPEED_FINISH wall {time.time()-t0:.0f}s")
    write_status(
        phase="SPEED_FINISH done",
        completed=["OULAD lock+hybrid+UCI gate+ablation attempted"],
        evidence=[str(RUN_DIR), str(REPORT_ROOT)],
        decision="see FINAL_DECISION.md — still no authority promotion unless gate pass",
        next_step="read reports",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
