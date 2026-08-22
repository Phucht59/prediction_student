"""OULAD ceiling continuation. DT hard-timeout. TemporalSummaryCatBoost extra budget."""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

import numpy as np
import optuna
import pandas as pd

from experiments.hybrid_superiority_v2.baselines import default_params, fit_eval, predictor_columns, sample_space
from experiments.hybrid_superiority_v2.data import baseline_frame, inner_partitions, scale_views
from experiments.hybrid_superiority_v2.hpo import _make_study
from experiments.hybrid_superiority_v2.io_utils import git_commit, utc_now, write_json
from experiments.hybrid_superiority_v2.protocol import stages_for, warm_for

from .paths import METRIC_DIR, OOF_DIR, RUN_DIR, ensure_dirs
from .protocol import BASELINE_ROSTER, DT_TIMEOUT_SEC, HPO_BUDGET, SCREEN_FOLD, protocol_hash
from .thermal import wait_if_hot

SCREEN_SEED = 42


def _fit_eval_call(kwargs: dict) -> dict:
    return fit_eval(**kwargs)


def fit_eval_guarded(name: str, *args, timeout: int = DT_TIMEOUT_SEC, **kwargs) -> dict[str, Any]:
    payload = dict(
        name=name,
        frame=args[0] if args else kwargs["frame"],
        columns=args[1] if len(args) > 1 else kwargs["columns"],
        categorical=args[2] if len(args) > 2 else kwargs["categorical"],
        fit_ids=args[3] if len(args) > 3 else kwargs["fit_ids"],
        stop_ids=args[4] if len(args) > 4 else kwargs["stop_ids"],
        valid_ids=args[5] if len(args) > 5 else kwargs["valid_ids"],
        seed=args[6] if len(args) > 6 else kwargs["seed"],
        params=args[7] if len(args) > 7 else kwargs.get("params"),
    )
    if name != "DT":
        wait_if_hot()
        return fit_eval(**payload)
    wait_if_hot()
    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_fit_eval_call, payload)
            return fut.result(timeout=timeout)
    except FuturesTimeout as exc:
        raise TimeoutError(f"DT_TIMEOUT_{timeout}s") from exc


def _study_name(domain: str, model: str) -> str:
    return f"c4_v21_baseline_{domain}_{model}_{protocol_hash()[:12]}"


def run_baselines_domain(domain: str, models: tuple[str, ...] | None = None, folds=(0, 1, 2), seeds=(42, 1201, 2026)) -> dict:
    ensure_dirs()
    lock_path = RUN_DIR / f"baseline_lock_{domain}.json"
    models = models or tuple(m for m in BASELINE_ROSTER if m != "TemporalSummaryCatBoost" or domain == "oulad")
    fit_ids, stop_ids, valid_ids = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    frames = {stage: baseline_frame(prepared, stage) for stage in stages_for(domain)}
    best_params: dict[str, dict] = {}
    for name in models:
        family = "CatBoost" if name == "TemporalSummaryCatBoost" else name
        budget = HPO_BUDGET.get(name, HPO_BUDGET.get(family, {"min_trials": 40, "plateau": 20}))
        n_min = int(budget["min_trials"])
        plateau = int(budget.get("plateau", 20))
        study = _make_study(_study_name(domain, name))

        def objective(trial, family=family, name=name):
            params = sample_space(family, trial, domain=domain)
            if family == "CatBoost":
                params["iterations"] = min(int(params.get("iterations", 300)), 400)
            aps = []
            for stage in warm_for(domain):
                frame = frames[stage]
                cols, cats = predictor_columns(frame)
                present = set(frame.record_id.astype(str))
                sf = [i for i in fit_ids if i in present]
                ss = [i for i in stop_ids if i in present]
                sv = [i for i in valid_ids if i in present]
                if len(sf) < 30 or len(sv) < 15:
                    continue
                try:
                    metrics = fit_eval_guarded(family, frame, cols, cats, sf, ss, sv, SCREEN_SEED, params)
                except TimeoutError:
                    raise optuna.TrialPruned("DT_TIMEOUT")
                aps.append(metrics["ap"])
            if not aps:
                raise optuna.TrialPruned("empty")
            return float(np.mean(aps))

        while True:
            values = [t.value for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
            n_done = len(values)
            plateau_hit = False
            if n_done >= n_min and n_done >= plateau:
                best = max(values)
                recent_best = max(values[-plateau:])
                plateau_hit = (best - recent_best) < 0.001
            print(f"[baseline] {domain} {name} done={n_done}/{n_min} plateau={plateau_hit}", flush=True)
            if n_done >= n_min and plateau_hit:
                break
            if n_done >= max(n_min * 2, n_min + plateau):
                break
            remain = min(8, max(1, n_min - n_done if n_done < n_min else plateau // 4 or 4))
            study.optimize(objective, n_trials=remain, catch=(Exception,))
            wait_if_hot()
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        best_params[name] = study.best_params if completed else default_params(family, SCREEN_SEED)

    rows, oof = [], []
    lock_folds = folds if domain == "uci" else (0, 1, 2)
    lock_seeds = seeds
    for fold in lock_folds:
        fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
        prepared = scale_views(domain, fit_ids)
        frames = {stage: baseline_frame(prepared, stage) for stage in stages_for(domain)}
        for seed in lock_seeds:
            for name in models:
                family = "CatBoost" if name == "TemporalSummaryCatBoost" else name
                params = best_params[name]
                print(f"[lock] {domain} {name} fold={fold} seed={seed}", flush=True)
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
                    try:
                        metrics = fit_eval_guarded(family, frame, cols, cats, sf, ss, sv, seed, params)
                    except TimeoutError:
                        continue
                    rows.append(
                        {
                            "domain": domain,
                            "model": name,
                            "fold": fold,
                            "seed": seed,
                            "stage": stage,
                            "ap": metrics["ap"],
                            "n": metrics["n"],
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
                print(f"  {time.time()-t0:.0f}s", flush=True)
                wait_if_hot()
    table = pd.DataFrame(rows)
    table.to_csv(METRIC_DIR / f"baseline_stage_metrics_{domain}.csv", index=False)
    oof_path = OOF_DIR / f"baseline_oof_{domain}.parquet"
    pd.DataFrame(oof).to_parquet(oof_path, index=False)
    ceiling = table.groupby(["stage", "model"])["ap"].mean().reset_index()
    best_by_stage = ceiling.loc[ceiling.groupby("stage")["ap"].idxmax()]
    lock = {
        "domain": domain,
        "protocol_hash": protocol_hash(),
        "git_commit": git_commit(),
        "frozen_at": utc_now(),
        "best_params": best_params,
        "stage_best_ap": {row.stage: {"model": row.model, "ap": float(row.ap)} for row in best_by_stage.itertuples()},
        "folds": list(lock_folds),
        "seeds": list(lock_seeds),
        "oof_path": str(oof_path),
        "outer_test_used": False,
        "roster": list(models),
        "speed_finish_not_used": True,
    }
    write_json(lock_path, lock)
    return lock
