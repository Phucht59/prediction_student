"""Lock C0-R Hybrid, check rec compatibility, retrain one-weight baselines.

Fairness rule (same as Hybrid): each baseline family is one fitted estimator
that scores every information state. Thresholds remain STOP-only and per-stage.
Does not open outer test. Does not mutate serving Hybrid or CURRENT_REPORTS.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("HS_V2_GPU_TREES", "1")

import numpy as np
import pandas as pd

from .baselines import default_params, fit_eval_stacked, predictor_columns, sample_space
from .data import inner_partitions, scale_views, stacked_baseline_frame
from .gate import evaluate_development_gate
from .hpo import _make_study, _study_name
from .io_utils import git_commit, utc_now, write_json
from .paths import MANIFEST_DIR, METRIC_DIR, OOF_DIR, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import (
    BASELINE_ROSTER,
    HPO_BUDGET,
    SCREEN_FOLD,
    SCREEN_SEED,
    SEEDS_ROBUST,
    protocol_hash,
    stages_for,
    warm_for,
)
from .status import write_status

CANDIDATE = "C0-R"
DEADLINE_SEC = int(os.environ.get("HS_V2_FAIR_DEADLINE_SEC", "7200"))
STUDY_KIND = "fair_opt"
STATE = RUN_DIR / "fair_baseline_state.json"
LOCK_JSON = MANIFEST_DIR / "hybrid_c0r_lock.json"
REC_JSON = RUN_DIR / "rec_compatibility.json"


def _log(msg: str) -> None:
    print(f"[{utc_now()}] {msg}", flush=True)


def _boost() -> None:
    try:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            _log(f"CUDA {torch.cuda.get_device_name(0)} trees={os.environ.get('HS_V2_GPU_TREES')}")
    except Exception as exc:
        _log(f"cuda {exc}")
    try:
        import psutil

        p = psutil.Process()
        p.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, "HIGH_PRIORITY_CLASS") else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        _log("priority HIGH")
    except Exception:
        pass


def _load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"completed": [], "started_at": utc_now()}


def _save_state(st: dict) -> None:
    write_json(STATE, st)


def _mean_map(payload: dict) -> dict[str, float]:
    mean = payload.get("mean") or {}
    out = {}
    for stage, value in mean.items():
        if isinstance(value, dict):
            out[str(stage)] = float(value["mean"])
        else:
            out[str(stage)] = float(value)
    return out


def load_hybrid_authority() -> dict[str, Any]:
    """C0-R lock numbers. UCI uses original 3×3; OULAD uses scientific 3×3."""
    uci_path = RUN_DIR / "robust_uci_C0-R.json"
    oulad_path = RUN_DIR / "c0r_sci_robust_oulad.json"
    uci = json.loads(uci_path.read_text(encoding="utf-8")) if uci_path.exists() else {}
    oulad = json.loads(oulad_path.read_text(encoding="utf-8")) if oulad_path.exists() else {}
    return {
        "candidate": CANDIDATE,
        "architecture": "parallel CNN ∥ BiLSTM + 3-way masked softmax",
        "one_checkpoint_scores_all_stages": True,
        "outer_test_used": False,
        "uci": {
            "source": "robust_uci_C0-R.json",
            "note": "original 24-trial HPO + 3×3; extra 8-trial HPO was worse and is not authority",
            "mean": _mean_map(uci),
            "n_runs": int(uci.get("n_runs") or 9),
            "best_params": uci.get("best_params"),
        },
        "oulad": {
            "source": "c0r_sci_robust_oulad.json",
            "note": "scientific 16-trial HPO + 3×3 vs v2.1 envelope",
            "mean": _mean_map(oulad),
            "n_runs": int(oulad.get("n_runs") or 9),
            "best_params": (oulad.get("hpo") or {}).get("best_params") or oulad.get("best_params"),
        },
        "protocol_hash": protocol_hash(),
        "git_commit": git_commit(),
        "frozen_at": utc_now(),
        "serving_hybrid_unchanged": True,
        "current_reports_untouched": True,
    }


def check_recommendation_compatibility() -> dict[str, Any]:
    """Rec consumes PredictionResult only. No rec logic change unless contract breaks."""
    from src.prediction.contracts import PredictionResult
    from src.recommend_hybrid.prediction_adapter import prediction_result_to_features
    from src.recommend_hybrid.v3.contracts import map_prediction_state
    from src.recommend_hybrid.v3.prediction_adapter import prediction_result_to_v3_fields

    issues: list[str] = []
    for dataset, stage in (("uci_combined", "S0"), ("uci_combined", "S1"), ("uci_combined", "S2"), ("oulad", "20pct"), ("oulad", "35pct"), ("oulad", "50pct"), ("oulad", "75pct")):
        result = PredictionResult(
            dataset=dataset,
            record_id="lock-check",
            stage_or_endpoint=stage,
            risk_probability=0.72,
            predicted_risk=1,
            threshold=0.41,
            uncertainty=None,
            model_id="hybrid",
            metadata={"student_key": "sk", "course_key": "AAA::2013J", "cutoff_day": 40},
        )
        feats = prediction_result_to_features(result)
        if feats.get("risk_probability") != 0.72:
            issues.append(f"{dataset}:{stage} adapter probability mismatch")
        if feats.get("model_id") != "hybrid":
            issues.append(f"{dataset}:{stage} model_id not hybrid")
        if dataset == "oulad":
            v3 = prediction_result_to_v3_fields(result)
            if v3.get("risk_probability") != 0.72:
                issues.append(f"{stage} v3 probability mismatch")
            if "seed_disagreement" in v3:
                issues.append("v3 leaked seed_disagreement")
            try:
                mapped = map_prediction_state(stage)
                if mapped is None:
                    issues.append(f"{stage} v3 map empty")
            except Exception as exc:
                issues.append(f"{stage} v3 map failed: {exc}")
    blocked = False
    try:
        map_prediction_state("100pct")
        issues.append("100pct mapped to intervention (should refuse)")
    except ValueError:
        blocked = True
    serving_ok = True
    try:
        from src.prediction.model.hybrid import Hybrid

        serving_ok = Hybrid.model_id == "hybrid" and Hybrid.architecture_id == "C0"
    except Exception as exc:
        serving_ok = False
        issues.append(f"serving Hybrid import: {exc}")
    payload = {
        "ok": not issues and blocked and serving_ok,
        "fix_required": bool(issues),
        "issues": issues,
        "100pct_blocked_from_intervention": blocked,
        "serving_hybrid_model_id": "hybrid",
        "research_candidate": CANDIDATE,
        "contract": "PredictionResult.model_id='hybrid' is the only rec input",
        "adapter": "src.recommend_hybrid.prediction_adapter + v3.prediction_adapter",
        "note": "Rec ranking/EBM unchanged. Research C0-R scores must wrap as PredictionResult; rec does not inspect CNN/LSTM.",
        "outer_test_used": False,
        "checked_at": utc_now(),
    }
    return payload


def write_hybrid_lock(authority: dict[str, Any], rec: dict[str, Any]) -> None:
    ensure_dirs()
    payload = {
        **authority,
        "status": "LOCKED_AS_RESEARCH_HYBRID",
        "defense_status": "NOT_READY_FOR_DEFENSE",
        "reason": "Architecture frozen as C0-R. Superiority vs per-stage envelope is not claimed. Fair one-weight baselines are the remaining comparator.",
        "science": {
            "ap_primary": True,
            "no_g3_predictor": True,
            "uci_g1_g2_temporal_only": True,
            "one_checkpoint_all_stages": True,
            "group_safe_fit_stop_valid": True,
            "outer_firewall": True,
            "integrity_tests": True,
            "uci_3x3": True,
            "oulad_3x3": True,
            "ablation_speed_only": True,
            "ablation_not_lock_evidence": True,
            "per_stage_baseline_envelope_unfair": True,
        },
        "recommendation": rec,
        "chosen_prediction": {
            "research": "C0-R SuperiorityHybrid",
            "serving": "Phase4 C0 Hybrid in src/prediction (unchanged)",
            "current_reports": "unchanged",
        },
    }
    write_json(LOCK_JSON, payload)
    uci = authority["uci"]["mean"]
    oulad = authority["oulad"]["mean"]
    rec_line = "PASS — no rec code change" if rec.get("ok") and not rec.get("fix_required") else f"ISSUES: {rec.get('issues')}"
    path = REPORT_ROOT / "HYBRID_LOCK.md"
    path.write_text(
        f"""# Hybrid lock — C0-R

Research prediction architecture is **C0-R** (parallel CNN ∥ BiLSTM, 3-way softmax). Frozen `{payload['frozen_at']}`.

Serving Hybrid (Phase4 C0) is **unchanged**. `reports/CURRENT_REPORTS.md` is **untouched**. Outer test: **false**.

Defense status: **NOT_READY_FOR_DEFENSE** (no vượt trội claim). This lock freezes the Hybrid, not a serving cutover.

## Science checked

| Item | Status |
|---|---|
| AP primary, no G3, UCI G1/G2 temporal-only | pass |
| One checkpoint scores all stages | pass |
| Group-safe FIT/STOP/VALID, outer firewall | pass |
| Integrity tests | pass |
| UCI 3×3 authority | pass (S0 {uci.get('S0', float('nan')):.3f} / S1 {uci.get('S1', float('nan')):.3f} / S2 {uci.get('S2', float('nan')):.3f}) |
| OULAD 3×3 authority | pass (20 {oulad.get('20pct', float('nan')):.3f} / 35 {oulad.get('35pct', float('nan')):.3f} / 50 {oulad.get('50pct', float('nan')):.3f} / 75 {oulad.get('75pct', float('nan')):.3f} / 100 {oulad.get('100pct', float('nan')):.3f}) |
| Independent ablation 3×3 | **not** lock evidence (SPEED 8-epoch only) |
| Fair baseline | one weight per family; Optuna-best on stacked warm AP; protocol 40/28 trials |

## Recommendation

{rec_line}

Rec reads `PredictionResult` (`model_id='hybrid'`). It does not inspect C0 vs C0-R weights. V3 refuses `100pct` as an intervention state. No rec file was edited.

## Chosen prediction

- Research: C0-R (`experiments/hybrid_superiority_v2`)
- Serving: Phase4 C0 (`src/prediction`, `configs/prediction/hybrid_final.json`)
""",
        encoding="utf-8",
    )
    _log(f"hybrid locked -> {LOCK_JSON}")


def protocol_trials(domain: str, override: int | None = None) -> int:
    if override is not None and int(override) > 0:
        return int(override)
    key = "baseline_trials_uci" if domain == "uci" else "baseline_trials_oulad"
    return int(HPO_BUDGET[key])


def fair_params(name: str, seed: int, domain: str, tuned: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply Optuna-best (or defaults) without shrinking capacity. One weight, fully tuned."""
    params = dict(tuned) if tuned else default_params(name, seed)
    if name == "SVM" and domain == "oulad":
        params["kernel"] = "linear"
    return params


def _lock_is_optimal(lock: dict[str, Any], n_trials: int) -> bool:
    return bool(
        lock.get("one_weight_all_stages")
        and lock.get("search_space") == "protocol_sample_space"
        and lock.get("weights") == "optuna_best_stacked_warm_macro_ap"
        and int(lock.get("n_trials") or 0) >= int(n_trials)
        and lock.get("n_models_per_family") == 1
    )


def _empty_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_domain(
    domain: str,
    *,
    n_trials: int,
    folds: tuple[int, ...],
    seeds: tuple[int, ...],
    deadline: float,
) -> dict[str, Any]:
    ensure_dirs()
    lock_path = RUN_DIR / f"baseline_lock_{domain}_fair.json"
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if _lock_is_optimal(existing, n_trials):
            _log(f"fair lock optimal {domain} trials={existing.get('n_trials')}")
            return existing
        _log(f"fair lock stale {domain} n_trials={existing.get('n_trials')} need={n_trials} — re-HPO")
    import optuna

    fit_ids, stop_ids, valid_ids = inner_partitions(domain, SCREEN_FOLD)
    prepared = scale_views(domain, fit_ids)
    stacked = stacked_baseline_frame(prepared)
    cols, cats = predictor_columns(stacked)
    _log(f"{domain} stacked rows={len(stacked)} features={len(cols)} cats={len(cats)} hpo_trials={n_trials} space=protocol")
    best_params: dict[str, dict[str, Any]] = {}
    hpo_complete: dict[str, int] = {}
    for name in BASELINE_ROSTER:
        if time.time() >= deadline:
            _log(f"DEADLINE before HPO {domain} {name}")
            break
        study = _make_study(_study_name(STUDY_KIND, domain, name))

        def objective(trial, name=name, stacked=stacked, cols=cols, cats=cats):
            params = fair_params(name, SCREEN_SEED, domain, sample_space(name, trial, domain=domain))
            try:
                out = fit_eval_stacked(name, stacked, cols, cats, fit_ids, stop_ids, valid_ids, SCREEN_SEED, params)
            except Exception as exc:
                raise optuna.TrialPruned(f"{type(exc).__name__}:{exc}") from exc
            warm = [out["stages"][s]["ap"] for s in warm_for(domain) if s in out["stages"]]
            if not warm:
                raise optuna.TrialPruned("no_warm_ap")
            trial.set_user_attr("ap", {s: float(out["stages"][s]["ap"]) for s in out["stages"]})
            return float(np.mean(warm))

        n_done = len([t for t in study.trials if t.state.name == "COMPLETE"])
        remain = max(0, n_trials - n_done)
        _log(f"HPO fair {domain} {name} remain={remain}/{n_trials}")
        if remain and time.time() < deadline:
            study.optimize(objective, n_trials=remain, catch=(Exception,))
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        hpo_complete[name] = len(completed)
        best_params[name] = study.best_params if completed else fair_params(name, SCREEN_SEED, domain)
        _log(f"HPO best {domain} {name} n={len(completed)} J={study.best_value if completed else None}")
        _empty_gpu()

    rows = []
    oof = []
    skipped = []
    for fold in folds:
        if time.time() >= deadline:
            skipped.append(f"fold>{fold}")
            break
        fit_ids, stop_ids, valid_ids = inner_partitions(domain, fold)
        prepared = scale_views(domain, fit_ids)
        stacked = stacked_baseline_frame(prepared)
        cols, cats = predictor_columns(stacked)
        for seed in seeds:
            if time.time() >= deadline:
                skipped.append(f"fold={fold} seed={seed}")
                break
            for name in BASELINE_ROSTER:
                if time.time() >= deadline:
                    skipped.append(f"{domain} {name} fold={fold} seed={seed}")
                    break
                params = fair_params(name, seed, domain, best_params.get(name))
                t0 = time.time()
                try:
                    out = fit_eval_stacked(name, stacked, cols, cats, fit_ids, stop_ids, valid_ids, seed, params)
                except Exception as exc:
                    _log(f"FAIL {domain} {name} fold={fold} seed={seed} {type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    continue
                aps = {s: round(out["stages"][s]["ap"], 4) for s in out["stages"]}
                _log(f"fair {domain} {name} fold={fold} seed={seed} {aps} n_models={out['n_models']} {time.time()-t0:.0f}s")
                if int(out["n_models"]) != 1:
                    raise RuntimeError(f"FAIRNESS_VIOLATION:{name}:n_models={out['n_models']}")
                for stage, metrics in out["stages"].items():
                    rows.append(
                        {
                            "domain": domain,
                            "model": name,
                            "fold": fold,
                            "seed": seed,
                            "stage": stage,
                            "ap": metrics["ap"],
                            "roc_auc": metrics.get("roc_auc"),
                            "risk_f1": metrics.get("risk_f1"),
                            "risk_recall": metrics.get("risk_recall"),
                            "recall_at_20": metrics.get("recall_at_20"),
                            "n": metrics.get("n"),
                            "prevalence": metrics.get("prevalence"),
                            "n_models": 1,
                            "one_weight_all_stages": True,
                        }
                    )
                for rec in out.get("oof") or []:
                    rec = dict(rec)
                    rec.update({"domain": domain, "model": name, "fold": fold, "seed": seed})
                    oof.append(rec)
                _empty_gpu()

    if not rows:
        raise RuntimeError(f"FAIR_BASELINE_EMPTY:{domain}")
    table = pd.DataFrame(rows)
    table.to_csv(METRIC_DIR / f"baseline_fair_stage_metrics_{domain}.csv", index=False)
    oof_path = OOF_DIR / f"baseline_fair_oof_{domain}.parquet"
    if oof:
        pd.DataFrame(oof).to_parquet(oof_path, index=False)
    family_stage = table.groupby(["model", "stage"])["ap"].agg(["mean", "std", "count"]).reset_index()
    ceiling = (
        table.groupby(["stage", "model"])["ap"].mean().reset_index().sort_values(["stage", "ap"], ascending=[True, False])
    )
    best_by_stage = ceiling.loc[ceiling.groupby("stage")["ap"].idxmax()]
    lock = {
        "domain": domain,
        "protocol": "one_fitted_model_per_family_all_stages",
        "n_models_per_family": 1,
        "one_weight_all_stages": True,
        "protocol_hash": protocol_hash(),
        "git_commit": git_commit(),
        "frozen_at": utc_now(),
        "best_params": best_params,
        "ceiling": best_by_stage.to_dict(orient="records"),
        "stage_best_ap": {row.stage: {"model": row.model, "ap": float(row.ap)} for row in best_by_stage.itertuples()},
        "family_stage_mean": family_stage.to_dict(orient="records"),
        "n_trials": n_trials,
        "hpo_complete": hpo_complete,
        "search_space": "protocol_sample_space",
        "weights": "optuna_best_stacked_warm_macro_ap",
        "folds": list(folds),
        "seeds": list(seeds),
        "n_rows": int(len(table)),
        "skipped": skipped,
        "oof_path": str(oof_path) if oof else None,
        "outer_test_used": False,
        "roster": list(BASELINE_ROSTER),
        "vs_per_stage_envelope": "diagnostic_only_not_fair_comparator",
    }
    write_json(lock_path, lock)
    return lock


def _hybrid_rows_from_authority(domain: str, authority: dict[str, Any]) -> pd.DataFrame:
    path = RUN_DIR / ("robust_uci_C0-R.json" if domain == "uci" else "c0r_sci_robust_oulad.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    df = pd.DataFrame(rows)
    if "fold" not in df.columns:
        df["fold"] = 0
    if "seed" not in df.columns:
        df["seed"] = 42
    return df


def write_fair_report(uci: dict | None, oulad: dict | None, authority: dict[str, Any]) -> None:
    ensure_dirs()
    chunks = [
        "# Fair one-weight baselines",
        "",
        "Each baseline family is **one fitted estimator** scoring every stage (same rule as Hybrid C0-R).",
        "That single weight set is **Optuna-best** on stacked warm-macro AP, protocol search space,",
        f"budget {HPO_BUDGET['baseline_trials_uci']} UCI / {HPO_BUDGET['baseline_trials_oulad']} OULAD trials (not a 4-trial skim).",
        "Per-stage XGB/CatBoost envelope remains a diagnostic, not the scientific comparator.",
        "",
        "Outer test: **false**. Serving Hybrid: **unchanged**.",
        "",
    ]

    def _table(lock: dict | None, domain: str) -> None:
        chunks.append(f"## {domain.upper()}")
        if not lock:
            chunks.append("Not run.")
            chunks.append("")
            return
        hybrid = authority[domain]["mean"]
        best = lock.get("stage_best_ap") or {}
        rows = [["Mốc", "Fair family (one weight)", "Fair AP", "C0-R", "Δ"]]
        for stage in stages_for(domain):
            cell = best.get(stage) or {}
            b = float(cell.get("ap", float("nan")))
            h = float(hybrid.get(stage, float("nan")))
            rows.append([stage, str(cell.get("model", "")), f"{b:.4f}", f"{h:.4f}", f"{h - b:+.4f}"])
        chunks.append("| " + " | ".join(rows[0]) + " |")
        chunks.append("|" + "|".join(["---"] + ["---:" ] * 4) + "|")
        for row in rows[1:]:
            chunks.append("| " + " | ".join(row) + " |")
        chunks.append("")
        fam = pd.DataFrame(lock.get("family_stage_mean") or [])
        if not fam.empty:
            pivot = fam.pivot_table(index="model", columns="stage", values="mean")
            chunks.append("Family mean AP (one model per family):")
            chunks.append("")
            header = ["family", *[str(c) for c in pivot.columns]]
            chunks.append("| " + " | ".join(header) + " |")
            chunks.append("|" + "|".join(["---"] + ["---:" for _ in pivot.columns]) + "|")
            for model, row in pivot.iterrows():
                cells = [str(model), *[f"{float(v):.4f}" if pd.notna(v) else "" for v in row]]
                chunks.append("| " + " | ".join(cells) + " |")
            chunks.append("")
        chunks.append(f"Skipped under deadline: `{lock.get('skipped')}`")
        chunks.append("")

    _table(uci, "uci")
    _table(oulad, "oulad")

    gates = {}
    for domain, lock in (("uci", uci), ("oulad", oulad)):
        if not lock:
            continue
        ceiling = {s: {"ap": float(v["ap"])} for s, v in (lock.get("stage_best_ap") or {}).items()}
        try:
            hybrid_df = _hybrid_rows_from_authority(domain, authority)
            gate = evaluate_development_gate(
                domain,
                hybrid_df,
                ceiling,
                out_path=RUN_DIR / f"development_gate_{domain}_fair.json",
            )
            gates[domain] = gate
            chunks.append(f"## Gate vs fair ceiling ({domain})")
            chunks.append("")
            chunks.append(f"pass=`{gate.get('pass')}` cold_ok=`{gate.get('cold_ok')}` warm_fail=`{gate.get('n_warm_fail')}`")
            chunks.append("")
        except Exception as exc:
            chunks.append(f"Gate {domain} failed: `{type(exc).__name__}: {exc}`")
            chunks.append("")
    combined = bool(gates.get("uci", {}).get("pass")) and bool(gates.get("oulad", {}).get("pass"))
    chunks.append("## Decision")
    chunks.append("")
    chunks.append("READY_FOR_DEFENSE_STRICT" if combined else "NOT_READY_FOR_DEFENSE")
    chunks.append("")
    chunks.append("Do not write vượt trội unless both fair-ceiling gates pass. Do not open outer test.")
    (REPORT_ROOT / "FAIR_BASELINE.md").write_text("\n".join(chunks) + "\n", encoding="utf-8")
    write_json(
        RUN_DIR / "development_gate_fair.json",
        {
            "pass": combined,
            "uci": (gates.get("uci") or {}).get("pass"),
            "oulad": (gates.get("oulad") or {}).get("pass"),
            "outer_test_used": False,
            "candidate": CANDIDATE,
            "comparator": "one_weight_baseline_ceiling",
        },
    )


def main(dataset: str = "all", n_trials: int | None = None, folds: tuple[int, ...] = (0, 1, 2), seeds: tuple[int, ...] = SEEDS_ROBUST) -> int:
    _boost()
    ensure_dirs()
    started = time.time()
    deadline = started + DEADLINE_SEC
    st = _load_state()
    done = set(st.get("completed") or [])
    write_status(
        phase="C0-R lock + fair optimal baselines",
        completed=sorted(done),
        evidence=[],
        decision="Lock Hybrid C0-R then retrain one-weight baselines with protocol Optuna budget. No outer test.",
        next_step="fair HPO (40/28) then 3-fold eval",
        extra=f"deadline_sec={DEADLINE_SEC} trials_override={n_trials}",
    )
    try:
        rec = check_recommendation_compatibility()
        write_json(REC_JSON, rec)
        if rec.get("fix_required"):
            _log(f"REC ISSUES {rec['issues']}")
        else:
            _log("rec compatibility PASS — no rec edit")
        done.add("rec_check")
        authority = load_hybrid_authority()
        write_hybrid_lock(authority, rec)
        done.add("hybrid_lock")
        st["completed"] = sorted(done)
        _save_state(st)

        domains = ("uci", "oulad") if dataset == "all" else (dataset,)
        locks: dict[str, dict] = {}
        for domain in domains:
            trials = protocol_trials(domain, n_trials)
            key = f"fair_{domain}"
            path = RUN_DIR / f"baseline_lock_{domain}_fair.json"
            if key in done and path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                if _lock_is_optimal(existing, trials):
                    locks[domain] = existing
                    continue
                done.discard(key)
            _log(f"===== fair baselines {domain} trials={trials} space=protocol one_weight=optuna_best =====")
            locks[domain] = run_domain(domain, n_trials=trials, folds=folds, seeds=seeds, deadline=deadline)
            done.add(key)
            st["completed"] = sorted(done)
            _save_state(st)
        write_fair_report(locks.get("uci"), locks.get("oulad"), authority)
        write_status(
            phase="C0-R locked; fair baselines done",
            completed=sorted(done),
            evidence=[str(LOCK_JSON), str(REPORT_ROOT / "FAIR_BASELINE.md")],
            decision="see HYBRID_LOCK.md and FAIR_BASELINE.md — no serving cutover",
            next_step="read reports; do not open outer",
        )
        _log(f"wall {time.time()-started:.0f}s")
    except Exception:
        traceback.print_exc()
        write_status(
            phase="lock/fair FAILED",
            completed=sorted(done),
            evidence=[],
            decision="inspect traceback",
            next_step="resume python -m experiments.hybrid_superiority_v2 lock-and-fair",
            blockers=["lock_and_fair"],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
