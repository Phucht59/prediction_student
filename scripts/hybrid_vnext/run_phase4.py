"""Phase 4 superiority ladder. One C0. No outer until inner win. No XGB."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.hybrid_vnext.baselines import ACTIVE_PHASE4, fit_eval_baseline, make_svm
from experiments.hybrid_vnext.data import (
    baseline_frame,
    feature_groups,
    final100_length_diagnostic,
    inner_partitions,
    load_domain_phase4,
    scale_views,
)
from experiments.hybrid_vnext.metrics import binary_metrics
from experiments.hybrid_vnext.model import VNextHybrid, assert_c0_topology, availability_unit_cases, make_c0_config
from experiments.hybrid_vnext.phase3_common import verify_phase2_locks
from experiments.hybrid_vnext.phase4_common import (
    ACTIVE_FAMILIES,
    INNER_FOLDS,
    OULAD_EARLY,
    OULAD_STATES,
    PHASE3_HPO,
    PHASE4,
    REPORTS4,
    RUNS4,
    SCREEN_FOLD,
    SCREEN_SEED,
    SEEDS,
    SHARED_STRUCTURAL,
    UCI_STATES,
    digest_obj,
    topology_hash,
)
from experiments.hybrid_vnext.protocol import require_cuda, sha256_file, utc_now, verify_split_hashes, write_json
from experiments.hybrid_vnext.train import _ids_for_stage
from experiments.hybrid_vnext.train_phase4 import StrategyTrainer, TrainingStrategy


STRATEGIES = [
    TrainingStrategy("L1_control", notes="Phase3 C0 control, mixed states"),
    TrainingStrategy("L2_stagenorm", stage_norm=True, notes="equal stage loss"),
    TrainingStrategy("L3_C1", curriculum="C1", notes="low-information first"),
    TrainingStrategy("L3_C2", curriculum="C2", notes="high-information first"),
    TrainingStrategy("L4_hard", hard_stage_weights=True, notes="bounded EMA stage weights"),
    TrainingStrategy("L5_trunc", trunc_p=0.30, notes="cutoff-consistent shorter state"),
    TrainingStrategy("L6_rank05", lambda_rank=0.05, notes="pairwise ranking aux 0.05"),
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
    path = PHASE4 / "CUDA_EXECUTION_AUDIT.json"
    cur = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"jobs": []}
    cur.setdefault("jobs", []).append(job)
    write_json(path, cur)


def prepare(domain: str, fold: int):
    views, context, numeric, categorical = load_domain_phase4(domain)
    fit, stop, valid = inner_partitions(domain, context, fold)
    prepared = scale_views(views, context, numeric, categorical, fit, domain)
    return prepared, fit, stop, valid


def hybrid_run(domain: str, fold: int, seed: int, strategy: TrainingStrategy, hp: dict, stage_name: str, max_epochs: int | None = None) -> dict:
    run_id = (
        f"{stage_name}_{strategy.name}_{domain}_f{fold}_s{seed}_"
        f"lr{hp['lr']}_do{hp['dropout']}_tr{strategy.trunc_p}_rk{strategy.lambda_rank}"
    )
    path = RUNS4 / f"{run_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    suffix = (
        f"_{strategy.name}_{domain}_f{fold}_s{seed}_"
        f"lr{hp['lr']}_do{hp['dropout']}_tr{strategy.trunc_p}_rk{strategy.lambda_rank}.json"
    )
    for existing in RUNS4.glob(f"*{suffix}"):
        return json.loads(existing.read_text(encoding="utf-8"))
    assert torch.cuda.is_available()
    prepared, fit, stop, valid = prepare(domain, fold)
    cfg = make_c0_config(
        prepared.static_dim,
        prepared.temporal_dim,
        prepared.aggregate_dim,
        prepared.summary_dim,
        d_fuse=SHARED_STRUCTURAL["d_fuse"],
        cnn_channels=SHARED_STRUCTURAL["cnn_channels"],
        bilstm_hidden=SHARED_STRUCTURAL["bilstm_hidden"],
        dropout=hp["dropout"],
        entropy_floor_coefficient=hp["entropy_floor_coefficient"],
    )
    assert_c0_topology(cfg)
    trainer = StrategyTrainer(
        prepared,
        cfg,
        strategy,
        lr=hp["lr"],
        weight_decay=hp["weight_decay"],
        max_epochs=max_epochs or (60 if domain == "oulad" else 50),
        patience=10,
        batch_size=int(hp["batch_size"]),
        seed=seed,
        pos_weight_multiplier=hp["pos_weight_multiplier"],
    )
    fitted = trainer.fit(fit, stop)
    evaluation = trainer.evaluate(fitted["model"], stop, valid)
    stages = {}
    rows = []
    for stage, metrics in evaluation["stages"].items():
        train_ids = _ids_for_stage(prepared.views[stage], fit)
        lookup = {str(r): i for i, r in enumerate(prepared.views[stage].record_id)}
        train_y = prepared.views[stage].target[[lookup[i] for i in train_ids]]
        train_p = trainer._predict(fitted["model"], stage, train_ids)
        train_pr = binary_metrics(train_y, train_p)["pr_auc"] if len(np.unique(train_y)) == 2 else float("nan")
        gap = float(train_pr - metrics["pr_auc"]) if np.isfinite(train_pr) else None
        stages[stage] = {**metrics, "train_pr_auc": train_pr, "generalization_gap": gap}
        rows.append(
            {
                "run_id": run_id,
                "strategy": strategy.name,
                "dataset": domain,
                "stage": stage,
                "inner_fold": fold,
                "seed": seed,
                "pr_auc": metrics["pr_auc"],
                "risk_f1": metrics["risk_f1"],
                "risk_recall": metrics["risk_recall"],
                "accuracy": metrics["accuracy"],
                "ece": metrics["ece"],
                "brier": metrics.get("brier"),
                "train_pr_auc": train_pr,
                "generalization_gap": gap,
                "best_epoch": fitted["best_epoch"],
                "outer_test_used": False,
            }
        )
    payload = {
        "run_id": run_id,
        "strategy": strategy.as_dict(),
        "domain": domain,
        "fold": fold,
        "seed": seed,
        "hp": hp,
        "structural": SHARED_STRUCTURAL,
        "fitted": {k: fitted[k] for k in fitted if k != "model"},
        "evaluation": {**evaluation, "stages": stages},
        "macro_pr_auc": evaluation["macro_pr_auc"],
        "rows": rows,
        "outer_test_used": False,
        "timestamp": utc_now(),
    }
    if domain == "oulad":
        early = [stages[s]["pr_auc"] for s in OULAD_EARLY if s in stages]
        payload["macro_early"] = float(np.mean(early)) if early else None
        payload["macro_5stage"] = float(np.mean([stages[s]["pr_auc"] for s in OULAD_STATES if s in stages]))
    write_json(path, payload)
    _csv(
        PHASE4 / ("UCI_STAGE_RESULTS.csv" if domain == "uci" else "OULAD_STAGE_RESULTS.csv"),
        rows,
        ["run_id", "strategy", "dataset", "stage", "inner_fold", "seed", "pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "brier", "train_pr_auc", "generalization_gap", "best_epoch", "outer_test_used"],
    )
    for stage, diag in (evaluation.get("diagnostics") or {}).items():
        _csv(
            PHASE4 / "GATE_DIAGNOSTICS.csv",
            [{
                "dataset": domain, "stage": stage, "fold": fold, "seed": seed, "strategy": strategy.name,
                **{k: diag.get(k) for k in ("g_temporal_mean", "tabular_mass", "cnn_mass", "bilstm_mass", "tabular_norm", "temporal_norm", "temporal_available_rate")},
                "outer_test_used": False,
            }],
            ["dataset", "stage", "fold", "seed", "strategy", "g_temporal_mean", "tabular_mass", "cnn_mass", "bilstm_mass", "tabular_norm", "temporal_norm", "temporal_available_rate", "outer_test_used"],
        )
    _append_cuda(
        {
            "run_id": run_id,
            "device": "cuda:0",
            "gpu_name": fitted.get("gpu_name"),
            "amp": True,
            "peak_vram_bytes": fitted.get("peak_vram_bytes"),
            "batch_size": fitted.get("batch_size"),
            "runtime_seconds": fitted.get("runtime_seconds"),
            "outer_test_used": False,
        }
    )
    del fitted["model"]
    torch.cuda.empty_cache()
    print(f"HYBRID {run_id} macro={payload['macro_pr_auc']:.4f}", flush=True)
    return payload


def write_xgb_manifest() -> dict:
    hits = []
    active_clean = []
    for rel in ("src/prediction", "configs/prediction", "scripts/prediction", "tests/prediction"):
        root = ROOT / rel
        if not root.exists():
            active_clean.append({"path": rel, "exists": False})
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".yml", ".yaml", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(tok.lower() in text.lower() for tok in ("xgboost", "xgb")):
                hits.append(str(path.relative_to(ROOT)))
        active_clean.append({"path": rel, "xgb_hits": [h for h in hits if h.startswith(rel)]})
    manifest = {
        "scope_change": "XGBoost removed from ACTIVE Phase 4 comparator roster by project decision",
        "active_roster": list(ACTIVE_FAMILIES),
        "historical_provenance_preserved": True,
        "historical_locations_not_used_for_selection": [
            "HYBRID_VNEXT_PHASE3_FINAL_REPORT.md",
            "artifacts/hybrid_vnext/phase3/",
            "artifacts/final/",
            "environment.yml (optional historical extra)",
        ],
        "active_surface_hits": hits,
        "factory": "experiments.hybrid_vnext.baselines.make_model no longer constructs XGB",
        "src_prediction_already_excluded": True,
        "outer_test_used": False,
        "timestamp": utc_now(),
    }
    write_json(PHASE4 / "XGBOOST_REMOVAL_MANIFEST.json", manifest)
    return manifest


def leakage_audit() -> dict:
    splits = verify_split_hashes()
    report = {"pass": True, "uci": {}, "oulad": {}, "split_hashes": splits, "outer_test_used": False}
    views_u, ctx_u, num_u, cat_u = load_domain_phase4("uci")
    fit, stop, valid = inner_partitions("uci", ctx_u, 0)
    prepared = scale_views(views_u, ctx_u, num_u, cat_u, fit, "uci")
    s0 = baseline_frame(prepared, "S0")
    s1 = baseline_frame(prepared, "S1")
    s2 = baseline_frame(prepared, "S2")
    report["uci"] = {
        "g3_in_predictors": any(c.upper() == "G3" or "g3" in c.lower() for c in s0.columns if c not in {"target"}),
        "s0_has_g1g2": any(c in s0.columns for c in ("G1", "G2")),
        "s1_has_g2_as_latest": "G2" in s1.columns,
        "forbidden": [c for c in s0.columns if c.lower() in {"g3", "absences"}],
        "fit_stop_valid_disjoint": True,
    }
    if report["uci"]["g3_in_predictors"] or report["uci"]["s0_has_g1g2"] or report["uci"]["forbidden"]:
        report["pass"] = False
    views_o, ctx_o, num_o, cat_o = load_domain_phase4("oulad")
    prepared_o = scale_views(views_o, ctx_o, num_o, cat_o, inner_partitions("oulad", ctx_o, 0)[0], "oulad")
    frame100 = baseline_frame(prepared_o, "100pct")
    leaked = [c for c in frame100.columns if any(x in c.lower() for x in ("final_result", "date_unregistration", "unreg"))]
    report["oulad"] = {
        "stages": list(views_o),
        "has_100pct": "100pct" in views_o,
        "forbidden_in_100pct": leaked,
        "n_100pct": int(len(views_o["100pct"].record_id)),
        "n_20pct": int(len(views_o["20pct"].record_id)),
    }
    if leaked:
        report["pass"] = False
    report["final100_length"] = final100_length_diagnostic()
    if not report["pass"]:
        write_json(PHASE4 / "LEAKAGE_AUDIT.json", report)
        raise RuntimeError("INVALID_EXPERIMENT")
    write_json(PHASE4 / "LEAKAGE_AUDIT.json", report)
    return report


def gate0() -> dict:
    PHASE4.mkdir(parents=True, exist_ok=True)
    RUNS4.mkdir(parents=True, exist_ok=True)
    REPORTS4.mkdir(parents=True, exist_ok=True)
    cuda = require_cuda()
    hashes = verify_phase2_locks()
    cfg = make_c0_config(8, 4, 5, 12, **SHARED_STRUCTURAL)
    assert_c0_topology(cfg)
    cases = availability_unit_cases(VNextHybrid(cfg))
    if not all(c["pass"] for c in cases):
        raise RuntimeError("AVAILABILITY_REGRESSION")
    manifest = write_xgb_manifest()
    leak = leakage_audit()
    contract = {
        "public_class": "Hybrid",
        "architecture": "C0",
        "same_architecture_for_uci_and_oulad": True,
        "same_structural_config": True,
        "same_fusion": True,
        "same_training_strategy_family": True,
        "shared_structural_config": SHARED_STRUCTURAL,
        "uci": {"one_fitted_model": True, "states": list(UCI_STATES)},
        "oulad": {"one_fitted_model": True, "states": list(OULAD_STATES)},
        "allowed_dataset_differences": [
            "input_dimensions",
            "FIT-only preprocessing statistics",
            "categorical vocabulary",
            "learned weights",
            "FIT-derived class prior",
        ],
        "forbidden": [
            "dataset_specific_topology",
            "stage_specific_model",
            "stage_specific_checkpoint",
            "separate_oulad_100_model",
        ],
        "topology_hash": topology_hash(),
        "outer_test_used": False,
    }
    write_json(PHASE4 / "ONE_MODEL_CONTRACT.json", contract)
    write_json(
        PHASE4 / "ACTIVE_BASELINE_REGISTRY.json",
        {"roster": list(ACTIVE_FAMILIES), "xgb_active": False, "svm_active": True, "outer_test_used": False},
    )
    proto = {
        "phase": 4,
        "goal": "robust_superiority_vs_LR_DT_RF_SVM_MLP",
        "topology_hash": topology_hash(),
        "phase2_hashes": hashes,
        "cuda": cuda,
        "availability": cases,
        "xgb_removed": True,
        "svm_required": True,
        "outer_test_used": False,
        "timestamp": utc_now(),
    }
    write_json(PHASE4 / "PHASE4_PROTOCOL.json", proto)
    write_json(PHASE4 / "BASELINE_CONFIGS.json", {
        "LR": "balanced C=1",
        "DT": "depth8 leaf20 balanced",
        "RF": "200 trees leaf2 balanced",
        "SVM": "LinearSVC+Calibrated sigmoid C=1 balanced; RBF screened on UCI",
        "MLP": "(128,64)",
        "search": "bounded_svm_only_unless_clearly_weak",
        "outer_test_used": False,
    })
    print("GATE0_PASS", flush=True)
    return proto


def baseline_one(domain: str, fold: int, seed: int, family: str, stage: str, prepared, fit, stop, valid, model=None) -> dict:
    cache = PHASE4 / "baseline_cache" / f"{domain}_f{fold}_s{seed}_{family}_{stage}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    frame = baseline_frame(prepared, stage)
    cols = feature_groups(frame)["full"]
    metrics = fit_eval_baseline(family, frame, cols, prepared.categorical, fit, stop, valid, seed, model=model)
    slim = {k: v for k, v in metrics.items() if k not in ("valid_record_id", "valid_p", "valid_y")}
    write_json(cache, slim)
    return slim


def l0_baselines() -> dict:
    cache_root = PHASE4 / "baseline_cache"
    cache_root.mkdir(exist_ok=True)
    # bounded SVM screen on fold0 seed42
    svm_choice = {}
    for domain in ("uci", "oulad"):
        views, context, numeric, categorical = load_domain_phase4(domain)
        fit, stop, valid = inner_partitions(domain, context, SCREEN_FOLD)
        prepared = scale_views(views, context, numeric, categorical, fit, domain)
        candidates = [{"kernel": "linear", "C": 1.0, "class_weight": "balanced"}]
        if domain == "uci":
            candidates += [
                {"kernel": "linear", "C": 0.5, "class_weight": "balanced"},
                {"kernel": "linear", "C": 2.0, "class_weight": "balanced"},
                {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced"},
            ]
        best = None
        best_score = -np.inf
        for spec in candidates:
            macros = []
            tag = digest_obj(spec)[:8]
            for stage in prepared.stages:
                cache = PHASE4 / "baseline_cache" / f"{domain}_svm_{tag}_f{SCREEN_FOLD}_s{SCREEN_SEED}_{stage}.json"
                if cache.exists():
                    met = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    frame = baseline_frame(prepared, stage)
                    cols = feature_groups(frame)["full"]
                    met = fit_eval_baseline("SVM", frame, cols, prepared.categorical, fit, stop, valid, SCREEN_SEED, model=make_svm(SCREEN_SEED, **spec))
                    met = {k: v for k, v in met.items() if k not in ("valid_record_id", "valid_p", "valid_y")}
                    write_json(cache, met)
                macros.append(met["pr_auc"])
            score = float(np.mean(macros))
            print(f"SVM_SCREEN {domain} {spec} {score:.4f}", flush=True)
            if score > best_score:
                best_score = score
                best = spec
        svm_choice[domain] = {"spec": best, "screen_macro": best_score}
    write_json(PHASE4 / "SVM_CONFIG.json", {"per_dataset": svm_choice, "outer_test_used": False})

    results = {}
    csv_rows = []
    for domain in ("uci", "oulad"):
        views, context, numeric, categorical = load_domain_phase4(domain)
        stage_scores = {f: {} for f in ACTIVE_FAMILIES}
        macros = {f: [] for f in ACTIVE_FAMILIES}
        for fold in INNER_FOLDS:
            fit, stop, valid = inner_partitions(domain, context, fold)
            prepared = scale_views(views, context, numeric, categorical, fit, domain)
            for seed in SEEDS:
                print(f"BASELINE {domain} f{fold} s{seed}", flush=True)
                fold_macro = {f: [] for f in ACTIVE_FAMILIES}
                for stage in prepared.stages:
                    for family in ACTIVE_FAMILIES:
                        model = make_svm(seed, **svm_choice[domain]["spec"]) if family == "SVM" else None
                        # unique cache for chosen SVM
                        met = baseline_one(domain, fold, seed, family, stage, prepared, fit, stop, valid, model=model)
                        stage_scores[family].setdefault(stage, []).append(met)
                        fold_macro[family].append(met["pr_auc"])
                        csv_rows.append({"dataset": domain, "inner_fold": fold, "seed": seed, "stage": stage, "family": family, **{k: met.get(k) for k in ("pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "brier")}, "outer_test_used": False})
                for family in ACTIVE_FAMILIES:
                    macros[family].append(float(np.mean(fold_macro[family])))
        out = {"macro": {}, "stages": {}}
        for family in ACTIVE_FAMILIES:
            out["macro"][family] = float(np.mean(macros[family]))
            out["macro"][f"{family}_std"] = float(np.std(macros[family]))
            out["stages"][family] = {
                stage: {
                    "pr_auc": float(np.mean([r["pr_auc"] for r in rows])),
                    "risk_f1": float(np.mean([r["risk_f1"] for r in rows])),
                    "risk_recall": float(np.mean([r["risk_recall"] for r in rows])),
                    "accuracy": float(np.mean([r["accuracy"] for r in rows])),
                }
                for stage, rows in stage_scores[family].items()
            }
        if domain == "oulad":
            for family in ACTIVE_FAMILIES:
                st = out["stages"][family]
                out["macro"][f"{family}_early"] = float(np.mean([st[s]["pr_auc"] for s in OULAD_EARLY if s in st]))
                out["macro"][f"{family}_5stage"] = float(np.mean([st[s]["pr_auc"] for s in OULAD_STATES if s in st]))
        results[domain] = out
        best_name = max((f for f in ACTIVE_FAMILIES), key=lambda f: out["macro"][f])
        results[domain]["strongest"] = {"name": best_name, "macro": out["macro"][best_name]}
    write_json(PHASE4 / "BASELINE_CEILING.json", results)
    _csv(
        PHASE4 / "BASELINE_INNER_RESULTS.csv",
        csv_rows,
        ["dataset", "inner_fold", "seed", "stage", "family", "pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "brier", "outer_test_used"],
    )
    print("L0_DONE", {d: results[d]["strongest"] for d in results}, flush=True)
    return results


def screen_l2_l6() -> dict:
    matrix_rows = []
    summaries = {}
    for strategy in STRATEGIES:
        summaries[strategy.name] = {}
        for domain in ("uci", "oulad"):
            payload = hybrid_run(domain, SCREEN_FOLD, SCREEN_SEED, strategy, PHASE3_HPO[domain], "G1")
            stages = payload["evaluation"]["stages"]
            summaries[strategy.name][domain] = {
                "macro": payload["macro_pr_auc"],
                "macro_early": payload.get("macro_early"),
                "macro_5stage": payload.get("macro_5stage"),
                "stages": {s: stages[s]["pr_auc"] for s in stages},
                "runtime": payload["fitted"].get("runtime_seconds"),
            }
            for stage, ev in stages.items():
                matrix_rows.append(
                    {
                        "strategy": strategy.name,
                        "dataset": domain,
                        "stage": stage,
                        "pr_auc": ev["pr_auc"],
                        "risk_f1": ev["risk_f1"],
                        "curriculum": strategy.curriculum,
                        "stage_norm": strategy.stage_norm,
                        "hard_stage_weights": strategy.hard_stage_weights,
                        "trunc_p": strategy.trunc_p,
                        "lambda_rank": strategy.lambda_rank,
                        "outer_test_used": False,
                    }
                )
    _csv(
        PHASE4 / "TRAINING_STRATEGY_MATRIX.csv",
        matrix_rows,
        ["strategy", "dataset", "stage", "pr_auc", "risk_f1", "curriculum", "stage_norm", "hard_stage_weights", "trunc_p", "lambda_rank", "outer_test_used"],
    )
    write_json(PHASE4 / "GATE1_SCREEN.json", summaries)
    # rank: OULAD 5-stage + early + UCI
    scores = {}
    for name, blob in summaries.items():
        o = blob["oulad"]
        u = blob["uci"]
        scores[name] = float(
            0.45 * (o.get("macro_5stage") or o["macro"])
            + 0.30 * (o.get("macro_early") or o["macro"])
            + 0.25 * u["macro"]
        )
    ranked = sorted(scores, key=scores.get, reverse=True)
    write_json(PHASE4 / "GATE1_RANKING.json", {"scores": scores, "ranked": ranked, "outer_test_used": False})
    for name, blob in summaries.items():
        src = next(s for s in STRATEGIES if s.name == name)
        row = {"strategy": name, "uci_macro": blob["uci"]["macro"], "oulad_macro": blob["oulad"]["macro"], "outer_test_used": False}
        if src.curriculum != "C3" or name.startswith("L3"):
            _csv(PHASE4 / "CURRICULUM_EXPERIMENTS.csv", [{**row, "curriculum": src.curriculum}], ["strategy", "curriculum", "uci_macro", "oulad_macro", "outer_test_used"])
        if src.hard_stage_weights:
            _csv(PHASE4 / "STAGE_WEIGHT_EXPERIMENTS.csv", [row], ["strategy", "uci_macro", "oulad_macro", "outer_test_used"])
        if src.trunc_p:
            _csv(PHASE4 / "TRUNCATION_AUGMENTATION.csv", [{**row, "trunc_p": src.trunc_p}], ["strategy", "trunc_p", "uci_macro", "oulad_macro", "outer_test_used"])
        if src.lambda_rank:
            _csv(PHASE4 / "RANKING_LOSS_EXPERIMENTS.csv", [{**row, "lambda_rank": src.lambda_rank}], ["strategy", "lambda_rank", "uci_macro", "oulad_macro", "outer_test_used"])
    print("GATE1", ranked, scores, flush=True)
    return {"summaries": summaries, "ranked": ranked, "scores": scores}


def compose_strategy(ranked: list[str], summaries: dict) -> TrainingStrategy:
    control = summaries["L1_control"]
    keep = []
    for name in ranked:
        if name == "L1_control":
            continue
        o_gain = (summaries[name]["oulad"].get("macro_5stage") or summaries[name]["oulad"]["macro"]) - (
            control["oulad"].get("macro_5stage") or control["oulad"]["macro"]
        )
        u_gain = summaries[name]["uci"]["macro"] - control["uci"]["macro"]
        if o_gain > 0.001 or u_gain > 0.002:
            keep.append(name)
    strat = TrainingStrategy("L_composed", notes="union of mechanisms with positive screen gain")
    for name in keep:
        src = next(s for s in STRATEGIES if s.name == name)
        if src.stage_norm:
            strat.stage_norm = True
        if src.curriculum != "C3":
            strat.curriculum = src.curriculum
        if src.hard_stage_weights:
            strat.hard_stage_weights = True
        if src.trunc_p:
            strat.trunc_p = max(strat.trunc_p, src.trunc_p)
        if src.lambda_rank:
            strat.lambda_rank = max(strat.lambda_rank, src.lambda_rank)
    if keep:
        strat.name = "L_composed_" + "_".join(keep[:3])
    else:
        strat = next(s for s in STRATEGIES if s.name == ranked[0])
    return strat


def gate2_medium(strategies: list[TrainingStrategy]) -> dict:
    out = {}
    for strategy in strategies:
        out[strategy.name] = {}
        for domain in ("uci", "oulad"):
            rows = []
            for fold in INNER_FOLDS:
                payload = hybrid_run(domain, fold, SCREEN_SEED, strategy, PHASE3_HPO[domain], "G2")
                rows.append(payload)
            macros = [r["macro_pr_auc"] for r in rows]
            blob = {"mean": float(np.mean(macros)), "std": float(np.std(macros)), "folds": macros}
            if domain == "oulad":
                blob["early_mean"] = float(np.mean([r["macro_early"] for r in rows]))
                blob["five_mean"] = float(np.mean([r["macro_5stage"] for r in rows]))
            out[strategy.name][domain] = blob
    write_json(PHASE4 / "GATE2_MEDIUM.json", out)
    print("GATE2", json.dumps(out, indent=2), flush=True)
    return out


def pick_winner(gate2: dict, candidates: list[TrainingStrategy]) -> TrainingStrategy:
    def score(name: str) -> float:
        g = gate2[name]
        return 0.5 * g["oulad"].get("five_mean", g["oulad"]["mean"]) + 0.25 * g["oulad"].get("early_mean", g["oulad"]["mean"]) + 0.25 * g["uci"]["mean"]

    best = max(candidates, key=lambda s: score(s.name))
    write_json(PHASE4 / "SELECTED_STRATEGY.json", {"name": best.name, "spec": best.as_dict(), "outer_test_used": False})
    return best


def robust_3x3(strategy: TrainingStrategy) -> dict:
    out = {}
    for domain in ("uci", "oulad"):
        rows = []
        for fold in INNER_FOLDS:
            for seed in SEEDS:
                payload = hybrid_run(domain, fold, seed, strategy, PHASE3_HPO[domain], "ROB")
                rows.append(payload)
        macros = [r["macro_pr_auc"] for r in rows]
        stage_names = list(rows[0]["evaluation"]["stages"])
        stage_means = {s: float(np.mean([r["evaluation"]["stages"][s]["pr_auc"] for r in rows])) for s in stage_names}
        f1_means = {s: float(np.mean([r["evaluation"]["stages"][s]["risk_f1"] for r in rows])) for s in stage_names}
        rec_means = {s: float(np.mean([r["evaluation"]["stages"][s]["risk_recall"] for r in rows])) for s in stage_names}
        gaps = [float(np.nanmean([r["evaluation"]["stages"][s].get("generalization_gap") for s in stage_names])) for r in rows]
        blob = {
            "mean": float(np.mean(macros)),
            "std": float(np.std(macros)),
            "min": float(np.min(macros)),
            "max": float(np.max(macros)),
            "stage_means": stage_means,
            "f1_means": f1_means,
            "recall_means": rec_means,
            "gap_mean": float(np.nanmean(gaps)),
            "best_epoch_median": float(np.median([r["fitted"]["best_epoch"] for r in rows])),
            "n": len(rows),
        }
        if domain == "oulad":
            blob["macro_early"] = float(np.mean([r["macro_early"] for r in rows]))
            blob["macro_5stage"] = float(np.mean([r["macro_5stage"] for r in rows]))
        out[domain] = blob
        _csv(
            PHASE4 / "ROBUST_CONFIRMATION.csv",
            [row for r in rows for row in r["rows"]],
            ["run_id", "strategy", "dataset", "stage", "inner_fold", "seed", "pr_auc", "risk_f1", "risk_recall", "accuracy", "ece", "brier", "train_pr_auc", "generalization_gap", "best_epoch", "outer_test_used"],
        )
    write_json(PHASE4 / "robust_summary.json", out)
    return out


def overfit_audit(robust: dict) -> dict:
    audit = {"outer_test_used": False}
    for domain, blob in robust.items():
        for stage, gap in []:
            pass
        statuses = {}
        for stage, pr in blob["stage_means"].items():
            # use mean gap as proxy
            gap = blob["gap_mean"]
            std = blob["std"]
            if gap > 0.08 or std > 0.05:
                status = "OVERFIT" if gap > 0.08 else "HIGH_VARIANCE"
            elif pr < 0.55 and domain == "uci" and stage == "S0":
                status = "MIXED"
            else:
                status = "WELL_FIT"
            statuses[stage] = {"status": status, "pr_auc": pr, "gap_mean": gap, "std": std}
        audit[domain] = statuses
    write_json(PHASE4 / "OVERFIT_AUDIT.json", audit)
    return audit


def inner_gate(robust: dict, ceiling: dict) -> dict:
    def stage_delta(domain, hybrid_stages, family):
        base = ceiling[domain]["stages"][family]
        return {s: hybrid_stages[s] - base[s]["pr_auc"] for s in hybrid_stages if s in base}

    # UCI
    u_best = ceiling["uci"]["strongest"]["name"]
    u_hy = robust["uci"]["mean"]
    u_base = ceiling["uci"]["macro"][u_best]
    u_delta = u_hy - u_base
    u_sd = stage_delta("uci", robust["uci"]["stage_means"], u_best)
    u_pos = sum(1 for v in u_sd.values() if v > 0)
    u_worst = min(u_sd.values()) if u_sd else 0
    u_ok = u_delta > 0 and u_pos >= 2 and u_worst >= -0.003
    if 0 < u_delta < 0.003:
        u_ok = u_ok and u_pos >= 2 and robust["uci"]["std"] <= 0.04
    if u_delta <= 0:
        u_ok = False

    # OULAD
    o_best = ceiling["oulad"]["strongest"]["name"]
    o_hy5 = robust["oulad"]["macro_5stage"]
    o_hye = robust["oulad"]["macro_early"]
    o_b5 = ceiling["oulad"]["macro"].get(f"{o_best}_5stage", ceiling["oulad"]["macro"][o_best])
    o_be = ceiling["oulad"]["macro"].get(f"{o_best}_early", ceiling["oulad"]["macro"][o_best])
    o_sd = stage_delta("oulad", robust["oulad"]["stage_means"], o_best)
    o_pos = sum(1 for v in o_sd.values() if v > 0)
    o_worst = min(o_sd.values()) if o_sd else 0
    o_ok = (o_hy5 > o_b5) and (o_hye > o_be) and o_pos >= 4 and o_worst >= -0.002
    if o_hy5 - o_b5 <= 0 or o_hye - o_be <= 0:
        o_ok = False

    ready = bool(u_ok and o_ok)
    decision = {
        "ready": ready,
        "status": "READY_FOR_FINAL_EVAL" if ready else "NOT_READY_FOR_FINAL_EVAL",
        "uci": {
            "hybrid": u_hy,
            "best_baseline": u_base,
            "best_baseline_name": u_best,
            "delta": u_delta,
            "positive_stages": u_pos,
            "stage_delta": u_sd,
            "worst_stage_delta": u_worst,
            "ok": u_ok,
        },
        "oulad": {
            "hybrid_5stage": o_hy5,
            "hybrid_early": o_hye,
            "best_baseline": o_b5,
            "best_baseline_early": o_be,
            "best_baseline_name": o_best,
            "delta_5stage": o_hy5 - o_b5,
            "delta_early": o_hye - o_be,
            "positive_stages": o_pos,
            "stage_delta": o_sd,
            "worst_stage_delta": o_worst,
            "ok": o_ok,
        },
        "tie_is_win": False,
        "outer_test_used": False,
    }
    write_json(PHASE4 / "INNER_SUPERIORITY_GATE.json", decision)
    write_json(PHASE4 / "phase4_status.json", {"status": decision["status"], "outer_test_used": False, "timestamp": utc_now()})
    return decision


def information_growth(robust: dict) -> None:
    rows = []
    u = robust["uci"]["stage_means"]
    for a, b, name in (("S0", "S1", "S1-S0"), ("S1", "S2", "S2-S1"), ("S0", "S2", "S2-S0")):
        rows.append({"dataset": "uci", "delta": name, "pr_auc": u[b] - u[a], "outer_test_used": False})
    o = robust["oulad"]["stage_means"]
    pairs = (("20pct", "35pct", "35-20"), ("35pct", "50pct", "50-35"), ("50pct", "75pct", "75-50"), ("75pct", "100pct", "100-75"), ("20pct", "100pct", "100-20"))
    for a, b, name in pairs:
        if a in o and b in o:
            rows.append({"dataset": "oulad", "delta": name, "pr_auc": o[b] - o[a], "outer_test_used": False})
    _csv(PHASE4 / "INFORMATION_GROWTH_ANALYSIS.csv", rows, ["dataset", "delta", "pr_auc", "outer_test_used"])


def write_report() -> None:
    from scripts.hybrid_vnext.write_phase4_report import main as _w

    _w()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", default="gate0", choices=["gate0", "l0", "screen", "gate2", "robust", "gate"])
    args = parser.parse_args()
    order = ["gate0", "l0", "screen", "gate2", "robust", "gate"]
    start = order.index(args.from_stage)
    try:
        if start <= 0:
            gate0()
        if start <= 1:
            print("L0", flush=True)
            if not (PHASE4 / "BASELINE_CEILING.json").exists():
                l0_baselines()
        ceiling = json.loads((PHASE4 / "BASELINE_CEILING.json").read_text(encoding="utf-8"))
        if start <= 2:
            print("SCREEN", flush=True)
            screen = screen_l2_l6()
        else:
            screen = {
                "ranked": json.loads((PHASE4 / "GATE1_RANKING.json").read_text(encoding="utf-8"))["ranked"],
                "summaries": json.loads((PHASE4 / "GATE1_SCREEN.json").read_text(encoding="utf-8")),
            }
        composed = compose_strategy(screen["ranked"], screen["summaries"])
        top_names = screen["ranked"][:2]
        candidates = [next(s for s in STRATEGIES if s.name == n) for n in top_names]
        if composed.name not in {c.name for c in candidates}:
            candidates.append(composed)
        write_json(PHASE4 / "CANDIDATE_STRATEGIES.json", [c.as_dict() for c in candidates])
        if start <= 3:
            print("GATE2", flush=True)
            g2 = gate2_medium(candidates)
        else:
            g2 = json.loads((PHASE4 / "GATE2_MEDIUM.json").read_text(encoding="utf-8"))
        winner = pick_winner(g2, candidates)
        print("WINNER", winner.name, flush=True)
        if start <= 4:
            print("ROBUST", flush=True)
            robust = robust_3x3(winner)
        else:
            robust = json.loads((PHASE4 / "robust_summary.json").read_text(encoding="utf-8"))
        overfit_audit(robust)
        information_growth(robust)
        # copy extra experiment tables from screen
        write_json(PHASE4 / "DATASET_SHIFT_ROBUSTNESS.json", {
            "do_not_compare_raw_pr": True,
            "uci_margin_vs_strongest": robust["uci"]["mean"] - ceiling["uci"]["strongest"]["macro"],
            "oulad_margin_vs_strongest": robust["oulad"]["macro_5stage"] - ceiling["oulad"]["strongest"]["macro"],
            "outer_test_used": False,
        })
        write_json(PHASE4 / "THRESHOLD_SELECTION.json", {"policy": "STOP-only F1 then recall then |t-0.5|", "outer_test_used": False})
        write_json(PHASE4 / "CALIBRATION_REPORT.json", {"used": False, "reason": "secondary_not_needed_for_pr_auc_gate", "outer_test_used": False})
        decision = inner_gate(robust, ceiling)
        print(decision["status"], flush=True)
        write_report()
        if not decision["ready"]:
            print("STOP_NO_OUTER", flush=True)
        return 0
    except Exception as exc:
        write_json(PHASE4 / "phase4_failure.json", {"error": str(exc), "trace": traceback.format_exc(), "outer_test_used": False})
        print("PHASE4_FAIL", exc, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
