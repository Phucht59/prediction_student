"""Phase 2 orchestrator: inner-only design screen. Does not touch prediction authority."""
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

from experiments.hybrid_vnext.baselines import fit_eval_baseline
from experiments.hybrid_vnext.data import (
    baseline_frame,
    feature_groups,
    final100_length_diagnostic,
    inner_partitions,
    load_domain,
    scale_views,
)
from experiments.hybrid_vnext.metrics import binary_metrics
from experiments.hybrid_vnext.model import VNextHybrid, availability_unit_cases, make_config
from experiments.hybrid_vnext.protocol import (
    ART,
    FOLDS,
    OULAD_PRIMARY,
    REPORTS,
    RUNS,
    SCREEN_FOLD,
    SCREEN_SEED,
    SEEDS,
    UCI_STAGES,
    git_branch,
    git_commit,
    require_cuda,
    run_metadata,
    utc_now,
    verify_split_hashes,
    write_json,
)
from experiments.hybrid_vnext.train import Trainer, _ids_for_stage


MATRIX = ART / "EXPERIMENT_MATRIX.csv"
MATRIX_FIELDS = [
    "run_id", "timestamp", "gate", "dataset", "stage", "architecture_id", "family", "feature_set",
    "inner_fold", "seed", "lr", "branch_mode", "temporal_mode", "pr_auc", "risk_f1", "risk_recall",
    "risk_precision", "balanced_accuracy", "ece", "stop_pr_auc", "train_pr_auc", "generalization_gap",
    "best_epoch", "parameter_count", "device", "amp", "batch_size", "outer_test_used", "status",
]


def append_rows(rows: list[dict]) -> None:
    ART.mkdir(parents=True, exist_ok=True)
    exists = MATRIX.exists()
    with MATRIX.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATRIX_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in MATRIX_FIELDS})


def save_run(run_id: str, payload: dict) -> None:
    write_json(RUNS / f"{run_id}.json", payload)


def gate0() -> dict:
    cuda = require_cuda()
    hashes = verify_split_hashes()
    cases = {}
    for architecture_id in ("C0", "C1", "C2", "C3"):
        cfg = make_config(architecture_id, static_dim=8, temporal_dim=4, aggregate_dim=5, summary_dim=12)
        model = VNextHybrid(cfg)
        results = availability_unit_cases(model)
        cases[architecture_id] = {
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "temporal_path": cfg.temporal_path,
            "fusion": cfg.fusion,
            "results": results,
            "pass": all(item["pass"] for item in results),
        }
    failed = [key for key, value in cases.items() if not value["pass"]]
    if failed:
        raise RuntimeError(f"AVAILABILITY_UNIT_FAIL:{failed}:{cases}")
    payload = run_metadata(
        gate=0,
        cuda=cuda,
        split_hashes=hashes,
        availability=cases,
        status="PASS",
    )
    write_json(ART / "CUDA_EXECUTION_AUDIT.json", {"gate0": payload, "jobs": []})
    write_json(ART / "gate0.json", payload)
    return payload


def _update_cuda_job(job: dict) -> None:
    path = ART / "CUDA_EXECUTION_AUDIT.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"jobs": []}
    current.setdefault("jobs", []).append(job)
    write_json(path, current)


def prepare(domain: str, inner_fold: int) -> tuple:
    views, context, numeric, categorical = load_domain(domain)
    fit, stop, valid = inner_partitions(domain, context, inner_fold)
    prepared = scale_views(views, context, numeric, categorical, fit, domain)
    return prepared, fit, stop, valid


def hybrid_run(*, domain, architecture_id, fold, seed, lr=2e-4, max_epochs=24, branch_mode="full", temporal_eval="identity", gate="G2") -> dict:
    run_id = f"{gate}_{domain}_{architecture_id}_{branch_mode}_f{fold}_s{seed}_{temporal_eval}_{lr}"
    existing = RUNS / f"{run_id}.json"
    if existing.exists():
        return json.loads(existing.read_text(encoding="utf-8"))
    prepared, fit, stop, valid = prepare(domain, fold)
    cfg = make_config(architecture_id, prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim, prepared.summary_dim, branch_mode=branch_mode)
    trainer = Trainer(prepared, cfg, lr=lr, max_epochs=max_epochs, seed=seed, amp=True)
    try:
        fitted = trainer.fit(fit, stop)
        model = fitted["model"]
        evaluation = trainer.evaluate(model, stop, valid, temporal_mode=temporal_eval)
    except (RuntimeError, ValueError) as exc:
        if "NONFINITE" not in str(exc) and "nonfinite" not in str(exc).lower() and "single-class" not in str(exc):
            raise
        torch.cuda.empty_cache()
        trainer = Trainer(prepared, cfg, lr=lr, max_epochs=max_epochs, seed=seed, amp=False)
        fitted = trainer.fit(fit, stop)
        model = fitted["model"]
        evaluation = trainer.evaluate(model, stop, valid, temporal_mode=temporal_eval)
        fitted["amp_fallback"] = True
    rows = []
    for stage, metrics in evaluation["stages"].items():
        train_ids = _ids_for_stage(prepared.views[stage], fit)
        lookup = {str(r): i for i, r in enumerate(prepared.views[stage].record_id)}
        train_y = prepared.views[stage].target[[lookup[i] for i in train_ids]]
        train_p = trainer._predict(model, stage, train_ids, temporal_eval)
        train_pr = binary_metrics(train_y, train_p)["pr_auc"] if len(np.unique(train_y)) == 2 else float("nan")
        row = {
            "run_id": f"{gate}_{domain}_{architecture_id}_{branch_mode}_f{fold}_s{seed}_{temporal_eval}_{lr}",
            "timestamp": utc_now(),
            "gate": gate,
            "dataset": domain,
            "stage": stage,
            "architecture_id": architecture_id,
            "family": "Hybrid",
            "feature_set": "parity" if cfg.tabular_mode == "parity" else "phase8",
            "inner_fold": fold,
            "seed": seed,
            "lr": lr,
            "branch_mode": branch_mode,
            "temporal_mode": temporal_eval,
            "pr_auc": metrics["pr_auc"],
            "risk_f1": metrics["risk_f1"],
            "risk_recall": metrics["risk_recall"],
            "risk_precision": metrics["risk_precision"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "ece": metrics["ece"],
            "stop_pr_auc": metrics["stop_pr_auc"],
            "train_pr_auc": train_pr,
            "generalization_gap": float(train_pr - metrics["pr_auc"]) if np.isfinite(train_pr) else None,
            "best_epoch": fitted["best_epoch"],
            "parameter_count": fitted["parameter_count"],
            "device": "cuda:0",
            "amp": True,
            "batch_size": fitted["batch_size"],
            "outer_test_used": False,
            "status": "ok",
        }
        rows.append(row)
    append_rows(rows)
    ckpt = ART / "checkpoints" / f"{rows[0]['run_id']}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": cfg.__dict__}, ckpt)
    payload = run_metadata(
        run_id=rows[0]["run_id"],
        domain=domain,
        architecture_id=architecture_id,
        fold=fold,
        seed=seed,
        lr=lr,
        branch_mode=branch_mode,
        temporal_eval=temporal_eval,
        feature_contract=prepared.feature_contract,
        checkpoint=str(ckpt),
        fitted={k: fitted[k] for k in ("best_epoch", "best_stop_macro_pr_auc", "parameter_count", "peak_vram_bytes", "runtime_seconds", "batch_size", "device", "amp", "gpu_name", "history")},
        evaluation=evaluation,
        rows=rows,
    )
    save_run(rows[0]["run_id"], payload)
    _update_cuda_job(
        {
            "run_id": rows[0]["run_id"],
            "device": "cuda:0",
            "gpu_name": fitted["gpu_name"],
            "amp": True,
            "peak_vram_bytes": fitted["peak_vram_bytes"],
            "batch_size": fitted["batch_size"],
            "parameter_count": fitted["parameter_count"],
            "outer_test_used": False,
        }
    )
    del model
    torch.cuda.empty_cache()
    return payload


def gate1() -> dict:
    write_json(ART / "FEATURE_PARITY_MANIFEST.json", {})
    result = {"baselines": {}, "hybrid": {}, "temporal_order": {}, "final100": None}
    for domain in ("uci", "oulad"):
        prepared, fit, stop, valid = prepare(domain, SCREEN_FOLD)
        write_json(ART / f"feature_contract_{domain}.json", prepared.feature_contract)
        result.setdefault("contracts", {})[domain] = prepared.feature_contract
        domain_base = {}
        models = ["XGB", "RF", "LR", "DT", "MLP"] + (["SVM"] if domain == "uci" else [])
        for stage in prepared.stages:
            frame = baseline_frame(prepared, stage)
            groups = feature_groups(frame)
            domain_base[stage] = {}
            for family in models:
                sets = groups if family == "XGB" else {"full": groups["full"]}
                for set_name, columns in sets.items():
                    if family != "XGB" and set_name != "full":
                        continue
                    if family == "XGB" and set_name not in {"static", "static_aggregate", "full"}:
                        continue
                    try:
                        metrics = fit_eval_baseline(family, frame, columns, prepared.categorical, fit, stop, valid, SCREEN_SEED)
                        status = "ok"
                    except Exception as exc:
                        metrics = {"error": str(exc)}
                        status = "fail"
                    row = {
                        "run_id": f"G1_{domain}_{family}_{set_name}_{stage}_f{SCREEN_FOLD}_s{SCREEN_SEED}",
                        "timestamp": utc_now(),
                        "gate": "G1",
                        "dataset": domain,
                        "stage": stage,
                        "architecture_id": family,
                        "family": family,
                        "feature_set": set_name,
                        "inner_fold": SCREEN_FOLD,
                        "seed": SCREEN_SEED,
                        "lr": None,
                        "branch_mode": None,
                        "temporal_mode": "identity",
                        "pr_auc": metrics.get("pr_auc"),
                        "risk_f1": metrics.get("risk_f1"),
                        "risk_recall": metrics.get("risk_recall"),
                        "risk_precision": metrics.get("risk_precision"),
                        "balanced_accuracy": metrics.get("balanced_accuracy"),
                        "ece": metrics.get("ece"),
                        "stop_pr_auc": metrics.get("stop_pr_auc"),
                        "train_pr_auc": metrics.get("train_pr_auc"),
                        "generalization_gap": metrics.get("generalization_gap"),
                        "best_epoch": None,
                        "parameter_count": None,
                        "device": "cpu",
                        "amp": False,
                        "batch_size": None,
                        "outer_test_used": False,
                        "status": status,
                    }
                    append_rows([row])
                    domain_base[stage][f"{family}:{set_name}"] = metrics
        result["baselines"][domain] = domain_base
        result["hybrid"][domain] = {}
        result["hybrid"][domain]["tabular"] = hybrid_run(domain=domain, architecture_id="C3", fold=SCREEN_FOLD, seed=SCREEN_SEED, branch_mode="tabular", gate="G1")
        c0 = hybrid_run(domain=domain, architecture_id="C0", fold=SCREEN_FOLD, seed=SCREEN_SEED, gate="G1")
        result["hybrid"][domain]["full_c0"] = c0
        prepared, fit, stop, valid = prepare(domain, SCREEN_FOLD)
        cfg = make_config("C0", prepared.static_dim, prepared.temporal_dim, prepared.aggregate_dim, prepared.summary_dim)
        trainer = Trainer(prepared, cfg, seed=SCREEN_SEED)
        model = VNextHybrid(cfg).to(torch.device("cuda"))
        blob = torch.load(c0["checkpoint"], map_location="cuda")
        model.load_state_dict(blob["state_dict"], strict=True)
        order = {
            "identity": {
                "macro": c0["evaluation"]["macro_pr_auc"],
                "stages": {key: value["pr_auc"] for key, value in c0["evaluation"]["stages"].items()},
            }
        }
        for mode in ("reverse", "shuffle"):
            evaluation = trainer.evaluate(model, stop, valid, temporal_mode=mode)
            order[mode] = {"macro": evaluation["macro_pr_auc"], "stages": {key: value["pr_auc"] for key, value in evaluation["stages"].items()}}
            append_rows(
                [
                    {
                        "run_id": f"G1_order_{domain}_C0_{mode}_f{SCREEN_FOLD}_s{SCREEN_SEED}",
                        "timestamp": utc_now(),
                        "gate": "G1_order",
                        "dataset": domain,
                        "stage": stage,
                        "architecture_id": "C0",
                        "family": "Hybrid",
                        "feature_set": "phase8",
                        "inner_fold": SCREEN_FOLD,
                        "seed": SCREEN_SEED,
                        "lr": 2e-4,
                        "branch_mode": "full",
                        "temporal_mode": mode,
                        "pr_auc": metrics["pr_auc"],
                        "risk_f1": metrics["risk_f1"],
                        "risk_recall": metrics["risk_recall"],
                        "risk_precision": metrics["risk_precision"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "ece": metrics["ece"],
                        "stop_pr_auc": metrics["stop_pr_auc"],
                        "train_pr_auc": None,
                        "generalization_gap": None,
                        "best_epoch": c0["fitted"]["best_epoch"],
                        "parameter_count": c0["fitted"]["parameter_count"],
                        "device": "cuda:0",
                        "amp": True,
                        "batch_size": c0["fitted"]["batch_size"],
                        "outer_test_used": False,
                        "status": "ok",
                    }
                    for stage, metrics in evaluation["stages"].items()
                ]
            )
        result["temporal_order"][domain] = order
        del model
        torch.cuda.empty_cache()
    result["final100"] = final100_length_diagnostic()
    write_json(ART / "FEATURE_PARITY_MANIFEST.json", result["contracts"] if "contracts" in result else {})
    write_json(ART / "gate1.json", result)
    return result


def summarize_payload(payload: dict) -> dict:
    return {
        "macro_pr_auc": payload["evaluation"]["macro_pr_auc"],
        "worst_pr_auc": payload["evaluation"]["worst_pr_auc"],
        "best_epoch": payload["fitted"]["best_epoch"],
        "parameter_count": payload["fitted"]["parameter_count"],
        "stages": {k: v["pr_auc"] for k, v in payload["evaluation"]["stages"].items()},
        "diagnostics": payload["evaluation"]["diagnostics"],
        "gap_mean": float(np.nanmean([row["generalization_gap"] for row in payload["rows"]])),
    }


def gate2() -> dict:
    screen = {}
    for domain in ("uci", "oulad"):
        screen[domain] = {}
        for architecture_id in ("C0", "C1", "C2", "C3"):
            if architecture_id == "C0":
                existing = json.loads((RUNS / f"G1_{domain}_C0_full_f{SCREEN_FOLD}_s{SCREEN_SEED}_identity_0.0002.json").read_text(encoding="utf-8"))
                screen[domain][architecture_id] = summarize_payload(existing)
                continue
            payload = hybrid_run(domain=domain, architecture_id=architecture_id, fold=SCREEN_FOLD, seed=SCREEN_SEED, gate="G2")
            screen[domain][architecture_id] = summarize_payload(payload)
    write_json(ART / "gate2.json", screen)
    return screen


def decide_survivors(screen: dict) -> list[str]:
    scores = {}
    for architecture_id in ("C0", "C1", "C2", "C3"):
        oulad = screen["oulad"][architecture_id]["macro_pr_auc"]
        uci = screen["uci"][architecture_id]["macro_pr_auc"]
        scores[architecture_id] = {"oulad": oulad, "uci": uci, "sum": oulad + uci}
    ranked = sorted(scores, key=lambda k: (scores[k]["oulad"], scores[k]["uci"]), reverse=True)
    survivors = [ranked[0]]
    if len(ranked) > 1 and scores[ranked[1]]["oulad"] >= scores[ranked[0]]["oulad"] - 0.008:
        survivors.append(ranked[1])
    write_json(ART / "survivors.json", {"scores": scores, "ranked": ranked, "survivors": survivors})
    return survivors


def gate3(survivors: list[str]) -> dict:
    robust = {}
    for architecture_id in survivors:
        robust[architecture_id] = {}
        for domain in ("uci", "oulad"):
            robust[architecture_id][domain] = []
            for fold in FOLDS:
                for seed in SEEDS:
                    payload = hybrid_run(domain=domain, architecture_id=architecture_id, fold=fold, seed=seed, gate="G3")
                    robust[architecture_id][domain].append(summarize_payload(payload) | {"fold": fold, "seed": seed})
    write_json(ART / "gate3.json", robust)
    return robust


def gate4(winner: str) -> dict:
    grid = {}
    for lr in (8e-4, 2e-4, 6e-5):
        grid[str(lr)] = {}
        for domain in ("uci", "oulad"):
            payload = hybrid_run(domain=domain, architecture_id=winner, fold=SCREEN_FOLD, seed=SCREEN_SEED, lr=lr, max_epochs=40, gate="G4")
            grid[str(lr)][domain] = summarize_payload(payload)
    write_json(ART / "gate4.json", grid)
    return grid


def select_topology(screen: dict, robust: dict, survivors: list[str]) -> dict:
    def pack(architecture_id: str) -> dict:
        oulad_macros = [item["macro_pr_auc"] for item in robust[architecture_id]["oulad"]]
        uci_macros = [item["macro_pr_auc"] for item in robust[architecture_id]["uci"]]
        oulad_worst = [item["worst_pr_auc"] for item in robust[architecture_id]["oulad"]]
        uci_gaps = [item["gap_mean"] for item in robust[architecture_id]["uci"]]
        return {
            "architecture_id": architecture_id,
            "oulad_mean": float(np.mean(oulad_macros)),
            "oulad_std": float(np.std(oulad_macros)),
            "oulad_worst_mean": float(np.mean(oulad_worst)),
            "uci_mean": float(np.mean(uci_macros)),
            "uci_std": float(np.std(uci_macros)),
            "uci_gap": float(np.mean(uci_gaps)),
            "params": robust[architecture_id]["oulad"][0]["parameter_count"],
        }

    packed = [pack(item) for item in survivors]
    packed.sort(key=lambda row: (row["oulad_mean"], row["oulad_worst_mean"], -row["uci_std"], row["uci_mean"]), reverse=True)
    winner = packed[0]
    baseline_c0 = screen["oulad"]["C0"]["macro_pr_auc"]
    tabular_ok = True
    reason = []
    if winner["oulad_mean"] + 1e-6 < baseline_c0 - 0.01:
        tabular_ok = False
        reason.append("oulad_below_c0")
    if winner["uci_mean"] + 1e-6 < screen["uci"]["C0"]["macro_pr_auc"] - 0.02:
        tabular_ok = False
        reason.append("uci_material_regression")
    decision = {
        "winner": winner,
        "all": packed,
        "eligible": tabular_ok,
        "reasons": reason,
        "outer_test_used": False,
    }
    write_json(ART / "selection.json", decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-gate", type=int, default=0)
    parser.add_argument("--only-gate", type=int, default=None)
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    try:
        if args.only_gate == 0 or (args.only_gate is None and args.from_gate <= 0):
            print("GATE0", flush=True)
            gate0()
            if args.only_gate == 0:
                return 0
        if args.only_gate == 1 or (args.only_gate is None and args.from_gate <= 1):
            print("GATE1", flush=True)
            gate1()
            if args.only_gate == 1:
                return 0
        print("GATE2", flush=True)
        screen = gate2()
        survivors = decide_survivors(screen)
        print("SURVIVORS", survivors, flush=True)
        if args.only_gate == 2:
            return 0
        print("GATE3", flush=True)
        robust = gate3(survivors)
        decision = select_topology(screen, robust, survivors)
        winner = decision["winner"]["architecture_id"]
        if args.only_gate == 3:
            return 0
        print("GATE4", winner, flush=True)
        gate4(winner)
        write_json(ART / "phase2_status.json", {"status": "COMPLETE", "winner": winner, "eligible": decision["eligible"], "outer_test_used": False})
        print("PHASE2_RUNS_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        write_json(ART / "phase2_failure.json", {"error": str(exc), "trace": traceback.format_exc(), "outer_test_used": False})
        print("PHASE2_FAIL", exc, flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
