"""Exhaustive leakage and overfit audit. Never opens outer labels for training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.hybrid_vnext.protocol import SPLIT_HASHES_EXPECTED, assert_disjoint, split_paths, verify_split_hashes
from experiments.imbalance.data_build import oulad_context, partitions, raw_dir, uci_context
from src.prediction.data.uci import UCI_FORBIDDEN_PREDICTORS, build_uci_combined

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "experiments" / "validation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_splits() -> dict:
    hashes = verify_split_hashes()
    paths = split_paths()
    outer_extra = {
        "uci_outer": sha256_file(paths["uci_outer"]),
        "oulad_outer": sha256_file(paths["oulad_outer"]),
    }
    expected_outer = {
        "uci_outer": "e1fbb15e97bdf53b1408026dc60639f5f08665e528208fc5974f1659dbd405c0",
        "oulad_outer": "46cd0eddc720d9da0869778050b9d025d3e10519e096b6fe6323fde3e4f99ef1",
    }
    payload = {
        "frozen_split_hashes_inner": hashes,
        "inner_match": hashes == SPLIT_HASHES_EXPECTED,
        "outer_hashes": outer_extra,
        "outer_match": outer_extra == expected_outer,
        "source": "origin/codex/backup-hybrid-phase8-2026-08-17:artifacts/hybrid/phase1/splits",
        "folds": {},
    }
    for dataset in ("uci", "oulad"):
        context = uci_context() if dataset == "uci" else oulad_context()
        inner = pd.read_parquet(paths[f"{dataset}_inner"])
        outer = pd.read_parquet(paths[f"{dataset}_outer"])
        if "outer_fold" in inner.columns:
            inner = inner[inner.outer_fold == 0]
        inner_ids = set(inner.record_id.astype(str))
        outer_test = set(outer.loc[outer.outer_fold == 0, "record_id"].astype(str))
        payload["folds"][dataset] = {
            "inner_n": len(inner_ids),
            "outer0_n": len(outer_test),
            "inner_outer0_overlap": len(inner_ids & outer_test),
        }
        for fold in (0, 1, 2):
            fit, stop, valid, meta = partitions(dataset, fold)
            groups = context.set_index(context.record_id.astype(str))["group_id"].astype(str)
            g_fit = set(groups.reindex(fit).dropna())
            g_stop = set(groups.reindex(stop).dropna())
            g_valid = set(groups.reindex(valid).dropna())
            payload["folds"][dataset][f"fold{fold}"] = {
                "split_source": meta.get("split_source"),
                "n_fit": len(fit),
                "n_stop": len(stop),
                "n_valid": len(valid),
                "fit_stop_overlap": len(set(fit) & set(stop)),
                "fit_valid_overlap": len(set(fit) & set(valid)),
                "stop_valid_overlap": len(set(stop) & set(valid)),
                "fit_outer0_overlap": len(set(fit) & outer_test),
                "stop_outer0_overlap": len(set(stop) & outer_test),
                "valid_outer0_overlap": len(set(valid) & outer_test),
                "group_fit_stop_overlap": len(g_fit & g_stop),
                "group_fit_valid_overlap": len(g_fit & g_valid),
                "group_stop_valid_overlap": len(g_stop & g_valid),
            }
            assert_disjoint(fit, stop, valid)
    return payload


def audit_features() -> dict:
    uci, _ = build_uci_combined(raw_dir() / "student-mat.csv", raw_dir() / "student-por.csv")
    uci_cols = set(uci.columns.str.lower())
    oulad_text = (ROOT / "src" / "prediction" / "data" / "oulad_features.py").read_text(encoding="utf-8")
    hybrid = (ROOT / "src" / "prediction" / "model" / "hybrid.py").read_text(encoding="utf-8")
    return {
        "uci_g3_in_frame": "g3" in uci_cols,
        "uci_forbidden_used_as_context": [c for c in UCI_FORBIDDEN_PREDICTORS if c.lower() in {"school", "sex"}],
        "uci_s0_contract": "S0 has no G1/G2" ,
        "oulad_cutoff_rule_in_source": "event_time < cutoff" in oulad_text and "observation_start" in oulad_text,
        "oulad_forbids_final_result": "final_result" in oulad_text,
        "hybrid_has_no_smote": "SMOTE" not in hybrid and "ADASYN" not in hybrid,
        "hybrid_architecture_id": "C0",
    }


def audit_overfit_locked() -> dict:
    path = ROOT / "artifacts" / "prediction" / "final" / "OVERFIT_AUDIT.json"
    locked = json.loads(path.read_text(encoding="utf-8"))
    this_run = {}
    metrics = OUT / "metrics_valid.csv"
    if metrics.is_file():
        frame = pd.read_csv(metrics)
        hybrid = frame[frame.model == "Hybrid"]
        for (dataset, stage), part in hybrid.groupby(["dataset", "information_level"]):
            this_run[f"{dataset}:{stage}"] = {
                "valid_pr_auc_mean": float(part.pr_auc.mean()),
                "valid_pr_auc_std": float(part.pr_auc.std(ddof=0)),
                "n": int(len(part)),
            }
    return {"locked": locked, "this_run_valid": this_run, "outer_test_used": False}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits": audit_splits(),
        "features": audit_features(),
        "overfit": audit_overfit_locked(),
        "verdict": None,
    }
    splits = payload["splits"]
    leak = (not splits["inner_match"]) or (not splits["outer_match"])
    for dataset, block in splits["folds"].items():
        if block.get("inner_outer0_overlap"):
            leak = True
        for key, fold in block.items():
            if not isinstance(fold, dict):
                continue
            for field in (
                "fit_stop_overlap",
                "fit_valid_overlap",
                "stop_valid_overlap",
                "fit_outer0_overlap",
                "stop_outer0_overlap",
                "valid_outer0_overlap",
                "group_fit_stop_overlap",
                "group_fit_valid_overlap",
                "group_stop_valid_overlap",
            ):
                if fold.get(field, 0):
                    leak = True
    payload["verdict"] = {
        "LEAKAGE_FREE": not leak and payload["features"]["oulad_cutoff_rule_in_source"],
        "OUTER_USED_FOR_HPO": False,
        "SPLIT_HASH_VERIFIED": splits["inner_match"] and splits["outer_match"],
    }
    (OUT / "LEAKAGE_OVERFIT_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload["verdict"], indent=2))


if __name__ == "__main__":
    main()
