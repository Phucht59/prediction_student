"""Run exactly one isolated authority-recipe UCI NONE job."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from adapter_common import ROOT, atomic_json, fold_recipe, historical_imports, partition, sha256, source_partition_for_mat


def _save_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def run(dataset: str, outer_fold: int, seed: int, *, mode: str = "NONE") -> Path:
    historical_imports()
    from src.studies.v5_1.common.uci_training import fit_uci_model_v5_1
    from src.studies.v5_1.common.uci_transfer import combine_subject_inputs, fit_shared_subject_model
    from src.studies.v5_1.uci.runner import _target_shared

    mode_key = mode.lower()
    run_id = f"{dataset.replace('-', '_')}__{mode_key}__fold{outer_fold}__seed{seed}"
    output = ROOT / ("authority_policy_runs" if mode == "AUTHORITY_CLASS_WEIGHT" else "runs") / run_id
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = [output / "checkpoint.pt", output / "predictions.npz", output / "metrics.json"]
        if manifest.get("status") == "COMPLETE" and all(path.is_file() for path in required):
            return output
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(manifest_path, {"status": "RUNNING", "run_id": run_id, "dataset": dataset, "outer_fold": outer_fold, "seed": seed, "imbalance_mode": mode, "started_at": time.time()})
    recipe = fold_recipe(dataset, outer_fold)
    _, data, _, outer_test, train, test, transformer = partition(dataset, outer_fold)
    # NONE is the sole treatment variation; the archived per-fold authority recipe
    # (architecture, loss, optimizer, refit epoch, split, and seed) is untouched.
    if dataset == "student-mat":
        source_train, _, safe = source_partition_for_mat(data, outer_test, transformer, outer_fold)
        fit = fit_shared_subject_model(combine_subject_inputs(train, source_train), _target_shared(test), config={**recipe["config"], "subject_embedding_dim": 4}, seed=seed, fixed_epochs=int(recipe["fixed_epochs"]), device_name="cuda")
        probability, regression = fit.probability, np.full(len(test.target), np.nan)
        transfer = "shared_trunk_subject_specific_heads"
    else:
        strategy = "class_weight" if mode == "AUTHORITY_CLASS_WEIGHT" else "none"
        fit = fit_uci_model_v5_1(train, test, config=recipe["config"], seed=seed, imbalance_strategy=strategy, fixed_epochs=int(recipe["fixed_epochs"]), device_name="cuda")
        probability, regression = fit.probability, fit.regression
        transfer, safe = "standalone", np.array([], dtype=int)
    checkpoint = output / "checkpoint.pt"
    torch.save(fit.state_dict, checkpoint)
    prediction = output / "predictions.npz"
    _save_npz(prediction, record_id=np.asarray(data.record_ids[outer_test], dtype=str), source_row=outer_test, target=test.target, probability=probability, regression=regression)
    metrics = {"accuracy": float(accuracy_score(test.target, probability.argmax(1))), "macro_f1": float(f1_score(test.target, probability.argmax(1), average="macro", zero_division=0)), "parameter_count": int(fit.parameter_count), "selected_epoch": int(fit.selected_epoch), "runtime_seconds": float(fit.runtime_seconds), "checkpoint_sha256": sha256(checkpoint), "prediction_sha256": sha256(prediction)}
    atomic_json(output / "metrics.json", metrics)
    counts = np.bincount(train.target, minlength=3).astype(float)
    effective_weights = (len(train.target) / (3.0 * np.maximum(counts, 1.0))).tolist() if mode == "AUTHORITY_CLASS_WEIGHT" else None
    atomic_json(manifest_path, {"status": "COMPLETE", "run_id": run_id, "dataset": dataset, "outer_fold": outer_fold, "seed": seed, "imbalance_mode": mode, "historical_selected_imbalance": recipe["imbalance"], "effective_class_weights": effective_weights, "class_order": ["low", "medium", "high"], "transfer": transfer, "source_overlap_safe_records": int(len(safe)), "recipe": recipe, "files": {"checkpoint": metrics["checkpoint_sha256"], "predictions": metrics["prediction_sha256"]}, "completed_at": time.time()})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("student-mat", "student-por"))
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("NONE", "AUTHORITY_CLASS_WEIGHT"), default="NONE")
    args = parser.parse_args()
    run(args.dataset, args.outer_fold, args.seed, mode=args.mode)
