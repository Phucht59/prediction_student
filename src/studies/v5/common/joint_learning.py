from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .artifacts import atomic_write_json, build_checksum_manifest, safe_v5_root, verify_checksum_manifest
from .metrics import multiclass_metrics
from .protocol import ROOT, load_json_yaml, load_study_protocol, sha256_file
from .uci_data import UCIData, context_preprocessor, load_uci
from .uci_training import UCIInputs, fit_uci_model


def _inputs(data, indices, transformer, mean, std, subject: str) -> UCIInputs:
    context = transformer.transform(data.context.iloc[indices]).astype(np.float32)
    subject_flag = np.zeros((len(indices), 2), dtype=np.float32)
    subject_flag[:, 0 if subject == "mat" else 1] = 1.0
    return UCIInputs(
        ((data.sequence[indices] - mean) / std).astype(np.float32),
        np.concatenate([context, subject_flag], axis=1),
        data.target[indices],
        data.raw_g3[indices],
    )


def _prepare_standalone(mat: UCIData, train, validation):
    transformer = context_preprocessor().fit(mat.context.iloc[train])
    mean = mat.sequence[train].mean(axis=(0, 1), keepdims=True)
    std = mat.sequence[train].std(axis=(0, 1), keepdims=True).clip(1e-6)
    return (
        _inputs(mat, train, transformer, mean, std, "mat"),
        _inputs(mat, validation, transformer, mean, std, "mat"),
    )


def _prepare_joint(mat: UCIData, por: UCIData, mat_train, mat_validation):
    validation_groups = set(mat.quasi_groups[mat_validation])
    por_indices = np.flatnonzero(~np.isin(por.quasi_groups, list(validation_groups)))
    transformer = context_preprocessor().fit(
        pd.concat([mat.context.iloc[mat_train], por.context.iloc[por_indices]], ignore_index=True)
    )
    combined_sequence = np.concatenate([mat.sequence[mat_train], por.sequence[por_indices]])
    mean = combined_sequence.mean(axis=(0, 1), keepdims=True)
    std = combined_sequence.std(axis=(0, 1), keepdims=True).clip(1e-6)
    mat_inputs = _inputs(mat, mat_train, transformer, mean, std, "mat")
    por_inputs = _inputs(por, por_indices, transformer, mean, std, "por")
    pretraining = UCIInputs(
        np.concatenate([mat_inputs.sequence, por_inputs.sequence]),
        np.concatenate([mat_inputs.context, por_inputs.context]),
        np.concatenate([mat_inputs.target, por_inputs.target]),
        np.concatenate([mat_inputs.raw_g3, por_inputs.raw_g3]),
    )
    return pretraining, mat_inputs, _inputs(mat, mat_validation, transformer, mean, std, "mat"), por_indices


def _score(inputs: UCIInputs, probability: np.ndarray) -> float:
    return float(multiclass_metrics(inputs.target, probability)["macro_f1"])


def run_joint_uci(*, device: str = "cuda", force: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = ROOT / "configs" / "joint_uci_v5.yaml"
    protocol = load_json_yaml(protocol_path)
    mat_protocol = load_study_protocol("student-mat")
    por_protocol = load_study_protocol("student-por")
    mat = load_uci(ROOT / mat_protocol["source"]["path"], "student-mat")
    por = load_uci(ROOT / por_protocol["source"]["path"], "student-por")
    artifact = safe_v5_root(ROOT / "artifacts" / "v5" / "joint_uci")
    fingerprint = sha256_file(protocol_path)
    state_path = artifact / "run_state.json"
    checksum_path = artifact / "artifact_checksums.json"
    if not force and state_path.is_file() and checksum_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
        if state.get("status") == "COMPLETE" and state.get("fingerprint") == fingerprint and verify_checksum_manifest(artifact, manifest):
            return {"study": "joint-uci", "status": "SKIPPED_VALID_CACHE", "artifact": str(artifact)}
    atomic_write_json(state_path, {"study": "joint-uci", "status": "RUNNING", "fingerprint": fingerprint})
    selected = json.loads((ROOT / "artifacts/v5/student_mat/selected_configs.json").read_text(encoding="utf-8"))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(mat_protocol["splits"]["fixed_split_seed"]))
    rows, leakage_rows = [], []
    for outer_fold, (outer_train, _) in enumerate(outer.split(np.zeros(len(mat.target)), mat.target)):
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(mat_protocol["splits"]["fixed_split_seed"]) + outer_fold)
        config = dict(selected[outer_fold]["cnn_bilstm"])
        fixed_epochs = int(selected[outer_fold]["fixed_epochs"])
        imbalance = str(selected[outer_fold]["imbalance"])
        splits = inner.split(outer_train, mat.target[outer_train], groups=mat.quasi_groups[outer_train])
        for inner_fold, (relative_train, relative_validation) in enumerate(splits):
            train_index, validation_index = outer_train[relative_train], outer_train[relative_validation]
            standalone_train, standalone_validation = _prepare_standalone(mat, train_index, validation_index)
            pretraining, finetuning, joint_validation, por_indices = _prepare_joint(mat, por, train_index, validation_index)
            validation_groups = set(mat.quasi_groups[validation_index])
            training_groups = set(mat.quasi_groups[train_index]) | set(por.quasi_groups[por_indices])
            overlap = training_groups & validation_groups
            leakage_rows.append({
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "mat_train_records": int(len(train_index)),
                "mat_validation_records": int(len(validation_index)),
                "por_pretraining_records": int(len(por_indices)),
                "excluded_por_overlap_records": int(len(por.target) - len(por_indices)),
                "overlapping_validation_groups": len(overlap),
            })
            if overlap:
                raise RuntimeError("Joint-learning group leakage detected")
            for seed in protocol["comparison"]["seeds"]:
                standalone = fit_uci_model(
                    standalone_train, standalone_validation, config=config, seed=int(seed),
                    imbalance_strategy=imbalance, fixed_epochs=fixed_epochs, device_name=device,
                )
                pretrained = fit_uci_model(
                    pretraining, finetuning, config=config, seed=int(seed), imbalance_strategy="none",
                    fixed_epochs=fixed_epochs, device_name=device,
                )
                finetune_config = dict(config)
                finetune_config["learning_rate"] = float(config["learning_rate"]) * float(protocol["comparison"]["finetuning_learning_rate_fraction"])
                finetune_epochs = max(3, int(round(fixed_epochs * float(protocol["comparison"]["finetuning_epochs_fraction"]))))
                joint = fit_uci_model(
                    finetuning, joint_validation, config=finetune_config, seed=int(seed),
                    imbalance_strategy=imbalance, fixed_epochs=finetune_epochs, device_name=device,
                    initial_state=pretrained.state_dict,
                )
                standalone_score, joint_score = _score(standalone_validation, standalone.probability), _score(joint_validation, joint.probability)
                rows.append({
                    "outer_fold": outer_fold, "inner_fold": inner_fold, "seed": int(seed),
                    "standalone_macro_f1": standalone_score, "joint_macro_f1": joint_score,
                    "delta_joint_minus_standalone": joint_score - standalone_score,
                    "selected_imbalance": imbalance, "pretraining_records": int(len(pretraining.target)),
                    "finetuning_records": int(len(finetuning.target)), "validation_records": int(len(joint_validation.target)),
                    "standalone_runtime_seconds": standalone.runtime_seconds,
                    "joint_runtime_seconds": pretrained.runtime_seconds + joint.runtime_seconds,
                    "standalone_checkpoint_sha256": standalone.checkpoint_sha256,
                    "pretrained_checkpoint_sha256": pretrained.checkpoint_sha256,
                    "joint_checkpoint_sha256": joint.checkpoint_sha256,
                })
        atomic_write_json(state_path, {"study": "joint-uci", "status": "RUNNING", "fingerprint": fingerprint, "completed_outer_folds": outer_fold + 1})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(artifact / "joint_learning_comparison.csv", index=False)
    seed_summary = comparison.groupby("seed", as_index=False).agg(
        standalone_macro_f1=("standalone_macro_f1", "mean"), joint_macro_f1=("joint_macro_f1", "mean"),
        delta=("delta_joint_minus_standalone", "mean"), delta_std=("delta_joint_minus_standalone", "std"),
    )
    fold_summary = comparison.groupby("outer_fold", as_index=False).agg(
        standalone_macro_f1=("standalone_macro_f1", "mean"), joint_macro_f1=("joint_macro_f1", "mean"),
        delta=("delta_joint_minus_standalone", "mean"),
    )
    seed_summary.to_csv(artifact / "joint_learning_seed_summary.csv", index=False)
    fold_summary.to_csv(artifact / "joint_learning_fold_summary.csv", index=False)
    positive_seeds, positive_folds = int((seed_summary.delta > 0).sum()), int((fold_summary.delta > 0).sum())
    overall_delta = float(comparison.delta_joint_minus_standalone.mean())
    keep_joint = positive_seeds == len(seed_summary) and positive_folds >= 4 and overall_delta > 0
    decision = {
        "decision": "KEEP_JOINT" if keep_joint else "KEEP_STANDALONE", "joint_selected": keep_joint,
        "overall_mean_delta_macro_f1": overall_delta, "positive_seed_count": positive_seeds,
        "seed_count": int(len(seed_summary)), "positive_outer_fold_count": positive_folds,
        "outer_fold_count": int(len(fold_summary)), "selection_scope": "inner_validation_only",
        "outer_results_used_for_selection": False,
        "reason": "Retain joint learning only when every seed and at least four of five outer-training partitions improve.",
    }
    atomic_write_json(artifact / "selection_decision.json", decision)
    atomic_write_json(artifact / "leakage_audit.json", {
        "status": "PASS" if all(row["overlapping_validation_groups"] == 0 for row in leakage_rows) else "FAIL",
        "comparisons": leakage_rows, "G3_used_as_input": False, "subject_indicator_added": True,
    })
    atomic_write_json(artifact / "protocol_snapshot.json", protocol)
    atomic_write_json(state_path, {"study": "joint-uci", "status": "COMPLETE", "fingerprint": fingerprint, "runtime_seconds": time.perf_counter() - started, "future_accessed": False})
    atomic_write_json(checksum_path, build_checksum_manifest(artifact))
    return {"study": "joint-uci", "status": "COMPLETE", "artifact": str(artifact), "decision": decision}


__all__ = ["run_joint_uci"]
