"""Run leakage-safe CNN/BiLSTM architecture and imbalance ablations.

The script appends deep-model evidence to a baseline evidence run created by
``run_final_evidence.py``.  All deep variants use G1/G2, the same train pool,
the same five OOF folds and train-only preprocessing/resampling.  The locked
test is scored only after the single-seed and ensemble candidates are selected
from OOF predictions.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_final_evidence import (
    ARTIFACT_ROOT,
    SEED,
    fit_temperature,
    load_frame,
    markdown_table,
    recommendation_evidence,
    sha256_file,
    split_frame,
    temperature_scale,
    write_json,
)
from src.evidence_metrics import bootstrap_confidence_intervals, classification_metrics
from src.models import create_model
from src.utils import set_seed


ENSEMBLE_SEEDS = (42, 123, 155, 156, 2025, 7, 99, 200, 300, 500, 1337)


@dataclass(frozen=True)
class DeepSpec:
    name: str
    architecture_variant: str
    oversample_method: str
    class_weight_mode: str


SINGLE_SPECS = (
    DeepSpec("cnn_only_none", "cnn_only", "none", "none"),
    DeepSpec("bilstm_only_none", "bilstm_only", "none", "none"),
    DeepSpec("cnn_bilstm_none", "cnn_bilstm", "none", "none"),
    DeepSpec("cnn_bilstm_class_weight", "cnn_bilstm", "none", "balanced"),
    DeepSpec("cnn_bilstm_smote", "cnn_bilstm", "smote", "none"),
    DeepSpec("cnn_bilstm_smote_class_weight", "cnn_bilstm", "smote", "balanced"),
)


MODEL_CONFIG = {
    "cnn_channels": 16,
    "cnn_kernel_size": 1,
    "lstm_hidden_dim": 16,
    "dropout": 0.25,
    "sequence_dropout": 0.10,
    "learning_rate": 0.003,
    "weight_decay": 0.001,
    "batch_size": 32,
    "max_epochs": 35,
    "patience": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="Defaults to artifacts/final/LATEST_RUN.txt")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true", help="Use 2 folds, 2 seeds and 5 epochs.")
    return parser.parse_args()


def class_weights(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=3).astype(float)
    weights = len(labels) / (len(counts) * np.maximum(counts, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def tensor_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(features[:, :, None], dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def predict_probabilities(model: nn.Module, features: np.ndarray) -> np.ndarray:
    model.eval()
    loader = tensor_loader(features, np.zeros(len(features), dtype=int), 128, shuffle=False, seed=SEED)
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for seq_x, _ in loader:
            probabilities.append(model.predict_proba(seq_x).cpu().numpy())
    return np.concatenate(probabilities, axis=0)


def train_predict(
    fit_frame: pd.DataFrame,
    score_frame: pd.DataFrame,
    spec: DeepSpec,
    *,
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, int]:
    set_seed(seed)
    labels = fit_frame["target_class"].to_numpy(dtype=int)
    positions = np.arange(len(fit_frame))
    model_positions, early_positions = train_test_split(
        positions, test_size=0.15, random_state=seed, stratify=labels
    )
    raw_model_x = fit_frame.iloc[model_positions][["G1", "G2"]].to_numpy(dtype=float)
    raw_early_x = fit_frame.iloc[early_positions][["G1", "G2"]].to_numpy(dtype=float)
    raw_score_x = score_frame[["G1", "G2"]].to_numpy(dtype=float)
    model_y = labels[model_positions]
    early_y = labels[early_positions]

    scaler = MinMaxScaler().fit(raw_model_x)
    model_x = scaler.transform(raw_model_x)
    early_x = scaler.transform(raw_early_x)
    score_x = scaler.transform(raw_score_x)
    original_model_y = model_y.copy()
    if spec.oversample_method == "smote":
        minimum = int(np.bincount(model_y, minlength=3).min())
        sampler = SMOTE(random_state=seed, k_neighbors=max(1, min(4, minimum - 1)))
        model_x, model_y = sampler.fit_resample(model_x, model_y)

    config = dict(MODEL_CONFIG)
    config["architecture_variant"] = spec.architecture_variant
    model = create_model("student", config).cpu()
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    weight = class_weights(original_model_y) if spec.class_weight_mode == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
    )
    train_loader = tensor_loader(model_x, model_y, int(config["batch_size"]), shuffle=True, seed=seed)
    early_loader = tensor_loader(early_x, early_y, 128, shuffle=False, seed=seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    epochs_without_improvement = 0
    for _ in range(max_epochs):
        model.train()
        for seq_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(seq_x), batch_y)
            loss.backward()
            optimizer.step()
        early_probabilities: list[np.ndarray] = []
        with torch.no_grad():
            model.eval()
            for seq_x, _ in early_loader:
                early_probabilities.append(model.predict_proba(seq_x).cpu().numpy())
        early_pred = np.concatenate(early_probabilities).argmax(axis=1)
        score = f1_score(early_y, early_pred, average="macro", zero_division=0)
        if score > best_f1 + 1e-8:
            best_f1 = float(score)
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    return predict_probabilities(model, score_x), parameter_count


def evaluate_oof_seed(
    train_pool: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: DeepSpec,
    seed: int,
    *,
    max_epochs: int,
    patience: int,
) -> tuple[np.ndarray, int, list[dict[str, Any]]]:
    probabilities = np.zeros((len(train_pool), 3), dtype=float)
    parameter_count = 0
    fold_rows: list[dict[str, Any]] = []
    labels = train_pool["target_class"].to_numpy(dtype=int)
    for fold_index, (fit_index, validation_index) in enumerate(folds):
        fold_probs, parameter_count = train_predict(
            train_pool.iloc[fit_index].copy(),
            train_pool.iloc[validation_index].copy(),
            spec,
            seed=seed + fold_index * 1000,
            max_epochs=max_epochs,
            patience=patience,
        )
        probabilities[validation_index] = fold_probs
        metrics = classification_metrics(labels[validation_index], fold_probs.argmax(axis=1), fold_probs)
        fold_rows.append(
            {
                "model": spec.name,
                "seed": seed,
                "fold": fold_index,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "ordinal_mae": metrics["ordinal_mae"],
                "brier": metrics["multiclass_brier_score"],
                "ece": metrics["ece"],
            }
        )
    return probabilities, parameter_count, fold_rows


def prediction_rows(
    model: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    fold_ids: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    labels = frame["target_class"].to_numpy(dtype=int)
    predictions = probabilities.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for index in range(len(frame)):
        row = {
            "model": model,
            "source_row_number": int(frame.iloc[index]["__source_row_number"]),
            "true_label": int(labels[index]),
            "predicted_label": int(predictions[index]),
            "confidence": float(probabilities[index].max()),
            "prob_low": float(probabilities[index, 0]),
            "prob_medium": float(probabilities[index, 1]),
            "prob_high": float(probabilities[index, 2]),
        }
        if fold_ids is not None:
            row["outer_fold"] = int(fold_ids[index])
        rows.append(row)
    return rows


def update_bundle_checksums(output_dir: Path, run_manifest: dict[str, Any], parameter_counts: dict[str, int]) -> None:
    checksums = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name not in {"run_manifest.json", "model_checksums.json"}
    }
    write_json(
        output_dir / "model_checksums.json",
        {
            "checkpoints": {},
            "checkpoint_policy": "Binary checkpoints are not committed; recreate from deep_selected_config.json.",
            "parameter_counts": parameter_counts,
            "artifact_checksums": checksums,
        },
    )
    run_manifest["artifact_checksums"] = checksums
    run_manifest["model_parameter_count"].update(parameter_counts)
    write_json(output_dir / "run_manifest.json", run_manifest)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    run_id = args.run_id or (ARTIFACT_ROOT / "LATEST_RUN.txt").read_text(encoding="utf-8").strip()
    output_dir = ARTIFACT_ROOT / run_id
    if not (output_dir / "run_manifest.json").exists():
        raise FileNotFoundError(f"Missing baseline evidence bundle: {output_dir}")
    frame = load_frame()
    train_pool, locked = split_frame(frame)
    cv_folds = 2 if args.smoke else args.cv_folds
    seeds = ENSEMBLE_SEEDS[:2] if args.smoke else ENSEMBLE_SEEDS
    max_epochs = 5 if args.smoke else int(MODEL_CONFIG["max_epochs"])
    patience = 2 if args.smoke else int(MODEL_CONFIG["patience"])
    folds = list(
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED).split(
            train_pool, train_pool["target_class"]
        )
    )
    fold_ids = np.zeros(len(train_pool), dtype=int)
    for fold_index, (_, validation_index) in enumerate(folds):
        fold_ids[validation_index] = fold_index

    cache: dict[tuple[str, int], np.ndarray] = {}
    parameter_counts: dict[str, int] = {}
    fold_rows: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    labels_train = train_pool["target_class"].to_numpy(dtype=int)
    for spec in SINGLE_SPECS:
        probabilities, count, rows = evaluate_oof_seed(
            train_pool, folds, spec, seeds[0], max_epochs=max_epochs, patience=patience
        )
        cache[(spec.name, seeds[0])] = probabilities
        parameter_counts[spec.name] = count
        fold_rows.extend(rows)
        metrics = classification_metrics(labels_train, probabilities.argmax(axis=1), probabilities)
        results[spec.name] = {"spec": asdict(spec), "oof_probabilities": probabilities, "oof_metrics": metrics}

    cnn_candidates = [spec for spec in SINGLE_SPECS if spec.architecture_variant == "cnn_bilstm"]
    selected_base = max(cnn_candidates, key=lambda spec: (results[spec.name]["oof_metrics"]["macro_f1"], -parameter_counts[spec.name]))
    for seed in seeds[1:]:
        probabilities, _, rows = evaluate_oof_seed(
            train_pool, folds, selected_base, seed, max_epochs=max_epochs, patience=patience
        )
        cache[(selected_base.name, seed)] = probabilities
        fold_rows.extend(rows)
    ensemble_probabilities = np.mean([cache[(selected_base.name, seed)] for seed in seeds], axis=0)
    ensemble_name = f"{selected_base.name}_ensemble_{len(seeds)}seed"
    ensemble_metrics = classification_metrics(labels_train, ensemble_probabilities.argmax(axis=1), ensemble_probabilities)
    results[ensemble_name] = {
        "spec": {**asdict(selected_base), "seed_list": list(seeds), "aggregation": "mean_probability"},
        "oof_probabilities": ensemble_probabilities,
        "oof_metrics": ensemble_metrics,
    }
    parameter_counts[ensemble_name] = parameter_counts[selected_base.name] * len(seeds)

    final_candidates = [selected_base.name, ensemble_name]
    selected_final = max(final_candidates, key=lambda name: results[name]["oof_metrics"]["macro_f1"])
    labels_locked = locked["target_class"].to_numpy(dtype=int)
    summary_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    locked_rows: list[dict[str, Any]] = []
    locked_probabilities_by_model: dict[str, np.ndarray] = {}
    for name, result in results.items():
        if name == ensemble_name:
            member_probabilities = []
            for seed in seeds:
                member, _ = train_predict(
                    train_pool,
                    locked,
                    selected_base,
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=patience,
                )
                member_probabilities.append(member)
            locked_probabilities = np.mean(member_probabilities, axis=0)
        else:
            spec = next(spec for spec in SINGLE_SPECS if spec.name == name)
            locked_probabilities, _ = train_predict(
                train_pool,
                locked,
                spec,
                seed=seeds[0],
                max_epochs=max_epochs,
                patience=patience,
            )
        locked_probabilities_by_model[name] = locked_probabilities
        oof_metrics = result["oof_metrics"]
        locked_metrics = classification_metrics(labels_locked, locked_probabilities.argmax(axis=1), locked_probabilities)
        temperature = fit_temperature(result["oof_probabilities"], labels_train)
        calibrated_locked = temperature_scale(locked_probabilities, temperature)
        calibrated_metrics = classification_metrics(labels_locked, calibrated_locked.argmax(axis=1), calibrated_locked)
        cis = bootstrap_confidence_intervals(
            labels_locked,
            locked_probabilities.argmax(axis=1),
            n_resamples=args.bootstrap_resamples,
            seed=SEED,
        )
        result.update(
            {
                "temperature": temperature,
                "locked_test_metrics": locked_metrics,
                "locked_test_calibrated_metrics": calibrated_metrics,
                "locked_test_confidence_intervals": cis,
            }
        )
        fold_f1 = [
            classification_metrics(
                labels_train[validation_index],
                result["oof_probabilities"][validation_index].argmax(axis=1),
                result["oof_probabilities"][validation_index],
            )["macro_f1"]
            for _, validation_index in folds
        ]
        summary_rows.append(
            {
                "model": name,
                "architecture_variant": result["spec"]["architecture_variant"],
                "oversample_method": result["spec"]["oversample_method"],
                "class_weight_mode": result["spec"]["class_weight_mode"],
                "seed_count": len(seeds) if name == ensemble_name else 1,
                "parameter_count_per_model": parameter_counts[selected_base.name] if name == ensemble_name else parameter_counts[name],
                "oof_macro_f1": oof_metrics["macro_f1"],
                "outer_fold_macro_f1_mean": float(np.mean(fold_f1)) if fold_f1 else oof_metrics["macro_f1"],
                "outer_fold_macro_f1_std": float(np.std(fold_f1, ddof=1)) if len(fold_f1) > 1 else 0.0,
                "locked_accuracy": locked_metrics["accuracy"],
                "locked_macro_f1": locked_metrics["macro_f1"],
                "locked_weighted_f1": locked_metrics["weighted_f1"],
                "locked_balanced_accuracy": locked_metrics["balanced_accuracy"],
                "locked_qwk": locked_metrics["quadratic_weighted_kappa"],
                "locked_ordinal_mae": locked_metrics["ordinal_mae"],
                "locked_two_step_errors": locked_metrics["two_step_errors"],
                "locked_pr_auc_macro": locked_metrics["pr_auc_macro"],
                "locked_brier": locked_metrics["multiclass_brier_score"],
                "locked_ece": locked_metrics["ece"],
                "temperature_oof": temperature,
                "locked_brier_calibrated": calibrated_metrics["multiclass_brier_score"],
                "locked_ece_calibrated": calibrated_metrics["ece"],
            }
        )
        oof_rows.extend(prediction_rows(name, train_pool, result["oof_probabilities"], fold_ids=fold_ids))
        locked_rows.extend(prediction_rows(name, locked, locked_probabilities))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "deep_ablation_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "deep_outer_fold_metrics.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output_dir / "deep_oof_predictions.csv", index=False)
    pd.DataFrame(locked_rows).to_csv(output_dir / "deep_locked_test_predictions.csv", index=False)
    serializable_results = {
        name: {key: value for key, value in result.items() if key != "oof_probabilities"}
        for name, result in results.items()
    }
    write_json(output_dir / "deep_classification_report.json", serializable_results)
    selected_config = {
        "selection_source": "train_pool_oof_only",
        "locked_test_used_for_selection": False,
        "scenario": "late_stage_G1_G2",
        "same_outer_folds_as_baselines": not args.smoke,
        "model_config": MODEL_CONFIG,
        "single_seed": seeds[0],
        "ensemble_seeds": list(seeds),
        "selected_imbalance_base": asdict(selected_base),
        "selected_final_deep_candidate": selected_final,
        "selection_metric": "OOF Macro-F1; locked test excluded",
        "parameter_counts": parameter_counts,
    }
    write_json(output_dir / "deep_selected_config.json", selected_config)

    original_ablation = pd.read_csv(output_dir / "ablation_results.csv")
    if "result_family" in original_ablation.columns:
        original_ablation = original_ablation[original_ablation["result_family"] == "classical"].copy()
    else:
        original_ablation.insert(0, "result_family", "classical")
    deep_for_merge = summary.copy()
    deep_for_merge.insert(0, "result_family", "deep")
    pd.concat([original_ablation, deep_for_merge], ignore_index=True, sort=False).to_csv(
        output_dir / "ablation_results.csv", index=False
    )

    selected_probabilities = locked_probabilities_by_model[selected_final]
    sorted_positions = np.argsort(locked["__source_row_number"].to_numpy())
    recommendation_metrics, recommendation_cases = recommendation_evidence(
        locked.iloc[sorted_positions].reset_index(drop=True),
        selected_probabilities[sorted_positions].argmax(axis=1),
        selected_probabilities[sorted_positions],
        source_model=selected_final,
    )
    write_json(output_dir / "recommendation_evaluation.json", recommendation_metrics)
    pd.DataFrame(recommendation_cases).to_csv(output_dir / "recommendation_expert_review_cases.csv", index=False)

    readme_path = output_dir / "README.md"
    base_readme = readme_path.read_text(encoding="utf-8").split("\n## Deep ablation\n", 1)[0].rstrip()
    deep_section = "\n".join(
        [
            "",
            "## Deep ablation",
            "",
            "CNN-only, BiLSTM-only and CNN-BiLSTM use the same G1/G2 folds. SMOTE and class weights are ablated independently; preprocessing and SMOTE fit only fold-training rows.",
            "",
            markdown_table(
                summary,
                [
                    "model",
                    "seed_count",
                    "parameter_count_per_model",
                    "oof_macro_f1",
                    "locked_accuracy",
                    "locked_macro_f1",
                    "locked_ordinal_mae",
                ],
            ),
            "",
            f"Selected from OOF only: `{selected_final}`.",
            "",
        ]
    )
    readme_path.write_text(base_readme + "\n" + deep_section, encoding="utf-8")

    run_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    run_manifest["deep_ablation"] = {
        "selected_config": selected_config,
        "results": summary_rows,
        "smoke": bool(args.smoke),
    }
    update_bundle_checksums(output_dir, run_manifest, parameter_counts)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "selected_base": selected_base.name,
                "selected_final_deep": selected_final,
                "deep_results": summary_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
