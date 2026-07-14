"""Hardened V3.1 smoke: one outer fold, one seed, never a scientific benchmark."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT_DIR
from src.data_pipeline import SOURCE_ROW_NUMBER_COLUMN, process_target_and_stratify
from src.evaluation.metrics import classification_metrics
from src.evaluation.model_v3_protocol import (
    MODEL_REGISTRY, build_expected_jobs, build_selection_study_contract, checksum,
    duplicate_jobs, legacy_intersection, map_g3_to_class, regression_metric_summary,
    validate_loader_rows, validate_selection_results, validate_shape_rows,
)
from src.evaluation.neural_sanity_v2_2 import loader_statistics
from src.evaluation.protocol import (
    file_checksum, load_fold_manifest, outer_folds_from_manifest, source_record_identity,
    validate_probability_matrix,
)
from src.models.ordinal_v3 import (
    SequenceOrdinalV3, TabularV3Model, TrainOnlyTargetScaler, multitask_loss, ordinal_bce_loss,
)
from src.postgres_data_source import load_dataset_version_from_postgres

AROOT = ROOT_DIR / "artifacts/model_v3_smoke"
LEGACY = ROOT_DIR / "artifacts/legacy_v1/legacy_manifest.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT_DIR, text=True).strip()


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def feature_contract(track: str, fold_checksum: str) -> dict:
    features = ["G1", "G2"] if track == "late_stage" else ["G1"]
    contract = {
        "contract_version": "v3_1_feature_1", "scenario": track,
        "cutoff": "after_G2" if track == "late_stage" else "after_G1",
        "feature_set_id": "+".join(features), "ordered_features": features,
        "preprocessing_contract": "StandardScaler fit on the current training partition only",
        "scaler_contract": "train_only_standard_scaler", "target_excluded": True,
        "temporal_availability_status": "allowed_by_frozen_feature_allowlist",
        "class_order": ["Low", "Medium", "High"], "dataset_version": 1,
        "fold_manifest_checksum": fold_checksum,
    }
    contract["semantic_checksum"] = checksum(contract)
    return contract


def make_model(family: str, input_dim: int, config: dict) -> torch.nn.Module:
    if family == "M4":
        return SequenceOrdinalV3(config["cnn_channels"], config["cnn_kernel_size"],
                                 config["lstm_hidden_dim"], config["dropout"], config["sequence_dropout"])
    model = MODEL_REGISTRY[family]
    return TabularV3Model(input_dim, int(config["hidden_width"]), int(config["hidden_layers"]),
                          float(config["dropout"]), model["ordinal"], model["regression"])


def tensor_x(values: np.ndarray, family: str) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor.unsqueeze(2) if family == "M4" else tensor


def loss_for(family: str, logits: torch.Tensor, regression: torch.Tensor | None,
             labels: torch.Tensor, scaled_g3: torch.Tensor, lam: float) -> torch.Tensor:
    classification = ordinal_bce_loss(logits, labels) if MODEL_REGISTRY[family]["ordinal"] else torch.nn.functional.cross_entropy(logits, labels)
    return multitask_loss(classification, regression, scaled_g3, lam) if regression is not None else classification


def fit_torch(family: str, config: dict, x: np.ndarray, labels: np.ndarray, scaled_g3: np.ndarray,
              *, seed: int, epochs: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = make_model(family, x.shape[1], config)
    model._model_id = family
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]),
                                 weight_decay=float(config["weight_decay"]))
    dataset = TensorDataset(tensor_x(x, family), torch.tensor(labels, dtype=torch.long),
                            torch.tensor(scaled_g3, dtype=torch.float32))
    model.train()
    for epoch in range(epochs):
        loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, drop_last=False,
                            generator=torch.Generator().manual_seed(seed + epoch))
        for xb, yb, rb in loader:
            optimizer.zero_grad()
            logits, reg = model(xb)
            loss_for(family, logits, reg, yb, rb, float(config.get("lambda", 0.0))).backward()
            optimizer.step()
    return model


def predict_torch(model: torch.nn.Module, family: str, values: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    model.eval()
    with torch.no_grad():
        logits, reg = model(tensor_x(values, family))
        probability = model.predict_proba(tensor_x(values, family)).cpu().numpy()
        cumulative = torch.sigmoid(logits).cpu().numpy() if MODEL_REGISTRY[family]["ordinal"] else None
        regression = None if reg is None else reg.cpu().numpy()
    return probability, cumulative, regression


def select_torch_config(family: str, train: pd.DataFrame, features: list[str], config: dict,
                        study: dict, trial_rows: list[dict]) -> dict:
    """Smoke uses one predeclared candidate but executes all three inner folds."""
    split = StratifiedKFold(n_splits=3, shuffle=True, random_state=study["study_seed"])
    values = train[features].to_numpy(float)
    labels = train.G3.to_numpy(int)
    raw_g3 = train._raw_g3.to_numpy(float)
    scores = []
    for inner_fold, (idx_train, idx_valid) in enumerate(split.split(values, labels)):
        x_scaler = StandardScaler().fit(values[idx_train])
        g3_scaler = TrainOnlyTargetScaler().fit(raw_g3[idx_train])
        model = fit_torch(family, config, x_scaler.transform(values[idx_train]), labels[idx_train],
                          g3_scaler.transform(raw_g3[idx_train]), seed=study["study_seed"] + inner_fold, epochs=2)
        probability, _, _ = predict_torch(model, family, x_scaler.transform(values[idx_valid]))
        score = classification_metrics(labels[idx_valid], probability.argmax(1), probability)["macro_f1"]
        scores.append(score)
        trial_rows.append({"study_id": study["study_id"], "trial_id": 0, "inner_fold": inner_fold,
                           "config_checksum": checksum(config), "inner_macro_f1": score})
    return {"study_id": study["study_id"], "model_family": family, "track": study["track"],
            "outer_fold": study["outer_fold"], "selected_trial_id": 0,
            "config": config, "config_checksum": checksum(config), "inner_macro_f1_mean": float(np.mean(scores))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = AROOT / args.run_id
    if root.exists():
        raise FileExistsError(root)
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked source tree must be clean.")

    root.mkdir(parents=True)
    source_commit = git("rev-parse", "HEAD")
    manifest = load_fold_manifest()
    raw, metadata = load_dataset_version_from_postgres("student-mat", 1)
    frame = process_target_and_stratify(raw.copy(), "G3", "student", "3class").drop(columns=["_strat_target"])
    wanted = {row["source_row_number"] for row in manifest["development_records"]}
    frame = frame[frame[SOURCE_ROW_NUMBER_COLUMN].isin(wanted)].sort_values(SOURCE_ROW_NUMBER_COLUMN).reset_index(drop=True)
    frame["_raw_g3"] = raw.loc[frame[SOURCE_ROW_NUMBER_COLUMN].astype(int), "G3"].to_numpy(float)
    development = {row["source_record_identity"] for row in manifest["development_records"]}
    legacy = set(json.loads(LEGACY.read_text(encoding="utf-8"))["current_79_record_ids"])
    intersection = legacy_intersection(development, legacy)
    if intersection:
        raise RuntimeError("Development/legacy-79 intersection is not empty.")

    folds = outer_folds_from_manifest(frame, manifest)
    train_idx, valid_idx = folds[0]
    train, valid = frame.iloc[train_idx], frame.iloc[valid_idx]
    contracts = {track: feature_contract(track, manifest["manifest_checksum"]) for track in ("late_stage", "early_warning")}
    target_contract = {
        "contract_version": "v3_1_target_1", "class_order": ["Low", "Medium", "High"],
        "class_mapping": {"Low": "G3<=9", "Medium": "10<=G3<=14", "High": "G3>=15"},
        "continuous_g3": {"description": "finer-grained supervision from the same underlying outcome",
                            "scaler": "fit on current training partition only", "primary_scale": "raw_0_20",
                            "clip_before_primary_rmse_r2": False},
    }
    target_contract["semantic_checksum"] = checksum(target_contract)
    search_contract = {"contract_version": "v3_1_search_1", "hidden_width": [8, 16, 32],
                       "hidden_layers": [1, 2], "dropout": [0.0, 0.15, 0.30],
                       "learning_rate": {"low": 0.0005, "high": 0.005, "distribution": "log_uniform"},
                       "weight_decay": {"low": 1e-6, "high": 1e-3, "distribution": "log_uniform"},
                       "batch_size": [16, 32], "max_epochs": 60, "patience": 10, "drop_last": False,
                       "multitask_lambda": [0.1, 0.3, 1.0], "trials_per_study": 20}
    search_contract["semantic_checksum"] = checksum(search_contract)
    selection_contract = build_selection_study_contract(args.run_id, manifest["manifest_checksum"], source_commit,
                                                        search_contract["semantic_checksum"], target_contract, smoke=True)
    dump(root / "selection_study_contract.json", selection_contract)
    source_s3 = json.loads((ROOT_DIR / "artifacts/benchmark_v2/benchmark-v2-full-20260713c/configs/selected_configs.json").read_text())["late_stage/cnn_bilstm_v2_tuned/fold0"]["config"]
    source_s3 = {**source_s3, "max_epochs": 40, "patience": 8, "scheduler_patience": 3}
    common = {"hidden_width": 16, "hidden_layers": 1, "dropout": 0.15, "learning_rate": 0.002,
              "weight_decay": 1e-4, "batch_size": 16, "max_epochs": 60, "patience": 10, "drop_last": False}
    configs = {"M0": dict(common), "M1": dict(common), "M2": {**common, "lambda": 0.3},
               "M3": {**common, "lambda": 0.3}, "M4": source_s3, "B0": {"alpha": 1.0, "alpha_grid": [0.01, 0.1, 1.0, 10.0]}}
    expected = build_expected_jobs(args.run_id, {0: len(valid)}, manifest["manifest_checksum"], source_commit,
                                   contracts, target_contract, smoke=True,
                                   config_checksums={key: checksum(value) for key, value in configs.items()},
                                   selection_contract_checksum=selection_contract["semantic_checksum"])
    dump(root / "expected_job_contract.json", expected)
    dump(root / "feature_contracts.json", contracts)
    dump(root / "target_supervision_contract.json", target_contract)
    dump(root / "search_space_contract.json", search_contract)
    run = {"run_id": args.run_id, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
           "source_commit": source_commit, "expected_jobs": len(expected["jobs"]),
           "expected_predictions": sum(x["expected_record_count"] for x in expected["jobs"]),
           "fold_manifest_checksum": manifest["manifest_checksum"], "dataset_checksum": metadata["content_hash"],
           "legacy_intersection_count": len(intersection), "full_benchmark": False,
           "scientific_eligibility": "smoke_only_not_for_model_ranking"}
    dump(root / "run_manifest.json", run)

    trial_rows: list[dict] = []
    selected_rows: list[dict] = []
    study_by_family = {study["model_family"]: study for study in selection_contract["studies"]}
    for family in ("M0", "M1", "M2", "M3"):
        selected_rows.append(select_torch_config(family, train, ["G1", "G2"], configs[family], study_by_family[family], trial_rows))
    trials = pd.DataFrame(trial_rows)
    selected = pd.DataFrame(selected_rows)
    trials.to_csv(root / "selection_trials.csv", index=False)
    selected.to_json(root / "selected_configs.json", orient="records", indent=2)
    selection_status = validate_selection_results(selection_contract, trials, selected)
    if any(selection_status.values()):
        raise RuntimeError(f"Invalid smoke selection evidence: {selection_status}")

    rows: list[dict] = []
    metric_rows: list[dict] = []
    diagnostics: list[dict] = []
    parameters: list[dict] = []
    loader_rows: list[dict] = []
    shape_rows: list[dict] = []
    ordinal_checks: list[bool] = []
    selected_by_family = {row.model_family: row for row in selected.itertuples(index=False)}
    for family in MODEL_REGISTRY:
        started = time.perf_counter()
        features = ["G1", "G2"]
        config = configs[family] if family not in selected_by_family else selected_by_family[family].config
        x_scaler = StandardScaler().fit(train[features])
        x_train, x_valid = x_scaler.transform(train[features]), x_scaler.transform(valid[features])
        target_scaler = TrainOnlyTargetScaler().fit(train._raw_g3)
        raw_prediction = None
        cumulative = None
        if family == "B0":
            ridge = Ridge(alpha=float(config["alpha"])).fit(x_train, train._raw_g3)
            raw_prediction = ridge.predict(x_valid)
            predicted = map_g3_to_class(raw_prediction)
            probability = np.eye(3, dtype=float)[predicted]
            parameter_count = int(ridge.coef_.size + 1)
            selected_epoch = None
        else:
            model = fit_torch(family, config, x_train, train.G3.to_numpy(int), target_scaler.transform(train._raw_g3),
                              seed=42, epochs=3)
            probability, cumulative, scaled_prediction = predict_torch(model, family, x_valid)
            predicted = probability.argmax(1)
            raw_prediction = None if scaled_prediction is None else target_scaler.inverse_transform(scaled_prediction)
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            selected_epoch = 3
            if family == "M4":
                kernel = int(config["cnn_kernel_size"])
                length = 2 + 2 * (kernel // 2) - kernel + 1
                shape_rows.append({"model_family": family, "cnn_kernel_size": kernel, "input_sequence_length": 2,
                                   "cnn_output_sequence_length": length, "bilstm_input_sequence_length": length})
        validate_probability_matrix(probability, predicted)
        metric = classification_metrics(valid.G3.to_numpy(int), predicted, probability)
        if raw_prediction is not None:
            metric.update(regression_metric_summary(valid._raw_g3.to_numpy(float), raw_prediction))
        metric_rows.append({"model_family": family, "track": "late_stage", "outer_fold": 0,
                            "training_seed": 0 if family == "B0" else 42,
                            **{k: v for k, v in metric.items() if k not in ["confusion_matrix", "per_class"]},
                            "confusion_matrix": json.dumps(metric["confusion_matrix"]), "per_class": json.dumps(metric["per_class"])})
        parameters.append({"model_family": family, "trainable_parameters": parameter_count,
                           "training_seconds": time.perf_counter() - started, "selected_epoch_smoke": selected_epoch,
                           "training_engine": MODEL_REGISTRY[family]["training_engine"]})
        diagnostics.append({"model_family": family, "target_scaler_fit_records": len(train),
                            "target_scaler_fit_record_ids_checksum": checksum(sorted(source_record_identity(1, x) for x in train[SOURCE_ROW_NUMBER_COLUMN])),
                            "outer_validation_target_scaler_fit_records": 0,
                            "regression_inverse_transform_verified": bool(raw_prediction is None or np.isfinite(raw_prediction).all()),
                            "outer_refit_selected_config_checksum": checksum(config),
                            "training_engine": MODEL_REGISTRY[family]["training_engine"]})
        loader_rows.append({"model_family": family, "phase": "outer_refit", **loader_statistics(len(train), int(config.get("batch_size", len(train))), False)})
        if cumulative is not None:
            ordinal_checks.append(bool(np.all(cumulative[:, 0] >= cumulative[:, 1] - 1e-7)))
        for position, (_, record) in enumerate(valid.iterrows()):
            rows.append({"run_id": args.run_id, "model_family": family, "track": "late_stage", "feature_set_id": "G1+G2",
                         "target_supervision_type": MODEL_REGISTRY[family]["target_supervision"], "training_engine": MODEL_REGISTRY[family]["training_engine"],
                         "outer_fold": 0, "training_seed": 0 if family == "B0" else 42,
                         "record_id": source_record_identity(1, record[SOURCE_ROW_NUMBER_COLUMN]), "true_label": int(record.G3),
                         "raw_g3": float(record._raw_g3), "predicted_label": int(predicted[position]),
                         "probability_low": float(probability[position, 0]), "probability_medium": float(probability[position, 1]),
                         "probability_high": float(probability[position, 2]), "predicted_g3_raw": None if raw_prediction is None else float(raw_prediction[position]),
                         "fold_manifest_checksum": manifest["manifest_checksum"], "feature_contract_checksum": contracts["late_stage"]["semantic_checksum"],
                         "target_contract_checksum": target_contract["semantic_checksum"], "config_checksum": checksum(config),
                         "source_commit": source_commit})

    predictions = pd.DataFrame(rows)
    metrics = pd.DataFrame(metric_rows)
    loader_frame, shape_frame = pd.DataFrame(loader_rows), pd.DataFrame(shape_rows)
    predictions.to_csv(root / "smoke_predictions.csv", index=False)
    metrics.to_csv(root / "smoke_metrics.csv", index=False)
    pd.DataFrame(parameters).to_csv(root / "parameter_count_comparison.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(root / "training_diagnostics.csv", index=False)
    loader_frame.to_csv(root / "loader_diagnostics.csv", index=False)
    shape_frame.to_csv(root / "shape_diagnostics.csv", index=False)
    expected_keys = {tuple(job[key] for key in ["model_family", "track", "outer_fold", "training_seed"]) for job in expected["jobs"]}
    actual_keys = set(tuple(row) for row in predictions[["model_family", "track", "outer_fold", "training_seed"]].drop_duplicates().itertuples(index=False, name=None))
    duplicate_status = {"expected_job_duplicates": duplicate_jobs(pd.DataFrame(expected["jobs"])),
                        "metric_duplicates": duplicate_jobs(metrics),
                        "prediction_duplicates": int(predictions.duplicated(["model_family", "track", "outer_fold", "training_seed", "record_id"]).sum())}
    probability = predictions[["probability_low", "probability_medium", "probability_high"]].to_numpy(float)
    recomputed = {}
    job_columns = ["model_family", "track", "outer_fold", "training_seed"]
    for key, group in predictions.groupby(job_columns):
        recomputed[key] = classification_metrics(group.true_label, group.predicted_label, group[["probability_low", "probability_medium", "probability_high"]].to_numpy())["macro_f1"]
    stored_by_job = {tuple(getattr(row, column) for column in job_columns): row.macro_f1 for row in metrics.itertuples(index=False)}
    config_valid = all(group.config_checksum.nunique() == 1 for _, group in predictions.groupby("model_family"))
    selection_config_valid = all(row.config_checksum == checksum(row.config) for row in selected.itertuples(index=False))
    validation = {"run_id": args.run_id, "expected_jobs": len(expected_keys), "actual_jobs": len(actual_keys),
                  "missing_jobs": len(expected_keys - actual_keys), "unexpected_jobs": len(actual_keys - expected_keys),
                  **duplicate_status, "expected_predictions": sum(j["expected_record_count"] for j in expected["jobs"]),
                  "actual_predictions": len(predictions),
                  "record_coverage_valid": all(set(group.record_id) == set(source_record_identity(1, x) for x in valid[SOURCE_ROW_NUMBER_COLUMN]) for _, group in predictions.groupby("model_family")),
                  "probability_contract_valid": bool(np.isfinite(probability).all() and (probability >= 0).all() and np.max(np.abs(probability.sum(1) - 1.0)) <= 1e-6),
                  "cumulative_ordering_valid": all(ordinal_checks), "regression_inverse_transform_valid": bool(pd.DataFrame(diagnostics).regression_inverse_transform_verified.all()),
                  "target_scaler_train_only": bool((pd.DataFrame(diagnostics).outer_validation_target_scaler_fit_records == 0).all()),
                  "loader_diagnostics_content_valid": validate_loader_rows(loader_frame), "shape_diagnostics_content_valid": validate_shape_rows(shape_frame),
                  "legacy_intersection_count": len(intersection), "selection_study_validation": selection_status,
                  "selected_config_propagation_valid": bool(config_valid and selection_config_valid),
                  "metric_recomputation_valid": bool(set(recomputed) == set(stored_by_job) and all(np.isclose(value, stored_by_job[key]) for key, value in recomputed.items())),
                  "overall_validation_status": "invalid"}
    required = [validation["missing_jobs"] == 0, validation["unexpected_jobs"] == 0,
                not any(duplicate_status.values()), validation["actual_predictions"] == validation["expected_predictions"],
                validation["record_coverage_valid"], validation["probability_contract_valid"], validation["cumulative_ordering_valid"],
                validation["regression_inverse_transform_valid"], validation["target_scaler_train_only"],
                validation["loader_diagnostics_content_valid"], validation["shape_diagnostics_content_valid"],
                validation["legacy_intersection_count"] == 0, not any(selection_status.values()),
                validation["selected_config_propagation_valid"], validation["metric_recomputation_valid"]]
    validation["overall_validation_status"] = "valid" if all(required) else "invalid"
    dump(root / "smoke_validation.json", validation)
    run.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat()})
    dump(root / "run_manifest.json", run)
    checksums = {str(path.relative_to(root)): file_checksum(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "checksums.json", checksums)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
