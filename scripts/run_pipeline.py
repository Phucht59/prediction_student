"""End-to-end thesis pipeline: CNN-BiLSTM classifier, learning paths, PostgreSQL."""

from __future__ import annotations

import argparse
import subprocess
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    precision_recall_curve,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATASETS,
    DEFAULT_SEED,
    EXPLANATIONS_DIR,
    FIXED_SEEDS,
    LOCKED_TEST_SIZE,
    METRICS_DIR,
    MANIFESTS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    ROOT_DIR,
    RECOMMENDATIONS_DIR,
    REPORTS_DIR,
    ensure_dirs,
)
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    apply_feature_engineering,
    attach_source_row_numbers,
    SOURCE_ROW_NUMBER_COLUMN,
    get_context_excluded_columns,
    get_sequence_columns,
    process_target_and_stratify,
)
from src.evaluation import (
    initialize_experiment_run_in_postgres,
    persist_evaluation_to_postgres,
    prepare_storage_context,
    project_uri,
)
from src.explainability import explain_model
from src.models import create_model
from src.estimator_factory import resolve_student_config, validate_resolved_config
from src.model_selection import (
    apply_probability_calibration,
    apply_threshold_policy,
    combine_seed_probabilities,
    fit_final_development_estimator,
    predict_with_fitted_estimator,
)
from src.postgres_data_source import (
    load_dataset_version_from_postgres,
    load_experiment_run,
    reconstruct_splits_from_run,
    verify_run_split_manifest,
)
from src.recommendation import generate_learning_path_report
from src.reproducibility import sha256_file
from src.train_pipeline import objective
from src.utils import set_seed, setup_logger
from src.loss_description import describe_effective_loss

logger = setup_logger("run_pipeline")


class LoadedStudy:
    def __init__(self, value: float, params: dict):
        self.best_value = value
        self.best_params = params


def load_selection_config(path_or_json: str | None) -> dict | None:
    if not path_or_json:
        return None
    path = Path(path_or_json)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(path_or_json)


def resolve_frozen_strategy_metadata(
    selected_strategy: dict | None,
    *,
    debug: bool = False,
) -> dict:
    selected_strategy = selected_strategy or {}
    seed_list = list(selected_strategy.get("seed_list") or (FIXED_SEEDS[:1] if debug else FIXED_SEEDS))
    probability_combination_method = selected_strategy.get("ensemble_method", "mean_probability")
    threshold_policy = selected_strategy.get("threshold_policy", {"type": "argmax"})
    calibration_policy = selected_strategy.get("calibration_policy", {"type": "none"})
    if len(seed_list) == 1:
        actual_aggregation_method = "single_model"
        strategy_name = selected_strategy.get("strategy_name") or f"single_seed_{seed_list[0]}"
    else:
        actual_aggregation_method = probability_combination_method
        strategy_name = selected_strategy.get("strategy_name") or probability_combination_method
    return {
        "strategy_name": strategy_name,
        "actual_seed_list": seed_list,
        "probability_combination_method": probability_combination_method,
        "actual_aggregation_method": actual_aggregation_method,
        "calibration_policy": calibration_policy,
        "threshold_policy": threshold_policy,
        "decision_rule": threshold_policy.get("type", "argmax"),
    }


def normalize_cnn_bilstm_classifier_params(params: dict) -> dict:
    """Upgrade a historical flat config to the canonical Strategy B contract."""
    try:
        validate_resolved_config(params)
        return dict(params)
    except (KeyError, TypeError, ValueError):
        pass
    forbidden = {
        "context_hidden_dim",
        "fusion_hidden_dim",
        "context_dropout",
        "fusion_dropout",
        "embedding_dim",
        "ablation_mode",
    }
    normalized = {key: value for key, value in dict(params).items() if key not in forbidden}
    normalized.setdefault("architecture_variant", "cnn_bilstm")
    normalized.setdefault("class_weight_mode", "none")
    # Historical files may contain only architecture dimensions.  Upgrade them
    # once here; all estimator consumers below still receive a complete config.
    historical_defaults = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 32,
        "oversample_method": "none",
        "loss": "cross_entropy",
        "smote_ratio": 1.0,
        "resampling_k_neighbors": 5,
        "cnn_kernel_size": 1,
        "dropout": 0.2,
        "sequence_dropout": 0.1,
        "max_epochs": 100,
        "patience": 15,
    }
    for key, value in historical_defaults.items():
        normalized.setdefault(key, value)
    return resolve_student_config(
        normalized,
        architecture_variant="cnn_bilstm",
        suggested_parameters={},
        scheduler_type="fixed_lr",
        swa_enabled=False,
        drop_last_train=False,
        evidence_role="final_corrected_full_development_estimator",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--target-mode", default="3class", choices=["3class"])
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--params-json", default=None, help="JSON file or JSON string used to skip Optuna")
    parser.add_argument(
        "--selection-config-json",
        default=None,
        help="Validation-only model-selection config used to freeze params, seeds, ensemble, and thresholds.",
    )
    parser.add_argument("--run-id", default=None, help="Existing UUID used to retry the same execution")
    parser.add_argument("--dataset-version-id", type=int, default=None)
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Development-only opt-out. Production thesis runs persist to PostgreSQL by default.",
    )
    parser.add_argument(
        "--allow-legacy-observed-evaluation",
        action="store_true",
        help=(
            "Explicitly authorize the historical 79-record evaluation path. "
            "It is disabled by default and forbidden for model selection, calibration or final confirmation."
        ),
    )
    return parser.parse_args()


def derive_target_frame(raw_frame, dataset_name: str, target_mode: str):
    spec = DATASETS[dataset_name]
    df_strat = process_target_and_stratify(
        attach_source_row_numbers(raw_frame),
        spec.target_col,
        spec.kind,
        target_mode,
    )
    return df_strat.dropna(subset=["_strat_target"])


def create_locked_split_from_frame(raw_frame, dataset_name: str, target_mode: str):
    df_strat = derive_target_frame(raw_frame, dataset_name, target_mode)
    train_pool, locked_test = train_test_split(
        df_strat,
        test_size=LOCKED_TEST_SIZE,
        stratify=df_strat["_strat_target"],
        random_state=DEFAULT_SEED,
    )
    return (
        train_pool.drop(columns=["_strat_target"]),
        locked_test.drop(columns=["_strat_target"]),
    )


def reconstruct_existing_run_splits(raw_frame, run_id: str, dataset_name: str, target_mode: str):
    target_frame = derive_target_frame(raw_frame, dataset_name, target_mode)
    target_frame = target_frame.drop(columns=["_strat_target"])
    return reconstruct_splits_from_run(target_frame, run_id)


def git_output(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT_DIR,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    return result.stdout.strip()


def git_reproducibility_context(run_id: str) -> dict:
    git_commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    if not status:
        return {
            "git_commit": git_commit,
            "working_tree_state": "clean",
            "source_diff_uri": None,
            "source_diff_hash": None,
        }

    diff_text = "\n".join(
        [
            "# git status --porcelain",
            status,
            "",
            "# git diff --binary HEAD",
            git_output("diff", "--binary", "HEAD"),
        ]
    )
    diff_path = MANIFESTS_DIR / "source_diffs" / f"{run_id}.diff"
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(diff_text, encoding="utf-8")
    return {
        "git_commit": git_commit,
        "working_tree_state": "dirty",
        "source_diff_uri": project_uri(diff_path),
        "source_diff_hash": sha256_file(diff_path),
    }


def environment_lock_context() -> dict:
    lock_path = ROOT_DIR / "environment.yml"
    if not lock_path.exists():
        lock_path = ROOT_DIR / "requirements.txt"
    if not lock_path.exists():
        raise FileNotFoundError("Missing environment.yml or requirements.txt for environment lock provenance.")
    return {
        "environment_lock_uri": project_uri(lock_path),
        "environment_lock_hash": sha256_file(lock_path),
    }


def artifact_manifest_context(run_id: str, dataset_name: str, target_mode: str, best_params: dict) -> dict:
    artifact_path = MANIFESTS_DIR / "artifacts" / f"{run_id}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "run_id": run_id,
        "dataset": dataset_name,
        "target_mode": target_mode,
        "model_artifact_directory": project_uri(MODELS_DIR),
        "model_artifact_pattern": f"{dataset_name}_3class_cnn_bilstm_classifier_seed*.pt",
        "best_params": best_params,
    }
    artifact_path.write_text(json.dumps(artifact_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"artifact_uri": project_uri(artifact_path)}


def load_or_create_run_manifest(run_id: str) -> dict:
    manifest_path = MANIFESTS_DIR / "runs" / f"{run_id}.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_study(args, train_pool, spec):
    if args.params_json:
        path = Path(args.params_json)
        params = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(args.params_json)
        best_value = float(params.pop("_best_value", 0.0))
        logger.info("Using provided parameters and skipping Optuna.")
        return LoadedStudy(best_value, params)

    import optuna

    target_trials = 1 if args.debug else (args.n_trials or (250 if spec.kind == "xapi" else 50))
    study_kwargs = {
        "direction": "maximize",
        "sampler": optuna.samplers.TPESampler(seed=DEFAULT_SEED, multivariate=True),
        "pruner": optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2),
    }
    if spec.kind == "xapi" and not args.debug:
        study_kwargs.update(
            study_name=f"{spec.name}_{args.target_mode}_cnn_bilstm_classifier",
            storage=f"sqlite:///{(MODELS_DIR / f'{spec.name}_{args.target_mode}_optuna.db').as_posix()}",
            load_if_exists=True,
        )
    study = optuna.create_study(**study_kwargs)
    finished_trials = sum(
        trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        for trial in study.trials
    )
    remaining_trials = max(0, target_trials - finished_trials)
    logger.info(
        "Optuna target: %s trials; finished: %s; remaining: %s.",
        target_trials,
        finished_trials,
        remaining_trials,
    )
    if remaining_trials:
        study.optimize(
            lambda trial: objective(trial, train_pool, spec, args.target_mode, cv_folds=5),
            n_trials=remaining_trials,
        )
    logger.info("Best CV F1-Macro: %.4f", study.best_value)
    logger.info("Best parameters: %s", study.best_params)
    return study


def prepare_datasets(train_pool, locked_test, spec, best_params):
    # This is kept if anything else needs it, but we won't use it for ensemble training anymore.
    train_engineered = apply_feature_engineering(train_pool, spec.kind)
    test_engineered = apply_feature_engineering(locked_test, spec.kind)
    preprocessor = DataPreprocessor(
        target_col=spec.target_col,
        oversample_method=best_params["oversample_method"],
        smote_ratio=best_params.get("smote_ratio", 1.0),
        resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
        oversampling_feature_columns=get_sequence_columns(spec.kind),
    )
    train_prepared = preprocessor.fit_transform(train_engineered)
    test_prepared = preprocessor.transform(test_engineered)

    selector = FeatureSelector(
        target_col=spec.target_col,
        use_feature_selection=True,
        required_features=get_sequence_columns(spec.kind),
    )
    train_selected = selector.fit_transform(
        train_prepared,
        preprocessor.numerical_cols,
        preprocessor.categorical_cols,
    )
    test_selected = selector.transform(test_prepared)

    train_dataset = StudentDataset(
        train_selected,
        spec.kind,
        spec.target_col,
        preprocessor.numerical_cols,
        preprocessor.categorical_cols,
    )
    test_dataset = StudentDataset(
        test_selected,
        spec.kind,
        spec.target_col,
        preprocessor.numerical_cols,
        preprocessor.categorical_cols,
    )
    cat_cardinalities = [
        len(preprocessor.label_encoders[column].classes_)
        for column in train_dataset.cat_cols
    ]
    return (
        preprocessor,
        selector,
        train_selected,
        test_selected,
        train_dataset,
        test_dataset,
        len(train_dataset.num_cols),
        cat_cardinalities,
    )


def train_seed_ensemble(
    spec,
    best_params,
    train_pool,
    locked_test,
    debug=False,
    seed_list: list[int] | None = None,
    ensemble_method: str = "mean_probability",
    seed_weights: dict[int, float] | None = None,
    calibration_policy: dict | None = None,
    threshold_policy: dict | None = None,
):
    """Fit every seed through the shared full-development estimator factory."""

    validate_resolved_config(best_params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = list(seed_list or (FIXED_SEEDS[:1] if debug else FIXED_SEEDS))
    all_probabilities = []
    probabilities_by_seed = {}
    last_model = None
    last_test_loader = None
    last_preprocessor = None
    last_train_selected = None

    for seed in seeds:
        logger.info("Training ensemble seed %s.", seed)
        fitted = fit_final_development_estimator(
            development_frame=train_pool,
            spec=spec,
            resolved_config=best_params,
            seed=int(seed),
        )
        seed_probabilities = predict_with_fitted_estimator(
            frame=locked_test,
            spec=spec,
            resolved_config=best_params,
            state_dict=fitted.refit_state_dict,
            preprocessor=fitted.preprocessor,
            selector=fitted.selector,
        )
        all_probabilities.append(seed_probabilities)
        probabilities_by_seed[int(seed)] = seed_probabilities
        test_engineered = apply_feature_engineering(locked_test.copy(), spec.kind)
        test_prepared = fitted.preprocessor.transform(test_engineered)
        test_selected = fitted.selector.transform(test_prepared)
        test_dataset = StudentDataset(
            test_selected,
            spec.kind,
            spec.target_col,
            fitted.preprocessor.numerical_cols,
            fitted.preprocessor.categorical_cols,
        )
        train_engineered = apply_feature_engineering(train_pool.copy(), spec.kind)
        train_prepared = fitted.preprocessor.transform(train_engineered)
        train_selected = fitted.selector.transform(train_prepared)
        last_model = fitted.model
        last_test_loader = DataLoader(
            test_dataset,
            batch_size=int(best_params["batch_size"]),
            shuffle=False,
        )
        last_preprocessor = fitted.preprocessor
        last_train_selected = train_selected

        model_path = MODELS_DIR / f"{spec.name}_3class_cnn_bilstm_classifier_seed{seed}.pt"
        torch.save(fitted.refit_state_dict, model_path)

    if probabilities_by_seed:
        mean_probabilities = combine_seed_probabilities(
            probabilities_by_seed,
            method=ensemble_method,
            seed_list=seeds,
            weights=seed_weights,
        )
    else:
        mean_probabilities = np.mean(np.asarray(all_probabilities), axis=0)
    mean_probabilities = apply_probability_calibration(mean_probabilities, calibration_policy)
    ensemble_predictions = apply_threshold_policy(mean_probabilities, threshold_policy)
    confidences = mean_probabilities.max(axis=1)
    return (
        np.asarray(ensemble_predictions, dtype=int),
        mean_probabilities,
        confidences,
        last_model,
        last_test_loader,
        device,
        last_preprocessor,
        last_train_selected
    )


def calculate_metrics(true_labels, predictions):
    return {
        "Accuracy": float(accuracy_score(true_labels, predictions)),
        "F1-Macro": float(f1_score(true_labels, predictions, average="macro")),
        "Precision-Macro": float(precision_score(true_labels, predictions, average="macro", zero_division=0)),
        "Recall-Macro": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
        "RMSE": float(np.sqrt(mean_squared_error(true_labels, predictions))),
        "R2": float(r2_score(true_labels, predictions)),
    }


def save_outputs(
    args,
    spec,
    study,
    best_params,
    locked_test,
    true_labels,
    predictions,
    probabilities,
    confidences,
    learning_paths,
    metrics,
):
    params_path = MODELS_DIR / f"{args.dataset}_{args.target_mode}_best_params.json"
    params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    metrics_path = METRICS_DIR / f"{args.dataset}_{args.target_mode}_locked_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    class_labels = list(range(probabilities.shape[1]))
    confusion_path = METRICS_DIR / f"{args.dataset}_{args.target_mode}_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(true_labels, predictions, labels=class_labels),
        index=[f"true_{label}" for label in class_labels],
        columns=[f"pred_{label}" for label in class_labels],
    ).to_csv(confusion_path)

    report_json_path = METRICS_DIR / f"{args.dataset}_{args.target_mode}_classification_report.json"
    report_json_path.write_text(
        json.dumps(
            classification_report(true_labels, predictions, zero_division=0, output_dict=True),
            indent=4,
        ),
        encoding="utf-8",
    )

    pr_rows = []
    true_array = np.asarray(true_labels)
    for class_index in class_labels:
        binary_true = (true_array == class_index).astype(int)
        precision_values, recall_values, thresholds = precision_recall_curve(
            binary_true,
            probabilities[:, class_index],
        )
        average_precision = float(average_precision_score(binary_true, probabilities[:, class_index]))
        padded_thresholds = list(thresholds) + [None]
        for precision_value, recall_value, threshold in zip(precision_values, recall_values, padded_thresholds):
            pr_rows.append(
                {
                    "class_label": class_index,
                    "precision": float(precision_value),
                    "recall": float(recall_value),
                    "threshold": None if threshold is None else float(threshold),
                    "average_precision": average_precision,
                }
            )
    pr_curve_path = METRICS_DIR / f"{args.dataset}_{args.target_mode}_precision_recall_curves.csv"
    pd.DataFrame(pr_rows).to_csv(pr_curve_path, index=False)

    predictions_frame = locked_test.reset_index(drop=True).copy()
    predictions_frame["True_Label"] = true_labels
    predictions_frame["Pred_Label"] = predictions
    predictions_frame["Confidence"] = confidences
    for class_index in range(probabilities.shape[1]):
        predictions_frame[f"Prob_Class_{class_index}"] = probabilities[:, class_index]
    predictions_path = PREDICTIONS_DIR / f"{args.dataset}_{args.target_mode}_predictions.csv"
    predictions_frame.to_csv(predictions_path, index=False)

    learning_path_path = RECOMMENDATIONS_DIR / f"{args.dataset}_{args.target_mode}_learning_paths.csv"
    learning_paths.to_csv(learning_path_path, index=False, encoding="utf-8-sig")

    report_path = REPORTS_DIR / f"{args.dataset}_{args.target_mode}_final_report.txt"
    report_path.write_text(
        "\n".join(
            [
                f"Dataset: {args.dataset}",
                f"Target Mode: {args.target_mode}",
                "Architecture: CNN-BiLSTM classifier",
                f"Loss: {describe_effective_loss(best_params)}",
                f"Optuna Best CV F1: {study.best_value:.4f}",
                f"Best Params: {json.dumps(best_params, indent=2)}",
                "",
                f"Final Locked Test F1-Macro: {metrics['F1-Macro']:.4f}",
                classification_report(true_labels, predictions, zero_division=0),
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Saved metrics, predictions, learning paths and report for %s.", args.dataset)


def main():
    args = parse_args()
    ensure_dirs()
    set_seed(DEFAULT_SEED)
    spec = DATASETS[args.dataset]
    selection_config = load_selection_config(getattr(args, "selection_config_json", None))
    logger.info("Starting approved thesis pipeline for %s.", args.dataset)
    if not selection_config and not args.debug:
        raise ValueError(
            "A frozen --selection-config-json is required for a non-debug final run. "
            "Run scripts/optimize_model_selection.py first; the locked test must not be "
            "opened by a command that is still running Optuna."
        )
    if args.skip_postgres and not args.debug:
        raise ValueError("--skip-postgres is debug-only; final runs must use PostgreSQL lineage persistence.")

    effective_dataset_version_id = args.dataset_version_id
    if effective_dataset_version_id is None and not args.run_id:
        raise ValueError("--dataset-version-id is required for a new DB-first run.")
    if not getattr(args, "allow_legacy_observed_evaluation", False):
        raise PermissionError(
            "The 79-record legacy_heldout_observed evaluation path is disabled. "
            "Phase A-B and all future model selection/calibration paths must use development records only."
        )

    run_id = args.run_id or str(uuid.uuid4())
    existing_run = load_experiment_run(run_id) if args.run_id else None
    if existing_run:
        raw_frame, dataset_version = load_dataset_version_from_postgres(
            args.dataset,
            int(existing_run["dataset_version_id"]),
        )
        verify_run_split_manifest(existing_run)
        train_pool, locked_test = reconstruct_existing_run_splits(
            raw_frame,
            run_id,
            args.dataset,
            args.target_mode,
        )
        logger.info("Loaded split membership for retry run %s from PostgreSQL.", run_id)
    else:
        raw_frame, dataset_version = load_dataset_version_from_postgres(
            args.dataset,
            effective_dataset_version_id,
        )
        train_pool, locked_test = create_locked_split_from_frame(
            raw_frame,
            args.dataset,
            args.target_mode,
        )
        logger.info(
            "Loaded dataset %s dataset_version_id=%s from PostgreSQL.",
            args.dataset,
            dataset_version["dataset_version_id"],
        )

    if selection_config:
        best_params = normalize_cnn_bilstm_classifier_params(selection_config["best_params"])
        study = LoadedStudy(float(selection_config.get("best_cv_f1_macro", selection_config.get("selected_strategy", {}).get("cv_f1_macro_mean", 0.0))), best_params)
        logger.info("Using validation-only selection config and skipping Optuna.")
    else:
        study = load_study(args, train_pool, spec)
        best_params = normalize_cnn_bilstm_classifier_params(study.best_params)

    selected_strategy = selection_config.get("selected_strategy", {}) if selection_config else {}
    strategy_metadata = resolve_frozen_strategy_metadata(selected_strategy, debug=args.debug)
    selected_seed_list = strategy_metadata["actual_seed_list"]
    selected_ensemble_method = strategy_metadata["probability_combination_method"]
    selected_calibration_policy = strategy_metadata["calibration_policy"]
    selected_threshold_policy = strategy_metadata["threshold_policy"]
    selected_seed_weights = selected_strategy.get("seed_weights")
    if selected_seed_weights:
        selected_seed_weights = {int(seed): float(weight) for seed, weight in selected_seed_weights.items()}

    storage_context = None
    if not args.skip_postgres:
        if existing_run:
            storage_context = {"run": existing_run}
        else:
            run_manifest = load_or_create_run_manifest(run_id)
            git_context = git_reproducibility_context(run_id)
            env_context = environment_lock_context()
            artifact_context = artifact_manifest_context(run_id, args.dataset, args.target_mode, best_params)
            train_config = {
                "architecture": "cnn_bilstm_classifier",
                "context_mlp_enabled": False,
                "sequence_columns": get_sequence_columns(spec.kind),
                "classifier_head": "linear",
                "model_selection_protocol": "nested_cv_or_validation_only",
                "selected_seed_list": selected_seed_list,
                "oversampling_policy": {
                    "method": best_params.get("oversample_method", "none"),
                    "smote_ratio": best_params.get("smote_ratio"),
                    "resampling_k_neighbors": best_params.get("resampling_k_neighbors"),
                },
                "target_mode": args.target_mode,
                "best_params": best_params,
                "fixed_seeds": selected_seed_list,
                "actual_seed_list": selected_seed_list,
                "debug": bool(args.debug),
                "selected_strategy_name": strategy_metadata["strategy_name"],
                "selected_ensemble_method": strategy_metadata["actual_aggregation_method"],
                "probability_combination_method": selected_ensemble_method,
                "actual_aggregation_method": strategy_metadata["actual_aggregation_method"],
                "selected_calibration_policy": selected_calibration_policy,
                "selected_threshold_policy": selected_threshold_policy,
                "decision_rule": strategy_metadata["decision_rule"],
                "selected_seed_weights": selected_seed_weights,
                "validation_only_selection": selection_config,
                "augmentation": {
                    "method": best_params.get("oversample_method", "none"),
                    "smote_ratio": best_params.get("smote_ratio"),
                    "resampling_k_neighbors": best_params.get("resampling_k_neighbors"),
                    "raw_train_records_before_oversampling": int(len(train_pool)),
                    "synthetic_samples_generated": None,
                    "note": "Synthetic samples are generated inside per-seed train folds and are not source records.",
                },
            }
            split_manifest_path = MANIFESTS_DIR / "splits" / f"{args.dataset}_{args.target_mode}_{run_id}.json"
            storage_context = prepare_storage_context(
                dataset_name=args.dataset,
                target_mode=args.target_mode,
                dataset_kind=spec.kind,
                target_col=spec.target_col,
                raw_path=None,
                csv_sep=spec.csv_sep,
                raw_frame=raw_frame,
                train_pool=train_pool,
                locked_test=locked_test,
                run_id=run_id,
                model_name="cnn_bilstm_classifier",
                train_config=train_config,
                artifact_uri=artifact_context["artifact_uri"],
                git_commit=git_context["git_commit"],
                working_tree_state=git_context["working_tree_state"],
                source_diff_uri=git_context["source_diff_uri"],
                source_diff_hash=git_context["source_diff_hash"],
                environment_lock_uri=env_context["environment_lock_uri"],
                environment_lock_hash=env_context["environment_lock_hash"],
                split_manifest_path=split_manifest_path,
                dataset_version=dataset_version,
                started_at=datetime.fromisoformat(run_manifest["started_at"]),
            )
            initialize_experiment_run_in_postgres(storage_context)

    # 2. Ensemble Training & Inference
    # Split train/val and preprocess PER seed to avoid SMOTE leakage
    (
        predictions,
        probabilities,
        confidences,
        best_model,
        test_loader,
        device,
        final_preprocessor,
        final_train_selected
    ) = train_seed_ensemble(
        spec,
        best_params,
        train_pool,
        locked_test,
        debug=args.debug,
        seed_list=selected_seed_list,
        ensemble_method=selected_ensemble_method,
        seed_weights=selected_seed_weights,
        calibration_policy=selected_calibration_policy,
        threshold_policy=selected_threshold_policy,
    )
    true_labels = locked_test[spec.target_col].astype(int).to_numpy()
    metrics = calculate_metrics(true_labels, predictions)
    logger.info("Locked-test F1-Macro: %.4f", metrics["F1-Macro"])

    # 5. Recommendation Paths
    learning_paths = generate_learning_path_report(
        original_features=locked_test,
        predictions=predictions,
        confidences=confidences,
        dataset_name=args.dataset,
    )
    save_outputs(
        args,
        spec,
        study,
        best_params,
        locked_test,
        true_labels,
        predictions,
        probabilities,
        confidences,
        learning_paths,
        metrics,
    )
    seq_cols = get_sequence_columns(spec.kind)
    context_exclusions = get_context_excluded_columns(spec.kind)
    num_cols = [
        c
        for c in final_preprocessor.numerical_cols
        if c in final_train_selected.columns and c not in seq_cols and c not in context_exclusions
    ]
    cat_cols = [
        c
        for c in final_preprocessor.categorical_cols
        if c in final_train_selected.columns and c not in seq_cols and c not in context_exclusions
    ]

    explain_model(
        best_model,
        test_loader,
        device,
        num_cols,
        cat_cols,
        EXPLANATIONS_DIR / f"{args.dataset}_{args.target_mode}_feature_importance.csv",
    )

    if args.skip_postgres:
        logger.warning("PostgreSQL persistence skipped by explicit command-line option.")
    else:
        run_id = persist_evaluation_to_postgres(
            dataset_name=args.dataset,
            model_name="cnn_bilstm_classifier",
            original_features=locked_test,
            true_labels=true_labels,
            predicted_labels=predictions,
            confidences=confidences,
            probabilities=probabilities,
            learning_paths=learning_paths,
            metrics=metrics,
            storage_context=storage_context,
        )
        logger.info("PostgreSQL run id: %s", run_id)


if __name__ == "__main__":
    main()
