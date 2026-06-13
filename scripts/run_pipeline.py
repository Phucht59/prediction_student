"""End-to-end thesis pipeline: CNN-BiLSTM + MLP, learning paths, PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATASETS,
    DEFAULT_SEED,
    EXPLANATIONS_DIR,
    FIXED_SEEDS,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    RAW_DIR,
    RECOMMENDATIONS_DIR,
    REPORTS_DIR,
    TrainingConfig,
    ensure_dirs,
)
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    apply_feature_engineering,
    create_and_save_locked_test,
    get_sequence_columns,
    load_splits,
)
from src.evaluation import persist_evaluation_to_postgres
from src.explainability import explain_model, generate_learning_path_report
from src.models import create_model
from src.train_pipeline import calculate_class_weights, objective, train_model
from src.utils import set_seed, setup_logger

logger = setup_logger("run_pipeline")


class LoadedStudy:
    def __init__(self, value: float, params: dict):
        self.best_value = value
        self.best_params = params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--target-mode", default="3class", choices=["3class"])
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--params-json", default=None, help="JSON file or JSON string used to skip Optuna")
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Development-only opt-out. Production thesis runs persist to PostgreSQL by default.",
    )
    return parser.parse_args()


def load_or_create_splits(dataset_name: str, target_mode: str):
    try:
        return load_splits(dataset_name, target_mode)
    except FileNotFoundError:
        spec = DATASETS[dataset_name]
        raw = pd.read_csv(RAW_DIR / spec.raw_file, sep=spec.csv_sep)
        create_and_save_locked_test(raw, dataset_name, target_mode)
        return load_splits(dataset_name, target_mode)


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
            study_name=f"{spec.name}_{args.target_mode}_cnn_bilstm_mlp",
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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(best_params["batch_size"])
    original_train_labels = train_pool[spec.target_col].astype(int).to_numpy()
    class_weights = calculate_class_weights(original_train_labels, num_classes=3).to(device)
    seeds = FIXED_SEEDS[:1] if debug else FIXED_SEEDS
    all_probabilities = []
    last_model = None
    last_test_loader = None
    last_preprocessor = None
    last_train_selected = None

    for seed in seeds:
        set_seed(seed)
        # 1. Tách validation trước khi resampling
        labels = train_pool[spec.target_col].astype(int).to_numpy()
        indices = np.arange(len(train_pool))
        train_indices, val_indices = train_test_split(
            indices,
            test_size=0.15,
            stratify=labels,
            random_state=seed,
        )
        
        train_sub = apply_feature_engineering(train_pool.iloc[train_indices].copy(), spec.kind)
        val_sub = apply_feature_engineering(train_pool.iloc[val_indices].copy(), spec.kind)
        test_engineered = apply_feature_engineering(locked_test.copy(), spec.kind)

        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=best_params["oversample_method"],
            smote_ratio=best_params.get("smote_ratio", 1.0),
            resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
        )
        train_prep = preprocessor.fit_transform(train_sub)
        val_prep = preprocessor.transform(val_sub)
        test_prep = preprocessor.transform(test_engineered)

        selector = FeatureSelector(
            target_col=spec.target_col,
            use_feature_selection=True,
            required_features=get_sequence_columns(spec.kind),
        )
        train_selected = selector.fit_transform(
            train_prep,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        val_selected = selector.transform(val_prep)
        test_selected = selector.transform(test_prep)

        train_ds = StudentDataset(train_selected, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
        val_ds = StudentDataset(val_selected, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
        test_ds = StudentDataset(test_selected, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_indices) > batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        cat_cardinalities = [len(preprocessor.label_encoders[col].classes_) for col in train_ds.cat_cols]
        num_numerical = len(train_ds.num_cols)

        from src.models import create_model, FocalLoss
        model = create_model(spec.kind, best_params, num_numerical, cat_cardinalities).to(device)
        if spec.kind == "xapi":
            criterion = nn.BCEWithLogitsLoss()
        else:
            if "focal_gamma" in best_params:
                criterion = FocalLoss(weight=class_weights, gamma=best_params["focal_gamma"])
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(best_params["learning_rate"]),
            weight_decay=float(best_params["weight_decay"]),
        )
        config = TrainingConfig(
            max_epochs=3 if debug else 100,
            patience=2 if debug else (25 if spec.kind == "xapi" else 15),
            scheduler_patience=1 if debug else (8 if spec.kind == "xapi" else 5),
        )
        logger.info("Training ensemble seed %s.", seed)
        model, _, _ = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            device,
        )

        model.eval()
        seed_probabilities = []
        with torch.no_grad():
            for seq_x, num_x, cat_x, _, _ in test_loader:
                probabilities = model.predict_proba(
                    seq_x.to(device),
                    num_x.to(device),
                    cat_x.to(device),
                )
                seed_probabilities.extend(probabilities.cpu().numpy())
        seed_probabilities = np.asarray(seed_probabilities)
        all_probabilities.append(seed_probabilities)
        last_model = model
        last_test_loader = test_loader
        last_preprocessor = preprocessor
        last_train_selected = train_selected

        model_path = MODELS_DIR / f"{spec.name}_3class_cnn_bilstm_mlp_seed{seed}.pt"
        torch.save(model.state_dict(), model_path)

    mean_probabilities = np.mean(np.asarray(all_probabilities), axis=0)
    ensemble_predictions = np.argmax(mean_probabilities, axis=1)
    confidences = mean_probabilities[np.arange(len(ensemble_predictions)), ensemble_predictions]
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
                "Architecture: CNN-BiLSTM + Context MLP",
                "Loss: Weighted CrossEntropyLoss",
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
    logger.info("Starting approved thesis pipeline for %s.", args.dataset)

    train_pool, locked_test = load_or_create_splits(args.dataset, args.target_mode)
    study = load_study(args, train_pool, spec)
    best_params = dict(study.best_params)

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
    )
    true_labels = locked_test[spec.target_col].astype(int).to_numpy()
    metrics = calculate_metrics(true_labels, predictions)
    logger.info("Locked-test F1-Macro: %.4f", metrics["F1-Macro"])

    # 5. Recommendation Paths
    learning_paths = generate_learning_path_report(
        original_features=locked_test,
        predictions=predictions,
        confidences=confidences,
        dataset_kind=spec.kind,
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
    num_cols = [c for c in final_preprocessor.numerical_cols if c in final_train_selected.columns and c not in seq_cols]
    cat_cols = [c for c in final_preprocessor.categorical_cols if c in final_train_selected.columns and c not in seq_cols]

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
            model_name="cnn_bilstm_mlp_ensemble",
            original_features=locked_test,
            true_labels=true_labels,
            predicted_labels=predictions,
            confidences=confidences,
            probabilities=probabilities,
            learning_paths=learning_paths,
            metrics=metrics,
        )
        logger.info("PostgreSQL run id: %s", run_id)


if __name__ == "__main__":
    main()
