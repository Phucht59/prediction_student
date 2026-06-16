import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, MODELS_DIR, ensure_dirs
from src.data_pipeline import (
    load_splits,
    apply_feature_engineering,
    get_sequence_columns,
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
)
from src.train_v27_pipeline import create_model_v27, train_model_v27
from src.losses_v27 import ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss
from src.utils import set_seed, setup_logger

logger = setup_logger("run_v27_ensemble")


def get_predictions_with_thresholds(probs, threshold_low, class_multipliers):
    adj_probs = probs * np.array(class_multipliers)
    is_class_0 = adj_probs[:, 0] >= threshold_low
    preds_other = np.argmax(adj_probs[:, 1:], axis=1) + 1
    preds = np.where(is_class_0, 0, preds_other)
    return preds


def run_ensemble_for_dataset(dataset_name: str):
    spec = DATASETS[dataset_name]
    logger.info(f"=== STARTING ENSEMBLE FOR {dataset_name} ===")

    # 1. Load best params
    best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
    if not best_params_path.exists():
        logger.error(f"Best parameters file not found at {best_params_path}. Please check.")
        return

    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    logger.info(f"Loaded best parameters: {best_params}")

    # 2. Load decision thresholds
    threshold_path = Path("outputs/experiments") / f"thresholds_{dataset_name}.json"
    if not threshold_path.exists():
        logger.error(f"Threshold file not found at {threshold_path}. Please check.")
        return

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    threshold_low = threshold_data["threshold_low"]
    class_multipliers = threshold_data["class_multipliers"]
    logger.info(f"Loaded thresholds: threshold_low={threshold_low:.4f}, multipliers={class_multipliers}")

    # 3. Load splits
    train_pool, locked_test = load_splits(dataset_name, "3class")
    logger.info(f"Loaded splits. Train Pool: {len(train_pool)}, Locked Test: {len(locked_test)}")

    seeds = [42, 43, 44, 45, 46]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(best_params.get("batch_size", 32))

    all_seed_probs = []
    all_seed_regs = []

    # Ground truth targets for evaluation
    # We will build test_ds using the preprocessor and selector from each seed
    for seed in seeds:
        logger.info(f"--- Training Seed {seed} ---")
        set_seed(seed)

        # Copy data
        train_fold = train_pool.copy()
        test_fold = locked_test.copy()

        # Apply feature engineering
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        test_fold = apply_feature_engineering(test_fold, spec.kind)

        # Fit preprocessor on train pool without oversampling
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=best_params.get("oversample_method", "none"),
            smote_ratio=best_params.get("smote_ratio", 1.0),
            resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
        )
        train_prep = preprocessor.fit_transform(train_fold, apply_oversampling=False)
        test_prep = preprocessor.transform(test_fold)

        # Feature selection
        sequence_columns = get_sequence_columns(spec.kind)
        selector = FeatureSelector(
            target_col=spec.target_col,
            use_feature_selection=True,
            required_features=sequence_columns,
        )
        train_selected = selector.fit_transform(
            train_prep,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        test_selected = selector.transform(test_prep)

        # Oversample only on train pool
        train_resampled = preprocessor.apply_oversampling(train_selected)

        # Create datasets
        train_ds = StudentDataset(
            train_resampled,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        test_ds = StudentDataset(
            test_selected,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        # We pass train_ds itself (shuffle=False) as val_loader to fit train_model_v27 interface
        val_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        num_numerical = len(train_ds.num_cols)
        cat_cardinalities = [
            len(preprocessor.label_encoders[col].classes_)
            for col in train_ds.cat_cols
        ]

        model = create_model_v27(spec.kind, best_params, num_numerical, cat_cardinalities).to(device)

        # Setup Loss
        original_train_labels = train_fold[spec.target_col].astype(int).to_numpy()
        class_counts = np.bincount(original_train_labels, minlength=3)
        class_loss_fn = ClassBalancedFocalLoss(class_counts=class_counts, beta=0.99, gamma=2.0)
        ordinal_loss_fn = OrdinalLoss()
        regression_loss_fn = nn.MSELoss()

        w_reg = 0.0 if spec.kind == "xapi" else 1.0
        criterion = JointHybridLoss(
            class_loss_fn=class_loss_fn,
            ordinal_loss_fn=ordinal_loss_fn,
            regression_loss_fn=regression_loss_fn,
            w_class=1.0,
            w_ord=1.0,
            w_reg=w_reg
        )

        optimizer = optim.Adam(
            model.parameters(),
            lr=best_params.get("learning_rate", 0.001),
            weight_decay=best_params.get("weight_decay", 0.0001)
        )

        class TrainingConfigV27:
            max_epochs = 40
            patience = 10
            scheduler_patience = 4
            scheduler_factor = 0.5

        config = TrainingConfigV27()

        model, history, best_score = train_model_v27(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            device
        )

        # Inference on locked test
        model.eval()
        seed_probs = []
        seed_regs = []
        with torch.no_grad():
            for batch in test_loader:
                seq_x, num_x, cat_x, _, _, _ = batch
                class_logits, _, reg_logits = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
                probs = torch.softmax(class_logits, dim=1).cpu().numpy()
                seed_probs.extend(probs)
                if spec.kind != "xapi":
                    seed_regs.extend(reg_logits.cpu().numpy())

        all_seed_probs.append(seed_probs)
        if spec.kind != "xapi":
            all_seed_regs.append(seed_regs)

    # 4. Average probabilities and regression outputs
    mean_probs = np.mean(all_seed_probs, axis=0)
    
    # Get ground truth from locked_test
    # Need to map target column to encoded label space using target_encoder of the last preprocessor (they are identical)
    y_true = preprocessor.target_encoder.transform(locked_test[spec.target_col])
    
    # Apply tuned decision thresholds
    preds = get_predictions_with_thresholds(mean_probs, threshold_low, class_multipliers)

    # 5. Evaluate Metrics
    acc = accuracy_score(y_true, preds)
    prec_macro = precision_score(y_true, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, preds, average="macro", zero_division=0)
    
    f1_classes = f1_score(y_true, preds, average=None, zero_division=0)
    f1_class_0 = float(f1_classes[0]) if len(f1_classes) > 0 else 0.0
    f1_class_1 = float(f1_classes[1]) if len(f1_classes) > 1 else 0.0
    f1_class_2 = float(f1_classes[2]) if len(f1_classes) > 2 else 0.0

    # Recall for class 0 (Low group)
    recalls = recall_score(y_true, preds, average=None, zero_division=0)
    recall_low = float(recalls[0]) if len(recalls) > 0 else 0.0

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "f1_class_0": f1_class_0,
        "f1_class_1": f1_class_1,
        "f1_class_2": f1_class_2,
        "recall_low": recall_low,
    }

    if spec.kind != "xapi":
        mean_regs = np.mean(all_seed_regs, axis=0)
        y_true_reg = locked_test["G3_raw"].values.astype(np.float32)
        rmse = np.sqrt(mean_squared_error(y_true_reg, mean_regs))
        r2 = r2_score(y_true_reg, mean_regs)
        metrics["rmse"] = float(rmse)
        metrics["r2"] = float(r2)

    logger.info(f"Ensemble metrics for {dataset_name}: {metrics}")

    # 6. Save metrics
    output_dir = Path("outputs/v27") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "ensemble_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved ensemble metrics to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Run Seed Ensembling V27")
    parser.add_argument("--dataset", choices=["student-mat", "student-por", "xapi"], help="Dataset name. If not set, runs all.")
    args = parser.parse_args()

    ensure_dirs()

    if args.dataset:
        run_ensemble_for_dataset(args.dataset)
    else:
        for ds in ["student-mat", "student-por", "xapi"]:
            run_ensemble_for_dataset(ds)


if __name__ == "__main__":
    main()
