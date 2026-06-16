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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, accuracy_score
import optuna

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATASETS, RAW_DIR, MODELS_DIR, DEFAULT_SEED, ensure_dirs
from src.data_pipeline import (
    load_splits,
    create_and_save_locked_test,
    apply_feature_engineering,
    get_sequence_columns,
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
)
from src.train_v27_pipeline import create_model_v27, train_model_v27, validate_epoch_v27
from src.losses_v27 import ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss
from src.utils import set_seed, setup_logger

logger = setup_logger("tune_v27_thresholds")

def get_predictions_with_thresholds(probs, threshold_low, class_multipliers):
    adj_probs = probs * np.array(class_multipliers)
    is_class_0 = adj_probs[:, 0] >= threshold_low
    preds_other = np.argmax(adj_probs[:, 1:], axis=1) + 1
    preds = np.where(is_class_0, 0, preds_other)
    return preds

def tune_dataset_thresholds(dataset_name):
    spec = DATASETS[dataset_name]
    ensure_dirs()
    
    # 1. Load best params
    best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
    if not best_params_path.exists():
        logger.error(f"Best parameters file not found at {best_params_path}. Please run run_v27_optuna.py first.")
        sys.exit(1)
        
    with open(best_params_path, "r", encoding="utf-8") as f:
        best_params = json.load(f)
    logger.info(f"Loaded best parameters for {dataset_name}: {best_params}")
    
    # Load splits
    train_pool, locked_test = load_splits(dataset_name, "3class")
    logger.info(f"Loaded splits. Train Pool: {len(train_pool)}")
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    
    oof_probs = np.zeros((len(train_pool), 3))
    oof_targets = np.zeros(len(train_pool), dtype=int)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels)):
        logger.info(f"--- FOLD {fold + 1} for out-of-fold prediction ---")
        set_seed(DEFAULT_SEED + fold)
        
        train_fold = train_pool.iloc[train_idx].copy()
        val_fold = train_pool.iloc[val_idx].copy()
        
        # Apply feature engineering
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        val_fold = apply_feature_engineering(val_fold, spec.kind)
        
        # DataPreprocessor fit-transform on train subset without oversampling
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=best_params.get("oversample_method", "none"),
            smote_ratio=best_params.get("smote_ratio", 1.0),
            resampling_k_neighbors=5,
        )
        train_prep = preprocessor.fit_transform(train_fold, apply_oversampling=False)
        val_prep = preprocessor.transform(val_fold)
        
        # Feature Selector fit-transform on train subset
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
        val_selected = selector.transform(val_prep)
        
        # Apply oversampling ONLY on selected features of the train fold
        train_resampled = preprocessor.apply_oversampling(train_selected)
        
        # Create StudentDataset
        train_ds = StudentDataset(
            train_resampled,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        val_ds = StudentDataset(
            val_selected,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        
        batch_size = int(best_params.get("batch_size", 32))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        num_numerical = len(train_ds.num_cols)
        cat_cardinalities = [
            len(preprocessor.label_encoders[col].classes_)
            for col in train_ds.cat_cols
        ]
        
        model = create_model_v27(spec.kind, best_params, num_numerical, cat_cardinalities).to(device)
        
        # Loss function
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
        
        # Get out-of-fold validation probabilities
        model.eval()
        fold_probs = []
        fold_targets = []
        with torch.no_grad():
            for batch in val_loader:
                seq_x, num_x, cat_x, labels, _, _ = batch
                probs = model.predict_proba(seq_x.to(device), num_x.to(device), cat_x.to(device))
                fold_probs.extend(probs.cpu().numpy())
                fold_targets.extend(labels.numpy())
                
        oof_probs[val_idx] = np.array(fold_probs)
        oof_targets[val_idx] = np.array(fold_targets)
        
    # 2. Evaluate before threshold tuning (standard argmax)
    raw_preds = np.argmax(oof_probs, axis=1)
    raw_f1 = f1_score(oof_targets, raw_preds, average="macro", zero_division=0)
    raw_recall_low = recall_score(oof_targets, raw_preds, labels=[0], average="macro", zero_division=0)
    raw_combined = 0.5 * raw_f1 + 0.5 * raw_recall_low
    
    logger.info(f"OOF Raw Metrics - F1-Macro: {raw_f1:.4f}, Recall-Low: {raw_recall_low:.4f}, Combined: {raw_combined:.4f}")
    
    # 3. Search for threshold_low and class_multipliers to maximize 0.5*F1_Macro + 0.5*Recall_Low
    logger.info("Searching for optimal thresholds and class multipliers...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def oof_objective(trial):
        threshold_low = trial.suggest_float("threshold_low", 0.0, 1.0)
        m0 = trial.suggest_float("m0", 0.1, 5.0)
        m1 = trial.suggest_float("m1", 0.1, 5.0)
        m2 = trial.suggest_float("m2", 0.1, 5.0)
        
        preds = get_predictions_with_thresholds(oof_probs, threshold_low, [m0, m1, m2])
        f1 = f1_score(oof_targets, preds, average="macro", zero_division=0)
        recall_low = recall_score(oof_targets, preds, labels=[0], average="macro", zero_division=0)
        return 0.5 * f1 + 0.5 * recall_low

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=DEFAULT_SEED))
    study.optimize(oof_objective, n_trials=2000)
    
    best_thresholds = study.best_params
    best_threshold_low = best_thresholds["threshold_low"]
    best_multipliers = [best_thresholds["m0"], best_thresholds["m1"], best_thresholds["m2"]]
    best_val = study.best_value
    
    # Evaluate after threshold tuning
    tuned_preds = get_predictions_with_thresholds(oof_probs, best_threshold_low, best_multipliers)
    tuned_f1 = f1_score(oof_targets, tuned_preds, average="macro", zero_division=0)
    tuned_recall_low = recall_score(oof_targets, tuned_preds, labels=[0], average="macro", zero_division=0)
    tuned_combined = 0.5 * tuned_f1 + 0.5 * tuned_recall_low
    
    logger.info(f"OOF Tuned Metrics - F1-Macro: {tuned_f1:.4f}, Recall-Low: {tuned_recall_low:.4f}, Combined: {tuned_combined:.4f}")
    logger.info(f"Best Threshold Low: {best_threshold_low:.4f}, Best Multipliers: {best_multipliers}")
    
    # Save the tuned thresholds
    output_dir = Path("outputs/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = output_dir / f"thresholds_{dataset_name}.json"
    
    threshold_data = {
        "threshold_low": float(best_threshold_low),
        "class_multipliers": [float(m) for m in best_multipliers]
    }
    
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Saved tuned thresholds to {threshold_path}")
    
    return {
        "dataset": dataset_name,
        "raw": {
            "f1_macro": raw_f1,
            "recall_low": raw_recall_low,
            "combined": raw_combined
        },
        "tuned": {
            "f1_macro": tuned_f1,
            "recall_low": tuned_recall_low,
            "combined": tuned_combined,
            "threshold_low": best_threshold_low,
            "class_multipliers": best_multipliers
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Tune decision thresholds for class 0 (Low) to maximize Combined Score")
    parser.add_argument("--dataset", choices=["student-mat", "student-por", "xapi"], help="Optional dataset name. If not set, tunes all three.")
    args = parser.parse_args()
    
    if args.dataset:
        datasets_to_tune = [args.dataset]
    else:
        datasets_to_tune = ["student-mat", "student-por", "xapi"]
        
    results = {}
    for dataset_name in datasets_to_tune:
        logger.info(f"=== TUNING THRESHOLDS FOR {dataset_name} ===")
        res = tune_dataset_thresholds(dataset_name)
        results[dataset_name] = res
        logger.info(f"=== COMPLETED TUNING FOR {dataset_name} ===\n")
        
    # Write summary report to stdout/logs
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
