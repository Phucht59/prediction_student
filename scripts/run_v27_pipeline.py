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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

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

logger = setup_logger("run_v27_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Run training pipeline V27 with JointHybridLoss")
    parser.add_argument("--dataset", choices=["student-mat", "student-por", "xapi"], required=True, help="Dataset name")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    spec = DATASETS[dataset_name]
    
    # Ensure raw directory and output directory exist
    ensure_dirs()
    
    # 1. Recreate splits to ensure G3_raw is included
    logger.info(f"Recreating splits for {dataset_name} to ensure G3_raw is preserved in CSV files...")
    raw = pd.read_csv(RAW_DIR / spec.raw_file, sep=spec.csv_sep)
    create_and_save_locked_test(raw, dataset_name, "3class")
    
    # Load splits
    train_pool, locked_test = load_splits(dataset_name, "3class")
    logger.info(f"Loaded splits. Train Pool: {len(train_pool)}, Locked Test: {len(locked_test)}")
    
    # Load best params
    best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
    if best_params_path.exists():
        with open(best_params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
        logger.info(f"Loaded best parameters from {best_params_path}")
    else:
        best_params = {
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 32,
            "oversample_method": "smote",
            "smote_ratio": 1.0,
            "cnn_channels": 32,
            "lstm_hidden_dim": 64,
            "context_hidden_dim": 64,
            "fusion_hidden_dim": 64,
            "dropout": 0.3,
        }
        logger.info("Using default parameters")
        
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels)):
        logger.info(f"=== FOLD {fold + 1} ===")
        set_seed(DEFAULT_SEED + fold)
        
        train_fold = train_pool.iloc[train_idx].copy()
        val_fold = train_pool.iloc[val_idx].copy()
        
        # Apply feature engineering
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        val_fold = apply_feature_engineering(val_fold, spec.kind)
        
        # Step 1: Preprocessor fit-transform on train subset without oversampling
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=best_params.get("oversample_method", "none"),
            smote_ratio=best_params.get("smote_ratio", 1.0),
            resampling_k_neighbors=best_params.get("resampling_k_neighbors", 5),
        )
        train_prep = preprocessor.fit_transform(train_fold, apply_oversampling=False)
        val_prep = preprocessor.transform(val_fold)
        
        # Step 2: Feature Selector fit-transform on train subset
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
        
        # Step 3: Apply oversampling ONLY on selected features of the train fold
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
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model_v27(spec.kind, best_params, num_numerical, cat_cardinalities).to(device)
        
        # Loss function with ClassBalancedFocalLoss, OrdinalLoss and regression loss
        # Use class frequencies from original training fold (prior to oversampling)
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
        
        # Setup simple calibration and early stopping
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
        
        # Compute final evaluation metrics
        val_loss, val_f1, val_acc = validate_epoch_v27(model, val_loader, criterion, device)
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                seq_x, num_x, cat_x, labels, _, _ = batch
                outputs = model(seq_x.to(device), num_x.to(device), cat_x.to(device))
                preds = torch.argmax(outputs[0], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        
        logger.info(f"Fold {fold+1} metrics - F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        fold_metrics.append({
            "fold": fold + 1,
            "f1_macro": val_f1,
            "accuracy": val_acc,
            "precision_macro": precision,
            "recall_macro": recall
        })
        
    avg_f1 = float(np.mean([m["f1_macro"] for m in fold_metrics]))
    avg_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
    avg_prec = float(np.mean([m["precision_macro"] for m in fold_metrics]))
    avg_rec = float(np.mean([m["recall_macro"] for m in fold_metrics]))
    
    results = {
        "dataset": dataset_name,
        "avg_metrics": {
            "accuracy": avg_acc,
            "precision_macro": avg_prec,
            "recall_macro": avg_rec,
            "f1_macro": avg_f1
        },
        "fold_metrics": fold_metrics
    }
    
    output_dir = Path("outputs/v27") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"=== COMPLETED TRAINING V27 FOR {dataset_name} ===")
    logger.info(f"Avg F1-Macro: {avg_f1:.4f}, Avg Accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
