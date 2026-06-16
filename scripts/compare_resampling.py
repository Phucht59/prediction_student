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
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC

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
from src.train_v27_pipeline import create_model_v27, train_model_v27
from src.losses_v27 import ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss
from src.utils import set_seed, setup_logger

logger = setup_logger("compare_resampling")


def run_resampling_experiment(dataset_name: str, method: str, best_params: dict, device: torch.device):
    spec = DATASETS[dataset_name]
    train_pool, _ = load_splits(dataset_name, "3class")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    
    fold_f1s = []
    fold_recalls = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels)):
        set_seed(DEFAULT_SEED + fold)
        
        train_fold = train_pool.iloc[train_idx].copy()
        val_fold = train_pool.iloc[val_idx].copy()
        
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        val_fold = apply_feature_engineering(val_fold, spec.kind)
        
        # Preprocess without oversampling
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method="none",
        )
        train_prep = preprocessor.fit_transform(train_fold, apply_oversampling=False)
        val_prep = preprocessor.transform(val_fold)
        
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
        val_selected = selector.transform(val_prep)
        
        # Perform custom resampling for comparison
        df_resampled = train_selected.copy()
        if method.lower() != "none":
            X = df_resampled.drop(columns=[spec.target_col])
            y_encoded = df_resampled[spec.target_col]
            remaining_cat_cols = [col for col in preprocessor.categorical_cols if col in X.columns]
            
            class_counts = pd.Series(y_encoded).value_counts()
            majority_count = class_counts.max()
            effective_k_neighbors = min(
                5,
                max(1, int(class_counts.min()) - 1),
            )
            strategy = {}
            for cls, count in class_counts.items():
                if count == majority_count:
                    strategy[cls] = count
                else:
                    target = int(majority_count * best_params.get("smote_ratio", 1.0))
                    strategy[cls] = max(count, target)
            
            if method.lower() == "random_oversampling":
                sampler = RandomOverSampler(sampling_strategy=strategy, random_state=42)
            elif method.lower() == "smotenc":
                cat_indices = [X.columns.get_loc(c) for c in remaining_cat_cols] if remaining_cat_cols else []
                if cat_indices:
                    sampler = SMOTENC(categorical_features=cat_indices, sampling_strategy=strategy, random_state=42, k_neighbors=effective_k_neighbors)
                else:
                    sampler = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=effective_k_neighbors)
            elif method.lower() == "smote":
                if remaining_cat_cols:
                    cat_indices = [X.columns.get_loc(c) for c in remaining_cat_cols]
                    sampler = SMOTENC(categorical_features=cat_indices, sampling_strategy=strategy, random_state=42, k_neighbors=effective_k_neighbors)
                else:
                    sampler = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=effective_k_neighbors)
            else:
                raise ValueError(f"Unknown method {method}")
                
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y_encoded)
                X = pd.DataFrame(X_resampled, columns=X.columns)
                # Ensure resampled categoricals are rounded and cast to int
                for col in remaining_cat_cols:
                    X[col] = X[col].round().astype(int)
                df_resampled = X.copy()
                df_resampled[spec.target_col] = y_resampled
            except Exception as e:
                logger.warning(f"Resampling failed: {e}. Proceeding without resampling.")
        
        # Create Dataset
        train_ds = StudentDataset(
            df_resampled,
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
        
        # Criterion
        original_train_labels = train_fold[spec.target_col].astype(int).to_numpy()
        counts = np.bincount(original_train_labels, minlength=3)
        class_loss_fn = ClassBalancedFocalLoss(class_counts=counts, beta=0.99, gamma=2.0)
        ordinal_loss_fn = OrdinalLoss()
        regression_loss_fn = nn.MSELoss()
        
        criterion = JointHybridLoss(
            class_loss_fn=class_loss_fn,
            ordinal_loss_fn=ordinal_loss_fn,
            regression_loss_fn=regression_loss_fn,
            w_class=1.0,
            w_ord=1.0,
            w_reg=1.0
        )
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=best_params.get("learning_rate", 0.001),
            weight_decay=best_params.get("weight_decay", 0.0001)
        )
        
        class TrainingConfigV27:
            max_epochs = 15  # 15 epochs for quick comparison
            patience = 5
            scheduler_patience = 2
            scheduler_factor = 0.5
            
        config = TrainingConfigV27()
        
        model, _, _ = train_model_v27(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            device
        )
        
        # Evaluate on validation
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
                
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        recalls = recall_score(all_labels, all_preds, average=None, zero_division=0)
        recall_low = recalls[0] if len(recalls) > 0 else 0.0
        
        fold_f1s.append(f1)
        fold_recalls.append(recall_low)
        
    avg_f1 = np.mean(fold_f1s)
    avg_recall_low = np.mean(fold_recalls)
    return float(avg_f1), float(avg_recall_low)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = ["student-mat", "student-por"]
    methods = ["None", "SMOTENC", "random_oversampling"]
    
    results = []
    
    for dataset_name in datasets:
        logger.info(f"=== Experimenting on {dataset_name} ===")
        # Load best params
        best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
        if best_params_path.exists():
            with open(best_params_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)
        else:
            best_params = {}
            
        for method in methods:
            logger.info(f"Running method: {method}...")
            f1, recall_low = run_resampling_experiment(dataset_name, method, best_params, device)
            logger.info(f"Dataset: {dataset_name}, Method: {method} -> Macro F1: {f1:.4f}, Recall Low: {recall_low:.4f}")
            results.append({
                "dataset": dataset_name,
                "method": method,
                "macro_f1": f1,
                "recall_low": recall_low
            })
            
    # Save to outputs/experiments/resampling_comparison.csv
    df_results = pd.DataFrame(results)
    output_dir = Path("outputs/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "resampling_comparison.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
