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

logger = setup_logger("run_v27_optuna")

def objective(trial, dataset_name, spec, train_pool):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    oversample_method = trial.suggest_categorical("oversample_method", ["none", "smote", "adasyn"])
    smote_ratio = trial.suggest_float("smote_ratio", 0.5, 1.5)
    cnn_channels = trial.suggest_categorical("cnn_channels", [16, 32, 64])
    lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [32, 64, 128])
    context_hidden_dim = trial.suggest_categorical("context_hidden_dim", [32, 64, 128])
    fusion_hidden_dim = trial.suggest_categorical("fusion_hidden_dim", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    params = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "oversample_method": oversample_method,
        "smote_ratio": smote_ratio,
        "cnn_channels": cnn_channels,
        "lstm_hidden_dim": lstm_hidden_dim,
        "context_hidden_dim": context_hidden_dim,
        "fusion_hidden_dim": fusion_hidden_dim,
        "dropout": dropout,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=DEFAULT_SEED)
    labels = train_pool[spec.target_col].astype(int).to_numpy()
    
    fold_f1s = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_pool, labels)):
        set_seed(DEFAULT_SEED + fold)
        
        train_fold = train_pool.iloc[train_idx].copy()
        val_fold = train_pool.iloc[val_idx].copy()
        
        # Feature engineering
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        val_fold = apply_feature_engineering(val_fold, spec.kind)
        
        # DataPreprocessor fit-transform on train (without oversampling)
        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=params["oversample_method"],
            smote_ratio=params["smote_ratio"],
            resampling_k_neighbors=5,
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
        
        # Oversampling ONLY on train selected
        train_resampled = preprocessor.apply_oversampling(train_selected)
        
        # StudentDataset
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
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        num_numerical = len(train_ds.num_cols)
        cat_cardinalities = [
            len(preprocessor.label_encoders[col].classes_)
            for col in train_ds.cat_cols
        ]
        
        model = create_model_v27(spec.kind, params, num_numerical, cat_cardinalities).to(device)
        
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
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"]
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
        
        _, val_f1, _ = validate_epoch_v27(model, val_loader, criterion, device)
        fold_f1s.append(val_f1)
        
    return float(np.mean(fold_f1s))

def main():
    parser = argparse.ArgumentParser(description="Tune StudentHybridV27 hyperparameters using Optuna")
    parser.add_argument("--dataset", choices=["student-mat", "student-por", "xapi"], required=True, help="Dataset name")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    
    dataset_name = args.dataset
    n_trials = args.n_trials
    
    spec = DATASETS[dataset_name]
    ensure_dirs()
    
    logger.info(f"Loading data for {dataset_name}...")
    train_pool, locked_test = load_splits(dataset_name, "3class")
    
    logger.info(f"Starting Optuna study for {dataset_name} with {n_trials} trials...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=DEFAULT_SEED))
    study.optimize(lambda trial: objective(trial, dataset_name, spec, train_pool), n_trials=n_trials)
    
    logger.info(f"Best trial: F1-Macro = {study.best_value:.4f}")
    logger.info(f"Best parameters: {study.best_params}")
    
    best_params_path = MODELS_DIR / f"{dataset_name}_3class_best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Saved best parameters to {best_params_path}")

if __name__ == "__main__":
    main()
