import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score
from src.config import *
from src.utils import setup_logger, set_seed
from src.data_pipeline import load_splits, apply_feature_engineering, V26FeatureSelector, V26Preprocessor, V27Dataset
from src.models import create_v27_model, HybridLoss

logger = setup_logger("train_pipeline")

import torch
import copy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


logger = setup_logger("v27_train")

class EarlyStopping:
    def __init__(self, patience=15, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_metric, model):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for seq_x, num_x, cat_x, labels, _ in dataloader:
        seq_x = seq_x.to(device) if seq_x is not None else None
        num_x = num_x.to(device) if num_x is not None else None
        cat_x = cat_x.to(device) if cat_x is not None else None
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, expected_val = model(seq_x, num_x, cat_x)
        
        loss, _, _ = criterion(logits, expected_val, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seq_x, num_x, cat_x, labels, _ in dataloader:
            seq_x = seq_x.to(device) if seq_x is not None else None
            num_x = num_x.to(device) if num_x is not None else None
            cat_x = cat_x.to(device) if cat_x is not None else None
            labels = labels.to(device)
            
            logits, expected_val = model(seq_x, num_x, cat_x)
            loss, _, _ = criterion(logits, expected_val, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = total_loss / len(dataloader)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    
    return val_loss, f1_macro, acc

def train_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    early_stopping = EarlyStopping(patience=config.patience)
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    
    for epoch in range(config.max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)
        
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    model.load_state_dict(early_stopping.best_state)
    return model, history, early_stopping.best_score


import optuna
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.preprocessing import V26Preprocessor
from src.feature_engineering import apply_feature_engineering
from src.feature_selection import V26FeatureSelector
from src.dataset import V27Dataset


from .model_factory import create_model
from .losses import HybridLoss
from .train import train_model
from src.config import TrainingConfig

logger = setup_logger("v27_optuna")

def calculate_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights = total / (num_classes * (counts + 1e-6))
    return torch.tensor(weights, dtype=torch.float32)

def objective(trial, df_train_pool: pd.DataFrame, spec, target_mode: str, cv_folds: int):
    # Common Hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    
    # Loss Config
    lambda_ordinal = trial.suggest_float("lambda_ordinal", 0.0, 0.5) # allow 0
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    oversample_method = trial.suggest_categorical("oversample_method", ["none", "smote", "adasyn"])
    
    config_dict = {
        "dropout": dropout
    }
    
    # Dataset specific Architecture config
    if spec.kind == "student":
        config_dict["context_architecture"] = trial.suggest_categorical("context_architecture", ["mlp_baseline", "deepfm", "dcnv2"])
        config_dict["sequence_architecture"] = trial.suggest_categorical("sequence_architecture", ["none", "dsc_bilstm_attention", "bilstm_attention", "self_attention_small"])
        config_dict["fm_embedding_dim"] = trial.suggest_categorical("fm_embedding_dim", [8, 16, 32])
        config_dict["dcn_cross_layers"] = trial.suggest_int("dcn_cross_layers", 1, 3)
        config_dict["context_hidden_dim"] = trial.suggest_categorical("context_hidden_dim", [32, 64, 128])
        config_dict["sequence_hidden_dim"] = trial.suggest_categorical("sequence_hidden_dim", [32, 64])
        config_dict["fusion_hidden_dim"] = trial.suggest_categorical("fusion_hidden_dim", [64, 128])
    else: # xapi
        config_dict["architecture"] = trial.suggest_categorical("architecture", ["ft_transformer", "deepfm", "dcnv2"])
        config_dict["d_token"] = trial.suggest_categorical("d_token", [16, 32, 64])
        config_dict["n_heads"] = trial.suggest_categorical("n_heads", [2, 4, 8])
        config_dict["n_layers"] = trial.suggest_int("n_layers", 1, 3)
        config_dict["ff_hidden_dim"] = trial.suggest_categorical("ff_hidden_dim", [64, 128])
        config_dict["attention_dropout"] = trial.suggest_float("attention_dropout", 0.1, 0.4)
        config_dict["residual_dropout"] = trial.suggest_float("residual_dropout", 0.1, 0.4)
        config_dict["pooling_type"] = trial.suggest_categorical("pooling_type", ["cls", "mean"])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    y_strat = df_train_pool[spec.target_col] if spec.target_col in df_train_pool.columns else np.zeros(len(df_train_pool))

    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_pool, y_strat)):
        train_fold = df_train_pool.iloc[train_idx].copy()
        val_fold = df_train_pool.iloc[val_idx].copy()
        
        train_fold = apply_feature_engineering(train_fold, spec.kind)
        val_fold = apply_feature_engineering(val_fold, spec.kind)
        
        preprocessor = V26Preprocessor(target_col=spec.target_col, oversample_method=oversample_method)
        train_prep = preprocessor.fit_transform(train_fold)
        val_prep = preprocessor.transform(val_fold)
        
        selector = V26FeatureSelector(target_col=spec.target_col, use_feature_selection=True)
        train_sel = selector.fit_transform(train_prep, preprocessor.numerical_cols, preprocessor.categorical_cols)
        val_sel = selector.transform(val_prep)
        
        num_classes = 3 if target_mode == "3class" else 5
        train_dataset = V27Dataset(train_sel, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
        val_dataset = V27Dataset(val_sel, spec.kind, spec.target_col, preprocessor.numerical_cols, preprocessor.categorical_cols)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        seq_cols_list = ["G1", "G2"] if spec.kind == "student" else []
        cat_cardinalities = [len(preprocessor.label_encoders[c].classes_) for c in preprocessor.categorical_cols if c in selector.selected_features and c not in seq_cols_list]
        num_numerical = len([c for c in preprocessor.numerical_cols if c in selector.selected_features and c not in seq_cols_list])
        
        model = create_model(spec.kind, config_dict, num_numerical, cat_cardinalities).to(device)
        
        class_weights = calculate_class_weights(train_prep[spec.target_col].values, num_classes).to(device)
        criterion = HybridLoss(class_weights=class_weights, gamma=focal_gamma, lambda_ordinal=lambda_ordinal)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        tconfig = TrainingConfig(max_epochs=40, patience=8) 
        
        _, _, best_val_f1 = train_model(model, train_loader, val_loader, criterion, optimizer, tconfig, device)
        fold_f1s.append(best_val_f1)
        
        # Stop early to save time if fold 1 is terrible
        if len(fold_f1s) == 1 and fold_f1s[0] < 0.4:
            raise optuna.exceptions.TrialPruned()
            
    return np.mean(fold_f1s)


