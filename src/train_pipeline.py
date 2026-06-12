"""Training and Optuna search for the approved CNN-BiLSTM + MLP model."""

from __future__ import annotations

import copy

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.config import TrainingConfig
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    apply_feature_engineering,
    get_sequence_columns,
)
from src.models import create_model
from src.utils import set_seed, setup_logger

logger = setup_logger("train_pipeline")


class EarlyStopping:
    def __init__(self, patience: int = 15, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_metric: float, model: nn.Module) -> None:
        if self.best_score is None or val_metric >= self.best_score + self.delta:
            self.best_score = val_metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for seq_x, num_x, cat_x, labels, _ in dataloader:
        seq_x = seq_x.to(device)
        num_x = num_x.to(device)
        cat_x = cat_x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(seq_x, num_x, cat_x)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for seq_x, num_x, cat_x, labels, _ in dataloader:
            seq_x = seq_x.to(device)
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            labels = labels.to(device)

            logits = model(seq_x, num_x, cat_x)
            total_loss += criterion(logits, labels).item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / max(len(dataloader), 1)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    return val_loss, f1_macro, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, config, device):
    early_stopping = EarlyStopping(patience=config.patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    for epoch in range(config.max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc = validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        scheduler.step(val_f1)
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %s", epoch + 1)
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
    return model, history, early_stopping.best_score or 0.0


def calculate_class_weights(y, num_classes):
    labels = np.asarray(y, dtype=int)
    counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def _model_dimensions(dataset: StudentDataset, preprocessor: DataPreprocessor):
    cat_cardinalities = [
        len(preprocessor.label_encoders[column].classes_)
        for column in dataset.cat_cols
    ]
    return len(dataset.num_cols), cat_cardinalities


def suggest_trial_params(trial, dataset_kind: str) -> dict:
    """Build the Optuna search space while keeping the wider search xAPI-only."""
    if dataset_kind == "xapi":
        return {
            # Strategy 1: Aggressive Optuna Expansion
            # Expand learning rate (5e-5 to 5e-2 log scale) and broaden weight decay
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            
            # Strategy 3: Advanced Resampling Strategy Tuning
            # Dynamically explore smote_ratio and k_neighbors
            "oversample_method": trial.suggest_categorical(
                "oversample_method", ["smote", "adasyn"]
            ),
            "smote_ratio": trial.suggest_float("smote_ratio", 0.3, 1.0),
            "resampling_k_neighbors": trial.suggest_int("resampling_k_neighbors", 2, 10),
            
            # Strategy 2: Fine-Tuning the CNN-BiLSTM (Sequential Branch)
            "cnn_channels": trial.suggest_categorical("cnn_channels", [16, 32, 64, 128]),
            # Expand cnn_kernel_size specifically for xapi [2, 3, 4]
            "cnn_kernel_size": trial.suggest_categorical("cnn_kernel_size", [2, 3, 4]),
            # Increase upper bound for sequence_hidden_dim (lstm_hidden_dim) up to 128
            "lstm_hidden_dim": trial.suggest_categorical(
                "lstm_hidden_dim", [32, 64, 96, 128]
            ),
            "context_hidden_dim": trial.suggest_categorical(
                "context_hidden_dim", [32, 64, 128, 256]
            ),
            # Increase upper bound for fusion_hidden_dim up to 256
            "fusion_hidden_dim": trial.suggest_categorical(
                "fusion_hidden_dim", [32, 64, 128, 256]
            ),
            # Optimize dropout ranges specifically for the sequence branch (0.1 to 0.6)
            "sequence_dropout": trial.suggest_float("sequence_dropout", 0.1, 0.6),
            "context_dropout": trial.suggest_float("context_dropout", 0.1, 0.5),
            "fusion_dropout": trial.suggest_float("fusion_dropout", 0.1, 0.6),
        }

    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "oversample_method": trial.suggest_categorical(
            "oversample_method", ["smote", "adasyn"]
        ),
        "smote_ratio": trial.suggest_float("smote_ratio", 0.4, 1.0),
        "resampling_k_neighbors": 5,
        "cnn_channels": trial.suggest_categorical("cnn_channels", [16, 32, 64]),
        "cnn_kernel_size": 3,
        "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [32, 64]),
        "context_hidden_dim": trial.suggest_categorical(
            "context_hidden_dim", [32, 64, 128]
        ),
        "fusion_hidden_dim": trial.suggest_categorical("fusion_hidden_dim", [32, 64, 128]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
    }


def objective(trial, df_train_pool: pd.DataFrame, spec, target_mode: str, cv_folds: int):
    if target_mode != "3class":
        raise ValueError("The approved thesis architecture supports the 3-class task only.")

    params = suggest_trial_params(trial, spec.kind)
    learning_rate = params["learning_rate"]
    weight_decay = params["weight_decay"]
    batch_size = params["batch_size"]
    oversample_method = params["oversample_method"]
    smote_ratio = params["smote_ratio"]
    model_config = params

    stratified_folds = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    target = df_train_pool[spec.target_col].astype(int).to_numpy()
    fold_f1s = []
    sequence_columns = get_sequence_columns(spec.kind)

    for fold_index, (train_index, val_index) in enumerate(
        stratified_folds.split(df_train_pool, target)
    ):
        set_seed(42 + trial.number * cv_folds + fold_index)
        train_fold = apply_feature_engineering(df_train_pool.iloc[train_index].copy(), spec.kind)
        val_fold = apply_feature_engineering(df_train_pool.iloc[val_index].copy(), spec.kind)

        preprocessor = DataPreprocessor(
            target_col=spec.target_col,
            oversample_method=oversample_method,
            smote_ratio=smote_ratio,
            resampling_k_neighbors=params["resampling_k_neighbors"],
        )
        train_prep = preprocessor.fit_transform(train_fold)
        val_prep = preprocessor.transform(val_fold)

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

        train_dataset = StudentDataset(
            train_selected,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        val_dataset = StudentDataset(
            val_selected,
            spec.kind,
            spec.target_col,
            preprocessor.numerical_cols,
            preprocessor.categorical_cols,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        num_numerical, cat_cardinalities = _model_dimensions(train_dataset, preprocessor)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(spec.kind, model_config, num_numerical, cat_cardinalities).to(device)

        original_train_labels = df_train_pool.iloc[train_index][spec.target_col].astype(int).to_numpy()
        class_weights = calculate_class_weights(original_train_labels, num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # Strategy 4: Early Stopping & Scheduler Calibration
        # Adjust patience to 25 epochs for the 150-trial run to give ReduceLROnPlateau headroom
        config = TrainingConfig(
            max_epochs=80 if spec.kind == "xapi" else 50,
            patience=25 if spec.kind == "xapi" else 15,
            scheduler_patience=8 if spec.kind == "xapi" else 5,
        )
        _, _, best_val_f1 = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            config,
            device,
        )
        fold_f1s.append(best_val_f1)

        trial.report(float(np.mean(fold_f1s)), step=fold_index)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if len(fold_f1s) == 1 and fold_f1s[0] < 0.4:
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(fold_f1s))
