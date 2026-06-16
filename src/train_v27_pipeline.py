import copy
import logging
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from src.config import TrainingConfig, DEFAULT_SEED
from src.data_pipeline import (
    DataPreprocessor,
    FeatureSelector,
    StudentDataset,
    apply_feature_engineering,
    get_sequence_columns,
)
from src.models_v27 import StudentHybridV27
from src.losses_v27 import ClassBalancedFocalLoss, OrdinalLoss, JointHybridLoss
from src.utils import set_seed, setup_logger

logger = setup_logger("train_v27_pipeline")


class EarlyStoppingV27:
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


def train_epoch_v27(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        seq_x, num_x, cat_x, labels, _, reg_label = batch
        seq_x = seq_x.to(device)
        num_x = num_x.to(device)
        cat_x = cat_x.to(device)
        labels = labels.to(device)
        reg_label = reg_label.to(device)

        optimizer.zero_grad()
        outputs = model(seq_x, num_x, cat_x)
        loss = criterion(outputs, labels, reg_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def validate_epoch_v27(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            seq_x, num_x, cat_x, labels, _, reg_label = batch
            seq_x = seq_x.to(device)
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            labels = labels.to(device)
            reg_label = reg_label.to(device)

            outputs = model(seq_x, num_x, cat_x)
            loss = criterion(outputs, labels, reg_label)
            total_loss += loss.item()

            preds = torch.argmax(outputs[0], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / max(len(dataloader), 1)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    return val_loss, f1_macro, accuracy


def train_model_v27(model, train_loader, val_loader, criterion, optimizer, config, device):
    from torch.optim.swa_utils import AveragedModel
    early_stopping = EarlyStoppingV27(patience=config.patience)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    swa_model = AveragedModel(model)
    swa_start = int(config.max_epochs * 0.6)

    for epoch in range(config.max_epochs):
        train_loss = train_epoch_v27(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_acc = validate_epoch_v27(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        scheduler.step(val_f1)
        early_stopping(val_f1, model)
        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
        
    swa_model.eval()
    with torch.no_grad():
        _, val_f1_swa, _ = validate_epoch_v27(swa_model, val_loader, criterion, device)
    
    if val_f1_swa > (early_stopping.best_score or 0.0):
        logger.info(f"SWA model is better ({val_f1_swa:.4f} > {early_stopping.best_score or 0.0:.4f}). Adopting SWA.")
        model.load_state_dict(swa_model.module.state_dict())
        return model, history, val_f1_swa

    return model, history, early_stopping.best_score or 0.0


def create_model_v27(dataset_kind: str, config: dict, num_numerical: int, cat_cardinalities: list) -> StudentHybridV27:
    embedding_dim = config.get("embedding_dim", None)
    num_classes = 3 # Both student-mat/student-por and xapi are mapped to 3 classes in target_mode="3class"
    return StudentHybridV27(
        num_classes=num_classes,
        seq_in_channels=1,
        num_numerical=num_numerical,
        cat_cardinalities=cat_cardinalities,
        cnn_channels=int(config.get("cnn_channels", 32)),
        cnn_kernel_size=int(config.get("cnn_kernel_size", 3)),
        lstm_hidden_dim=int(config.get("lstm_hidden_dim", 64)),
        context_hidden_dim=int(config.get("context_hidden_dim", 64)),
        fusion_hidden_dim=int(config.get("fusion_hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.3)),
        sequence_dropout=config.get("sequence_dropout", None),
        context_dropout=config.get("context_dropout", None),
        fusion_dropout=config.get("fusion_dropout", None),
        embedding_dim=embedding_dim,
    )
