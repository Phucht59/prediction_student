from __future__ import annotations

import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset

from src.evaluation.metrics import classification_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_class_weights(y_train, num_classes: int = 3) -> torch.FloatTensor:
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.asarray(y_train),
    )
    return torch.FloatTensor(weights)


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-cross_entropy)
        loss = ((1.0 - pt) ** self.gamma) * cross_entropy
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        classification_loss_type: str = "focal_loss",
        class_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        regression_weight: float = 0.3,
        classification_weight: float = 1.0,
        regression_loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        if classification_loss_type == "focal_loss":
            self.classification_loss = FocalLoss(
                gamma=focal_gamma,
                weight=class_weight,
                label_smoothing=label_smoothing,
                reduction="mean",
            )
        elif classification_loss_type == "cross_entropy":
            self.classification_loss = nn.CrossEntropyLoss(
                weight=class_weight,
                label_smoothing=label_smoothing,
            )
        else:
            raise ValueError(f"Unsupported classification_loss_type: {classification_loss_type}")

        if regression_loss_type == "smooth_l1":
            self.regression_loss = nn.SmoothL1Loss()
        elif regression_loss_type == "mse":
            self.regression_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported regression_loss_type: {regression_loss_type}")

        self.regression_weight = regression_weight
        self.classification_weight = classification_weight

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        y_class: torch.Tensor,
        y_reg_scaled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = self.classification_loss(outputs["logits"], y_class)
        reg_loss = self.regression_loss(outputs["regression"].view(-1), y_reg_scaled.view(-1))
        total_loss = self.classification_weight * cls_loss + self.regression_weight * reg_loss
        return total_loss, cls_loss, reg_loss


def make_tensor_dataset_for_static(X, y) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def make_tensor_dataset_for_hybrid(static_X, grade_seq, y) -> TensorDataset:
    return TensorDataset(
        torch.tensor(static_X, dtype=torch.float32),
        torch.tensor(grade_seq, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )


def unpack_batch(batch, device: torch.device, is_hybrid: bool):
    if is_hybrid:
        static_x, grade_seq, y = batch
        return (static_x.to(device), grade_seq.to(device)), y.to(device)
    x, y = batch
    return x.to(device), y.to(device)


def forward_model(model, inputs, is_hybrid: bool):
    if is_hybrid:
        static_x, grade_seq = inputs
        return model(static_x, grade_seq)
    return model(inputs)


def _mixup_inputs(inputs, y: torch.Tensor, alpha: float, num_classes: int, is_hybrid: bool):
    if alpha <= 0.0 or y.shape[0] < 2:
        return inputs, F.one_hot(y, num_classes=num_classes).float()
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(y.shape[0], device=y.device)
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[index]
    if is_hybrid:
        static_x, grade_seq = inputs
        mixed_inputs = (
            lam * static_x + (1.0 - lam) * static_x[index],
            lam * grade_seq + (1.0 - lam) * grade_seq[index],
        )
    else:
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, mixed_y


def _soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    loss_matrix = -soft_targets * log_probs
    if weight is not None:
        loss_matrix = loss_matrix * weight.view(1, -1)
    return loss_matrix.sum(dim=1).mean()


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
    is_hybrid: bool = False,
    mixup_alpha: float = 0.0,
    num_classes: int = 3,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch in data_loader:
        inputs, y = unpack_batch(batch, device, is_hybrid)
        optimizer.zero_grad()
        if mixup_alpha > 0.0:
            mixed_inputs, mixed_y = _mixup_inputs(inputs, y, mixup_alpha, num_classes, is_hybrid)
            logits = forward_model(model, mixed_inputs, is_hybrid)
            class_weight = getattr(criterion, "weight", None)
            loss = _soft_cross_entropy(logits, mixed_y, class_weight)
        else:
            logits = forward_model(model, inputs, is_hybrid)
            loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size

    return total_loss / max(total_rows, 1)


@torch.no_grad()
def compute_average_loss(model, data_loader, criterion, device, is_hybrid: bool = False) -> float:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    for batch in data_loader:
        inputs, y = unpack_batch(batch, device, is_hybrid)
        logits = forward_model(model, inputs, is_hybrid)
        loss = criterion(logits, y)
        batch_size = int(y.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def predict_model(model, data_loader, device, is_hybrid: bool = False):
    model.eval()
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []

    for batch in data_loader:
        inputs, y = unpack_batch(batch, device, is_hybrid)
        logits = forward_model(model, inputs, is_hybrid)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        y_true_parts.append(y.detach().cpu().numpy())
        y_pred_parts.append(predictions.detach().cpu().numpy())
        y_prob_parts.append(probabilities.detach().cpu().numpy())

    return (
        np.concatenate(y_true_parts),
        np.concatenate(y_pred_parts),
        np.concatenate(y_prob_parts),
    )


def evaluate_model(model, data_loader, device, is_hybrid: bool = False) -> dict[str, float]:
    y_true, y_pred, _ = predict_model(model, data_loader, device, is_hybrid)
    return classification_metrics(y_true, y_pred)


class EarlyStopping:
    def __init__(self, patience: int = 12, mode: str = "max") -> None:
        if mode != "max":
            raise ValueError("Only mode='max' is supported.")
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.best_state_dict: dict | None = None
        self.counter = 0

    def step(self, score: float, model, epoch: int) -> bool:
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def restore_best_weights(self, model) -> None:
        if self.best_state_dict is None:
            return
        model.load_state_dict(copy.deepcopy(self.best_state_dict))


def save_training_curve(history, output_path, title) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_f1 = [row["val_f1_macro"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="val_f1_macro", color="#4C78A8")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
