"""CNN-BiLSTM + MLP model approved for the student-performance thesis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AttentionPooling1D(nn.Module):
    """Pool Bi-LSTM outputs with a small, interpretable attention layer."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        attention_hidden = max(8, hidden_dim // 2)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.score(sequence), dim=1)
        pooled = torch.sum(sequence * weights, dim=1)
        return pooled, weights


class StudentHybridModel(nn.Module):
    """Pure CNN-BiLSTM sequence branch fused with a context MLP branch."""

    def __init__(
        self,
        num_classes: int,
        seq_in_channels: int,
        num_numerical: int,
        cat_cardinalities: list[int],
        cnn_channels: int = 32,
        cnn_kernel_size: int = 3,
        lstm_hidden_dim: int = 64,
        context_hidden_dim: int = 64,
        fusion_hidden_dim: int = 64,
        dropout: float = 0.3,
        sequence_dropout: float | None = None,
        context_dropout: float | None = None,
        fusion_dropout: float | None = None,
        embedding_dim: int | None = None,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        sequence_dropout = dropout if sequence_dropout is None else sequence_dropout
        context_dropout = dropout if context_dropout is None else context_dropout
        fusion_dropout = dropout if fusion_dropout is None else fusion_dropout

        self.embeddings = nn.ModuleList()
        embedding_total_dim = 0
        for cardinality in cat_cardinalities:
            dim = embedding_dim if embedding_dim else max(2, min(50, (cardinality + 1) // 2))
            self.embeddings.append(nn.Embedding(num_embeddings=cardinality, embedding_dim=dim))
            embedding_total_dim += dim

        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=seq_in_channels,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(sequence_dropout),
        )
        self.sequence_bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        sequence_output_dim = lstm_hidden_dim * 2
        self.sequence_pool = AttentionPooling1D(sequence_output_dim)

        context_input_dim = num_numerical + embedding_total_dim
        self.context_input_dim = max(1, context_input_dim)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.context_input_dim, context_hidden_dim),
            nn.ReLU(),
            nn.Dropout(context_dropout),
            nn.Linear(context_hidden_dim, context_hidden_dim),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(sequence_output_dim + context_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
        )
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    def _prepare_context(
        self,
        num_x: torch.Tensor | None,
        cat_x: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if self.num_numerical > 0:
            if num_x is None or num_x.shape[1] < self.num_numerical:
                raise ValueError("Numerical context does not match the configured model input.")
            parts.append(num_x[:, : self.num_numerical].float())

        if self.cat_cardinalities:
            if cat_x is None or cat_x.shape[1] < len(self.cat_cardinalities):
                raise ValueError("Categorical context does not match the configured model input.")
            embedded_categorical = []
            for index, emb_layer in enumerate(self.embeddings):
                values = cat_x[:, index].long()
                # Ensure no index out of bounds
                cardinality = self.cat_cardinalities[index]
                values = torch.clamp(values, 0, cardinality - 1)
                embedded_categorical.append(emb_layer(values))
            parts.append(torch.cat(embedded_categorical, dim=1))

        if not parts:
            return torch.zeros(batch_size, 1, device=device)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        seq_x: torch.Tensor,
        num_x: torch.Tensor | None,
        cat_x: torch.Tensor | None,
    ) -> torch.Tensor:
        if seq_x is None:
            raise ValueError("Sequential input is required by the CNN-BiLSTM architecture.")

        sequence = seq_x.float().transpose(1, 2)
        sequence = self.sequence_cnn(sequence).transpose(1, 2)
        sequence, _ = self.sequence_bilstm(sequence)
        sequence_vector, _ = self.sequence_pool(sequence)

        context = self._prepare_context(
            num_x=num_x,
            cat_x=cat_x,
            batch_size=seq_x.shape[0],
            device=seq_x.device,
        )
        context_vector = self.context_mlp(context)

        fused = self.fusion(torch.cat([sequence_vector, context_vector], dim=1))
        return self.classifier(fused)

    def predict_proba(
        self,
        seq_x: torch.Tensor,
        num_x: torch.Tensor | None,
        cat_x: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.classifier.out_features == 2:
            logits = self.forward(seq_x, num_x, cat_x)
            probs_gt = torch.sigmoid(logits)
            p_gt_low = probs_gt[:, 0]
            p_gt_medium = probs_gt[:, 1]
            p_low = 1.0 - p_gt_low
            p_medium = torch.clamp(p_gt_low - p_gt_medium, min=0.0)
            p_high = p_gt_medium
            probs = torch.stack([p_low, p_medium, p_high], dim=1)
            return probs / probs.sum(dim=1, keepdim=True)
        return torch.softmax(self.forward(seq_x, num_x, cat_x), dim=1)


def create_model(
    dataset_kind: str,
    config: dict[str, Any],
    num_numerical: int,
    cat_cardinalities: list[int],
) -> StudentHybridModel:
    """Create the only model architecture allowed by the approved proposal."""
    embedding_dim = config.get("embedding_dim", None)
    num_classes = 2 if dataset_kind == "xapi" else 3
    return StudentHybridModel(
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
