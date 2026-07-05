"""CNN-BiLSTM classifier aligned with the thesis proposal."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class StudentHybridModel(nn.Module):
    """Sequence-only CNN-BiLSTM classifier.

    The class name is retained for backwards-compatible imports, but the active
    architecture intentionally has no profile/context MLP, embedding branch or
    fusion layer. The classifier consumes only the chronological grade sequence.
    """

    architecture = "cnn_bilstm_classifier"
    context_mlp_enabled = False
    classifier_head = "linear"

    def __init__(
        self,
        num_classes: int,
        seq_in_channels: int,
        num_numerical: int = 0,
        cat_cardinalities: list[int] | None = None,
        cnn_channels: int = 32,
        cnn_kernel_size: int = 3,
        lstm_hidden_dim: int = 64,
        dropout: float = 0.3,
        sequence_dropout: float | None = None,
        ablation_mode: str = "sequence_only",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_numerical = 0
        self.cat_cardinalities: list[int] = []
        self.ablation_mode = "sequence_only"
        self.sequence_columns: list[str] = []
        sequence_dropout = dropout if sequence_dropout is None else sequence_dropout

        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=seq_in_channels,
                out_channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                padding=cnn_kernel_size // 2,
            ),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
        )
        self.sequence_bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.sequence_dropout = nn.Dropout(float(sequence_dropout))
        self.sequence_output_dim = lstm_hidden_dim * 2
        self.classifier = nn.Linear(self.sequence_output_dim, num_classes)

    def forward(
        self,
        seq_x: torch.Tensor,
        num_x: torch.Tensor | None = None,
        cat_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if seq_x is None:
            raise ValueError("Sequential grade input is required by the CNN-BiLSTM classifier.")
        if seq_x.ndim != 3:
            raise ValueError("Sequential input must have shape [batch, timesteps, channels].")

        sequence = seq_x.float().transpose(1, 2)
        sequence = self.sequence_cnn(sequence).transpose(1, 2)
        _, (hidden, _) = self.sequence_bilstm(sequence)
        sequence_vector = torch.cat([hidden[-2], hidden[-1]], dim=1)
        sequence_vector = self.sequence_dropout(sequence_vector)
        return self.classifier(sequence_vector)

    def predict_proba(
        self,
        seq_x: torch.Tensor,
        num_x: torch.Tensor | None = None,
        cat_x: torch.Tensor | None = None,
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
    num_numerical: int = 0,
    cat_cardinalities: list[int] | None = None,
) -> StudentHybridModel:
    """Create the final thesis classifier: sequence input -> CNN -> Bi-LSTM -> linear head."""
    num_classes = 2 if dataset_kind == "xapi" else 3
    return StudentHybridModel(
        num_classes=num_classes,
        seq_in_channels=1,
        num_numerical=0,
        cat_cardinalities=[],
        cnn_channels=int(config.get("cnn_channels", 32)),
        cnn_kernel_size=int(config.get("cnn_kernel_size", 3)),
        lstm_hidden_dim=int(config.get("lstm_hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.3)),
        sequence_dropout=config.get("sequence_dropout", None),
        ablation_mode="sequence_only",
    )
