"""Compact neural candidates for the UCI G1/G2 student-grade studies.

The implementations intentionally contain no BatchNorm.  Ordered heads use an
ordered-cutpoint parameterization, so monotone cumulative probabilities follow
from construction and never require clamping.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


STUDENT_GRADE_NEURAL_CANDIDATES = {"N0", "N1", "N2", "N3", "A1", "A2"}
ORDINAL_CANDIDATES = {"N1", "N3"}


class OrderedCutpointHead(nn.Module):
    """Map a representation to cumulative logits with ordered cutpoints."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(int(input_dim), 1)
        self.threshold_base = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        self.threshold_delta_unconstrained = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def thresholds(self) -> torch.Tensor:
        first = self.threshold_base
        second = first + F.softplus(self.threshold_delta_unconstrained)
        return torch.stack([first, second])

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        latent = self.score(representation)
        return latent - self.thresholds().view(1, 2)

    @staticmethod
    def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
        cumulative = torch.sigmoid(logits)
        p_gt_low = cumulative[:, 0]
        p_gt_medium = cumulative[:, 1]
        probabilities = torch.stack(
            [1.0 - p_gt_low, p_gt_low - p_gt_medium, p_gt_medium],
            dim=1,
        )
        return probabilities / probabilities.sum(dim=1, keepdim=True)


class StudentGradeSequenceModel(nn.Module):
    """CNN–BiLSTM and matched sequence ablations for N0/N1/A1/A2."""

    def __init__(
        self,
        *,
        candidate_id: str,
        cnn_channels: int,
        cnn_kernel_size: int,
        lstm_hidden_dim: int,
        normalization: str,
        dropout: float,
        sequence_dropout: float,
    ):
        super().__init__()
        if candidate_id not in {"N0", "N1", "A1", "A2"}:
            raise ValueError(f"Unsupported sequence candidate: {candidate_id}")
        if cnn_kernel_size not in {1, 2}:
            raise ValueError("The G1/G2 sequence length 2 permits only kernel size 1 or 2.")
        if normalization not in {"none", "layer_norm"}:
            raise ValueError("Student-grade normalization must be none or layer_norm.")
        self.candidate_id = candidate_id
        self.architecture_variant = {
            "N0": "cnn_bilstm", "N1": "cnn_bilstm", "A1": "cnn_only", "A2": "bilstm_only"
        }[candidate_id]
        self.normalization_name = normalization
        self.input_sequence_length = 2
        self.sequence_cnn: nn.Conv1d | None = None
        self.sequence_norm: nn.LayerNorm | nn.Identity = nn.Identity()
        self.sequence_bilstm: nn.LSTM | None = None
        if candidate_id != "A2":
            self.sequence_cnn = nn.Conv1d(1, int(cnn_channels), int(cnn_kernel_size), padding=0)
            self.cnn_output_sequence_length = self.input_sequence_length - int(cnn_kernel_size) + 1
            self.sequence_norm = (
                nn.LayerNorm(int(cnn_channels)) if normalization == "layer_norm" else nn.Identity()
            )
            recurrent_input = int(cnn_channels)
        else:
            self.cnn_output_sequence_length = self.input_sequence_length
            recurrent_input = 1
        self.activation = nn.ReLU()
        self.sequence_dropout = nn.Dropout(float(sequence_dropout))
        if candidate_id != "A1":
            self.sequence_bilstm = nn.LSTM(
                input_size=recurrent_input,
                hidden_size=int(lstm_hidden_dim),
                batch_first=True,
                bidirectional=True,
            )
            representation_dim = 2 * int(lstm_hidden_dim)
        else:
            representation_dim = int(cnn_channels)
        self.head_dropout = nn.Dropout(float(dropout))
        self.head: nn.Module = (
            OrderedCutpointHead(representation_dim)
            if candidate_id == "N1"
            else nn.Linear(representation_dim, 3)
        )

    def encode(self, seq_x: torch.Tensor) -> torch.Tensor:
        if seq_x.ndim != 3 or seq_x.shape[1:] != (2, 1):
            raise ValueError("Student-grade sequence input must have shape [batch, 2, 1].")
        sequence = seq_x.float()
        if self.sequence_cnn is not None:
            sequence = self.sequence_cnn(sequence.transpose(1, 2)).transpose(1, 2)
            sequence = self.activation(sequence)
            # LayerNorm is applied to the final channel dimension of [B, L, C].
            sequence = self.sequence_norm(sequence)
            sequence = self.sequence_dropout(sequence)
        if self.candidate_id == "A1":
            representation = sequence.mean(dim=1)
        else:
            assert self.sequence_bilstm is not None
            _, (hidden, _) = self.sequence_bilstm(sequence)
            representation = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.head_dropout(representation)

    def forward(self, seq_x: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(seq_x))

    def predict_proba(self, seq_x: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        logits = self.forward(seq_x)
        if self.candidate_id == "N1":
            return OrderedCutpointHead.probabilities_from_logits(logits)
        return torch.softmax(logits, dim=1)


class StudentGradeMLPModel(nn.Module):
    """Raw G1/G2 vector MLP for N2/N3."""

    def __init__(
        self,
        *,
        candidate_id: str,
        hidden_dim: int,
        num_layers: int,
        normalization: str,
        dropout: float,
    ):
        super().__init__()
        if candidate_id not in {"N2", "N3"}:
            raise ValueError(f"Unsupported MLP candidate: {candidate_id}")
        if num_layers not in {1, 2}:
            raise ValueError("Tiny MLP supports one or two hidden layers.")
        if normalization not in {"none", "layer_norm"}:
            raise ValueError("Student-grade normalization must be none or layer_norm.")
        self.candidate_id = candidate_id
        self.architecture_variant = "mlp"
        self.normalization_name = normalization
        self.input_sequence_length = 2
        self.cnn_output_sequence_length = 0
        layers: list[nn.Module] = []
        input_dim = 2
        for _ in range(int(num_layers)):
            layers.append(nn.Linear(input_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            if normalization == "layer_norm":
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(nn.Dropout(float(dropout)))
            input_dim = int(hidden_dim)
        self.encoder = nn.Sequential(*layers)
        self.head: nn.Module = (
            OrderedCutpointHead(input_dim) if candidate_id == "N3" else nn.Linear(input_dim, 3)
        )

    def forward(self, seq_x: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        if seq_x.ndim != 3 or seq_x.shape[1:] != (2, 1):
            raise ValueError("Student-grade MLP input must have shape [batch, 2, 1].")
        representation = self.encoder(seq_x.float().reshape(len(seq_x), 2))
        return self.head(representation)

    def predict_proba(self, seq_x: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        logits = self.forward(seq_x)
        if self.candidate_id == "N3":
            return OrderedCutpointHead.probabilities_from_logits(logits)
        return torch.softmax(logits, dim=1)


def create_student_grade_model(config: dict) -> nn.Module:
    candidate_id = str(config["candidate_id"])
    if candidate_id not in STUDENT_GRADE_NEURAL_CANDIDATES:
        raise ValueError(f"Unknown student-grade neural candidate: {candidate_id}")
    if candidate_id in {"N2", "N3"}:
        return StudentGradeMLPModel(
            candidate_id=candidate_id,
            hidden_dim=int(config["hidden_dim"]),
            num_layers=int(config["num_layers"]),
            normalization=str(config["normalization"]),
            dropout=float(config["dropout"]),
        )
    return StudentGradeSequenceModel(
        candidate_id=candidate_id,
        cnn_channels=int(config["cnn_channels"]),
        cnn_kernel_size=int(config["cnn_kernel_size"]),
        lstm_hidden_dim=int(config["lstm_hidden_dim"]),
        normalization=str(config["normalization"]),
        dropout=float(config["dropout"]),
        sequence_dropout=float(config["sequence_dropout"]),
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
