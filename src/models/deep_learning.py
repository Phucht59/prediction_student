from __future__ import annotations

import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SafeBatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.shape[0] <= 1:
            return nn.functional.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(input)


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, num_classes)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, num_classes), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CNNBiLSTMTabularClassifier(nn.Module):
    """CNN-BiLSTM over encoded tabular features as a pseudo-sequence.

    This is not a true semester time-series model. It treats the processed
    feature vector as an ordered pseudo-sequence to test whether the CNN-BiLSTM
    architecture is useful on the encoded tabular representation.
    """

    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 64,
        lstm_hidden: int = 64,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.use_pool = input_dim >= 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2) if self.use_pool else nn.Identity()
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        lstm_output, _ = self.lstm(x)
        representation = lstm_output.mean(dim=1)
        return self.classifier(representation)


class StudentCNNBiLSTMV2(nn.Module):
    """Student tabular CNN-BiLSTM with an explicit grade representation branch.

    The BiLSTM reads the encoded feature vector as a pseudo-sequence for
    representation learning. When G1/G2 are available, the grade branch can
    read a richer 2-step sequence such as [G1, trend, normalized grade, velocity]
    instead of treating G1/G2 as a one-dimensional temporal signal.
    """

    def __init__(
        self,
        input_dim: int,
        grade_feature_dim: int = 0,
        grade_seq_len: int = 0,
        grade_input_dim: int = 0,
        conv_channels: int = 64,
        lstm_hidden: int = 32,
        n_lstm_layers: int = 1,
        grade_hidden: int = 32,
        fusion_hidden: int = 64,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1.")
        if grade_feature_dim < 0:
            raise ValueError("grade_feature_dim must be non-negative.")
        if grade_seq_len < 0:
            raise ValueError("grade_seq_len must be non-negative.")
        if grade_input_dim < 0:
            raise ValueError("grade_input_dim must be non-negative.")
        if n_lstm_layers < 1:
            raise ValueError("n_lstm_layers must be at least 1.")
        if bool(grade_seq_len) != bool(grade_input_dim):
            raise ValueError("grade_seq_len and grade_input_dim must either both be positive or both be zero.")

        self.input_dim = input_dim
        self.grade_feature_dim = grade_feature_dim
        self.grade_seq_len = grade_seq_len
        self.grade_input_dim = grade_input_dim
        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            SafeBatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) if input_dim >= 2 else nn.Identity(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            SafeBatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.use_grade_sequence_branch = grade_seq_len > 0 and grade_input_dim > 0
        self.use_grade_mlp_branch = grade_feature_dim > 0 and not self.use_grade_sequence_branch
        if self.use_grade_sequence_branch:
            self.grade_lstm = nn.LSTM(
                input_size=grade_input_dim,
                hidden_size=grade_hidden,
                num_layers=n_lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_lstm_layers > 1 else 0.0,
            )
            self.grade_branch = nn.Sequential(
                nn.Linear(grade_hidden * 2, grade_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_input_dim = lstm_hidden * 2 + grade_hidden
        elif self.use_grade_mlp_branch:
            self.grade_branch = nn.Sequential(
                nn.Linear(grade_feature_dim, grade_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            fusion_input_dim = lstm_hidden * 2 + grade_hidden
        else:
            self.grade_branch = None
            fusion_input_dim = lstm_hidden * 2

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            SafeBatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, grade_features: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x width mismatch: expected {self.input_dim}, got {x.shape[1]}.")
        conv_x = self.conv(x.unsqueeze(1))
        sequence_x = conv_x.transpose(1, 2)
        sequence_output, _ = self.lstm(sequence_x)
        feature_repr = sequence_output.mean(dim=1)

        if not self.use_grade_sequence_branch and not self.use_grade_mlp_branch:
            return self.classifier(feature_repr)
        if grade_features is None:
            raise ValueError("grade_features is required when the grade branch is enabled.")
        if self.use_grade_sequence_branch:
            if grade_features.ndim != 3 or grade_features.shape[1:] != (self.grade_seq_len, self.grade_input_dim):
                raise ValueError(
                    "grade_features shape mismatch: expected "
                    f"(*, {self.grade_seq_len}, {self.grade_input_dim}), got {tuple(grade_features.shape)}."
                )
            grade_output, _ = self.grade_lstm(grade_features)
            grade_repr = self.grade_branch(grade_output.mean(dim=1))
            return self.classifier(torch.cat([feature_repr, grade_repr], dim=1))

        if grade_features.ndim != 2 or grade_features.shape[1] != self.grade_feature_dim:
            raise ValueError(
                "grade_features shape mismatch: expected "
                f"(*, {self.grade_feature_dim}), got {tuple(grade_features.shape)}."
            )
        grade_repr = self.grade_branch(grade_features)
        return self.classifier(torch.cat([feature_repr, grade_repr], dim=1))


class HybridCNNBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        static_input_dim: int,
        grade_seq_len: int,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        static_hidden: int = 64,
        dropout: float = 0.3,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if grade_seq_len < 1:
            raise ValueError("grade_seq_len must be at least 1.")

        self.grade_conv = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.grade_lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.static_branch = nn.Sequential(
            nn.Linear(static_input_dim, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + static_hidden, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_hidden, num_classes),
        )

    def forward(self, static_x: torch.Tensor, grade_seq: torch.Tensor) -> torch.Tensor:
        grade_x = grade_seq.transpose(1, 2)
        grade_x = self.grade_conv(grade_x)
        grade_x = grade_x.transpose(1, 2)
        grade_output, _ = self.grade_lstm(grade_x)
        temporal_repr = grade_output.mean(dim=1)
        static_repr = self.static_branch(static_x)
        combined = torch.cat([temporal_repr, static_repr], dim=1)
        return self.classifier(combined)


class CNNBiLSTMXAPI(nn.Module):
    """CLS-XAPI / CNN-BiLSTM-XAPI classifier over one processed xAPI input.

    All xAPI features after preprocessing are consumed as one ordered
    pseudo-sequence. The model intentionally does not split the four behavior
    counters into a separate branch.
    """

    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 16,
        n_conv_blocks: int = 1,
        lstm_hidden: int = 16,
        n_lstm_layers: int = 1,
        dense_hidden: int = 64,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1.")
        if n_conv_blocks not in {1, 2}:
            raise ValueError("n_conv_blocks must be 1 or 2.")
        if n_lstm_layers < 1:
            raise ValueError("n_lstm_layers must be at least 1.")

        self.input_dim = input_dim
        self.n_conv_blocks = n_conv_blocks
        self._lstm_input_size = conv_channels * (2 if n_conv_blocks == 2 else 1)
        if n_conv_blocks == 1:
            conv_layers = [
                nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
                SafeBatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2) if input_dim >= 2 else nn.Identity(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                SafeBatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2) if input_dim >= 4 else nn.Identity(),
            ]
        else:
            conv_layers = [
                nn.Conv1d(1, conv_channels, kernel_size=3, padding=1),
                SafeBatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
                SafeBatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
                SafeBatchNorm1d(conv_channels * 2),
                nn.ReLU(),
                nn.Conv1d(conv_channels * 2, conv_channels * 2, kernel_size=3, padding=1),
                SafeBatchNorm1d(conv_channels * 2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            ]
        self.conv_block = nn.Sequential(*conv_layers)
        self.bilstm1 = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.bilstm2 = (
            nn.LSTM(
                input_size=lstm_hidden * 2,
                hidden_size=lstm_hidden,
                num_layers=n_lstm_layers - 1,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_lstm_layers > 2 else 0.0,
            )
            if n_lstm_layers > 1
            else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, dense_hidden),
            SafeBatchNorm1d(dense_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x width mismatch: expected {self.input_dim}, got {x.shape[1]}.")

        sequence = self.conv_block(x.unsqueeze(1)).transpose(1, 2)
        sequence, _ = self.bilstm1(sequence)
        sequence = self.dropout(sequence)
        if self.bilstm2 is not None:
            sequence, _ = self.bilstm2(sequence)
            sequence = self.dropout(sequence)
        representation = sequence.mean(dim=1)
        return self.classifier(representation)


class CLSXAPI(CNNBiLSTMXAPI):
    """Alias class name for the paper-aligned xAPI CNN-BiLSTM model."""
