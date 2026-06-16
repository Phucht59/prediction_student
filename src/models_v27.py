import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any

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


class GatedFusion(nn.Module):
    """Fuses the sequence vector and context vector dynamically using a sigmoid gate."""

    def __init__(self, seq_dim: int, ctx_dim: int, out_dim: int):
        super().__init__()
        self.proj_seq = nn.Linear(seq_dim, out_dim)
        self.proj_ctx = nn.Linear(ctx_dim, out_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(seq_dim + ctx_dim, out_dim),
            nn.Sigmoid()
        )
        
    def forward(self, seq_vec: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        h_seq = self.proj_seq(seq_vec)
        h_ctx = self.proj_ctx(ctx_vec)
        concat = torch.cat([seq_vec, ctx_vec], dim=1)
        gate = self.gate_layer(concat)
        fused = gate * h_seq + (1.0 - gate) * h_ctx
        return fused


class StudentHybridV27(nn.Module):
    """V27 Hybrid Model integrating GatedFusion and three output heads."""

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

        self.fusion = GatedFusion(sequence_output_dim, context_hidden_dim, fusion_hidden_dim)
        
        # Output heads: classification, ordinal, regression
        self.class_head = nn.Linear(fusion_hidden_dim, num_classes)
        self.ordinal_head = nn.Linear(fusion_hidden_dim, num_classes - 1)
        self.reg_head = nn.Linear(fusion_hidden_dim, 1)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        fused = self.fusion(sequence_vector, context_vector)
        
        class_logits = self.class_head(fused)
        ordinal_logits = self.ordinal_head(fused)
        reg_logits = self.reg_head(fused).squeeze(-1)
        
        return class_logits, ordinal_logits, reg_logits

    def predict_proba(
        self,
        seq_x: torch.Tensor,
        num_x: torch.Tensor | None,
        cat_x: torch.Tensor | None,
    ) -> torch.Tensor:
        class_logits, _, _ = self.forward(seq_x, num_x, cat_x)
        return torch.softmax(class_logits, dim=1)
