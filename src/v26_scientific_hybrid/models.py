import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
    def forward(self, x):
        attn_weights = F.softmax(self.attention(x), dim=1)
        return torch.sum(x * attn_weights, dim=1)

class ContextEncoder(nn.Module):
    def __init__(self, num_numeric, categorical_cardinalities, hidden_dim, dropout=0.3):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality + 1) // 2))
            for cardinality in categorical_cardinalities
        ])
        total_embed_dim = sum(emb.embedding_dim for emb in self.cat_embeddings)
        self.mlp = nn.Sequential(
            nn.Linear(num_numeric + total_embed_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
    def forward(self, numeric_x, categorical_x):
        cat_embeds = [emb(categorical_x[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_x = torch.cat(cat_embeds, dim=1) if cat_embeds else torch.empty(numeric_x.shape[0], 0, device=numeric_x.device)
        x = torch.cat([numeric_x, cat_x], dim=1)
        return self.mlp(x)

class HybridCNNBiLSTMAttentionOrdinalV2(nn.Module):
    def __init__(
        self,
        seq_input_dim,
        num_numeric_context,
        categorical_cardinalities,
        num_classes,
        conv_channels=64,
        conv_kernel_size=2,
        bilstm_hidden=64,
        bilstm_layers=1,
        attention_dim=64,
        context_hidden=128,
        fusion_hidden=128,
        dropout=0.3,
        use_attention=True,
        use_context=True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_context = use_context

        actual_kernel_size = min(conv_kernel_size, seq_input_dim)
        
        self.sequence_cnn = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=actual_kernel_size, padding="same"),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bilstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=bilstm_hidden,
            num_layers=bilstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        if self.use_attention:
            self.attention = AttentionPooling(input_dim=bilstm_hidden * 2)

        if self.use_context:
            self.context_encoder = ContextEncoder(
                num_numeric=num_numeric_context,
                categorical_cardinalities=categorical_cardinalities,
                hidden_dim=context_hidden,
                dropout=dropout,
            )

        fusion_input_dim = bilstm_hidden * 2 + (context_hidden if self.use_context else 0)

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.GELU(),
        )

        self.classification_head = nn.Linear(fusion_hidden // 2, num_classes)
        self.ordinal_head = nn.Linear(fusion_hidden // 2, 1)

    def forward(self, seq_x, numeric_x, categorical_x):
        z = seq_x.unsqueeze(1) 
        z = self.sequence_cnn(z).transpose(1, 2)
        z, _ = self.bilstm(z)
        
        seq_emb = self.attention(z) if self.use_attention else torch.mean(z, dim=1)

        if self.use_context:
            ctx_emb = self.context_encoder(numeric_x, categorical_x)
            fused = torch.cat([seq_emb, ctx_emb], dim=1)
        else:
            fused = seq_emb
            
        fused = self.fusion(fused)
        return self.classification_head(fused), self.ordinal_head(fused).squeeze(-1)
