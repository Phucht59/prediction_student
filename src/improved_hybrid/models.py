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
        # x shape: [batch_size, seq_len, input_dim]
        attn_weights = self.attention(x) # [batch, seq, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum over sequence
        weighted = torch.sum(x * attn_weights, dim=1) # [batch, input_dim]
        return weighted

class ContextEncoder(nn.Module):
    def __init__(self, num_numeric, categorical_cardinalities, hidden_dim, dropout=0.3):
        super().__init__()
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, min(50, (cardinality + 1) // 2))
            for cardinality in categorical_cardinalities
        ])
        
        total_embed_dim = sum(emb.embedding_dim for emb in self.cat_embeddings)
        total_input_dim = num_numeric + total_embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU()
        )
        
    def forward(self, numeric_x, categorical_x):
        # categorical_x: [batch, num_cat_features]
        cat_embeds = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embeds.append(emb(categorical_x[:, i]))
            
        if cat_embeds:
            cat_x = torch.cat(cat_embeds, dim=1)
            x = torch.cat([numeric_x, cat_x], dim=1)
        else:
            x = numeric_x
            
        return self.mlp(x)

class HybridCNNBiLSTMAttentionOrdinal(nn.Module):
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

        # In case seq_input_dim is very small, we adjust kernel size
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

        fusion_input_dim = bilstm_hidden * 2
        if self.use_context:
            fusion_input_dim += context_hidden

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
        # seq_x: [batch, seq_len]
        # We need [batch, channels=1, seq_len] for Conv1D to treat features as a spatial sequence
        z = seq_x.unsqueeze(1) 
        z = self.sequence_cnn(z) # [batch, channels, seq_len]
        
        # Transpose for LSTM: [batch, seq_len, channels]
        z = z.transpose(1, 2)
        
        z, _ = self.bilstm(z) # [batch, seq_len, hidden*2]
        
        if self.use_attention:
            seq_emb = self.attention(z) # [batch, hidden*2]
        else:
            # Mean pooling if no attention
            seq_emb = torch.mean(z, dim=1)

        if self.use_context:
            ctx_emb = self.context_encoder(numeric_x, categorical_x) # [batch, context_hidden]
            fused = torch.cat([seq_emb, ctx_emb], dim=1)
        else:
            fused = seq_emb
            
        fused = self.fusion(fused)

        logits = self.classification_head(fused)
        ordinal_score = self.ordinal_head(fused).squeeze(-1)
        
        return logits, ordinal_score
