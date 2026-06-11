import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Tuple
from src.utils import setup_logger

logger = setup_logger("models")


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha # tensor of weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class HybridLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2.0, lambda_ordinal=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=gamma)
        self.smooth_l1 = nn.SmoothL1Loss()
        self.lambda_ordinal = lambda_ordinal
        
    def forward(self, logits, expected_val, target_class):
        f_loss = self.focal(logits, target_class)
        o_loss = self.smooth_l1(expected_val, target_class.float())
        total_loss = f_loss + self.lambda_ordinal * o_loss
        return total_loss, f_loss, o_loss



class TabularTokenizer(nn.Module):
    """
    Tokenizes both numerical and categorical features into embeddings of size `d_token`.
    This is required for FT-Transformer and useful for DeepFM.
    """
    def __init__(self, num_numerical: int, cat_cardinalities: list, d_token: int):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.d_token = d_token
        
        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cat_cardinalities
        ])
        
        # Numerical embeddings: Each numerical feature i has a learned vector W_i of size d_token.
        # Plus a bias b_i of size d_token.
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.Tensor(num_numerical, d_token))
            self.num_biases = nn.Parameter(torch.Tensor(num_numerical, d_token))
            nn.init.kaiming_uniform_(self.num_weights, a=math.sqrt(5))
            nn.init.zeros_(self.num_biases)

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor):
        """
        num_x: (batch, num_numerical)
        cat_x: (batch, num_categorical)
        Returns:
            tokens: (batch, num_features, d_token)
        """
        tokens = []
        
        # Process numerical features: x_i * W_i + b_i
        if self.num_numerical > 0:
            # num_x shape: (batch, num_num, 1)
            # num_weights shape: (1, num_num, d_token)
            x = num_x.unsqueeze(-1)
            w = self.num_weights.unsqueeze(0)
            b = self.num_biases.unsqueeze(0)
            num_tokens = x * w + b  # shape: (batch, num_num, d_token)
            tokens.append(num_tokens)
            
        # Process categorical features
        if len(self.cat_cardinalities) > 0:
            cat_tokens = []
            for i, emb in enumerate(self.cat_embeddings):
                # Clamp to avoid out-of-bounds indices for unseen categories
                c = torch.clamp(cat_x[:, i], 0, self.cat_cardinalities[i] - 1)
                t = emb(c) # (batch, d_token)
                cat_tokens.append(t.unsqueeze(1))
            cat_tokens = torch.cat(cat_tokens, dim=1) # (batch, num_cat, d_token)
            tokens.append(cat_tokens)
            
        if len(tokens) == 0:
            return None
            
        return torch.cat(tokens, dim=1) # (batch, num_num + num_cat, d_token)



class DeepFM(nn.Module):
    """
    DeepFM model for tabular data.
    Takes numerical and categorical features and computes Linear, FM, and Deep representations.
    Returns either logits or a representation vector for fusion.
    """
    def __init__(self, 
                 num_numerical: int, 
                 cat_cardinalities: list, 
                 d_token: int = 8, 
                 mlp_hidden: int = 64,
                 dropout: float = 0.3,
                 output_logits: bool = False,
                 num_classes: int = 3):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.output_logits = output_logits
        
        # Tokenizer for FM and Deep parts
        self.tokenizer = TabularTokenizer(num_numerical, cat_cardinalities, d_token)
        
        # Linear Part
        if num_numerical > 0:
            self.linear_num = nn.Linear(num_numerical, 1)
        self.linear_cat = nn.ModuleList([
            nn.Embedding(card, 1) for card in cat_cardinalities
        ])
        
        # Deep Part
        num_features = num_numerical + len(cat_cardinalities)
        deep_in_dim = num_features * d_token
        
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_in_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Final Output Layer if outputting logits
        if self.output_logits:
            # linear (1) + fm (1) + deep (mlp_hidden)
            self.head = nn.Linear(1 + 1 + mlp_hidden, num_classes)
        else:
            self.out_dim = 1 + 1 + mlp_hidden

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor):
        batch_size = num_x.shape[0] if num_x is not None and num_x.shape[0] > 0 else cat_x.shape[0]
        
        # 1. Linear Part
        linear_out = torch.zeros(batch_size, 1, device=num_x.device if num_x is not None else cat_x.device)
        if self.num_numerical > 0:
            linear_out += self.linear_num(num_x)
        
        for i, emb in enumerate(self.linear_cat):
            c = torch.clamp(cat_x[:, i], 0, self.cat_cardinalities[i] - 1)
            linear_out += emb(c) # (batch, 1)
            
        # Tokenize for FM and Deep
        tokens = self.tokenizer(num_x, cat_x) # (batch, num_features, d_token)
        
        # 2. FM Part
        # 0.5 * sum((sum(v_i x_i))^2 - sum((v_i x_i)^2))
        sum_of_square = torch.sum(tokens ** 2, dim=1) # (batch, d_token)
        square_of_sum = torch.sum(tokens, dim=1) ** 2 # (batch, d_token)
        fm_out = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True) # (batch, 1)
        
        # 3. Deep Part
        deep_in = tokens.view(batch_size, -1) # flatten
        deep_out = self.deep_mlp(deep_in) # (batch, mlp_hidden)
        
        # Concat all representations
        fused = torch.cat([linear_out, fm_out, deep_out], dim=1)
        
        if self.output_logits:
            return self.head(fused)
        return fused



class CrossNetwork(nn.Module):
    """
    DCN-V2 Cross Network.
    x_{l+1} = x_0 * (x_l W_l + b_l) + x_l
    """
    def __init__(self, in_features: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_features, in_features) * 0.01) for _ in range(num_layers)
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(in_features)) for _ in range(num_layers)
        ])
        
    def forward(self, x0):
        xl = x0
        for i in range(self.num_layers):
            xl_w = torch.matmul(xl, self.weights[i]) + self.biases[i]
            xl = x0 * xl_w + xl
        return xl

class DCNv2(nn.Module):
    """
    Deep & Cross Network V2.
    """
    def __init__(self, 
                 num_numerical: int, 
                 cat_cardinalities: list, 
                 d_token: int = 8, 
                 cross_layers: int = 2,
                 mlp_hidden: int = 64,
                 dropout: float = 0.3,
                 output_logits: bool = False,
                 num_classes: int = 3):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.output_logits = output_logits
        
        self.tokenizer = TabularTokenizer(num_numerical, cat_cardinalities, d_token)
        
        num_features = num_numerical + len(cat_cardinalities)
        in_dim = num_features * d_token
        
        # Cross Network
        self.cross = CrossNetwork(in_dim, cross_layers)
        
        # Deep Network
        self.deep_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if self.output_logits:
            # We use stacked architecture: out = concat(cross_out, deep_out) -> logits
            self.head = nn.Linear(in_dim + mlp_hidden, num_classes)
        else:
            self.out_dim = in_dim + mlp_hidden

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor):
        batch_size = num_x.shape[0] if num_x is not None and num_x.shape[0] > 0 else cat_x.shape[0]
        tokens = self.tokenizer(num_x, cat_x)
        x0 = tokens.view(batch_size, -1)
        
        cross_out = self.cross(x0)
        deep_out = self.deep_mlp(x0)
        
        fused = torch.cat([cross_out, deep_out], dim=1)
        
        if self.output_logits:
            return self.head(fused)
        return fused



class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer.
    """
    def __init__(self, 
                 num_numerical: int, 
                 cat_cardinalities: list, 
                 d_token: int = 32, 
                 n_heads: int = 4,
                 n_layers: int = 3,
                 ff_hidden_dim: int = 64,
                 attention_dropout: float = 0.2,
                 residual_dropout: float = 0.1,
                 output_logits: bool = True,
                 num_classes: int = 3,
                 pooling_type: str = "cls"):
        super().__init__()
        self.num_numerical = num_numerical
        self.cat_cardinalities = cat_cardinalities
        self.output_logits = output_logits
        self.pooling_type = pooling_type.lower()
        
        self.tokenizer = TabularTokenizer(num_numerical, cat_cardinalities, d_token)
        
        # CLS token
        if self.pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
            nn.init.kaiming_uniform_(self.cls_token)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, 
            nhead=n_heads, 
            dim_feedforward=ff_hidden_dim, 
            dropout=attention_dropout, 
            activation="gelu", 
            batch_first=True,
            norm_first=True # Better for stable training in FT-Transformer
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.final_ln = nn.LayerNorm(d_token)
        
        if self.output_logits:
            self.head = nn.Linear(d_token, num_classes)
        else:
            self.out_dim = d_token

    def forward(self, num_x: torch.Tensor, cat_x: torch.Tensor):
        batch_size = num_x.shape[0] if num_x is not None and num_x.shape[0] > 0 else cat_x.shape[0]
        
        tokens = self.tokenizer(num_x, cat_x) # (batch, num_features, d_token)
        
        if self.pooling_type == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (batch, 1, d_token)
            tokens = torch.cat([cls_tokens, tokens], dim=1) # (batch, 1 + num_features, d_token)
        
        # Transformer pass
        out = self.transformer(tokens) # (batch, seq_len, d_token)
        
        # Pooling
        if self.pooling_type == "cls":
            repr_vec = out[:, 0, :] # Extract CLS token
        elif self.pooling_type == "mean":
            repr_vec = out.mean(dim=1)
        else:
            # Fallback to cls or mean
            repr_vec = out.mean(dim=1)
            
        repr_vec = self.final_ln(repr_vec)
        
        if self.output_logits:
            return self.head(repr_vec)
        return repr_vec


import torch
import torch.nn as nn
import math

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class AttentionPooling1D(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(x * weights, dim=1)
        return pooled, weights

class StudentV27HybridModel(nn.Module):
    def __init__(self, 
                 num_classes: int,
                 seq_in_channels: int,
                 num_numerical: int,
                 cat_cardinalities: list,
                 context_architecture: str = "mlp_baseline",
                 sequence_architecture: str = "dsc_bilstm_attention",
                 fm_embedding_dim: int = 8,
                 dcn_cross_layers: int = 2,
                 context_hidden_dim: int = 64,
                 sequence_hidden_dim: int = 64,
                 fusion_hidden_dim: int = 64,
                 dropout: float = 0.3):
        super().__init__()
        self.context_architecture = context_architecture.lower()
        self.sequence_architecture = sequence_architecture.lower()
        
        # 1. Sequence Branch
        self.use_seq = self.sequence_architecture != "none"
        if self.use_seq:
            if "dsc" in self.sequence_architecture:
                self.seq_fe = nn.Sequential(
                    DepthwiseSeparableConv1d(seq_in_channels, sequence_hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(sequence_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                bilstm_in = sequence_hidden_dim
            else:
                self.seq_fe = nn.Identity()
                bilstm_in = seq_in_channels
                
            if "bilstm" in self.sequence_architecture:
                self.bilstm = nn.LSTM(bilstm_in, sequence_hidden_dim, batch_first=True, bidirectional=True)
                self.seq_attn = AttentionPooling1D(sequence_hidden_dim * 2)
                seq_out_dim = sequence_hidden_dim * 2
            elif "self_attention" in self.sequence_architecture:
                self.bilstm = nn.Identity()
                self.mha = nn.MultiheadAttention(embed_dim=bilstm_in, num_heads=1, batch_first=True)
                self.seq_attn = AttentionPooling1D(bilstm_in)
                seq_out_dim = bilstm_in
        else:
            seq_out_dim = 0
            
        # 2. Context Branch
        if self.context_architecture == "deepfm":
            self.context_branch = DeepFM(num_numerical, cat_cardinalities, d_token=fm_embedding_dim, 
                                         mlp_hidden=context_hidden_dim, dropout=dropout, output_logits=False)
            context_out_dim = self.context_branch.out_dim
        elif self.context_architecture == "dcnv2":
            self.context_branch = DCNv2(num_numerical, cat_cardinalities, d_token=fm_embedding_dim,
                                        cross_layers=dcn_cross_layers, mlp_hidden=context_hidden_dim, 
                                        dropout=dropout, output_logits=False)
            context_out_dim = self.context_branch.out_dim
        else:
            # Baseline MLP
            self.embeddings = nn.ModuleList([nn.Embedding(card, fm_embedding_dim) for card in cat_cardinalities])
            mlp_in = num_numerical + len(cat_cardinalities) * fm_embedding_dim
            self.context_branch = nn.Sequential(
                nn.Linear(mlp_in, context_hidden_dim),
                nn.BatchNorm1d(context_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(context_hidden_dim, context_hidden_dim)
            )
            context_out_dim = context_hidden_dim

        # 3. Fusion
        fusion_in = seq_out_dim + context_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)
        self.register_buffer('class_indices', torch.arange(num_classes).float())

    def forward(self, seq_x, num_x, cat_x):
        # Context path
        if self.context_architecture in ["deepfm", "dcnv2"]:
            context_vector = self.context_branch(num_x, cat_x)
        else:
            emb_outs = []
            for i, emb in enumerate(self.context_branch.embeddings if hasattr(self.context_branch, 'embeddings') else self.embeddings):
                c = torch.clamp(cat_x[:, i], 0, emb.num_embeddings - 1)
                emb_outs.append(emb(c))
            if len(emb_outs) > 0:
                cat_vector = torch.cat(emb_outs, dim=1)
                context_vector = torch.cat([num_x, cat_vector], dim=1)
            else:
                context_vector = num_x
            context_vector = self.context_branch(context_vector)
            
        # Sequence path
        if self.use_seq and seq_x is not None:
            # seq_x: (batch, seq, ch)
            s_x = seq_x.transpose(1, 2)
            s_x = self.seq_fe(s_x)
            s_x = s_x.transpose(1, 2)
            
            if "bilstm" in self.sequence_architecture:
                lstm_out, _ = self.bilstm(s_x)
                seq_vector, _ = self.seq_attn(lstm_out)
            elif "self_attention" in self.sequence_architecture:
                attn_out, _ = self.mha(s_x, s_x, s_x)
                seq_vector, _ = self.seq_attn(attn_out)
                
            fused = torch.cat([seq_vector, context_vector], dim=1)
        else:
            fused = context_vector
            
        fused = self.fusion(fused)
        logits = self.classifier(fused)
        
        probs = torch.softmax(logits, dim=1)
        expected_val = torch.sum(probs * self.class_indices, dim=1)
        
        return logits, expected_val

def create_model(dataset_kind: str, config: dict, num_numerical: int, cat_cardinalities: list):
    """
    Model factory based on dataset kind.
    """
    if dataset_kind == "student":
        model = StudentV27HybridModel(
            num_classes=3,
            seq_in_channels=1, # G1, G2 feature dimension is 1 per timestep
            num_numerical=num_numerical,
            cat_cardinalities=cat_cardinalities,
            context_architecture=config.get("context_architecture", "mlp_baseline"),
            sequence_architecture=config.get("sequence_architecture", "dsc_bilstm_attention"),
            fm_embedding_dim=config.get("fm_embedding_dim", 8),
            dcn_cross_layers=config.get("dcn_cross_layers", 2),
            context_hidden_dim=config.get("context_hidden_dim", 64),
            sequence_hidden_dim=config.get("sequence_hidden_dim", 64),
            fusion_hidden_dim=config.get("fusion_hidden_dim", 64),
            dropout=config.get("dropout", 0.3)
        )
    elif dataset_kind == "xapi":
        arch = config.get("architecture", "ft_transformer")
        if arch == "ft_transformer":
            model = FTTransformer(
                num_numerical=num_numerical,
                cat_cardinalities=cat_cardinalities,
                d_token=config.get("d_token", 32),
                n_heads=config.get("n_heads", 4),
                n_layers=config.get("n_layers", 3),
                ff_hidden_dim=config.get("ff_hidden_dim", 64),
                attention_dropout=config.get("attention_dropout", 0.2),
                residual_dropout=config.get("residual_dropout", 0.1),
                output_logits=True,
                num_classes=3,
                pooling_type=config.get("pooling_type", "cls")
            )
        elif arch == "deepfm":
            model = DeepFM(num_numerical, cat_cardinalities, 
                           d_token=config.get("d_token", 8), 
                           mlp_hidden=config.get("ff_hidden_dim", 64), 
                           dropout=config.get("residual_dropout", 0.3), 
                           output_logits=True, num_classes=3)
        elif arch == "dcnv2":
            model = DCNv2(num_numerical, cat_cardinalities, 
                          d_token=config.get("d_token", 8), 
                          cross_layers=config.get("n_layers", 2), 
                          mlp_hidden=config.get("ff_hidden_dim", 64), 
                          dropout=config.get("residual_dropout", 0.3), 
                          output_logits=True, num_classes=3)
        else:
            raise ValueError(f"Unknown architecture for xapi: {arch}")
            
        # Add ordinal expected val logic for xAPI model wrapper
        # We can dynamically add it here or modify the models to return it.
        # Let's wrap it nicely.
        class XAPIWrapper(nn.Module):
            def __init__(self, core_model, num_classes):
                super().__init__()
                self.core_model = core_model
                self.register_buffer('class_indices', torch.arange(num_classes).float())
            def forward(self, seq_x, num_x, cat_x):
                # Ignore seq_x for xapi
                logits = self.core_model(num_x, cat_x)
                probs = torch.softmax(logits, dim=1)
                expected_val = torch.sum(probs * self.class_indices, dim=1)
                return logits, expected_val
        
        model = XAPIWrapper(model, 3)
    else:
        raise ValueError("dataset_kind must be student or xapi")
        
    return model


