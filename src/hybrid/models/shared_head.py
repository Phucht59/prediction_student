"""Phase 6E controlled UCI Hybrid with one shared prediction head."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .components import BiLSTMBranch,ResidualCNNBranch

@dataclass(frozen=True)
class SharedHeadConfig:
    temporal_dim:int;context_dim:int;d_model:int;cnn_channels:int;cnn_blocks:int;bilstm_hidden:int;bilstm_layers:int;context_hidden:int;head_hidden:int;dropout:float

class SharedHeadHybrid(nn.Module):
    model_id="hybrid";display_name="Hybrid"
    def __init__(self,config:SharedHeadConfig):
        super().__init__();self.config=config
        self.temporal_adapter=nn.Sequential(nn.Linear(config.temporal_dim,config.d_model),nn.LayerNorm(config.d_model))
        self.cnn_projection=nn.Identity() if config.d_model==config.cnn_channels else nn.Linear(config.d_model,config.cnn_channels)
        dilations={1:(1,),2:(1,2),3:(1,2,4)}[config.cnn_blocks]
        self.cnn=ResidualCNNBranch(config.cnn_channels,2,dilations,config.dropout);self.bilstm=BiLSTMBranch(config.d_model,config.bilstm_hidden,config.bilstm_layers)
        self.context=nn.Sequential(nn.Linear(config.context_dim,config.context_hidden),nn.LayerNorm(config.context_hidden),nn.GELU(),nn.Dropout(config.dropout))
        width=2*config.cnn_channels+4*config.bilstm_hidden+config.context_hidden+5
        self.head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,config.head_hidden),nn.GELU(),nn.Dropout(config.dropout),nn.Linear(config.head_hidden,1))
    @staticmethod
    def uci_residual(temporal,lengths):
        g1=temporal[:,0,0];g2=temporal[:,1,0];a1=(lengths>=1).to(temporal.dtype);a2=(lengths>=2).to(temporal.dtype);g1=g1*a1;g2=g2*a2
        return torch.stack((g1,g2,(g2-g1)*a2,a1,a2),-1)
    def forward(self,temporal,mask,lengths,context,stage_index=None):
        adapted=self.temporal_adapter(temporal)*mask.unsqueeze(-1).to(temporal.dtype);cnn_input=self.cnn_projection(adapted)*mask.unsqueeze(-1).to(temporal.dtype)
        temporal_rep=torch.cat((self.cnn(cnn_input,mask),self.bilstm(adapted,mask,lengths)),-1);context_rep=self.context(context);residual=self.uci_residual(temporal,lengths)
        return self.head(torch.cat((temporal_rep,context_rep,residual),-1)).squeeze(-1).float()
