"""Small ordinal and multi-task models for the frozen V3 protocol."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TrainOnlyTargetScaler:
    """Standardize continuous G3 using outer/inner training statistics only."""
    def __init__(self): self.mean_:float|None=None;self.scale_:float|None=None
    def fit(self,values):
        array=np.asarray(values,dtype=float);self.mean_=float(array.mean());self.scale_=float(array.std(ddof=0)) or 1.0;return self
    def transform(self,values):
        if self.mean_ is None: raise RuntimeError("Target scaler is not fitted.")
        return (np.asarray(values,dtype=float)-self.mean_)/self.scale_
    def inverse_transform(self,values):
        if self.mean_ is None: raise RuntimeError("Target scaler is not fitted.")
        return np.asarray(values,dtype=float)*self.scale_+self.mean_


def coral_targets(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Encode class k as indicators [k > 0, ..., k > K-2]."""
    labels = labels.long().reshape(-1)
    if torch.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("Labels are outside the ordinal class range.")
    thresholds = torch.arange(num_classes - 1, device=labels.device)
    return (labels[:, None] > thresholds[None, :]).float()


def ordinal_logits_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert monotone cumulative logits for P(y>0), P(y>1) to class probabilities."""
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("Three-class ordinal logits must have shape [n, 2].")
    cumulative = torch.sigmoid(logits)
    if torch.any(cumulative[:, 1] > cumulative[:, 0] + 1e-7):
        raise ValueError("Cumulative probabilities are not monotone.")
    probabilities = torch.stack((1-cumulative[:, 0], cumulative[:, 0]-cumulative[:, 1], cumulative[:, 1]), dim=1)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


class SmallBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, hidden_layers: int, dropout: float):
        super().__init__(); layers=[]; width=input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(width, hidden_width), nn.ReLU(), nn.Dropout(dropout)]); width=hidden_width
        self.network=nn.Sequential(*layers); self.output_dim=width
    def forward(self, x): return self.network(x.float())


class OrderedHead(nn.Module):
    """Rank-consistent head: one score and two ordered learned thresholds."""
    def __init__(self, input_dim: int):
        super().__init__(); self.score=nn.Linear(input_dim,1); self.threshold_start=nn.Parameter(torch.tensor(-0.5)); self.threshold_gap_raw=nn.Parameter(torch.tensor(0.0))
    def forward(self,x):
        score=self.score(x); t0=self.threshold_start; t1=t0+F.softplus(self.threshold_gap_raw)
        return torch.cat((score-t0,score-t1),dim=1)


class TabularV3Model(nn.Module):
    def __init__(self,input_dim:int,hidden_width:int=16,hidden_layers:int=1,dropout:float=.15,ordinal:bool=False,regression:bool=False):
        super().__init__(); self.ordinal=ordinal;self.regression=regression;self.backbone=SmallBackbone(input_dim,hidden_width,hidden_layers,dropout)
        self.classification_head=OrderedHead(self.backbone.output_dim) if ordinal else nn.Linear(self.backbone.output_dim,3)
        self.regression_head=nn.Linear(self.backbone.output_dim,1) if regression else None
    def forward(self,x):
        features=self.backbone(x); logits=self.classification_head(features); regression=None if self.regression_head is None else self.regression_head(features).squeeze(1)
        return logits,regression
    def predict_proba(self,x):
        logits,_=self.forward(x); return ordinal_logits_to_probabilities(logits) if self.ordinal else torch.softmax(logits,dim=1)


class SequenceOrdinalV3(nn.Module):
    def __init__(self,cnn_channels:int,cnn_kernel_size:int,lstm_hidden_dim:int,dropout:float,sequence_dropout:float):
        super().__init__();self.kernel=cnn_kernel_size
        self.cnn=nn.Sequential(nn.Conv1d(1,cnn_channels,cnn_kernel_size,padding=cnn_kernel_size//2),nn.BatchNorm1d(cnn_channels),nn.ReLU())
        self.sequence_dropout=nn.Dropout(sequence_dropout);self.bilstm=nn.LSTM(cnn_channels,lstm_hidden_dim,batch_first=True,bidirectional=True);self.head_dropout=nn.Dropout(dropout);self.ordinal_head=OrderedHead(lstm_hidden_dim*2)
    def forward(self,x):
        sequence=self.sequence_dropout(self.cnn(x.float().transpose(1,2)).transpose(1,2));_,(hidden,_)=self.bilstm(sequence);features=self.head_dropout(torch.cat((hidden[-2],hidden[-1]),dim=1));return self.ordinal_head(features),None
    def predict_proba(self,x): return ordinal_logits_to_probabilities(self.forward(x)[0])


def ordinal_bce_loss(logits:torch.Tensor,labels:torch.Tensor)->torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits,coral_targets(labels))


def multitask_loss(classification_loss:torch.Tensor,regression_prediction:torch.Tensor,regression_target:torch.Tensor,lambda_regression:float)->torch.Tensor:
    return classification_loss+float(lambda_regression)*F.mse_loss(regression_prediction,regression_target)
