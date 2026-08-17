"""Mask-safe components for the parallel CNN and BiLSTM Hybrid."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    result = (values * weights).sum(dim=1) / count
    return torch.where(mask.any(dim=1, keepdim=True), result, torch.zeros_like(result))


def masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~mask.unsqueeze(-1), torch.finfo(values.dtype).min)
    result = masked.max(dim=1).values
    return torch.where(mask.any(dim=1, keepdim=True), result, torch.zeros_like(result))


def masked_mean_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.cat((masked_mean(values, mask), masked_max(values, mask)), dim=-1)


def temporal_summaries(values: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean/max/sum/last/delta plus normalized length, with exact zero for empty sequences."""
    weights=mask.unsqueeze(-1).to(values.dtype);summed=(values*weights).sum(1);mean=masked_mean(values,mask);maximum=masked_max(values,mask)
    safe=(lengths-1).clamp_min(0);batch=torch.arange(values.shape[0],device=values.device);last=values[batch,safe]* (lengths>0).unsqueeze(-1)
    previous_index=(lengths-2).clamp_min(0);previous=values[batch,previous_index];delta=(last-previous)*(lengths>1).unsqueeze(-1)
    normalized=(lengths.to(values.dtype)/values.shape[1]).unsqueeze(-1);summary=torch.cat((mean,maximum,summed,last,delta,normalized),-1)
    return summary*(lengths>0).unsqueeze(-1).to(values.dtype)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _conv(self, layer: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        left = self.pad // 2
        return layer(F.pad(x, (left, self.pad - left)))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        keep = mask.unsqueeze(1).to(sequence.dtype)
        residual = sequence * keep
        x = self._conv(self.conv1, residual) * keep
        x = self.dropout(self.activation(x)) * keep
        x = self._conv(self.conv2, x) * keep
        return (residual + self.dropout(x)) * keep


class ResidualCNNBranch(nn.Module):
    def __init__(self, channels: int = 64, kernel_size: int = 2, dilations=(1, 2, 4), dropout: float = 0.2):
        super().__init__()
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(channels, kernel_size, dilation, dropout) for dilation in dilations
        )

    def forward(self, adapted: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = adapted.transpose(1, 2)
        for block in self.blocks:
            x = block(x, mask)
        return masked_mean_max(x.transpose(1, 2), mask)


class BiLSTMBranch(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=0.0 if num_layers == 1 else 0.2)

    def forward(self, adapted: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch, timesteps, _ = adapted.shape
        output = adapted.new_zeros((batch, timesteps, self.lstm.hidden_size * 2))
        positive = lengths > 0
        if positive.any():
            subset = adapted[positive]
            subset_lengths = lengths[positive]
            packed = pack_padded_sequence(subset, subset_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed)
            unpacked, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=timesteps)
            output[positive] = unpacked.to(output.dtype)
        output = output * mask.unsqueeze(-1).to(output.dtype)
        return masked_mean_max(output, mask)
