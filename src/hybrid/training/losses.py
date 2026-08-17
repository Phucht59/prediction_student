"""Frozen Hybrid V1 loss helpers."""
import numpy as np
import torch


def binary_pos_weight(target: np.ndarray) -> torch.Tensor:
    positive = int(np.sum(target))
    negative = int(len(target) - positive)
    if positive == 0 or negative == 0:
        raise ValueError("Both classes are required for Hybrid training")
    return torch.tensor([negative / positive], dtype=torch.float32)
