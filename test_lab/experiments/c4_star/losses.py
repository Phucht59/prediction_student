"""C4-STAR losses. No ADASYN on Hybrid tensors."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_rank_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    pos = logits[target > 0.5]
    neg = logits[target < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    n = min(pos.numel() * neg.numel(), 2048)
    idx_p = torch.randint(0, pos.numel(), (n,), device=logits.device)
    idx_n = torch.randint(0, neg.numel(), (n,), device=logits.device)
    return F.relu(margin - (pos[idx_p] - neg[idx_n])).mean()


def kd_bce(student_logits: torch.Tensor, teacher_prob: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft-target KD. Uses BCE-with-logits so AMP FP16 autocast is safe."""
    t = max(float(temperature), 1e-3)
    teacher = teacher_prob.float().clamp(1e-4, 1 - 1e-4)
    logits = student_logits.float() / t
    return F.binary_cross_entropy_with_logits(logits, teacher) * (t * t)


def gate_reg(alpha: torch.Tensor, floor: float = 0.0, ceil: float = 1.0) -> torch.Tensor:
    return (F.relu(floor - alpha) + F.relu(alpha - ceil)).mean()


def ssl_reconstruct(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    keep = mask.unsqueeze(-1).to(pred.dtype)
    if keep.sum() <= 0:
        return pred.new_zeros(())
    return ((pred - target).pow(2) * keep).sum() / keep.sum().clamp_min(1.0)
