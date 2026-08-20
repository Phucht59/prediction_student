"""Training losses. SMOTE/ADASYN are not applied to Hybrid tensors."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def pairwise_rank_loss(logits: torch.Tensor, target: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    pos = logits[target > 0.5]
    neg = logits[target < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    # Sample a bounded number of pairs for stability.
    n = min(pos.numel() * neg.numel(), 2048)
    idx_p = torch.randint(0, pos.numel(), (n,), device=logits.device)
    idx_n = torch.randint(0, neg.numel(), (n,), device=logits.device)
    return F.relu(margin - (pos[idx_p] - neg[idx_n])).mean()


def kd_kl(student_logits: torch.Tensor, teacher_prob: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-3)
    teacher = teacher_prob.clamp(1e-4, 1 - 1e-4)
    teacher_logits = torch.log(teacher) - torch.log1p(-teacher)
    p = torch.sigmoid(student_logits / t)
    q = torch.sigmoid(teacher_logits / t)
    kl = q * (q.clamp_min(1e-8).log() - p.clamp_min(1e-8).log()) + (1 - q) * (
        (1 - q).clamp_min(1e-8).log() - (1 - p).clamp_min(1e-8).log()
    )
    return (t * t) * kl.mean()


def gate_live_penalty(gate: torch.Tensor, temporal_available: torch.Tensor, floor: float = 0.05) -> torch.Tensor:
    avail = temporal_available.to(gate.dtype)
    if avail.sum() <= 0:
        return gate.new_zeros(())
    return (F.relu(floor - gate) * avail).sum() / avail.sum().clamp_min(1.0)
