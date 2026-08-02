from __future__ import annotations
import torch
import torch.nn.functional as F


def ordinal_loss(logits, target, weight):
    thresholds = torch.stack((1 - target[:, 0], target[:, 2]), 1)
    # first threshold is y>0; second conditional term yields monotonic probability
    raw = F.binary_cross_entropy_with_logits(logits, thresholds, reduction="none").mean(1)
    return (raw * weight).sum() / weight.sum().clamp_min(1e-8)


def soft_distribution_loss(logits, target, weight):
    raw = -(target * F.log_softmax(logits, 1)).sum(1)
    return (raw * weight).sum() / weight.sum().clamp_min(1e-8)


def query_ranking_loss(score, expected, query, weight, margin=.05):
    """Pairwise loss only within the original candidate query."""
    total = score.new_zeros(()); n = 0
    for q in torch.unique(query):
        idx = torch.where(query == q)[0]
        if len(idx) < 2: continue
        delta = expected[idx, None] - expected[None, idx]
        pairs = delta > .05
        if pairs.any():
            total = total + (F.relu(margin - (score[idx, None] - score[None, idx]))[pairs]).mean()
            n += 1
    return total / max(n, 1)


def total_loss(outputs, target, weight, query, lambdas):
    return (lambdas["soft"] * soft_distribution_loss(outputs["distribution_logits"], target, weight)
            + lambdas["ordinal"] * ordinal_loss(outputs["ordinal_logits"], target, weight)
            + lambdas["rank"] * query_ranking_loss(outputs["ranking_score"], target @ target.new_tensor([0., 1., 2.]), query, weight))
