"""Phase 4 context--action model; all heads share one evidence-gated representation."""
from __future__ import annotations

import torch
from torch import nn


class EvidenceGatedRecommender(nn.Module):
    """Ordinal, distribution and ranking heads in one bounded-size model.

    ``x`` must contain only frozen prediction/evidence values and their masks.  It
    deliberately receives action metadata separately so an action cannot become a
    global row-level shortcut.
    """

    def __init__(self, feature_dim: int, actions: int, datasets: int, stages: int,
                 hidden: int = 128, dropout: float = .2) -> None:
        super().__init__()
        self.action = nn.Embedding(actions, 24)
        self.dataset = nn.Embedding(datasets, 8)
        self.stage = nn.Embedding(stages, 8)
        self.context = nn.Sequential(nn.Linear(feature_dim + 16, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(27, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
        self.film = nn.Linear(hidden, hidden * 2)
        self.fusion = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.residual = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.ordinal = nn.Linear(hidden, 2)
        self.distribution = nn.Linear(hidden, 3)
        self.ranking = nn.Linear(hidden, 1, bias=False)  # no unrestricted global action bias

    def representation(self, x: torch.Tensor, action: torch.Tensor, dataset: torch.Tensor,
                       stage: torch.Tensor, action_meta: torch.Tensor) -> torch.Tensor:
        c = self.context(torch.cat((x, self.dataset(dataset), self.stage(stage)), dim=1))
        a = self.action_encoder(torch.cat((self.action(action), action_meta), dim=1))
        scale, shift = self.film(c).chunk(2, dim=1)
        modulated = a * (1 + torch.tanh(scale)) + shift
        h = self.fusion(torch.cat((c, modulated, c * modulated, torch.abs(c - modulated)), dim=1))
        return h + self.gate(h) * self.residual(h)

    def forward(self, x, action, dataset, stage, action_meta):
        h = self.representation(x, action, dataset, stage, action_meta)
        ordinal_logits = self.ordinal(h)
        distribution_logits = self.distribution(h)
        relevance = self.ranking(h).squeeze(1)
        return {"ordinal_logits": ordinal_logits, "distribution_logits": distribution_logits,
                "ranking_score": relevance, "representation": h}

    @staticmethod
    def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
        # Cumulative probabilities are constrained to p(y>1) <= p(y>0).
        p_gt_0 = torch.sigmoid(logits[:, 0])
        p_gt_1 = p_gt_0 * torch.sigmoid(logits[:, 1])
        return torch.stack((1 - p_gt_0, p_gt_0 - p_gt_1, p_gt_1), dim=1)
