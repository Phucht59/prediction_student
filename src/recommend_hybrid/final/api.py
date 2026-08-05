"""Public, fail-closed API for conditional action ranking.

Eligibility is an upstream policy or instructor decision. This module never
decides whether a learner should receive a recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RankingResult:
    status: str
    actions: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "actions": [dict(x) for x in self.actions]}


class ConditionalHybridActionRanker:
    """Rank caller-supplied eligible actions; do not issue recommendations."""

    module_boundary = "conditional_hybrid_action_ranker"
    runtime_authorized = False

    def rank_actions(
        self,
        learner_state: Mapping[str, Any],
        eligible_actions: Sequence[Mapping[str, Any]],
        policy_authorized: bool = True,
    ) -> RankingResult:
        if not policy_authorized:
            return RankingResult("ELIGIBILITY_REQUIRED", ())
        if not eligible_actions:
            return RankingResult("NO_ELIGIBLE_ACTIONS", ())
        model_scores = learner_state.get("action_scores", learner_state.get("action_logits"))
        if model_scores is not None and len(model_scores) != len(eligible_actions):
            raise ValueError("action score output must align with eligible_actions")
        scored: list[dict[str, Any]] = []
        for index, action in enumerate(eligible_actions):
            item = dict(action)
            item.setdefault("action_id", index)
            score = (
                model_scores[index]
                if model_scores is not None
                else item.get("score", item.get("action_probability", 0.0))
            )
            try:
                item["score"] = float(score)
            except (TypeError, ValueError):
                item["score"] = 0.0
            item["rank"] = 0
            scored.append(item)
        scored.sort(key=lambda value: (-value["score"], str(value["action_id"])))
        ranked = []
        for rank, item in enumerate(scored, 1):
            item["rank"] = rank
            ranked.append(item)
        return RankingResult("RANKED_ELIGIBLE_ACTIONS", tuple(ranked))


__all__ = ["ConditionalHybridActionRanker", "RankingResult"]
