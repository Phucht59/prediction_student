"""Public, fail-closed API for offline conditional action ranking.

Eligibility is an upstream policy or instructor decision. This module never
decides whether a learner should receive a recommendation, and the validated
conditional evidence does not authorize production runtime use.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

MODEL_SCORE_AUTHORITY = "integrated_conditional_action_head"
OFFLINE_EXECUTION_CONTEXT = "offline_evaluation"
SCIENTIFIC_ACTION_ORDER = (
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
)
ACTION_ALIASES = {
    "ASSESSMENT_COMPLETION": "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY": "STUDY_REGULARITY",
    "STUDY_SCHEDULE": "STUDY_REGULARITY",
    "VLE_ENGAGEMENT": "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE": "QUIZ_OR_RETRIEVAL_PRACTICE",
    "RETRIEVAL_PRACTICE": "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW": "CONTENT_REVIEW",
    "LEARNING_CONSOLIDATION": "CONTENT_REVIEW",
}
_ACTION_INDEX = {
    action_id: index for index, action_id in enumerate(SCIENTIFIC_ACTION_ORDER)
}


@dataclass(frozen=True)
class RankingResult:
    status: str
    actions: tuple[Mapping[str, Any], ...]
    score_authority: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "actions": [dict(item) for item in self.actions],
            "score_authority": self.score_authority,
        }


class ConditionalHybridActionRanker:
    """Rank eligible actions from verified integrated-head outputs only.

    The input score vector must contain the five fixed scientific action slots
    in ``SCIENTIFIC_ACTION_ORDER``. Caller-authored per-action scores are never
    used. This API is restricted to offline evaluation because the end-to-end
    recommendation runtime is not scientifically authorized.
    """

    module_boundary = "conditional_hybrid_action_ranker"
    runtime_authorized = False
    required_score_authority = MODEL_SCORE_AUTHORITY

    def rank_actions(
        self,
        learner_state: Mapping[str, Any],
        eligible_actions: Sequence[Mapping[str, Any]],
        policy_authorized: bool = False,
        *,
        execution_context: str = OFFLINE_EXECUTION_CONTEXT,
    ) -> RankingResult:
        if execution_context != OFFLINE_EXECUTION_CONTEXT:
            return RankingResult("RUNTIME_NOT_AUTHORIZED", ())
        if not policy_authorized:
            return RankingResult("ELIGIBILITY_REQUIRED", ())
        if not eligible_actions:
            return RankingResult("NO_ELIGIBLE_ACTIONS", ())

        score_authority = learner_state.get("score_authority")
        if score_authority != self.required_score_authority:
            return RankingResult("MODEL_OUTPUT_AUTHORITY_REQUIRED", ())

        model_scores = learner_state.get("action_logits")
        if model_scores is None:
            model_scores = learner_state.get("action_scores")
        if model_scores is None:
            return RankingResult(
                "MODEL_OUTPUT_REQUIRED",
                (),
                score_authority=self.required_score_authority,
            )
        if len(model_scores) != len(SCIENTIFIC_ACTION_ORDER):
            raise ValueError(
                "integrated action-head output must contain exactly five canonical slots"
            )

        canonical_scores: list[float] = []
        for raw_score in model_scores:
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise ValueError("action-head scores must be numeric") from exc
            if not isfinite(score):
                raise ValueError("action-head scores must be finite")
            canonical_scores.append(score)

        scored: list[dict[str, Any]] = []
        for action in eligible_actions:
            item = dict(action)
            raw_action_id = item.get("action_id")
            if raw_action_id is None:
                return RankingResult(
                    "SUPPORTED_ACTION_ID_REQUIRED",
                    (),
                    score_authority=self.required_score_authority,
                )
            canonical_action_id = ACTION_ALIASES.get(str(raw_action_id))
            if canonical_action_id is None:
                return RankingResult(
                    "UNSUPPORTED_ACTION_SET",
                    (),
                    score_authority=self.required_score_authority,
                )
            item["canonical_action_id"] = canonical_action_id
            item["score"] = canonical_scores[_ACTION_INDEX[canonical_action_id]]
            item["score_authority"] = self.required_score_authority
            item["rank"] = 0
            scored.append(item)

        scored.sort(
            key=lambda value: (
                -value["score"],
                _ACTION_INDEX[value["canonical_action_id"]],
            )
        )
        ranked: list[dict[str, Any]] = []
        for rank, item in enumerate(scored, 1):
            item["rank"] = rank
            ranked.append(item)
        return RankingResult(
            "RANKED_ELIGIBLE_ACTIONS_OFFLINE",
            tuple(ranked),
            score_authority=self.required_score_authority,
        )


__all__ = [
    "ACTION_ALIASES",
    "ConditionalHybridActionRanker",
    "MODEL_SCORE_AUTHORITY",
    "OFFLINE_EXECUTION_CONTEXT",
    "RankingResult",
    "SCIENTIFIC_ACTION_ORDER",
]
