"""Public, fail-closed API for offline conditional action ranking.

Eligibility is an upstream policy or instructor decision. This module never
decides whether a learner should receive a recommendation, and the validated
conditional evidence does not authorize production runtime use.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from .model import ACTION_COUNT, ActionAwareOutput

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
class IntegratedActionScoreOutput:
    """One learner's five canonical scores produced by the integrated head."""

    scores: tuple[float, ...]
    score_authority: str = MODEL_SCORE_AUTHORITY

    def __post_init__(self) -> None:
        if self.score_authority != MODEL_SCORE_AUTHORITY:
            raise ValueError("unexpected action score authority")
        if len(self.scores) != ACTION_COUNT:
            raise ValueError(
                "integrated action-head output must contain exactly five canonical slots"
            )
        if not all(isfinite(float(value)) for value in self.scores):
            raise ValueError("action-head scores must be finite")

    @classmethod
    def from_head_output(
        cls,
        output: ActionAwareOutput,
        *,
        batch_index: int = 0,
    ) -> "IntegratedActionScoreOutput":
        """Create a score envelope directly from a neural-head forward output."""

        logits = output.action_logits
        if logits.ndim != 2 or logits.shape[1] != ACTION_COUNT:
            raise ValueError("action_logits must have shape [B, 5]")
        if not 0 <= batch_index < logits.shape[0]:
            raise IndexError("batch_index is outside the action-head output")
        values = logits[batch_index].detach().cpu().tolist()
        return cls(tuple(float(value) for value in values))


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
    """Rank eligible actions from integrated-head outputs only.

    The input score vector contains the five fixed scientific action slots in
    ``SCIENTIFIC_ACTION_ORDER``. Caller-authored scores attached to action
    dictionaries are ignored. This API is restricted to offline evaluation
    because the end-to-end recommendation runtime is not authorized.
    """

    module_boundary = "conditional_hybrid_action_ranker"
    runtime_authorized = False
    required_score_authority = MODEL_SCORE_AUTHORITY

    def rank_actions(
        self,
        model_output: IntegratedActionScoreOutput | None,
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
        if model_output is None:
            return RankingResult("MODEL_OUTPUT_REQUIRED", ())
        if not isinstance(model_output, IntegratedActionScoreOutput):
            return RankingResult("INTEGRATED_HEAD_OUTPUT_REQUIRED", ())

        canonical_scores = model_output.scores
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
            item.pop("score", None)
            item.pop("action_probability", None)
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
    "IntegratedActionScoreOutput",
    "MODEL_SCORE_AUTHORITY",
    "OFFLINE_EXECUTION_CONTEXT",
    "RankingResult",
    "SCIENTIFIC_ACTION_ORDER",
]
