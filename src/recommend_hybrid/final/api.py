"""Public, fail-closed API for conditional action ranking.

Eligibility is an upstream policy or instructor decision. This module never
decides whether a learner should receive a recommendation, and it never falls
back to caller-authored action scores when integrated-head output is absent.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from typing import Any

from .actions import ACTION_COUNT, ACTION_INDEX, ACTION_ORDER, canonical_action_id

EXPECTED_MODEL_ID = "conditional_hybrid_action_ranker"


@dataclass(frozen=True)
class RankingResult:
    status: str
    actions: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "actions": [dict(x) for x in self.actions]}


def _sigmoid(value: float) -> float:
    clipped = min(40.0, max(-40.0, value))
    return 1.0 / (1.0 + exp(-clipped))


def _finite_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numeric model output") from exc
    if not isfinite(number):
        raise ValueError(f"{field} must contain finite model output")
    return number


def _score_map(raw_output: object, *, logits: bool) -> dict[str, float]:
    """Map integrated-head output onto its fixed scientific action identities."""

    values: dict[str, float] = {}
    if isinstance(raw_output, Mapping):
        for supplied_action_id, raw_value in raw_output.items():
            canonical = canonical_action_id(supplied_action_id)
            if canonical in values:
                raise ValueError(f"duplicate model output for action {canonical}")
            number = _finite_float(raw_value, field="action model output")
            values[canonical] = _sigmoid(number) if logits else number
    else:
        if isinstance(raw_output, (str, bytes)) or not isinstance(raw_output, Sequence):
            raise ValueError("action model output must be a mapping or a sequence")
        if len(raw_output) != ACTION_COUNT:
            raise ValueError(
                f"action model output must contain exactly {ACTION_COUNT} canonical slots"
            )
        for canonical, raw_value in zip(ACTION_ORDER, raw_output, strict=True):
            number = _finite_float(raw_value, field="action model output")
            values[canonical] = _sigmoid(number) if logits else number

    if not logits:
        outside_probability_range = [
            action_id for action_id, value in values.items() if not 0.0 <= value <= 1.0
        ]
        if outside_probability_range:
            raise ValueError("action_scores must be probabilities in [0, 1]")
    return values


class ConditionalHybridActionRanker:
    """Rank policy-authorized scientific actions using integrated-head output."""

    module_boundary = "conditional_hybrid_action_ranker"
    runtime_authorized = False
    model_id = EXPECTED_MODEL_ID

    def rank_actions(
        self,
        learner_state: Mapping[str, Any],
        eligible_actions: Sequence[Mapping[str, Any]],
        policy_authorized: bool = False,
    ) -> RankingResult:
        if not policy_authorized:
            return RankingResult("ELIGIBILITY_REQUIRED", ())
        if not eligible_actions:
            return RankingResult("NO_ELIGIBLE_ACTIONS", ())

        supplied_model_id = learner_state.get("model_id")
        if supplied_model_id not in (None, EXPECTED_MODEL_ID):
            return RankingResult("MODEL_AUTHORITY_MISMATCH", ())

        raw_scores = learner_state.get("action_scores")
        raw_logits = learner_state.get("action_logits")
        if raw_scores is not None and raw_logits is not None:
            return RankingResult("AMBIGUOUS_MODEL_OUTPUT", ())
        if raw_scores is None and raw_logits is None:
            return RankingResult("MODEL_SCORES_REQUIRED", ())

        scores = _score_map(
            raw_scores if raw_scores is not None else raw_logits,
            logits=raw_scores is None,
        )

        scored: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action in eligible_actions:
            item = dict(action)
            canonical = canonical_action_id(item.get("action_id"))
            if canonical in seen:
                raise ValueError(f"duplicate eligible action identity {canonical}")
            seen.add(canonical)
            if canonical not in scores:
                return RankingResult("MODEL_SCORE_MISSING", ())
            item["model_action_id"] = canonical
            item["model_action_index"] = ACTION_INDEX[canonical]
            item["score"] = scores[canonical]
            item["score_source"] = EXPECTED_MODEL_ID
            item["rank"] = 0
            scored.append(item)

        scored.sort(
            key=lambda value: (
                -float(value["score"]),
                int(value["model_action_index"]),
            )
        )
        ranked: list[dict[str, Any]] = []
        for rank, item in enumerate(scored, 1):
            item["rank"] = rank
            ranked.append(item)
        return RankingResult("RANKED_ELIGIBLE_ACTIONS", tuple(ranked))


__all__ = [
    "ConditionalHybridActionRanker",
    "EXPECTED_MODEL_ID",
    "RankingResult",
]
