from __future__ import annotations

from src.recommend_hybrid.hybrid_only_final.scorer import (
    HybridActionEvidence,
    HybridOnlyScoreConfig,
    score_hybrid_actions,
    semantic_evidence_strength,
)


def config(**overrides) -> HybridOnlyScoreConfig:
    values = {
        "version": "test",
        "risk_weight": 0.8,
        "evidence_weight": 0.2,
        "need_weight": 0.1,
        "certainty_weight": 0.1,
        "workload_weight": 0.05,
        "minimum_risk_reduction": 0.01,
        "maximum_uncertainty": 0.2,
        "minimum_evidence": 0.4,
        "minimum_top_margin": 0.01,
        "minimum_top_score": 0.1,
        "risk_scale": 0.1,
        "need_scale": 1.0,
        "uncertainty_scale": 0.1,
        "workload_scale_minutes": 150.0,
    }
    values.update(overrides)
    return HybridOnlyScoreConfig(**values)


def action(
    action_id: str,
    risk: float,
    *,
    evidence: float = 0.5,
    need: float = 0.5,
    uncertainty: float = 0.05,
    workload: int = 90,
    **overrides,
) -> HybridActionEvidence:
    values = {
        "action_id": action_id,
        "risk_reduction": risk,
        "evidence_strength": evidence,
        "need_score": need,
        "uncertainty": uncertainty,
        "workload_minutes": workload,
    }
    values.update(overrides)
    return HybridActionEvidence(**values)


def test_semantic_evidence_treats_due_assessment_as_direct_evidence() -> None:
    assert semantic_evidence_strength("ASSESSMENT_COMPLETION", 0.1) == 0.8
    assert semantic_evidence_strength("VLE_ENGAGEMENT", 0.7) == 0.7


def test_scorer_selects_highest_reliable_hybrid_action() -> None:
    decision = score_hybrid_actions(
        [
            action("STUDY_SCHEDULE", 0.08, workload=30),
            action("VLE_ENGAGEMENT", 0.04, workload=90),
        ],
        config(),
    )
    assert decision.issued
    assert decision.selected_action is not None
    assert decision.selected_action.action_id == "STUDY_SCHEDULE"


def test_candidate_filters_apply_before_ranking() -> None:
    decision = score_hybrid_actions(
        [
            action("STUDY_SCHEDULE", 0.20, uncertainty=0.8),
            action("VLE_ENGAGEMENT", 0.05, uncertainty=0.02),
        ],
        config(maximum_uncertainty=0.2, minimum_top_margin=0.0),
    )
    assert decision.selected_action is not None
    assert decision.selected_action.action_id == "VLE_ENGAGEMENT"


def test_scorer_abstains_on_small_top_margin() -> None:
    decision = score_hybrid_actions(
        [
            action("STUDY_SCHEDULE", 0.05),
            action("VLE_ENGAGEMENT", 0.049),
        ],
        config(minimum_top_margin=0.2),
    )
    assert not decision.issued
    assert decision.abstention_reason == "TOP_MARGIN_BELOW_THRESHOLD"


def test_scorer_abstains_when_no_action_has_enough_hybrid_reduction() -> None:
    decision = score_hybrid_actions(
        [action("STUDY_SCHEDULE", 0.001)],
        config(minimum_risk_reduction=0.01),
    )
    assert not decision.issued
    assert decision.abstention_reason == "NO_ELIGIBLE_CONFIDENT_ACTION"


def test_scorer_is_deterministic() -> None:
    candidates = [
        action("STUDY_SCHEDULE", 0.06, workload=30),
        action("VLE_ENGAGEMENT", 0.04, workload=90),
    ]
    first = score_hybrid_actions(candidates, config())
    second = score_hybrid_actions(reversed(candidates), config())
    assert first == second
