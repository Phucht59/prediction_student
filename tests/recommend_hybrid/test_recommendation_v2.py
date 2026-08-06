from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipelines import oulad
from src.recommend_hybrid.v2.eligibility import (
    EligibilityDecision,
    apply_eligibility_policy,
    normalized_binary_entropy,
    select_eligibility_policy,
)
from src.recommend_hybrid.v2.evaluation import eligibility_metrics, ranking_metrics
from src.recommend_hybrid.v2.ranking import RankingWeights, utility_scores
from src.recommend_hybrid.v2.simulation import SimulationStrength, simulate_action_inputs
from src.recommend_hybrid.v2.taxonomy import LEARNED_ACTIONS, audit_taxonomy


def test_eligibility_is_validation_selected_and_full_population_metrics_are_finite() -> None:
    target = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int8)
    risk = np.array([0.1, 0.2, 0.4, 0.55, 0.7, 0.8, 0.9, 0.6])
    need = np.array([0.1, 0.2, 0.2, 0.5, 0.7, 0.8, 0.9, 0.1])
    policy, _ = select_eligibility_policy(
        validation_target=target,
        validation_risk_probability=risk,
        validation_need_score=need,
        validation_entropy=normalized_binary_entropy(risk),
        validation_seed_disagreement=np.zeros(len(risk)),
        risk_thresholds=(0.45, 0.55),
        minimum_needs=(0.2, 0.4),
        defer_entropies=(0.95,),
        defer_disagreements=(0.2,),
    )
    decisions = apply_eligibility_policy(
        risk_probability=risk,
        need_score=need,
        predictive_entropy=normalized_binary_entropy(risk),
        seed_disagreement=np.zeros(len(risk)),
        policy=policy,
    )
    assert set(decisions).issubset({decision.value for decision in EligibilityDecision})
    metrics = eligibility_metrics(target=target, risk_probability=risk, decisions=decisions)
    assert metrics["population"] == len(target)
    assert 0.0 <= metrics["intervention_rate"] <= 1.0
    assert metrics["risk_coverage_curve"]


def test_utility_ranking_reports_more_than_precision_at_one() -> None:
    probability = np.array([[0.8, 0.4, 0.2], [0.3, 0.7, 0.6]])
    need = np.array([[0.9, 0.3, 0.4], [0.5, 0.8, 0.7]])
    reduction = np.array([[0.7, 0.1, 0.2], [0.2, 0.6, 0.5]])
    confidence = np.full_like(probability, 0.8)
    workload = np.array([[0.8, 0.2, 0.4], [0.8, 0.2, 0.4]])
    uncertainty = np.full_like(probability, 0.1)
    mask = np.ones_like(probability, dtype=bool)
    target = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int8)
    scores = utility_scores(
        action_probability=probability,
        need_severity=need,
        simulated_risk_reduction=reduction,
        evidence_confidence=confidence,
        workload=workload,
        uncertainty=uncertainty,
        mask=mask,
        weights=RankingWeights(0.45, 0.2, 0.2, 0.1, 0.05, 0.05),
    )
    metrics = ranking_metrics(scores, target, mask)
    assert metrics["precision_at_1"] == 1.0
    assert metrics["recall_at_3"] == 1.0
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["action_diversity"] == 2


def test_taxonomy_keeps_governance_outside_learned_ranker() -> None:
    rows: list[dict[str, object]] = []
    for record in range(40):
        for stage in ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"):
            for action_index, action in enumerate(LEARNED_ACTIONS):
                rows.append(
                    {
                        "record_id": str(record),
                        "stage": stage,
                        "action_id": action,
                        "silver_label": int((record + action_index) % 3 == 0),
                    }
                )
    audit = audit_taxonomy(
        pd.DataFrame(rows),
        stages=("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75"),
        minimum_positive=20,
    )
    assert audit["learned_action_count"] == 5
    assert audit["governance_routes_ranked_by_action_head"] is False
    assert audit["research_candidates_activated"] is False


def _synthetic_stage_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, weeks = 3, 5
    base = np.zeros((rows, weeks, len(oulad.BASE_CHANNELS)), dtype=np.float32)
    index = {name: position for position, name in enumerate(oulad.BASE_CHANNELS)}
    base[:, :3, index["total_clicks"]] = 4.0
    base[:, :3, index["active_days"]] = 1.0
    base[:, :3, index["unique_sites"]] = 1.0
    base[:, :3, index["unique_activity_types"]] = 1.0
    base[:, :3, index["content_clicks"]] = 2.0
    base[:, :, index["score_missing_mask"]] = 1.0
    lengths = np.array([5, 4, 3], dtype=np.int64)
    mask = np.arange(weeks)[None, :] < lengths[:, None]
    base[~mask] = 0.0
    full = oulad._dynamic(base, mask)
    context = np.array(
        [[0.5, 5.0, 5.0, 0.0], [0.5, 4.0, 6.0, 0.0], [0.5, 3.0, 7.0, 0.0]],
        dtype=np.float32,
    )
    return full, lengths, context


def test_simulation_rebuilds_dynamic_and_aggregate_without_future_edits() -> None:
    full, lengths, context = _synthetic_stage_inputs()
    simulated = simulate_action_inputs(
        full_sequence=full,
        lengths=lengths,
        stage_context=context,
        action_id="STUDY_REGULARITY",
        strength=SimulationStrength.MODERATE,
        applicable=np.array([True, True, False]),
    )
    assert simulated.full_sequence.shape == full.shape
    assert simulated.raw_aggregate.shape == (3, 165)
    assert simulated.constraint_violations == ()
    for row, length in enumerate(lengths):
        assert np.all(simulated.full_sequence[row, int(length) :, :] == 0.0)
    assert np.any(simulated.full_sequence[0] != full[0])
    assert np.array_equal(simulated.full_sequence[2], full[2])
