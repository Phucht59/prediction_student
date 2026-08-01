from dataclasses import replace

from src.recommend_hybrid.candidate_generator import HybridCandidateGenerator
from src.recommend_hybrid.contracts import CandidateStatus, Stage


def test_candidate_generator_has_no_scores(catalog, prediction_context, observed_state):
    candidates = HybridCandidateGenerator(catalog).generate(prediction_context, observed_state)
    assert candidates
    assert all(not hasattr(candidate, "score") for candidate in candidates)


def test_candidate_generator_stage_filter(catalog, prediction_context, observed_state):
    generator = HybridCandidateGenerator(catalog)
    candidates = generator.generate(prediction_context, observed_state)
    targeted = next(item for item in candidates if item.action.action_id == "TARGETED_PRACTICE")
    assert targeted.status is CandidateStatus.PREREQUISITE_NOT_MET


def test_candidate_generator_missing_evidence(catalog, prediction_context):
    from src.recommend_hybrid.observed_state import ObservedStateBuilder

    observed = ObservedStateBuilder().build(
        stage=Stage.MIDDLE_50, cutoff_day=100, activity_events=(), assessment_events=()
    )
    candidates = HybridCandidateGenerator(catalog).generate(prediction_context, observed)
    activity = next(item for item in candidates if item.action.action_id == "VLE_ENGAGEMENT")
    assert activity.status is CandidateStatus.MISSING_REQUIRED_EVIDENCE


def test_final_stage_has_no_intervention(catalog, prediction_context, observed_state):
    prediction = replace(prediction_context, stage=Stage.FINAL_EVALUATION)
    observed = replace(observed_state, stage=Stage.FINAL_EVALUATION, course_progress=1.0)
    generator = HybridCandidateGenerator(catalog)
    evaluations = generator.generate(prediction, observed)
    assert generator.eligible(evaluations) == ()
    assert all(item.status is CandidateStatus.INELIGIBLE_STAGE for item in evaluations)
