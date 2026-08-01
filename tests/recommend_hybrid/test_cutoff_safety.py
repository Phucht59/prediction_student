from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.observed_state import AssessmentEvent, ObservedStateBuilder


def test_unknown_score_release_makes_grade_unavailable():
    state = ObservedStateBuilder().build(
        stage=Stage.LATE_75,
        cutoff_day=150,
        activity_events=(),
        assessment_events=(AssessmentEvent(50, 45, score=90, score_release_day=None),),
    )
    assert state.grade_trend is None
    assert "grade_trend" in state.missing_evidence
