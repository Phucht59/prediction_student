import pytest

from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.exceptions import PostCutoffDataError, SensitiveFeatureError
from src.recommend_hybrid.observed_state import ActivityEvent, ObservedStateBuilder


def test_no_post_cutoff_events():
    with pytest.raises(PostCutoffDataError):
        ObservedStateBuilder().build(
            stage=Stage.EARLY_35,
            cutoff_day=50,
            activity_events=(ActivityEvent(50, 1),),
            assessment_events=(),
        )


def test_missing_evidence_not_zero_imputed():
    state = ObservedStateBuilder().build(
        stage=Stage.EARLY_20,
        cutoff_day=30,
        activity_events=(),
        assessment_events=(),
    )
    assert state.activity_level is None
    assert state.inactivity_streak is None
    assert state.assessment_progress is None
    assert "activity_level" in state.missing_evidence


def test_sensitive_features_rejected():
    with pytest.raises(SensitiveFeatureError):
        ObservedStateBuilder().build(
            stage=Stage.MIDDLE_50,
            cutoff_day=100,
            activity_events=(),
            assessment_events=(),
            source_fields=("gender",),
        )


def test_lineage_required(observed_state):
    available = set(observed_state.available_evidence)
    lineage = {item.feature for item in observed_state.feature_lineage}
    assert available <= lineage
    assert all(
        item.observation_end is None or item.observation_end < observed_state.cutoff_day
        for item in observed_state.feature_lineage
    )
