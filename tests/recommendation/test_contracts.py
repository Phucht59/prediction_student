from __future__ import annotations

import pandas as pd
import pytest

from src.recommendation.contracts.prediction import PredictionArtifactAdapter
from src.recommendation.contracts.state import make_case_id
from src.recommendation.state.builder import StudentStateBuilder
from src.recommendation.state.validation import validate_student_state


def _prediction_rows(*, final: bool = False, score: float = 0.4) -> pd.DataFrame:
    stage = "FINAL-100" if final else "20pct"
    rows = []
    for seed in (42, 1201, 2026):
        rows.append({
            "record_id": "r1", "group_id": "student-1", "score": score + seed / 100000,
            "model": "Hybrid", "domain": "oulad", "stage": stage, "outer_fold": 0,
            "seed": seed, "threshold": 0.5,
        })
    return pd.DataFrame(rows)


def _features() -> pd.DataFrame:
    return pd.DataFrame([{
        "dataset": "oulad", "student_id": "student-1", "record_id": "r1", "stage": "20pct",
        "outer_fold": 0, "module": "AAA", "presentation": "2013J", "enrollment_identity": "r1",
        "inactive_streak": 1.0, "active_days_ratio": 0.5,
        "assessment_completion": 0.25, "course_progress": 0.2, "missing_assessments": 2,
        "quiz_activity": 3.0, "vle_available": True, "source_feature_version": "test",
    }])


def test_case_id_is_deterministic_and_identity_based():
    assert make_case_id("oulad", "r1", "20pct") == make_case_id("oulad", "r1", "20pct")
    assert make_case_id("oulad", "r1", "20pct") != make_case_id("oulad", "r1", "35pct")


def test_adapter_aggregates_frozen_seeds_and_excludes_final_by_scope(tmp_path):
    path = tmp_path / "predictions.parquet"
    _prediction_rows().to_parquet(path, index=False)
    adapter = PredictionArtifactAdapter.from_parquet(path, dataset="oulad", stages=("20pct",))
    assert len(adapter.records) == 1
    assert adapter.records.iloc[0].prediction_seed_count == 3
    assert adapter.records.iloc[0].risk_probability == pytest.approx(0.4 + (42 + 1201 + 2026) / 3 / 100000)


def test_final_is_rejected_when_not_in_adapter_scope(tmp_path):
    path = tmp_path / "predictions.parquet"
    _prediction_rows(final=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="no frozen prediction rows"):
        PredictionArtifactAdapter.from_parquet(path, dataset="oulad", stages=("20pct",))


def test_builder_join_and_validation():
    predictions = pd.DataFrame([{
        "dataset": "oulad", "student_id": "student-1", "record_id": "r1", "stage": "20pct",
        "outer_fold": 0, "risk_probability": 0.4, "prediction_threshold": 0.5,
        "prediction_source_version": "source#sha256=x", "prediction_seed_count": 3,
    }])
    state = StudentStateBuilder().build(predictions, _features())
    assert state.loc[0, "case_id"] == make_case_id("oulad", "r1", "20pct")
    assert validate_student_state(state) == []
    assert state.loc[0, "risk_band"] == "medium"


def test_duplicate_detection():
    predictions = pd.DataFrame([{
        "dataset": "oulad", "student_id": "student-1", "record_id": "r1", "stage": "20pct",
        "outer_fold": 0, "risk_probability": 0.4, "prediction_source_version": "source",
    }, {
        "dataset": "oulad", "student_id": "student-1", "record_id": "r1", "stage": "20pct",
        "outer_fold": 0, "risk_probability": 0.4, "prediction_source_version": "source",
    }])
    with pytest.raises(ValueError, match="not unique"):
        StudentStateBuilder().build(predictions, _features())


def test_leakage_blacklist_is_rejected():
    predictions = pd.DataFrame([{
        "dataset": "oulad", "student_id": "student-1", "record_id": "r1", "stage": "20pct",
        "outer_fold": 0, "risk_probability": 0.4, "prediction_source_version": "source",
    }])
    bad = _features().assign(target=0)
    with pytest.raises(ValueError, match="forbidden source fields"):
        StudentStateBuilder().build(predictions, bad)


def test_probability_bounds_are_validated():
    predictions = pd.DataFrame([{
        "dataset": "oulad", "student_id": "student-1", "record_id": "r1", "stage": "20pct",
        "outer_fold": 0, "risk_probability": 1.2, "prediction_source_version": "source",
    }])
    with pytest.raises(ValueError, match="probability must be in"):
        StudentStateBuilder().build(predictions, _features())
