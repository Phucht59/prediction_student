import json

import numpy as np
import pandas as pd

from src.recommendation import (
    FORBIDDEN_INPUT_COLUMNS,
    POLICY_VERSION,
    build_recommendation,
    generate_learning_path_report,
    recommendation_to_legacy_row,
    validate_recommendation_schema,
)
from scripts import materialize_recommendation_policy as materializer


def _student_features() -> dict:
    return {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "studytime": 1,
        "failures": 1,
        "schoolsup": "no",
        "famsup": "no",
        "paid": "no",
        "activities": "yes",
        "internet": "no",
        "guardian": "other",
        "Dalc": 2,
        "Walc": 4,
        "goout": 4,
        "absences": 12,
        "G1": 8,
        "G2": 7,
        "G3": 4,
        "true_label": 0,
        "__source_row_number": 10,
        "record_id": 123,
        "dataset_version_id": 1,
        "run_id": "not-for-policy",
    }


def test_recommender_ignores_true_label_target_and_lineage_metadata():
    features_a = _student_features()
    features_b = {
        **features_a,
        "G3": 20,
        "true_label": 2,
        "__source_row_number": 999,
        "record_id": 999,
        "dataset_version_id": 999,
        "run_id": "different",
    }

    rec_a = build_recommendation(features_a, predicted_class=0, confidence=0.81)
    rec_b = build_recommendation(features_b, predicted_class=0, confidence=0.81)

    assert rec_a == rec_b
    serialized = json.dumps(rec_a, ensure_ascii=False)
    assert not any(column in serialized for column in FORBIDDEN_INPUT_COLUMNS)


def test_recommender_excludes_sensitive_advisory_inputs():
    features_a = _student_features()
    features_b = {
        **features_a,
        "school": "MS",
        "sex": "M",
        "address": "R",
        "guardian": "mother",
        "paid": "yes",
        "Dalc": 5,
        "Walc": 5,
        "goout": 1,
    }

    assert build_recommendation(features_a, predicted_class=0, confidence=0.81) == build_recommendation(
        features_b, predicted_class=0, confidence=0.81
    )


def test_low_medium_high_prediction_policies_differ():
    features = _student_features()

    low_perf = build_recommendation(features, predicted_class=0, confidence=0.82)
    medium_perf = build_recommendation(features, predicted_class=1, confidence=0.82)
    high_perf = build_recommendation(features, predicted_class=2, confidence=0.82)

    assert [low_perf["risk_band"], medium_perf["risk_band"], high_perf["risk_band"]] == [
        "High",
        "Medium",
        "Low",
    ]
    assert low_perf["weekly_plan"] != medium_perf["weekly_plan"]
    assert medium_perf["weekly_plan"] != high_perf["weekly_plan"]


def test_low_confidence_adds_cautious_wording():
    rec = build_recommendation(_student_features(), predicted_class=0, confidence=0.42)

    assert rec["confidence_level"] == "low"
    action_text = json.dumps(rec["recommended_actions"], ensure_ascii=False).lower()
    assert "verify" in action_text
    assert "advisor" in action_text


def test_recommendation_output_is_deterministic_and_valid_json_schema():
    features = _student_features()

    first = build_recommendation(features, predicted_class=1, confidence=0.67)
    second = build_recommendation(features, predicted_class=1, confidence=0.67)

    assert first == second
    validate_recommendation_schema(first)
    assert "advisor" in first["disclaimer"].lower()
    assert json.loads(json.dumps(first, ensure_ascii=False)) == first


def test_every_action_has_rationale_and_no_vague_study_more_text():
    rec = build_recommendation(_student_features(), predicted_class=0, confidence=0.91)
    risk_codes = []

    for action in rec["recommended_actions"]:
        assert action["action"]
        assert action["frequency"]
        assert action["duration"]
        assert action["reason"]
        assert "study more" not in action["action"].lower()
        assert "study more" not in action["reason"].lower()
        risk_codes.append(action["risk_code"])

    assert len(risk_codes) == len(set(risk_codes))


def test_policy_version_is_carried_to_persistence_row():
    recommendation = build_recommendation(_student_features(), predicted_class=0, confidence=0.77)
    row = recommendation_to_legacy_row(
        row_index=0,
        predicted_class=0,
        confidence=0.77,
        recommendation=recommendation,
    )

    assert row["policy_version"] == POLICY_VERSION
    payload = json.loads(row["learning_path"])
    assert payload["explanation"]["policy_version"] == POLICY_VERSION


def test_generate_learning_path_report_exposes_standardized_policy_rows():
    frame = pd.DataFrame(
        [
            _student_features(),
            {**_student_features(), "absences": 0, "studytime": 3, "G2": 14, "__source_row_number": 25},
        ]
    )
    frame.index = [50, 60]
    report = generate_learning_path_report(
        original_features=frame,
        predictions=np.array([0, 2]),
        confidences=np.array([0.84, 0.62]),
        dataset_name="student-mat",
    )

    assert report["source_row_index"].tolist() == [10, 25]
    assert list(report["policy_version"]) == [POLICY_VERSION, POLICY_VERSION]
    assert set(report["risk_band"]) == {"High", "Low"}
    for value in report["learning_path"]:
        validate_recommendation_schema(json.loads(value))


def test_materializer_query_does_not_select_ground_truth():
    query = materializer.PREDICTION_CONTEXT_QUERY.lower()

    assert "true_label" not in query
    assert "target_label" not in query
    assert "select" in query
    assert "update" not in query
    assert "delete" not in query


def test_materializer_does_not_pass_target_or_database_metadata(monkeypatch):
    captured = {}

    def fake_build_recommendation(features, predicted_class, confidence, probability=None):
        captured["features"] = dict(features)
        return build_recommendation(features, predicted_class, confidence, probability)

    monkeypatch.setattr(materializer, "build_recommendation", fake_build_recommendation)
    row = {
        "prediction_id": 1,
        "predicted_label": 0,
        "confidence": 0.8,
        "probability": {"Low": 0.8, "Medium": 0.1, "High": 0.1},
        "raw_payload": _student_features(),
    }

    payload = materializer.build_materialized_payload(row, POLICY_VERSION)

    assert payload["policy_version"] == POLICY_VERSION
    assert not (set(captured["features"]) & FORBIDDEN_INPUT_COLUMNS)
    assert "G3" not in captured["features"]
    assert "record_id" not in captured["features"]
    assert "dataset_version_id" not in captured["features"]


def test_materializer_output_is_independent_of_source_row_order():
    rows = [
        {
            "prediction_id": 1,
            "source_row_number": 20,
            "predicted_label": 0,
            "confidence": 0.8,
            "probability": {"Low": 0.8, "Medium": 0.1, "High": 0.1},
            "raw_payload": _student_features(),
        },
        {
            "prediction_id": 2,
            "source_row_number": 10,
            "predicted_label": 2,
            "confidence": 0.7,
            "probability": {"Low": 0.1, "Medium": 0.2, "High": 0.7},
            "raw_payload": {**_student_features(), "absences": 0, "studytime": 3, "G2": 14},
        },
    ]

    forward = {
        row["prediction_id"]: materializer.build_materialized_payload(row, POLICY_VERSION)
        for row in rows
    }
    reverse = {
        row["prediction_id"]: materializer.build_materialized_payload(row, POLICY_VERSION)
        for row in reversed(rows)
    }

    assert forward == reverse


class _FakeRecommendationCursor:
    def __init__(self, existing):
        self.existing = existing
        self.rowcount = 0
        self.executed = []

    def execute(self, query, params):
        self.executed.append(query.strip().lower())
        if query.strip().lower().startswith("insert"):
            self.rowcount = 0

    def fetchone(self):
        return self.existing


def test_materializer_retry_identical_row_is_idempotent():
    row = {
        "prediction_id": 1,
        "predicted_label": 0,
        "confidence": 0.8,
        "probability": {"Low": 0.8, "Medium": 0.1, "High": 0.1},
        "raw_payload": _student_features(),
    }
    payload = materializer.build_materialized_payload(row, POLICY_VERSION)
    cursor = _FakeRecommendationCursor(
        {
            "risk_band": payload["risk_band"],
            "learning_path": payload["learning_path"],
            "explanation": payload["explanation"],
        }
    )

    result = materializer.insert_or_compare_recommendation(cursor, payload)

    assert result == "identical"
    assert any(query.startswith("insert") for query in cursor.executed)
    assert not any(query.startswith("update") or query.startswith("delete") for query in cursor.executed)
