import pytest

from src.database.connection import DatabaseSettings, connect_with_retry, load_dotenv
from src.database.live_runtime import (
    lookup_case,
    normalize_student_key,
    predict_case,
    query_id_for,
    recommend_case,
)


def test_normalize_student_and_query_id():
    assert normalize_student_key("631334") == "OULAD:631334"
    assert normalize_student_key("OULAD:631334") == "OULAD:631334"
    assert query_id_for("OULAD:631334", "CCC", "2014B", "20pct") == "631334::CCC::2014B::EARLY_20"


def _live_available() -> bool:
    try:
        load_dotenv()
        connection = connect_with_retry(DatabaseSettings.from_environment(), attempts=1)
    except Exception:
        return False
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 FROM prediction.prediction LIMIT 1")
            return cursor.fetchone() is not None
    except Exception:
        connection.rollback()
        return False
    finally:
        connection.close()


def test_lookup_and_predict_frozen_c0():
    if not _live_available():
        pytest.skip("live student_db with C0 predictions is not available")
    payload = lookup_case("631334", "CCC", "2014B", "20")
    assert payload["ok"] is True
    assert payload["enrollment"]["dataset_key"] == "oulad"
    assert payload["cases"]
    prediction = payload["cases"][0]["prediction"]
    assert 0.0 <= float(prediction["risk_probability"]) <= 1.0
    served = predict_case("631334", "CCC", "2014B", "20")
    assert served["ok"] is True
    assert served["refit"] is False
    assert served["prediction"]["prediction_id"] == prediction["prediction_id"]


def test_recommend_uses_frozen_v3_without_changing_c0():
    if not _live_available():
        pytest.skip("live student_db with C0 predictions is not available")
    before = predict_case("631334", "CCC", "2014B", "20")
    payload = recommend_case("631334", "CCC", "2014B", "20", persist=False)
    assert payload["ok"] is True
    assert payload["refit"] is False
    assert payload["ranker"] == "Five-EBM-C0"
    assert payload["decision"]["route"] in {
        "RECOMMEND",
        "HUMAN_REVIEW",
        "NO_FEASIBLE_ACTION",
        "INSUFFICIENT_EVIDENCE",
    }
    after = predict_case("631334", "CCC", "2014B", "20")
    assert after["prediction"]["risk_probability"] == before["prediction"]["risk_probability"]
