import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_no_fake_expert_review(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM recommendation.review WHERE review_type='expert'")
        assert cursor.fetchone()[0] == 0


def test_expert_status_pending():
    payload = json.loads(
        (ROOT / "artifacts/final/final_results.json").read_text(encoding="utf-8")
    )
    assert payload["recommendation"]["expert_status"]["value"] == "PENDING_EXPERT_LABELS"


def test_future_oulad_locked():
    payload = json.loads(
        (ROOT / "artifacts/final/final_results.json").read_text(encoding="utf-8")
    )
    assert str(payload["future_oulad"]).startswith("LOCKED")
    assert payload["future_oulad_executed"] is False
