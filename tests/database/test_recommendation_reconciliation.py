import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _expected_actions() -> int:
    total = 0
    with (
        ROOT / "artifacts/final/recommendation/recommendation_plans.jsonl"
    ).open(encoding="utf-8") as handle:
        for line in handle:
            total += len(json.loads(line)["recommended_actions"])
    return total


def test_risk_profile_count(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM recommendation.risk_profile")
        assert cursor.fetchone()[0] == 15378


def test_plan_count(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM recommendation.plan")
        assert cursor.fetchone()[0] == 15378


def test_action_count(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM recommendation.action")
        assert cursor.fetchone()[0] == _expected_actions()


def test_no_orphan_plan(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) FROM recommendation.plan p
            LEFT JOIN recommendation.risk_profile r USING(risk_profile_id)
            WHERE r.risk_profile_id IS NULL
            """
        )
        assert cursor.fetchone()[0] == 0


def test_no_orphan_action(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) FROM recommendation.action a
            LEFT JOIN recommendation.plan p USING(plan_id)
            WHERE p.plan_id IS NULL
            """
        )
        assert cursor.fetchone()[0] == 0


def test_no_duplicate_risk_profile(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) FROM (
              SELECT run_id,record_pk FROM recommendation.risk_profile
              GROUP BY run_id,record_pk HAVING count(*)>1
            ) duplicate
            """
        )
        assert cursor.fetchone()[0] == 0
