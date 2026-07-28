from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "system.schema_migration",
    "catalog.dataset",
    "catalog.dataset_version",
    "catalog.record",
    "ml.model",
    "ml.run",
    "ml.artifact",
    "ml.metric",
    "recommendation.policy",
    "recommendation.risk_profile",
    "recommendation.plan",
    "recommendation.action",
    "recommendation.review",
    "recommendation.expert_review_case",
    "recommendation.expert_plan_review",
    "recommendation.expert_action_review",
}


def test_expected_16_core_tables(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_schema||'.'||table_name FROM information_schema.tables
            WHERE table_type='BASE TABLE'
              AND table_schema IN ('system','catalog','ml','recommendation')
            """
        )
        assert {row[0] for row in cursor.fetchall()} == EXPECTED


def test_no_application_tables_in_public(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'"
        )
        assert cursor.fetchone()[0] == 0


def test_no_versioned_table_names(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema IN ('system','catalog','ml','recommendation')
              AND table_name ~* '(v4|v5|v6|phase_|study_)'
            """
        )
        assert cursor.fetchall() == []
