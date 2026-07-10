import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_SQL = (PROJECT_ROOT / "database" / "migrations" / "001_create_source_ml_schema.sql").read_text(
    encoding="utf-8"
)
MIGRATION_002_SQL = (
    PROJECT_ROOT / "database" / "migrations" / "002_allow_append_only_recommendation_policy_versions.sql"
).read_text(encoding="utf-8")
MIGRATION_003_SQL = (
    PROJECT_ROOT / "database" / "migrations" / "003_add_source_record_targets.sql"
).read_text(encoding="utf-8")
DEFAULT_WINDOWS_PSQL = Path("C:/Program Files/PostgreSQL/17/bin/psql.exe")
PSQL_PATH = os.getenv("PSQL_PATH") or shutil.which("psql") or (
    str(DEFAULT_WINDOWS_PSQL) if DEFAULT_WINDOWS_PSQL.exists() else None
)

pytestmark = pytest.mark.skipif(
    not os.getenv("POSTGRES_TEST_DSN") or not os.getenv("POSTGRES_TEST_APP_DSN") or not PSQL_PATH,
    reason="Set POSTGRES_TEST_DSN, POSTGRES_TEST_APP_DSN, and provide psql to run PostgreSQL lineage integration tests.",
)


def app_role_dsn() -> str:
    return os.environ["POSTGRES_TEST_APP_DSN"]


def run_sql(sql: str, *, schema: str | None = None, expect_ok: bool = True, dsn: str | None = None) -> str:
    dsn = dsn or os.environ["POSTGRES_TEST_DSN"]
    if expect_ok:
        full_sql = f'SET search_path TO "{schema}";\n{sql}' if schema else sql
    else:
        if schema is None:
            raise ValueError("Expected-failure SQL must run inside a schema for these tests.")
        full_sql = f"""
        SET search_path TO "{schema}";
        DO $expected_failure$
        DECLARE
            failed boolean := false;
        BEGIN
            BEGIN
                EXECUTE $statement${sql}$statement$;
            EXCEPTION WHEN OTHERS THEN
                failed := true;
            END;
            IF NOT failed THEN
                RAISE EXCEPTION 'expected SQL statement to fail, but it succeeded';
            END IF;
        END
        $expected_failure$;
        """
    result = subprocess.run(
        [PSQL_PATH, "-qAt", "-v", "ON_ERROR_STOP=1", "-d", dsn, "-c", full_sql],
        text=True,
        capture_output=True,
        timeout=20,
    )
    if expect_ok and result.returncode != 0:
        raise AssertionError(f"psql failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nSQL:\n{sql}")
    if not expect_ok and result.returncode != 0:
        raise AssertionError(f"expected-failure wrapper failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nSQL:\n{sql}")
    return result.stdout.strip() if result.stdout is not None else ""


def run_sql_file(sql: str, *, schema: str) -> None:
    dsn = os.environ["POSTGRES_TEST_DSN"]
    with tempfile.NamedTemporaryFile("w", suffix=".sql", encoding="utf-8", delete=False) as handle:
        handle.write(f'SET search_path TO "{schema}";\n')
        handle.write(sql)
        path = handle.name
    try:
        result = subprocess.run(
            [PSQL_PATH, "-qAt", "-v", "ON_ERROR_STOP=1", "-d", dsn, "-f", path],
            text=True,
            capture_output=True,
            timeout=20,
        )
        if result.returncode != 0:
            raise AssertionError(f"psql failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    finally:
        Path(path).unlink(missing_ok=True)


def create_schema() -> str:
    schema = f"source_ml_test_{uuid.uuid4().hex}"
    run_sql(f'CREATE SCHEMA "{schema}";')
    run_sql_file(MIGRATION_SQL, schema=schema)
    run_sql_file(MIGRATION_002_SQL, schema=schema)
    run_sql_file(MIGRATION_003_SQL, schema=schema)
    return schema


def drop_schema(schema: str) -> None:
    run_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')


def insert_dataset_version(schema: str, *, dataset_code="student-mat", row_count=2, suffix="") -> int:
    output = run_sql(
        f"""
        INSERT INTO source_dataset_versions (
            dataset_code, source_locator, content_hash,
            ingestion_contract, ingestion_contract_hash,
            row_count, metadata
        )
        VALUES ('{dataset_code}', '{dataset_code}{suffix}.csv', 'content{suffix}', '{{}}', 'contract{suffix}', {row_count}, '{{}}')
        RETURNING dataset_version_id;
        """,
        schema=schema,
    )
    return int(output.splitlines()[-1])


def insert_running_run(schema: str, *, run_id: str, dataset_version_id: int, dsn: str | None = None) -> None:
    run_sql(
        f"""
        INSERT INTO ml_experiment_runs (
            run_id, dataset_version_id, model_name, task_type,
            target_definition, target_definition_hash,
            split_manifest_uri, split_manifest_hash,
            git_commit, working_tree_state,
            environment_lock_uri, environment_lock_hash,
            train_config, artifact_uri,
            status, started_at
        )
        VALUES (
            '{run_id}', {dataset_version_id}, 'model', 'classification',
            '{{"label_mapping":{{"0":"Low","1":"Medium","2":"High"}}}}', 'target',
            'split.json', 'split',
            'commit', 'clean',
            'env.yml', 'envhash',
            '{{}}', 'artifact',
            'running', NOW()
        );
        """,
        schema=schema,
        dsn=dsn,
    )


def setup_completed_run(schema: str, *, dsn: str | None = None) -> tuple[str, int, list[int], int]:
    run_id = str(uuid.uuid4())
    dataset_version_id = insert_dataset_version(schema, suffix=uuid.uuid4().hex[:8])
    run_sql(
        f"""
        INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
        VALUES
            ({dataset_version_id}, 0, '{{"G1": 8, "G2": 9}}'),
            ({dataset_version_id}, 1, '{{"G1": 12, "G2": 13}}');
        """,
        schema=schema,
        dsn=dsn,
    )
    insert_running_run(schema, run_id=run_id, dataset_version_id=dataset_version_id, dsn=dsn)
    record_ids = [
        int(value)
        for value in run_sql(
            f"SELECT record_id FROM source_records WHERE dataset_version_id = {dataset_version_id} ORDER BY source_row_number;",
            schema=schema,
            dsn=dsn,
        ).splitlines()
    ]
    run_sql(
        f"""
        INSERT INTO ml_run_record_splits (run_id, dataset_version_id, record_id, split_name)
        VALUES
            ('{run_id}', {dataset_version_id}, {record_ids[0]}, 'train'),
            ('{run_id}', {dataset_version_id}, {record_ids[1]}, 'test');
        """,
        schema=schema,
        dsn=dsn,
    )
    prediction_id = int(
        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {record_ids[1]}, 'test', 1, 1, 0.9, '{{"Low":0.05,"Medium":0.9,"High":0.05}}')
            RETURNING prediction_id;
            """,
            schema=schema,
            dsn=dsn,
        ).splitlines()[-1]
    )
    run_sql(
        f"UPDATE ml_experiment_runs SET status = 'completed', completed_at = NOW() WHERE run_id = '{run_id}';",
        schema=schema,
        dsn=dsn,
    )
    return run_id, dataset_version_id, record_ids, prediction_id


def test_postgres_lineage_hard_gates():
    schema = create_schema()
    run_id = str(uuid.uuid4())
    try:
        dataset_version_id = insert_dataset_version(schema)

        run_sql(
            f"""
            INSERT INTO ml_experiment_runs (
                run_id, dataset_version_id, model_name, task_type,
                target_definition, target_definition_hash,
                split_manifest_uri, split_manifest_hash,
                git_commit, working_tree_state,
                environment_lock_uri, environment_lock_hash,
                train_config, artifact_uri,
                status, started_at, completed_at
            )
            VALUES (
                '{run_id}', {dataset_version_id}, 'model', 'classification',
                '{{}}', 'target', 'split.json', 'split',
                'commit', 'clean',
                'env.yml', 'envhash',
                '{{}}', 'artifact',
                'completed', NOW(), NOW()
            );
            """,
            schema=schema,
            expect_ok=False,
        )

        run_sql(
            f"""
            INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
            VALUES ({dataset_version_id}, 0, '{{"G3": 8}}'), ({dataset_version_id}, 1, '{{"G3": 12}}');
            """,
            schema=schema,
        )
        insert_running_run(schema, run_id=run_id, dataset_version_id=dataset_version_id)

        run_sql(
            f"INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload) VALUES ({dataset_version_id}, 2, '{{}}');",
            schema=schema,
            expect_ok=False,
        )

        ids = run_sql(
            f"SELECT record_id FROM source_records WHERE dataset_version_id = {dataset_version_id} ORDER BY source_row_number;",
            schema=schema,
        ).splitlines()
        train_record_id, test_record_id = [int(value) for value in ids]
        run_sql(
            f"""
            INSERT INTO ml_run_record_splits (
                run_id, dataset_version_id, record_id, split_name
            )
            VALUES
                ('{run_id}', {dataset_version_id}, {train_record_id}, 'train'),
                ('{run_id}', {dataset_version_id}, {test_record_id}, 'test');
            """,
            schema=schema,
        )

        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {test_record_id}, 'train', 1, 1, 0.9, '{{"Low":0.05,"Medium":0.9,"High":0.05}}');
            """,
            schema=schema,
            expect_ok=False,
        )

        run_sql(
            f"UPDATE ml_experiment_runs SET status = 'completed', completed_at = NOW() WHERE run_id = '{run_id}';",
            schema=schema,
            expect_ok=False,
        )

        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {test_record_id}, 'test', 1, 1, 0.9, '{{"Low":0.05,"Medium":0.9,"High":0.05}}');
            """,
            schema=schema,
        )
        run_sql(
            f"UPDATE ml_experiment_runs SET status = 'completed', completed_at = NOW() WHERE run_id = '{run_id}';",
            schema=schema,
        )

        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {train_record_id}, 'train', 0, 0, 0.8, '{{"Low":0.8,"Medium":0.1,"High":0.1}}');
            """,
            schema=schema,
            expect_ok=False,
        )
    finally:
        drop_schema(schema)


def test_postgres_rejects_incomplete_source_row_range_and_missing_membership():
    schema = create_schema()
    try:
        bad_dataset_id = insert_dataset_version(schema, row_count=2, suffix="bad_range")
        run_sql(
            f"""
            INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
            VALUES ({bad_dataset_id}, 1, '{{}}'), ({bad_dataset_id}, 2, '{{}}');
            """,
            schema=schema,
        )
        run_id = str(uuid.uuid4())
        run_sql(
            f"""
            INSERT INTO ml_experiment_runs (
                run_id, dataset_version_id, model_name, task_type,
                target_definition, target_definition_hash,
                split_manifest_uri, split_manifest_hash,
                git_commit, working_tree_state,
                environment_lock_uri, environment_lock_hash,
                train_config, artifact_uri,
                status, started_at
            )
            VALUES (
                '{run_id}', {bad_dataset_id}, 'model', 'classification',
                '{{}}', 'target', 'split.json', 'split',
                'commit', 'clean',
                'env.yml', 'envhash',
                '{{}}', 'artifact',
                'running', NOW()
            );
            """,
            schema=schema,
            expect_ok=False,
        )

        dataset_id = insert_dataset_version(schema, row_count=2, suffix="missing_membership")
        run_sql(
            f"""
            INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
            VALUES ({dataset_id}, 0, '{{}}'), ({dataset_id}, 1, '{{}}');
            """,
            schema=schema,
        )
        run_id = str(uuid.uuid4())
        insert_running_run(schema, run_id=run_id, dataset_version_id=dataset_id)
        first_record_id, second_record_id = [
            int(value)
            for value in run_sql(
                f"SELECT record_id FROM source_records WHERE dataset_version_id = {dataset_id} ORDER BY source_row_number;",
                schema=schema,
            ).splitlines()
        ]
        run_sql(
            f"""
            INSERT INTO ml_run_record_splits (run_id, dataset_version_id, record_id, split_name)
            VALUES ('{run_id}', {dataset_id}, {first_record_id}, 'test');
            """,
            schema=schema,
        )
        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {first_record_id}, 'test', 1, 1, 0.9, '{{"Low":0.05,"Medium":0.9,"High":0.05}}');
            """,
            schema=schema,
        )
        run_sql(
            f"UPDATE ml_experiment_runs SET status = 'completed', completed_at = NOW() WHERE run_id = '{run_id}';",
            schema=schema,
            expect_ok=False,
        )

        run_sql(
            f"""
            INSERT INTO ml_run_record_splits (run_id, dataset_version_id, record_id, split_name)
            VALUES ('{run_id}', {dataset_id}, {second_record_id}, 'train');
            """,
            schema=schema,
        )
        run_sql(
            f"UPDATE ml_experiment_runs SET status = 'completed', completed_at = NOW() WHERE run_id = '{run_id}';",
            schema=schema,
        )
    finally:
        drop_schema(schema)


def test_app_role_is_insert_only_for_source_split_prediction_ledgers():
    schema = create_schema()
    run_id = str(uuid.uuid4())
    app_dsn = app_role_dsn()
    try:
        dataset_version_id = int(
            run_sql(
                """
                INSERT INTO source_dataset_versions (
                    dataset_code, source_locator, content_hash,
                    ingestion_contract, ingestion_contract_hash,
                    row_count, metadata
                )
                VALUES ('student-mat', 'project://data/raw/student-mat.csv', 'content_app', '{}', 'contract_app', 2, '{}')
                RETURNING dataset_version_id;
                """,
                schema=schema,
                dsn=app_dsn,
            ).splitlines()[-1]
        )
        run_sql(
            f"""
            INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
            VALUES ({dataset_version_id}, 0, '{{"G3": 8}}'), ({dataset_version_id}, 1, '{{"G3": 12}}');
            """,
            schema=schema,
            dsn=app_dsn,
        )
        insert_running_run(schema, run_id=run_id, dataset_version_id=dataset_version_id, dsn=app_dsn)
        record_ids = [
            int(value)
            for value in run_sql(
                f"SELECT record_id FROM source_records WHERE dataset_version_id = {dataset_version_id} ORDER BY source_row_number;",
                schema=schema,
                dsn=app_dsn,
            ).splitlines()
        ]
        run_sql(
            f"""
            INSERT INTO ml_run_record_splits (run_id, dataset_version_id, record_id, split_name)
            VALUES
                ('{run_id}', {dataset_version_id}, {record_ids[0]}, 'train'),
                ('{run_id}', {dataset_version_id}, {record_ids[1]}, 'test');
            """,
            schema=schema,
            dsn=app_dsn,
        )
        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {record_ids[1]}, 'test', 1, 1, 0.9, '{{"Low":0.05,"Medium":0.9,"High":0.05}}');
            """,
            schema=schema,
            dsn=app_dsn,
        )

        run_sql(
            f"UPDATE source_records SET raw_payload = '{{}}' WHERE record_id = {record_ids[0]};",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"DELETE FROM source_records WHERE record_id = {record_ids[0]};",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"UPDATE ml_run_record_splits SET split_name = 'test' WHERE run_id = '{run_id}' AND record_id = {record_ids[0]};",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"DELETE FROM ml_run_record_splits WHERE run_id = '{run_id}' AND record_id = {record_ids[0]};",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"UPDATE ml_predictions SET confidence = 0.1 WHERE run_id = '{run_id}' AND record_id = {record_ids[1]};",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"DELETE FROM ml_predictions WHERE run_id = '{run_id}' AND record_id = {record_ids[1]};",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
    finally:
        drop_schema(schema)


def test_completed_run_allows_append_only_recommendation_policy_versions_only():
    schema = create_schema()
    app_dsn = app_role_dsn()
    try:
        run_id, dataset_version_id, record_ids, prediction_id = setup_completed_run(schema, dsn=app_dsn)

        run_sql(
            f"""
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (
                {prediction_id}, 'policy_v1', 'Medium',
                '{{"weekly_plan":[]}}',
                '{{"policy_version":"policy_v1"}}'
            );
            """,
            schema=schema,
            dsn=app_dsn,
        )
        run_sql(
            f"""
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (
                {prediction_id}, 'policy_v2', 'Medium',
                '{{"weekly_plan":["review"]}}',
                '{{"policy_version":"policy_v2"}}'
            );
            """,
            schema=schema,
            dsn=app_dsn,
        )
        assert (
            run_sql(
                f"SELECT COUNT(*) FROM ml_recommendations WHERE prediction_id = {prediction_id};",
                schema=schema,
                dsn=app_dsn,
            ).splitlines()[-1]
            == "2"
        )

        run_sql(
            f"""
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (
                {prediction_id}, 'policy_v2', 'High',
                '{{"weekly_plan":["changed"]}}',
                '{{"policy_version":"policy_v2"}}'
            );
            """,
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"UPDATE ml_recommendations SET risk_band = 'High' WHERE prediction_id = {prediction_id} AND policy_version = 'policy_v1';",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"DELETE FROM ml_recommendations WHERE prediction_id = {prediction_id} AND policy_version = 'policy_v1';",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            """
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (
                999999999, 'orphan_policy', 'Medium',
                '{"weekly_plan":[]}',
                '{"policy_version":"orphan_policy"}'
            );
            """,
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )

        run_sql(
            f"INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload) VALUES ({dataset_version_id}, 2, '{{}}');",
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"""
            INSERT INTO ml_run_record_splits (run_id, dataset_version_id, record_id, split_name)
            VALUES ('{run_id}', {dataset_version_id}, {record_ids[0]}, 'validation');
            """,
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"""
            INSERT INTO ml_predictions (
                run_id, record_id, split_name, true_label,
                predicted_label, confidence, probability
            )
            VALUES ('{run_id}', {record_ids[0]}, 'train', 0, 0, 0.8, '{{"Low":0.8,"Medium":0.1,"High":0.1}}');
            """,
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
        run_sql(
            f"""
            INSERT INTO ml_run_metrics (run_id, split_name, metric_name, metric_value, metric_context)
            VALUES ('{run_id}', 'test', 'Accuracy', 1.0, '{{}}');
            """,
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
    finally:
        drop_schema(schema)


def test_failed_run_cannot_materialize_recommendation_policy_versions():
    schema = create_schema()
    app_dsn = app_role_dsn()
    try:
        run_id = str(uuid.uuid4())
        dataset_version_id = insert_dataset_version(schema, suffix="failed_policy")
        run_sql(
            f"""
            INSERT INTO source_records (dataset_version_id, source_row_number, raw_payload)
            VALUES
                ({dataset_version_id}, 0, '{{"G1": 8}}'),
                ({dataset_version_id}, 1, '{{"G1": 12}}');
            """,
            schema=schema,
            dsn=app_dsn,
        )
        insert_running_run(schema, run_id=run_id, dataset_version_id=dataset_version_id, dsn=app_dsn)
        record_ids = [
            int(value)
            for value in run_sql(
                f"SELECT record_id FROM source_records WHERE dataset_version_id = {dataset_version_id} ORDER BY source_row_number;",
                schema=schema,
                dsn=app_dsn,
            ).splitlines()
        ]
        run_sql(
            f"""
            INSERT INTO ml_run_record_splits (run_id, dataset_version_id, record_id, split_name)
            VALUES
                ('{run_id}', {dataset_version_id}, {record_ids[0]}, 'train'),
                ('{run_id}', {dataset_version_id}, {record_ids[1]}, 'test');
            """,
            schema=schema,
            dsn=app_dsn,
        )
        prediction_id = int(
            run_sql(
                f"""
                INSERT INTO ml_predictions (
                    run_id, record_id, split_name, true_label,
                    predicted_label, confidence, probability
                )
                VALUES ('{run_id}', {record_ids[1]}, 'test', 1, 1, 0.9, '{{"Low":0.05,"Medium":0.9,"High":0.05}}')
                RETURNING prediction_id;
                """,
                schema=schema,
                dsn=app_dsn,
            ).splitlines()[-1]
        )
        run_sql(
            f"UPDATE ml_experiment_runs SET status = 'failed', completed_at = NOW() WHERE run_id = '{run_id}';",
            schema=schema,
            dsn=app_dsn,
        )
        run_sql(
            f"""
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (
                {prediction_id}, 'policy_after_failed', 'Medium',
                '{{"weekly_plan":[]}}',
                '{{"policy_version":"policy_after_failed"}}'
            );
            """,
            schema=schema,
            dsn=app_dsn,
            expect_ok=False,
        )
    finally:
        drop_schema(schema)
