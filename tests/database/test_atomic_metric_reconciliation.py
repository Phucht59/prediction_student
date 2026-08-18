from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import database_final as db


ROOT = Path(__file__).resolve().parents[2]


def _rows():
    return db._canonical_metric_rows()


def _counts(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
              (SELECT count(*) FROM ml.model),
              (SELECT count(*) FROM ml.run),
              (SELECT count(*) FROM ml.metric),
              (SELECT count(*) FROM recommendation.risk_profile),
              (SELECT count(*) FROM recommendation.plan),
              (SELECT count(*) FROM recommendation.action),
              (SELECT count(*) FROM recommendation.review)
            """
        )
        return tuple(int(value) for value in cursor.fetchone())


def test_metric_natural_key_normalizes_nullable_fields():
    sequence = (
        "run",
        "macro_f1",
        0.5,
        "overall",
        "ensemble",
        None,
        None,
        None,
        None,
        None,
        {},
    )
    mapping = {
        "run_id": "run",
        "metric_name": "macro_f1",
        "scope": "overall",
        "aggregation": "ensemble",
        "class_label": None,
        "budget": None,
        "fold": None,
        "seed": None,
    }
    assert db.normalize_metric_natural_key(sequence) == (
        "run",
        "macro_f1",
        "overall",
        "ensemble",
        "<NULL>",
        "<NULL>",
        "<NULL>",
        "<NULL>",
        "<NULL>",
    )
    assert db.normalize_metric_natural_key(sequence) == (
        db.normalize_metric_natural_key(mapping)
    )


def test_canonical_metric_rows_are_unique_and_complete():
    rows = _rows()
    keys = [db.normalize_metric_natural_key(row) for row in rows]
    assert len(rows) == 995
    assert len(keys) == len(set(keys))


def test_stale_metric_is_updated_and_replay_is_idempotent(
    final_connection,
):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE ml.metric
            SET metric_value=-123.0, unit='stale', detail='{"stale": true}'
            WHERE run_id='final:student_mat:logistic_regression'
              AND metric_name='macro_f1'
              AND scope='overall'
              AND aggregation='ensemble'
              AND class_label IS NULL
              AND budget IS NULL
              AND fold IS NULL
              AND seed IS NULL
            """
        )
        first = db._reconcile_canonical_metrics(cursor, _rows())
        assert first["updated_rows"] >= 1
        cursor.execute(
            """
            SELECT metric_value,unit,detail
            FROM ml.metric
            WHERE run_id='final:student_mat:logistic_regression'
              AND metric_name='macro_f1'
              AND scope='overall'
              AND aggregation='ensemble'
              AND class_label IS NULL
              AND budget IS NULL
              AND fold IS NULL
              AND seed IS NULL
            """
        )
        value, unit, detail = cursor.fetchone()
        assert value == pytest.approx(0.8951984435013155, abs=1e-15)
        assert unit is None
        assert detail["value"] == pytest.approx(
            0.8951984435013155, abs=1e-15
        )
        second = db._reconcile_canonical_metrics(cursor, _rows())
        assert second["inserted_rows"] == 0
        assert second["updated_rows"] == 0
        assert second["deleted_stale_rows"] == 0


def test_missing_metric_is_inserted(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            DELETE FROM ml.metric
            WHERE run_id='final:student_por:mlp'
              AND metric_name='macro_f1'
              AND scope='overall'
              AND aggregation='ensemble'
              AND class_label IS NULL
              AND budget IS NULL
              AND fold IS NULL
              AND seed IS NULL
            """
        )
        result = db._reconcile_canonical_metrics(cursor, _rows())
        assert result["inserted_rows"] >= 1
        assert result["missing_rows"] == 0


def test_extra_canonical_scope_metric_is_deleted(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO ml.metric(
              run_id,metric_name,metric_value,scope,aggregation,detail
            ) VALUES (
              'final:student_mat:mlp','obsolete_metric',1.0,
              'overall','ensemble','{}'
            )
            """
        )
        result = db._reconcile_canonical_metrics(cursor, _rows())
        assert result["deleted_stale_rows"] >= 1
        assert result["extra_rows"] == 0


def test_reconciliation_preserves_recommendation_tables(
    final_connection,
):
    before = _counts(final_connection)
    with final_connection.cursor() as cursor:
        result = db._reconcile_canonical_metrics(cursor, _rows())
    after = _counts(final_connection)
    assert result["all_metrics_match_canonical_json"] is True
    assert after[3:] == before[3:]
    assert after[3:] == (15378, 15378, 27355, 0)


def test_safe_revalidated_and_mlp_values_win(final_connection):
    expected = {
        ("student_mat", "logistic_regression"): 0.8951984435013155,
        ("student_mat", "decision_tree"): 0.9024249717831561,
        ("student_mat", "random_forest"): 0.8998331533550239,
        ("student_mat", "hist_gradient_boosting"): 0.8696861886517059,
        ("student_mat", "svm"): 0.8710297543355626,
        ("student_mat", "xgboost"): 0.8815265387865198,
        ("student_por", "logistic_regression"): 0.8378767571281749,
        ("student_por", "decision_tree"): 0.8460876710013042,
        ("student_por", "random_forest"): 0.8513805162482108,
        ("student_por", "hist_gradient_boosting"): 0.8440556722833327,
        ("student_por", "svm"): 0.8501911605004389,
        ("student_por", "xgboost"): 0.8676585159508289,
        ("student_mat", "mlp"): 0.8595069898734821,
        ("student_por", "mlp"): 0.8303986867455508,
        ("oulad", "mlp"): 0.8282857900281345,
        ("oulad", "cnn_bilstm"): 0.8280835945631038,
    }
    with final_connection.cursor() as cursor:
        db._reconcile_canonical_metrics(cursor, _rows())
        for (dataset, model), value in expected.items():
            cursor.execute(
                """
                SELECT metric_value
                FROM ml.metric
                WHERE run_id=%s
                  AND metric_name='macro_f1'
                  AND scope='overall'
                  AND aggregation='ensemble'
                  AND class_label IS NULL
                  AND budget IS NULL
                  AND fold IS NULL
                  AND seed IS NULL
                """,
                (f"final:{dataset}:{model}",),
            )
            assert cursor.fetchone()[0] == pytest.approx(
                value, abs=1e-15
            )


class _FakeConnection:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _patch_cutover_dependencies(monkeypatch, connection):
    monkeypatch.setattr(db, "_connect", lambda *_args, **_kwargs: connection)
    monkeypatch.setattr(
        db, "_validate_backup_manifest", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(db, "_read_json", lambda *_args, **_kwargs: {
        "status": "PASS"
    })
    monkeypatch.setattr(db, "_assert_locked_sources", lambda: None)
    monkeypatch.setattr(
        db,
        "_apply_migrations_connection",
        lambda _connection: {"applied": [], "skipped": []},
    )
    monkeypatch.setattr(
        db,
        "load_canonical",
        lambda *_args, **_kwargs: {
            "metric_reconciliation": {
                "all_metrics_match_canonical_json": True
            }
        },
    )
    monkeypatch.setattr(
        db, "_legacy_cutover_in_transaction", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        db,
        "_validate_atomic_cutover",
        lambda *_args, **_kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        db, "validate_database", lambda *_args, **_kwargs: {"status": "PASS"}
    )
    monkeypatch.setattr(
        db, "_permission_validation", lambda *_args, **_kwargs: {
            "status": "PASS"
        }
    )
    monkeypatch.setattr(db, "_is_disposable", lambda *_args: True)


def test_atomic_cutover_rolls_back_on_validation_hook_failure(
    monkeypatch,
):
    connection = _FakeConnection()
    _patch_cutover_dependencies(monkeypatch, connection)

    def fail(_connection):
        raise db.FinalDatabaseError("controlled validation failure")

    with pytest.raises(
        db.FinalDatabaseError, match="controlled validation failure"
    ):
        db.cutover(
            "postgresql://redacted/disposable_test",
            confirm=True,
            drop_empty=False,
            _validation_hook=fail,
        )
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_atomic_cutover_commits_only_after_validation(monkeypatch):
    connection = _FakeConnection()
    _patch_cutover_dependencies(monkeypatch, connection)
    result = db.cutover(
        "postgresql://redacted/disposable_test",
        confirm=True,
        drop_empty=False,
    )
    assert result["status"] == "PASS"
    assert connection.commits == 1
    assert connection.rollbacks == 0


def test_identity_sequence_replay_is_deterministic(final_connection):
    dsn = os.getenv("FINAL_DATABASE_URL")
    if not dsn:
        pytest.skip("FINAL_DATABASE_URL is required")
    with final_connection.cursor() as cursor:
        db._synchronize_identity_sequences(cursor)
    final_connection.commit()
    middle = db._sequence_state(dsn)
    with final_connection.cursor() as cursor:
        db._synchronize_identity_sequences(cursor)
    final_connection.commit()
    after = db._sequence_state(dsn)
    assert middle == after


def test_database_checksum_manifest_replays():
    dsn = os.getenv("FINAL_DATABASE_URL")
    if not dsn:
        pytest.skip("FINAL_DATABASE_URL is required")
    assert db._validate_database_checksum_manifest(dsn)["status"] == "PASS"
