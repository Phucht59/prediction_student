from pathlib import Path

from scripts import database_final

ROOT = Path(__file__).resolve().parents[2]


def test_stage_migration_uses_run_record_stage_prediction_key() -> None:
    sql = (
        ROOT / "database/final/migrations/012_add_stage_aware_prediction.sql"
    ).read_text(encoding="utf-8")
    assert "UNIQUE (run_id, record_pk, prediction_stage)" in sql
    assert "prediction_stage TEXT NOT NULL" in sql


def test_stage_metric_key_uses_nulls_not_distinct() -> None:
    sql = (
        ROOT / "database/final/migrations/012_add_stage_aware_prediction.sql"
    ).read_text(encoding="utf-8")
    assert "prediction_stage" in database_final.METRIC_NATURAL_KEY_COLUMNS
    assert "NULLS NOT DISTINCT" in sql


def test_stage_and_overall_views_are_explicit() -> None:
    sql = (
        ROOT / "database/final/migrations/012_add_stage_aware_prediction.sql"
    ).read_text(encoding="utf-8")
    assert "ml.stage_model_results" in sql
    assert "mt.scope = 'stage'" in sql


def test_recommendation_provenance_defaults_to_frozen_oulad_stage() -> None:
    sql = (
        ROOT / "database/final/migrations/012_add_stage_aware_prediction.sql"
    ).read_text(encoding="utf-8")
    assert "DEFAULT 'F2_MIDDLE'" in sql


def test_database_replacement_never_targets_canonical_for_writes() -> None:
    source = (ROOT / "scripts/unified_database.py").read_text(encoding="utf-8")
    assert "source_opened_read_only" in source
    assert "canonical_database_modified" in source
    assert "CREATE DATABASE" in source
