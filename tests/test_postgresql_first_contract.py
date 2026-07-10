from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_python_paths_do_not_read_training_csv_directly():
    """CSV parsing is restricted to ingestion/legacy split utilities."""
    forbidden = [
        ROOT / "scripts" / "run_pipeline.py",
        ROOT / "scripts" / "optimize_model_selection.py",
        ROOT / "src" / "model_selection.py",
        ROOT / "src" / "recommendation.py",
        ROOT / "src" / "explainability.py",
        ROOT / "scripts" / "build_final_evidence_bundle.py",
    ]
    for path in forbidden:
        assert "read_csv" not in path.read_text(encoding="utf-8"), path


def test_ingestion_is_the_only_production_dataset_csv_reader():
    source = (ROOT / "src" / "postgres_data_source.py").read_text(encoding="utf-8")
    assert "pd.read_csv" in source
    assert "ingest_dataset_csv_to_postgres" in source


def test_target_storage_migration_is_present():
    migration = (ROOT / "database" / "migrations" / "003_add_source_record_targets.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS source_record_targets" in migration
    assert "FOREIGN KEY (dataset_version_id, record_id)" in migration
