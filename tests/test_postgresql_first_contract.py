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
    reader = (ROOT / "src" / "ingestion" / "csv_reader.py").read_text(encoding="utf-8")
    assert "read_csv" in source
    assert "pd.read_csv" in reader
    assert "ingest_dataset_csv_to_postgres" in source


def test_target_storage_migration_is_present():
    migration = (ROOT / "database" / "migrations" / "003_add_source_record_targets.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS source_record_targets" in migration
    assert "FOREIGN KEY (dataset_version_id, record_id)" in migration


def test_legacy_architectures_are_not_active_source():
    forbidden = [
        "src/models_v27.py",
        "src/losses_v27.py",
        "src/train_v27_pipeline.py",
        "scripts/run_recommender_pipeline.py",
        "src/recommender/risk_head.py",
    ]
    assert all(not (ROOT / path).exists() for path in forbidden)


def test_final_loader_has_no_raw_target_fallback():
    source = (ROOT / "src" / "postgres_data_source.py").read_text(encoding="utf-8")
    assert "final paths do not fall back to raw_payload" in source
    assert "backwards-compatible fallback" not in source
