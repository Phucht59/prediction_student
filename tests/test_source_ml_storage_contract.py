import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_pipeline
from src import data_pipeline
from src import postgres_data_source
from src.data_pipeline import (
    DataPreprocessor,
    SOURCE_ROW_NUMBER_COLUMN,
    StudentDataset,
    attach_source_row_numbers,
    create_and_save_locked_test,
    split_sidecar_matches_current_raw,
)
from src.evaluation.evaluation import (
    build_ingestion_contract,
    build_split_manifest,
    build_target_definition,
    prepare_storage_context,
    sha256_file,
    sha256_json,
    validate_run_artifact_provenance,
    validate_split_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = PROJECT_ROOT / "database" / "migrations" / "001_create_source_ml_schema.sql"


class _FakeCursor:
    def __init__(self, version=None, rows=None):
        self.version = version
        self.rows = rows or []
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        if "FROM source_dataset_versions" in query:
            self.result = self.version
        elif "FROM source_records" in query:
            self.result = self.rows
        else:
            raise AssertionError(f"unexpected query: {query}")

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result


class _FakeConnection:
    def __init__(self, version=None, rows=None):
        self.cursor_obj = _FakeCursor(version=version, rows=rows)

    def cursor(self, *args, **kwargs):
        return self.cursor_obj

    def close(self):
        pass


def test_source_ml_migration_contains_hard_gate_constraints():
    sql = MIGRATION_PATH.read_text(encoding="utf-8")
    sql_lower = sql.lower()
    required_fragments = [
        "primary key (run_id, record_id)",
        "unique (run_id, record_id, split_name)",
        "foreign key (run_id, record_id, split_name)",
        "references ml_run_record_splits(run_id, record_id, split_name)",
        "status <> 'running'",
        "experiment runs must be inserted as running",
        "reject_source_record_insert_after_run",
        "count(distinct source_row_number)",
        "test_prediction_count <> test_count",
        "split_name IN ('train', 'validation', 'test', 'excluded')",
        "source_diff_uri IS NULL AND source_diff_hash IS NULL",
        "source_diff_uri IS NOT NULL AND source_diff_hash IS NOT NULL",
        "environment_lock_uri",
        "ingestion_contract_hash_algorithm",
        "student_predict_app",
        "grant select, insert",
        "grant update(status, completed_at)",
    ]
    for fragment in required_fragments:
        assert fragment.lower() in sql_lower


def test_legacy_paper_schema_is_not_modified_by_new_migration():
    legacy_schema = (PROJECT_ROOT / "database" / "schema.sql").read_text(encoding="utf-8")
    migration_sql = MIGRATION_PATH.read_text(encoding="utf-8").lower()

    assert "paper_predictions" in legacy_schema
    assert "paper_learning_recommendations" in legacy_schema
    assert "alter table paper_" not in migration_sql
    assert "drop table paper_" not in migration_sql


def test_source_row_number_is_protected_from_preprocessing_and_model_input():
    frame = attach_source_row_numbers(
        pd.DataFrame(
            [
                {"G1": 8, "G2": 9, "G3": 0, "studytime": 1, "school": "GP"},
                {"G1": 12, "G2": 13, "G3": 1, "studytime": 2, "school": "MS"},
                {"G1": 15, "G2": 16, "G3": 2, "studytime": 3, "school": "GP"},
            ]
        )
    )

    preprocessor = DataPreprocessor(target_col="G3", oversample_method="none")
    prepared = preprocessor.fit_transform(frame, apply_oversampling=False)

    assert SOURCE_ROW_NUMBER_COLUMN in frame.columns
    assert SOURCE_ROW_NUMBER_COLUMN not in prepared.columns
    assert SOURCE_ROW_NUMBER_COLUMN not in preprocessor.numerical_cols
    assert SOURCE_ROW_NUMBER_COLUMN not in preprocessor.categorical_cols

    dataset = StudentDataset(
        prepared,
        kind="student",
        target_col="G3",
        numerical_cols=preprocessor.numerical_cols,
        categorical_cols=preprocessor.categorical_cols,
    )
    assert SOURCE_ROW_NUMBER_COLUMN not in dataset.num_cols
    assert SOURCE_ROW_NUMBER_COLUMN not in dataset.cat_cols


def test_split_manifest_partitions_all_source_records_without_synthetic_membership():
    raw = attach_source_row_numbers(
        pd.DataFrame(
            [
                {"G1": 8, "G2": 9, "G3": 8},
                {"G1": 10, "G2": 11, "G3": 11},
                {"G1": 13, "G2": 14, "G3": 14},
                {"G1": 15, "G2": 16, "G3": 16},
            ]
        )
    )
    train = raw.iloc[[0, 1]].copy()
    test = raw.iloc[[2]].copy()
    target_definition = build_target_definition("student-mat", "G3", "3class", "student")

    manifest = build_split_manifest(
        dataset_version_identity={
            "dataset_code": "student-mat",
            "hash_algorithm": "sha256",
            "content_hash": "abc",
            "ingestion_contract_hash_algorithm": "sha256",
            "ingestion_contract_hash": "def",
        },
        target_definition=target_definition,
        split_protocol={"name": "unit", "random_seed": 42},
        raw_frame=raw,
        train_pool=train,
        locked_test=test,
    )
    validate_split_manifest(manifest)

    membership = manifest["membership"]
    all_rows = {
        item["source_row_number"]
        for split_name in ("train", "validation", "test")
        for item in membership[split_name]
    } | {item["source_row_number"] for item in membership["excluded"]}

    assert all_rows == {0, 1, 2, 3}
    assert manifest["counts"]["train"] == 2
    assert manifest["counts"]["validation"] == 0
    assert manifest["counts"]["test"] == 1
    assert manifest["counts"]["excluded"] == 1
    assert "synthetic" not in json.dumps(manifest).lower()


def test_ingestion_contract_is_parser_semantics_only():
    contract = build_ingestion_contract(";", ["G1", "G2", "G3"])

    assert "source_file_name" not in contract
    assert "source_locator" not in contract
    assert contract["source_format"] == "csv"
    assert contract["delimiter"] == ";"
    assert contract["canonical_columns"] == ["G1", "G2", "G3"]
    assert contract["schema_fingerprint"] == sha256_json(["G1", "G2", "G3"])


def test_ingestion_contract_hash_participates_in_dataset_identity():
    base_contract = {
        "source_format": "csv",
        "delimiter": ";",
        "encoding": "utf-8",
        "header_policy": "first_row_header",
        "null_value_policy": "pandas_default",
        "parser": "pandas.read_csv",
        "parser_version": "unit",
        "canonical_columns": ["G1", "G2", "G3"],
    }
    changed_contract = dict(base_contract)
    changed_contract["delimiter"] = ","

    assert sha256_json(base_contract) != sha256_json(changed_contract)


def test_prepare_storage_context_uses_single_source_row_mapping_with_non_default_index(tmp_path):
    raw_path = tmp_path / "renamed-source.csv"
    raw_frame = pd.DataFrame(
        [
            {"G1": 8, "G2": 9, "G3": 8},
            {"G1": 10, "G2": 11, "G3": 11},
            {"G1": 13, "G2": 14, "G3": 14},
        ],
        index=[10, 20, 30],
    )
    raw_frame.to_csv(raw_path, sep=";", index=False)
    raw_with_source = attach_source_row_numbers(raw_frame)
    train_pool = raw_with_source.iloc[[2, 0]].copy()
    locked_test = raw_with_source.iloc[[1]].copy()

    environment_lock = tmp_path / "environment.yml"
    environment_lock.write_text("dependencies: []\n", encoding="utf-8")
    artifact_manifest = tmp_path / "artifact.json"
    artifact_manifest.write_text("{}", encoding="utf-8")

    context = prepare_storage_context(
        dataset_name="student-mat",
        target_mode="3class",
        dataset_kind="student",
        target_col="G3",
        raw_path=raw_path,
        csv_sep=";",
        raw_frame=raw_with_source,
        train_pool=train_pool,
        locked_test=locked_test,
        run_id="00000000-0000-0000-0000-000000000001",
        model_name="unit",
        train_config={},
        artifact_uri=str(artifact_manifest),
        git_commit="commit",
        working_tree_state="clean",
        source_diff_uri=None,
        source_diff_hash=None,
        environment_lock_uri=str(environment_lock),
        environment_lock_hash=sha256_file(environment_lock),
        split_manifest_path=tmp_path / "split.json",
    )

    records = context["source_records"]
    assert [record["source_row_number"] for record in records] == [0, 1, 2]
    assert [record["raw_payload"]["G1"] for record in records] == [8, 10, 13]
    assert {item["source_row_number"] for item in context["split_manifest"]["membership"]["train"]} == {0, 2}
    assert {item["source_row_number"] for item in context["split_manifest"]["membership"]["test"]} == {1}
    assert not Path(context["dataset_version"]["source_locator"]).is_absolute()


def test_dataset_version_identity_survives_raw_file_rename(tmp_path):
    first_path = tmp_path / "student-mat.csv"
    second_path = tmp_path / "renamed-student-mat.csv"
    csv_text = "G1;G2;G3\n8;9;8\n10;11;11\n13;14;14\n"
    first_path.write_text(csv_text, encoding="utf-8")
    second_path.write_text(csv_text, encoding="utf-8")
    raw_frame = attach_source_row_numbers(pd.read_csv(first_path, sep=";"))
    train_pool = raw_frame.iloc[[0, 2]].copy()
    locked_test = raw_frame.iloc[[1]].copy()
    environment_lock = tmp_path / "environment.yml"
    environment_lock.write_text("dependencies: []\n", encoding="utf-8")
    artifact_manifest = tmp_path / "artifact.json"
    artifact_manifest.write_text("{}", encoding="utf-8")

    common_kwargs = {
        "dataset_name": "student-mat",
        "target_mode": "3class",
        "dataset_kind": "student",
        "target_col": "G3",
        "csv_sep": ";",
        "raw_frame": raw_frame,
        "train_pool": train_pool,
        "locked_test": locked_test,
        "run_id": "00000000-0000-0000-0000-000000000002",
        "model_name": "unit",
        "train_config": {},
        "artifact_uri": str(artifact_manifest),
        "git_commit": "commit",
        "working_tree_state": "clean",
        "source_diff_uri": None,
        "source_diff_hash": None,
        "environment_lock_uri": str(environment_lock),
        "environment_lock_hash": sha256_file(environment_lock),
    }
    first_context = prepare_storage_context(
        **common_kwargs,
        raw_path=first_path,
        split_manifest_path=tmp_path / "split-a.json",
    )
    second_context = prepare_storage_context(
        **common_kwargs,
        raw_path=second_path,
        split_manifest_path=tmp_path / "split-b.json",
    )

    first_version = first_context["dataset_version"]
    second_version = second_context["dataset_version"]
    identity_keys = {
        "dataset_code",
        "hash_algorithm",
        "content_hash",
        "ingestion_contract_hash_algorithm",
        "ingestion_contract_hash",
        "row_count",
        "ingestion_contract",
        "metadata",
    }
    assert {key: first_version[key] for key in identity_keys} == {
        key: second_version[key] for key in identity_keys
    }
    assert first_version["metadata"] == {}
    assert second_version["metadata"] == {}
    assert first_version["source_locator"] != second_version["source_locator"]


def test_artifact_provenance_rejects_missing_or_wrong_hash(tmp_path):
    environment_lock = tmp_path / "environment.yml"
    environment_lock.write_text("dependencies: []\n", encoding="utf-8")
    artifact_manifest = tmp_path / "artifact.json"
    artifact_manifest.write_text("{}", encoding="utf-8")
    diff_file = tmp_path / "source.diff"
    diff_file.write_text("diff", encoding="utf-8")

    valid_run = {
        "environment_lock_uri": str(environment_lock),
        "environment_lock_hash": sha256_file(environment_lock),
        "artifact_uri": str(artifact_manifest),
        "working_tree_state": "dirty",
        "source_diff_uri": str(diff_file),
        "source_diff_hash": sha256_file(diff_file),
    }
    validate_run_artifact_provenance(valid_run)

    wrong_env = dict(valid_run)
    wrong_env["environment_lock_hash"] = "0" * 64
    try:
        validate_run_artifact_provenance(wrong_env)
    except ValueError as exc:
        assert "environment_lock_uri SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("Wrong environment lock hash should be rejected.")

    missing_artifact = dict(valid_run)
    missing_artifact["artifact_uri"] = str(tmp_path / "missing.json")
    try:
        validate_run_artifact_provenance(missing_artifact)
    except FileNotFoundError as exc:
        assert "artifact_uri" in str(exc)
    else:
        raise AssertionError("Missing artifact URI should be rejected.")

    unsupported_remote = dict(valid_run)
    unsupported_remote["artifact_uri"] = "s3://bucket/artifact.json"
    try:
        validate_run_artifact_provenance(unsupported_remote)
    except ValueError as exc:
        assert "unsupported remote URI scheme" in str(exc)
    else:
        raise AssertionError("Unsupported remote artifact URI should be rejected.")


def test_stale_split_sidecar_rejects_raw_byte_change_with_same_row_count(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    raw_path = raw_dir / "student-mat.csv"
    rows = ["G1;G2;G3"]
    rows.extend([f"{grade};{grade};8" for grade in range(1, 6)])
    rows.extend([f"{grade};{grade};11" for grade in range(6, 11)])
    rows.extend([f"{grade};{grade};16" for grade in range(11, 16)])
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(data_pipeline, "PROCESSED_DIR", processed_dir)

    raw_frame = attach_source_row_numbers(pd.read_csv(raw_path, sep=";"))
    create_and_save_locked_test(
        raw_frame,
        "student-mat",
        "3class",
        raw_path=raw_path,
        csv_sep=";",
    )
    assert split_sidecar_matches_current_raw("student-mat", "3class", raw_path=raw_path, csv_sep=";")

    changed = raw_path.read_text(encoding="utf-8").replace("1;1;8", "2;1;8", 1)
    raw_path.write_text(changed, encoding="utf-8")
    assert not split_sidecar_matches_current_raw("student-mat", "3class", raw_path=raw_path, csv_sep=";")


def test_postgres_loader_reconstructs_dataframe_without_database_metadata(monkeypatch):
    version = {
        "dataset_version_id": 10,
        "dataset_code": "student-mat",
        "source_locator": "project://data/raw/student-mat.csv",
        "hash_algorithm": "sha256",
        "content_hash": "content",
        "ingestion_contract": {
            "canonical_columns": ["G1", "G2", "G3"],
        },
        "ingestion_contract_hash_algorithm": "sha256",
        "ingestion_contract_hash": "contract",
        "row_count": 2,
        "metadata": {},
        "created_at": "now",
    }
    rows = [
        {"source_row_number": 0, "raw_payload": {"G1": 8, "G2": 9, "G3": 8}},
        {"source_row_number": 1, "raw_payload": {"G1": 10, "G2": 11, "G3": 11}},
    ]
    monkeypatch.setattr(postgres_data_source, "_connect", lambda: _FakeConnection(version, rows))

    frame, metadata = postgres_data_source.load_dataset_version_from_postgres("student-mat")

    assert metadata["dataset_version_id"] == 10
    assert frame.columns.tolist() == [SOURCE_ROW_NUMBER_COLUMN, "G1", "G2", "G3"]
    assert "record_id" not in frame.columns
    assert "dataset_version_id" not in frame.columns
    assert frame[SOURCE_ROW_NUMBER_COLUMN].tolist() == [0, 1]

    preprocessor = DataPreprocessor(target_col="G3", oversample_method="none")
    prepared = preprocessor.fit_transform(frame, apply_oversampling=False)
    assert SOURCE_ROW_NUMBER_COLUMN not in prepared.columns


def test_postgres_loader_missing_dataset_fails_without_csv_fallback(monkeypatch):
    monkeypatch.setattr(postgres_data_source, "_connect", lambda: _FakeConnection(None, []))
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CSV fallback used")))

    with pytest.raises(RuntimeError, match="dataset version not found"):
        postgres_data_source.load_dataset_version_from_postgres("student-mat")


def test_run_pipeline_normal_mode_does_not_fallback_to_csv(monkeypatch):
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id=None,
        dataset_version_id=None,
        seed_from_csv=False,
        skip_postgres=False,
    )
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        run_pipeline,
        "load_dataset_version_from_postgres",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dataset version not found")),
    )
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CSV fallback used")))

    with pytest.raises(RuntimeError, match="dataset version not found"):
        run_pipeline.main()


def test_seed_from_csv_reloads_from_postgres_before_training(monkeypatch):
    calls = []
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id=None,
        dataset_version_id=None,
        seed_from_csv=True,
        skip_postgres=False,
    )
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        run_pipeline,
        "ingest_dataset_csv_to_postgres",
        lambda dataset: calls.append(("seed", dataset)) or {"dataset_version_id": 1, "row_count": 395},
    )

    def stop_after_postgres_load(dataset, dataset_version_id=None):
        calls.append(("load_postgres", dataset, dataset_version_id))
        raise RuntimeError("stop after postgres load")

    monkeypatch.setattr(run_pipeline, "load_dataset_version_from_postgres", stop_after_postgres_load)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("CSV read after seed")))

    with pytest.raises(RuntimeError, match="stop after postgres load"):
        run_pipeline.main()
    assert calls == [("seed", "student-mat"), ("load_postgres", "student-mat", 1)]


def test_seed_from_csv_uses_seeded_dataset_version_not_latest(monkeypatch):
    calls = []
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id=None,
        dataset_version_id=None,
        seed_from_csv=True,
        skip_postgres=False,
    )
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        run_pipeline,
        "ingest_dataset_csv_to_postgres",
        lambda dataset: calls.append(("seed", dataset)) or {"dataset_version_id": 7, "row_count": 395},
    )

    def stop_after_versioned_load(dataset, dataset_version_id=None):
        calls.append(("load_postgres", dataset, dataset_version_id))
        raise RuntimeError("stop after postgres load")

    monkeypatch.setattr(run_pipeline, "load_dataset_version_from_postgres", stop_after_versioned_load)

    with pytest.raises(RuntimeError, match="stop after postgres load"):
        run_pipeline.main()
    assert calls == [("seed", "student-mat"), ("load_postgres", "student-mat", 7)]


def test_seed_from_csv_rejects_conflicting_dataset_version_id(monkeypatch):
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id=None,
        dataset_version_id=8,
        seed_from_csv=True,
        skip_postgres=False,
    )
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        run_pipeline,
        "ingest_dataset_csv_to_postgres",
        lambda dataset: {"dataset_version_id": 7, "row_count": 395},
    )
    monkeypatch.setattr(
        run_pipeline,
        "load_dataset_version_from_postgres",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("conflict should fail before load")),
    )

    with pytest.raises(ValueError, match="dataset_version_id=7"):
        run_pipeline.main()


def test_seed_from_csv_allows_matching_dataset_version_id(monkeypatch):
    calls = []
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id=None,
        dataset_version_id=7,
        seed_from_csv=True,
        skip_postgres=False,
    )
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        run_pipeline,
        "ingest_dataset_csv_to_postgres",
        lambda dataset: {"dataset_version_id": 7, "row_count": 395},
    )

    def stop_after_versioned_load(dataset, dataset_version_id=None):
        calls.append(("load_postgres", dataset, dataset_version_id))
        raise RuntimeError("stop after postgres load")

    monkeypatch.setattr(run_pipeline, "load_dataset_version_from_postgres", stop_after_versioned_load)

    with pytest.raises(RuntimeError, match="stop after postgres load"):
        run_pipeline.main()
    assert calls == [("load_postgres", "student-mat", 7)]


def test_seed_from_csv_rejects_run_id_retry(monkeypatch):
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id="00000000-0000-0000-0000-000000000123",
        dataset_version_id=None,
        seed_from_csv=True,
        skip_postgres=False,
    )
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        run_pipeline,
        "ingest_dataset_csv_to_postgres",
        lambda dataset: (_ for _ in ()).throw(AssertionError("seed should not run for retry")),
    )

    with pytest.raises(ValueError, match="cannot be combined with --run-id"):
        run_pipeline.main()


def test_retry_uses_postgres_split_ledger_without_creating_new_split(monkeypatch):
    args = SimpleNamespace(
        dataset="student-mat",
        target_mode="3class",
        n_trials=1,
        debug=True,
        params_json=None,
        run_id="00000000-0000-0000-0000-000000000123",
        dataset_version_id=None,
        seed_from_csv=False,
        skip_postgres=False,
    )
    raw_frame = attach_source_row_numbers(
        pd.DataFrame(
            [
                {"G1": 8, "G2": 9, "G3": 8},
                {"G1": 10, "G2": 11, "G3": 11},
            ]
        )
    )
    calls = []
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_pipeline, "load_experiment_run", lambda run_id: {"run_id": run_id, "dataset_version_id": 1})
    monkeypatch.setattr(run_pipeline, "load_dataset_version_from_postgres", lambda dataset, version_id: (raw_frame, {"dataset_version_id": version_id}))
    monkeypatch.setattr(run_pipeline, "verify_run_split_manifest", lambda run: calls.append("verify_manifest"))
    monkeypatch.setattr(run_pipeline, "create_locked_split_from_frame", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("created new split")))

    def stop_after_retry_split(*args, **kwargs):
        calls.append("reconstruct_split")
        raise RuntimeError("stop after retry split")

    monkeypatch.setattr(run_pipeline, "reconstruct_existing_run_splits", stop_after_retry_split)

    with pytest.raises(RuntimeError, match="stop after retry split"):
        run_pipeline.main()
    assert calls == ["verify_manifest", "reconstruct_split"]
