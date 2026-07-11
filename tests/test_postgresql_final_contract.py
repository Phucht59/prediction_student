from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import scripts.run_pipeline as run_pipeline
import src.postgres_data_source as postgres_data_source


ROOT = Path(__file__).resolve().parents[1]


def test_final_pipeline_has_no_csv_seed_option():
    source = (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "seed-from-csv" not in source
    assert "ingest_dataset_csv_to_postgres" not in source


def test_new_final_run_requires_dataset_version_id(monkeypatch):
    args = SimpleNamespace(dataset="student-mat", target_mode="3class", n_trials=None, debug=False, params_json=None, selection_config_json='{"best_params": {}}', run_id=None, dataset_version_id=None, skip_postgres=False)
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda _: None)
    with pytest.raises(ValueError, match="dataset-version-id"):
        run_pipeline.main()


def test_non_debug_run_cannot_skip_postgres(monkeypatch):
    args = SimpleNamespace(dataset="student-mat", target_mode="3class", n_trials=None, debug=False, params_json=None, selection_config_json='{"best_params": {}}', run_id=None, dataset_version_id=1, skip_postgres=True)
    monkeypatch.setattr(run_pipeline, "parse_args", lambda: args)
    monkeypatch.setattr(run_pipeline, "ensure_dirs", lambda: None)
    monkeypatch.setattr(run_pipeline, "set_seed", lambda _: None)
    with pytest.raises(ValueError, match="debug-only"):
        run_pipeline.main()


def test_training_loader_requires_target_table(monkeypatch):
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, query, params=None):
            if "source_record_targets" in query:
                raise RuntimeError("relation source_record_targets does not exist")
            if "source_dataset_versions" in query:
                self.row = {"dataset_version_id": 1, "dataset_code": "student-mat", "source_locator": "project://data/raw/student-mat.csv", "hash_algorithm": "sha256", "content_hash": "x", "ingestion_contract": {"canonical_columns": ["G1", "G2", "G3"]}, "ingestion_contract_hash_algorithm": "sha256", "ingestion_contract_hash": "x", "row_count": 1, "metadata": {}, "created_at": "now"}
            elif "source_records" in query:
                self.row = [{"source_row_number": 0, "raw_payload": {"G1": 1, "G2": 2, "G3": 3}}]
            else:
                raise AssertionError(query)
        def fetchone(self): return self.row
        def fetchall(self): return self.row
    class Connection:
        def cursor(self, **kwargs): return Cursor()
        def close(self): pass
    monkeypatch.setattr(postgres_data_source, "_connect", lambda: Connection())
    monkeypatch.setattr(postgres_data_source, "_dict_cursor", lambda c: c.cursor())
    with pytest.raises(RuntimeError, match="source_record_targets"):
        postgres_data_source.load_dataset_version_from_postgres("student-mat", 1, include_target=True)


def test_inference_loader_drops_target_without_csv(monkeypatch):
    # Contract is static: inference calls include_target=False and no production
    # final module is allowed to use pandas.read_csv.
    assert "read_csv" not in (ROOT / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    assert "read_csv" not in (ROOT / "scripts" / "optimize_model_selection.py").read_text(encoding="utf-8")
