from pathlib import Path
import pytest

import src.postgres_data_source as postgres_data_source


ROOT = Path(__file__).resolve().parents[1]


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
