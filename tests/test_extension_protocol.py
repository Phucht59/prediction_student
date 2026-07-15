from __future__ import annotations

import json
from pathlib import Path

from src.studies.common.hashing import sha256_file, stable_record_id


ROOT = Path(__file__).resolve().parents[1]


def test_extension_protocol_is_frozen_and_preserves_study_a():
    protocol = json.loads((ROOT / "configs" / "extension_protocol_v1.yaml").read_text(encoding="utf-8"))
    assert protocol["status"] == "frozen_before_model_results"
    assert protocol["study_a"] == {
        "immutable": True,
        "development_records": 316,
        "legacy_heldout_observed_records": 79,
        "legacy_observed_access_allowed": False,
        "official_artifact_mutation_allowed": False,
    }
    assert protocol["study_c"]["forecasts"] == [
        {"id": "F1_EARLY", "fraction": 0.20},
        {"id": "F2_MIDDLE", "fraction": 0.50},
        {"id": "F3_LATE", "fraction": 0.80},
    ]


def test_raw_manifest_matches_frozen_source_hashes():
    protocol = json.loads((ROOT / "configs" / "extension_protocol_v1.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "manifests" / "extension_raw_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS"
    assert len(manifest["files"]) == 9
    for item in manifest["files"]:
        source = protocol["sources"][item["logical_dataset"]]
        assert item["sha256"] == source["sha256"]
        assert sha256_file(ROOT / source["path"]) == source["sha256"]
        assert item["duplicate_column_status"] == "PASS"


def test_stable_record_id_is_deterministic_and_order_sensitive():
    assert stable_record_id("AAA", "2014J", 1) == stable_record_id("AAA", "2014J", 1)
    assert stable_record_id("AAA", "2014J", 1) != stable_record_id("AAA", 1, "2014J")


def test_migration_005_is_additive_and_append_only():
    text = (ROOT / "database" / "migrations" / "005_oulad_lineage_and_snapshot_registry.sql").read_text(encoding="utf-8")
    for table in ["source_dataset_files", "prediction_cohorts", "cutoff_feature_snapshots", "snapshot_record_index", "split_manifest_registry"]:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in text
    assert "reject_append_only_update_delete" in text
    assert "DROP TABLE" not in text.upper()
