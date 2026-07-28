import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_all_metrics_match_canonical_json():
    with (ROOT / "artifacts/final/database/metric_reconciliation.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(row["status"] == "PASS" for row in rows)


def test_all_per_class_metrics_loaded(final_connection):
    final = json.loads((ROOT / "artifacts/final/final_results.json").read_text(encoding="utf-8"))
    expected = sum(
        len(model["per_class"]) * 4
        for dataset in final["datasets"].values()
        for model in dataset["models"]
    )
    with final_connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM ml.metric WHERE scope='per_class'")
        assert cursor.fetchone()[0] == expected


def test_all_oulad_top_k_loaded(final_connection):
    with final_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT count(*) FROM ml.metric mt
            JOIN ml.run r USING(run_id)
            JOIN ml.model m USING(model_id)
            JOIN catalog.dataset d USING(dataset_id)
            WHERE d.slug='oulad' AND mt.scope='top_k'
            """
        )
        assert cursor.fetchone()[0] == 10 * 3 * 4


def test_artifact_checksums_match():
    validation = json.loads(
        (ROOT / "artifacts/final/database/migration_validation.json").read_text(encoding="utf-8")
    )
    assert validation["checks"]["artifact_checksums_match"] is True
