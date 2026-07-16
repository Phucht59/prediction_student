from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import Json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATABASE_URL, POSTGRES_CONFIG


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def connect():
    return psycopg2.connect(DATABASE_URL) if DATABASE_URL else psycopg2.connect(**POSTGRES_CONFIG)


def dataset_version(cursor, dataset_code: str, source_locator: str, content_hash: str, row_count: int, contract: dict, metadata: dict) -> int:
    contract_hash = digest_bytes(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode())
    cursor.execute(
        """INSERT INTO source_dataset_versions
        (dataset_code, source_locator, content_hash, ingestion_contract, ingestion_contract_hash, row_count, metadata)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (dataset_code, hash_algorithm, content_hash, ingestion_contract_hash_algorithm, ingestion_contract_hash)
        DO NOTHING""",
        (dataset_code, source_locator, content_hash, Json(contract), contract_hash, row_count, Json(metadata)),
    )
    cursor.execute(
        """SELECT dataset_version_id FROM source_dataset_versions
        WHERE dataset_code=%s AND content_hash=%s AND ingestion_contract_hash=%s""",
        (dataset_code, content_hash, contract_hash),
    )
    return int(cursor.fetchone()[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-b-run", required=True)
    parser.add_argument("--study-c-run", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifests/extension_raw_manifest.json").read_text(encoding="utf-8"))
    files = {entry["logical_dataset"]: entry for entry in manifest["files"]}
    protocol_path = ROOT / "configs/extension_protocol_v1.yaml"
    protocol_hash = digest_file(protocol_path)
    oulad_names = [name for name in files if name.startswith("oulad_")]
    combined_oulad_hash = digest_bytes("\n".join(f"{name}:{files[name]['sha256']}" for name in sorted(oulad_names)).encode())
    migration = ROOT / "database/migrations/005_oulad_lineage_and_snapshot_registry.sql"
    report: dict[str, object] = {"migration": migration.relative_to(ROOT).as_posix(), "migration_sha256": digest_file(migration), "database_registration": "PENDING"}
    try:
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(migration.read_text(encoding="utf-8"))
                por = files["student_por"]
                por_version = dataset_version(cursor, "student-por-extension-v1", por["relative_repository_path"], por["sha256"], por["row_count"], {"delimiter": por["delimiter"], "target": "G3 Low/Medium/High", "features": ["G1", "G2"]}, {"study": "B", "raw_immutable": True})
                oulad_version = dataset_version(cursor, "oulad-release-extension-v1", "data/raw/oulad", combined_oulad_hash, files["oulad_student_info"]["row_count"], {"protocol": "extension_protocol_v1", "landmarks": [0.2, 0.5, 0.8], "event_window": "0<=date<cutoff", "target_separate": True}, {"study": "C", "raw_files": len(oulad_names), "raw_immutable": True})
                for name in sorted(oulad_names):
                    entry = files[name]
                    cursor.execute(
                        """INSERT INTO source_dataset_files
                        (dataset_version_id, logical_name, relative_path, sha256, row_count, schema_json)
                        VALUES (%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (dataset_version_id, logical_name, sha256) DO NOTHING""",
                        (oulad_version, name, entry["relative_repository_path"], entry["sha256"], entry["row_count"], Json({"columns": entry["column_names"], "delimiter": entry["delimiter"], "encoding": entry["encoding"]})),
                    )
                for forecast in ["F1_EARLY", "F2_MIDDLE", "F3_LATE"]:
                    snapshot = json.loads((ROOT / f"data/processed/study_c_oulad/manifests/{forecast}.json").read_text(encoding="utf-8"))
                    cohort_id = f"oulad-extension-v1-{forecast.lower()}"
                    cursor.execute(
                        """INSERT INTO prediction_cohorts
                        (dataset_version_id, cohort_id, forecast_id, cutoff_contract, cohort_hash, record_count)
                        VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (cohort_id) DO NOTHING""",
                        (oulad_version, cohort_id, forecast, Json({"fraction": snapshot["forecast_fraction"], "rule": snapshot["cutoff_rule"]}), snapshot["cohort_hash"], snapshot["row_count"]),
                    )
                    cursor.execute("SELECT prediction_cohort_id FROM prediction_cohorts WHERE cohort_id=%s", (cohort_id,))
                    cohort_key = int(cursor.fetchone()[0])
                    flat_path = f"data/processed/study_c_oulad/flat/{forecast}.parquet"
                    cursor.execute(
                        """INSERT INTO cutoff_feature_snapshots
                        (prediction_cohort_id, feature_contract_hash, target_hash, parquet_relative_path, parquet_sha256, sequence_length, channel_order, channel_order_hash, row_count, status)
                        SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,'validated'
                        WHERE NOT EXISTS (SELECT 1 FROM cutoff_feature_snapshots WHERE prediction_cohort_id=%s AND feature_contract_hash=%s AND parquet_sha256=%s)""",
                        (cohort_key, snapshot["feature_contract_hash"], snapshot["target_hash"], flat_path, snapshot["checksums"]["flat"], snapshot["sequence_length"], Json(snapshot["channel_order"]), snapshot["channel_order_hash"], snapshot["row_count"], cohort_key, snapshot["feature_contract_hash"], snapshot["checksums"]["flat"]),
                    )
                    for role, relative in [("outer_cv", "data/processed/study_c_oulad/manifests/split_manifest.csv"), ("future_presentation", "data/processed/study_c_oulad/manifests/future_test_manifest.csv")]:
                        cursor.execute(
                            """INSERT INTO split_manifest_registry
                            (prediction_cohort_id, split_role, manifest_relative_path, manifest_sha256)
                            VALUES (%s,%s,%s,%s) ON CONFLICT (prediction_cohort_id, split_role, manifest_sha256) DO NOTHING""",
                            (cohort_key, role, relative, digest_file(ROOT / relative)),
                        )
                commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
                for run_id, version, study, path in [
                    (args.study_b_run, por_version, "study_b_student_por", f"artifacts/study_b_student_por/{args.study_b_run}"),
                    (args.study_c_run, oulad_version, "study_c_oulad", f"artifacts/study_c_oulad/{args.study_c_run}"),
                ]:
                    cursor.execute(
                        """INSERT INTO study_extension_runs
                        (extension_run_id, dataset_version_id, study_id, protocol_sha256, artifact_relative_path, source_commit, status, metadata)
                        VALUES (%s,%s,%s,%s,%s,%s,'completed',%s) ON CONFLICT (extension_run_id) DO NOTHING""",
                        (run_id, version, study, protocol_hash, path, commit, Json({"legacy_observed_accessed": False, "evidence_scope": "development_and_domain_shift"})),
                    )
                cursor.execute("SELECT count(*) FROM source_dataset_files WHERE dataset_version_id=%s", (oulad_version,)); source_file_rows = int(cursor.fetchone()[0])
                cursor.execute("SELECT count(*) FROM prediction_cohorts WHERE dataset_version_id=%s", (oulad_version,)); cohort_rows = int(cursor.fetchone()[0])
                cursor.execute("SELECT count(*) FROM cutoff_feature_snapshots c JOIN prediction_cohorts p USING (prediction_cohort_id) WHERE p.dataset_version_id=%s", (oulad_version,)); snapshot_rows = int(cursor.fetchone()[0])
                cursor.execute("SELECT count(*) FROM study_extension_runs WHERE extension_run_id IN (%s,%s)", (args.study_b_run, args.study_c_run)); run_rows = int(cursor.fetchone()[0])
        report |= {"status": "PASS", "database_registration": "PASS", "migration_execution": "PASS", "dataset_version_ids": {"student_por": por_version, "oulad": oulad_version}, "registered": {"source_files": source_file_rows, "cohorts": cohort_rows, "snapshots": snapshot_rows, "runs": run_rows}, "destructive_operations": False}
    except Exception as exc:
        report |= {"status": "PENDING", "database_registration": "PENDING", "migration_execution": "FAIL", "error_type": type(exc).__name__, "error": str(exc).splitlines()[0], "destructive_operations": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
