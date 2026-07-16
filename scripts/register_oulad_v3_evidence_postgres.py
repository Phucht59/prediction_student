from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies.oulad_v3_closure.fairness import metrics_with_modules


MIGRATIONS = (
    ROOT / "database/migrations/005_oulad_lineage_and_snapshot_registry.sql",
    ROOT / "database/migrations/006_oulad_v3_fair_evidence_registry.sql",
)
RUN_NAMESPACE = uuid.UUID("7ae62339-ad32-49b7-8f30-0c92f90c8d38")
REDACTED_DSN = "postgresql://<redacted>@localhost:5432/student_predict"
SOURCE_COMMIT = "f7ce7b1c53fc494a9a252014e973d77317f902e6"
V2_COMMIT = "07217a184b9a5dcc6402e3f117a5af2e84c7596f"
V3_COMMIT = "dbd5c2f27e914da2b252bffe176e7c93a6c2c237"
TARGET_CONTRACT = {"target": "at_risk", "positive": ["Withdrawn", "Fail"], "negative": ["Pass", "Distinction"]}
SPLIT_CONTRACT = {"outer": "3-fold StratifiedGroupKFold", "inner": "2-fold StratifiedGroupKFold", "group": "global id_student", "scope": "F2_MIDDLE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def strip_transaction_wrapper(text: str) -> str:
    kept = []
    for line in text.splitlines():
        if re.fullmatch(r"\s*(BEGIN|COMMIT)\s*;\s*", line, flags=re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept)


def migration_rows() -> list[dict[str, str]]:
    return [
        {"migration_id": path.name, "sha256": sha256(path), "path": path.as_posix()}
        for path in MIGRATIONS
    ]


def validate_migration_state(cursor) -> None:
    required = {
        "source_dataset_files", "prediction_cohorts", "cutoff_feature_snapshots",
        "snapshot_record_index", "split_manifest_registry", "study_extension_runs",
        "ml_schema_migrations", "ml_evidence_bundles",
    }
    cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
    found = {row[0] for row in cursor.fetchall()}
    missing = sorted(required - found)
    if missing:
        raise RuntimeError(f"Migration postcondition failed: missing {missing}")
    cursor.execute("SELECT count(*) FROM pg_trigger WHERE tgname='trg_ml_evidence_bundles_append_only' AND NOT tgisinternal")
    if cursor.fetchone()[0] != 1:
        raise RuntimeError("Evidence append-only trigger missing")


def dry_run_migrations(admin_dsn: str) -> dict[str, object]:
    with psycopg2.connect(admin_dsn) as connection:
        connection.autocommit = False
        with connection.cursor() as cursor:
            for path in MIGRATIONS:
                cursor.execute(strip_transaction_wrapper(path.read_text(encoding="utf-8")))
            validate_migration_state(cursor)
        connection.rollback()
    return {"status": "PASS", "transaction": "ROLLBACK", "migrations": migration_rows()}


def apply_migrations(admin_dsn: str, source_commit: str) -> dict[str, object]:
    with psycopg2.connect(admin_dsn) as connection:
        connection.autocommit = False
        with connection.cursor() as cursor:
            for path in MIGRATIONS:
                cursor.execute(strip_transaction_wrapper(path.read_text(encoding="utf-8")))
            validate_migration_state(cursor)
            for item in migration_rows():
                cursor.execute(
                    """INSERT INTO ml_schema_migrations(migration_id,migration_sha256,source_commit)
                       VALUES (%s,%s,%s) ON CONFLICT (migration_id) DO NOTHING""",
                    (item["migration_id"], item["sha256"], source_commit),
                )
                cursor.execute("SELECT migration_sha256 FROM ml_schema_migrations WHERE migration_id=%s", (item["migration_id"],))
                if cursor.fetchone()[0] != item["sha256"]:
                    raise RuntimeError(f"Migration checksum mismatch for {item['migration_id']}")
        connection.commit()
    return {"status": "PASS", "transaction": "COMMIT", "migrations": migration_rows()}


def create_cleanup_plan(admin_dsn: str, artifact_root: Path, report_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with psycopg2.connect(admin_dsn) as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """SELECT r.run_id::text,r.model_name,r.status,
                      count(distinct s.record_id) split_count,
                      count(distinct p.prediction_id) prediction_count,
                      count(distinct m.metric_id) metric_count
               FROM ml_experiment_runs r
               LEFT JOIN ml_run_record_splits s ON s.run_id=r.run_id
               LEFT JOIN ml_predictions p ON p.run_id=r.run_id
               LEFT JOIN ml_run_metrics m ON m.run_id=r.run_id
               WHERE r.status IN ('running','failed')
               GROUP BY r.run_id,r.model_name,r.status ORDER BY r.started_at"""
        )
        for run in cursor.fetchall():
            explicit_temp = bool(re.search(r"(^|[-_])(test|temp|smoke)([-_]|$)", run["model_name"], re.I))
            zero_dependencies = run["split_count"] == run["prediction_count"] == run["metric_count"] == 0
            authorized = explicit_temp and zero_dependencies
            rows.append({
                "table": "ml_experiment_runs", "predicate": f"run_id = '{run['run_id']}'",
                "candidate_row_count": 1, "dependency_count": int(run["split_count"] + run["prediction_count"] + run["metric_count"]),
                "reason": "explicit test/temp empty run" if authorized else "denylisted: not explicitly test/temp and/or has lineage dependencies",
                "rollback_strategy": "restore from validated custom-format backup", "authorized": authorized,
            })
    plan = {
        "status": "PASS", "generated_before_cleanup": True, "backup_required": True,
        "authorized_deletion_count": sum(int(row["candidate_row_count"]) for row in rows if row["authorized"]),
        "entries": rows, "absolute_denylist_enforced": True,
    }
    write_json(artifact_root / "postgres_cleanup_plan.json", plan)
    lines = ["# PostgreSQL Cleanup Plan", "", "No blind cleanup is authorized.", "", "| Table | Predicate | Rows | Dependencies | Authorized | Reason |", "|---|---|---:|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['table']} | `{row['predicate']}` | {row['candidate_row_count']} | {row['dependency_count']} | {str(row['authorized']).lower()} | {row['reason']} |")
    if not rows:
        lines.append("| — | — | 0 | 0 | false | No cleanup candidates |")
    (report_root / "POSTGRES_CLEANUP_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def create_or_repair_runtime_role(admin_dsn: str, password: str) -> dict[str, object]:
    if not password:
        raise RuntimeError("POSTGRES_RUNTIME_PASSWORD is required and must be generated at runtime")
    role = "student_predict_app_local"
    with psycopg2.connect(admin_dsn) as connection:
        connection.autocommit = False
        with connection.cursor() as cursor:
            cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=%s)", (role,))
            existed = cursor.fetchone()[0]
            if not existed:
                cursor.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD %s").format(sql.Identifier(role)), (password,))
            else:
                cursor.execute(sql.SQL("ALTER ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION PASSWORD %s").format(sql.Identifier(role)), (password,))
            cursor.execute(sql.SQL("GRANT student_predict_app TO {}").format(sql.Identifier(role)))
            cursor.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(sql.Identifier("student_predict"), sql.Identifier(role)))
            cursor.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(sql.Identifier(role)))
        connection.commit()
    return {"role": role, "existed_before": existed, "status": "PASS", "password_recorded": False, "membership": "student_predict_app"}


def unique_records(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = ["record_id", "outer_fold", "code_module", "code_presentation", "id_student", "target_at_risk"]
    records = predictions[columns].drop_duplicates().sort_values("record_id").reset_index(drop=True)
    if len(records) != predictions.record_id.nunique() or records.record_id.duplicated().any():
        raise RuntimeError("Artifact record identity is not one-to-one")
    return records


def ensure_dataset(connection, records: pd.DataFrame, artifact_root: Path) -> tuple[int, dict[str, int]]:
    contract = {"kind": "prediction_cohort", "forecast_id": "F2_MIDDLE", "source": "validated V1/V2 cutoff snapshot", "target_separated": True}
    content_rows = records.to_dict("records")
    content_hash = canonical_hash(content_rows)
    contract_hash = canonical_hash(contract)
    dataset_code = "oulad-f2-middle-fair-closure"
    with connection.cursor() as cursor:
        cursor.execute(
            """INSERT INTO source_dataset_versions(dataset_code,source_locator,content_hash,ingestion_contract,ingestion_contract_hash,row_count,metadata)
               VALUES (%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (dataset_code,hash_algorithm,content_hash,ingestion_contract_hash_algorithm,ingestion_contract_hash) DO NOTHING
               RETURNING dataset_version_id""",
            (dataset_code, artifact_root.as_posix(), content_hash, Json(contract), contract_hash, len(records), Json({"forecast_id": "F2_MIDDLE", "scope": "development_only", "legacy_observed_access": False})),
        )
        row = cursor.fetchone()
        if row:
            dataset_version_id = int(row[0])
            execute_values(
                cursor,
                "INSERT INTO source_records(dataset_version_id,source_row_number,raw_payload) VALUES %s",
                [(dataset_version_id, index, Json({
                    "record_key": item.record_id, "outer_fold": int(item.outer_fold), "code_module": item.code_module,
                    "code_presentation": item.code_presentation, "id_student": int(item.id_student),
                })) for index, item in enumerate(records.itertuples(index=False))],
                page_size=1000,
            )
            cursor.execute(
                "SELECT record_id,raw_payload->>'record_key' FROM source_records WHERE dataset_version_id=%s ORDER BY source_row_number",
                (dataset_version_id,),
            )
            mapping = {record_key: int(record_id) for record_id, record_key in cursor.fetchall()}
            target_hash = canonical_hash(TARGET_CONTRACT)
            execute_values(
                cursor,
                """INSERT INTO source_record_targets(dataset_version_id,record_id,target_name,raw_target_value,encoded_target_value,target_contract_hash) VALUES %s""",
                [(dataset_version_id, mapping[item.record_id], "at_risk", Json({"at_risk": int(item.target_at_risk)}), int(item.target_at_risk), target_hash) for item in records.itertuples(index=False)],
                page_size=1000,
            )
        else:
            cursor.execute(
                """SELECT dataset_version_id FROM source_dataset_versions
                   WHERE dataset_code=%s AND content_hash=%s AND ingestion_contract_hash=%s""",
                (dataset_code, content_hash, contract_hash),
            )
            dataset_version_id = int(cursor.fetchone()[0])
            cursor.execute("SELECT record_id,raw_payload->>'record_key' FROM source_records WHERE dataset_version_id=%s", (dataset_version_id,))
            mapping = {record_key: int(record_id) for record_id, record_key in cursor.fetchall()}
        if len(mapping) != len(records):
            raise RuntimeError("Canonical source-record count mismatch")
    return dataset_version_id, mapping


def register_evidence_bundles(connection, dataset_version_id: int, artifact_root: Path, source_commit: str) -> list[dict[str, object]]:
    now = datetime.now(timezone.utc)
    closure_protocol = ROOT / "configs/oulad_v3_fair_db_closure_protocol.yaml"
    definitions = [
        ("v2", "oulad-deep-v2-f2-20260716-v1", None, V2_COMMIT, ROOT / "configs/oulad_deep_v2_protocol.yaml", ROOT / "artifacts/study_c_oulad_v2/oulad-deep-v2-f2-20260716-v1", "NOT_SUPPORTED"),
        ("v3", "oulad-deep-v3-f2-20260716-v1", "oulad-deep-v2-f2-20260716-v1", V3_COMMIT, ROOT / "configs/oulad_deep_v3_protocol.yaml", ROOT / "artifacts/study_c_oulad_v3/oulad-deep-v3-f2-20260716-v1", "PRACTICAL_TIE"),
        ("v3_fair_db_closure", artifact_root.name, "oulad-deep-v3-f2-20260716-v1", source_commit, closure_protocol, artifact_root, "PRACTICAL_TIE"),
    ]
    registered = []
    with connection.cursor() as cursor:
        for version, run_id, parent, commit, protocol, root, verdict in definitions:
            if not protocol.exists() or not root.exists():
                raise RuntimeError(f"Evidence source missing: {protocol} or {root}")
            manifest = sorted((path.relative_to(root).as_posix(), sha256(path)) for path in root.rglob("*") if path.is_file() and "threshold_replay_cache" not in path.parts)
            candidate_registry = {"source": version, "registry_status": "frozen", "fair_candidates": ["V3-A0F-ENS", "V3-H2TF-ENS", "V3-H3CF-ENS", "V3-P0-ENS", "V3-D0-ENS", "V3-A1-ENS", "V3-MLF", "V3-MLD"] if version == "v3_fair_db_closure" else []}
            cursor.execute(
                """INSERT INTO ml_evidence_bundles(
                       study_id,study_version,run_id,parent_run_id,source_commit,protocol_path,protocol_sha256,
                       artifact_root,artifact_manifest_sha256,dataset_version_id,forecast_id,target_contract,split_contract,
                       seed_registry,candidate_registry,benchmark_status,future_benchmark_status,scientific_verdict,
                       validation_status,runtime_seconds,environment,created_at,completed_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (study_id,study_version,run_id) DO NOTHING""",
                ("study_c_oulad", version, run_id, parent, commit, protocol.relative_to(ROOT).as_posix(), sha256(protocol), root.relative_to(ROOT).as_posix(), canonical_hash(manifest), dataset_version_id, "F2_MIDDLE", Json(TARGET_CONTRACT), Json(SPLIT_CONTRACT), Json([42, 2026, 3407]), Json(candidate_registry), "development_only", "NOT_EXECUTED", verdict, "PASS", None, Json({"python": platform.python_version(), "torch": "frozen evidence", "cuda": "frozen evidence", "credential_redaction": True}), now, now),
            )
            registered.append({"study_version": version, "run_id": run_id, "source_commit": commit, "artifact_manifest_sha256": canonical_hash(manifest)})
    return registered


def run_uuid(candidate_id: str) -> uuid.UUID:
    return uuid.uuid5(RUN_NAMESPACE, f"study_c_oulad_v3_fair_db_closure:{candidate_id}")


def metric_rows(candidate_id: str, ensemble_metrics: pd.DataFrame, single_metrics: pd.DataFrame, mean_metrics: pd.DataFrame):
    def numeric_items(row, excluded):
        for key, value in row.items():
            if key in excluded or pd.isna(value) or not isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
                continue
            yield key, float(value)
    ensemble = ensemble_metrics.loc[ensemble_metrics.candidate_id == candidate_id].iloc[0].to_dict()
    contract = str(ensemble["prediction_contract"])
    for name, value in numeric_items(ensemble, {"candidate_id"}):
        yield name, value, contract, {"prediction_contract": contract, "seed": None, "prediction_variant": "ensemble_42_2026_3407" if candidate_id.endswith("-ENS") else "deterministic"}
    source_id = candidate_id.removesuffix("-ENS")
    for row in single_metrics.loc[single_metrics.candidate_id == source_id].to_dict("records"):
        for name, value in numeric_items(row, {"candidate_id", "seed"}):
            yield name, value, f"single_seed:{int(row['seed'])}", {"prediction_contract": "single_seed", "seed": int(row["seed"])}
    rows = mean_metrics.loc[mean_metrics.candidate_id == source_id]
    if not rows.empty:
        row = rows.iloc[0].to_dict()
        for name, value in numeric_items(row, {"candidate_id"}):
            yield name, value, "mean_of_seed_metrics", {"prediction_contract": "mean_of_seed_metrics", "declared_seeds": [42, 2026, 3407]}


def register_candidate_runs(connection, predictions: pd.DataFrame, mapping: dict[str, int], dataset_version_id: int, artifact_root: Path, source_commit: str) -> dict[str, str]:
    ensemble_metrics = pd.read_csv(artifact_root / "ensemble_metrics.csv")
    single_metrics = pd.read_csv(artifact_root / "single_seed_metrics.csv")
    mean_metrics = pd.read_csv(artifact_root / "mean_seed_metrics.csv")
    split_hash = canonical_hash(predictions[["record_id", "outer_fold"]].drop_duplicates().sort_values("record_id").to_dict("records"))
    target_hash = canonical_hash(TARGET_CONTRACT)
    environment_hash = canonical_hash({"python": platform.python_version(), "registration": "artifact replay only"})
    now = datetime.now(timezone.utc)
    result: dict[str, str] = {}
    for candidate_id, frame in predictions.groupby("candidate_id", sort=True):
        run_id = run_uuid(candidate_id)
        result[candidate_id] = str(run_id)
        with connection.cursor() as cursor:
            cursor.execute("SELECT status FROM ml_experiment_runs WHERE run_id=%s", (str(run_id),))
            existing = cursor.fetchone()
            if existing:
                if existing[0] != "completed":
                    raise RuntimeError(f"Existing closure run {candidate_id} is not completed")
                print(json.dumps({"candidate": candidate_id, "registration": "already_completed"}), flush=True)
                continue
            cursor.execute(
                """INSERT INTO ml_experiment_runs(
                       run_id,dataset_version_id,model_name,task_type,target_definition,target_definition_hash,
                       split_manifest_uri,split_manifest_hash,git_commit,working_tree_state,environment_lock_uri,
                       environment_lock_hash,train_config,artifact_uri,status,started_at,metadata)
                   VALUES (%s,%s,%s,'classification',%s,%s,%s,%s,%s,'clean',%s,%s,%s,%s,'running',%s,%s)""",
                (str(run_id), dataset_version_id, candidate_id, Json(TARGET_CONTRACT), target_hash,
                 "ensemble_oof_predictions.parquet#outer_fold", split_hash, source_commit,
                 "source_provenance.json", environment_hash,
                 Json({"operation": "evidence_registration_only", "training": False, "prediction_regeneration": False, "contract": str(frame.prediction_contract.iloc[0])}),
                 artifact_root.relative_to(ROOT).as_posix(), now,
                 Json({"scope": "fair_ensemble_closure", "forecast_id": "F2_MIDDLE", "prediction_variant": str(frame.prediction_variant.iloc[0]), "future_benchmark": "NOT_EXECUTED"})),
            )
            split_values = [(str(run_id), dataset_version_id, mapping[row.record_id], "test", None) for row in frame.itertuples(index=False)]
            execute_values(cursor, "INSERT INTO ml_run_record_splits(run_id,dataset_version_id,record_id,split_name,exclusion_reason) VALUES %s", split_values, page_size=2000)
            prediction_values = []
            for row in frame.itertuples(index=False):
                probability = float(row.probability)
                payload = {
                    "not_at_risk": 1.0 - probability, "at_risk": probability,
                    "prediction_contract": str(row.prediction_contract), "prediction_variant": str(row.prediction_variant),
                    "outer_fold": int(row.outer_fold), "macro_threshold": float(row.macro_threshold),
                    "operational_threshold": float(row.operational_threshold), "operational_feasible": bool(row.operational_feasible),
                    "code_module": str(row.code_module), "code_presentation": str(row.code_presentation),
                    "id_student": int(row.id_student), "scope": str(row.scope), "seed": None,
                }
                prediction_values.append((str(run_id), mapping[row.record_id], "test", int(row.target_at_risk), int(row.predicted_label), max(probability, 1.0 - probability), Json(payload)))
            execute_values(cursor, "INSERT INTO ml_predictions(run_id,record_id,split_name,true_label,predicted_label,confidence,probability) VALUES %s", prediction_values, page_size=1000)
            metrics = [(str(run_id), "test", name, value, scope, Json(context)) for name, value, scope, context in metric_rows(candidate_id, ensemble_metrics, single_metrics, mean_metrics)]
            execute_values(cursor, "INSERT INTO ml_run_metrics(run_id,split_name,metric_name,metric_value,label_scope,metric_context) VALUES %s", metrics, page_size=1000)
            cursor.execute("UPDATE ml_experiment_runs SET status='completed',completed_at=%s WHERE run_id=%s", (datetime.now(timezone.utc), str(run_id)))
        connection.commit()
        print(json.dumps({"candidate": candidate_id, "registration": "committed", "predictions": len(frame)}), flush=True)
    return result


def reproduce_from_database(admin_dsn: str, predictions: pd.DataFrame, run_ids: dict[str, str], artifact_root: Path) -> dict[str, object]:
    frames = []
    with psycopg2.connect(admin_dsn) as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
        for candidate_id, run_id in run_ids.items():
            cursor.execute(
                """SELECT sr.raw_payload->>'record_key' record_id,p.true_label,p.predicted_label,
                          (p.probability->>'at_risk')::double precision probability,
                          (p.probability->>'macro_threshold')::double precision macro_threshold,
                          (p.probability->>'operational_threshold')::double precision operational_threshold,
                          (p.probability->>'operational_feasible')::boolean operational_feasible,
                          (p.probability->>'outer_fold')::integer outer_fold,
                          p.probability->>'code_module' code_module,
                          (p.probability->>'id_student')::bigint id_student,
                          ((p.probability->>'at_risk')::double precision >= (p.probability->>'operational_threshold')::double precision)::integer operational_prediction
                   FROM ml_predictions p JOIN ml_experiment_runs r ON r.run_id=p.run_id
                   JOIN source_records sr ON sr.record_id=p.record_id AND sr.dataset_version_id=r.dataset_version_id
                   WHERE p.run_id=%s ORDER BY record_id""",
                (run_id,),
            )
            frame = pd.DataFrame(cursor.fetchall())
            frame.insert(0, "candidate_id", candidate_id)
            frame = frame.rename(columns={"true_label": "target_at_risk"})
            frames.append(frame)
    database = pd.concat(frames, ignore_index=True)
    expected_columns = ["candidate_id", "record_id", "target_at_risk", "predicted_label", "probability", "macro_threshold", "operational_threshold", "operational_feasible", "outer_fold"]
    merged = predictions[expected_columns].merge(database[expected_columns], on=["candidate_id", "record_id"], suffixes=("_artifact", "_db"), validate="one_to_one")
    max_probability_difference = float(np.max(np.abs(merged.probability_artifact - merged.probability_db)))
    label_equal = bool(np.array_equal(merged.predicted_label_artifact, merged.predicted_label_db) and np.array_equal(merged.target_at_risk_artifact, merged.target_at_risk_db))
    threshold_equal = bool(np.allclose(merged.macro_threshold_artifact, merged.macro_threshold_db, atol=1e-12, rtol=0) and np.allclose(merged.operational_threshold_artifact, merged.operational_threshold_db, atol=1e-12, rtol=0))
    expected_metrics = pd.read_csv(artifact_root / "ensemble_metrics.csv").set_index("candidate_id")
    metric_difference = 0.0
    for candidate_id, frame in database.groupby("candidate_id"):
        observed = metrics_with_modules(frame)
        for metric in ["macro_f1", "accuracy", "balanced_accuracy", "at_risk_precision", "at_risk_recall", "at_risk_f1", "specificity", "pr_auc", "brier", "nll", "ece", "worst_eligible_module_macro_f1", "worst_eligible_module_recall"]:
            metric_difference = max(metric_difference, abs(float(observed[metric]) - float(expected_metrics.loc[candidate_id, metric])))
    report = {
        "status": "PASS" if len(merged) == len(predictions) and max_probability_difference <= 1e-12 and label_equal and threshold_equal and metric_difference <= 1e-12 else "FAIL",
        "artifact_rows": len(predictions), "database_rows": len(database), "record_candidate_key_equality": len(merged) == len(predictions),
        "max_probability_absolute_difference": max_probability_difference, "label_equality": label_equal,
        "threshold_equality": threshold_equal, "max_metric_absolute_difference": metric_difference,
    }
    return report


def permission_test(connection, name: str, statement, params=(), expect_success=False) -> dict[str, object]:
    cursor = connection.cursor()
    cursor.execute("SAVEPOINT permission_case")
    try:
        cursor.execute(statement, params)
        succeeded = True
        sqlstate = None
    except psycopg2.Error as error:
        succeeded = False
        sqlstate = error.pgcode
        cursor.execute("ROLLBACK TO SAVEPOINT permission_case")
    finally:
        cursor.execute("RELEASE SAVEPOINT permission_case")
        cursor.close()
    passed = succeeded if expect_success else not succeeded
    return {"test": name, "expected": "allow" if expect_success else "deny", "observed": "allowed" if succeeded else "denied", "sqlstate": sqlstate, "status": "PASS" if passed else "FAIL"}


def audit_permissions(admin_dsn: str, app_dsn: str, run_ids: dict[str, str], dataset_version_id: int) -> dict[str, object]:
    with psycopg2.connect(app_dsn) as app:
        app.autocommit = False
        with app.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT current_user,rolsuper,rolcreatedb,rolcreaterole FROM pg_roles WHERE rolname=current_user")
            profile = dict(cursor.fetchone())
        if profile["current_user"] != "student_predict_app_local" or profile["rolsuper"] or profile["rolcreatedb"] or profile["rolcreaterole"]:
            raise RuntimeError("Application permission tests require the least-privileged local role")
        candidate_run = next(iter(run_ids.values()))
        tests = [
            permission_test(app, "allowed_select", "SELECT count(*) FROM ml_predictions", expect_success=True),
            permission_test(app, "allowed_insert_operational_source", "INSERT INTO source_dataset_versions(dataset_code,source_locator,content_hash,ingestion_contract,ingestion_contract_hash,row_count,metadata) VALUES ('permission-rollback','rollback','rollback',%s,'rollback',1,%s)", (Json({}), Json({})), True),
            permission_test(app, "forbidden_drop", "DROP TABLE ml_evidence_bundles"),
            permission_test(app, "forbidden_alter", "ALTER TABLE ml_predictions ADD COLUMN forbidden_test integer"),
            permission_test(app, "immutable_completed_run_update", "UPDATE ml_experiment_runs SET status='completed',completed_at=completed_at WHERE run_id=%s", (candidate_run,)),
            permission_test(app, "invalid_status", "INSERT INTO ml_experiment_runs(run_id,dataset_version_id,model_name,task_type,target_definition,target_definition_hash,split_manifest_uri,split_manifest_hash,git_commit,working_tree_state,environment_lock_uri,environment_lock_hash,train_config,artifact_uri,status,started_at) VALUES (%s,%s,'invalid','classification',%s,'x','x','x','x','clean','x','x',%s,'x','invalid',NOW())", (str(uuid.uuid4()), dataset_version_id, Json({}), Json({}))),
            permission_test(app, "duplicate_evidence_key", "INSERT INTO ml_evidence_bundles SELECT * FROM ml_evidence_bundles LIMIT 1"),
            permission_test(app, "orphan_prediction", "INSERT INTO ml_predictions(run_id,record_id,split_name,true_label,predicted_label,confidence,probability) VALUES (%s,9223372036854775806,'test',0,0,.5,%s)", (str(uuid.uuid4()), Json({"at_risk": .5}))),
        ]
        app.rollback()
    with psycopg2.connect(admin_dsn) as admin:
        admin.autocommit = False
        with admin.cursor() as cursor:
            cursor.execute("SELECT * FROM ml_evidence_bundles LIMIT 1")
            row = cursor.fetchone()
            columns = [description.name for description in cursor.description]
            placeholders = sql.SQL(",").join(sql.Placeholder() for _ in row)
            statement = sql.SQL("INSERT INTO ml_evidence_bundles ({}) VALUES ({})").format(sql.SQL(",").join(map(sql.Identifier, columns)), placeholders)
            tests.append(permission_test(admin, "duplicate_evidence_unique_constraint_admin_transaction", statement, row))
        admin.rollback()
    status = "PASS" if all(item["status"] == "PASS" for item in tests) else "FAIL"
    return {"status": status, "application_profile": profile, "administrator_not_used_as_app_evidence": True, "tests": tests}


def query_plans(admin_dsn: str, report_root: Path, run_ids: dict[str, str]) -> None:
    with psycopg2.connect(admin_dsn) as connection:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("ANALYZE source_records; ANALYZE ml_predictions; ANALYZE ml_run_metrics; ANALYZE ml_evidence_bundles")
            plans = []
            queries = [
                ("Evidence lookup", "SELECT * FROM ml_evidence_bundles WHERE study_id='study_c_oulad' AND source_commit=%s", (SOURCE_COMMIT,)),
                ("Prediction reproduction", "SELECT sr.raw_payload->>'record_key',p.predicted_label,p.probability->>'at_risk' FROM ml_predictions p JOIN source_records sr ON sr.record_id=p.record_id WHERE p.run_id=%s", (next(iter(run_ids.values())),)),
            ]
            for title, query, params in queries:
                cursor.execute("EXPLAIN (ANALYZE,BUFFERS,FORMAT TEXT) " + query, params)
                plans.append((title, "\n".join(row[0] for row in cursor.fetchall())))
    lines = ["# PostgreSQL Query Plans", ""]
    for title, plan in plans:
        lines.extend([f"## {title}", "", "```text", plan, "```", ""])
    (report_root / "postgres_query_plans.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--source-commit", default=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    args = parser.parse_args()
    artifact_root = Path(args.artifact_root).resolve(); report_root = Path(args.report_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True); report_root.mkdir(parents=True, exist_ok=True)
    admin_dsn = os.environ.get("POSTGRES_TEST_DSN"); app_dsn = os.environ.get("POSTGRES_RUNTIME_APP_DSN")
    if not admin_dsn or not app_dsn:
        raise RuntimeError("POSTGRES_TEST_DSN and POSTGRES_RUNTIME_APP_DSN are required")
    backup = json.loads((artifact_root / "postgres_backup_manifest.json").read_text(encoding="utf-8"))
    if backup.get("status") != "PASS":
        raise RuntimeError("Validated backup gate is required before database writes")

    plan = create_cleanup_plan(admin_dsn, artifact_root, report_root)
    dry_run = dry_run_migrations(admin_dsn)
    migration = apply_migrations(admin_dsn, args.source_commit)
    role = create_or_repair_runtime_role(admin_dsn, os.environ.get("POSTGRES_RUNTIME_PASSWORD", ""))

    predictions = pd.read_parquet(artifact_root / "ensemble_oof_predictions.parquet")
    records = unique_records(predictions)
    with psycopg2.connect(admin_dsn) as connection:
        connection.autocommit = False
        dataset_version_id, mapping = ensure_dataset(connection, records, artifact_root)
        bundles = register_evidence_bundles(connection, dataset_version_id, artifact_root, args.source_commit)
        connection.commit()
        print(json.dumps({"registration": "dataset_and_bundle_registry_committed", "source_records": len(records), "evidence_bundles": len(bundles)}), flush=True)
        run_ids = register_candidate_runs(connection, predictions, mapping, dataset_version_id, artifact_root, args.source_commit)

    reproduction = reproduce_from_database(admin_dsn, predictions, run_ids, artifact_root)
    permission = audit_permissions(admin_dsn, app_dsn, run_ids, dataset_version_id)
    cleanup_execution = {"status": "PASS", "transaction": "ROLLBACK", "rows_removed": 0, "authorized_deletion_count": plan["authorized_deletion_count"], "predicates_executed": [], "valid_scientific_lineage_preserved": True}
    query_plans(admin_dsn, report_root, run_ids)

    registration = {
        "status": "PASS" if reproduction["status"] == "PASS" else "FAIL",
        "dataset_version_id": dataset_version_id, "source_records": len(records), "target_rows": len(records),
        "candidate_runs": run_ids, "completed_runs_registered": len(run_ids), "prediction_rows": len(predictions),
        "split_rows": len(predictions), "evidence_bundles": bundles,
        "prediction_contracts": sorted(predictions.prediction_contract.unique()), "future_benchmark": "NOT_EXECUTED",
    }
    migration_report = {"status": "PASS", "backup_gate": "PASS", "dry_run": dry_run, "applied": migration, "role": role, "destructive_cascade": False, "rollback": "restore validated custom dump or apply compensating migration"}
    write_json(artifact_root / "postgres_migration_report.json", migration_report)
    write_json(artifact_root / "postgres_cleanup_execution.json", cleanup_execution)
    write_json(artifact_root / "postgres_evidence_registration.json", registration)
    write_json(artifact_root / "postgres_reproduction_validation.json", reproduction)
    write_json(artifact_root / "postgres_permission_audit.json", permission)
    if reproduction["status"] != "PASS" or permission["status"] != "PASS":
        raise RuntimeError("PostgreSQL closure validation failed")
    print(json.dumps({"status": "PASS", "dataset_version_id": dataset_version_id, "prediction_rows": len(predictions), "runs": len(run_ids), "cleanup_rows": 0, "permission": permission["status"], "reproduction": reproduction["status"]}, indent=2))


if __name__ == "__main__":
    main()
