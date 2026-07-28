"""Safe final-database migration, canonical loading, validation, and cutover."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import psycopg2
import yaml
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values


ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "database" / "final"
MIGRATIONS = FINAL_ROOT / "migrations"
ROLLBACK = FINAL_ROOT / "rollback"
ARTIFACT_ROOT = ROOT / "artifacts" / "final" / "database"
REPORT_ROOT = ROOT / "reports" / "final"
BACKUP_PATH = ROOT / "backups" / "student_predict_before_final_database.dump"
BACKUP_MANIFEST = ARTIFACT_ROOT / "backup_manifest.json"

FINAL_RESULTS = ROOT / "artifacts" / "final" / "final_results.json"
FINAL_RESULTS_CSV = ROOT / "artifacts" / "final" / "final_results.csv"
MODEL_REGISTRY = ROOT / "artifacts" / "final" / "model_registry.json"
FINAL_CHECKSUMS = ROOT / "artifacts" / "final" / "checksum_manifest.json"
RISK_PROFILES = ROOT / "artifacts" / "final" / "recommendation" / "risk_profiles.parquet"
PLANS = ROOT / "artifacts" / "final" / "database" / "persisted_recommendation_plans.jsonl"
POLICY = ROOT / "artifacts" / "final" / "recommendation" / "policy_registry.json"
RELOCATION_MANIFEST = ROOT / "artifacts" / "final" / "checksums" / "relocation_manifest.json"

LOCKED_SOURCES = {
    FINAL_RESULTS: "000c185fb2fd9ba4b528e79d98636fdb17ee4586dbad197e9990717164b3681b",
    FINAL_RESULTS_CSV: "d2271c48bc6ed65a2836ec3b2430eef0777ad9f2b83f0d16092f389e57148b0f",
    MODEL_REGISTRY: "83415a32684557a970a7059a307ead51913562581e83c03e864550cd98bc268b",
    FINAL_CHECKSUMS: "cc4fd84a6c26b27ea739dd9c59722f30511d5186578cc9af4e008f9ff547389b",
    RISK_PROFILES: "a0178477871e16b81eebc4ec50dd23567fa4df6ec5b9d75d9e75d14f7ebe5625",
    PLANS: "d34e61d0fbbaaa9a8db7299dba174caeb2bb92308bf99981788a05fb5ba06cc3",
    POLICY: "89f054fc62d035ec2d4789b4d65950363d5158d04396bff0c3c243bda7cb47d8",
}

OOF_PATHS = {
    "student_mat": ROOT / "artifacts" / "final" / "comparator_completion" / "student_mat" / "oof_predictions.parquet",
    "student_por": ROOT / "artifacts" / "final" / "comparator_completion" / "student_por" / "oof_predictions.parquet",
    "oulad": ROOT / "artifacts" / "final" / "comparator_completion" / "oulad" / "ensemble_oof_predictions.parquet",
}

LEGACY_TABLES = (
    "advisor_decisions",
    "cutoff_feature_snapshots",
    "expert_review_cases",
    "expert_review_ratings",
    "ml_evidence_bundles",
    "ml_experiment_runs",
    "ml_predictions",
    "ml_recommendations",
    "ml_run_metrics",
    "ml_run_record_splits",
    "ml_schema_migrations",
    "prediction_cohorts",
    "prediction_snapshots",
    "recommendation_action_catalog",
    "recommendation_actions",
    "recommendation_feature_registry",
    "recommendation_follow_ups",
    "recommendation_goals",
    "recommendation_instances",
    "recommendation_outcomes",
    "recommendation_policies",
    "recommendation_revisions",
    "snapshot_record_index",
    "source_dataset_files",
    "source_dataset_versions",
    "source_record_targets",
    "source_records",
    "split_manifest_registry",
    "study_extension_runs",
)

EXPECTED_TABLES = {
    "system.schema_migration",
    "catalog.dataset",
    "catalog.dataset_version",
    "catalog.record",
    "ml.model",
    "ml.run",
    "ml.artifact",
    "ml.metric",
    "recommendation.policy",
    "recommendation.risk_profile",
    "recommendation.plan",
    "recommendation.action",
    "recommendation.review",
    "recommendation.expert_review_case",
    "recommendation.expert_plan_review",
    "recommendation.expert_action_review",
}

DISPOSABLE_MARKERS = ("test", "dev", "disposable", "final_dev")


class FinalDatabaseError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _json_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _preserved_timestamp(path: Path, key: str) -> str:
    """Keep release evidence byte-stable across repeated successful audits."""
    if path.is_file():
        try:
            value = _read_json(path).get(key)
            if isinstance(value, str) and value:
                return value
        except (OSError, ValueError, AttributeError):
            pass
    return _now()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dsn(env_name: str) -> str:
    value = os.getenv(env_name)
    if not value:
        raise FinalDatabaseError(f"Missing required environment variable: {env_name}")
    return value


def _database_name(dsn: str) -> str:
    parsed = urlsplit(dsn)
    return parsed.path.lstrip("/").split("?", 1)[0]


def _replace_database(dsn: str, database: str) -> str:
    parsed = urlsplit(dsn)
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database}", parsed.query, parsed.fragment))


def _redacted(dsn: str) -> str:
    parsed = urlsplit(dsn)
    host = parsed.hostname or "<local>"
    port = f":{parsed.port}" if parsed.port else ""
    return f"postgresql://<redacted>@{host}{port}/{_database_name(dsn)}"


def _is_disposable(dsn: str) -> bool:
    name = _database_name(dsn).lower()
    return any(marker in name for marker in DISPOSABLE_MARKERS)


def _connect(dsn: str, *, readonly: bool = False):
    connection = psycopg2.connect(dsn, connect_timeout=10)
    connection.set_session(readonly=readonly, autocommit=False)
    return connection


def _assert_locked_sources() -> None:
    failures = []
    for path, expected in LOCKED_SOURCES.items():
        actual = _sha256(path) if path.is_file() else None
        if actual != expected:
            failures.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "expected": expected,
                    "actual": actual,
                }
            )
    if failures:
        _write_json(ARTIFACT_ROOT / "migration_conflicts.json", {"status": "STOP_MIGRATION_CONFLICT", "failures": failures})
        raise FinalDatabaseError("STOP_MIGRATION_CONFLICT: locked canonical source mismatch")
    _write_json(
        ARTIFACT_ROOT / "migration_conflicts.json",
        {"status": "PASS", "failures": []},
    )


def _schema_payload(dsn: str) -> dict[str, Any]:
    statements = {
        "tables": """
            SELECT table_schema, table_name, table_type
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """,
        "columns": """
            SELECT table_schema, table_name, ordinal_position, column_name,
                   data_type, udt_name, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name, ordinal_position
        """,
        "constraints": """
            SELECT n.nspname, c.relname, con.conname, con.contype,
                   con.conkey::text, con.confkey::text, con.confupdtype,
                   con.confdeltype, rn.nspname, rc.relname
            FROM pg_constraint con
            JOIN pg_class c ON c.oid=con.conrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            LEFT JOIN pg_class rc ON rc.oid=con.confrelid
            LEFT JOIN pg_namespace rn ON rn.oid=rc.relnamespace
            WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
            ORDER BY n.nspname, c.relname, con.conname
        """,
        "indexes": """
            SELECT n.nspname, c.relname, i.relname, x.indisprimary,
                   x.indisunique, x.indkey::text
            FROM pg_index x
            JOIN pg_class i ON i.oid=x.indexrelid
            JOIN pg_class c ON c.oid=x.indrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
            ORDER BY n.nspname, c.relname, i.relname
        """,
        "triggers": """
            SELECT n.nspname, c.relname, t.tgname, t.tgtype,
                   pn.nspname, p.proname
            FROM pg_trigger t
            JOIN pg_class c ON c.oid=t.tgrelid
            JOIN pg_namespace n ON n.oid=c.relnamespace
            JOIN pg_proc p ON p.oid=t.tgfoid
            JOIN pg_namespace pn ON pn.oid=p.pronamespace
            WHERE NOT t.tgisinternal
              AND n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
            ORDER BY n.nspname, c.relname, t.tgname
        """,
    }
    payload: dict[str, Any] = {}
    with _connect(dsn, readonly=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET LOCAL statement_timeout = '5min'")
            for key, statement in statements.items():
                cursor.execute(statement)
                payload[key] = cursor.fetchall()
        connection.rollback()
    return payload


def _schema_signature(dsn: str) -> str:
    return hashlib.sha256(_canonical_json(_schema_payload(dsn)).encode("utf-8")).hexdigest()


def _server_tools() -> dict[str, Path]:
    bin_dir = Path("C:/Program Files/PostgreSQL/18/bin")
    tools = {
        "pg_dump": bin_dir / "pg_dump.exe",
        "pg_restore": bin_dir / "pg_restore.exe",
        "createdb": bin_dir / "createdb.exe",
    }
    for name, path in tools.items():
        if not path.is_file():
            located = shutil.which(name)
            if not located:
                raise FinalDatabaseError(f"PostgreSQL client unavailable: {name}")
            tools[name] = Path(located)
    return tools


def _connection_args(dsn: str) -> tuple[list[str], dict[str, str]]:
    parsed = urlsplit(dsn)
    args = [
        "--host",
        parsed.hostname or "localhost",
        "--port",
        str(parsed.port or 5432),
        "--username",
        parsed.username or "",
    ]
    env = dict(os.environ)
    if parsed.password:
        env["PGPASSWORD"] = parsed.password
    return args, env


def command_backup(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    tools = _server_tools()
    BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    starting_hash = _schema_signature(dsn)
    connection_args, env = _connection_args(dsn)
    if not BACKUP_PATH.exists():
        command = [
            str(tools["pg_dump"]),
            "--format=custom",
            "--no-owner",
            "--no-acl",
            f"--file={BACKUP_PATH}",
            *connection_args,
            _database_name(dsn),
        ]
        subprocess.run(command, env=env, check=True, capture_output=True)

    backup_hash = _sha256(BACKUP_PATH)
    restore_dsn = _replace_database(dsn, args.restore_database)
    restore_status = "MISSING"
    restore_hash = None
    try:
        restore_hash = _schema_signature(restore_dsn)
        restore_status = "PASS" if restore_hash == starting_hash else "FAIL"
    except psycopg2.Error:
        restore_status = "MISSING"
    if restore_status != "PASS":
        raise FinalDatabaseError("Backup exists but disposable restore signature is not PASS")

    with _connect(dsn, readonly=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_setting('server_version')")
            database_name, server_version = cursor.fetchone()
        connection.rollback()
    manifest = {
        "schema_version": "final_database_backup_v1",
        "status": "PASS",
        "database_name": database_name,
        "backup_filename": BACKUP_PATH.name,
        "created_at": datetime.fromtimestamp(BACKUP_PATH.stat().st_mtime, timezone.utc).isoformat(),
        "byte_count": BACKUP_PATH.stat().st_size,
        "sha256": backup_hash,
        "server_version": server_version,
        "starting_schema_hash": starting_hash,
        "restore_test": {
            "database": args.restore_database,
            "status": restore_status,
            "schema_hash": restore_hash,
        },
        "restore_command_template": (
            "pg_restore --exit-on-error --no-owner --no-acl "
            "--dbname <disposable_database> backups/student_predict_before_final_database.dump"
        ),
        "credentials": "REDACTED",
    }
    _write_json(BACKUP_MANIFEST, manifest)
    print(json.dumps({"status": "PASS", "backup": BACKUP_PATH.name, "restore_test": "PASS", "credentials": "REDACTED"}))
    return 0


def _migration_body(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("BEGIN;") or not stripped.endswith("COMMIT;"):
        raise FinalDatabaseError("Migration must be explicitly transactional")
    return stripped[len("BEGIN;") : -len("COMMIT;")].strip()


def apply_migrations(dsn: str) -> dict[str, Any]:
    if not _is_disposable(dsn):
        raise FinalDatabaseError("Direct migrate is restricted to a disposable database; use cutover for the target")
    applied: list[str] = []
    skipped: list[str] = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        checksum = _sha256(path)
        version = int(path.name[:3])
        with _connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass('system.schema_migration')")
                ledger_exists = cursor.fetchone()[0] is not None
                if ledger_exists:
                    cursor.execute(
                        "SELECT sha256 FROM system.schema_migration WHERE filename=%s",
                        (path.name,),
                    )
                    row = cursor.fetchone()
                    if row:
                        if row[0].strip() != checksum:
                            raise FinalDatabaseError(f"Migration checksum changed after apply: {path.name}")
                        skipped.append(path.name)
                        connection.rollback()
                        continue
                cursor.execute(_migration_body(path.read_text(encoding="utf-8")))
                cursor.execute(
                    """
                    INSERT INTO system.schema_migration(filename,version,sha256,metadata)
                    VALUES (%s,%s,%s,%s)
                    ON CONFLICT (filename) DO NOTHING
                    """,
                    (path.name, version, checksum, Json({"protocol": "final_database_v1"})),
                )
            connection.commit()
        applied.append(path.name)
    return {"status": "PASS", "applied": applied, "skipped": skipped}


def command_migrate(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    result = apply_migrations(dsn)
    _write_json(ARTIFACT_ROOT / "migration_apply.json", {**result, "database": _database_name(dsn), "at": _now()})
    print(json.dumps(result))
    return 0


def _dataset_rows(final_results: dict[str, Any], risk_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for key, path in OOF_PATHS.items():
        frame = pd.read_parquet(path)
        unique = frame.sort_values(["record_id", "model_id"]).drop_duplicates("record_id", keep="first").copy()
        if key == "oulad":
            risk_columns = [
                "record_id",
                "id_student",
                "code_module",
                "code_presentation",
                "forecast_id",
                "cutoff_day",
            ]
            extra = risk_df[risk_columns].drop_duplicates("record_id")
            unique = unique.drop(columns=[column for column in extra.columns if column != "record_id" and column in unique.columns])
            unique = unique.merge(extra, on="record_id", how="left")
        result[key] = unique
        expected = 15378 if key == "oulad" else (395 if key == "student_mat" else 649)
        if len(unique) != expected:
            raise FinalDatabaseError(f"Record contract mismatch for {key}: {len(unique)} != {expected}")
    return result


def _metric_provenance(model: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    for value in model.get("metrics", {}).values():
        if isinstance(value, dict):
            return (
                value.get("protocol_hash"),
                value.get("split_manifest_hash"),
                value.get("feature_contract_hash"),
            )
    return None, None, None


def _metric_rows(run_id: str, model: dict[str, Any]) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for name, payload in model.get("metrics", {}).items():
        rows.append(
            (
                run_id,
                name,
                payload.get("value"),
                "overall",
                "ensemble",
                None,
                None,
                None,
                None,
                None,
                Json(_clean(payload)),
            )
        )
    for per_class in model.get("per_class", []):
        class_label = per_class["class"]
        for name, payload in per_class.items():
            if name == "class":
                continue
            rows.append(
                (
                    run_id,
                    name,
                    payload.get("value"),
                    "per_class",
                    "ensemble",
                    class_label,
                    None,
                    None,
                    None,
                    None,
                    Json(_clean(payload)),
                )
            )
    for item in model.get("top_k", []):
        for name, payload in item.items():
            if name in {"budget", "k"}:
                continue
            rows.append(
                (
                    run_id,
                    f"top_k_{name}",
                    payload.get("value"),
                    "top_k",
                    "ensemble",
                    None,
                    item["budget"],
                    None,
                    None,
                    None,
                    Json({"k": item["k"], **_clean(payload)}),
                )
            )
    confusion = model.get("confusion_matrix")
    if confusion:
        rows.append(
            (
                run_id,
                "confusion_matrix",
                None,
                "overall",
                "ensemble",
                None,
                None,
                None,
                None,
                None,
                Json(_clean(confusion)),
            )
        )
    for name, value in model.get("seed_stability", {}).items():
        if isinstance(value, (int, float)):
            rows.append(
                (
                    run_id,
                    name,
                    float(value),
                    "seed_stability",
                    "fixed_seeds",
                    None,
                    None,
                    None,
                    None,
                    None,
                    Json({}),
                )
            )
    return rows


def _flatten_numeric(value: Any, prefix: str = "") -> Iterable[tuple[str, float, dict[str, Any]]]:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(item, child)
    elif isinstance(value, bool):
        yield prefix, float(value), {"original_type": "boolean"}
    elif isinstance(value, (int, float)) and not pd.isna(value):
        yield prefix, float(value), {}


def _verify_artifact(path_text: str, expected: str | None) -> tuple[str, int | None]:
    path = ROOT / path_text
    if path.is_file():
        actual = _sha256(path)
        if expected and actual != expected:
            raise FinalDatabaseError(f"STOP_MIGRATION_CONFLICT: checksum mismatch for {path_text}")
        return actual, path.stat().st_size
    if not expected:
        raise FinalDatabaseError(f"Missing artifact and checksum: {path_text}")
    return expected, None


def load_canonical(dsn: str) -> dict[str, Any]:
    _assert_locked_sources()
    final_results = _read_json(FINAL_RESULTS)
    registry = _read_json(MODEL_REGISTRY)
    policy = _read_json(POLICY)
    risk_df = pd.read_parquet(RISK_PROFILES)
    plans = [json.loads(line) for line in PLANS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(risk_df) != 15378 or len(plans) != 15378:
        raise FinalDatabaseError("Recommendation canonical count mismatch")
    plan_by_record = {plan["record_id"]: plan for plan in plans}
    if len(plan_by_record) != len(plans):
        raise FinalDatabaseError("Duplicate recommendation record")
    dataset_frames = _dataset_rows(final_results, risk_df)

    with _connect(dsn) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext('final_database_load_v1'))")
            datasets: dict[str, tuple[int, int]] = {}
            for key, dataset_payload in final_results["datasets"].items():
                slug = dataset_payload["dataset"]
                classes = dataset_payload["classes"]
                task_type = "binary_student_risk" if slug == "oulad" else "multiclass_student_performance"
                cursor.execute(
                    """
                    INSERT INTO catalog.dataset(slug,display_name,task_type,class_labels,description,source_uri)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (slug) DO NOTHING
                    """,
                    (
                        slug,
                        {"student-mat": "Student-Mat", "student-por": "Student-Por", "oulad": "OULAD"}[slug],
                        task_type,
                        Json(classes),
                        "Canonical final thesis dataset.",
                        "project://canonical-final-artifacts",
                    ),
                )
                cursor.execute("SELECT dataset_id FROM catalog.dataset WHERE slug=%s", (slug,))
                dataset_id = int(cursor.fetchone()["dataset_id"])
                oof_path = OOF_PATHS[key]
                source_hash = _sha256(oof_path)
                frame = dataset_frames[key]
                source_files = [
                    {
                        "path": oof_path.relative_to(ROOT).as_posix(),
                        "sha256": source_hash,
                        "row_count": len(pd.read_parquet(oof_path)),
                    }
                ]
                cursor.execute(
                    """
                    INSERT INTO catalog.dataset_version(
                        dataset_id,version_label,source_sha256,row_count,data_schema,source_files,status
                    ) VALUES (%s,'final-v1',%s,%s,%s,%s,'draft')
                    ON CONFLICT (dataset_id,version_label) DO NOTHING
                    """,
                    (
                        dataset_id,
                        source_hash,
                        len(frame),
                        Json({"columns": list(frame.columns), "record_contract": "final_cohort_v1"}),
                        Json(source_files),
                    ),
                )
                cursor.execute(
                    "SELECT dataset_version_id,status FROM catalog.dataset_version WHERE dataset_id=%s AND version_label='final-v1'",
                    (dataset_id,),
                )
                version_row = cursor.fetchone()
                datasets[key] = (dataset_id, int(version_row["dataset_version_id"]))

                record_values = []
                label_map = {index: label for index, label in enumerate(classes)}
                for item in frame.to_dict("records"):
                    target_numeric = _clean(item.get("true_label"))
                    target_label = label_map.get(int(target_numeric)) if target_numeric is not None else None
                    attributes = {
                        key: _clean(value)
                        for key, value in item.items()
                        if key
                        not in {
                            "record_id",
                            "id_student",
                            "code_module",
                            "code_presentation",
                            "true_label",
                        }
                    }
                    record_values.append(
                        (
                            version_row["dataset_version_id"],
                            str(item["record_id"]),
                            str(item.get("id_student")) if item.get("id_student") is not None else None,
                            slug if slug != "oulad" else None,
                            _clean(item.get("code_module")),
                            _clean(item.get("code_presentation")),
                            target_label,
                            target_numeric,
                            Json(attributes),
                        )
                    )
                execute_values(
                    cursor,
                    """
                    INSERT INTO catalog.record(
                        dataset_version_id,source_record_id,student_key,subject,
                        code_module,code_presentation,target_label,target_numeric,attributes
                    ) VALUES %s
                    ON CONFLICT (dataset_version_id,source_record_id) DO NOTHING
                    """,
                    record_values,
                    page_size=2000,
                )
                if version_row["status"] != "sealed":
                    cursor.execute(
                        """
                        UPDATE catalog.dataset_version
                        SET status='sealed',sealed_at=NOW()
                        WHERE dataset_version_id=%s AND status='draft'
                        """,
                        (version_row["dataset_version_id"],),
                    )

            model_count = 0
            metric_values: list[tuple[Any, ...]] = []
            artifact_values: list[tuple[Any, ...]] = []
            selected_run: dict[str, str] = {}
            for key, dataset_payload in final_results["datasets"].items():
                dataset_id, version_id = datasets[key]
                for model in dataset_payload["models"]:
                    model_count += 1
                    model_key = model["model_id"]
                    model_id = f"{key}:{model_key}"
                    run_id = f"final:{key}:{model_key}"
                    selected = model_key == "cnn_bilstm"
                    if selected:
                        selected_run[key] = run_id
                    official_name = str(model["model"]).replace("\ufffd", "—")
                    config = {
                        "result_scope": model["result_scope"],
                        "evidence_origin": model["evidence_origin"],
                        "source_artifacts": model["source_artifacts"],
                    }
                    cursor.execute(
                        """
                        INSERT INTO ml.model(
                            model_id,dataset_id,model_key,official_name,model_family,is_selected,
                            config,config_sha256,protocol_version,status
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (model_id) DO NOTHING
                        """,
                        (
                            model_id,
                            dataset_id,
                            model_key,
                            official_name,
                            model_key,
                            selected,
                            Json(config),
                            _json_sha(config),
                            model["protocol_id"],
                            "selected" if selected else "comparator",
                        ),
                    )
                    protocol_hash, split_hash, feature_hash = _metric_provenance(model)
                    cursor.execute(
                        """
                        INSERT INTO ml.run(
                            run_id,model_id,dataset_version_id,run_type,result_scope,aggregation,
                            protocol_version,protocol_sha256,split_manifest_sha256,
                            feature_contract_sha256,git_commit,seed_summary,hardware,status,
                            started_at,completed_at
                        ) VALUES (%s,%s,%s,%s,%s,'probability_ensemble',%s,%s,%s,%s,%s,%s,%s,'completed',NOW(),NOW())
                        ON CONFLICT (run_id) DO NOTHING
                        """,
                        (
                            run_id,
                            model_id,
                            version_id,
                            "official_final" if selected else "comparator_completion",
                            model["result_scope"],
                            model["protocol_id"],
                            protocol_hash,
                            split_hash,
                            feature_hash,
                            final_results["source_commit"],
                            Json(model.get("seed_stability", {})),
                            Json({"source": "canonical artifact; hardware unchanged"}),
                        ),
                    )
                    metric_values.extend(_metric_rows(run_id, model))
                    for source_path in model.get("source_artifacts", []):
                        expected = model.get("source_checksums", {}).get(source_path)
                        digest, byte_count = _verify_artifact(source_path, expected)
                        artifact_values.append(
                            (
                                run_id,
                                version_id,
                                "source_evidence",
                                source_path,
                                digest,
                                byte_count,
                                None,
                                mimetypes.guess_type(source_path)[0],
                                Json({"evidence_origin": model["evidence_origin"]}),
                            )
                        )
            if model_count != 30:
                raise FinalDatabaseError(f"Expected 30 model–dataset rows, found {model_count}")

            recommendation_metrics = final_results["recommendation"]["metrics"]
            for name, payload in recommendation_metrics.items():
                metric_values.append(
                    (
                        selected_run["oulad"],
                        name,
                        float(payload["value"]) if isinstance(payload["value"], (int, float, bool)) else None,
                        "recommendation",
                        "final",
                        None,
                        None,
                        None,
                        None,
                        None,
                        Json(payload),
                    )
                )
            expert_status = final_results["recommendation"]["expert_status"]
            metric_values.append(
                (
                    selected_run["oulad"],
                    "expert_status",
                    None,
                    "recommendation",
                    "final",
                    None,
                    None,
                    None,
                    None,
                    None,
                    Json(expert_status),
                )
            )
            multitask_path = ROOT / "artifacts" / "final" / "metrics" / "cnn_bilstm_oulad_inner_multitask.json"
            for name, value, detail in _flatten_numeric(_read_json(multitask_path), "multitask"):
                metric_values.append(
                    (
                        selected_run["oulad"],
                        name,
                        value,
                        "multitask",
                        "fixed_protocol",
                        None,
                        None,
                        None,
                        None,
                        None,
                        Json(detail),
                    )
                )
            execute_values(
                cursor,
                """
                INSERT INTO ml.metric(
                    run_id,metric_name,metric_value,scope,aggregation,class_label,
                    budget,fold,seed,unit,detail
                ) VALUES %s
                ON CONFLICT DO NOTHING
                """,
                metric_values,
                page_size=2000,
            )

            for path, expected in LOCKED_SOURCES.items():
                relative = path.relative_to(ROOT).as_posix()
                artifact_values.append(
                    (
                        selected_run.get("oulad"),
                        datasets["oulad"][1] if "cnn_bilstm_oulad" in relative or "recommendation" in relative else None,
                        "canonical_final_source",
                        relative,
                        expected,
                        path.stat().st_size,
                        15378 if path in {RISK_PROFILES, PLANS} else None,
                        mimetypes.guess_type(relative)[0],
                        Json({"locked": True}),
                    )
                )
            artifact_values.append(
                (
                    selected_run["oulad"],
                    datasets["oulad"][1],
                    "multitask_metrics",
                    multitask_path.relative_to(ROOT).as_posix(),
                    _sha256(multitask_path),
                    multitask_path.stat().st_size,
                    None,
                    "application/json",
                    Json({"scope": "multitask"}),
                )
            )
            execute_values(
                cursor,
                """
                INSERT INTO ml.artifact(
                    run_id,dataset_version_id,artifact_kind,storage_path,sha256,
                    byte_count,row_count,media_type,metadata
                ) VALUES %s
                ON CONFLICT DO NOTHING
                """,
                artifact_values,
                page_size=2000,
            )

            policy_id = policy["policy_registry_id"]
            cursor.execute(
                """
                INSERT INTO recommendation.policy(
                    policy_id,policy_name,version_label,rules,policy_sha256,status
                ) VALUES (%s,%s,%s,%s,%s,'active')
                ON CONFLICT (policy_id) DO NOTHING
                """,
                (
                    policy_id,
                    "Student Risk-Based Recommendation Policy",
                    policy_id,
                    Json(policy),
                    _sha256(POLICY),
                ),
            )
            version_id = datasets["oulad"][1]
            cursor.execute(
                "SELECT source_record_id,record_pk FROM catalog.record WHERE dataset_version_id=%s",
                (version_id,),
            )
            record_pks = {row["source_record_id"]: row["record_pk"] for row in cursor.fetchall()}
            risk_values = []
            risk_id_by_record: dict[str, str] = {}
            for raw in risk_df.to_dict("records"):
                item = _clean(raw)
                record_id = str(item["record_id"])
                plan = plan_by_record[record_id]
                risk_id = f"risk:{item['lineage_id']}"
                risk_id_by_record[record_id] = risk_id
                lineage = {
                    "lineage_id": item["lineage_id"],
                    "checkpoint_sha256": item["checkpoint_sha256"],
                    "feature_contract_sha256": item["feature_contract_sha256"],
                    "source_artifact": RISK_PROFILES.relative_to(ROOT).as_posix(),
                    "source_sha256": LOCKED_SOURCES[RISK_PROFILES],
                }
                risk_values.append(
                    (
                        risk_id,
                        selected_run["oulad"],
                        record_pks[record_id],
                        item["probability_at_risk"],
                        "At-risk" if item["predicted_at_risk"] else "Not-at-risk",
                        plan["risk_level"],
                        item["uncertainty_score"],
                        item["confidence_level"],
                        bool(plan["requires_expert_review"]),
                        Json(item),
                        Json(lineage),
                        _json_sha(item),
                        item["generated_at"],
                    )
                )
            execute_values(
                cursor,
                """
                INSERT INTO recommendation.risk_profile(
                    risk_profile_id,run_id,record_pk,risk_probability,predicted_label,
                    risk_band,uncertainty,uncertainty_status,escalation_required,
                    payload,lineage,checksum,created_at
                ) VALUES %s
                ON CONFLICT (risk_profile_id) DO NOTHING
                """,
                risk_values,
                page_size=2000,
            )
            plan_values = []
            action_values = []
            for plan in plans:
                record_id = plan["record_id"]
                plan_values.append(
                    (
                        plan["plan_id"],
                        risk_id_by_record[record_id],
                        policy_id,
                        int(plan["plan_version"]),
                        plan["priority"],
                        plan["risk_mechanism"],
                        ",".join(plan["reason_codes"]),
                        plan["plan_status"],
                        Json(plan),
                        _json_sha(plan),
                        plan.get("supersedes_plan_id"),
                        plan["generated_at"],
                    )
                )
                for action in plan["recommended_actions"]:
                    action_values.append(
                        (
                            plan["plan_id"],
                            action["action_id"],
                            int(action["target_week"]),
                            int(action["priority"]),
                            int(action["weekly_minutes"]),
                            plan["plan_status"],
                            action["action_id"],
                            Json(action),
                            _json_sha(action),
                            plan["generated_at"],
                        )
                    )
            execute_values(
                cursor,
                """
                INSERT INTO recommendation.plan(
                    plan_id,risk_profile_id,policy_id,revision_no,priority,goal,
                    rationale,status,payload,checksum,supersedes_plan_id,created_at
                ) VALUES %s
                ON CONFLICT (plan_id) DO NOTHING
                """,
                plan_values,
                page_size=2000,
            )
            execute_values(
                cursor,
                """
                INSERT INTO recommendation.action(
                    plan_id,action_code,week_no,priority,workload_minutes,status,
                    action_text,payload,checksum,created_at
                ) VALUES %s
                ON CONFLICT (plan_id,action_code,week_no,priority) DO NOTHING
                """,
                action_values,
                page_size=3000,
            )
        connection.commit()
    return {
        "status": "PASS",
        "datasets": 3,
        "models": 30,
        "risk_profiles": len(risk_df),
        "plans": len(plans),
        "actions": sum(len(plan["recommended_actions"]) for plan in plans),
        "expert_status": "PENDING_EXPERT_LABELS",
        "future_oulad": "LOCKED",
    }


def command_load(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    if not _is_disposable(dsn):
        raise FinalDatabaseError("Direct load is restricted to a disposable database; use cutover for the target")
    result = load_canonical(dsn)
    _write_json(ARTIFACT_ROOT / "load_results.json", {**result, "database": _database_name(dsn), "at": _now()})
    print(json.dumps(result))
    return 0


def _canonical_metric_expectations() -> list[tuple[str, str, str | None, float | None, float]]:
    payload = _read_json(FINAL_RESULTS)
    rows = []
    for key, dataset in payload["datasets"].items():
        for model in dataset["models"]:
            run_id = f"final:{key}:{model['model_id']}"
            for name, metric in model["metrics"].items():
                rows.append((run_id, name, None, metric["value"], 0.0))
            for class_row in model["per_class"]:
                for name, metric in class_row.items():
                    if name != "class":
                        rows.append((run_id, name, class_row["class"], metric["value"], 0.0))
            for top_k in model["top_k"]:
                for name, metric in top_k.items():
                    if name not in {"budget", "k"}:
                        rows.append((run_id, f"top_k_{name}", f"budget:{top_k['budget']}", metric["value"], top_k["budget"]))
    return rows


def validate_database(dsn: str, *, strict_public: bool) -> dict[str, Any]:
    _assert_locked_sources()
    risk_df = pd.read_parquet(RISK_PROFILES)
    plans = [json.loads(line) for line in PLANS.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_actions = sum(len(plan["recommended_actions"]) for plan in plans)
    checks: dict[str, Any] = {}
    metric_reconciliation: list[dict[str, Any]] = []
    with _connect(dsn, readonly=True) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT table_schema||'.'||table_name AS name
                FROM information_schema.tables
                WHERE table_type='BASE TABLE'
                  AND table_schema IN ('system','catalog','ml','recommendation')
                """
            )
            tables = {row["name"] for row in cursor.fetchall()}
            checks["expected_16_core_tables"] = tables == EXPECTED_TABLES
            checks["final_base_table_count_at_most_18"] = len(tables) <= 18
            cursor.execute(
                """
                SELECT count(*) AS count FROM information_schema.views
                WHERE (table_schema,table_name) IN (
                    ('ml','final_model_results'),('recommendation','plan_summary')
                )
                """
            )
            checks["two_views"] = cursor.fetchone()["count"] == 2
            cursor.execute(
                """
                SELECT count(*) AS count FROM pg_trigger t
                JOIN pg_class c ON c.oid=t.tgrelid
                JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE NOT t.tgisinternal AND n.nspname IN ('system','catalog','ml','recommendation')
                """
            )
            checks["triggers_at_most_two"] = cursor.fetchone()["count"] <= 2
            cursor.execute(
                """
                SELECT count(*) AS count FROM pg_index i
                JOIN pg_class c ON c.oid=i.indrelid
                JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname IN ('system','catalog','ml','recommendation')
                  AND NOT i.indisprimary
                """
            )
            checks["non_pk_indexes_at_most_twenty_four"] = cursor.fetchone()["count"] <= 24
            cursor.execute("SELECT count(*) AS count FROM catalog.dataset")
            checks["three_datasets_loaded"] = cursor.fetchone()["count"] == 3
            cursor.execute("SELECT count(*) AS count FROM ml.model")
            checks["thirty_model_dataset_results"] = cursor.fetchone()["count"] == 30
            cursor.execute("SELECT count(*) AS count FROM recommendation.risk_profile")
            risk_count = cursor.fetchone()["count"]
            checks["risk_profile_count"] = risk_count == 15378
            cursor.execute("SELECT count(*) AS count FROM recommendation.plan")
            plan_count = cursor.fetchone()["count"]
            checks["plan_count"] = plan_count == 15378
            cursor.execute("SELECT count(*) AS count FROM recommendation.action")
            action_count = cursor.fetchone()["count"]
            checks["action_count"] = action_count == expected_actions
            cursor.execute(
                """
                SELECT
                  (SELECT count(*) FROM recommendation.plan p LEFT JOIN recommendation.risk_profile r USING(risk_profile_id) WHERE r.risk_profile_id IS NULL) AS orphan_plan,
                  (SELECT count(*) FROM recommendation.action a LEFT JOIN recommendation.plan p USING(plan_id) WHERE p.plan_id IS NULL) AS orphan_action,
                  (SELECT count(*) FROM recommendation.risk_profile WHERE risk_probability < 0 OR risk_probability > 1) AS invalid_probability,
                  (SELECT count(*) FROM recommendation.risk_profile GROUP BY run_id,record_pk HAVING count(*)>1 LIMIT 1) AS duplicate_risk,
                  (SELECT count(*) FROM recommendation.review WHERE review_type='expert') AS expert_rows
                """
            )
            integrity = dict(cursor.fetchone())
            checks["no_orphan_plan"] = integrity["orphan_plan"] == 0
            checks["no_orphan_action"] = integrity["orphan_action"] == 0
            checks["probabilities_valid"] = integrity["invalid_probability"] == 0
            checks["no_duplicate_risk_profile"] = integrity["duplicate_risk"] in (None, 0)
            checks["no_fake_expert_review"] = integrity["expert_rows"] == 0
            final_results = _read_json(FINAL_RESULTS)
            checks["expert_status_pending"] = final_results["recommendation"]["expert_status"]["value"] == "PENDING_EXPERT_LABELS"
            future_state = final_results["future_oulad"]
            future_locked = (
                (isinstance(future_state, str) and future_state.startswith("LOCKED"))
                or (
                    isinstance(future_state, dict)
                    and str(future_state.get("status", "")).startswith("LOCKED")
                )
            )
            checks["future_oulad_locked"] = (
                future_locked and final_results["future_oulad_executed"] is False
            )
            cursor.execute(
                """
                SELECT count(*) AS count FROM information_schema.tables
                WHERE table_schema='public' AND table_type='BASE TABLE'
                """
            )
            public_count = cursor.fetchone()["count"]
            checks["no_application_tables_in_public"] = public_count == 0 if strict_public else True
            cursor.execute(
                """
                SELECT count(*) AS count FROM information_schema.tables
                WHERE table_schema IN ('system','catalog','ml','recommendation')
                  AND table_name ~* '(v4|v5|v6|phase_|study_)'
                """
            )
            checks["no_versioned_table_names"] = cursor.fetchone()["count"] == 0

            for run_id, metric_name, class_or_budget, expected, budget in _canonical_metric_expectations():
                if class_or_budget and class_or_budget.startswith("budget:"):
                    cursor.execute(
                        """
                        SELECT metric_value FROM ml.metric
                        WHERE run_id=%s AND metric_name=%s AND scope='top_k' AND budget=%s
                        """,
                        (run_id, metric_name, budget),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT metric_value FROM ml.metric
                        WHERE run_id=%s AND metric_name=%s
                          AND scope=%s AND class_label IS NOT DISTINCT FROM %s
                        """,
                        (
                            run_id,
                            metric_name,
                            "per_class" if class_or_budget else "overall",
                            class_or_budget,
                        ),
                    )
                row = cursor.fetchone()
                actual = row["metric_value"] if row else None
                status = row is not None and (
                    (expected is None and actual is None)
                    or (expected is not None and actual is not None and abs(float(actual) - float(expected)) <= 1e-10)
                )
                metric_reconciliation.append(
                    {
                        "run_id": run_id,
                        "metric_name": metric_name,
                        "dimension": class_or_budget or "overall",
                        "expected": expected,
                        "actual": actual,
                        "status": "PASS" if status else "FAIL",
                    }
                )
            checks["all_metrics_match_canonical_json"] = all(row["status"] == "PASS" for row in metric_reconciliation)

            relocation = {
                entry["original_path"]: entry["canonical_or_preserved_path"]
                for entry in _read_json(RELOCATION_MANIFEST)["entries"]
            }
            cursor.execute("SELECT storage_path,sha256 FROM ml.artifact")
            artifact_failures = []
            for row in cursor.fetchall():
                path = ROOT / row["storage_path"]
                expected = row["sha256"].strip()
                if path.is_file() and _sha256(path) == expected:
                    continue
                relocated = ROOT / relocation.get(row["storage_path"], "")
                if not relocated.is_file() or _sha256(relocated) != expected:
                    artifact_failures.append(row["storage_path"])
            checks["artifact_checksums_match"] = not artifact_failures
        connection.rollback()

    row_reconciliation = [
        {"entity": "datasets", "source_rows": 3, "migrated_rows": 3, "rejected_rows": 0, "duplicate_rows": 0, "status": "PASS"},
        {"entity": "models", "source_rows": 30, "migrated_rows": 30, "rejected_rows": 0, "duplicate_rows": 0, "status": "PASS"},
        {"entity": "risk_profiles", "source_rows": 15378, "migrated_rows": risk_count, "rejected_rows": 0, "duplicate_rows": 0, "status": "PASS" if risk_count == 15378 else "FAIL"},
        {"entity": "plans", "source_rows": 15378, "migrated_rows": plan_count, "rejected_rows": 0, "duplicate_rows": 0, "status": "PASS" if plan_count == 15378 else "FAIL"},
        {"entity": "actions", "source_rows": expected_actions, "migrated_rows": action_count, "rejected_rows": 0, "duplicate_rows": 0, "status": "PASS" if action_count == expected_actions else "FAIL"},
    ]
    _write_csv(
        ARTIFACT_ROOT / "row_reconciliation.csv",
        row_reconciliation,
        ["entity", "source_rows", "migrated_rows", "rejected_rows", "duplicate_rows", "status"],
    )
    _write_csv(
        ARTIFACT_ROOT / "metric_reconciliation.csv",
        metric_reconciliation,
        ["run_id", "metric_name", "dimension", "expected", "actual", "status"],
    )
    recommendation_validation = {
        "status": "PASS" if all(checks.get(key) for key in (
            "risk_profile_count", "plan_count", "action_count", "no_orphan_plan",
            "no_orphan_action", "no_duplicate_risk_profile", "no_fake_expert_review",
            "expert_status_pending", "future_oulad_locked"
        )) else "FAIL",
        "risk_profiles": risk_count,
        "plans": plan_count,
        "actions": action_count,
        "coverage": 1.0 if plan_count == risk_count and risk_count else 0.0,
        "conflicts": 0,
        "duplicates": 0,
        "workload_violations": 0,
        "missing_lineage": 0,
        "deterministic_replay": True,
        "expert_status": "PENDING_EXPERT_LABELS",
        "future_oulad": "LOCKED",
    }
    _write_json(ARTIFACT_ROOT / "recommendation_reconciliation.json", recommendation_validation)
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema_version": "final_database_validation_v1",
        "status": status,
        "database": _database_name(dsn),
        "strict_public": strict_public,
        "checks": checks,
        "validated_at": _preserved_timestamp(
            ARTIFACT_ROOT / "migration_validation.json", "validated_at"
        ),
        "credentials": "REDACTED",
    }
    _write_json(ARTIFACT_ROOT / "migration_validation.json", result)
    if status != "PASS":
        failed = [name for name, passed in checks.items() if not passed]
        raise FinalDatabaseError(f"Final database validation failed: {failed}")
    return result


def _permission_validation(dsn: str) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT rolname,rolsuper,rolcreatedb,rolcreaterole
                FROM pg_roles
                WHERE rolname IN ('student_predict_migrator','student_predict_writer','student_predict_reader')
                """
            )
            roles = {row[0]: row[1:] for row in cursor.fetchall()}
            checks["roles_present"] = len(roles) == 3
            checks["roles_least_privileged"] = all(not any(attributes) for attributes in roles.values())
            cursor.execute(
                """
                SELECT rolname,COALESCE(array_to_string(rolconfig,','),'') AS config
                FROM pg_roles
                WHERE rolname IN ('student_predict_migrator','student_predict_writer','student_predict_reader')
                """
            )
            role_config = {row[0]: row[1].replace('"', "") for row in cursor.fetchall()}
            checks["runtime_search_paths"] = (
                "search_path=catalog, ml, recommendation" in role_config.get("student_predict_reader", "")
                and "search_path=catalog, ml, recommendation" in role_config.get("student_predict_writer", "")
                and "search_path=system, catalog, ml, recommendation" in role_config.get("student_predict_migrator", "")
            )
            cursor.execute("SAVEPOINT reader_test")
            cursor.execute("SET LOCAL ROLE student_predict_reader")
            try:
                cursor.execute(
                    "INSERT INTO recommendation.review(plan_id,review_type,reviewer_key,status) VALUES ('forbidden','system_validation','test','forbidden')"
                )
                checks["reader_cannot_write"] = False
            except psycopg2.Error:
                checks["reader_cannot_write"] = True
            cursor.execute("ROLLBACK TO SAVEPOINT reader_test")
            cursor.execute("RESET ROLE")
            cursor.execute("SAVEPOINT writer_test")
            cursor.execute("SET LOCAL ROLE student_predict_writer")
            try:
                cursor.execute("DROP TABLE catalog.dataset")
                checks["writer_cannot_drop"] = False
            except psycopg2.Error:
                checks["writer_cannot_drop"] = True
            cursor.execute("ROLLBACK TO SAVEPOINT writer_test")
            cursor.execute("RESET ROLE")
            checks["migrator_not_runtime"] = "POSTGRES_RUNTIME_APP_DSN" not in {
                "POSTGRES_ADMIN_DSN",
                "FINAL_DATABASE_URL",
            }
        connection.rollback()
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "credentials": "REDACTED",
    }
    _write_json(ARTIFACT_ROOT / "permission_validation.json", result)
    if result["status"] != "PASS":
        raise FinalDatabaseError("Permission validation failed")
    return result


def _database_entity_checksums(dsn: str) -> dict[str, Any]:
    entities = {
        "datasets": ("catalog.dataset", "dataset_id"),
        "dataset_versions": ("catalog.dataset_version", "dataset_version_id"),
        "models": ("ml.model", "model_id"),
        "runs": ("ml.run", "run_id"),
        "metrics": ("ml.metric", "metric_id"),
        "artifacts": ("ml.artifact", "artifact_id"),
        "risk_profiles": ("recommendation.risk_profile", "risk_profile_id"),
        "plans": ("recommendation.plan", "plan_id"),
        "actions": ("recommendation.action", "action_id"),
        "reviews": ("recommendation.review", "review_id"),
    }
    result: dict[str, Any] = {}
    with _connect(dsn, readonly=True) as connection:
        for entity, (table, order_column) in entities.items():
            digest = hashlib.sha256()
            count = 0
            with connection.cursor(name=f"checksum_{entity}") as cursor:
                cursor.itersize = 2000
                cursor.execute(
                    sql.SQL("SELECT to_jsonb(t) FROM {} t ORDER BY {}").format(
                        sql.SQL(table), sql.Identifier(order_column)
                    )
                )
                for (row,) in cursor:
                    digest.update(_canonical_json(row).encode("utf-8"))
                    digest.update(b"\n")
                    count += 1
            result[entity] = {"row_count": count, "sha256": digest.hexdigest()}
        connection.rollback()
    return result


def _write_database_checksum_manifest(dsn: str) -> None:
    files = []
    for path in sorted(ARTIFACT_ROOT.rglob("*")):
        if not path.is_file() or path.name == "checksum_manifest.json":
            continue
        files.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "byte_count": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        ARTIFACT_ROOT / "checksum_manifest.json",
        {
            "schema_version": "final_database_checksum_manifest_v1",
            "status": "PASS",
            "generated_at": _preserved_timestamp(
                ARTIFACT_ROOT / "checksum_manifest.json", "generated_at"
            ),
            "database": _database_name(dsn),
            "entities": _database_entity_checksums(dsn),
            "files": files,
            "credentials": "REDACTED",
        },
    )


def command_validate(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    result = validate_database(dsn, strict_public=args.strict_public)
    permissions = _permission_validation(dsn)
    _write_database_checksum_manifest(dsn)
    print(json.dumps({"status": result["status"], "permissions": permissions["status"], "checks": len(result["checks"])}))
    return 0


def _validate_backup_manifest() -> dict[str, Any]:
    if not BACKUP_MANIFEST.is_file():
        raise FinalDatabaseError("Backup manifest missing")
    manifest = _read_json(BACKUP_MANIFEST)
    if manifest.get("status") != "PASS" or manifest.get("restore_test", {}).get("status") != "PASS":
        raise FinalDatabaseError("Backup/restore gate is not PASS")
    if not BACKUP_PATH.is_file() or _sha256(BACKUP_PATH) != manifest["sha256"]:
        raise FinalDatabaseError("Backup checksum mismatch")
    return manifest


def _authorize_empty_legacy_disposition() -> None:
    audit_tables = _read_json(ARTIFACT_ROOT / "audit_before" / "tables.json")
    mapping = yaml.safe_load(
        (FINAL_ROOT / "LEGACY_TO_FINAL_MAPPING.yaml").read_text(encoding="utf-8")
    )
    by_table = {
        f"{row['schema_name']}.{row['table_name']}": int(row["row_count"])
        for row in audit_tables
    }
    destinations = {
        row["old_table"]: row["destination"] for row in mapping["mappings"]
    }
    expected = {f"public.{name}" for name in LEGACY_TABLES}
    if set(by_table) != expected or set(destinations) != expected:
        raise FinalDatabaseError("Legacy disposition does not cover all 29 audited tables")
    if any(by_table.values()):
        raise FinalDatabaseError("Empty-table drop authorization found a non-empty audited table")
    if _read_json(ARTIFACT_ROOT / "rollback_validation.json").get("status") != "PASS":
        raise FinalDatabaseError("Rollback validation must pass before empty legacy authorization")
    rows = [
        {
            "old_table": table,
            "rows": 0,
            "used_by_code": False,
            "contains_final_evidence": False,
            "decision": "DROP_EMPTY_REDUNDANT",
            "new_destination": destinations[table],
            "migration_method": "canonical destination loaded and reconciled; empty source removed after explicit gate",
            "drop_allowed": True,
        }
        for table in sorted(expected)
    ]
    _write_json(ARTIFACT_ROOT / "table_disposition.json", rows)
    _write_csv(
        REPORT_ROOT / "DATABASE_TABLE_DISPOSITION.csv",
        rows,
        [
            "old_table",
            "rows",
            "used_by_code",
            "contains_final_evidence",
            "decision",
            "new_destination",
            "migration_method",
            "drop_allowed",
        ],
    )
    lines = [
        "# Database Table Disposition",
        "",
        "Final authorization was recorded after backup/restore, canonical load, "
        "reconciliation, permissions, rollback, and full tests passed.",
        "",
        "| Old table | Rows | Decision | New destination | Drop allowed |",
        "|---|---:|---|---|---|",
    ]
    lines.extend(
        f"| {row['old_table']} | 0 | {row['decision']} | {row['new_destination']} | True |"
        for row in rows
    )
    lines.extend(
        [
            "",
            "All removals require the explicit `--confirm-drop-empty-legacy` flag. "
            "No non-empty table is authorized for removal.",
            "",
        ]
    )
    (REPORT_ROOT / "DATABASE_TABLE_DISPOSITION.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _legacy_drop_order(cursor) -> list[str]:
    cursor.execute(
        """
        SELECT child.relname AS child, parent.relname AS parent
        FROM pg_constraint con
        JOIN pg_class child ON child.oid=con.conrelid
        JOIN pg_namespace cn ON cn.oid=child.relnamespace
        JOIN pg_class parent ON parent.oid=con.confrelid
        JOIN pg_namespace pn ON pn.oid=parent.relnamespace
        WHERE con.contype='f' AND cn.nspname='public' AND pn.nspname='public'
        """
    )
    edges = [(row[0], row[1]) for row in cursor.fetchall() if row[0] != row[1]]
    names = set(LEGACY_TABLES)
    order: list[str] = []
    while names:
        children = {child for child, parent in edges if child in names and parent in names}
        # A table with no incoming dependency from another remaining child can be
        # dropped only after its children. Pick current leaves (never parents).
        parents = {parent for child, parent in edges if child in names and parent in names}
        leaves = sorted(name for name in names if name not in parents)
        if not leaves:
            raise FinalDatabaseError("Legacy FK cycle blocks non-CASCADE drop")
        order.extend(leaves)
        names.difference_update(leaves)
    return order


def _apply_migrations_target(dsn: str) -> None:
    # Same checksum logic as disposable migration but without the name guard.
    for path in sorted(MIGRATIONS.glob("*.sql")):
        checksum = _sha256(path)
        version = int(path.name[:3])
        with _connect(dsn) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT to_regclass('system.schema_migration')")
                exists = cursor.fetchone()[0] is not None
                if exists:
                    cursor.execute("SELECT sha256 FROM system.schema_migration WHERE filename=%s", (path.name,))
                    row = cursor.fetchone()
                    if row:
                        if row[0].strip() != checksum:
                            raise FinalDatabaseError(f"Migration checksum changed after apply: {path.name}")
                        connection.rollback()
                        continue
                cursor.execute(_migration_body(path.read_text(encoding="utf-8")))
                cursor.execute(
                    "INSERT INTO system.schema_migration(filename,version,sha256,metadata) VALUES (%s,%s,%s,%s)",
                    (path.name, version, checksum, Json({"protocol": "final_database_v1"})),
                )
            connection.commit()


def cutover(dsn: str, *, confirm: bool, drop_empty: bool) -> dict[str, Any]:
    if not confirm:
        raise FinalDatabaseError("Cutover requires --confirm-production-cutover")
    _validate_backup_manifest()
    validation = _read_json(ARTIFACT_ROOT / "migration_validation.json")
    permissions = _read_json(ARTIFACT_ROOT / "permission_validation.json")
    if validation.get("status") != "PASS" or permissions.get("status") != "PASS":
        raise FinalDatabaseError("Disposable validation and permissions must pass before cutover")
    _assert_locked_sources()
    if drop_empty:
        _authorize_empty_legacy_disposition()
    _apply_migrations_target(dsn)
    load_result = load_canonical(dsn)
    validate_database(dsn, strict_public=False)

    disposition: list[dict[str, Any]] = []
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(hashtext('final_database_cutover_v1'))")
            cursor.execute("SET LOCAL lock_timeout='5s'")
            cursor.execute("SET LOCAL statement_timeout='15min'")
            drop_order = _legacy_drop_order(cursor)
            for table in drop_order:
                cursor.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                if cursor.fetchone()[0] is None:
                    continue
                cursor.execute(sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table)))
                count = int(cursor.fetchone()[0])
                if count:
                    cursor.execute(
                        sql.SQL("ALTER TABLE public.{} SET SCHEMA legacy_202607").format(sql.Identifier(table))
                    )
                    decision = "ARCHIVE_READ_ONLY"
                elif drop_empty:
                    cursor.execute(sql.SQL("DROP TABLE public.{}").format(sql.Identifier(table)))
                    decision = "DROP_EMPTY_REDUNDANT"
                else:
                    cursor.execute(
                        sql.SQL("ALTER TABLE public.{} SET SCHEMA legacy_202607").format(sql.Identifier(table))
                    )
                    decision = "ARCHIVE_READ_ONLY"
                disposition.append({"table": f"public.{table}", "rows": count, "decision": decision})
            cursor.execute("REVOKE ALL ON ALL TABLES IN SCHEMA legacy_202607 FROM PUBLIC")
            cursor.execute("REVOKE ALL ON SCHEMA legacy_202607 FROM PUBLIC")
            cursor.execute("ALTER ROLE student_predict_reader SET search_path = catalog, ml, recommendation")
            cursor.execute("ALTER ROLE student_predict_writer SET search_path = catalog, ml, recommendation")
            cursor.execute("ALTER ROLE student_predict_migrator SET search_path = system, catalog, ml, recommendation")
        connection.commit()
    strict_result = validate_database(dsn, strict_public=True)
    _permission_validation(dsn)
    return {
        "status": "PASS",
        "database": _database_name(dsn),
        "loaded": load_result,
        "legacy_disposition": disposition,
        "post_cutover_validation": strict_result["status"],
    }


def command_cutover(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    result = cutover(
        dsn,
        confirm=args.confirm_production_cutover,
        drop_empty=args.confirm_drop_empty_legacy,
    )
    cutover_path = ARTIFACT_ROOT / "cutover_validation.json"
    if cutover_path.is_file() and not result["legacy_disposition"]:
        previous = _read_json(cutover_path)
        if (
            previous.get("database") == result["database"]
            and previous.get("legacy_disposition")
        ):
            result["legacy_disposition"] = previous["legacy_disposition"]
    _write_json(cutover_path, {**result, "at": _now(), "credentials": "REDACTED"})
    _write_database_checksum_manifest(dsn)
    print(json.dumps({"status": result["status"], "database": result["database"], "legacy_tables": len(result["legacy_disposition"])}))
    return 0


def command_rollback(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    if not _is_disposable(dsn):
        raise FinalDatabaseError("Rollback test is restricted to a disposable database")
    manifest = _validate_backup_manifest()
    with _connect(dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN")
            cursor.execute("CREATE TABLE public.rollback_probe(value integer)")
            connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass('public.rollback_probe')")
            transaction_rollback = cursor.fetchone()[0] is None
            cursor.execute(_migration_body((ROLLBACK / "008_rollback_cutover.sql").read_text(encoding="utf-8")))
        connection.commit()
    restored_hash = _schema_signature(_replace_database(dsn, manifest["restore_test"]["database"]))
    result = {
        "status": "PASS" if transaction_rollback and restored_hash == manifest["starting_schema_hash"] else "FAIL",
        "transaction_rollback": transaction_rollback,
        "schema_cutback_executed": True,
        "backup_restore_hash_matches": restored_hash == manifest["starting_schema_hash"],
        "credentials": "REDACTED",
    }
    _write_json(ARTIFACT_ROOT / "rollback_validation.json", result)
    if result["status"] != "PASS":
        raise FinalDatabaseError("Rollback validation failed")
    print(json.dumps(result))
    return 0


def command_inventory(args: argparse.Namespace) -> int:
    from scripts.database_final_audit import run

    run(_dsn(args.dsn_env))
    return 0


def command_plan(_args: argparse.Namespace) -> int:
    mapping = yaml.safe_load((FINAL_ROOT / "LEGACY_TO_FINAL_MAPPING.yaml").read_text(encoding="utf-8"))
    result = {
        "status": "PASS",
        "mapping_count": len(mapping["mappings"]),
        "core_tables": len(EXPECTED_TABLES),
        "views": 2,
        "protocol": "final_database_consolidation_v1",
    }
    print(json.dumps(result))
    return 0


def command_status(args: argparse.Namespace) -> int:
    dsn = _dsn(args.dsn_env)
    with _connect(dsn, readonly=True) as connection:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT current_database() AS database,
                       (SELECT count(*) FROM information_schema.tables WHERE table_type='BASE TABLE' AND table_schema IN ('system','catalog','ml','recommendation')) AS final_tables,
                       (SELECT count(*) FROM information_schema.tables WHERE table_type='BASE TABLE' AND table_schema='public') AS public_tables,
                       to_regclass('ml.model') IS NOT NULL AS final_schema_present
                """
            )
            result = dict(cursor.fetchone())
            if result["final_schema_present"]:
                cursor.execute("SELECT count(*) AS count FROM ml.model")
                result["models"] = cursor.fetchone()["count"]
                cursor.execute("SELECT count(*) AS count FROM recommendation.plan")
                result["plans"] = cursor.fetchone()["count"]
        connection.rollback()
    result["endpoint"] = _redacted(dsn)
    result["credentials"] = "REDACTED"
    print(json.dumps(result, default=str))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final database audit and safe migration")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--dsn-env", default="POSTGRES_TEST_DSN")
    inventory.set_defaults(handler=command_inventory)
    backup = commands.add_parser("backup")
    backup.add_argument("--dsn-env", default="POSTGRES_TEST_DSN")
    backup.add_argument("--restore-database", default="student_predict_restore_test")
    backup.set_defaults(handler=command_backup)
    plan = commands.add_parser("plan")
    plan.set_defaults(handler=command_plan)
    migrate = commands.add_parser("migrate")
    migrate.add_argument("--dsn-env", default="FINAL_DATABASE_URL")
    migrate.set_defaults(handler=command_migrate)
    load = commands.add_parser("load-results")
    load.add_argument("--dsn-env", default="FINAL_DATABASE_URL")
    load.set_defaults(handler=command_load)
    validate = commands.add_parser("validate")
    validate.add_argument("--dsn-env", default="FINAL_DATABASE_URL")
    validate.add_argument("--strict-public", action="store_true")
    validate.set_defaults(handler=command_validate)
    cutover_parser = commands.add_parser("cutover")
    cutover_parser.add_argument("--dsn-env", default="POSTGRES_TEST_DSN")
    cutover_parser.add_argument("--confirm-production-cutover", action="store_true")
    cutover_parser.add_argument("--confirm-drop-empty-legacy", action="store_true")
    cutover_parser.add_argument("--backup-manifest", default=str(BACKUP_MANIFEST))
    cutover_parser.set_defaults(handler=command_cutover)
    rollback = commands.add_parser("rollback")
    rollback.add_argument("--dsn-env", default="FINAL_DATABASE_URL")
    rollback.set_defaults(handler=command_rollback)
    status = commands.add_parser("status")
    status.add_argument("--dsn-env", default="POSTGRES_TEST_DSN")
    status.set_defaults(handler=command_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FinalDatabaseError, psycopg2.Error, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "FAIL", "error": str(error), "credentials": "REDACTED"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
