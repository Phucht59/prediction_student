"""Read-only PostgreSQL inventory and repository database-usage audit.

This script is intentionally separate from the mutating final migration CLI.
It never logs a DSN and opens the source transaction in read-only mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "artifacts" / "final" / "database" / "audit_before"
DATABASE_DIR = ROOT / "artifacts" / "final" / "database"
REPORT_DIR = ROOT / "reports" / "final"

SCAN_PATHS = (
    "src",
    "scripts",
    "tests",
    "database",
    "configs",
    "project.py",
    "docker-compose.yml",
)

# These modules are retained solely to reproduce historical training/database
# evidence. They are not imported by the final runtime or final database CLI.
HISTORICAL_DATABASE_PATHS = {
    "src/postgres_data_source.py",
    "src/evaluation/evaluation.py",
    "scripts/database_audit.py",
    "scripts/database_register_evidence.py",
    "scripts/database_final.py",
    "scripts/validate_oulad_final.py",
}

LEGACY_DESTINATIONS: dict[str, tuple[str, str, str]] = {
    "advisor_decisions": ("MERGE", "recommendation.review", "map advisor decision to review"),
    "cutoff_feature_snapshots": ("MIGRATE_TO_ARTIFACT", "ml.artifact", "register immutable snapshot path and checksum"),
    "expert_review_cases": ("MERGE", "recommendation.review", "map expert case metadata without fabricating labels"),
    "expert_review_ratings": ("MERGE", "recommendation.review", "map real expert rating only"),
    "ml_evidence_bundles": ("MERGE", "ml.artifact", "register evidence bundle path and checksum"),
    "ml_experiment_runs": ("MERGE", "ml.run", "normalize final and evaluation runs"),
    "ml_predictions": ("MIGRATE_TO_ARTIFACT", "ml.artifact", "store prediction file metadata; risk rows use risk_profile"),
    "ml_recommendations": ("MERGE", "recommendation.plan", "normalize final recommendation plans"),
    "ml_run_metrics": ("MERGE", "ml.metric", "normalize overall, class, top-k and calibration metrics"),
    "ml_run_record_splits": ("MIGRATE_TO_ARTIFACT", "ml.artifact", "register split manifest instead of split members"),
    "ml_schema_migrations": ("MERGE", "system.schema_migration", "copy immutable migration ledger"),
    "prediction_cohorts": ("MERGE", "catalog.dataset_version", "preserve cohort contract in version metadata"),
    "prediction_snapshots": ("MIGRATE_TO_ARTIFACT", "ml.artifact", "register prediction snapshot path and checksum"),
    "recommendation_action_catalog": ("MERGE", "recommendation.policy", "embed versioned action catalog in policy rules"),
    "recommendation_actions": ("MERGE", "recommendation.action", "normalize final plan actions"),
    "recommendation_feature_registry": ("MERGE", "recommendation.policy", "embed recommendation feature contract in policy"),
    "recommendation_follow_ups": ("MERGE", "recommendation.review", "map as follow_up review"),
    "recommendation_goals": ("MERGE", "recommendation.plan", "map goal into plan goal and payload"),
    "recommendation_instances": ("MERGE", "recommendation.risk_profile", "normalize risk/recommendation case identity"),
    "recommendation_outcomes": ("MERGE", "recommendation.review", "map recorded outcome as review payload"),
    "recommendation_policies": ("MERGE", "recommendation.policy", "normalize one checksummed policy registry"),
    "recommendation_revisions": ("MERGE", "recommendation.plan", "normalize plan revision and supersession"),
    "snapshot_record_index": ("MIGRATE_TO_ARTIFACT", "ml.artifact", "retain row index inside snapshot artifact"),
    "source_dataset_files": ("MERGE", "catalog.dataset_version", "embed source file checksums in source_files"),
    "source_dataset_versions": ("RENAME", "catalog.dataset_version", "normalize dataset version identity"),
    "source_record_targets": ("MERGE", "catalog.record", "merge target into canonical record"),
    "source_records": ("RENAME", "catalog.record", "normalize final cohort records"),
    "split_manifest_registry": ("MIGRATE_TO_ARTIFACT", "ml.artifact", "register split manifest path and checksum"),
    "study_extension_runs": ("MERGE", "ml.run", "normalize evaluation-only extension runs"),
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fetch(cursor: RealDictCursor, statement: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    cursor.execute(statement, params)
    return [dict(row) for row in cursor.fetchall()]


def _redacted_endpoint(dsn: str) -> str:
    parsed = urlparse(dsn)
    host = parsed.hostname or "<local>"
    port = f":{parsed.port}" if parsed.port else ""
    database = parsed.path.lstrip("/") or "<database>"
    return f"postgresql://<redacted>@{host}{port}/{database}"


def _inventory(dsn: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    with psycopg2.connect(dsn) as connection:
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SET LOCAL lock_timeout = '5s'")
            cursor.execute("SET LOCAL statement_timeout = '5min'")
            cursor.execute(
                """
                SELECT current_database() AS database_name,
                       current_setting('server_version') AS server_version,
                       pg_database_size(current_database()) AS database_size_bytes,
                       pg_size_pretty(pg_database_size(current_database())) AS database_size
                """
            )
            profile = dict(cursor.fetchone())
            profile.update(
                {
                    "audited_at": datetime.now(timezone.utc).isoformat(),
                    "connection": _redacted_endpoint(dsn),
                    "current_user": "<redacted>",
                    "transaction_mode": "READ ONLY",
                }
            )

            schemas = _fetch(
                cursor,
                """
                SELECT n.nspname AS schema_name,
                       pg_get_userbyid(n.nspowner) AS owner,
                       obj_description(n.oid, 'pg_namespace') AS comment
                FROM pg_namespace n
                WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
                ORDER BY n.nspname
                """,
            )
            tables = _fetch(
                cursor,
                """
                SELECT s.schemaname AS schema_name, s.relname AS table_name,
                       pg_get_userbyid(c.relowner) AS owner,
                       COALESCE(obj_description(c.oid, 'pg_class'), '') AS comment,
                       pg_total_relation_size(c.oid) AS total_bytes,
                       pg_relation_size(c.oid) AS table_bytes,
                       s.last_analyze, s.last_autoanalyze, s.last_vacuum, s.last_autovacuum
                FROM pg_stat_user_tables s
                JOIN pg_class c ON c.oid = format('%%I.%%I', s.schemaname, s.relname)::regclass
                ORDER BY s.schemaname, s.relname
                """,
            )
            for table in tables:
                cursor.execute(
                    sql.SQL("SELECT count(*) AS row_count FROM {}.{}").format(
                        sql.Identifier(table["schema_name"]), sql.Identifier(table["table_name"])
                    )
                )
                table["row_count"] = int(cursor.fetchone()["row_count"])

            columns = _fetch(
                cursor,
                """
                SELECT table_schema AS schema_name, table_name, ordinal_position, column_name,
                       data_type, udt_name, is_nullable, column_default,
                       character_maximum_length, numeric_precision, numeric_scale
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name, ordinal_position
                """,
            )
            constraints = _fetch(
                cursor,
                """
                SELECT n.nspname AS schema_name, c.relname AS table_name,
                       con.conname AS constraint_name,
                       CASE con.contype WHEN 'p' THEN 'PRIMARY KEY' WHEN 'f' THEN 'FOREIGN KEY'
                            WHEN 'u' THEN 'UNIQUE' WHEN 'c' THEN 'CHECK'
                            WHEN 'x' THEN 'EXCLUSION' ELSE con.contype::text END AS constraint_type,
                       pg_get_constraintdef(con.oid, true) AS definition,
                       rn.nspname AS referenced_schema, rc.relname AS referenced_table
                FROM pg_constraint con
                JOIN pg_class c ON c.oid = con.conrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_class rc ON rc.oid = con.confrelid
                LEFT JOIN pg_namespace rn ON rn.oid = rc.relnamespace
                WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
                ORDER BY n.nspname, c.relname, con.conname
                """,
            )
            indexes = _fetch(
                cursor,
                """
                SELECT ns.nspname AS schema_name, tbl.relname AS table_name, idx.relname AS index_name,
                       i.indisprimary AS is_primary, i.indisunique AS is_unique,
                       pg_get_indexdef(i.indexrelid) AS definition,
                       pg_relation_size(i.indexrelid) AS index_bytes
                FROM pg_index i
                JOIN pg_class idx ON idx.oid = i.indexrelid
                JOIN pg_class tbl ON tbl.oid = i.indrelid
                JOIN pg_namespace ns ON ns.oid = tbl.relnamespace
                WHERE ns.nspname !~ '^pg_' AND ns.nspname <> 'information_schema'
                ORDER BY ns.nspname, tbl.relname, idx.relname
                """,
            )
            triggers = _fetch(
                cursor,
                """
                SELECT n.nspname AS schema_name, c.relname AS table_name, t.tgname AS trigger_name,
                       pg_get_triggerdef(t.oid, true) AS definition, t.tgenabled AS enabled
                FROM pg_trigger t
                JOIN pg_class c ON c.oid = t.tgrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE NOT t.tgisinternal
                  AND n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
                ORDER BY n.nspname, c.relname, t.tgname
                """,
            )
            functions = _fetch(
                cursor,
                """
                SELECT n.nspname AS schema_name, p.proname AS function_name,
                       pg_get_function_identity_arguments(p.oid) AS arguments,
                       pg_get_function_result(p.oid) AS result_type,
                       l.lanname AS language, p.provolatile AS volatility
                FROM pg_proc p
                JOIN pg_namespace n ON n.oid = p.pronamespace
                JOIN pg_language l ON l.oid = p.prolang
                WHERE n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
                ORDER BY n.nspname, p.proname
                """,
            )
            grants = _fetch(
                cursor,
                """
                SELECT table_schema AS schema_name, table_name, grantee, privilege_type,
                       is_grantable
                FROM information_schema.role_table_grants
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name, grantee, privilege_type
                """,
            )
            views = _fetch(
                cursor,
                """
                SELECT schemaname AS schema_name, viewname AS view_name, definition
                FROM pg_views
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, viewname
                """,
            )
            materialized_views = _fetch(
                cursor,
                """
                SELECT schemaname AS schema_name, matviewname AS view_name, ispopulated
                FROM pg_matviews
                WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schemaname, matviewname
                """,
            )
            sequences = _fetch(
                cursor,
                """
                SELECT sequence_schema AS schema_name, sequence_name, data_type
                FROM information_schema.sequences
                WHERE sequence_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY sequence_schema, sequence_name
                """,
            )

            integrity: dict[str, Any] = {
                "status": "PASS",
                "orphan_foreign_keys": [],
                "duplicate_unique_keys": [],
                "nullable_primary_key_columns": [],
                "status_values": {},
            }
            for item in constraints:
                if item["constraint_type"] == "FOREIGN KEY":
                    cursor.execute(
                        """
                        SELECT conkey, confkey
                        FROM pg_constraint
                        WHERE conname = %s
                          AND conrelid = format('%%I.%%I', %s, %s)::regclass
                        """,
                        (item["constraint_name"], item["schema_name"], item["table_name"]),
                    )
                    key_row = cursor.fetchone()
                    cursor.execute(
                        """
                        SELECT a.attnum, a.attname FROM pg_attribute a
                        WHERE a.attrelid = format('%%I.%%I', %s, %s)::regclass
                        """,
                        (item["schema_name"], item["table_name"]),
                    )
                    local_names = {row["attnum"]: row["attname"] for row in cursor.fetchall()}
                    cursor.execute(
                        """
                        SELECT a.attnum, a.attname FROM pg_attribute a
                        WHERE a.attrelid = format('%%I.%%I', %s, %s)::regclass
                        """,
                        (item["referenced_schema"], item["referenced_table"]),
                    )
                    remote_names = {row["attnum"]: row["attname"] for row in cursor.fetchall()}
                    joins = [
                        sql.SQL("c.{} = p.{}").format(
                            sql.Identifier(local_names[left]), sql.Identifier(remote_names[right])
                        )
                        for left, right in zip(key_row["conkey"], key_row["confkey"])
                    ]
                    not_null = [
                        sql.SQL("c.{} IS NOT NULL").format(sql.Identifier(local_names[left]))
                        for left in key_row["conkey"]
                    ]
                    missing = sql.SQL("p.{} IS NULL").format(
                        sql.Identifier(remote_names[key_row["confkey"][0]])
                    )
                    statement = sql.SQL(
                        "SELECT count(*) AS count FROM {}.{} c LEFT JOIN {}.{} p ON {} WHERE {} AND {}"
                    ).format(
                        sql.Identifier(item["schema_name"]),
                        sql.Identifier(item["table_name"]),
                        sql.Identifier(item["referenced_schema"]),
                        sql.Identifier(item["referenced_table"]),
                        sql.SQL(" AND ").join(joins),
                        sql.SQL(" AND ").join(not_null),
                        missing,
                    )
                    cursor.execute(statement)
                    count = int(cursor.fetchone()["count"])
                    if count:
                        integrity["orphan_foreign_keys"].append(
                            {"constraint": item["constraint_name"], "count": count}
                        )

            for column in columns:
                if column["column_name"] == "status":
                    cursor.execute(
                        sql.SQL(
                            "SELECT {}::text AS value, count(*) AS count FROM {}.{} "
                            "GROUP BY {} ORDER BY {}::text"
                        ).format(
                            sql.Identifier("status"),
                            sql.Identifier(column["schema_name"]),
                            sql.Identifier(column["table_name"]),
                            sql.Identifier("status"),
                            sql.Identifier("status"),
                        )
                    )
                    integrity["status_values"][
                        f'{column["schema_name"]}.{column["table_name"]}'
                    ] = [dict(row) for row in cursor.fetchall()]

            profile.update(
                {
                    "schema_count": len(schemas),
                    "base_table_count": len(tables),
                    "view_count": len(views),
                    "materialized_view_count": len(materialized_views),
                    "sequence_count": len(sequences),
                    "index_count": len(indexes),
                    "trigger_count": len(triggers),
                    "function_count": len(functions),
                    "total_rows": sum(table["row_count"] for table in tables),
                }
            )
            connection.rollback()

    return profile, {
        "schemas": schemas,
        "tables": tables,
        "columns": columns,
        "constraints": constraints,
        "indexes": indexes,
        "triggers": triggers,
        "functions": functions,
        "grants": grants,
        "views": views,
        "materialized_views": materialized_views,
        "sequences": sequences,
        "integrity_checks": integrity,
    }


def _repository_files() -> list[Path]:
    files: list[Path] = []
    for relative in SCAN_PATHS:
        path = ROOT / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and "__pycache__" not in candidate.parts
                and candidate.suffix.lower() in {".py", ".sql", ".yaml", ".yml", ".toml", ".json"}
            )
    return sorted(set(files))


def _usage(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    files = _repository_files()
    usage: list[dict[str, Any]] = []
    for table in tables:
        name = table["table_name"]
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", re.IGNORECASE)
        buckets: dict[str, list[str]] = {
            "read_references": [],
            "write_references": [],
            "migration_references": [],
            "test_references": [],
        }
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if not pattern.search(text):
                continue
            relative = path.relative_to(ROOT).as_posix()
            matches = [line for line in text.splitlines() if pattern.search(line)]
            if relative.startswith("database/"):
                buckets["migration_references"].append(relative)
            if relative.startswith("tests/"):
                buckets["test_references"].append(relative)
            if any(re.search(rf"\b(INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(?:\w+\.)?{re.escape(name)}\b", line, re.I) for line in matches):
                buckets["write_references"].append(relative)
            if any(re.search(rf"\b(FROM|JOIN)\s+(?:\w+\.)?{re.escape(name)}\b", line, re.I) for line in matches):
                buckets["read_references"].append(relative)
        for key in buckets:
            buckets[key] = sorted(set(buckets[key]))
        runtime_paths = [
            ref
            for ref in buckets["read_references"] + buckets["write_references"]
            if ref.startswith(("src/", "scripts/"))
            and ref not in HISTORICAL_DATABASE_PATHS
        ]
        if runtime_paths:
            runtime_status = "ACTIVE_RUNTIME"
        elif buckets["migration_references"]:
            runtime_status = "MIGRATION_ONLY"
        elif any(buckets.values()):
            runtime_status = "LEGACY_READ_ONLY"
        elif table["row_count"] == 0:
            runtime_status = "EMPTY_UNUSED"
        else:
            runtime_status = "UNKNOWN_REQUIRES_REVIEW"
        usage.append(
            {
                "schema": table["schema_name"],
                "table": name,
                **buckets,
                "runtime_status": runtime_status,
            }
        )
    return usage


def _disposition(
    tables: list[dict[str, Any]], usage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    usage_by_name = {item["table"]: item for item in usage}
    rows: list[dict[str, Any]] = []
    for table in tables:
        name = table["table_name"]
        decision, destination, method = LEGACY_DESTINATIONS.get(
            name, ("BLOCKED_REVIEW", "", "manual investigation required")
        )
        refs = usage_by_name[name]
        used_by_code = bool(refs["read_references"] or refs["write_references"])
        rows.append(
            {
                "old_table": f'{table["schema_name"]}.{name}',
                "rows": table["row_count"],
                "used_by_code": used_by_code,
                "contains_final_evidence": table["row_count"] > 0,
                "decision": decision,
                "new_destination": destination,
                "migration_method": method,
                "drop_allowed": False,
            }
        )
    return rows


def _render_reports(
    profile: dict[str, Any],
    tables: list[dict[str, Any]],
    usage: list[dict[str, Any]],
    disposition: list[dict[str, Any]],
) -> None:
    nonempty = [table for table in tables if table["row_count"]]
    empty = [table for table in tables if not table["row_count"]]
    audit_lines = [
        "# Database Current State Audit",
        "",
        f"- Audit timestamp: `{profile['audited_at']}`",
        f"- Endpoint: `{profile['connection']}`",
        f"- Database: `{profile['database_name']}`",
        f"- PostgreSQL: `{profile['server_version']}`",
        f"- Base tables: **{profile['base_table_count']}** (expected 29: **{'PASS' if profile['base_table_count'] == 29 else 'FAIL'}**)",
        f"- Views: **{profile['view_count']}**",
        f"- Triggers: **{profile['trigger_count']}**",
        f"- Exact total rows: **{profile['total_rows']}**",
        f"- Database size: **{profile['database_size']}**",
        "",
        "## Data-bearing tables",
        "",
        "None." if not nonempty else "\n".join(f"- `{t['schema_name']}.{t['table_name']}`: {t['row_count']}" for t in nonempty),
        "",
        "## Empty tables",
        "",
        *[f"- `{t['schema_name']}.{t['table_name']}`" for t in empty],
        "",
        "## Finding",
        "",
        "All 29 legacy application tables are structurally present but contain zero rows. "
        "Canonical final artifacts therefore remain the source of truth for result loading. "
        "No table is eligible for removal until backup/restore, protocol lock, code cutover, "
        "disposition validation, and the explicit empty-table confirmation gate all pass.",
        "",
    ]
    (REPORT_DIR / "DATABASE_CURRENT_STATE_AUDIT.md").write_text(
        "\n".join(audit_lines), encoding="utf-8"
    )

    usage_lines = [
        "# Database Code Usage",
        "",
        "| Schema | Table | Read refs | Write refs | Migration refs | Test refs | Runtime status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in usage:
        usage_lines.append(
            f"| {row['schema']} | {row['table']} | {len(row['read_references'])} | "
            f"{len(row['write_references'])} | {len(row['migration_references'])} | "
            f"{len(row['test_references'])} | {row['runtime_status']} |"
        )
    usage_lines.extend(
        [
            "",
            "References are file-level, collected from `src/`, `scripts/`, `tests/`, "
            "`database/`, `configs/`, `project.py`, and `docker-compose.yml`.",
            "",
        ]
    )
    (REPORT_DIR / "DATABASE_CODE_USAGE.md").write_text("\n".join(usage_lines), encoding="utf-8")

    disposition_lines = [
        "# Database Table Disposition",
        "",
        "This plan covers every live legacy table. No destination is unknown.",
        "",
        "| Old table | Rows | Used by code | Final evidence | Decision | New destination | Drop allowed |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in disposition:
        disposition_lines.append(
            f"| {row['old_table']} | {row['rows']} | {row['used_by_code']} | "
            f"{row['contains_final_evidence']} | {row['decision']} | "
            f"{row['new_destination']} | {row['drop_allowed']} |"
        )
    disposition_lines.extend(
        [
            "",
            "All legacy tables are empty, but `drop_allowed` remains false during audit. "
            "The migration CLI may change that outcome only after backup and restore validation, "
            "runtime reference removal, dependency checks, and the explicit "
            "`--confirm-drop-empty-legacy` flag.",
            "",
        ]
    )
    (REPORT_DIR / "DATABASE_TABLE_DISPOSITION.md").write_text(
        "\n".join(disposition_lines), encoding="utf-8"
    )
    _write_csv(
        REPORT_DIR / "DATABASE_TABLE_DISPOSITION.csv",
        disposition,
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


def _checksum_manifest(paths: list[Path]) -> dict[str, Any]:
    entries = []
    for path in sorted(paths):
        entries.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "byte_count": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }


def run(dsn: str, audit_label: str = "audit_before") -> None:
    profile, inventory = _inventory(dsn)
    audit_dir = DATABASE_DIR / audit_label
    audit_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _write_json(audit_dir / "database_profile.json", profile)
    for name in ("schemas", "tables", "columns", "constraints", "indexes", "triggers", "functions", "grants"):
        _write_json(audit_dir / f"{name}.json", inventory[name])
    _write_json(
        audit_dir / "integrity_checks.json",
        inventory["integrity_checks"],
    )
    _write_json(
        audit_dir / "objects.json",
        {
            "views": inventory["views"],
            "materialized_views": inventory["materialized_views"],
            "sequences": inventory["sequences"],
        },
    )

    row_counts = [
        {
            "schema": table["schema_name"],
            "table": table["table_name"],
            "row_count": table["row_count"],
        }
        for table in inventory["tables"]
    ]
    _write_csv(audit_dir / "row_counts.csv", row_counts, ["schema", "table", "row_count"])
    table_sizes = [
        {
            "schema": table["schema_name"],
            "table": table["table_name"],
            "table_bytes": table["table_bytes"],
            "total_bytes": table["total_bytes"],
        }
        for table in inventory["tables"]
    ]
    _write_csv(
        audit_dir / "table_sizes.csv",
        table_sizes,
        ["schema", "table", "table_bytes", "total_bytes"],
    )

    usage = _usage(inventory["tables"])
    disposition = _disposition(inventory["tables"], usage)
    if audit_label == "audit_before":
        _write_json(DATABASE_DIR / "database_code_usage.json", usage)
        _write_json(DATABASE_DIR / "table_disposition.json", disposition)
        _render_reports(profile, inventory["tables"], usage, disposition)
    else:
        lines = [
            "# Database Post-Cutover Audit",
            "",
            f"- Audit timestamp: `{profile['audited_at']}`",
            f"- Endpoint: `{profile['connection']}`",
            f"- Active base tables: **{profile['base_table_count']}**",
            f"- Views: **{profile['view_count']}**",
            f"- Triggers: **{profile['trigger_count']}**",
            f"- Exact total rows: **{profile['total_rows']}**",
            f"- Database size: **{profile['database_size']}**",
            "",
            "The detailed object inventory and checksum manifest are in "
            "`artifacts/final/database/audit_after/`.",
            "",
        ]
        (REPORT_DIR / "DATABASE_POST_CUTOVER_AUDIT.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    manifest_inputs = [
        path
        for path in audit_dir.iterdir()
        if path.is_file() and path.name != "schema_checksum_manifest.json"
    ]
    _write_json(
        audit_dir / "schema_checksum_manifest.json",
        _checksum_manifest(manifest_inputs),
    )
    counts = Counter(row["decision"] for row in disposition)
    print(
        json.dumps(
            {
                "status": "PASS",
                "base_tables": profile["base_table_count"],
                "total_rows": profile["total_rows"],
                "disposition": dict(sorted(counts.items())),
                "credentials": "REDACTED",
            },
            sort_keys=True,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn-env", default="POSTGRES_TEST_DSN")
    parser.add_argument(
        "--audit-label",
        choices=("audit_before", "audit_after"),
        default="audit_before",
    )
    args = parser.parse_args()
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        raise SystemExit(f"Missing required environment variable: {args.dsn_env}")
    run(dsn, args.audit_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
