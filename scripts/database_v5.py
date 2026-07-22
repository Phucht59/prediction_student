from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.database.connection import DatabaseSettings, connect_with_retry, transaction
from src.database.services import IngestionService
from src.studies.v5.common.artifacts import atomic_write_json
from src.studies.v5.common.protocol import sha256_file


MIGRATIONS = ROOT / "database" / "v5" / "migrations"
SCHEMAS = ["source", "education", "feature", "experiment", "evaluation", "recommendation"]


def _settings(*, mutating: bool) -> DatabaseSettings:
    load_dotenv(ROOT / ".env")
    return DatabaseSettings.from_environment(require_v5_dsn=mutating)


def _migration_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^\s*BEGIN;", "", text, count=1, flags=re.IGNORECASE)
    text = re.sub(r"COMMIT;\s*$", "", text, count=1, flags=re.IGNORECASE)
    return text


def migrate() -> dict[str, object]:
    settings = _settings(mutating=True)
    applied = []
    with transaction(settings) as connection:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            digest = sha256_file(path)
            with connection.cursor() as cursor:
                if path.name != "001_create_schemas.sql":
                    cursor.execute("SELECT sha256 FROM source.schema_migration WHERE version=%s", (path.name.split("_", 1)[0],))
                    row = cursor.fetchone()
                    if row:
                        if row[0] != digest:
                            raise RuntimeError(f"Applied migration checksum changed: {path.name}")
                        continue
                cursor.execute(_migration_body(path))
                cursor.execute(
                    """INSERT INTO source.schema_migration(version,filename,sha256) VALUES (%s,%s,%s)
                       ON CONFLICT (version) DO NOTHING""",
                    (path.name.split("_", 1)[0], path.name, digest),
                )
            applied.append(path.name)
    return {"status": "PASS", "applied": applied, "migration_count": len(list(MIGRATIONS.glob('*.sql')))}


def seed() -> dict[str, object]:
    settings = _settings(mutating=True)
    service = IngestionService(settings)
    rows = []
    for slug, name, filename, count in [
        ("student-mat", "UCI Student Mathematics", "student-mat.csv", 395),
        ("student-por", "UCI Student Portuguese", "student-por.csv", 649),
        ("oulad", "Open University Learning Analytics Dataset", "studentInfo.csv", 32593),
    ]:
        rows.append(
            service.register_source(
                slug=slug,
                display_name=name,
                source_path=ROOT / "data" / "raw" / filename,
                version_label="v5-frozen-20260718",
                row_count=count,
                data_schema={"registered_by": "scripts/database_v5.py", "full_ingestion": slug != "oulad"},
                license_note="See dataset source documentation; metadata registration is not a license grant.",
            )
        )
    return {"status": "PASS", "datasets": rows}


def audit() -> dict[str, object]:
    settings = _settings(mutating=False)
    connection = connect_with_retry(settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), current_user")
            database, user = cursor.fetchone()
            cursor.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = ANY(%s) ORDER BY schema_name",
                (SCHEMAS,),
            )
            schemas = [row[0] for row in cursor.fetchall()]
            cursor.execute(
                """SELECT table_schema, count(*) FROM information_schema.tables
                   WHERE table_schema = ANY(%s) AND table_type='BASE TABLE'
                   GROUP BY table_schema ORDER BY table_schema""",
                (SCHEMAS,),
            )
            tables = dict(cursor.fetchall())
            cursor.execute("SELECT rolname FROM pg_roles WHERE rolname = ANY(%s) ORDER BY rolname", (["student_predict_migrator", "student_predict_writer", "student_predict_reader"],))
            roles = [row[0] for row in cursor.fetchall()]
        result = {
            "status": "PASS" if set(schemas) == set(SCHEMAS) and len(roles) == 3 else "FAIL",
            "database": database,
            "user": user,
            "schemas": schemas,
            "tables_by_schema": tables,
            "roles": roles,
            "credentials_logged": False,
        }
    finally:
        connection.close()
    output = ROOT / "reports" / "v5" / "final" / "database_audit.json"
    atomic_write_json(output, result)
    return result


def reset(confirm_disposable: bool) -> dict[str, object]:
    if not confirm_disposable:
        raise RuntimeError("Reset requires --confirm-disposable")
    settings = _settings(mutating=True)
    connection = connect_with_retry(settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database()")
            database = str(cursor.fetchone()[0]).lower()
            if not any(marker in database for marker in ["test", "dev", "disposable", "_v5"]):
                raise RuntimeError(f"Refusing reset for non-disposable database name: {database}")
            for schema in reversed(SCHEMAS):
                cursor.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        connection.commit()
    finally:
        connection.close()
    return {"status": "PASS", "removed_schemas": SCHEMAS, "recoverable": "restore from explicit backup only"}


def start() -> dict[str, object]:
    completed = subprocess.run(["docker", "compose", "up", "-d", "postgres-v5"], cwd=ROOT, check=False)
    return {"status": "PASS" if completed.returncode == 0 else "FAIL", "returncode": completed.returncode}


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("start")
    commands.add_parser("migrate")
    commands.add_parser("seed")
    commands.add_parser("audit")
    reset_parser = commands.add_parser("reset")
    reset_parser.add_argument("--confirm-disposable", action="store_true")
    args = parser.parse_args()
    handlers = {"start": start, "migrate": migrate, "seed": seed, "audit": audit}
    result = reset(args.confirm_disposable) if args.command == "reset" else handlers[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

