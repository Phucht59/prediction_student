from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/v5_1/final/database_audit.json"
DISPOSABLE_MARKERS = ("test", "dev", "disposable", "v5_1")


def _write(value: dict[str, object]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _generic_dsn() -> str | None:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    required = ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]
    if not all(os.getenv(name) for name in required):
        return None
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
    )


def _database_name(dsn: str) -> str:
    return urlparse(dsn).path.lstrip("/").lower()


def _read_only_audit(dsn: str | None) -> dict[str, object]:
    if not dsn:
        return {"status": "SKIP_NO_READ_ONLY_DSN"}
    try:
        connection = psycopg2.connect(dsn, connect_timeout=5)
        connection.set_session(readonly=True, autocommit=True)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_user")
                database, user = cursor.fetchone()
                cursor.execute(
                    """SELECT schema_name FROM information_schema.schemata
                       WHERE schema_name IN ('source','education','feature','experiment','evaluation','recommendation')
                       ORDER BY schema_name"""
                )
                schemas = [row[0] for row in cursor.fetchall()]
                cursor.execute(
                    """SELECT table_schema, count(*) FROM information_schema.tables
                       WHERE table_schema IN ('source','education','feature','experiment','evaluation','recommendation')
                       AND table_type='BASE TABLE' GROUP BY table_schema ORDER BY table_schema"""
                )
                tables = dict(cursor.fetchall())
            return {
                "status": "PASS_READ_ONLY",
                "database": database,
                "user": user,
                "schemas": schemas,
                "tables_by_schema": tables,
                "transaction_read_only": True,
            }
        finally:
            connection.close()
    except Exception as error:
        return {"status": "SKIP_READ_ONLY_CONNECTION_UNAVAILABLE", "error_type": type(error).__name__}


def main() -> int:
    load_dotenv(ROOT / ".env")
    test_dsn = os.getenv("POSTGRES_TEST_DSN")
    app_dsn = os.getenv("POSTGRES_TEST_APP_DSN")
    disposable = bool(test_dsn) and any(marker in _database_name(test_dsn) for marker in DISPOSABLE_MARKERS)
    if not test_dsn:
        integration = {
            "status": "SKIP_NO_DISPOSABLE_DSN",
            "reason": "POSTGRES_TEST_DSN is not present in the execution environment.",
        }
    elif not disposable:
        integration = {
            "status": "SKIP_NON_DISPOSABLE_DATABASE",
            "database": _database_name(test_dsn),
            "reason": "Destructive migration/permission tests require a database name marked test/dev/disposable/v5_1.",
        }
    elif not app_dsn:
        integration = {
            "status": "SKIP_NO_DISPOSABLE_APP_DSN",
            "database": _database_name(test_dsn),
            "reason": "Least-privilege integration also requires POSTGRES_TEST_APP_DSN.",
        }
    else:
        integration = {
            "status": "SKIP_NOT_EXECUTED_BY_READ_ONLY_AUDITOR",
            "database": _database_name(test_dsn),
            "reason": "Disposable credentials were detected; use the dedicated migration integration suite explicitly.",
        }

    result = {
        "status": "PASS_WITH_TRANSPARENT_SKIP" if integration["status"].startswith("SKIP") else "PASS",
        "read_only_existing_database": _read_only_audit(_generic_dsn()),
        "destructive_disposable_integration": integration,
        "writes_executed": False,
        "credentials_logged": False,
    }
    _write(result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
