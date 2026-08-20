"""Drop unused leftover Postgres objects; keep only live student_db tables in use."""
from __future__ import annotations

from pathlib import Path

from psycopg2 import sql

from src.database.connection import DatabaseSettings, connect_with_retry, load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PRUNE_SQL = ROOT / "database" / "live" / "002_prune_unused.sql"
KEEP_DATABASES = {"postgres", "student_db"}


def _drop_leftover_databases(settings: DatabaseSettings) -> None:
    admin = DatabaseSettings(
        host=settings.host,
        port=settings.port,
        database="postgres",
        user=settings.user,
        password=settings.password,
    )
    connection = connect_with_retry(admin)
    try:
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT datname
                FROM pg_database
                WHERE datistemplate = false
                ORDER BY 1
                """
            )
            names = [row[0] for row in cursor.fetchall()]
            for name in names:
                if name in KEEP_DATABASES:
                    continue
                print("DROP DATABASE", name)
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid <> pg_backend_pid()",
                    (name,),
                )
                cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
    finally:
        connection.close()


def _print_remaining(settings: DatabaseSettings) -> None:
    connection = connect_with_retry(settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_database(), pg_size_pretty(pg_database_size(current_database()))")
            print("database", *cursor.fetchone())
            cursor.execute(
                """
                SELECT n.nspname, c.relname, pg_size_pretty(pg_total_relation_size(c.oid))
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')
                  AND c.relkind = 'r'
                ORDER BY n.nspname, c.relname
                """
            )
            print("schema.table\tsize\trows")
            for schema, table, size in cursor.fetchall():
                cursor.execute(
                    sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                )
                count = cursor.fetchone()[0]
                print(f"{schema}.{table}\t{size}\t{count}")
            cursor.execute(
                """
                SELECT datname, pg_size_pretty(pg_database_size(datname))
                FROM pg_database
                WHERE datistemplate = false
                ORDER BY 1
                """
            )
            print("remaining_databases")
            for row in cursor.fetchall():
                print(" ", row[0], row[1])
    finally:
        connection.close()


def main() -> int:
    load_dotenv()
    settings = DatabaseSettings.from_environment()
    if settings.database != "student_db":
        raise RuntimeError(f"refusing to prune database {settings.database!r}; expected student_db")
    sql = PRUNE_SQL.read_text(encoding="utf-8")
    connection = connect_with_retry(settings)
    try:
        connection.autocommit = False
        with connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = 0")
            cursor.execute(sql)
        connection.commit()
        print("PRUNED student_db unused schemas/tables")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    _drop_leftover_databases(settings)
    _print_remaining(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
