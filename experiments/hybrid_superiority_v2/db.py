"""PostgreSQL research logger. Credentials come from env; never logged."""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote_plus

import psycopg2
from psycopg2.extras import Json, execute_values

from .io_utils import load_dotenv, sha256_json, utc_now
from .paths import PROJECT_ROOT
from .protocol import PROTOCOL_ID, protocol_hash, protocol_payload


MIGRATION = PROJECT_ROOT / "database" / "migrations" / "009_research_hybrid_superiority_v2.sql"


def settings() -> dict[str, Any]:
    load_dotenv()
    return {
        "host": os.environ.get("POSTGRES_HOST") or os.environ.get("DB_HOST") or "localhost",
        "port": int(os.environ.get("POSTGRES_PORT") or os.environ.get("DB_PORT") or "5432"),
        "dbname": os.environ.get("POSTGRES_DB") or os.environ.get("DB_NAME") or "student_db",
        "user": os.environ.get("POSTGRES_USER") or os.environ.get("DB_USER") or "postgres",
        "password": os.environ.get("POSTGRES_PASSWORD") or os.environ.get("DB_PASSWORD") or "",
    }


def redacted_settings() -> dict[str, Any]:
    raw = settings()
    return {k: ("present" if k == "password" and v else v) for k, v in raw.items() if k != "password"} | {
        "password_present": bool(raw["password"])
    }


def optuna_storage_url() -> str:
    load_dotenv()
    explicit = os.environ.get("OPTUNA_STORAGE_URL")
    if explicit:
        return explicit
    s = settings()
    user = quote_plus(s["user"])
    password = quote_plus(s["password"])
    return (
        f"postgresql+psycopg2://{user}:{password}@{s['host']}:{s['port']}/{s['dbname']}"
        "?options=-csearch_path%3Doptuna_hs_v2"
    )


@contextmanager
def connect() -> Iterator:
    s = settings()
    conn = psycopg2.connect(connect_timeout=8, **s)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def migrate() -> dict[str, Any]:
    sql = MIGRATION.read_text(encoding="utf-8")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                """
                INSERT INTO research.protocol (protocol_id, version, sha256, git_commit, payload_jsonb)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (protocol_id) DO UPDATE
                SET sha256 = EXCLUDED.sha256, payload_jsonb = EXCLUDED.payload_jsonb
                """,
                (PROTOCOL_ID, PROTOCOL_ID, protocol_hash(), os.popen("git rev-parse HEAD").read().strip(), Json(protocol_payload())),
            )
            cur.execute("SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('research','optuna_hs_v2','recommendation')")
            schemas = [r[0] for r in cur.fetchall()]
    return {"ok": True, "schemas": schemas, "protocol_id": PROTOCOL_ID, "protocol_hash": protocol_hash()}


def log_event(event_type: str, payload: dict[str, Any], *, run_uuid: str | None = None, level: str = "info") -> None:
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO research.event_log (run_uuid, level, event_type, payload_jsonb) VALUES (%s,%s,%s,%s)",
                    (run_uuid, level, event_type, Json(payload)),
                )
    except Exception:
        return


def insert_run(row: dict[str, Any]) -> str:
    run_uuid = row.get("run_uuid") or str(uuid.uuid4())
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research.run (
                    run_uuid, study_id, trial_number, dataset, model, outer_fold, inner_fold, seed,
                    git_commit, config_hash, data_hash, status, started_at, ended_at, device_jsonb,
                    parameter_count, peak_vram, runtime_seconds, outer_test_used
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (run_uuid) DO UPDATE SET status = EXCLUDED.status, ended_at = EXCLUDED.ended_at
                """,
                (
                    run_uuid,
                    row.get("study_id"),
                    row.get("trial_number"),
                    row.get("dataset"),
                    row.get("model"),
                    row.get("outer_fold"),
                    row.get("inner_fold"),
                    row.get("seed"),
                    row.get("git_commit"),
                    row.get("config_hash"),
                    row.get("data_hash"),
                    row.get("status"),
                    row.get("started_at") or utc_now(),
                    row.get("ended_at"),
                    Json(row.get("device_jsonb") or {}),
                    row.get("parameter_count"),
                    row.get("peak_vram"),
                    row.get("runtime_seconds"),
                    bool(row.get("outer_test_used", False)),
                ),
            )
    return run_uuid


def insert_metrics(run_uuid: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    values = [(run_uuid, r["split"], r["stage"], r["metric_name"], r["value"]) for r in rows]
    with connect() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO research.metric (run_uuid, split, stage, metric_name, value) VALUES %s ON CONFLICT DO NOTHING",
                values,
            )


def health() -> dict[str, Any]:
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user")
            db, user = cur.fetchone()
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
    return {"ok": True, "database": db, "user": user, "version": version[:80], **redacted_settings()}
