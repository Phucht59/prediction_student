from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from dotenv import dotenv_values

from src.database.connection import DatabaseSettings, connect_with_retry

from .contract import ARTIFACT_ROOT, ROOT, SCHEMA_VERSION, atomic_json, sha256_file


MIGRATION = (
    ROOT
    / "database/final/migrations/011_create_v6_2_expert_review_validation.sql"
)


def _dsn_candidates() -> tuple[str | None, bool, str]:
    explicit = os.getenv("POSTGRES_TEST_DSN") or os.getenv("POSTGRES_TEST_APP_DSN")
    if explicit:
        return explicit, True, "EXPLICIT_DISPOSABLE_TEST_DSN"
    values = dotenv_values(ROOT / ".env") if (ROOT / ".env").is_file() else {}
    read_only = (
        os.getenv("POSTGRES_RUNTIME_APP_DSN")
        or values.get("DATABASE_URL")
    )
    if not read_only and all(
        values.get(name)
        for name in (
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_DB",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
        )
    ):
        read_only = (
            "postgresql://"
            f"{quote_plus(str(values['POSTGRES_USER']))}:"
            f"{quote_plus(str(values['POSTGRES_PASSWORD']))}@"
            f"{values['POSTGRES_HOST']}:{values['POSTGRES_PORT']}/"
            f"{quote_plus(str(values['POSTGRES_DB']))}"
        )
    return str(read_only) if read_only else None, False, "READ_ONLY_RUNTIME_DSN"


def _static_migration_checks() -> dict[str, Any]:
    text = MIGRATION.read_text(encoding="utf-8")
    required = [
        "recommendation.expert_review_case",
        "recommendation.expert_plan_review",
        "recommendation.expert_action_review",
        "UNIQUE (case_id, reviewer_key)",
        "UNIQUE (case_id, reviewer_key, action_code)",
        "^E[0-9]{2,}$",
    ]
    return {
        "status": "PASS" if all(token in text for token in required) else "FAIL",
        "migration": str(MIGRATION.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256_file(MIGRATION),
        "additive_only": not any(
            token in text.upper()
            for token in ("DROP TABLE", "TRUNCATE", "DELETE FROM", "ALTER TABLE")
        ),
        "required_constraints_present": {
            token: token in text for token in required
        },
    }


def _table_count(connection: Any, relation: str) -> int | None | str:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT to_regclass(%s)", (relation,))
            if cursor.fetchone()[0] is None:
                connection.rollback()
                return None
            cursor.execute(
                f"SELECT count(*) FROM {relation}"
            )  # relation is an internal constant
            count = int(cursor.fetchone()[0])
        connection.rollback()
        return count
    except Exception:
        connection.rollback()
        return "NO_SELECT_PRIVILEGE"


def audit_database(*, apply_migration: bool = False) -> dict[str, Any]:
    static = _static_migration_checks()
    dsn, disposable, source = _dsn_candidates()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_STATIC_ONLY_NO_DSN",
        "credentials_recorded": False,
        "production_write": False,
        "dsn_source": source if dsn else "NONE",
        "disposable_dsn_confirmed": disposable,
        "migration_applied": False,
        "static_migration_validation": static,
        "tables": {},
        "fake_expert_records": None,
        "expert_records_verified_empty": False,
    }
    if not dsn:
        atomic_json(ARTIFACT_ROOT / "database_audit.json", result)
        return result
    if apply_migration and not disposable:
        raise RuntimeError(
            "Refusing schema mutation without POSTGRES_TEST_DSN or "
            "POSTGRES_TEST_APP_DSN"
        )
    try:
        connection = connect_with_retry(DatabaseSettings(dsn=dsn), attempts=1)
    except Exception as exc:
        result["status"] = "PASS_STATIC_ONLY_DATABASE_CONNECTION_UNAVAILABLE"
        result["connection_error_type"] = type(exc).__name__
        atomic_json(ARTIFACT_ROOT / "database_audit.json", result)
        return result
    try:
        if apply_migration:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute(MIGRATION.read_text(encoding="utf-8"))
            result["migration_applied"] = True
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT current_database(), current_user, "
                "current_setting('server_version_num')"
            )
            database_name, _user, server_version = cursor.fetchone()
            result["database"] = {
                "name": database_name,
                "server_version_num": server_version,
                "user_redacted": True,
            }
            relations = (
                "ml.model",
                "ml.run",
                "ml.metric",
                "recommendation.risk_profile",
                "recommendation.plan",
                "recommendation.action",
                "recommendation.review",
                "recommendation.expert_review_case",
                "recommendation.expert_plan_review",
                "recommendation.expert_action_review",
                "v6_prediction_runs",
                "v6_student_risk_profiles",
                "v6_recommendation_plans",
                "v6_expert_evaluations",
            )
        result["tables"] = {
            relation: _table_count(connection, relation) for relation in relations
        }
        expert_relations = (
            "recommendation.expert_plan_review",
            "recommendation.expert_action_review",
            "v6_expert_evaluations",
        )
        counts = [
            result["tables"].get(relation)
            for relation in expert_relations
            if isinstance(result["tables"].get(relation), int)
        ]
        result["fake_expert_records"] = int(sum(counts)) if counts else None
        result["expert_records_verified_empty"] = bool(counts) and sum(counts) == 0
        result["status"] = (
            "PASS"
            if static["status"] == "PASS"
            and result["fake_expert_records"] in {0, None}
            and (result["migration_applied"] or not apply_migration)
            else "FAIL"
        )
        if not disposable and not result["migration_applied"]:
            result["status"] = "PASS_READ_ONLY_MIGRATION_PENDING_DISPOSABLE_DSN"
    finally:
        connection.close()
    atomic_json(ARTIFACT_ROOT / "database_audit.json", result)
    return result


__all__ = ["audit_database"]
