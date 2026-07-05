"""Append materialized learning recommendations for an existing prediction run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psycopg2 import errors
from psycopg2.extras import Json, RealDictCursor

from src.evaluation.evaluation import _connect, canonical_json
from src.recommendation import (
    POLICY_VERSION,
    build_recommendation,
    prepare_recommendation_features,
    validate_recommendation_schema,
)


PREDICTION_CONTEXT_QUERY = """
SELECT
    p.prediction_id,
    p.predicted_label,
    p.confidence,
    p.probability,
    p.split_name,
    sr.source_row_number,
    sr.raw_payload
FROM ml_predictions p
JOIN source_records sr ON sr.record_id = p.record_id
WHERE p.run_id = %s
ORDER BY sr.source_row_number, p.prediction_id
"""


def _same_json(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def load_prediction_context(connection, run_id: str) -> list[dict[str, Any]]:
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(PREDICTION_CONTEXT_QUERY, (run_id,))
        return [dict(row) for row in cursor.fetchall()]


def build_materialized_payload(row: dict[str, Any], policy_version: str) -> dict[str, Any]:
    if policy_version != POLICY_VERSION:
        raise ValueError(f"Unsupported policy_version {policy_version!r}; expected {POLICY_VERSION!r}.")
    features = prepare_recommendation_features(dict(row["raw_payload"]))
    recommendation = build_recommendation(
        features=features,
        predicted_class=int(row["predicted_label"]),
        confidence=float(row["confidence"]),
        probability=dict(row["probability"]),
    )
    validate_recommendation_schema(recommendation)
    return {
        "prediction_id": int(row["prediction_id"]),
        "policy_version": policy_version,
        "risk_band": recommendation["risk_band"],
        "learning_path": recommendation,
        "explanation": {
            "policy_version": policy_version,
            "confidence_level": recommendation["confidence_level"],
            "prediction_basis": recommendation["explanation"]["prediction_basis"],
            "risk_factor_basis": recommendation["explanation"]["risk_factor_basis"],
            "scope_note": recommendation["explanation"]["scope_note"],
        },
    }


def insert_or_compare_recommendation(cursor, payload: dict[str, Any]) -> str:
    try:
        cursor.execute(
            """
            INSERT INTO ml_recommendations (
                prediction_id, policy_version, risk_band, learning_path, explanation
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (prediction_id, policy_version) DO NOTHING
            """,
            (
                payload["prediction_id"],
                payload["policy_version"],
                payload["risk_band"],
                Json(payload["learning_path"]),
                Json(payload["explanation"]),
            ),
        )
    except errors.IntegrityConstraintViolation as exc:
        if (
            "Recommendation can only be inserted while its parent run is running" in str(exc)
            or "Recommendation can only be inserted while its parent run is running or completed" in str(exc)
        ):
            raise RuntimeError(
                "Database rejected append-only materialization because the parent run is not materializable. "
                "The current schema trigger allows recommendation materialization only for running or completed runs."
            ) from exc
        raise
    inserted = cursor.rowcount == 1
    cursor.execute(
        """
        SELECT risk_band, learning_path, explanation
        FROM ml_recommendations
        WHERE prediction_id = %s
          AND policy_version = %s
        """,
        (payload["prediction_id"], payload["policy_version"]),
    )
    existing = cursor.fetchone()
    if existing is None:
        raise RuntimeError(
            "Recommendation insert was not persisted. If the parent run is already completed, "
            "the database trigger may be enforcing running-only inserts."
        )
    existing_dict = dict(existing)
    mismatches = [
        key
        for key in ("risk_band", "learning_path", "explanation")
        if not _same_json(payload[key], existing_dict[key])
    ]
    if mismatches:
        raise RuntimeError(
            "Existing recommendation differs for "
            f"prediction_id={payload['prediction_id']} policy_version={payload['policy_version']}: "
            + ", ".join(mismatches)
        )
    return "inserted" if inserted else "identical"


def materialize_recommendations(run_id: str, policy_version: str) -> dict[str, Any]:
    connection = _connect()
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT current_database(), current_user")
            database_name, user_name = cursor.fetchone().values()
        rows = load_prediction_context(connection, run_id)
        if not rows:
            raise RuntimeError(f"No predictions found for run_id={run_id}.")
        if any(row["split_name"] != "test" for row in rows):
            raise RuntimeError("Materialization only supports persisted test predictions.")
        inserted = 0
        identical = 0
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            for row in rows:
                payload = build_materialized_payload(row, policy_version)
                result = insert_or_compare_recommendation(cursor, payload)
                inserted += int(result == "inserted")
                identical += int(result == "identical")
        connection.commit()
        return {
            "database": database_name,
            "user": user_name,
            "run_id": run_id,
            "policy_version": policy_version,
            "predictions": len(rows),
            "inserted": inserted,
            "identical": identical,
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append versioned learning recommendations for an existing run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--policy-version", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize_recommendations(args.run_id, args.policy_version)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
