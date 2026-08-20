"""Serve frozen Hybrid C0 + Five-EBM-C0 through live student_db.

C0 probabilities are read from prediction.prediction (OOF dump). Hybrid is
not refit. V3 uses the frozen EBM artifacts and persists the decision.
"""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd
from psycopg2.extras import Json, RealDictCursor

from src.database.connection import DatabaseSettings, connect_with_retry, load_dotenv
from src.recommend_hybrid.v3.contracts import (
    RecommendationDecision,
    RiskThresholds,
    SafetyThresholds,
    map_prediction_state,
)
from src.recommend_hybrid.v3.features_io import features_from_row
from src.recommend_hybrid.v3.pipeline import RecommendationV3Pipeline
from src.recommend_hybrid.v3.ranker import FiveEBMC0Ranker

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "artifacts" / "recommend_hybrid" / "v3"
FEATURES_PATH = V3 / "data" / "learner_stage_features.parquet"
RANKER_DIR = V3 / "ranker" / "final_models"
ROUTER_PATH = V3 / "router" / "ROUTER_CONFIG.json"

DB_STAGE_TO_PCT = {
    "20": "20pct",
    "35": "35pct",
    "50": "50pct",
    "75": "75pct",
    "20pct": "20pct",
    "35pct": "35pct",
    "50pct": "50pct",
    "75pct": "75pct",
    "EARLY_20": "20pct",
    "EARLY_35": "35pct",
    "MIDDLE_50": "50pct",
    "LATE_75": "75pct",
}
PCT_TO_DB_STAGE = {
    "20pct": "20",
    "35pct": "35",
    "50pct": "50",
    "75pct": "75",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def normalize_student_key(student: str) -> str:
    text = str(student).strip()
    if text.upper().startswith(("OULAD:", "UCI:")):
        prefix, rest = text.split(":", 1)
        return f"{prefix.upper()}:{rest}"
    if text.isdigit():
        return f"OULAD:{text}"
    return text


def normalize_prediction_stage(stage: str) -> str:
    key = str(stage).strip()
    if key not in DB_STAGE_TO_PCT:
        raise ValueError(f"unknown stage {stage!r}; use 20/35/50/75")
    return DB_STAGE_TO_PCT[key]


def query_id_for(student_key: str, course: str, presentation: str, stage_pct: str) -> str:
    numeric = student_key.split(":", 1)[-1] if student_key.startswith("OULAD:") else student_key
    v3_stage = map_prediction_state(stage_pct).value
    return f"{numeric}::{course}::{presentation}::{v3_stage}"


def _settings() -> DatabaseSettings:
    load_dotenv()
    return DatabaseSettings.from_environment()


@lru_cache(maxsize=1)
def _feature_table() -> pd.DataFrame:
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"missing V3 feature table: {FEATURES_PATH}")
    frame = pd.read_parquet(FEATURES_PATH)
    if "query_id" not in frame.columns:
        raise RuntimeError("learner_stage_features.parquet has no query_id")
    return frame


@lru_cache(maxsize=1)
def _pipeline() -> RecommendationV3Pipeline:
    payload = json.loads(ROUTER_PATH.read_text(encoding="utf-8"))
    return RecommendationV3Pipeline(
        FiveEBMC0Ranker.from_artifacts(RANKER_DIR),
        RiskThresholds(
            maximum_automatic_uncertainty=float(payload["risk"]["maximum_automatic_uncertainty"]),
            minimum_risk_margin=float(payload["risk"]["minimum_risk_margin"]),
        ),
        SafetyThresholds(
            minimum_top1_score=float(payload["safety"]["minimum_top1_score"]),
            minimum_top1_margin=float(payload["safety"]["minimum_top1_margin"]),
            maximum_uncertainty=float(payload["safety"]["maximum_uncertainty"]),
        ),
        review_k=3,
    )


def _fetch_enrollment(cursor, student_key: str, course: str, presentation: str) -> dict | None:
    cursor.execute(
        """
        SELECT
            s.student_id, s.external_student_id,
            c.course_id, c.course_code, c.presentation, c.dataset_key,
            e.enrollment_id, e.status
        FROM catalog.enrollment e
        JOIN catalog.student s ON s.student_id = e.student_id
        JOIN catalog.course c ON c.course_id = e.course_id
        WHERE s.external_student_id = %s
          AND c.course_code = %s
          AND c.presentation = %s
        """,
        (student_key, course, presentation),
    )
    row = cursor.fetchone()
    return dict(row) if row else None


def _fetch_predictions(cursor, enrollment_id, stage_pct: str | None) -> list[dict]:
    sql = """
        SELECT
            p.prediction_id, p.enrollment_id, p.stage, p.risk_probability,
            p.predicted_risk, p.threshold, p.uncertainty, p.predicted_at, p.metadata
        FROM prediction.prediction p
        WHERE p.enrollment_id = %s
    """
    params: list[Any] = [enrollment_id]
    if stage_pct is not None:
        sql += " AND p.stage = %s"
        params.append(PCT_TO_DB_STAGE[stage_pct])
    sql += " ORDER BY p.stage"
    cursor.execute(sql, tuple(params))
    return [dict(row) for row in cursor.fetchall()]


def _fetch_recommendation(cursor, prediction_id) -> dict | None:
    cursor.execute(
        """
        SELECT recommendation_id, prediction_id, risk_band, route_status, generated_at, metadata
        FROM recommendation.recommendation
        WHERE prediction_id = %s
        """,
        (prediction_id,),
    )
    rec = cursor.fetchone()
    if rec is None:
        return None
    payload = dict(rec)
    cursor.execute(
        """
        SELECT i.rank_position, a.action_key, i.score, i.feasible, i.explanation
        FROM recommendation.recommendation_item i
        JOIN recommendation.action a ON a.action_id = i.action_id
        WHERE i.recommendation_id = %s
        ORDER BY i.rank_position
        """,
        (payload["recommendation_id"],),
    )
    payload["items"] = [dict(row) for row in cursor.fetchall()]
    return payload


def lookup_case(
    student: str,
    course: str,
    presentation: str,
    stage: str | None = None,
) -> dict[str, Any]:
    student_key = normalize_student_key(student)
    stage_pct = normalize_prediction_stage(stage) if stage else None
    connection = connect_with_retry(_settings())
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            enrollment = _fetch_enrollment(cursor, student_key, course, presentation)
            if enrollment is None:
                return {
                    "ok": False,
                    "error": "ENROLLMENT_NOT_FOUND",
                    "student": student_key,
                    "course": course,
                    "presentation": presentation,
                }
            predictions = _fetch_predictions(cursor, enrollment["enrollment_id"], stage_pct)
            results = []
            for prediction in predictions:
                rec = _fetch_recommendation(cursor, prediction["prediction_id"])
                results.append({"prediction": prediction, "recommendation": rec})
            return jsonable(
                {
                    "ok": True,
                    "source": "student_db",
                    "enrollment": enrollment,
                    "query_id": (
                        query_id_for(student_key, course, presentation, stage_pct)
                        if stage_pct
                        else None
                    ),
                    "cases": results,
                }
            )
    finally:
        connection.close()


def predict_case(student: str, course: str, presentation: str, stage: str) -> dict[str, Any]:
    payload = lookup_case(student, course, presentation, stage)
    if not payload.get("ok"):
        return payload
    cases = payload.get("cases") or []
    if not cases:
        payload["ok"] = False
        payload["error"] = "C0_PREDICTION_NOT_FOUND"
        payload["note"] = "Frozen Hybrid C0 is served from prediction.prediction; it is not refit."
        return payload
    prediction = cases[0]["prediction"]
    return jsonable(
        {
            "ok": True,
            "source": "student_db",
            "model": "Hybrid C0",
            "refit": False,
            "enrollment": payload["enrollment"],
            "query_id": payload["query_id"],
            "prediction": prediction,
        }
    )


def _features_for_query(query_id: str, prediction: dict) -> Any:
    table = _feature_table()
    matched = table.loc[table["query_id"].astype(str) == str(query_id)]
    if matched.empty:
        raise KeyError(query_id)
    row = matched.iloc[0].copy()
    row["risk_probability"] = float(prediction["risk_probability"])
    row["predicted_risk"] = int(bool(prediction["predicted_risk"]))
    row["prediction_threshold"] = float(prediction["threshold"])
    row["uncertainty"] = float(prediction["uncertainty"])
    return features_from_row(row)


def _decision_payload(decision: RecommendationDecision) -> dict[str, Any]:
    plan = None
    if decision.plan is not None:
        plan = asdict(decision.plan)
        plan["observed_evidence"] = list(decision.plan.observed_evidence)
    return {
        "student_key": decision.student_key,
        "course_key": decision.course_key,
        "stage": decision.stage.value,
        "risk_route": decision.risk_route.value,
        "route": decision.route.value,
        "reason_codes": list(decision.reason_codes),
        "protocol_version": decision.protocol_version,
        "ranked_actions": [
            {"action": item.action.value, "score": float(item.score)}
            for item in decision.ranked_actions
        ],
        "plan": plan,
    }


def _persist_decision(cursor, prediction_id, decision: RecommendationDecision, query_id: str) -> str:
    cursor.execute(
        "SELECT action_id, action_key FROM recommendation.action WHERE is_active IS TRUE"
    )
    actions = {str(row["action_key"]): row["action_id"] for row in cursor.fetchall()}
    for key in [item.action.value for item in decision.ranked_actions]:
        if key not in actions:
            cursor.execute(
                """
                INSERT INTO recommendation.action (action_key, action_name, description, is_active)
                VALUES (%s,%s,%s,TRUE) RETURNING action_id
                """,
                (key, key.replace("_", " ").title(), "V3 canonical action"),
            )
            actions[key] = cursor.fetchone()["action_id"]
    cursor.execute(
        """
        DELETE FROM recommendation.recommendation_item
        WHERE recommendation_id IN (
            SELECT recommendation_id FROM recommendation.recommendation
            WHERE prediction_id = %s
        )
        """,
        (prediction_id,),
    )
    cursor.execute("DELETE FROM recommendation.recommendation WHERE prediction_id = %s", (prediction_id,))
    cursor.execute(
        """
        INSERT INTO recommendation.recommendation (prediction_id, risk_band, route_status, metadata)
        VALUES (%s,%s,%s,%s) RETURNING recommendation_id
        """,
        (
            prediction_id,
            "C0",
            decision.route.value,
            Json(
                {
                    "query_id": query_id,
                    "authority": "Five-EBM-C0",
                    "source": "runtime",
                    "risk_route": decision.risk_route.value,
                    "reason_codes": list(decision.reason_codes),
                    "protocol_version": decision.protocol_version,
                }
            ),
        ),
    )
    rec_id = cursor.fetchone()["recommendation_id"]
    for rank, item in enumerate(decision.ranked_actions, start=1):
        action_id = actions.get(item.action.value)
        if action_id is None:
            continue
        explanation: dict[str, Any] = {"action_id": item.action.value}
        if rank == 1 and decision.plan is not None:
            explanation["plan"] = _decision_payload(decision)["plan"]
        cursor.execute(
            """
            INSERT INTO recommendation.recommendation_item
                (recommendation_id, action_id, rank_position, score, explanation, feasible)
            VALUES (%s,%s,%s,%s,%s,%s)
            """,
            (rec_id, action_id, rank, float(item.score), Json(explanation), True),
        )
    return str(rec_id)


def recommend_case(
    student: str,
    course: str,
    presentation: str,
    stage: str,
    *,
    persist: bool = True,
) -> dict[str, Any]:
    student_key = normalize_student_key(student)
    stage_pct = normalize_prediction_stage(stage)
    query_id = query_id_for(student_key, course, presentation, stage_pct)
    connection = connect_with_retry(_settings())
    try:
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            enrollment = _fetch_enrollment(cursor, student_key, course, presentation)
            if enrollment is None:
                return {
                    "ok": False,
                    "error": "ENROLLMENT_NOT_FOUND",
                    "student": student_key,
                    "course": course,
                    "presentation": presentation,
                }
            predictions = _fetch_predictions(cursor, enrollment["enrollment_id"], stage_pct)
            if not predictions:
                return {
                    "ok": False,
                    "error": "C0_PREDICTION_NOT_FOUND",
                    "student": student_key,
                    "query_id": query_id,
                }
            prediction = predictions[0]
            try:
                features = _features_for_query(query_id, prediction)
            except KeyError:
                return {
                    "ok": False,
                    "error": "V3_FEATURES_NOT_FOUND",
                    "query_id": query_id,
                }
            features = replace(
                features,
                risk_probability=float(prediction["risk_probability"]),
                predicted_risk=int(bool(prediction["predicted_risk"])),
                prediction_threshold=float(prediction["threshold"]),
                uncertainty=float(prediction["uncertainty"]),
            )
            decision = _pipeline().recommend(features)
            rec_id = None
            if persist:
                rec_id = _persist_decision(cursor, prediction["prediction_id"], decision, query_id)
                connection.commit()
            return jsonable(
                {
                    "ok": True,
                    "source": "runtime",
                    "persist": persist,
                    "refit": False,
                    "c0_source": "prediction.prediction",
                    "ranker": "Five-EBM-C0",
                    "enrollment": enrollment,
                    "query_id": query_id,
                    "prediction": prediction,
                    "recommendation_id": rec_id,
                    "decision": _decision_payload(decision),
                }
            )
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


__all__ = [
    "jsonable",
    "lookup_case",
    "normalize_prediction_stage",
    "normalize_student_key",
    "predict_case",
    "query_id_for",
    "recommend_case",
]
