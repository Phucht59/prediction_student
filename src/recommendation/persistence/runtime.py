"""Transactional recommendation runtime persistence. Reuses catalog upserts."""

from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy import text

from src.database.connection import engine, transaction
from src.database.repository import upsert_course, upsert_student


def _json(value: Any) -> Any:
    from src.database.repository import _json as encode

    return encode(value)


def ensure_bundle(bundle_version: str, freeze_version: str, checksums: Mapping[str, str]) -> str:
    with transaction() as connection:
        existing = connection.execute(
            text("SELECT bundle_id FROM recommendation.bundle WHERE bundle_version=:v"),
            {"v": bundle_version},
        ).scalar_one_or_none()
        if existing:
            return str(existing)
        return str(connection.execute(
            text(
                "INSERT INTO recommendation.bundle(bundle_version,freeze_version,checksums) "
                "VALUES (:v,:f,:c) RETURNING bundle_id"
            ),
            {"v": bundle_version, "f": freeze_version, "c": _json(dict(checksums))},
        ).scalar_one())


def upsert_enrollment_external(student_id: str, module: str, presentation: str, external_enrollment_id: str) -> str:
    sid = upsert_student(f"OULAD:{student_id}")
    cid = upsert_course(module, presentation, module)
    with transaction() as connection:
        connection.execute(
            text(
                "INSERT INTO catalog.enrollment(student_id,course_id,external_enrollment_id,enrolled_at,status) "
                "VALUES (:s,:c,:e,NOW(),'ACTIVE') "
                "ON CONFLICT (student_id,course_id) DO UPDATE "
                "SET external_enrollment_id=COALESCE(catalog.enrollment.external_enrollment_id,EXCLUDED.external_enrollment_id), status='ACTIVE'"
            ),
            {"s": sid, "c": cid, "e": external_enrollment_id},
        )
        enrollment_id = connection.execute(
            text("SELECT enrollment_id FROM catalog.enrollment WHERE student_id=:s AND course_id=:c"),
            {"s": sid, "c": cid},
        ).scalar_one()
        return str(enrollment_id)


def persist_recommendation(result: dict, payload: Mapping[str, Any], *, bundle_version: str, state_version: str) -> str:
    from ..finalization import FREEZE_VERSION as freeze_version

    enrollment_identity = str(payload.get("enrollment_identity") or payload.get("record_id") or result["enrollment_identity"])
    if not enrollment_identity:
        raise ValueError("enrollment identity is required to persist")
    enrollment_id = upsert_enrollment_external(
        str(payload.get("student_id") or result.get("student_id")),
        str(payload.get("module") or result.get("module")),
        str(payload.get("presentation") or result.get("presentation")),
        enrollment_identity,
    )
    checksums = (result.get("checksums") or {})
    bundle_id = ensure_bundle(bundle_version, freeze_version, checksums)
    request_key = f"{enrollment_identity}|{result['stage']}|{bundle_version}|{state_version}"
    with transaction() as connection:
        snapshot_id = connection.execute(
            text(
                """
                INSERT INTO recommendation.state_snapshot(
                    enrollment_id,stage,state_version,case_id,risk_probability,
                    inactive_streak,active_days_ratio,recent_activity,activity_trend,
                    assessment_completion,missing_assessments,quiz_activity,vle_available,features,source_lineage
                ) VALUES (
                    :e,:g,:v,:c,:r,:i,:a,:n,:t,:ac,:m,:q,:vl,:f,:s
                )
                ON CONFLICT (enrollment_id,stage,state_version) DO UPDATE SET
                    case_id=EXCLUDED.case_id,
                    risk_probability=EXCLUDED.risk_probability,
                    features=EXCLUDED.features
                RETURNING snapshot_id
                """
            ),
            {
                "e": enrollment_id,
                "g": result["stage"],
                "v": state_version,
                "c": result.get("case_id"),
                "r": result["risk_probability"],
                "i": payload.get("inactive_streak"),
                "a": payload.get("active_days_ratio"),
                "n": payload.get("recent_activity"),
                "t": payload.get("activity_trend"),
                "ac": payload.get("assessment_completion"),
                "m": payload.get("missing_assessments"),
                "q": payload.get("quiz_activity"),
                "vl": payload.get("vle_available"),
                "f": _json({key: payload.get(key) for key in (
                    "inactive_streak", "active_days_ratio", "recent_activity", "activity_trend",
                    "assessment_completion", "missing_assessments", "quiz_activity", "vle_available",
                )}),
                "s": _json({"bundle_version": bundle_version}),
            },
        ).scalar_one()
        run_id = connection.execute(
            text(
                """
                INSERT INTO recommendation.run(enrollment_id,snapshot_id,bundle_id,stage,request_key,plan_status,risk_probability)
                VALUES (:e,:s,:b,:g,:k,:p,:r)
                ON CONFLICT (request_key) DO UPDATE SET
                    plan_status=EXCLUDED.plan_status,
                    risk_probability=EXCLUDED.risk_probability
                RETURNING run_id
                """
            ),
            {
                "e": enrollment_id,
                "s": snapshot_id,
                "b": bundle_id,
                "g": result["stage"],
                "k": request_key,
                "p": result["plan_status"],
                "r": result["risk_probability"],
            },
        ).scalar_one()
        action_ids = {
            row.action_key: row.action_id
            for row in connection.execute(text("SELECT action_id, action_key FROM recommendation.action")).all()
        }
        for action in result["actions"]:
            aid = action_ids[action["action_id"]]
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation.score(run_id,action_id,raw_score,relevance_score,rank,feasibility_status,release_status,quality_warning,model_version)
                    VALUES (:r,:a,:w,:v,:n,:f,:l,:q,:m)
                    ON CONFLICT (run_id,action_id) DO UPDATE SET
                        raw_score=EXCLUDED.raw_score,
                        relevance_score=EXCLUDED.relevance_score,
                        rank=EXCLUDED.rank,
                        feasibility_status=EXCLUDED.feasibility_status,
                        release_status=EXCLUDED.release_status,
                        quality_warning=EXCLUDED.quality_warning
                    """
                ),
                {
                    "r": run_id,
                    "a": aid,
                    "w": action["raw_score"],
                    "v": action["relevance_score"],
                    "n": action["rank"],
                    "f": action["feasibility_status"],
                    "l": action["release_status"],
                    "q": action["quality_warning"],
                    "m": action.get("model_version") or bundle_version,
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation.explanation(run_id,action_id,intercept,top_positive,top_negative,payload,explanation_version)
                    VALUES (:r,:a,:i,:p,:n,:x,:v)
                    ON CONFLICT (run_id,action_id) DO UPDATE SET
                        intercept=EXCLUDED.intercept,
                        top_positive=EXCLUDED.top_positive,
                        top_negative=EXCLUDED.top_negative,
                        payload=EXCLUDED.payload
                    """
                ),
                {
                    "r": run_id,
                    "a": aid,
                    "i": action.get("intercept"),
                    "p": _json(action.get("top_positive_reasons") or []),
                    "n": _json(action.get("top_negative_reasons") or []),
                    "x": _json({"raw_score": action["raw_score"], "relevance_score": action["relevance_score"]}),
                    "v": "ebm_local_v1",
                },
            )
        top_keys = [item["action_id"] for item in result.get("top_actions") or []]
        connection.execute(
            text(
                """
                INSERT INTO recommendation.plan(run_id,plan_status,top_action_keys)
                VALUES (:r,:p,:t)
                ON CONFLICT (run_id) DO UPDATE SET plan_status=EXCLUDED.plan_status, top_action_keys=EXCLUDED.top_action_keys
                """
            ),
            {"r": run_id, "p": result["plan_status"], "t": top_keys},
        )
        return str(run_id)


def fetch_plan(run_id: str) -> dict | None:
    with engine.connect() as connection:
        run = connection.execute(
            text("SELECT * FROM recommendation.run WHERE run_id=:r"),
            {"r": run_id},
        ).mappings().one_or_none()
        if not run:
            return None
        scores = connection.execute(
            text(
                "SELECT a.action_key, s.* FROM recommendation.score s "
                "JOIN recommendation.action a ON a.action_id=s.action_id WHERE s.run_id=:r ORDER BY s.rank"
            ),
            {"r": run_id},
        ).mappings().all()
        return {"run": dict(run), "scores": [dict(row) for row in scores]}
