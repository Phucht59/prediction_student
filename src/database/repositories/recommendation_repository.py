from __future__ import annotations

from psycopg2.extras import Json, RealDictCursor

from .base import Repository


class RecommendationRepository(Repository):
    def plan_for_record(
        self, source_record_id: str, prediction_stage: str = "F2_MIDDLE"
    ) -> dict | None:
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT ps.*,p.payload
                FROM recommendation.plan_summary ps
                JOIN recommendation.plan p USING(plan_id)
                WHERE ps.source_record_id=%s AND ps.prediction_stage=%s
                """,
                (source_record_id, prediction_stage),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def add_review(
        self,
        *,
        plan_id: str,
        review_type: str,
        reviewer_key: str,
        status: str,
        decision: str | None = None,
        reason: str | None = None,
        payload: dict | None = None,
    ) -> int:
        return int(
            self.scalar(
                """
                INSERT INTO recommendation.review(
                    plan_id,review_type,reviewer_key,status,decision,reason,payload
                ) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING review_id
                """,
                (
                    plan_id,
                    review_type,
                    reviewer_key,
                    status,
                    decision,
                    reason,
                    Json(payload or {}),
                ),
            )
        )


__all__ = ["RecommendationRepository"]

