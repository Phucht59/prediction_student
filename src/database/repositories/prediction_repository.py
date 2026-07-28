from __future__ import annotations

from psycopg2.extras import RealDictCursor

from .base import Repository


class PredictionRepository(Repository):
    def predict(self, source_record_id: str, prediction_stage: str) -> list[dict]:
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT d.slug AS dataset,m.model_key,m.official_name,m.is_selected,
                       p.prediction_stage,p.predicted_label,p.probabilities,p.outer_fold
                FROM ml.prediction p
                JOIN ml.run r ON r.run_id=p.run_id
                JOIN ml.model m ON m.model_id=r.model_id
                JOIN catalog.dataset d ON d.dataset_id=m.dataset_id
                JOIN catalog.record cr ON cr.record_pk=p.record_pk
                WHERE cr.source_record_id=%s AND p.prediction_stage=%s
                ORDER BY m.is_selected DESC,m.model_key
                """,
                (source_record_id, prediction_stage),
            )
            return [dict(row) for row in cursor.fetchall()]


__all__ = ["PredictionRepository"]
