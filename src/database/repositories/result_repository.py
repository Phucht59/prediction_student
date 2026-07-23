from __future__ import annotations

from psycopg2.extras import RealDictCursor

from .base import Repository


class ResultRepository(Repository):
    def metrics(self, run_id: str) -> list[dict]:
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT metric_name,metric_value,scope,aggregation,class_label,
                       budget,fold,seed,unit,detail
                FROM ml.metric WHERE run_id=%s
                ORDER BY scope,metric_name,class_label,budget,fold,seed
                """,
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def artifacts(self, run_id: str) -> list[dict]:
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT artifact_kind,storage_path,sha256,byte_count,row_count,
                       media_type,metadata
                FROM ml.artifact WHERE run_id=%s
                ORDER BY artifact_kind,storage_path
                """,
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]


__all__ = ["ResultRepository"]
