from __future__ import annotations

from psycopg2.extras import Json

from .base import Repository


class EvaluationRepository(Repository):
    def register_prediction_set(self, training_run_id: int, scope: str, aggregation: str, path: str | None, sha256: str | None, row_count: int, probability_schema: dict) -> int:
        return int(
            self.scalar(
                """INSERT INTO evaluation.prediction_set(training_run_id,scope,aggregation,storage_path,sha256,row_count,probability_schema)
                   VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING prediction_set_id""",
                (training_run_id, scope, aggregation, path, sha256, row_count, Json(probability_schema)),
            )
        )

    def register_metric(self, training_run_id: int, prediction_set_id: int | None, name: str, value: float, scope: str, aggregation: str, fold: int | None, seed: int | None, class_label: str | None, detail: dict | None = None) -> int:
        return int(
            self.scalar(
                """INSERT INTO evaluation.metric(training_run_id,prediction_set_id,metric_name,metric_value,scope,aggregation,fold,seed,class_label,detail)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING metric_id""",
                (training_run_id, prediction_set_id, name, value, scope, aggregation, fold, seed, class_label, Json(detail or {})),
            )
        )


__all__ = ["EvaluationRepository"]

