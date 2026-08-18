from __future__ import annotations

from psycopg2.extras import RealDictCursor

from ..models import ModelResult
from .base import Repository


class ModelRepository(Repository):
    def final_results(
        self,
        dataset_slug: str | None = None,
        prediction_stage: str | None = None,
    ) -> list[ModelResult]:
        view = "ml.stage_model_results" if prediction_stage is not None else "ml.final_model_results"
        stage_clause = "AND prediction_stage=%s" if prediction_stage is not None else ""
        statement = """
            SELECT * FROM {view}
            WHERE (%s IS NULL OR dataset=%s)
              {stage_clause}
            ORDER BY dataset,is_selected DESC,model_key
        """.format(view=view, stage_clause=stage_clause)
        parameters = (
            (dataset_slug, dataset_slug, prediction_stage)
            if prediction_stage is not None
            else (dataset_slug, dataset_slug)
        )
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(statement, parameters)
            rows = cursor.fetchall()
        metric_names = (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "macro_f1",
            "pr_auc",
            "roc_auc",
            "brier",
            "nll",
            "ece",
        )
        return [
            ModelResult(
                dataset=row["dataset"],
                model_key=row["model_key"],
                official_name=row["official_name"],
                is_selected=row["is_selected"],
                metrics={name: row.get(name) for name in metric_names},
            )
            for row in rows
        ]


__all__ = ["ModelRepository"]
