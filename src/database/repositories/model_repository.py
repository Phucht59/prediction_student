from __future__ import annotations

from psycopg2.extras import RealDictCursor

from ..models import ModelResult
from .base import Repository


class ModelRepository(Repository):
    def final_results(self, dataset_slug: str | None = None) -> list[ModelResult]:
        statement = """
            SELECT * FROM ml.final_model_results
            WHERE (%s IS NULL OR dataset=%s)
            ORDER BY dataset,is_selected DESC,model_key
        """
        with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(statement, (dataset_slug, dataset_slug))
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
                metrics={name: row[name] for name in metric_names},
            )
            for row in rows
        ]


__all__ = ["ModelRepository"]
