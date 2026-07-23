from __future__ import annotations

from ..connection import DatabaseSettings, transaction
from ..repositories.evaluation_repository import EvaluationRepository


class PredictionService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def register_metric_bundle(self, *, training_run_id: int, prediction_set_id: int | None, scope: str, aggregation: str, metrics: dict[str, float], fold: int | None = None, seed: int | None = None) -> list[int]:
        with transaction(self.settings) as connection:
            repository = EvaluationRepository(connection)
            return [
                repository.register_metric(training_run_id, prediction_set_id, name, float(value), scope, aggregation, fold, seed, None)
                for name, value in metrics.items()
            ]


__all__ = ["PredictionService"]

