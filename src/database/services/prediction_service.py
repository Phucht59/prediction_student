from __future__ import annotations

from ..connection import DatabaseSettings, transaction
from ..repositories.prediction_repository import PredictionRepository


class PredictionService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def predict(self, source_record_id: str, prediction_stage: str) -> list[dict]:
        with transaction(self.settings) as connection:
            return PredictionRepository(connection).predict(
                source_record_id, prediction_stage
            )


__all__ = ["PredictionService"]
