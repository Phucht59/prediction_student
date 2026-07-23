from __future__ import annotations

from ..connection import DatabaseSettings, transaction


class MigrationValidationService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def core_counts(self) -> dict[str, int]:
        with transaction(self.settings) as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                  (SELECT count(*) FROM catalog.dataset),
                  (SELECT count(*) FROM ml.model),
                  (SELECT count(*) FROM recommendation.risk_profile),
                  (SELECT count(*) FROM recommendation.plan),
                  (SELECT count(*) FROM recommendation.action)
                """
            )
            datasets, models, risk_profiles, plans, actions = cursor.fetchone()
        return {
            "datasets": datasets,
            "models": models,
            "risk_profiles": risk_profiles,
            "plans": plans,
            "actions": actions,
        }


__all__ = ["MigrationValidationService"]
