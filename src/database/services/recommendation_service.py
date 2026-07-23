from __future__ import annotations

from ..contracts import ReviewType
from ..connection import DatabaseSettings, transaction
from ..repositories.recommendation_repository import RecommendationRepository


class RecommendationService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def plan_for_record(self, source_record_id: str) -> dict | None:
        with transaction(self.settings) as connection:
            return RecommendationRepository(connection).plan_for_record(source_record_id)

    def add_advisor_review(
        self,
        *,
        plan_id: str,
        reviewer_key: str,
        status: str,
        decision: str | None = None,
        reason: str | None = None,
    ) -> int:
        with transaction(self.settings) as connection:
            return RecommendationRepository(connection).add_review(
                plan_id=plan_id,
                review_type=ReviewType.ADVISOR.value,
                reviewer_key=reviewer_key,
                status=status,
                decision=decision,
                reason=reason,
            )


__all__ = ["RecommendationService"]

