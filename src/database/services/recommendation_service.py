from __future__ import annotations

from ..connection import DatabaseSettings, transaction
from ..repositories import RecommendationRepository


class RecommendationService:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings

    def create_revision(self, *, case_id: int, revision_no: int, goal: str, rationale: str, supersedes_plan_id: int | None = None) -> int:
        with transaction(self.settings) as connection:
            return RecommendationRepository(connection).create_plan(
                case_id, revision_no, goal, rationale, supersedes_plan_id
            )


__all__ = ["RecommendationService"]

