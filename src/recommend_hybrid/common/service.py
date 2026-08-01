"""Application service for generating, persisting, retrieving and replaying plans."""

from __future__ import annotations

from src.recommend_hybrid.persistence import PlanRepository
from src.recommend_hybrid.pipeline import PlanRequest, RecommendHybridPipeline

from .plan_contracts import LearningPlan


class HybridRecommendationService:
    def __init__(self, pipeline: RecommendHybridPipeline, repository: PlanRepository) -> None:
        self.pipeline = pipeline
        self.repository = repository

    def generate(self, request: PlanRequest, *, dry_run: bool = False) -> LearningPlan:
        plan = self.pipeline.generate(request)
        if not dry_run:
            self.repository.save(plan)
        return plan

    def retrieve(self, plan_id: str) -> LearningPlan | None:
        return self.repository.retrieve(plan_id)

    def replay(self, plan_id: str) -> LearningPlan | None:
        return self.repository.replay(plan_id)


__all__ = ["HybridRecommendationService"]
