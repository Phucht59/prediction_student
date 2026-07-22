"""Read-only facade over a frozen recommendation implementation."""

from __future__ import annotations

from typing import Any


class StudentRiskRecommendationSystem:
    system_id = "student_risk_recommendation_system"
    official_name = "Student Risk-Based Recommendation System"

    def __init__(self, frozen_engine: Any) -> None:
        self.frozen_engine = frozen_engine

    def recommend(self, risk_profile: Any) -> Any:
        return self.frozen_engine.recommend(risk_profile)
