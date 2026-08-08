"""Production recommendation namespace tests."""

from src import recommend_hybrid
from src.recommend_hybrid import final


def test_public_api_points_to_released_pipeline() -> None:
    assert recommend_hybrid.RecommendationPipeline is final.ExplainableRecommendationPipeline
    assert final.RecommendationPipeline is final.ExplainableRecommendationPipeline


def test_final_route_status_contract() -> None:
    assert {status.value for status in final.RouteStatus} == {
        "RECOMMEND",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW",
        "NO_FEASIBLE_ACTION",
    }


def test_legacy_conditional_ranker_is_not_public_authority() -> None:
    assert not hasattr(recommend_hybrid, "ConditionalHybridActionRanker")
    assert not hasattr(final, "ConditionalHybridActionRanker")
