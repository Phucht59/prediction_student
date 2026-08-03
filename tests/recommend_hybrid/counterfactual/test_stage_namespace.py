from scripts.recommend_hybrid.evaluate_counterfactual_recommender import (
    STAGES,
)
from scripts.recommend_hybrid.evaluate_historical_trajectories import (
    NEXT_OOF_STAGE,
    STAGE_PATH,
)
from src.recommend_hybrid.contracts import Stage


def test_middle_stage_keeps_bundle_and_reporting_names_separate():
    assert "M1_MIDDLE_FROZEN" in STAGES
    canonical, cutoff, reporting_alias = STAGES["M1_MIDDLE_FROZEN"]
    assert canonical is Stage.MIDDLE_50
    assert cutoff == 50.0
    assert reporting_alias == "M1_MIDDLE_50PCT"

    assert STAGE_PATH["EARLY_35"] == (
        "E2_EARLY_35PCT",
        "M1_MIDDLE_FROZEN",
    )
    assert STAGE_PATH["MIDDLE_50"][0] == "M1_MIDDLE_FROZEN"
    assert NEXT_OOF_STAGE["EARLY_35"] == "M1_MIDDLE_50PCT"
