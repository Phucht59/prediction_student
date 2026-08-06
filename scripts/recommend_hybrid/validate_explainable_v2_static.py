"""Static protocol and contract validation for explainable recommendation V2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.explainable_v2 import (
    CanonicalAction,
    ExplainableRecommendationPipeline,
    FixedActionRanker,
    RecommendationFeatures,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
)


def main() -> int:
    config = yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/explainable_v2.yaml").read_text(encoding="utf-8")
    )
    catalog = yaml.safe_load(
        (ROOT / "configs/recommend_hybrid/actions_v2.yaml").read_text(encoding="utf-8")
    )

    expected_actions = [action.value for action in CanonicalAction]
    configured_actions = list(config["canonical_actions"])
    catalog_actions = [row["action_id"] for row in catalog["actions"]]
    if configured_actions != expected_actions or catalog_actions != expected_actions:
        raise RuntimeError("canonical action order differs across code and configuration")
    if config["runtime_authorized"] is not False:
        raise RuntimeError("V2 must remain runtime_authorized=false")
    if config["ranker"]["action_id_as_feature"] is not False:
        raise RuntimeError("raw action identity is forbidden as a Five-EBM feature")
    if config["risk_authority"]["architecture_mutation_allowed"] is not False:
        raise RuntimeError("frozen Hybrid architecture mutation is forbidden")
    if config["risk_authority"]["checkpoint_replacement_allowed"] is not False:
        raise RuntimeError("frozen Hybrid checkpoint replacement is forbidden")

    risk = RiskThresholds(0.35, 0.65, 0.40, 0.10)
    safety = SafetyThresholds(0.60, 0.10, 0.40, 0.10, 0.30, 0.95)
    scores = {
        CanonicalAction.ASSESSMENT_COMPLETION: 0.90,
        CanonicalAction.RECOVER_ENGAGEMENT: 0.70,
        CanonicalAction.STUDY_REGULARITY: 0.55,
        CanonicalAction.TARGETED_CONTENT_REVIEW: 0.40,
        CanonicalAction.QUIZ_RETRIEVAL_PRACTICE: 0.30,
    }
    pipeline = ExplainableRecommendationPipeline(
        FixedActionRanker(scores), risk, safety, top_k=3
    )
    fixture = RecommendationFeatures(
        student_key="static-validation",
        course_key="AAA:2014J",
        stage=Stage.EARLY_35,
        cutoff_day=40,
        risk_probability=0.80,
        hybrid_uncertainty=0.10,
        seed_disagreement=0.02,
        course_progress=0.35,
        assessment_progress=0.30,
        assessments_due=2,
        assessment_window_open=True,
        time_to_deadline_days=10,
        inactivity_streak=8,
        active_day_rate=0.20,
        recent_activity_trend=-0.40,
        regularity_score=0.25,
        content_coverage=0.40,
        knowledge_gap_evidence=True,
        quiz_activity=0.10,
        quiz_available=True,
        vle_access_available=True,
        study_material_available=True,
        label_conflict=0.10,
        ood_score=0.20,
    )
    decision = pipeline.recommend(fixture)
    if decision.route is not RouteStatus.RECOMMEND:
        raise RuntimeError("valid high-risk fixture must produce RECOMMEND")

    result = {
        "status": "PASS",
        "protocol_status": config["protocol_status"],
        "canonical_actions": expected_actions,
        "frozen_hybrid_mutated": False,
        "runtime_authorized": False,
        "fixture_route": decision.route.value,
        "fixture_top_actions": [item.action.value for item in decision.ranked_actions],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
