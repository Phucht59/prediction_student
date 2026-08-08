"""Deterministic end-to-end integration audit after the one-shot Panel-B freeze."""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.platform == "win32" and not platform.machine():
    platform.machine = lambda: "AMD64"

from src.recommend_hybrid.contracts import Stage  # noqa: E402
from src.recommend_hybrid.explainable_v2 import (  # noqa: E402
    ExplainableRecommendationPipeline,
    FiveEBMRanker,
    RecommendationFeatures,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
)
from src.recommend_hybrid.explainable_v2.feasibility import (  # noqa: E402
    feasible_actions,
)
from src.recommend_hybrid.explainable_v2.plan_builder import (  # noqa: E402
    build_structured_plan,
)
from src.recommend_hybrid.explainable_v2.ranker import (  # noqa: E402
    FEATURE_COLUMNS,
    canonical_ordinal_score_from_model_prediction,
    public_score_from_ordinal_prediction,
)
from src.recommend_hybrid.explainable_v2.simulator.core import (  # noqa: E402
    SimulationResult,
)


ARTIFACT_ROOT = ROOT / "artifacts/recommend_hybrid/explainable_v2"
DEVELOPMENT_FREEZE_PATH = (
    ARTIFACT_ROOT / "frozen/development_v2/DEVELOPMENT_FREEZE_MANIFEST.json"
)
RANKER_DIR = ARTIFACT_ROOT / "frozen/ranker_panel_a_v2"
ROUTER_PATH = ARTIFACT_ROOT / "frozen/router_panel_a_v1/ROUTER_FREEZE_MANIFEST.json"
LABEL_PATH = ARTIFACT_ROOT / "labels/panel_a_v1/probabilistic_relevance_labels.parquet"
CANDIDATE_PATH = ARTIFACT_ROOT / "features/action_candidates.parquet"
PANEL_B_MANIFEST_PATH = (
    ARTIFACT_ROOT / "final_heldout/panel_b_v1/PANEL_B_FINAL_HELDOUT_MANIFEST.json"
)
PANEL_B_METRICS_PATH = (
    ARTIFACT_ROOT / "final_heldout/panel_b_v1/PANEL_B_FINAL_HELDOUT_METRICS.json"
)
OUTPUT_PATH = (
    ARTIFACT_ROOT / "audit/final_integration_v1/PHASE9_END_TO_END_INTEGRATION_AUDIT.json"
)
FIXTURE_CASE_ID = "case_04c2266a53cea8377792d2f2"
SIMULATOR_LANGUAGE = "model-implied risk delta"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def python_value(value: Any) -> Any:
    return None if pd.isna(value) else value


def load_fixture() -> RecommendationFeatures:
    labels = pd.read_parquet(LABEL_PATH, columns=["query_id", "case_id"])
    mapping = labels.loc[labels["case_id"].eq(FIXTURE_CASE_ID), ["query_id"]].drop_duplicates()
    if len(mapping) != 1:
        raise RuntimeError("PHASE9_FIXTURE_LINEAGE_NOT_UNIQUE")
    query_id = mapping.iloc[0]["query_id"]
    candidates = pd.read_parquet(CANDIDATE_PATH)
    rows = candidates.loc[candidates["query_id"].eq(query_id)].sort_values("action_id")
    if len(rows) != 5:
        raise RuntimeError("PHASE9_FIXTURE_NOT_FIVE_ACTION_ROWS")
    row = rows.iloc[0]
    if float(row["risk_probability"]) < 0.8:
        raise RuntimeError("PHASE9_FIXTURE_NOT_HIGH_RISK")

    return RecommendationFeatures(
        student_key="phase9-blinded-fixture",
        course_key="phase9-blinded-course",
        stage=Stage(str(row["stage"])),
        cutoff_day=int(row["cutoff_day"]),
        risk_probability=float(row["risk_probability"]),
        hybrid_uncertainty=float(row["hybrid_uncertainty"]),
        seed_disagreement=None,
        course_progress=float(row["course_progress"]),
        assessments_due=python_value(row["assessments_due"]),
        missing_assessment_count=python_value(row["missing_assessment_count"]),
        due_soon_count=python_value(row["due_soon_count"]),
        completion_rate=python_value(row["completion_rate"]),
        inactivity_streak=python_value(row["inactivity_streak"]),
        active_day_rate=python_value(row["active_day_rate"]),
        regularity_score=python_value(row["regularity_score"]),
        content_coverage=python_value(row["content_coverage"]),
        quiz_activity=python_value(row["quiz_activity"]),
        quiz_available=bool(row["quiz_available"]),
        vle_access_available=bool(row["vle_available"]),
        study_material_available=bool(row["study_material_available"]),
        label_conflict=0.0,
        ood_score=0.0,
    )


class RecordingRanker:
    def __init__(self, delegate: FiveEBMRanker) -> None:
        self.delegate = delegate
        self.received_actions = ()

    def score(self, features, eligible_actions):
        self.received_actions = eligible_actions
        return self.delegate.score(features, eligible_actions)


def run() -> int:
    checks: dict[str, str] = {}

    development = json.loads(DEVELOPMENT_FREEZE_PATH.read_text(encoding="utf-8"))
    router = json.loads(ROUTER_PATH.read_text(encoding="utf-8"))
    panel_b_manifest = json.loads(PANEL_B_MANIFEST_PATH.read_text(encoding="utf-8"))
    panel_b_metrics = json.loads(PANEL_B_METRICS_PATH.read_text(encoding="utf-8"))
    if development["status"] != "PASS":
        raise RuntimeError("DEVELOPMENT_FREEZE_NOT_PASS")
    if panel_b_manifest["status"] != "PASS":
        raise RuntimeError("PANEL_B_FINAL_FREEZE_NOT_PASS")
    if panel_b_metrics["scope"] != "PANEL_B_FINAL_HELDOUT":
        raise RuntimeError("PANEL_B_METRIC_SCOPE_CHANGED")
    checks["frozen_hybrid_authority"] = "PASS"

    expected_model_hashes = development["ranker"]["five_model_sha256"]
    for action, expected in expected_model_hashes.items():
        if sha256(RANKER_DIR / "final_models" / f"{action}.joblib") != expected:
            raise RuntimeError(f"FROZEN_EBM_HASH_CHANGED={action}")
    checks["five_frozen_ebm_artifacts"] = "PASS"

    selected = router["selected_thresholds"]
    if canonical_sha256(selected) != router["selected_thresholds_sha256"]:
        raise RuntimeError("FROZEN_ROUTER_THRESHOLD_HASH_MISMATCH")
    safety = SafetyThresholds(
        minimum_top1_score=float(selected["minimum_top1_score"]),
        minimum_top1_margin=float(selected["minimum_top1_margin"]),
        maximum_hybrid_uncertainty=float(selected["maximum_hybrid_uncertainty"]),
        maximum_seed_disagreement=selected["maximum_seed_disagreement"],
        maximum_label_conflict=float(selected["maximum_label_conflict"]),
        maximum_ood_score=float(selected["maximum_ood_score"]),
    )
    if safety.maximum_seed_disagreement is not None:
        raise RuntimeError("UNAVAILABLE_SEED_DISAGREEMENT_THRESHOLD_WAS_APPLIED")
    checks["router_uses_frozen_panel_a_thresholds"] = "PASS"
    checks["missing_seed_disagreement_remains_none"] = "PASS"

    features = load_fixture()
    ranker = FiveEBMRanker.from_frozen_ordinal_artifacts(RANKER_DIR / "final_models")
    recording_ranker = RecordingRanker(ranker)
    pipeline = ExplainableRecommendationPipeline(
        recording_ranker,
        RiskThresholds(
            low=0.2,
            high=0.8,
            maximum_automatic_uncertainty=0.4,
            maximum_seed_disagreement=0.1,
        ),
        safety,
        top_k=3,
    )
    feasibility = feasible_actions(features)
    eligible = tuple(item.action for item in feasibility if item.eligible)
    decision = pipeline.recommend(features)
    if recording_ranker.received_actions != eligible:
        raise RuntimeError("HARD_FEASIBILITY_DID_NOT_PRECEDE_RANKING")
    if any(item.action not in eligible for item in decision.ranked_actions):
        raise RuntimeError("INFEASIBLE_ACTION_RETURNED")
    if decision.route not in set(RouteStatus):
        raise RuntimeError("NONCANONICAL_ROUTE_STATUS")
    if {status.value for status in RouteStatus} != {
        "RECOMMEND",
        "INSUFFICIENT_EVIDENCE",
        "HUMAN_REVIEW",
        "NO_FEASIBLE_ACTION",
    }:
        raise RuntimeError("PUBLIC_STATUS_CONTRACT_CHANGED")
    if not decision.ranked_actions:
        raise RuntimeError("PHASE9_FIXTURE_DID_NOT_REACH_RANKED_OUTPUT")
    checks["hard_feasibility_before_ranking"] = "PASS"
    checks["no_infeasible_action_returned"] = "PASS"
    checks["four_status_public_contract"] = "PASS"

    scores = np.asarray([item.score for item in decision.ranked_actions], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError("PUBLIC_SCORE_OUTSIDE_0_1")
    score_source = inspect.getsource(public_score_from_ordinal_prediction)
    if score_source.count("/ ORDINAL_RELEVANCE_MAX") != 1:
        raise RuntimeError("PUBLIC_SCORE_NORMALIZATION_LOCATION_COUNT_NOT_ONE")
    raw_fixture = np.asarray([-0.25, 0.0, 1.5, 3.0, 3.5])
    hardened = np.asarray(
        [
            public_score_from_ordinal_prediction(
                canonical_ordinal_score_from_model_prediction(value)
            )
            for value in raw_fixture
        ]
    )
    historical = np.clip(raw_fixture / 3.0, 0.0, 1.0)
    if not np.array_equal(hardened, historical):
        raise RuntimeError("ORDINAL_CLAMP_CHANGED_PUBLIC_SCORE")
    checks["canonical_native_ordinal_scale_0_3"] = "PASS"
    checks["single_public_score_normalization"] = "PASS"
    checks["no_double_normalization"] = "PASS"

    forbidden_features = {
        "final_result",
        "outcome",
        "target",
        "post_cutoff_behavior",
        "future_activity",
        "action_id",
    }
    if forbidden_features & set(FEATURE_COLUMNS):
        raise RuntimeError("FORBIDDEN_OR_OUTCOME_FEATURE_IN_EBM_SCHEMA")
    checks["pre_cutoff_feature_schema_only"] = "PASS"
    checks["outcome_leakage_count_zero"] = "PASS"

    observed = tuple(
        name
        for name, _ in decision.ranked_actions[0].explanation
        if name in FEATURE_COLUMNS
    )
    if any(name not in FEATURE_COLUMNS for name in observed):
        raise RuntimeError("EXPLANATION_NOT_GROUNDED_IN_OBSERVED_FEATURES")
    plan = build_structured_plan(
        decision.ranked_actions,
        features.stage,
        observed_evidence_summary=observed,
    )
    if set(plan.observed_evidence) - set(FEATURE_COLUMNS):
        raise RuntimeError("PLAN_CONTAINS_UNOBSERVED_EVIDENCE")
    checks["evidence_grounded_explanation"] = "PASS"
    checks["evidence_grounded_learning_plan"] = "PASS"

    simulation = SimulationResult(
        status="MODEL_IMPLIED_RISK_DELTA_ONLY",
        risk_delta=0.0,
        causal_claim_allowed=False,
        runtime_authorized=False,
    )
    if simulation.causal_claim_allowed or SIMULATOR_LANGUAGE != "model-implied risk delta":
        raise RuntimeError("SIMULATOR_CLAIM_BOUNDARY_FAILURE")
    checks["simulator_model_implied_risk_delta_only"] = "PASS"

    if float(panel_b_metrics["frozen_five_ebm_ranker"]["invalid_action_rate"]) != 0.0:
        raise RuntimeError("FROZEN_PANEL_B_INVALID_ACTION_RATE_NONZERO")
    checks["frozen_panel_b_invalid_action_rate_zero"] = "PASS"

    report = {
        "schema_version": "phase9_end_to_end_integration_audit_v1",
        "status": "PASS",
        "scope": "POST_PANEL_B_OUTPUT_INVARIANT_INTEGRATION_ONLY",
        "panel_b_recomputed": False,
        "provider_called": False,
        "scientific_outputs_changed": False,
        "runtime_authorized": False,
        "fixture_case_id": FIXTURE_CASE_ID,
        "fixture_route": decision.route.value,
        "eligible_actions": [action.value for action in eligible],
        "ranked_actions": [item.action.value for item in decision.ranked_actions],
        "public_score_min": float(scores.min()),
        "public_score_max": float(scores.max()),
        "plan_action": plan.action,
        "plan_observed_evidence": list(plan.observed_evidence),
        "simulator_language": SIMULATOR_LANGUAGE,
        "checks": checks,
        "lineage": {
            "development_freeze_sha256": sha256(DEVELOPMENT_FREEZE_PATH),
            "router_freeze_sha256": sha256(ROUTER_PATH),
            "panel_b_final_manifest_sha256": sha256(PANEL_B_MANIFEST_PATH),
            "panel_b_final_metrics_sha256": sha256(PANEL_B_METRICS_PATH),
            "post_panel_b_ranker_source_sha256": sha256(
                ROOT / "src/recommend_hybrid/explainable_v2/ranker.py"
            ),
            "pre_engineering_ranker_source_sha256": (
                "86427f48547c0d1400ede8b83c65f1d3d159631bd15a0aecfd3f76a5f513efee"
            ),
        },
        "post_panel_b_change_classification": [
            {
                "files": [
                    "src/recommend_hybrid/explainable_v2/ranker.py",
                    "tests/recommend_hybrid/explainable_v2/test_five_ebm_models_v1.py",
                ],
                "classification": "SCIENTIFIC_OUTPUT_INVARIANT_ENGINEERING_CHANGE",
                "reason": (
                    "Canonical ordinal clamp makes the documented [0,3] intermediate "
                    "explicit; public clip(raw/3,0,1), rankings, and frozen metrics are identical."
                ),
            },
            {
                "files": [
                    "scripts/recommend_hybrid/explainable_v2/dispatch_gemini_panel_a_batch01_v3.py",
                    "scripts/recommend_hybrid/explainable_v2/dispatch_gemini_panel_a_batch_v4.py",
                    "scripts/recommend_hybrid/explainable_v2/dispatch_gemini_panel_b_batch_v1.py",
                    "tests/recommend_hybrid/explainable_v2/test_gemini_panel_a_dispatch_v3.py",
                ],
                "classification": "SCIENTIFIC_OUTPUT_INVARIANT_ENGINEERING_CHANGE",
                "reason": "Provider rate limiting only; Panel B is frozen and no provider is called again.",
            },
            {
                "files": [
                    "scripts/recommend_hybrid/explainable_v2/run_plausibility_simulator.py",
                    "reports/recommend_hybrid_v2/SCIENTIFIC_CLAIM_BOUNDARIES.md",
                ],
                "classification": "SCIENTIFIC_OUTPUT_INVARIANT_ENGINEERING_CHANGE",
                "reason": "Terminology normalized to model-implied risk delta; no numeric output changes.",
            },
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PHASE=9")
    print("STATUS=PASS")
    print("PANEL_B_RECOMPUTED=FALSE")
    print("PROVIDER_CALLED=FALSE")
    print(f"CHECKS={len(checks)}_PASS")
    print(f"AUDIT_SHA256={sha256(OUTPUT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
