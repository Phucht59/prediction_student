from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.recommendation.v5_2.engine import build_recommendation, validate_recommendation

from .contract import ARTIFACT_ROOT, REPORT_ROOT, atomic_json, atomic_text
from .decision_policy import POLICY_THRESHOLDS, POLICY_VERSION, apply_decision_policy
from .risk_profile import SCHEMA_VERSION, validate_risk_profile


RECOMMENDATION_ROOT = ARTIFACT_ROOT / "recommendation"
ENGINE_VERSION = "v5_2"
PLAN_SCHEMA = "recommendation_plan_v2"
FIXED_TIME = "2026-07-21T00:00:00+00:00"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def recommendation_input(
    profile: dict[str, Any], *, current_state_cutoff_day: int | None = None
) -> dict[str, Any]:
    validate_risk_profile(profile)
    if current_state_cutoff_day is not None and current_state_cutoff_day > profile["cutoff_day"]:
        raise ValueError("Stale risk profile: a newer student state exists")
    return {
        "schema_version": "recommendation_input_v2",
        "student_risk_profile": profile,
        "student_learning_state": {
            "activity_level": float(1 - profile["withdrawal_risk_horizon"]),
            "inactivity_streak": int(round(profile["withdrawal_risk_horizon"] * 10)),
            "assessment_progress": float(1 - profile["probability_fail"]),
            "grade_trend": float(
                profile["probability_pass"]
                + profile["probability_distinction"]
                - profile["probability_fail"]
            ),
        },
        "previous_plan": None,
        "advisor_constraints": {
            "max_actions": POLICY_THRESHOLDS["workload"]["max_actions"],
            "max_weekly_minutes": POLICY_THRESHOLDS["workload"]["max_weekly_minutes"],
        },
        "policy_version": POLICY_VERSION,
        "recommendation_engine_version": ENGINE_VERSION,
    }


def validate_recommendation_input(value: dict[str, Any]) -> None:
    if value.get("schema_version") != "recommendation_input_v2":
        raise ValueError("Recommendation input schema mismatch")
    if value.get("policy_version") != POLICY_VERSION:
        raise ValueError("Recommendation policy mismatch")
    if value.get("recommendation_engine_version") != ENGINE_VERSION:
        raise ValueError("Recommendation engine mismatch")
    validate_risk_profile(value["student_risk_profile"])


def _workload_guard(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    minutes = 0
    for action in actions:
        action_minutes = int(action["weekly_minutes"])
        if len(selected) >= int(POLICY_THRESHOLDS["workload"]["max_actions"]):
            break
        if minutes + action_minutes > int(
            POLICY_THRESHOLDS["workload"]["max_weekly_minutes"]
        ):
            continue
        selected.append(action)
        minutes += action_minutes
    return selected


def validate_plan(plan: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "plan_id",
        "record_id",
        "risk_profile_lineage_id",
        "plan_version",
        "plan_status",
        "risk_level",
        "risk_mechanism",
        "priority",
        "recommended_actions",
        "reason_codes",
        "expected_weekly_minutes",
        "monitoring_horizon_days",
        "requires_expert_review",
        "escalation_reason",
        "prediction_model_version",
        "recommendation_engine_version",
        "policy_version",
        "generated_at",
        "supersedes_plan_id",
    }
    missing = required - set(plan)
    if missing:
        raise ValueError(f"Plan fields missing: {sorted(missing)}")
    actions = plan["recommended_actions"]
    identifiers = [action["action_id"] for action in actions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate recommendation action")
    if any(not action.get("reason_codes") for action in actions):
        raise ValueError("Recommendation action missing reason")
    if len(actions) > POLICY_THRESHOLDS["workload"]["max_actions"]:
        raise ValueError("Recommendation action cap exceeded")
    if plan["expected_weekly_minutes"] > POLICY_THRESHOLDS["workload"][
        "max_weekly_minutes"
    ]:
        raise ValueError("Recommendation workload cap exceeded")
    if not plan["risk_profile_lineage_id"]:
        raise ValueError("Recommendation lineage missing")


def generate_plan(profile: dict[str, Any]) -> dict[str, Any]:
    input_value = recommendation_input(profile)
    validate_recommendation_input(input_value)
    policy = apply_decision_policy(profile)
    deep = [1 - profile["probability_at_risk"], profile["probability_at_risk"]]
    ml = [
        1 - profile["ml_cross_check_probability"],
        profile["ml_cross_check_probability"],
    ]
    engine = build_recommendation(
        student_or_enrollment_id=profile["record_id"],
        dataset="oulad",
        prediction_set_id=profile["lineage_id"],
        deep_model_registry_id=profile["model_version"],
        ml_model_registry_id="V3-MLF-operational-cross-check",
        deep_probability=deep,
        ml_probability=ml,
        features=input_value["student_learning_state"],
        input_snapshot_at=profile["generated_at"],
        prediction_created_at=profile["generated_at"],
        created_at=FIXED_TIME,
    )
    validate_recommendation(engine)
    actions = _workload_guard(engine["ranked_actions"])
    if policy["risk_mechanism"] == "UNCERTAIN_RISK":
        advisor = [action for action in actions if action["action_id"] == "ADVISOR_ESCALATION"]
        actions = advisor or actions[:1]
    reason_codes = sorted(
        set(profile["reason_codes"])
        | {reason for action in actions for reason in action["reason_codes"]}
        | set(policy["escalation_reasons"])
    )
    body: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "record_id": profile["record_id"],
        "risk_profile_lineage_id": profile["lineage_id"],
        "plan_version": 1,
        "plan_status": "ACTIVE",
        "risk_level": policy["risk_level"],
        "risk_mechanism": policy["risk_mechanism"],
        "priority": policy["priority"],
        "recommended_actions": actions,
        "reason_codes": reason_codes,
        "expected_weekly_minutes": sum(int(action["weekly_minutes"]) for action in actions),
        "monitoring_horizon_days": 7 if policy["priority"] == "IMMEDIATE" else 14 if policy["priority"] == "HIGH" else 28,
        "requires_expert_review": bool(
            policy["requires_expert_review"] or engine["requires_advisor_review"]
        ),
        "escalation_reason": ",".join(policy["escalation_reasons"]) or None,
        "prediction_model_version": profile["model_version"],
        "recommendation_engine_version": ENGINE_VERSION,
        "policy_version": POLICY_VERSION,
        "generated_at": FIXED_TIME,
        "supersedes_plan_id": None,
        "v5_2_engine_recommendation_id": engine["recommendation_id"],
        "effectiveness": "NOT_ESTABLISHED",
        "causal_claim": "PROHIBITED",
    }
    body["plan_id"] = _hash(body)[:24]
    validate_plan(body)
    return body


def generate_recommendations() -> dict[str, Any]:
    state_path = RECOMMENDATION_ROOT / "run_state.json"
    plans_path = RECOMMENDATION_ROOT / "plans.jsonl"
    if state_path.is_file() and plans_path.is_file():
        cached = json.loads(state_path.read_text(encoding="utf-8"))
        if cached.get("status") == "COMPLETE":
            return cached
    started = time.perf_counter()
    profiles = pd.read_parquet(ARTIFACT_ROOT / "prediction/risk_profiles.parquet")
    plans = [generate_plan(row) for row in profiles.to_dict(orient="records")]
    plans_path.parent.mkdir(parents=True, exist_ok=True)
    plans_path.write_text(
        "".join(json.dumps(plan, ensure_ascii=False, allow_nan=False) + "\n" for plan in plans),
        encoding="utf-8",
    )
    replay = [generate_plan(row) for row in profiles.to_dict(orient="records")]
    conflicts = sum(
        len(plan["recommended_actions"])
        != len({action["action_id"] for action in plan["recommended_actions"]})
        for plan in plans
    )
    workload_violations = sum(
        plan["expected_weekly_minutes"]
        > POLICY_THRESHOLDS["workload"]["max_weekly_minutes"]
        for plan in plans
    )
    missing_lineage = sum(not plan["risk_profile_lineage_id"] for plan in plans)
    duplicate_plans = len(plans) - len({plan["plan_id"] for plan in plans})
    metrics = {
        "schema_version": "v6_recommendation_technical_metrics_v1",
        "status": "PASS"
        if not (conflicts or workload_violations or missing_lineage or duplicate_plans)
        and plans == replay
        else "FAIL",
        "plans_generated": len(plans),
        "coverage": len(plans) / max(1, len(profiles)),
        "escalation_rate": float(np.mean([plan["requires_expert_review"] for plan in plans])),
        "conflicts": conflicts,
        "duplicate_plans": duplicate_plans,
        "workload_violations": workload_violations,
        "missing_lineage": missing_lineage,
        "deterministic_replay": plans == replay,
        "mean_latency_ms": (time.perf_counter() - started) * 1000 / max(1, len(plans)),
        "causal_effectiveness_claimed": False,
    }
    atomic_json(RECOMMENDATION_ROOT / "technical_metrics.json", metrics)
    atomic_json(RECOMMENDATION_ROOT / "action_metrics.json", {
        "status": "PENDING_EXPERT_LABELS",
        "action_precision": None,
        "action_recall": None,
        "action_f1": None,
        "top_3_action_recall": None,
        "plan_approval_rate": None,
        "escalation_f1": None,
        "expert_agreement": None,
    })
    state = {
        **metrics,
        "schema_version": "v6_recommendation_run_state_v1",
        "input_schema": "recommendation_input_v2",
        "profile_schema": SCHEMA_VERSION,
        "plan_schema": PLAN_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "policy_version": POLICY_VERSION,
    }
    atomic_json(state_path, state)
    atomic_text(
        REPORT_ROOT / "RECOMMENDATION_FINAL_REPORT.md",
        f"""# V6 recommendation technical report

The frozen V5.2 engine was adapted to canonical `student_risk_profile_v1`
through `recommendation_input_v2`; its action taxonomy and safeguards were
retained and the V6 versioned risk policy/workload guard was applied.

- Plans: {len(plans)}
- Coverage: {metrics['coverage']:.3%}
- Expert-review/escalation rate: {metrics['escalation_rate']:.3%}
- Conflicts: {conflicts}
- Duplicates: {duplicate_plans}
- Workload violations: {workload_violations}
- Missing lineage: {missing_lineage}
- Deterministic replay: {metrics['deterministic_replay']}
- Technical verdict: **{metrics['status']}**

No student-outcome or causal recommendation-effectiveness claim is made.
""",
    )
    return state


def load_plans(path: Path | None = None) -> list[dict[str, Any]]:
    source = path or RECOMMENDATION_ROOT / "plans.jsonl"
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]


__all__ = [
    "generate_plan",
    "generate_recommendations",
    "load_plans",
    "recommendation_input",
    "validate_plan",
    "validate_recommendation_input",
]
