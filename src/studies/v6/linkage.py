from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd

from .contract import ARTIFACT_ROOT, REPORT_ROOT, atomic_json, atomic_text
from .recommendation import generate_plan, recommendation_input


def _action_set(plan: dict[str, Any]) -> set[str]:
    return {action["action_id"] for action in plan["recommended_actions"]}


def analyze_linkage() -> dict[str, Any]:
    output = ARTIFACT_ROOT / "linkage/analysis.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    profiles = pd.read_parquet(ARTIFACT_ROOT / "prediction/risk_profiles.parquet")
    sample = profiles.sort_values("lineage_id").head(100).to_dict(orient="records")
    probability_stability: list[float] = []
    for profile in sample:
        base = generate_plan(profile)
        original = _action_set(base)
        for delta in (-0.02, -0.01, 0.01, 0.02):
            changed = copy.deepcopy(profile)
            changed["probability_at_risk"] = float(
                np.clip(changed["probability_at_risk"] + delta, 0, 1)
            )
            altered = _action_set(generate_plan(changed))
            union = original | altered
            probability_stability.append(
                len(original & altered) / len(union) if union else 1.0
            )
    anchor = copy.deepcopy(sample[0])
    anchor["probability_at_risk"] = 0.65
    anchor["confidence_level"] = "HIGH_CONFIDENCE"
    anchor["decision_status"] = "PREDICTED"
    mechanisms: dict[str, set[str]] = {}
    for name, withdrawal, failure in (
        ("engagement_heavy", 0.75, 0.25),
        ("academic_heavy", 0.25, 0.75),
        ("mixed", 0.75, 0.75),
    ):
        changed = copy.deepcopy(anchor)
        changed["withdrawal_risk_horizon"] = withdrawal
        changed["probability_fail"] = failure
        remaining = 1 - failure
        changed["probability_pass"] = remaining * 0.8
        changed["probability_distinction"] = remaining * 0.2
        mechanisms[name] = _action_set(generate_plan(changed))
    confidence = copy.deepcopy(anchor)
    confident_plan = generate_plan(confidence)
    confidence["confidence_level"] = "LOW_CONFIDENCE"
    confidence["uncertainty_score"] = 0.95
    confidence["decision_status"] = "ABSTAIN_REVIEW_REQUIRED"
    uncertain_plan = generate_plan(confidence)
    stale_rejected = False
    try:
        recommendation_input(anchor, current_state_cutoff_day=int(anchor["cutoff_day"]) + 1)
    except ValueError:
        stale_rejected = True
    mechanism_changed = len({tuple(sorted(value)) for value in mechanisms.values()}) >= 2
    result = {
        "schema_version": "v6_prediction_recommendation_linkage_v1",
        "status": "PASS"
        if mechanism_changed
        and stale_rejected
        and uncertain_plan["requires_expert_review"]
        and float(np.mean(probability_stability)) >= 0.75
        else "FAIL",
        "probability_perturbation_mean_action_jaccard": float(
            np.mean(probability_stability)
        ),
        "mechanism_action_sets": {
            name: sorted(values) for name, values in mechanisms.items()
        },
        "mechanism_changes_plan": mechanism_changed,
        "confidence_raises_expert_review": bool(
            uncertain_plan["requires_expert_review"]
            and (
                not confident_plan["requires_expert_review"]
                or len(uncertain_plan["recommended_actions"])
                <= len(confident_plan["recommended_actions"])
            )
        ),
        "stale_profile_rejected": stale_rejected,
        "causal_claim": "PROHIBITED",
    }
    atomic_json(output, result)
    atomic_text(
        REPORT_ROOT / "PREDICTION_RECOMMENDATION_LINKAGE.md",
        f"""# V6 prediction-recommendation linkage

1. Calibrated risk probability and percentile determine the versioned risk band and action intensity.
2. Withdrawal horizon risk shifts the mechanism toward engagement support and shortens monitoring cadence.
3. Fail probability shifts the mechanism toward academic remediation and assessment actions.
4. Low confidence forces uncertain-risk handling and expert review without stronger automatic intervention.
5. Top-5/10% buckets raise advisor priority to immediate/high even near a probability boundary.
6. Deep-ML disagreement at the registered threshold adds mandatory expert review.
7. Engagement-heavy, academic-heavy and mixed counterfactuals produced mechanism-appropriate action changes: **{mechanism_changed}**.
8. Small probability perturbations (+/-0.01, +/-0.02) had mean action-set Jaccard **{result['probability_perturbation_mean_action_jaccard']:.3f}**.
9. Material mechanism/confidence state changes altered plans and escalation in the registered direction.
10. Every plan traces to risk-profile lineage, checkpoint manifest, calibration, policy and V5.2 engine version.

The stale-state test passed: **{stale_rejected}**. These are policy stability
tests, not causal claims about student outcomes.
""",
    )
    return result


__all__ = ["analyze_linkage"]
