"""Explainable rule-based learning recommendations for persisted predictions.

The recommendation policy is downstream of the frozen classifier. It uses the
predicted class, classifier confidence and observable source features only. It
does not read true labels, target values, source row numbers or database IDs.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High"}
RISK_BANDS = {0: "High", 1: "Medium", 2: "Low"}
POLICY_VERSION = "student_mat_rule_policy_v3"
STUDENT_RECOMMENDATION_FEATURES = {
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "activities",
    "internet",
    "absences",
    "G1",
    "G2",
}
DATASET_KIND = {
    "student-mat": "student",
    "student-por": "student",
    "xapi": "xapi",
}
FORBIDDEN_INPUT_COLUMNS = {
    "G3",
    "G3_raw",
    "true_label",
    "target_label",
    "__source_row_number",
    "record_id",
    "dataset_version_id",
    "prediction_id",
    "run_id",
}
ADVISORY_EXCLUDED_COLUMNS = {
    "school",
    "sex",
    "address",
    "guardian",
    "paid",
    "Dalc",
    "Walc",
    "goout",
}


@dataclass(frozen=True)
class RiskFactor:
    code: str
    group: str
    priority: int
    evidence: str
    reason: str


@dataclass(frozen=True)
class Action:
    action: str
    frequency: str
    duration: str
    reason: str
    risk_code: str


def _number(features: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = features.get(name, default)
    try:
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def _text(features: dict[str, Any], name: str, default: str = "") -> str:
    value = features.get(name, default)
    return default if pd.isna(value) else str(value).strip()


def sanitize_features(features: dict[str, Any]) -> dict[str, Any]:
    """Remove target, lineage and socially sensitive advisory inputs."""
    excluded = FORBIDDEN_INPUT_COLUMNS | ADVISORY_EXCLUDED_COLUMNS
    return {key: value for key, value in features.items() if key not in excluded}


def prepare_recommendation_features(features: dict[str, Any]) -> dict[str, Any]:
    """Allowlist source features used by the Student-Mat recommendation policy."""
    sanitized = sanitize_features(features)
    return {key: sanitized[key] for key in sorted(STUDENT_RECOMMENDATION_FEATURES) if key in sanitized}


def extract_features(frame: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """Return observable non-target features for legacy recommender scripts."""
    if dataset_name not in DATASET_KIND:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    rows: list[list[float]] = []
    for raw_record in frame.to_dict("records"):
        record = sanitize_features(raw_record)
        if DATASET_KIND[dataset_name] == "student":
            rows.append(
                [
                    _number(record, "absences"),
                    _number(record, "studytime", 1.0),
                    _number(record, "failures"),
                    _number(record, "G1"),
                    _number(record, "G2"),
                    _number(record, "Dalc", 1.0),
                    _number(record, "Walc", 1.0),
                    _number(record, "goout", 1.0),
                ]
            )
        else:
            rows.append(
                [
                    _number(record, "raisedhands"),
                    _number(record, "VisITedResources"),
                    _number(record, "AnnouncementsView"),
                    _number(record, "Discussion"),
                    float(_text(record, "StudentAbsenceDays").lower() == "above-7"),
                    float(_text(record, "ParentAnsweringSurvey").lower() == "no"),
                    float(_text(record, "ParentschoolSatisfaction").lower() == "bad"),
                ]
            )
    return np.asarray(rows, dtype=np.float32)


def confidence_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.55:
        return "medium"
    return "low"


def cautious_prefix(level: str) -> str:
    if level == "low":
        return "Because model confidence is low, verify this with a teacher/advisor before applying: "
    if level == "medium":
        return "Use this as a guided recommendation and monitor weekly evidence: "
    return ""


def identify_student_risks(features: dict[str, Any], predicted_class: int) -> list[RiskFactor]:
    """Find auditable Student-Mat risk factors from observed dataset columns."""
    risks: list[RiskFactor] = []
    absences = _number(features, "absences")
    studytime = _number(features, "studytime", 1.0)
    failures = _number(features, "failures")
    g1 = _number(features, "G1")
    g2 = _number(features, "G2")
    internet = _text(features, "internet").lower()
    schoolsup = _text(features, "schoolsup").lower()
    famsup = _text(features, "famsup").lower()
    activities = _text(features, "activities").lower()

    absence_ratio = absences / max(studytime, 0.5)
    if absences >= 10 or absence_ratio >= 5:
        risks.append(
            RiskFactor(
                "attendance_absences",
                "attendance/absences",
                1,
                f"absences={absences:.0f}, absences_per_studytime={absence_ratio:.1f}",
                "High absence load can make the student miss prerequisite practice and feedback.",
            )
        )
    if studytime <= 1:
        risks.append(
            RiskFactor(
                "low_study_time",
                "study time",
                2,
                f"studytime={studytime:.0f}/4",
                "The recorded weekly study-time band is the lowest category.",
            )
        )
    if g2 < 10 or (g1 > 0 and g2 < g1):
        risks.append(
            RiskFactor(
                "prior_grade_gap",
                "previous grades",
                1,
                f"G1={g1:.0f}, G2={g2:.0f}",
                "Earlier period grades suggest content gaps or a downward trend before the final prediction.",
            )
        )
    if failures > 0:
        risks.append(
            RiskFactor(
                "failure_history",
                "failures",
                1,
                f"failures={failures:.0f}",
                "Previous course failures indicate accumulated academic risk.",
            )
        )
    if famsup == "no" and schoolsup == "no":
        risks.append(
            RiskFactor(
                "support_gap",
                "family/parent support",
                2,
                "famsup=no, schoolsup=no",
                "The student has neither family educational support nor school support marked in the source record.",
            )
        )
    elif famsup == "no" or schoolsup == "no":
        risks.append(
            RiskFactor(
                "partial_support_gap",
                "family/parent support",
                3,
                f"famsup={famsup or 'unknown'}, schoolsup={schoolsup or 'unknown'}",
                "One support channel is absent, so weekly check-ins should be explicit.",
            )
        )
    if internet == "no":
        risks.append(
            RiskFactor(
                "limited_internet_access",
                "internet/educational support",
                2,
                "internet=no",
                "Limited internet access can reduce access to online practice and learning materials.",
            )
        )
    if activities == "yes" and studytime <= 1:
        risks.append(
            RiskFactor(
                "workload_balance",
                "extracurricular/workload",
                3,
                "activities=yes with low studytime",
                "Extracurricular workload may need scheduling around minimum study blocks.",
            )
        )
    if not risks:
        risks.append(
            RiskFactor(
                "prediction_monitoring",
                "prediction basis",
                4,
                f"predicted_class={CLASS_NAMES[int(predicted_class)]}",
                "No specific source-feature risk crossed a rule threshold, so the recommendation focuses on monitoring the prediction band.",
            )
        )
    return sorted(risks, key=lambda risk: (risk.priority, risk.code))[:5]


def _action_for_risk(risk: RiskFactor, band: str, level: str) -> Action:
    prefix = cautious_prefix(level)
    if risk.code == "attendance_absences":
        return Action(
            action=prefix + "Create a missed-lesson recovery checklist and confirm attendance with the advisor.",
            frequency="2 check-ins per week",
            duration="4 weeks",
            reason="Absence-related risk needs fast recovery of missed content before adding new workload.",
            risk_code=risk.code,
        )
    if risk.code in {"prior_grade_gap", "failure_history", "no_extra_academic_support"}:
        return Action(
            action=prefix + "Run a topic-level diagnostic, then complete targeted exercises for the two weakest topics.",
            frequency="3 practice sessions per week",
            duration="45 minutes per session for 4 weeks",
            reason="Earlier grades/failures point to specific academic gaps rather than an unfocused workload increase.",
            risk_code=risk.code,
        )
    if risk.code == "low_study_time":
        return Action(
            action=prefix + "Schedule fixed study blocks before the next class and log completed tasks.",
            frequency="4 days per week",
            duration="30-45 minutes per block for 4 weeks",
            reason="The source study-time band is low, so the intervention defines measurable time blocks.",
            risk_code=risk.code,
        )
    if risk.code in {"support_gap", "partial_support_gap", "guardian_followup"}:
        return Action(
            action=prefix + "Set a short advisor-family progress update and agree on one weekly support task.",
            frequency="1 update per week",
            duration="10-15 minutes for 4 weeks",
            reason="Support-channel risks are best addressed through coordination, not extra assignments alone.",
            risk_code=risk.code,
        )
    if risk.code == "limited_internet_access":
        return Action(
            action=prefix + "Prepare offline practice materials or reserve school computer-lab time.",
            frequency="2 resource checks per week",
            duration="4 weeks",
            reason="The action directly addresses access to learning materials.",
            risk_code=risk.code,
        )
    if risk.code in {"workload_balance", "time_management"}:
        return Action(
            action=prefix + "Move one non-urgent activity away from assessment days and protect study blocks.",
            frequency="weekly planning session",
            duration="15 minutes per week for 4 weeks",
            reason="The issue is scheduling pressure, so the intervention changes the calendar rather than adding vague workload.",
            risk_code=risk.code,
        )
    if risk.code == "alcohol_weekend_pattern":
        return Action(
            action=prefix + "Review weekend routine with an advisor and keep school-night study/sleep times stable.",
            frequency="1 reflection and check-in per week",
            duration="4 weeks",
            reason="The rule flags a behavior pattern that may disrupt preparation; the recommendation remains supportive.",
            risk_code=risk.code,
        )
    return Action(
        action=prefix + ("Maintain the current learning routine and verify progress evidence." if band == "Low" else "Review prediction evidence with the advisor."),
        frequency="1 review per week",
        duration="4 weeks",
        reason="No stronger feature-specific risk was detected, so the safest action is monitoring with evidence.",
        risk_code=risk.code,
    )


def build_recommendation(
    features: dict[str, Any],
    predicted_class: int,
    confidence: float,
    probability: dict[str, float] | None = None,
) -> dict[str, Any]:
    clean_features = sanitize_features(features)
    band = RISK_BANDS[int(predicted_class)]
    level = confidence_level(float(confidence))
    risks = identify_student_risks(clean_features, int(predicted_class))
    priority_risks = [asdict(risk) for risk in risks[:3]]
    actions = [_action_for_risk(risk, band, level) for risk in risks[:3]]

    if band == "High":
        weekly_plan = [
            "Week 1: confirm risk evidence and start the highest-priority remediation action.",
            "Week 2: review attendance/practice evidence and adjust the weakest-topic exercise set.",
            "Week 3: add a timed practice task only if Week 2 evidence improves.",
            "Week 4: compare progress evidence with the advisor and decide whether to continue support.",
        ]
    elif band == "Medium":
        weekly_plan = [
            "Week 1: verify the main risk factor and set one measurable weekly target.",
            "Week 2: complete targeted practice and monitor whether the risk indicator changes.",
            "Week 3: keep the effective action and remove actions that did not match the evidence.",
            "Week 4: review progress and decide whether to reduce or intensify support.",
        ]
    else:
        weekly_plan = [
            "Week 1: maintain current study routine and check the most relevant risk signal.",
            "Week 2: complete one enrichment or consolidation activity tied to prior grades.",
            "Week 3: keep weekly evidence of attendance, practice and assessment preparation.",
            "Week 4: review whether the student can continue with light monitoring only.",
        ]

    prediction_basis = [
        f"predicted_class={CLASS_NAMES[int(predicted_class)]}",
        f"risk_band={band}",
        f"confidence={float(confidence):.4f}",
        f"confidence_level={level}",
    ]
    if probability:
        prediction_basis.append("probability=" + json.dumps(probability, sort_keys=True))

    return {
        "risk_band": band,
        "confidence_level": level,
        "priority_risks": priority_risks,
        "weekly_plan": weekly_plan,
        "recommended_actions": [asdict(action) for action in actions],
        "explanation": {
            "prediction_basis": prediction_basis,
            "risk_factor_basis": [risk["evidence"] for risk in priority_risks],
            "policy_version": POLICY_VERSION,
            "scope_note": "Rule-based advisory support; not evidence of causal improvement.",
        },
        "disclaimer": "Advisory support only; a teacher or advisor must review before action.",
    }


def validate_recommendation_schema(payload: dict[str, Any]) -> None:
    required = {"risk_band", "confidence_level", "priority_risks", "weekly_plan", "recommended_actions", "explanation", "disclaimer"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Missing recommendation fields: {sorted(missing)}")
    if payload["risk_band"] not in {"Low", "Medium", "High"}:
        raise ValueError("risk_band must be Low, Medium, or High.")
    if payload["confidence_level"] not in {"high", "medium", "low"}:
        raise ValueError("confidence_level must be high, medium, or low.")
    if "advisor" not in payload["disclaimer"].lower() and "teacher" not in payload["disclaimer"].lower():
        raise ValueError("Disclaimer must require human advisory review.")
    if not payload["priority_risks"]:
        raise ValueError("Recommendation must include at least one risk factor or explicit monitoring reason.")
    if not payload["recommended_actions"]:
        raise ValueError("Recommendation must include at least one action.")
    for action in payload["recommended_actions"]:
        for key in ["action", "frequency", "duration", "reason", "risk_code"]:
            if not action.get(key):
                raise ValueError(f"Action missing {key}.")
    explanation = payload["explanation"]
    if explanation.get("policy_version") != POLICY_VERSION:
        raise ValueError("Incorrect policy version.")
    forbidden_text = json.dumps(payload, ensure_ascii=False)
    for forbidden in FORBIDDEN_INPUT_COLUMNS:
        if forbidden in forbidden_text:
            raise ValueError(f"Forbidden metadata leaked into recommendation: {forbidden}")
    for forbidden in ADVISORY_EXCLUDED_COLUMNS:
        if f'"{forbidden}"' in forbidden_text:
            raise ValueError(f"Excluded advisory input leaked into recommendation: {forbidden}")


def recommendation_to_legacy_row(
    *,
    row_index: int,
    predicted_class: int,
    confidence: float,
    recommendation: dict[str, Any],
) -> dict[str, Any]:
    validate_recommendation_schema(recommendation)
    headline = {
        "High": "Priority learning support plan",
        "Medium": "Guided consolidation learning plan",
        "Low": "Maintenance and enrichment learning plan",
    }[recommendation["risk_band"]]
    return {
        "source_row_index": row_index,
        "predicted_class": int(predicted_class),
        "predicted_class_name": CLASS_NAMES[int(predicted_class)],
        "confidence": round(float(confidence), 6),
        "confidence_level": recommendation["confidence_level"],
        "risk_band": recommendation["risk_band"],
        "headline": headline,
        "policy_version": POLICY_VERSION,
        "risk_factors": json.dumps(recommendation["priority_risks"], ensure_ascii=False),
        "risk_scores": json.dumps(
            {risk["code"]: {"priority": risk["priority"], "group": risk["group"]} for risk in recommendation["priority_risks"]},
            ensure_ascii=False,
        ),
        "learning_path": json.dumps(recommendation, ensure_ascii=False),
    }


def generate_learning_path_report(
    original_features: pd.DataFrame,
    predictions: np.ndarray,
    confidences: np.ndarray,
    dataset_name: str,
    train_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate one standardized rule-based recommendation per prediction."""
    if dataset_name not in {"student-mat", "student-por"}:
        raise ValueError("The standardized recommendation policy currently supports student datasets.")
    rows = []
    for row_index, record in enumerate(original_features.reset_index(drop=True).to_dict("records")):
        source_row_index = int(record.get("__source_row_number", row_index))
        recommendation = build_recommendation(
            record,
            int(predictions[row_index]),
            float(confidences[row_index]),
        )
        rows.append(
            recommendation_to_legacy_row(
                row_index=source_row_index,
                predicted_class=int(predictions[row_index]),
                confidence=float(confidences[row_index]),
                recommendation=recommendation,
            )
        )
    return pd.DataFrame(rows)


def reference_risk_targets(frame: pd.DataFrame, dataset_name: str) -> np.ndarray:
    """Compatibility helper for offline structural recommender evaluation."""
    if dataset_name not in {"student-mat", "student-por"}:
        raise ValueError("The standardized recommendation policy currently supports student datasets.")
    targets: list[list[float]] = []
    for record in frame.to_dict("records"):
        clean = sanitize_features(record)
        risk_codes = {risk.code for risk in identify_student_risks(clean, predicted_class=1)}
        targets.append(
            [
                float("attendance_absences" in risk_codes),
                float("failure_history" in risk_codes),
                float("prior_grade_gap" in risk_codes),
                float("low_study_time" in risk_codes),
                float("alcohol_weekend_pattern" in risk_codes),
                float("time_management" in risk_codes or "workload_balance" in risk_codes),
            ]
        )
    return np.asarray(targets, dtype=np.float32)


class MLPLearningPathEngine:
    """Backward-compatible facade over the deterministic rule policy.

    The name is retained for old offline evaluation scripts; no neural
    recommender is trained or loaded here.
    """

    risk_codes = (
        "attendance_absences",
        "failure_history",
        "prior_grade_gap",
        "low_study_time",
        "alcohol_weekend_pattern",
        "time_management",
    )

    def __init__(self, dataset_name: str, train_frame: pd.DataFrame | None = None):
        if dataset_name not in {"student-mat", "student-por"}:
            raise ValueError("The standardized recommendation policy currently supports student datasets.")
        self.dataset_name = dataset_name
        self.train_frame = train_frame
        self.checkpoint = {
            "schema_version": 3,
            "policy_version": POLICY_VERSION,
            "architecture": "deterministic_rule_policy",
            "epochs_completed": 0,
            "best_validation_loss": None,
            "seed": None,
        }

    def predict_scores(self, frame: pd.DataFrame) -> np.ndarray:
        return reference_risk_targets(frame, self.dataset_name)

    def generate(self, features: dict[str, Any], predicted_class: int, confidence: float) -> dict[str, Any]:
        recommendation = build_recommendation(features, predicted_class, confidence)
        return {
            "predicted_class": int(predicted_class),
            "predicted_class_name": CLASS_NAMES[int(predicted_class)],
            "confidence": round(float(confidence), 6),
            "risk_band": recommendation["risk_band"],
            "headline": {
                "High": "Priority learning support plan",
                "Medium": "Guided consolidation learning plan",
                "Low": "Maintenance and enrichment learning plan",
            }[recommendation["risk_band"]],
            "risk_factors": recommendation["priority_risks"],
            "risk_scores": {
                risk["code"]: {"priority": risk["priority"], "group": risk["group"]}
                for risk in recommendation["priority_risks"]
            },
            "learning_path": [
                {
                    "phase": f"Week {index + 1}",
                    "goal": step,
                    "actions": recommendation["recommended_actions"][min(index, len(recommendation["recommended_actions"]) - 1)]["action"],
                }
                for index, step in enumerate(recommendation["weekly_plan"])
            ],
            "standardized_output": recommendation,
        }


def load_or_train_recommendation_model(
    dataset_name: str,
    train_frame: pd.DataFrame | None = None,
    force_retrain: bool = False,
) -> tuple[MLPLearningPathEngine, dict[str, Any]]:
    """Compatibility shim; the current policy is deterministic and has no model checkpoint."""
    engine = MLPLearningPathEngine(dataset_name, train_frame=train_frame)
    return engine, engine.checkpoint


def structural_validity_metrics(recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(recommendations)
    if total == 0:
        return {}
    valid = 0
    with_action = 0
    with_explanation = 0
    low_conf_cautious = 0
    contradictions = 0
    leaks = 0
    band_counts = Counter()
    confidence_counts = Counter()
    risk_counter = Counter()
    for payload in recommendations:
        try:
            validate_recommendation_schema(payload)
            valid += 1
        except ValueError:
            pass
        with_action += int(bool(payload.get("recommended_actions")))
        with_explanation += int(bool(payload.get("explanation", {}).get("prediction_basis")))
        band_counts[payload.get("risk_band", "unknown")] += 1
        confidence_counts[payload.get("confidence_level", "unknown")] += 1
        for risk in payload.get("priority_risks", []):
            risk_counter[risk.get("code", "unknown")] += 1
        if payload.get("confidence_level") == "low":
            text = json.dumps(payload.get("recommended_actions", []), ensure_ascii=False).lower()
            low_conf_cautious += int("verify" in text or "advisor" in text or "monitor" in text)
        serialized = json.dumps(payload, ensure_ascii=False)
        leaks += int(
            any(forbidden in serialized for forbidden in FORBIDDEN_INPUT_COLUMNS)
            or any(f'"{forbidden}"' in serialized for forbidden in ADVISORY_EXCLUDED_COLUMNS)
        )
        action_risks = [action.get("risk_code") for action in payload.get("recommended_actions", [])]
        contradictions += int(len(action_risks) != len(set(action_risks)))
    return {
        "total": total,
        "valid_schema_rate": valid / total,
        "with_explanation_rate": with_explanation / total,
        "with_specific_action_rate": with_action / total,
        "no_contradiction_rate": 1.0 - contradictions / total,
        "no_sensitive_metadata_leak_rate": 1.0 - leaks / total,
        "low_confidence_cautious_rate": None if confidence_counts["low"] == 0 else low_conf_cautious / confidence_counts["low"],
        "risk_band_coverage": dict(band_counts),
        "confidence_coverage": dict(confidence_counts),
        "top_risk_factors": risk_counter.most_common(10),
    }
