"""Config-driven evidence severity and action eligibility evaluation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from .policy_contracts import (
    EligibilityStatus,
    EvidenceAvailability,
    EvidenceItem,
    EvidenceSeverity,
)

SEVERITY_RANK = {
    EvidenceSeverity.MISSING: -1,
    EvidenceSeverity.NONE: 0,
    EvidenceSeverity.LOW: 1,
    EvidenceSeverity.MEDIUM: 2,
    EvidenceSeverity.HIGH: 3,
    EvidenceSeverity.CRITICAL: 4,
}


def _matches(value: Any, operator: str, threshold: Any) -> bool:
    if operator == "ge":
        return value >= threshold
    if operator == "gt":
        return value > threshold
    if operator == "le":
        return value <= threshold
    if operator == "lt":
        return value < threshold
    if operator == "eq":
        return value == threshold
    if operator == "present":
        return bool(value) is bool(threshold)
    raise ValueError(f"unsupported policy operator: {operator}")


def apply_severity_rules(
    evidence: tuple[EvidenceItem, ...], rules: Mapping[str, Any]
) -> tuple[EvidenceItem, ...]:
    result: list[EvidenceItem] = []
    for item in evidence:
        if item.availability is not EvidenceAvailability.AVAILABLE:
            result.append(item)
            continue
        definition = rules.get(item.feature_name)
        if definition is None:
            result.append(replace(item, severity=EvidenceSeverity.NONE))
            continue
        severity = EvidenceSeverity(definition.get("default", "NONE"))
        for rule in definition.get("rules", []):
            if _matches(item.observed_value, rule["operator"], rule["value"]):
                severity = EvidenceSeverity(rule["severity"])
                break
        result.append(replace(item, severity=severity))
    return tuple(result)


def _trigger_matches(item: EvidenceItem, trigger: Mapping[str, Any]) -> bool:
    if item.availability is not EvidenceAvailability.AVAILABLE:
        return False
    rank = SEVERITY_RANK[item.severity]
    minimum = SEVERITY_RANK[EvidenceSeverity(trigger["minimum_severity"])]
    maximum = SEVERITY_RANK[EvidenceSeverity(trigger.get("maximum_severity", "CRITICAL"))]
    return minimum <= rank <= maximum


def evaluate_eligibility(
    action_rule: Mapping[str, Any],
    *,
    stage: str,
    evidence_by_name: Mapping[str, EvidenceItem],
) -> tuple[EligibilityStatus, tuple[EvidenceItem, ...], tuple[str, ...]]:
    if stage not in action_rule["stages"]:
        return EligibilityStatus.INELIGIBLE_STAGE, (), ()
    triggers_any = tuple(action_rule.get("trigger_any", ()))
    triggers_all = tuple(action_rule.get("trigger_all", ()))
    referenced = {trigger["feature"] for trigger in (*triggers_any, *triggers_all)}
    missing = tuple(
        sorted(
            feature
            for feature in referenced
            if feature not in evidence_by_name
            or evidence_by_name[feature].availability
            in {EvidenceAvailability.MISSING, EvidenceAvailability.INSUFFICIENT_EVIDENCE}
        )
    )
    all_support = tuple(
        evidence_by_name[trigger["feature"]]
        for trigger in triggers_all
        if trigger["feature"] in evidence_by_name
        and _trigger_matches(evidence_by_name[trigger["feature"]], trigger)
    )
    any_support = tuple(
        evidence_by_name[trigger["feature"]]
        for trigger in triggers_any
        if trigger["feature"] in evidence_by_name
        and _trigger_matches(evidence_by_name[trigger["feature"]], trigger)
    )
    if missing and (triggers_all or len(missing) == len(referenced)):
        return EligibilityStatus.MISSING_REQUIRED_EVIDENCE, (), missing
    if triggers_all and len(all_support) != len(triggers_all):
        return EligibilityStatus.NOT_APPLICABLE, (), missing
    if triggers_any and not any_support:
        return EligibilityStatus.NOT_APPLICABLE, (), missing
    supporting = tuple(dict.fromkeys((*all_support, *any_support)))
    if not supporting:
        return EligibilityStatus.NOT_APPLICABLE, (), missing
    status = (
        EligibilityStatus.REQUIRES_HUMAN_CONTACT
        if action_rule.get("requires_human_contact", False)
        else EligibilityStatus.ELIGIBLE
    )
    return status, supporting, missing


def detect_contradictions(
    evidence_by_name: Mapping[str, EvidenceItem], rules: tuple[Mapping[str, Any], ...]
) -> tuple[str, ...]:
    reasons: list[str] = []
    for rule in rules:
        conditions = rule.get("conditions", ())
        if conditions and all(
            condition["feature"] in evidence_by_name
            and evidence_by_name[condition["feature"]].availability
            is EvidenceAvailability.AVAILABLE
            and _matches(
                evidence_by_name[condition["feature"]].observed_value,
                condition["operator"],
                condition["value"],
            )
            for condition in conditions
        ):
            reasons.append(rule["reason_code"])
    return tuple(reasons)


__all__ = [
    "SEVERITY_RANK",
    "apply_severity_rules",
    "detect_contradictions",
    "evaluate_eligibility",
]
