"""Blinded expert-case export and immutable raw-label normalization."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ApprovalStatus,
    CandidateEvaluation,
    ExpertActionRating,
    ExpertCase,
    ExpertCaseReview,
    PlanStatus,
)
from .exceptions import ExpertLabelValidationError
from .exceptions import ContractValidationError

EXPORT_VERSION = "recommend_hybrid_expert_export_v1"
ALLOWED_EXPERTS = ("expert_01", "expert_02")
RATING_FIELDS = (
    "case_id",
    "action_id",
    "expert_id",
    "relevance_score",
    "approval_status",
    "missing_action",
    "safety_concern",
    "escalation_required",
    "reason_support",
    "comment",
)
REVIEW_FIELDS = (
    "case_id",
    "expert_id",
    "plan_score",
    "overall_status",
    "missing_actions",
    "safety_concerns",
    "review_comment",
)


def pseudonymous_case_id(
    student_key: str, course_key: str, stage: str, secret: bytes
) -> str:
    if len(secret) < 16:
        raise ExpertLabelValidationError("blinding secret must contain at least 16 bytes")
    message = "\x1f".join((student_key, course_key, stage)).encode("utf-8")
    return "case_" + hmac.new(secret, message, hashlib.sha256).hexdigest()[:24]


def _probability_band(probability: float) -> str:
    if probability < 1 / 3:
        return "LOWER_THIRD"
    if probability < 2 / 3:
        return "MIDDLE_THIRD"
    return "UPPER_THIRD"


def _uncertainty_band(value: float) -> str:
    if value < 0.33:
        return "LOW"
    if value < 0.66:
        return "MEDIUM"
    return "HIGH"


def blinded_case_payload(case: ExpertCase) -> dict[str, Any]:
    """Remove identifiers, model internals, future outcomes and exact probabilities."""
    context = case.prediction_context
    observed = case.observed_state
    risk_probability = context.class_probabilities[1]
    return {
        "schema_version": EXPORT_VERSION,
        "case_id": case.case_id,
        "stage": context.stage.value,
        "cutoff_day": context.cutoff_day,
        "predicted_class": context.predicted_class,
        "risk_probability_band": _probability_band(risk_probability),
        "confidence_band": _probability_band(context.confidence),
        "uncertainty_band": _uncertainty_band(context.uncertainty),
        "seed_disagreement_band": _uncertainty_band(context.seed_disagreement),
        "observed_state": observed.to_dict(),
        "candidate_actions": [
            {
                "action_id": item.action.action_id,
                "category": item.action.category,
                "title": item.action.title,
                "description": item.action.description,
                "weekly_minutes": item.action.weekly_minutes,
                "requires_human_review": item.action.requires_human_review,
                "success_criterion": item.action.success_criterion,
                "eligibility_status": item.status.value,
                "reason_codes": list(item.reason_codes),
            }
            for item in case.candidate_actions
        ],
        "blinding": dict(case.blinding_metadata),
    }


def export_expert_package(
    cases: Sequence[ExpertCase],
    output_root: Path,
    *,
    reviewers: Sequence[str] = ALLOWED_EXPERTS,
    shuffle_secret: bytes,
) -> dict[str, Any]:
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ExpertLabelValidationError("expert cases must be non-empty and unique")
    if not reviewers or set(reviewers) - set(ALLOWED_EXPERTS):
        raise ExpertLabelValidationError("reviewer ID is not approved")
    exports = output_root / "exports"
    templates = output_root / "templates"
    exports.mkdir(parents=True, exist_ok=True)
    templates.mkdir(parents=True, exist_ok=True)
    payloads = [blinded_case_payload(case) for case in cases]
    case_path = exports / "expert_cases.json"
    case_path.write_text(json.dumps(payloads, indent=2, ensure_ascii=False), encoding="utf-8")
    for reviewer in reviewers:
        seed = int.from_bytes(
            hmac.new(shuffle_secret, reviewer.encode(), hashlib.sha256).digest()[:8],
            byteorder="big",
        )
        rng = random.Random(seed)
        rows: list[dict[str, Any]] = []
        for payload in payloads:
            actions = list(payload["candidate_actions"])
            rng.shuffle(actions)
            for order, action in enumerate(actions, start=1):
                rows.append(
                    {
                        "case_id": payload["case_id"],
                        "action_id": action["action_id"],
                        "expert_id": reviewer,
                        "candidate_order": order,
                        "relevance_score": "",
                        "approval_status": "",
                        "missing_action": "",
                        "safety_concern": "",
                        "escalation_required": "",
                        "reason_support": "",
                        "comment": "",
                    }
                )
        rating_path = templates / f"{reviewer}_action_ratings.csv"
        with rating_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("candidate_order", *RATING_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        review_path = templates / f"{reviewer}_case_reviews.csv"
        with review_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            for payload in payloads:
                writer.writerow(
                    {
                        "case_id": payload["case_id"],
                        "expert_id": reviewer,
                        "plan_score": "",
                        "overall_status": "",
                        "missing_actions": "",
                        "safety_concerns": "",
                        "review_comment": "",
                    }
                )
    return {
        "schema_version": EXPORT_VERSION,
        "case_count": len(cases),
        "reviewer_templates": list(reviewers),
        "fabricated_labels": 0,
        "exact_probability_blinded": True,
        "future_outcome_blinded": True,
        "student_identifier_blinded": True,
        "case_export": case_path.as_posix(),
    }


def _boolean(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ExpertLabelValidationError(f"{field} must be a boolean")


def _known_cases(case_export: Path) -> dict[str, set[str]]:
    cases = json.loads(case_export.read_text(encoding="utf-8"))
    return {
        item["case_id"]: {action["action_id"] for action in item["candidate_actions"]}
        for item in cases
    }


def import_expert_ratings(
    raw_path: Path,
    case_export: Path,
    normalized_path: Path,
    *,
    allowed_experts: Sequence[str] = ALLOWED_EXPERTS,
) -> tuple[ExpertActionRating, ...]:
    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    known = _known_cases(case_export)
    with raw_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(RATING_FIELDS).issubset(reader.fieldnames or []):
            raise ExpertLabelValidationError("raw rating schema is incomplete")
        source = list(reader)
    ratings: list[ExpertActionRating] = []
    identities: set[tuple[str, str, str]] = set()
    for row in source:
        if any(row.get(field, "").strip() == "" for field in RATING_FIELDS[:-1]):
            raise ExpertLabelValidationError("required rating field is empty")
        case_id, action_id, expert_id = row["case_id"], row["action_id"], row["expert_id"]
        if case_id not in known or action_id not in known[case_id]:
            raise ExpertLabelValidationError("unknown case/action pair")
        if expert_id not in allowed_experts:
            raise ExpertLabelValidationError("unknown expert ID")
        identity = (case_id, action_id, expert_id)
        if identity in identities:
            raise ExpertLabelValidationError("duplicate expert action rating")
        identities.add(identity)
        try:
            score = int(row["relevance_score"])
            approval = ApprovalStatus(row["approval_status"])
        except (ValueError, KeyError) as exc:
            raise ExpertLabelValidationError("invalid score or approval status") from exc
        try:
            rating = ExpertActionRating(
                case_id=case_id,
                action_id=action_id,
                expert_id=expert_id,
                relevance_score=score,
                approval_status=approval,
                missing_action=_boolean(row["missing_action"], "missing_action"),
                safety_concern=_boolean(row["safety_concern"], "safety_concern"),
                escalation_required=_boolean(
                    row["escalation_required"], "escalation_required"
                ),
                reason_support=row["reason_support"].strip(),
                comment=row.get("comment", "").strip(),
            )
        except ContractValidationError as exc:
            raise ExpertLabelValidationError(str(exc)) from exc
        if rating.approval_status is ApprovalStatus.APPROVE and rating.relevance_score < 1:
            raise ExpertLabelValidationError("approval contradicts non-positive relevance")
        ratings.append(rating)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(
        json.dumps([rating.to_dict() for rating in ratings], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_before:
        raise ExpertLabelValidationError("raw expert file was modified")
    return tuple(ratings)


def import_expert_case_reviews(
    raw_path: Path,
    case_export: Path,
    normalized_path: Path,
    *,
    allowed_experts: Sequence[str] = ALLOWED_EXPERTS,
) -> tuple[ExpertCaseReview, ...]:
    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    known = _known_cases(case_export)
    with raw_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(REVIEW_FIELDS).issubset(reader.fieldnames or []):
            raise ExpertLabelValidationError("raw case-review schema is incomplete")
        source = list(reader)
    reviews: list[ExpertCaseReview] = []
    identities: set[tuple[str, str]] = set()
    for row in source:
        if any(row.get(field, "").strip() == "" for field in REVIEW_FIELDS[:4]):
            raise ExpertLabelValidationError("required case-review field is empty")
        case_id, expert_id = row["case_id"], row["expert_id"]
        if case_id not in known or expert_id not in allowed_experts:
            raise ExpertLabelValidationError("unknown case or expert ID")
        identity = (case_id, expert_id)
        if identity in identities:
            raise ExpertLabelValidationError("duplicate expert case review")
        identities.add(identity)
        try:
            review = ExpertCaseReview(
                case_id=case_id,
                expert_id=expert_id,
                plan_score=int(row["plan_score"]),
                overall_status=PlanStatus(row["overall_status"]),
                missing_actions=tuple(
                    item.strip() for item in row["missing_actions"].split("|") if item.strip()
                ),
                safety_concerns=tuple(
                    item.strip() for item in row["safety_concerns"].split("|") if item.strip()
                ),
                review_comment=row["review_comment"].strip(),
            )
        except (ContractValidationError, ValueError) as exc:
            raise ExpertLabelValidationError(str(exc)) from exc
        if review.overall_status is PlanStatus.APPROVED and review.safety_concerns:
            raise ExpertLabelValidationError("approved review contradicts safety concern")
        reviews.append(review)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_path.write_text(
        json.dumps([review.to_dict() for review in reviews], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_before:
        raise ExpertLabelValidationError("raw expert case-review file was modified")
    return tuple(reviews)


__all__ = [
    "ALLOWED_EXPERTS",
    "EXPORT_VERSION",
    "RATING_FIELDS",
    "blinded_case_payload",
    "export_expert_package",
    "import_expert_case_reviews",
    "import_expert_ratings",
    "pseudonymous_case_id",
]
