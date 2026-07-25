from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .contract import (
    ARTIFACT_ROOT,
    REPORT_ROOT,
    ROOT,
    SCHEMA_VERSION,
    atomic_json,
    atomic_text,
    canonical_sha256,
)
from .recommendation import load_plans


EXPERT_ROOT = ARTIFACT_ROOT / "expert_evaluation"
EXPERT_SCHEMA = "v6_2_expert_review_v1"
CASE_SCHEMA = "v6_2_expert_case_v1"
PLAN_SHEET = "Plan Review"
ACTION_SHEET = "Action Review"
REVIEWER_PATTERN = re.compile(r"^E[0-9]{2,}$")
PLAN_SCORE_VALUES = {1, 2, 3, 4, 5}
ACTION_RELEVANCE_VALUES = {"APPROVE", "PARTIAL", "UNSURE", "REJECT"}
YES_NO_VALUES = {"YES", "NO"}
ESCALATION_VALUES = {"CORRECT", "OVER_ESCALATED", "UNDER_ESCALATED", "UNSURE"}
REASON_SUPPORT_VALUES = {"SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNSURE"}
SAFETY_VALUES = {"SAFE", "CONCERN", "UNSAFE", "UNSURE"}


class ReviewImportError(ValueError):
    """Raised when a real expert file violates the preregistered contract."""


def _band(value: float, low: float, high: float) -> str:
    return "LOW" if value < low else "HIGH" if value >= high else "MEDIUM"


def _category(value: float | None, thresholds: tuple[float, float]) -> str:
    if value is None or pd.isna(value):
        return "NOT_AVAILABLE"
    return _band(float(value), thresholds[0], thresholds[1])


def _course_aliases(profiles: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    modules = {
        name: f"Course {chr(65 + index)}"
        for index, name in enumerate(sorted(profiles["code_module"].astype(str).unique()))
    }
    presentations = {
        name: f"Offering {index + 1}"
        for index, name in enumerate(
            sorted(profiles["code_presentation"].astype(str).unique())
        )
    }
    return modules, presentations


def _select_case_indices(frame: pd.DataFrame, count: int) -> list[int]:
    dimensions = [
        "risk_level",
        "confidence_level",
        "plan_status",
        "fail_risk_band",
        "disagreement_band",
        "code_module",
        "code_presentation",
    ]
    records = {
        int(index): {
            dimension: str(row[dimension]) for dimension in dimensions
        }
        | {"plan_id": str(row["plan_id"])}
        for index, row in frame.iterrows()
    }
    candidates = list(records)
    selected: list[int] = []
    counts: dict[str, Counter[str]] = {
        dimension: Counter() for dimension in dimensions
    }
    while candidates and len(selected) < min(count, len(frame)):
        def score(index: int) -> tuple[float, str]:
            row = records[index]
            rarity = sum(
                1.0 / (1.0 + counts[dimension][row[dimension]])
                for dimension in dimensions
            )
            return (-rarity, row["plan_id"])

        chosen = min(candidates, key=score)
        candidates.remove(chosen)
        selected.append(chosen)
        for dimension in dimensions:
            counts[dimension][records[chosen][dimension]] += 1
    return selected


def _review_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": EXPERT_SCHEMA,
        "schema_version": EXPERT_SCHEMA,
        "case_schema": CASE_SCHEMA,
        "reviewer_id": {
            "type": "string",
            "pattern": "^E[0-9]{2,}$",
            "description": "Pseudonymous reviewer identifier; no name or contact data.",
        },
        "plan_review": {
            "required": [
                "schema_version",
                "reviewer_id",
                "case_id",
                "q1_plan_score",
                "q3_missing_action",
                "q3_missing_action_text",
                "q4_escalation",
                "q5_reason_support",
                "q6_safety_workload",
                "q6_safety_note",
            ],
            "enums": {
                "q1_plan_score": sorted(PLAN_SCORE_VALUES),
                "q3_missing_action": sorted(YES_NO_VALUES),
                "q4_escalation": sorted(ESCALATION_VALUES),
                "q5_reason_support": sorted(REASON_SUPPORT_VALUES),
                "q6_safety_workload": sorted(SAFETY_VALUES),
            },
            "conditional_requirements": {
                "q3_missing_action_text": "required non-empty when q3_missing_action=YES",
                "q6_safety_note": "required non-empty when q6_safety_workload is CONCERN or UNSAFE",
            },
        },
        "action_review": {
            "required": [
                "schema_version",
                "reviewer_id",
                "case_id",
                "action_id",
                "q2_action_relevance",
            ],
            "enums": {
                "q2_action_relevance": sorted(ACTION_RELEVANCE_VALUES),
            },
        },
        "prohibited_fields": [
            "reviewer_name",
            "email",
            "student_id",
            "record_id",
            "outcome",
            "model_name",
            "prediction_probability",
        ],
        "synthetic_or_inferred_labels_permitted": False,
    }


def export_expert_package(case_count: int = 60) -> dict[str, Any]:
    plans = load_plans()
    profiles = pd.read_parquet(
        ROOT / "artifacts/v6/prediction/risk_profiles.parquet"
    ).set_index("lineage_id", drop=False)
    rows: list[dict[str, Any]] = []
    for plan in plans:
        profile = profiles.loc[plan["risk_profile_lineage_id"]]
        rows.append(
            {
                "plan_id": plan["plan_id"],
                "risk_profile_lineage_id": plan["risk_profile_lineage_id"],
                "risk_level": plan["risk_level"],
                "confidence_level": str(profile["confidence_level"]),
                "plan_status": plan["plan_status"],
                "fail_risk_band": _band(float(profile["probability_fail"]), 0.35, 0.55),
                "disagreement_band": _band(
                    float(profile["deep_ml_disagreement"]), 0.10, 0.25
                ),
                "code_module": str(profile["code_module"]),
                "code_presentation": str(profile["code_presentation"]),
                "cutoff_day": int(profile["cutoff_day"]),
            }
        )
    selection_frame = pd.DataFrame(rows)
    selected_indices = _select_case_indices(selection_frame, case_count)
    selected = selection_frame.loc[selected_indices].reset_index(drop=True)
    plan_by_id = {plan["plan_id"]: plan for plan in plans}
    modules, presentations = _course_aliases(
        pd.read_parquet(ROOT / "artifacts/v6/prediction/risk_profiles.parquet")
    )
    case_rows: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    for number, row in enumerate(selected.itertuples(index=False), 1):
        case_id = f"CASE-{number:03d}"
        plan = plan_by_id[row.plan_id]
        action_ids = [action["action_id"] for action in plan["recommended_actions"]]
        actions = [
            f"{action['action_id']}: {action['action_text']} "
            f"({action['weekly_minutes']} min/week)"
            for action in plan["recommended_actions"]
        ]
        reason_codes = sorted(
            {
                reason
                for action in plan["recommended_actions"]
                for reason in action["reason_codes"]
            }
        )
        case_rows.append(
            {
                "schema_version": CASE_SCHEMA,
                "case_id": case_id,
                "course_context": modules[row.code_module],
                "offering_context": presentations[row.code_presentation],
                "course_progress_day": row.cutoff_day,
                "risk_band": row.risk_level,
                "confidence_band": row.confidence_level,
                "decision_support_status": row.plan_status,
                "fail_risk_band": row.fail_risk_band,
                "model_disagreement_band": row.disagreement_band,
                "observed_activity": plan["observed_evidence"][
                    "activity_level_band"
                ],
                "observed_inactivity_weeks": plan["observed_evidence"][
                    "inactivity_streak_weeks"
                ],
                "observed_assessment_progress": plan["observed_evidence"][
                    "assessment_progress_band"
                ],
                "observed_grade_trend": plan["observed_evidence"][
                    "grade_trend_band"
                ],
                "observed_evidence_status": "; ".join(
                    plan["partial_evidence_reasons"]
                )
                or "COMPLETE_FOR_REGISTERED_RULES",
                "reason_codes": "; ".join(reason_codes) or "NONE_ABSTAINED",
                "proposed_action_ids": "; ".join(action_ids) or "NONE_ABSTAINED",
                "proposed_actions": "\n".join(actions)
                or "No automatic action; human review is required.",
                "weekly_minutes": plan["expected_weekly_minutes"],
                "priority": plan["priority"],
                "human_review_required": "YES"
                if plan["requires_expert_review"]
                else "NO",
                "scope_note": (
                    "Historical pre-cutoff decision support; outcome and model "
                    "identity are withheld. Effectiveness is not established."
                ),
            }
        )
        registry.append(
            {
                "case_id": case_id,
                "plan_id": row.plan_id,
                "risk_profile_lineage_id": row.risk_profile_lineage_id,
                "strata": {
                    "risk": row.risk_level,
                    "confidence": row.confidence_level,
                    "abstention": row.plan_status,
                    "fail_risk": row.fail_risk_band,
                    "disagreement": row.disagreement_band,
                    "module": row.code_module,
                    "presentation": row.code_presentation,
                },
            }
        )
    cases = pd.DataFrame(case_rows)
    EXPERT_ROOT.mkdir(parents=True, exist_ok=True)
    cases.to_csv(EXPERT_ROOT / "expert_review_cases.csv", index=False)
    atomic_json(EXPERT_ROOT / "expert_review_schema.json", _review_schema())
    atomic_json(
        EXPERT_ROOT / "case_registry.json",
        {
            "schema_version": EXPERT_SCHEMA,
            "cases": registry,
            "selection_rule": "deterministic greedy marginal stratum coverage",
            "selection_dimensions": [
                "risk",
                "confidence",
                "abstention",
                "fail_risk",
                "disagreement",
                "module",
                "presentation",
            ],
            "outcome_used_for_selection": False,
            "demographics_used_for_selection": False,
        },
    )
    mapping_rows: list[dict[str, Any]] = []
    for reviewer_number, reviewer_id in enumerate(("E01", "E02"), 1):
        order = cases.sample(frac=1, random_state=6200 + reviewer_number).reset_index(
            drop=True
        )
        plan_template = pd.DataFrame(
            {
                "schema_version": EXPERT_SCHEMA,
                "reviewer_id": reviewer_id,
                "randomized_order": np.arange(1, len(order) + 1),
                "case_id": order["case_id"],
                "q1_plan_score": "",
                "q3_missing_action": "",
                "q3_missing_action_text": "",
                "q4_escalation": "",
                "q5_reason_support": "",
                "q6_safety_workload": "",
                "q6_safety_note": "",
                "reviewer_comment": "",
            }
        )
        action_template_rows: list[dict[str, Any]] = []
        for case in order.to_dict(orient="records"):
            action_ids = [
                value.strip()
                for value in str(case["proposed_action_ids"]).split(";")
                if value.strip() and value.strip() != "NONE_ABSTAINED"
            ]
            for action_id in action_ids:
                action_template_rows.append(
                    {
                        "schema_version": EXPERT_SCHEMA,
                        "reviewer_id": reviewer_id,
                        "randomized_order": int(
                            plan_template.loc[
                                plan_template["case_id"].eq(case["case_id"]),
                                "randomized_order",
                            ].iloc[0]
                        ),
                        "case_id": case["case_id"],
                        "action_id": action_id,
                        "q2_action_relevance": "",
                        "action_comment": "",
                    }
                )
            mapping_rows.append(
                {
                    "reviewer_id": reviewer_id,
                    "randomized_order": int(
                        plan_template.loc[
                            plan_template["case_id"].eq(case["case_id"]),
                            "randomized_order",
                        ].iloc[0]
                    ),
                    "case_id": case["case_id"],
                }
            )
        plan_template.to_csv(
            EXPERT_ROOT / f"plan_review_template_{reviewer_id}.csv", index=False
        )
        pd.DataFrame(action_template_rows).to_csv(
            EXPERT_ROOT / f"action_review_template_{reviewer_id}.csv", index=False
        )
        order.merge(
            plan_template[["case_id", "randomized_order"]],
            on="case_id",
            how="left",
            validate="one_to_one",
        ).sort_values("randomized_order").to_csv(
            EXPERT_ROOT / f"expert_review_cases_{reviewer_id}.csv", index=False
        )
    pd.DataFrame(mapping_rows).to_csv(
        EXPERT_ROOT / "case_randomization_map.csv", index=False
    )
    instructions = """# V6.2 blinded expert review instructions

## Purpose and scope

Evaluate whether each proposed plan is relevant, supported by the displayed
historical pre-cutoff evidence, safe, and feasible. This is a scientific review
of decision-support output. It is **not** evidence that an action causes better
student outcomes, and no student outcome is shown.

The cases hide model identity, exact prediction probabilities, source record
identifiers, student identifiers, and outcomes. Reviewer IDs must remain
pseudonymous (`E01`, `E02`, ...). Do not add names, email addresses, or other
personal information.

## Independent review

Complete the assigned randomized order independently before discussing cases
with another reviewer. Do not infer a missing behavior from a risk band. A
reason such as `LOW_VLE_ENGAGEMENT` is acceptable only when the displayed
observed pre-cutoff evidence supports it.

## Questions

1. `q1_plan_score` — overall plan quality, integer 1 (very poor) to 5 (very good).
2. `q2_action_relevance` — one value for every proposed action:
   `APPROVE`, `PARTIAL`, `UNSURE`, or `REJECT`.
3. `q3_missing_action` — `YES` or `NO`; if `YES`, describe the omitted action.
4. `q4_escalation` — `CORRECT`, `OVER_ESCALATED`, `UNDER_ESCALATED`, or `UNSURE`.
5. `q5_reason_support` — `SUPPORTED`, `PARTIAL`, `UNSUPPORTED`, or `UNSURE`.
6. `q6_safety_workload` — `SAFE`, `CONCERN`, `UNSAFE`, or `UNSURE`; a note is
   required for `CONCERN` or `UNSAFE`.

Do not change `schema_version`, `reviewer_id`, `case_id`, `action_id`, or
`randomized_order`. Blank templates are intentionally pending and contain no
synthetic labels.
"""
    atomic_text(EXPERT_ROOT / "expert_review_instructions.md", instructions)
    atomic_text(REPORT_ROOT / "EXPERT_REVIEW_HANDOFF.md", instructions)
    status = {
        "schema_version": EXPERT_SCHEMA,
        "status": "PENDING_REAL_EXPERT_LABELS",
        "cases": len(cases),
        "reviewers_prepared": 2,
        "model_identity_exposed": False,
        "exact_probabilities_exposed": False,
        "internal_record_ids_exposed": False,
        "outcomes_exposed": False,
        "demographics_exposed": False,
        "synthetic_labels_created": False,
        "llm_labels_created": False,
        "heuristic_labels_created": False,
        "package_sha256": canonical_sha256(case_rows),
    }
    atomic_json(EXPERT_ROOT / "expert_review_status.json", status)
    return status


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value)) or str(
        value
    ).strip() == ""


def _read_review(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        if path.suffix.lower() == ".xlsx":
            plan = pd.read_excel(path, sheet_name=PLAN_SHEET, dtype=object)
            action = pd.read_excel(path, sheet_name=ACTION_SHEET, dtype=object)
            return plan, action
        if path.is_dir():
            plan_files = sorted(path.glob("plan_review*.csv"))
            action_files = sorted(path.glob("action_review*.csv"))
            if not plan_files or not action_files:
                raise ReviewImportError("Review directory lacks plan/action CSV files")
            return (
                pd.concat(
                    [pd.read_csv(item, dtype=object) for item in plan_files],
                    ignore_index=True,
                ),
                pd.concat(
                    [pd.read_csv(item, dtype=object) for item in action_files],
                    ignore_index=True,
                ),
            )
    except ReviewImportError:
        raise
    except Exception as exc:  # corrupt workbooks must fail loudly
        raise ReviewImportError(f"Cannot read expert review file: {path.name}") from exc
    raise ReviewImportError("Expert review input must be an .xlsx file or directory")


def _assert_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame)
    if missing:
        raise ReviewImportError(f"{label} fields missing: {sorted(missing)}")


def _validate_reviews(
    plan: pd.DataFrame,
    action: pd.DataFrame,
    cases: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plan_required = set(_review_schema()["plan_review"]["required"])
    action_required = set(_review_schema()["action_review"]["required"])
    _assert_columns(plan, plan_required, "Plan review")
    _assert_columns(action, action_required, "Action review")
    if plan.empty:
        raise ReviewImportError("Expert review contains no plan rows")
    answer_fields = plan_required - {"schema_version", "reviewer_id", "case_id"}
    if plan[list(answer_fields)].map(_is_blank).all(axis=None):
        return plan.iloc[0:0].copy(), action.iloc[0:0].copy()
    conditionally_blank = {"q3_missing_action_text", "q6_safety_note"}
    nonblank_required = plan_required - conditionally_blank
    if plan[list(nonblank_required)].map(_is_blank).any(axis=None):
        raise ReviewImportError("Completed plan review has missing required values")
    if not action.empty and action[list(action_required)].map(_is_blank).any(axis=None):
        raise ReviewImportError("Completed action review has missing required values")
    if not plan["schema_version"].eq(EXPERT_SCHEMA).all() or (
        not action.empty and not action["schema_version"].eq(EXPERT_SCHEMA).all()
    ):
        raise ReviewImportError("Expert review schema version mismatch")
    reviewers = set(plan["reviewer_id"].astype(str))
    if not reviewers or any(not REVIEWER_PATTERN.fullmatch(value) for value in reviewers):
        raise ReviewImportError("Reviewer IDs must be pseudonymous E## identifiers")
    if not set(action["reviewer_id"].astype(str)).issubset(reviewers):
        raise ReviewImportError("Action review contains an unknown reviewer")
    if plan.duplicated(["reviewer_id", "case_id"]).any():
        raise ReviewImportError("Duplicate reviewer/case plan review")
    if action.duplicated(["reviewer_id", "case_id", "action_id"]).any():
        raise ReviewImportError("Duplicate reviewer/case/action review")
    known_cases = set(cases["case_id"].astype(str))
    if not set(plan["case_id"].astype(str)).issubset(known_cases) or not set(
        action["case_id"].astype(str)
    ).issubset(known_cases):
        raise ReviewImportError("Expert review contains an unknown case")
    scores: list[int] = []
    for value in plan["q1_plan_score"]:
        if isinstance(value, bool):
            raise ReviewImportError("Plan score must be an integer 1-5")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ReviewImportError("Plan score must be an integer 1-5") from exc
        if not numeric.is_integer() or int(numeric) not in PLAN_SCORE_VALUES:
            raise ReviewImportError("Plan score must be an integer 1-5")
        scores.append(int(numeric))
    plan = plan.copy()
    plan["q1_plan_score"] = scores
    enum_checks = {
        "q3_missing_action": YES_NO_VALUES,
        "q4_escalation": ESCALATION_VALUES,
        "q5_reason_support": REASON_SUPPORT_VALUES,
        "q6_safety_workload": SAFETY_VALUES,
    }
    for column, allowed in enum_checks.items():
        if not plan[column].astype(str).isin(allowed).all():
            raise ReviewImportError(f"Invalid enum value in {column}")
    if not action.empty and not action["q2_action_relevance"].astype(str).isin(
        ACTION_RELEVANCE_VALUES
    ).all():
        raise ReviewImportError("Invalid enum value in q2_action_relevance")
    for row in plan.itertuples(index=False):
        if row.q3_missing_action == "YES" and _is_blank(row.q3_missing_action_text):
            raise ReviewImportError("Missing-action text is required when answer is YES")
        if row.q6_safety_workload in {"CONCERN", "UNSAFE"} and _is_blank(
            row.q6_safety_note
        ):
            raise ReviewImportError("Safety note is required for concern/unsafe")
    proposed = {
        str(row.case_id): {
            value.strip()
            for value in str(row.proposed_action_ids).split(";")
            if value.strip() and value.strip() != "NONE_ABSTAINED"
        }
        for row in cases.itertuples(index=False)
    }
    for row in action.itertuples(index=False):
        if str(row.action_id) not in proposed[str(row.case_id)]:
            raise ReviewImportError("Action review contains an action not proposed for case")
    expected_pairs = {
        (reviewer, case_id, action_id)
        for reviewer in reviewers
        for case_id, action_ids in proposed.items()
        if case_id in set(plan.loc[plan["reviewer_id"].eq(reviewer), "case_id"].astype(str))
        for action_id in action_ids
    }
    actual_pairs = {
        (str(row.reviewer_id), str(row.case_id), str(row.action_id))
        for row in action.itertuples(index=False)
    }
    if expected_pairs != actual_pairs:
        raise ReviewImportError("Action review coverage is incomplete or excessive")
    return plan, action


def _pairwise_kappa(
    frame: pd.DataFrame,
    column: str,
    *,
    item_columns: list[str],
    weights: str | None = None,
) -> dict[str, Any]:
    reviewers = sorted(frame["reviewer_id"].astype(str).unique())
    values: list[float] = []
    pairs: list[dict[str, Any]] = []
    for first, second in combinations(reviewers, 2):
        left = frame.loc[frame["reviewer_id"].eq(first)].set_index(item_columns)
        right = frame.loc[frame["reviewer_id"].eq(second)].set_index(item_columns)
        shared = left.index.intersection(right.index)
        if not len(shared):
            continue
        score = float(
            cohen_kappa_score(
                left.loc[shared, column],
                right.loc[shared, column],
                weights=weights,
            )
        )
        if np.isfinite(score):
            values.append(score)
            pairs.append(
                {
                    "reviewer_a": first,
                    "reviewer_b": second,
                    "shared_items": len(shared),
                    "kappa": score,
                }
            )
    return {
        "status": "COMPLETE" if values else "PENDING_AT_LEAST_TWO_REVIEWERS",
        "mean_pairwise_kappa": float(np.mean(values)) if values else None,
        "pairs": pairs,
    }


def _metric_snapshot(plan: pd.DataFrame, action: pd.DataFrame) -> dict[str, float]:
    scores = plan["q1_plan_score"].to_numpy(dtype=float)
    action_values = action["q2_action_relevance"].astype(str)
    escalation = plan["q4_escalation"].astype(str)
    return {
        "plan_score_mean": float(np.mean(scores)),
        "plan_score_median": float(np.median(scores)),
        "plan_score_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "plan_score_ge_4_rate": float(np.mean(scores >= 4)),
        "plan_score_le_2_rate": float(np.mean(scores <= 2)),
        "action_approve_rate": float(np.mean(action_values.eq("APPROVE")))
        if len(action_values)
        else float("nan"),
        "action_partial_rate": float(np.mean(action_values.eq("PARTIAL")))
        if len(action_values)
        else float("nan"),
        "action_unsure_rate": float(np.mean(action_values.eq("UNSURE")))
        if len(action_values)
        else float("nan"),
        "action_reject_rate": float(np.mean(action_values.eq("REJECT")))
        if len(action_values)
        else float("nan"),
        "missing_action_rate": float(
            np.mean(plan["q3_missing_action"].astype(str).eq("YES"))
        ),
        "escalation_correct_rate": float(np.mean(escalation.eq("CORRECT"))),
        "over_escalation_rate": float(np.mean(escalation.eq("OVER_ESCALATED"))),
        "under_escalation_rate": float(np.mean(escalation.eq("UNDER_ESCALATED"))),
    }


def _bootstrap(
    plan: pd.DataFrame,
    action: pd.DataFrame,
    *,
    draws: int = 2000,
    seed: int = 6202,
) -> dict[str, Any]:
    cases = sorted(plan["case_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    metrics: defaultdict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        sampled = rng.choice(cases, size=len(cases), replace=True)
        plan_parts: list[pd.DataFrame] = []
        action_parts: list[pd.DataFrame] = []
        for occurrence, case_id in enumerate(sampled):
            plan_part = plan.loc[plan["case_id"].astype(str).eq(case_id)].copy()
            action_part = action.loc[action["case_id"].astype(str).eq(case_id)].copy()
            # Resample cases, not individual reviewer rows; every reviewer for a
            # selected case stays together. Occurrence distinguishes duplicates.
            plan_part["_bootstrap_case"] = f"{occurrence}:{case_id}"
            action_part["_bootstrap_case"] = f"{occurrence}:{case_id}"
            plan_parts.append(plan_part)
            action_parts.append(action_part)
        snapshot = _metric_snapshot(
            pd.concat(plan_parts, ignore_index=True),
            pd.concat(action_parts, ignore_index=True),
        )
        for name, value in snapshot.items():
            if np.isfinite(value):
                metrics[name].append(value)
    return {
        name: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        }
        for name, values in sorted(metrics.items())
    }


def import_and_score_expert_reviews(path: Path | None = None) -> dict[str, Any]:
    cases_path = EXPERT_ROOT / "expert_review_cases.csv"
    if not cases_path.is_file():
        export_expert_package()
    cases = pd.read_csv(cases_path, dtype=object)
    if path is None or not path.exists():
        result = {
            "schema_version": EXPERT_SCHEMA,
            "status": "PENDING_REAL_EXPERT_LABELS",
            "reviewers": 0,
            "cases_scored": 0,
            "plan_metrics": None,
            "action_metrics": None,
            "omission_metric": {
                "name": "expert_reported_missing_action_rate",
                "value": None,
                "not_action_recall": True,
            },
            "inter_rater": {
                "status": "PENDING_AT_LEAST_TWO_REVIEWERS",
            },
            "bootstrap": None,
            "synthetic_labels_created": False,
        }
        atomic_json(EXPERT_ROOT / "expert_metrics.json", result)
        return result
    raw_plan, raw_action = _read_review(path)
    plan, action = _validate_reviews(raw_plan, raw_action, cases)
    if plan.empty:
        return import_and_score_expert_reviews(None)
    snapshot = _metric_snapshot(plan, action)
    categories = (
        action.merge(
            cases[["case_id", "proposed_action_ids"]],
            on="case_id",
            how="left",
            validate="many_to_one",
        )
        if not action.empty
        else action
    )
    per_action: dict[str, Any] = {}
    if not categories.empty:
        for action_id, group in categories.groupby("action_id", sort=True):
            values = group["q2_action_relevance"].astype(str)
            per_action[str(action_id)] = {
                "reviews": len(group),
                "approve_rate": float(values.eq("APPROVE").mean()),
                "partial_rate": float(values.eq("PARTIAL").mean()),
                "unsure_rate": float(values.eq("UNSURE").mean()),
                "reject_rate": float(values.eq("REJECT").mean()),
            }
    result = {
        "schema_version": EXPERT_SCHEMA,
        "status": "COMPLETE_REAL_EXPERT_LABELS",
        "reviewers": int(plan["reviewer_id"].nunique()),
        "cases_scored": int(plan["case_id"].nunique()),
        "case_coverage": float(plan["case_id"].nunique() / len(cases)),
        "plan_metrics": {
            key: value for key, value in snapshot.items() if key.startswith("plan_")
        },
        "action_metrics": {
            key: value
            for key, value in snapshot.items()
            if key.startswith("action_")
        }
        | {"per_action_category": per_action},
        "omission_metric": {
            "name": "expert_reported_missing_action_rate",
            "value": snapshot["missing_action_rate"],
            "not_action_recall": True,
        },
        "escalation_metrics": {
            key: value
            for key, value in snapshot.items()
            if "escalation" in key
        },
        "inter_rater": {
            "plan_score_weighted_kappa": _pairwise_kappa(
                plan,
                "q1_plan_score",
                item_columns=["case_id"],
                weights="quadratic",
            ),
            "action_relevance_nominal_kappa": _pairwise_kappa(
                action,
                "q2_action_relevance",
                item_columns=["case_id", "action_id"],
            ),
            "escalation_nominal_kappa": _pairwise_kappa(
                plan,
                "q4_escalation",
                item_columns=["case_id"],
            ),
        },
        "bootstrap": {
            "unit": "case_id with all reviewer rows retained",
            "draws": 2000,
            "seed": 6202,
            "confidence_intervals": _bootstrap(plan, action),
        },
        "synthetic_labels_created": False,
    }
    atomic_json(EXPERT_ROOT / "expert_metrics.json", result)
    return result


__all__ = [
    "ACTION_RELEVANCE_VALUES",
    "CASE_SCHEMA",
    "EXPERT_ROOT",
    "EXPERT_SCHEMA",
    "ReviewImportError",
    "export_expert_package",
    "import_and_score_expert_reviews",
]
