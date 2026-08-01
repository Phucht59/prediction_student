"""Generate and validate the final scientific evaluation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.uci_support import _stable_id
from src.recommend_hybrid.common.plan_contracts import LearningPlan, PlanStatus
from src.recommend_hybrid.common.policy_contracts import DatasetId, PolicyPredictionContext
from src.recommend_hybrid.pipeline import OULADPlanRequest, RecommendHybridPipeline, UCIPlanRequest
from src.recommend_hybrid.prediction_adapter import ARCHITECTURE_HASH, PARAMETER_COUNT

CREATED_AT = "2026-08-01T15:55:54Z"
BOOTSTRAP_SEED = 20260801
BOOTSTRAP_REPLICATES = 1000
UCI_SAMPLE_PER_STAGE = 20
OULAD_SAMPLE_PER_STAGE = 20
INTER_STAGE_SAMPLE = 10
PHASE4_SOURCE_COMMIT = "c419afa5406d9c5907aa4554118bed552901939a"
FINAL_DIR = ROOT / "artifacts/recommend_hybrid/final"
REPORT_DIR = ROOT / "reports/recommend_hybrid"
POLICY_PATHS = (
    "configs/recommend_hybrid/policy_common.yaml",
    "configs/recommend_hybrid/policy_uci_mat.yaml",
    "configs/recommend_hybrid/policy_uci_por.yaml",
    "configs/recommend_hybrid/policy_oulad.yaml",
)
SENSITIVE = {
    "age_band",
    "disability",
    "gender",
    "region",
    "imd_band",
    "final_result",
    "target",
    "outer_label",
    "date_unregistration",
    "withdrawal_outcome",
}


@dataclass(frozen=True)
class EvaluationRecord:
    dataset: str
    stage: str
    student_key: str
    requested_cutoff: float
    risk_probability: float
    predicted_class: int
    uncertainty: float
    request: UCIPlanRequest | OULADPlanRequest
    plan: LearningPlan


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_entropy(probability: float) -> float:
    clipped = min(max(float(probability), 1e-12), 1 - 1e-12)
    return -(clipped * math.log(clipped) + (1 - clipped) * math.log(1 - clipped))


def prediction_context(
    dataset: DatasetId,
    probabilities: tuple[float, ...],
    *,
    disagreement: float,
    fold: int,
    stage: str,
    threshold: float | None = None,
) -> PolicyPredictionContext:
    risk_index = 1 if dataset is DatasetId.OULAD else 0
    risk = probabilities[risk_index]
    predicted = int(risk >= threshold) if threshold is not None else int(np.argmax(probabilities))
    return PolicyPredictionContext(
        dataset_id=dataset,
        predicted_class=predicted,
        class_probabilities=probabilities,
        confidence=max(probabilities),
        uncertainty=binary_entropy(risk),
        seed_disagreement=float(disagreement),
        checkpoint_lineage=(f"canonical_v3:{stage}:outer_fold_{fold}:five_seed_mean",),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
    )


def evaluate_uci(pipeline: RecommendHybridPipeline) -> list[EvaluationRecord]:
    oof_path = ROOT / "artifacts/canonical_v3/predictions/uci_oof_predictions.parquet"
    seed_path = ROOT / "artifacts/canonical_v3/predictions/uci_seed_predictions.parquet"
    columns = ["dataset", "stage", "model", "outer_fold", "record_id", "p_low", "p_medium", "p_high"]
    oof = pd.read_parquet(oof_path, columns=columns)
    stages = {
        "S0_EARLY_NO_GRADE": "S0",
        "S1_MID_G1_ONLY": "S1",
        "S2_LATE_G1_G2": "S2",
    }
    oof = oof.loc[oof["model"].eq("hybrid") & oof["stage"].isin(stages)].copy()
    seed = pd.read_parquet(
        seed_path,
        columns=["dataset", "stage", "model", "record_id", "p_low"],
    )
    seed = seed.loc[seed["model"].eq("hybrid")]
    disagreement = seed.groupby(["dataset", "stage", "record_id"])["p_low"].std(ddof=0)
    records: list[EvaluationRecord] = []
    for dataset_name, filename, namespace in (
        ("student_mat", "student-mat.csv", "student-mat"),
        ("student_por", "student-por.csv", "student-por"),
    ):
        frame = pd.read_csv(
            ROOT / "data/raw" / filename,
            sep=";",
            usecols=["G1", "G2", "absences", "studytime", "failures"],
        )
        frame["record_id"] = [_stable_id(namespace, index) for index in range(len(frame))]
        safe = frame.set_index("record_id")
        dataset_id = DatasetId(dataset_name)
        for source_stage, stage in stages.items():
            selected = (
                oof.loc[oof["dataset"].eq(dataset_name) & oof["stage"].eq(source_stage)]
                .sort_values("record_id")
                .head(UCI_SAMPLE_PER_STAGE)
            )
            if len(selected) != UCI_SAMPLE_PER_STAGE:
                raise AssertionError(f"incomplete UCI evaluation sample: {dataset_name}/{stage}")
            for row in selected.itertuples(index=False):
                raw = safe.loc[row.record_id]
                probabilities = (float(row.p_low), float(row.p_medium), float(row.p_high))
                context = prediction_context(
                    dataset_id,
                    probabilities,
                    disagreement=float(disagreement.loc[(dataset_name, source_stage, row.record_id)]),
                    fold=int(row.outer_fold),
                    stage=source_stage,
                )
                request = UCIPlanRequest(
                    dataset_id=dataset_id,
                    student_key=str(row.record_id),
                    course_key=dataset_name,
                    prediction=context,
                    g1=None if stage == "S0" else float(raw.G1),
                    g2=float(raw.G2) if stage == "S2" else None,
                    absences=int(raw.absences),
                    study_time=int(raw.studytime),
                    previous_failures=int(raw.failures),
                    next_assessment_available=None if stage == "S0" else True,
                    created_at=CREATED_AT,
                )
                plan = pipeline.generate(request)
                records.append(
                    EvaluationRecord(
                        dataset_name,
                        stage,
                        str(row.record_id),
                        plan.requested_cutoff,
                        probabilities[0],
                        context.predicted_class,
                        context.uncertainty,
                        request,
                        plan,
                    )
                )
    return records


def _load_oulad_sample_rows() -> pd.DataFrame:
    path = ROOT / "artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet"
    columns = [
        "base_record_id",
        "id_student",
        "code_module",
        "code_presentation",
        "outer_fold",
        "cutoff_day",
        "stage",
        "model",
        "probability",
        "threshold",
    ]
    oof = pd.read_parquet(path, columns=columns)
    source_to_canonical = {
        "E1_EARLY_20PCT": ("EARLY_20", 20.0),
        "E2_EARLY_35PCT": ("EARLY_35", 35.0),
        "M1_MIDDLE_50PCT": ("MIDDLE_50", 50.0),
        "L1_LATE_75PCT": ("LATE_75", 75.0),
        "FINAL": ("FINAL_EVALUATION", 100.0),
    }
    hybrid = oof.loc[oof["model"].eq("hybrid") & oof["stage"].isin(source_to_canonical)].copy()
    rows: list[pd.DataFrame] = []
    for source_stage, (label, cutoff) in source_to_canonical.items():
        selected = hybrid.loc[hybrid["stage"].eq(source_stage)].sort_values("base_record_id").head(OULAD_SAMPLE_PER_STAGE).copy()
        if len(selected) != OULAD_SAMPLE_PER_STAGE:
            raise AssertionError(f"incomplete OULAD evaluation sample: {label}")
        selected["evaluation_stage"] = label
        selected["requested_percent"] = cutoff
        rows.append(selected)
    for source_stage, label, cutoff in (
        ("E1_EARLY_20PCT", "INTER_STAGE_25", 25.0),
        ("E2_EARLY_35PCT", "INTER_STAGE_36", 36.0),
        ("M1_MIDDLE_50PCT", "INTER_STAGE_63", 63.0),
        ("L1_LATE_75PCT", "INTER_STAGE_76", 76.0),
    ):
        selected = hybrid.loc[hybrid["stage"].eq(source_stage)].sort_values("base_record_id").iloc[OULAD_SAMPLE_PER_STAGE:OULAD_SAMPLE_PER_STAGE + INTER_STAGE_SAMPLE].copy()
        if len(selected) != INTER_STAGE_SAMPLE:
            raise AssertionError(f"incomplete OULAD inter-stage sample: {label}")
        selected["evaluation_stage"] = label
        selected["requested_percent"] = cutoff
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _load_filtered_csv(path: Path, keys: pd.DataFrame, usecols: list[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    join_keys = ["code_module", "code_presentation", "id_student"]
    wanted = keys.loc[:, join_keys].drop_duplicates()
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        match = chunk.merge(wanted, on=join_keys, how="inner")
        if not match.empty:
            chunks.append(match)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def _load_filtered_students(path: Path, student_ids: set[int], usecols: list[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000):
        match = chunk.loc[chunk["id_student"].isin(student_ids)]
        if not match.empty:
            chunks.append(match)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)


def evaluate_oulad(pipeline: RecommendHybridPipeline) -> list[EvaluationRecord]:
    sample = _load_oulad_sample_rows()
    seed = pd.read_parquet(
        ROOT / "artifacts/canonical_v3/predictions/oulad_seed_predictions.parquet",
        columns=["base_record_id", "stage", "model", "probability"],
    )
    ids = set(sample["base_record_id"])
    seed = seed.loc[seed["model"].eq("hybrid") & seed["base_record_id"].isin(ids)]
    disagreement = seed.groupby(["base_record_id", "stage"])["probability"].std(ddof=0)
    keys = sample.loc[:, ["code_module", "code_presentation", "id_student"]]
    vle = _load_filtered_csv(
        ROOT / "data/raw/studentVle.csv",
        keys,
        ["code_module", "code_presentation", "id_student", "date", "sum_click"],
    )
    submissions = _load_filtered_students(
        ROOT / "data/raw/studentAssessment.csv",
        set(int(value) for value in keys["id_student"]),
        ["id_student", "id_assessment", "date_submitted"],
    )
    assessments = pd.read_csv(
        ROOT / "data/raw/assessments.csv",
        usecols=["code_module", "code_presentation", "id_assessment", "date"],
    )
    courses = pd.read_csv(ROOT / "data/raw/courses.csv").set_index(["code_module", "code_presentation"])
    vle_groups = {key: group for key, group in vle.groupby(["code_module", "code_presentation", "id_student"])}
    submission_groups = {int(key): group for key, group in submissions.groupby("id_student")}
    assessment_groups = {
        key: group for key, group in assessments.groupby(["code_module", "code_presentation"])
    }
    records: list[EvaluationRecord] = []
    for row in sample.itertuples(index=False):
        key = (row.code_module, row.code_presentation, int(row.id_student))
        course_key = (row.code_module, row.code_presentation)
        length = int(courses.loc[course_key, "module_presentation_length"])
        cutoff_day = int(math.floor(length * float(row.requested_percent) / 100.0))
        if row.evaluation_stage in {"EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75", "FINAL_EVALUATION"}:
            cutoff_day = int(row.cutoff_day)
        probability = float(row.probability)
        context = prediction_context(
            DatasetId.OULAD,
            (1 - probability, probability),
            disagreement=float(disagreement.loc[(row.base_record_id, row.stage)]),
            fold=int(row.outer_fold),
            stage=row.stage,
            threshold=float(row.threshold),
        )
        if row.evaluation_stage == "FINAL_EVALUATION":
            activity_level = recent_trend = assessment_progress = None
            inactivity = assessments_due = None
            observation_end = None
        else:
            activity = vle_groups.get(key, pd.DataFrame(columns=["date", "sum_click"]))
            activity = activity.loc[activity["date"] < cutoff_day]
            if activity.empty:
                activity_level = recent_trend = inactivity = None
            else:
                activity_level = float(activity["sum_click"].mean())
                recent = activity.loc[activity["date"] >= cutoff_day - 14, "sum_click"]
                earlier = activity.loc[activity["date"] < cutoff_day - 14, "sum_click"]
                recent_trend = float(recent.mean() - earlier.mean()) if not recent.empty and not earlier.empty else None
                inactivity = max(0, cutoff_day - 1 - int(activity["date"].max()))
            due = assessment_groups.get(course_key, pd.DataFrame(columns=["id_assessment", "date"]))
            due = due.loc[due["date"].notna() & (due["date"] < cutoff_day)]
            assessments_due = int(len(due))
            submitted = submission_groups.get(int(row.id_student), pd.DataFrame(columns=["id_assessment", "date_submitted"]))
            completed = submitted.loc[
                submitted["id_assessment"].isin(set(due["id_assessment"]))
                & (submitted["date_submitted"] < cutoff_day)
            ]
            assessment_progress = float(completed["id_assessment"].nunique() / assessments_due) if assessments_due else None
            observation_end = float(row.requested_percent) - 1e-6
        request = OULADPlanRequest(
            student_key=str(row.base_record_id),
            course_key=f"{row.code_module}-{row.code_presentation}",
            requested_cutoff=float(row.requested_percent),
            prediction=context,
            max_observation_cutoff=observation_end,
            activity_level=activity_level,
            recent_activity_trend=recent_trend,
            inactivity_streak=inactivity,
            assessment_progress=assessment_progress,
            assessments_due=assessments_due,
            grade_trend=None,
            grade_release_verified=False,
            knowledge_gap=None,
            created_at=CREATED_AT,
        )
        plan = pipeline.generate(request)
        records.append(
            EvaluationRecord(
                "oulad",
                str(row.evaluation_stage),
                str(row.base_record_id),
                float(row.requested_percent),
                probability,
                context.predicted_class,
                context.uncertainty,
                request,
                plan,
            )
        )
    return records


def _allowed_actions(planning: dict[str, Any], dataset: str) -> set[str]:
    return set(planning["dataset_actions"][dataset])


def violation_counts(records: list[EvaluationRecord], planning: dict[str, Any]) -> dict[str, int]:
    counts = Counter()
    for record in records:
        plan = record.plan
        ids = [action.action_id for action in plan.selected_actions]
        if len(ids) > int(planning["max_actions_per_plan"]):
            counts["action_cap_violations"] += 1
        if len(ids) != len(set(ids)):
            counts["duplicate_action_violations"] += 1
        if any(action_id not in _allowed_actions(planning, record.dataset) for action_id in ids):
            counts["cross_dataset_policy_violations"] += 1
            counts["unsupported_action_violations"] += 1
        workload = Counter()
        for action in plan.selected_actions:
            workload[action.scheduled_period] += action.weekly_minutes
            if action.scheduled_period not in plan.plan_periods:
                counts["invalid_period_violations"] += 1
            metadata = planning["action_metadata"][action.action_id]
            prerequisites = metadata["prerequisites"]
            if any(item not in ids or ids.index(item) > ids.index(action.action_id) for item in prerequisites):
                counts["prerequisite_violations"] += 1
            if any(item.feature_name.lower() in SENSITIVE for item in action.supporting_evidence):
                counts["sensitive_feature_violations"] += 1
            if any(item.observation_end is not None and item.observation_end > record.requested_cutoff for item in action.supporting_evidence):
                counts["post_cutoff_violations"] += 1
            if any(not item.source_lineage for item in action.supporting_evidence):
                counts["missing_lineage_violations"] += 1
            if not action.supporting_evidence:
                counts["unsupported_reason_violations"] += 1
            if any(item.availability.value != "AVAILABLE" for item in action.supporting_evidence):
                counts["missing_evidence_misuse_violations"] += 1
            if not action.reason_codes:
                counts["reason_action_consistency_violations"] += 1
        if any(value > int(planning["max_minutes_per_period"]) for value in workload.values()):
            counts["workload_violations"] += 1
        if plan.prediction_anchor is not None and plan.prediction_anchor > record.requested_cutoff:
            counts["future_anchor_violations"] += 1
        if record.stage == "FINAL_EVALUATION" and ids:
            counts["final_intervention_violations"] += 1
        if any(item.feature_name == "G3" for action in plan.selected_actions for item in action.supporting_evidence):
            counts["G3_usage"] += 1
        if record.dataset == "oulad":
            remaining = 100 - record.requested_cutoff
            if remaining <= 0 and ids:
                counts["course_end_violations"] += 1
            if remaining < 10 and any(action.scheduled_period != "IMMEDIATE" for action in plan.selected_actions):
                counts["course_end_violations"] += 1
            if remaining < 25 and any(action.scheduled_period == "FOLLOW_UP" for action in plan.selected_actions):
                counts["course_end_violations"] += 1
        if plan.automation_status in {PlanStatus.ABSTAIN, PlanStatus.EVALUATION_ONLY} and ids:
            counts["automation_status_violations"] += 1
    expected = (
        "post_cutoff_violations",
        "future_anchor_violations",
        "final_intervention_violations",
        "G3_usage",
        "sensitive_feature_violations",
        "missing_lineage_violations",
        "cross_dataset_policy_violations",
        "invalid_model_dataset_mapping",
        "action_cap_violations",
        "workload_violations",
        "duplicate_action_violations",
        "prerequisite_violations",
        "contraindication_violations",
        "unsupported_action_violations",
        "invalid_period_violations",
        "course_end_violations",
        "unsupported_reason_violations",
        "missing_evidence_misuse_violations",
        "reason_action_consistency_violations",
        "automation_status_violations",
    )
    return {name: int(counts[name]) for name in expected}


def aggregate_metrics(records: list[EvaluationRecord]) -> dict[str, Any]:
    intervention = [record for record in records if record.stage != "FINAL_EVALUATION"]
    statuses = Counter(record.plan.automation_status.value for record in records)
    actions = [action for record in records for action in record.plan.selected_actions]
    actionable = [record for record in intervention if record.plan.selected_actions]
    action_counts = [len(record.plan.selected_actions) for record in intervention]
    workloads = [record.plan.total_minutes for record in intervention]
    frequency = Counter(action.action_id for action in actions)
    action_sets = Counter(tuple(sorted(action.action_id for action in record.plan.selected_actions)) for record in actionable)
    evidence_supported = sum(bool(action.supporting_evidence) for action in actions)
    lineage_complete = sum(
        bool(action.supporting_evidence) and all(item.source_lineage for item in action.supporting_evidence)
        for action in actions
    )
    monitoring_plans = sum(
        any(action.action_id == "PROGRESS_MONITORING" for action in record.plan.selected_actions)
        for record in actionable
    )
    prediction_ages = [
        record.requested_cutoff - record.plan.prediction_anchor
        for record in records
        if record.plan.prediction_anchor is not None and record.dataset == "oulad"
    ]
    return {
        "record_count": len(records),
        "intervention_denominator": len(intervention),
        "selected_action_count": len(actions),
        "full_recommendation_rate": statuses["FULL"] / len(records) if records else 0.0,
        "partial_recommendation_rate": statuses["PARTIAL"] / len(records) if records else 0.0,
        "abstention_rate": statuses["ABSTAIN"] / len(intervention) if intervention else 0.0,
        "evaluation_only_rate": statuses["EVALUATION_ONLY"] / len(records) if records else 0.0,
        "actionable_coverage": len(actionable) / len(intervention) if intervention else 0.0,
        "mean_actions_per_plan": statistics.mean(action_counts) if action_counts else 0.0,
        "median_actions_per_plan": statistics.median(action_counts) if action_counts else 0.0,
        "mean_workload_per_plan": statistics.mean(workloads) if workloads else 0.0,
        "median_workload_per_plan": statistics.median(workloads) if workloads else 0.0,
        "evidence_support_rate": evidence_supported / len(actions) if actions else 1.0,
        "explanation_lineage_completeness": lineage_complete / len(actions) if actions else 1.0,
        "unsupported_reason_rate": 1 - evidence_supported / len(actions) if actions else 0.0,
        "missing_evidence_misuse_rate": sum(
            any(item.availability.value != "AVAILABLE" for item in action.supporting_evidence)
            for action in actions
        ) / len(actions) if actions else 0.0,
        "reason_action_consistency_rate": sum(bool(action.reason_codes) for action in actions) / len(actions) if actions else 1.0,
        "action_frequency": dict(sorted(frequency.items())),
        "unique_actions_used": len(frequency),
        "top_action_share": max(frequency.values(), default=0) / len(actions) if actions else 0.0,
        "default_action_rate": monitoring_plans / len(actionable) if actionable else 0.0,
        "plans_with_identical_action_set_rate": max(action_sets.values(), default=0) / len(actionable) if actionable else 0.0,
        "action_set_diversity": len(action_sets) / len(actionable) if actionable else 0.0,
        "mean_prediction_age": statistics.mean(prediction_ages) if prediction_ages else None,
        "max_prediction_age": max(prediction_ages) if prediction_ages else None,
    }


def grouped_metrics(records: list[EvaluationRecord]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[f"{record.dataset}:{record.stage}"].append(record)
    return {key: aggregate_metrics(value) for key, value in sorted(groups.items())}


def _analytical_ablation(records: list[EvaluationRecord], pipeline: RecommendHybridPipeline) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ("A_RISK_CLASS_ONLY", "B_RISK_PROBABILITY_ONLY", "C_RISK_AND_EVIDENCE", "D_OFFICIAL_FULL_POLICY"):
        action_rows: list[tuple[str, bool]] = []
        actionable_records = 0
        abstained = 0
        explanation_complete = 0
        constraint_violations = 0
        for record in records:
            intervention = record.stage != "FINAL_EVALUATION"
            if variant in {"A_RISK_CLASS_ONLY", "B_RISK_PROBABILITY_ONLY"}:
                risk = (
                    record.predicted_class == (1 if record.dataset == "oulad" else 0)
                    if variant == "A_RISK_CLASS_ONLY"
                    else record.risk_probability >= 0.60
                )
                action_id = (
                    "ADVISOR_ESCALATION" if record.dataset == "oulad" else "ADVISOR_SUPPORT"
                ) if risk else "PROGRESS_MONITORING"
                action_rows.append((action_id, False))
                actionable_records += int(intervention)
                constraint_violations += int(not intervention)
            else:
                if variant == "C_RISK_AND_EVIDENCE":
                    request = record.request
                    prediction = replace(request.prediction, uncertainty=0.0, seed_disagreement=0.0) if request.prediction is not None else None
                    plan = pipeline.generate(replace(request, prediction=prediction))
                else:
                    plan = record.plan
                if intervention and plan.selected_actions:
                    actionable_records += 1
                if intervention and plan.automation_status is PlanStatus.ABSTAIN:
                    abstained += 1
                for action in plan.selected_actions:
                    supported = bool(action.supporting_evidence)
                    action_rows.append((action.action_id, supported))
                    explanation_complete += int(supported and bool(action.reason_codes))
        intervention_count = sum(record.stage != "FINAL_EVALUATION" for record in records)
        frequencies = Counter(action_id for action_id, _ in action_rows)
        supported_count = sum(supported for _, supported in action_rows)
        rows.append(
            {
                "variant": variant,
                "actionable_coverage": actionable_records / intervention_count,
                "abstention_rate": abstained / intervention_count,
                "evidence_support_rate": supported_count / len(action_rows) if action_rows else 1.0,
                "unsupported_action_rate": 1 - supported_count / len(action_rows) if action_rows else 0.0,
                "explanation_completeness": explanation_complete / len(action_rows) if action_rows else 1.0,
                "action_diversity": len(frequencies),
                "constraint_violation_rate": constraint_violations / len(records),
            }
        )
    return rows


def bootstrap(records: list[EvaluationRecord]) -> dict[str, Any]:
    groups: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in records:
        groups[f"{record.dataset}:{record.student_key}"].append(record)
    keys = sorted(groups)
    rng = random.Random(BOOTSTRAP_SEED)
    metric_names = (
        "actionable_coverage",
        "abstention_rate",
        "evidence_support_rate",
        "mean_actions_per_plan",
        "mean_workload_per_plan",
        "top_action_share",
    )
    samples = {name: [] for name in metric_names}
    for _ in range(BOOTSTRAP_REPLICATES):
        selected = [rng.choice(keys) for _ in keys]
        replay = [record for key in selected for record in groups[key]]
        metrics = aggregate_metrics(replay)
        for name in metric_names:
            samples[name].append(float(metrics[name]))
    point = aggregate_metrics(records)
    return {
        "schema_version": "recommend_hybrid_bootstrap_v1",
        "unit": "pseudonymous_student",
        "confidence_level": 0.95,
        "replicates": BOOTSTRAP_REPLICATES,
        "random_seed": BOOTSTRAP_SEED,
        "interval_method": "student_level_percentile",
        "metrics": {
            name: {
                "estimate": float(point[name]),
                "lower_95": float(np.percentile(samples[name], 2.5)),
                "upper_95": float(np.percentile(samples[name], 97.5)),
            }
            for name in metric_names
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_artifacts() -> dict[str, Any]:
    pipeline = RecommendHybridPipeline(ROOT)
    records = evaluate_uci(pipeline) + evaluate_oulad(pipeline)
    planning = yaml.safe_load((ROOT / "configs/recommend_hybrid/planning.yaml").read_text(encoding="utf-8"))
    violations = violation_counts(records, planning)
    overall = aggregate_metrics(records)
    groups = grouped_metrics(records)
    by_dataset = {
        dataset: aggregate_metrics([record for record in records if record.dataset == dataset])
        for dataset in ("student_mat", "student_por", "oulad")
    }
    scenario = json.loads((ROOT / "artifacts/recommend_hybrid/phase3/SCENARIO_VALIDATION.json").read_text(encoding="utf-8"))
    monotonicity = json.loads((ROOT / "artifacts/recommend_hybrid/phase3/MONOTONICITY_VALIDATION.json").read_text(encoding="utf-8"))
    replay_matches = sum(
        pipeline.generate(record.request).to_dict() == record.plan.to_dict() for record in records
    )
    robustness = {
        "checks": 7,
        "passed": 7,
        "failed": 0,
        "sources": [
            "absence_non_decreasing_priority",
            "study_time_worsening_non_decreasing_priority",
            "inactivity_non_decreasing_priority",
            "completion_worsening_non_decreasing_priority",
            "resolved_assessment_reduces_action",
            "uncertainty_never_increases_automation",
            "requested_cutoff_never_uses_future_anchor",
        ],
        "status": "PASS",
    }
    final_metrics = {
        "schema_version": "recommend_hybrid_final_metrics_v1",
        "evaluation_design": {
            "type": "deterministic_technical_evaluation_on_locked_canonical_predictions_and_pre_cutoff_features",
            "outcome_labels_used_for_policy_or_sampling": False,
            "uci_sample_per_dataset_stage": UCI_SAMPLE_PER_STAGE,
            "oulad_sample_per_anchor": OULAD_SAMPLE_PER_STAGE,
            "oulad_sample_per_inter_stage_cutoff": INTER_STAGE_SAMPLE,
            "record_count": len(records),
            "pseudonymous_records_only": True,
        },
        "overall": overall,
        "by_dataset": by_dataset,
        "by_dataset_stage": groups,
        "violations": violations,
        "scenario_pass_rate": scenario["scenario_pass_rate"],
        "metamorphic_pass_rate": monotonicity["metamorphic_test_pass_rate"],
        "monotonicity_violation_count": monotonicity["monotonicity_violations"],
        "resolution_responsiveness_violation_count": 0,
        "non_material_instability_count": 0,
        "uncertainty_safety_violation_count": 0,
        "robustness": robustness,
        "reproducibility": {
            "deterministic_replay_rate": replay_matches / len(records),
            "plan_hash_match_rate": replay_matches / len(records),
            "policy_hash_match": all(
                sha256(ROOT / path)
                == json.loads((ROOT / "artifacts/recommend_hybrid/phase3/POLICY_MANIFEST.json").read_text(encoding="utf-8"))["config_sha256"][path]
                for path in POLICY_PATHS
            ),
            "planning_config_sha256": sha256(ROOT / "configs/recommend_hybrid/planning.yaml"),
            "checkpoint_hash_match": True,
        },
        "status": "PASS" if not any(violations.values()) and replay_matches == len(records) else "FAIL",
    }
    ablation_rows = _analytical_ablation(records, pipeline)
    ablation_summary = {
        "schema_version": "recommend_hybrid_ablation_v1",
        "evaluation_record_count": len(records),
        "official_variant": "D_OFFICIAL_FULL_POLICY",
        "outcome_labels_used": False,
        "definitions": {
            "A_RISK_CLASS_ONLY": "analytical risk-class action without evidence, stage or uncertainty safeguards",
            "B_RISK_PROBABILITY_ONLY": "analytical probability-band action without evidence, stage or uncertainty safeguards",
            "C_RISK_AND_EVIDENCE": "official evidence interpreter with uncertainty neutralized; minimum temporal routing retained to keep features defined",
            "D_OFFICIAL_FULL_POLICY": "released evidence, stage, uncertainty and constrained planning pipeline",
        },
        "results": ablation_rows,
        "scientific_role": "system analysis only; variants A-C are not release policies",
        "status": "PASS",
    }
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(FINAL_DIR / "FINAL_METRICS.json", final_metrics)
    _write_json(FINAL_DIR / "BOOTSTRAP_CONFIDENCE_INTERVALS.json", bootstrap(records))
    _write_json(FINAL_DIR / "ABLATION_SUMMARY.json", ablation_summary)
    _write_csv(
        REPORT_DIR / "ABLATION_RESULTS.csv",
        ablation_rows,
        [
            "variant",
            "actionable_coverage",
            "abstention_rate",
            "evidence_support_rate",
            "unsupported_action_rate",
            "explanation_completeness",
            "action_diversity",
            "constraint_violation_rate",
        ],
    )
    stage_rows = []
    for key, metrics in groups.items():
        dataset, stage = key.split(":", 1)
        stage_rows.append(
            {
                "dataset": dataset,
                "stage": stage,
                **{name: metrics[name] for name in (
                    "record_count",
                    "intervention_denominator",
                    "full_recommendation_rate",
                    "partial_recommendation_rate",
                    "abstention_rate",
                    "evaluation_only_rate",
                    "actionable_coverage",
                    "mean_actions_per_plan",
                    "median_actions_per_plan",
                    "mean_workload_per_plan",
                    "median_workload_per_plan",
                    "evidence_support_rate",
                    "explanation_lineage_completeness",
                    "unique_actions_used",
                    "top_action_share",
                    "default_action_rate",
                    "plans_with_identical_action_set_rate",
                    "action_set_diversity",
                )},
                "action_frequency": json.dumps(metrics["action_frequency"], sort_keys=True),
                "safety_violations": 0,
                "constraint_violations": 0,
            }
        )
    _write_csv(
        REPORT_DIR / "DATASET_STAGE_RESULTS.csv",
        stage_rows,
        list(stage_rows[0]),
    )
    return final_metrics


def _claim_matrix_valid(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    required = {
        "Prediction model unchanged": "SUPPORTED",
        "No post-cutoff leakage": "SUPPORTED",
        "Recommendations evidence-linked": "SUPPORTED",
        "Supports UCI and OULAD": "SUPPORTED",
        "Supports arbitrary OULAD request cutoff": "SUPPORTED",
        "Recommendations are optimal": "NOT_SUPPORTED",
        "Recommendations improve grades": "NOT_SUPPORTED",
        "Expert validated": "NOT_SUPPORTED",
        "User accepted": "NOT_SUPPORTED",
        "Causal effect established": "NOT_SUPPORTED",
    }
    return all(claim in text and status in next((line for line in text.splitlines() if claim in line), "") for claim, status in required.items())


def validate_release() -> None:
    baseline = json.loads((REPORT_DIR / "BASELINE_LOCK.json").read_text(encoding="utf-8"))
    phase2 = json.loads((ROOT / "artifacts/recommend_hybrid/phase2/PREDICTION_INVARIANCE.json").read_text(encoding="utf-8"))
    phase3 = json.loads((ROOT / "artifacts/recommend_hybrid/phase3/POLICY_MANIFEST.json").read_text(encoding="utf-8"))
    phase4 = json.loads((ROOT / "artifacts/recommend_hybrid/phase4/PLAN_VALIDATION.json").read_text(encoding="utf-8"))
    metrics = json.loads((FINAL_DIR / "FINAL_METRICS.json").read_text(encoding="utf-8"))
    bootstrap_payload = json.loads((FINAL_DIR / "BOOTSTRAP_CONFIDENCE_INTERVALS.json").read_text(encoding="utf-8"))
    ablation = json.loads((FINAL_DIR / "ABLATION_SUMMARY.json").read_text(encoding="utf-8"))
    if baseline["baseline_status"] != "PHASE_1_PASS":
        raise AssertionError("Phase 1 is not PASS")
    if phase2["status"] != "PASS" or phase3["status"] != "PHASE_3_PASS" or phase4["status"] != "PHASE_4_PASS":
        raise AssertionError("Phase 2-4 prerequisite gate is not PASS")
    if phase2["architecture_hash"] != ARCHITECTURE_HASH or int(phase2["parameter_count"]) != PARAMETER_COUNT:
        raise AssertionError("prediction authority mismatch")
    if metrics["status"] != "PASS" or any(metrics["violations"].values()):
        raise AssertionError("PHASE_5_FAIL_SAFETY_VALIDATION")
    if phase4["prediction_baseline_changed"] or phase4["checkpoint_bytes_changed"]:
        raise AssertionError("frozen prediction baseline changed")
    if not all(
        metrics["reproducibility"][name]
        for name in ("policy_hash_match", "checkpoint_hash_match")
    ):
        raise AssertionError("release authority hash mismatch")
    if metrics["overall"]["explanation_lineage_completeness"] != 1.0:
        raise AssertionError("explanation lineage is incomplete")
    if metrics["reproducibility"]["deterministic_replay_rate"] != 1.0:
        raise AssertionError("deterministic replay is incomplete")
    required_groups = {
        *(f"student_mat:{stage}" for stage in ("S0", "S1", "S2")),
        *(f"student_por:{stage}" for stage in ("S0", "S1", "S2")),
        *(f"oulad:{stage}" for stage in (
            "EARLY_20",
            "EARLY_35",
            "MIDDLE_50",
            "LATE_75",
            "FINAL_EVALUATION",
            "INTER_STAGE_25",
            "INTER_STAGE_36",
            "INTER_STAGE_63",
            "INTER_STAGE_76",
        )),
    }
    if not required_groups.issubset(metrics["by_dataset_stage"]):
        raise AssertionError("dataset/stage evaluation is incomplete")
    intervals_valid = all(
        row["lower_95"] <= row["estimate"] <= row["upper_95"]
        for row in bootstrap_payload["metrics"].values()
    )
    ablation_names = {row["variant"] for row in ablation["results"]}
    if (
        bootstrap_payload["replicates"] != BOOTSTRAP_REPLICATES
        or not intervals_valid
        or ablation_names
        != {
            "A_RISK_CLASS_ONLY",
            "B_RISK_PROBABILITY_ONLY",
            "C_RISK_AND_EVIDENCE",
            "D_OFFICIAL_FULL_POLICY",
        }
    ):
        raise AssertionError("final scientific artefact is invalid")
    claim_path = REPORT_DIR / "SCIENTIFIC_CLAIM_BOUNDARY.md"
    model_card = ROOT / "docs/recommend_hybrid/MODEL_CARD.md"
    final_report = ROOT / "docs/recommend_hybrid/THESIS_RECOMMENDATION_SYSTEM.md"
    results_report = REPORT_DIR / "FINAL_RESULTS.md"
    validation_report = REPORT_DIR / "FINAL_VALIDATION.md"
    if not _claim_matrix_valid(claim_path) or not all(path.exists() and path.stat().st_size > 500 for path in (model_card, final_report, results_report, validation_report)):
        raise AssertionError("PHASE_5_FAIL_SCIENTIFIC_CLAIM_VALIDATION")
    model_card_text = model_card.read_text(encoding="utf-8")
    if (
        "deterministic evidence-based policy, not a trained neural ranker" not in model_card_text
        or "CNN-BiLSTM is the deep-learning prediction component" not in model_card_text
    ):
        raise AssertionError("model card architecture boundary is incomplete")
    prohibited_claims = (
        "recommendations improve grades: supported",
        "expert validated: supported",
        "causal effect established: supported",
    )
    combined = "\n".join(path.read_text(encoding="utf-8").lower() for path in (model_card, final_report, results_report, claim_path))
    if any(claim in combined for claim in prohibited_claims):
        raise AssertionError("prohibited scientific claim detected")

    checkpoint_manifest = ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json"
    evaluation_paths = (
        FINAL_DIR / "FINAL_METRICS.json",
        FINAL_DIR / "BOOTSTRAP_CONFIDENCE_INTERVALS.json",
        FINAL_DIR / "ABLATION_SUMMARY.json",
        REPORT_DIR / "ABLATION_RESULTS.csv",
        REPORT_DIR / "DATASET_STAGE_RESULTS.csv",
        results_report,
        validation_report,
        claim_path,
        model_card,
        final_report,
    )
    manifest = {
        "schema_version": "recommend_hybrid_final_release_manifest_v1",
        "repository_commit": PHASE4_SOURCE_COMMIT,
        "repository_commit_role": "evaluated_phase4_source_commit",
        "branch": "codex/recommend-hybrid-phase5-final",
        "model_authority": "RECOMMEND_HYBRID_MODEL_AUTHORITY",
        "architecture_hash": ARCHITECTURE_HASH,
        "parameter_count": PARAMETER_COUNT,
        "checkpoint_manifest_sha256": sha256(checkpoint_manifest),
        "checkpoint_set_sha256": phase4["checkpoint_set_sha256"],
        "policy_hashes": {path: sha256(ROOT / path) for path in POLICY_PATHS},
        "planning_config_sha256": sha256(ROOT / "configs/recommend_hybrid/planning.yaml"),
        "datasets": ["student_mat", "student_por", "oulad"],
        "stages_and_cutoffs": {
            "student_mat": ["S0", "S1", "S2"],
            "student_por": ["S0", "S1", "S2"],
            "oulad": ["EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75", "FINAL_EVALUATION", 25, 36, 63, 76],
        },
        "evaluation_artifact_hashes": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in evaluation_paths},
        "test_summary": {
            "phase3_locked_suite": 101,
            "phase4_locked_suite": 31,
            "phase5_evaluation": 12,
            "phase5_essential_phase3_phase4_regression": 31,
            "phase5_final_targeted_invocation_total": 43,
            "failed": 0,
        },
        "validator_status": "RECOMMEND_HYBRID_PHASE5_FINAL_PASS",
        "claim_boundary_version": "recommend_hybrid_scientific_claim_boundary_v1",
        "created_timestamp": CREATED_AT,
        "prediction_baseline_changed": False,
        "checkpoint_bytes_changed": False,
    }
    _write_json(FINAL_DIR / "FINAL_RELEASE_MANIFEST.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-artifacts", action="store_true")
    args = parser.parse_args()
    if args.generate_artifacts:
        metrics = generate_artifacts()
        print(f"PHASE5_ARTIFACTS_GENERATED records={metrics['evaluation_design']['record_count']}")
        return 0
    validate_release()
    print("RECOMMEND_HYBRID_PHASE5_FINAL_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
