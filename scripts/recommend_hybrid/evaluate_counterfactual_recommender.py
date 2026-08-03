"""Evaluate the OULAD counterfactual recommender on outer-fold validation rows.

The evaluator never uses target, final_result, withdrawal outcome, or any other
post-cutoff label to generate, score, rank, or select an action. Outcome-based
trajectory analysis is intentionally a separate script and a separate claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipelines.oulad import BASE_CHANNELS, STATIC_COLUMNS, _build_bundle
from src.recommend_hybrid.common.policy_contracts import (
    DatasetId,
    PolicyPredictionContext,
)
from src.recommend_hybrid.contracts import Stage
from src.recommend_hybrid.counterfactual.evaluation import (
    CounterfactualEvaluationRow,
    aggregate_counterfactual_metrics,
    grouped_counterfactual_metrics,
)
from src.recommend_hybrid.counterfactual.plan_builder import (
    OULADCounterfactualPlanBuilder,
)
from src.recommend_hybrid.counterfactual.reference_profile import (
    OULADReferenceProfile,
    OULADReferenceProfileBuilder,
)
from src.recommend_hybrid.oulad.policy import RecommendHybridOULAD
from src.recommend_hybrid.prediction_adapter import HybridPredictionAdapter

OUT = ROOT / "artifacts/recommend_hybrid/counterfactual"
REPORT = ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_EVALUATION.md"
CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"

# Bundle keys follow the frozen training pipeline. The middle-stage reporting
# alias remains M1_MIDDLE_50PCT because that is the name used by OOF artifacts.
STAGES: dict[str, tuple[Stage, float, str]] = {
    "E1_EARLY_20PCT": (Stage.EARLY_20, 20.0, "E1_EARLY_20PCT"),
    "E2_EARLY_35PCT": (Stage.EARLY_35, 35.0, "E2_EARLY_35PCT"),
    "M1_MIDDLE_FROZEN": (Stage.MIDDLE_50, 50.0, "M1_MIDDLE_50PCT"),
    "L1_LATE_75PCT": (Stage.LATE_75, 75.0, "L1_LATE_75PCT"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _course_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["code_module"].astype(str)
        + "-"
        + frame["code_presentation"].astype(str)
    )


def _round_robin_indices(
    frame: pd.DataFrame,
    *,
    limit: int | None,
) -> list[int]:
    ordered = frame.sort_values(
        ["code_module", "code_presentation", "base_record_id"]
    )
    queues = [
        list(group.index)
        for _, group in ordered.groupby(
            ["code_module", "code_presentation"],
            sort=True,
        )
    ]
    selected: list[int] = []
    position = 0
    while queues and (limit is None or len(selected) < limit):
        retained: list[list[int]] = []
        for queue in queues:
            if position < len(queue):
                selected.append(int(queue[position]))
                if limit is not None and len(selected) >= limit:
                    break
            if position + 1 < len(queue):
                retained.append(queue)
        queues = retained
        position += 1
    return selected


def _load_assessment_dates() -> dict[tuple[str, str], np.ndarray]:
    frame = pd.read_csv(
        ROOT / "data/raw/assessments.csv",
        usecols=["code_module", "code_presentation", "date"],
    )
    frame = frame.loc[frame["date"].notna()].copy()
    return {
        (str(module), str(presentation)): np.sort(
            group["date"].to_numpy(dtype=float)
        )
        for (module, presentation), group in frame.groupby(
            ["code_module", "code_presentation"],
            sort=True,
        )
    }


def _reference_profiles(
    data: Any,
    *,
    fold: int,
    profile_stage: str,
    builder: OULADReferenceProfileBuilder,
) -> dict[str, OULADReferenceProfile]:
    frame = data.frame.copy()
    frame["course_key"] = _course_key(frame)
    profiles: dict[str, OULADReferenceProfile] = {}
    training = frame.loc[frame["outer_fold"].ne(fold)]
    for course_key, group in training.groupby("course_key", sort=True):
        indices = group.index.to_numpy(dtype=int)
        profiles[str(course_key)] = builder.build(
            sequence=data.sequence[indices],
            lengths=data.lengths[indices],
            fold=fold,
            stage=profile_stage,
            course_key=str(course_key),
            sample_role="TRAIN",
        )
    return profiles


def _policy_context(output: Any) -> PolicyPredictionContext:
    probabilities = tuple(
        float(value)
        for value in output.probabilities[0].detach().cpu().tolist()
    )
    return PolicyPredictionContext(
        dataset_id=DatasetId.OULAD,
        predicted_class=int(output.predicted_class[0].detach().cpu().item()),
        class_probabilities=probabilities,
        confidence=float(output.confidence[0].detach().cpu().item()),
        uncertainty=float(output.uncertainty[0].detach().cpu().item()),
        seed_disagreement=float(
            output.seed_disagreement[0].detach().cpu().item()
        ),
        checkpoint_lineage=tuple(
            reference.checkpoint_id
            for reference in output.checkpoint_references
        ),
        architecture_authority="RECOMMEND_HYBRID_MODEL_AUTHORITY",
        representation_lineage=(
            "student_state_embedding:64",
            "tabular_expert_embedding:32",
        ),
    )


def _observed_features(
    sequence: np.ndarray,
    length: int,
    *,
    cutoff_day: int,
    assessment_dates: np.ndarray,
) -> dict[str, float | int | None]:
    index = {name: position for position, name in enumerate(BASE_CHANNELS)}
    observed = sequence[:length]
    clicks = observed[:, index["total_clicks"]].astype(float)
    activity_level = float(clicks.sum() / max(1, length * 7))
    recent_weeks = min(2, length)
    recent_rate = float(
        clicks[-recent_weeks:].sum() / max(1, recent_weeks * 7)
    )
    previous = clicks[max(0, length - 2 * recent_weeks) : length - recent_weeks]
    recent_activity_trend = (
        float(recent_rate - previous.sum() / (len(previous) * 7))
        if len(previous)
        else None
    )
    inactivity_streak = int(
        round(
            float(
                observed[-1, index["days_since_last_vle_activity"]]
            )
        )
    )
    due_count = int(np.sum(assessment_dates < cutoff_day))
    completed = int(
        round(
            float(
                observed[:, index["submitted_assessment_count"]].sum()
            )
        )
    )
    assessment_progress = (
        min(1.0, completed / due_count) if due_count > 0 else None
    )
    return {
        "activity_level": activity_level,
        "recent_activity_trend": recent_activity_trend,
        "inactivity_streak": inactivity_streak,
        "assessment_progress": assessment_progress,
        "assessments_due": due_count,
    }


def _model_inputs(
    data: Any,
    frame: pd.DataFrame,
    index: int,
    adapter: HybridPredictionAdapter,
) -> dict[str, torch.Tensor]:
    aggregate = adapter.transform_aggregate(
        data.aggregate[index : index + 1]
    )
    row = frame.loc[[index]]
    static = adapter.transform_static(
        {column: row[column].tolist() for column in STATIC_COLUMNS}
    )
    return {
        "sequence": torch.from_numpy(
            data.sequence[index : index + 1].astype(np.float32)
        ),
        "lengths": torch.from_numpy(
            data.lengths[index : index + 1].astype(np.int64)
        ),
        "mask": torch.from_numpy(
            data.mask[index : index + 1].astype(np.float32)
        ),
        "aggregate": torch.from_numpy(aggregate),
        "static": torch.from_numpy(static),
    }


def _bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | int | None]:
    if not values:
        return {
            "mean": None,
            "lower_95": None,
            "upper_95": None,
            "replicates": replicates,
        }
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = np.empty(replicates, dtype=float)
    for index in range(replicates):
        means[index] = generator.choice(
            array,
            size=len(array),
            replace=True,
        ).mean()
    return {
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "replicates": replicates,
    }


def evaluate(
    *,
    max_records_per_fold_stage: int | None,
    folds: tuple[int, ...],
    stages: tuple[str, ...],
    seeds: tuple[int, ...],
    bootstrap_replicates: int,
    output_dir: Path = OUT,
    report_path: Path = REPORT,
) -> dict[str, Any]:
    bundle = _build_bundle()
    assessment_dates = _load_assessment_dates()
    policy = RecommendHybridOULAD(ROOT)
    plan_builder = OULADCounterfactualPlanBuilder(ROOT)
    reference_builder = OULADReferenceProfileBuilder()
    evaluation_rows: list[CounterfactualEvaluationRow] = []
    action_rows: list[dict[str, Any]] = []

    for bundle_stage in stages:
        canonical_stage, requested_cutoff, profile_stage = STAGES[bundle_stage]
        data = bundle.stages[bundle_stage]
        frame = data.frame.copy()
        frame["course_key"] = _course_key(frame)
        for fold in folds:
            adapter = HybridPredictionAdapter.from_manifest(
                ROOT,
                stage=canonical_stage,
                fold=fold,
                seeds=seeds,
            )
            profiles = _reference_profiles(
                data,
                fold=fold,
                profile_stage=profile_stage,
                builder=reference_builder,
            )
            validation = frame.loc[frame["outer_fold"].eq(fold)]
            indices = _round_robin_indices(
                validation,
                limit=max_records_per_fold_stage,
            )
            for index in indices:
                row = frame.loc[index]
                course_key = str(row["course_key"])
                profile = profiles.get(course_key)
                if profile is None:
                    raise RuntimeError(
                        f"missing training reference profile: fold={fold} "
                        f"stage={bundle_stage} course={course_key}"
                    )
                inputs = _model_inputs(data, frame, index, adapter)
                baseline_output = adapter.predict(inputs)
                context = _policy_context(baseline_output)
                course_tuple = (
                    str(row["code_module"]),
                    str(row["code_presentation"]),
                )
                observed = _observed_features(
                    data.sequence[index],
                    int(data.lengths[index]),
                    cutoff_day=int(row["cutoff_day"]),
                    assessment_dates=assessment_dates.get(
                        course_tuple,
                        np.array([], dtype=float),
                    ),
                )
                policy_result = policy.recommend(
                    student_key=str(row["base_record_id"]),
                    course_key=course_key,
                    requested_cutoff=requested_cutoff,
                    prediction=context,
                    max_observation_cutoff=requested_cutoff - 1e-6,
                    activity_level=observed["activity_level"],
                    recent_activity_trend=observed[
                        "recent_activity_trend"
                    ],
                    inactivity_streak=observed["inactivity_streak"],
                    assessment_progress=observed["assessment_progress"],
                    assessments_due=observed["assessments_due"],
                    grade_trend=None,
                    grade_release_verified=False,
                    knowledge_gap=None,
                )
                result = plan_builder.build(
                    policy_result,
                    course_key=course_key,
                    created_at="2026-08-03T20:30:00Z",
                    model_inputs=inputs,
                    reference_profile=profile,
                    prediction_authority=adapter,
                )
                ranking = result.ranking
                top = (
                    ranking.ranked_actions[0]
                    if ranking is not None and ranking.ranked_actions
                    else None
                )
                baseline_risk = float(
                    baseline_output.probabilities[0, 1].detach().cpu().item()
                )
                evaluation_rows.append(
                    CounterfactualEvaluationRow(
                        student_key=str(row["base_record_id"]),
                        course_key=course_key,
                        stage=canonical_stage.value,
                        fold=fold,
                        baseline_risk=baseline_risk,
                        decision_threshold=float(adapter.decision_threshold),
                        status=result.status.value,
                        top_action_id=top.action_id if top else None,
                        top_counterfactual_risk=(
                            top.counterfactual_risk if top else None
                        ),
                        top_risk_reduction=(
                            top.risk_reduction if top else None
                        ),
                        top_utility_score=(
                            top.utility_score if top else None
                        ),
                        selected_action_count=len(
                            result.plan.selected_actions
                        ),
                        selected_workload_minutes=result.plan.total_minutes,
                        reference_profile_id=result.reference_profile_id,
                        fallback_reasons=result.fallback_reasons,
                    )
                )
                if ranking is None:
                    continue
                selected_ids = {
                    item.action_id for item in result.plan.selected_actions
                }
                for action in (
                    *ranking.ranked_actions,
                    *ranking.rejected_actions,
                ):
                    action_rows.append(
                        {
                            "student_key": str(row["base_record_id"]),
                            "course_key": course_key,
                            "stage": canonical_stage.value,
                            "fold": fold,
                            "action_id": action.action_id,
                            "utility_status": action.status.value,
                            "baseline_risk": action.baseline_risk,
                            "counterfactual_risk": action.counterfactual_risk,
                            "risk_reduction": action.risk_reduction,
                            "utility_score": action.utility_score,
                            "evidence_strength": action.evidence_strength,
                            "uncertainty_penalty": action.uncertainty_penalty,
                            "workload_minutes": action.workload_minutes,
                            "selected_in_plan": action.action_id in selected_ids,
                            "reason_codes": "|".join(action.reason_codes),
                            "reference_profile_id": profile.profile_id,
                        }
                    )

    metrics = aggregate_counterfactual_metrics(evaluation_rows)
    reductions = [
        float(row.top_risk_reduction)
        for row in evaluation_rows
        if row.top_risk_reduction is not None
    ]
    metrics["mean_risk_reduction_bootstrap_95"] = _bootstrap_mean_ci(
        reductions,
        seed=20260803,
        replicates=bootstrap_replicates,
    )
    payload = {
        "schema_version": "counterfactual_oulad_evaluation_v2",
        "generated_at": _utc_now(),
        "claim_boundary": CLAIM_BOUNDARY,
        "configuration": {
            "folds": list(folds),
            "bundle_stages": list(stages),
            "reporting_stage_aliases": {
                key: STAGES[key][2] for key in stages
            },
            "seeds": list(seeds),
            "max_records_per_fold_stage": max_records_per_fold_stage,
            "bootstrap_replicates": bootstrap_replicates,
            "sampling": "deterministic_course_round_robin",
            "reference_scope": "TRAINING_FOLD_COURSE_STAGE_ONLY",
        },
        "overall": metrics,
        "grouped": grouped_counterfactual_metrics(evaluation_rows),
        "scientific_guards": {
            "target_used_for_ranking": False,
            "final_result_used_for_ranking": False,
            "date_unregistration_used_for_ranking": False,
            "outer_validation_labels_used_for_tuning": False,
            "expert_labels_required": False,
            "silver_labels_used": False,
            "causal_effect_claimed": False,
        },
        "status": "PASS" if evaluation_rows else "FAIL",
    }
    _write_json(output_dir / "evaluation.json", payload)
    _write_csv(
        output_dir / "evaluation_rows.csv",
        [row.to_dict() for row in evaluation_rows],
    )
    _write_csv(output_dir / "action_scores.csv", action_rows)
    _write_report(payload, report_path=report_path)
    return payload


def _write_report(
    payload: dict[str, Any],
    *,
    report_path: Path = REPORT,
) -> None:
    overall = payload["overall"]
    bootstrap = overall["mean_risk_reduction_bootstrap_95"]
    lines = [
        "# Counterfactual recommender evaluation",
        "",
        f"- Status: `{payload['status']}`",
        f"- Claim boundary: `{payload['claim_boundary']}`",
        f"- Records: `{overall['record_count']}`",
        f"- Scored coverage: `{overall['scored_coverage']:.4f}`",
        f"- Mean top-action risk reduction: "
        f"`{overall['mean_top_risk_reduction']:.6f}`",
        f"- Median top-action risk reduction: "
        f"`{overall['median_top_risk_reduction']:.6f}`",
        f"- Success@0.01: `{overall['success_at_0_01']:.4f}`",
        f"- Success@0.05: `{overall['success_at_0_05']:.4f}`",
        f"- Threshold crossing rate: "
        f"`{overall['threshold_crossing_rate']:.4f}`",
        f"- Bootstrap mean 95% CI: "
        f"`[{bootstrap['lower_95']}, {bootstrap['upper_95']}]`",
        "",
        "The evaluator measures changes in risk predicted by the frozen "
        "Hybrid CNN–BiLSTM under feasible input counterfactuals. Targets and "
        "post-cutoff outcomes are not used to rank actions, and the result is "
        "not a causal treatment-effect estimate.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_stage_tuple(value: str) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown bundle stages: {unknown}")
    return stages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-records-per-fold-stage",
        type=int,
        default=100,
        help="Use 0 for all outer-validation rows.",
    )
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument(
        "--seeds",
        default="42,1201,2026,3407,7319",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Write evaluation artifacts to this directory.",
    )
    args = parser.parse_args()
    limit = (
        None
        if args.max_records_per_fold_stage == 0
        else args.max_records_per_fold_stage
    )
    if limit is not None and limit <= 0:
        raise ValueError("max records must be positive or zero for all")
    payload = evaluate(
        max_records_per_fold_stage=limit,
        folds=_parse_int_tuple(args.folds),
        stages=_parse_stage_tuple(args.stages),
        seeds=_parse_int_tuple(args.seeds),
        bootstrap_replicates=args.bootstrap_replicates,
        output_dir=args.output_dir,
        report_path=(
            REPORT if args.output_dir == OUT else args.output_dir / "evaluation.md"
        ),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
