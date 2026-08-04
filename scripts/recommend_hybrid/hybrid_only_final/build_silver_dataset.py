"""Build cutoff-safe hybrid-only candidates and direct future silver labels.

No model is trained here.  The only predictive quantities are frozen hybrid
risk and uncertainty plus frozen hybrid counterfactual risk reduction.  Future
behavior is used solely to evaluate whether an issued action aligns with the
next observed OULAD trajectory.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/recommend_hybrid/hybrid_only_final"
sys.path.insert(0, str(ROOT))

from src.pipelines.oulad import BASE_CHANNELS, _build_bundle  # noqa: E402

TRANSITIONS = [
    ("E1_EARLY_20PCT", "E2_EARLY_35PCT", "EARLY_20"),
    ("E2_EARLY_35PCT", "M1_MIDDLE_FROZEN", "EARLY_35"),
    ("M1_MIDDLE_FROZEN", "L1_LATE_75PCT", "MIDDLE_50"),
]
FAMILIES = [
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
]
FAMILY_TO_RUNTIME_ACTION = {
    "ASSESSMENT_COMPLETION": "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY": "STUDY_SCHEDULE",
    "VLE_ENGAGEMENT": "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE": "RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW": "LEARNING_CONSOLIDATION",
}
WORKLOAD = {
    "ASSESSMENT_COMPLETION": 150,
    "STUDY_REGULARITY": 30,
    "VLE_ENGAGEMENT": 90,
    "QUIZ_OR_RETRIEVAL_PRACTICE": 75,
    "CONTENT_REVIEW": 90,
}
VLE_TYPES = {
    "QUIZ_OR_RETRIEVAL_PRACTICE": {"quiz", "externalquiz", "questionnaire"},
    "CONTENT_REVIEW": {"resource", "oucontent", "page", "subpage", "url", "folder", "glossary"},
}
INDEX = {name: idx for idx, name in enumerate(BASE_CHANNELS)}
MIN_RELATIVE_IMPROVEMENT = 0.05
MIN_ABSOLUTE_IMPROVEMENT = 1e-6


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sum(sequence: np.ndarray, start: int, stop: int, channel: str) -> float:
    return float(sequence[start:stop, INDEX[channel]].sum())


def _improved(current: float, future: float) -> bool:
    required = max(MIN_ABSOLUTE_IMPROVEMENT, abs(current) * MIN_RELATIVE_IMPROVEMENT)
    return bool(future - current >= required)


def main() -> None:
    dataset_out = OUT / "dataset"
    dataset_out.mkdir(parents=True, exist_ok=True)
    bundle = _build_bundle()

    assessments = pd.read_csv(
        ROOT / "data/raw/assessments.csv",
        usecols=["code_module", "code_presentation", "date", "weight"],
    )
    assessments = assessments[assessments["weight"] > 0]
    assessment_schedule = {
        (str(module), str(presentation)): group[["date", "weight"]].to_numpy()
        for (module, presentation), group in assessments.groupby(
            ["code_module", "code_presentation"], sort=False
        )
    }
    vle = pd.read_csv(
        ROOT / "data/raw/vle.csv",
        usecols=["code_module", "code_presentation", "activity_type", "week_from", "week_to"],
    )
    vle_schedule = {
        (str(module), str(presentation)): group
        for (module, presentation), group in vle.groupby(
            ["code_module", "code_presentation"], sort=False
        )
    }

    predictions = pd.read_parquet(
        ROOT / "artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet"
    )
    predictions = predictions[
        (predictions["model"] == "hybrid")
        & predictions["stage"].isin(
            ["E1_EARLY_20PCT", "E2_EARLY_35PCT", "M1_MIDDLE_50PCT"]
        )
    ].copy()
    predictions["stage_key"] = predictions["stage"].replace(
        {"M1_MIDDLE_50PCT": "M1_MIDDLE_FROZEN"}
    )
    risk_map = predictions.groupby(["base_record_id", "stage_key"])["probability"].mean().to_dict()

    seed_predictions = pd.read_parquet(
        ROOT / "artifacts/canonical_v3/predictions/oulad_seed_predictions.parquet"
    )
    seed_predictions = seed_predictions[
        (seed_predictions["model"] == "hybrid")
        & seed_predictions["stage"].isin(
            ["E1_EARLY_20PCT", "E2_EARLY_35PCT", "M1_MIDDLE_50PCT"]
        )
    ]
    uncertainty_map = (
        seed_predictions.groupby(["base_record_id", "stage"])["probability"]
        .std()
        .fillna(0.0)
        .to_dict()
    )

    counterfactual = pd.read_parquet(
        ROOT / "artifacts/recommend_hybrid/counterfactual/full_cohort/action_scores.parquet",
        columns=["student_key", "stage", "action_id", "risk_reduction"],
    )
    counterfactual_map = {
        (str(row.student_key), str(row.stage), str(row.action_id)): float(row.risk_reduction)
        for row in counterfactual.itertuples()
    }

    stage_indexes = {
        stage: {str(value): idx for idx, value in enumerate(data.frame["base_record_id"])}
        for stage, data in bundle.stages.items()
    }
    groups: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    for source_stage, target_stage, stage in TRANSITIONS:
        source = bundle.stages[source_stage]
        target = bundle.stages[target_stage]
        for source_index, record in source.frame.iterrows():
            learner = str(record.base_record_id)
            target_index = stage_indexes[target_stage].get(learner)
            if target_index is None:
                continue
            current = source.sequence[source_index]
            future = target.sequence[target_index]
            current_length = int(source.lengths[source_index])
            future_length = int(target.lengths[target_index])
            if future_length <= current_length:
                continue
            future_steps = future_length - current_length
            history_start = max(0, current_length - future_steps)
            cutoff_day = float(record.cutoff_day)
            target_day = float(target.frame.iloc[target_index].cutoff_day)
            history_start_day = max(0.0, cutoff_day - (target_day - cutoff_day))
            module = str(record.code_module)
            presentation = str(record.code_presentation)
            group_id = f"{learner}|{stage}"

            schedule = vle_schedule.get((module, presentation), pd.DataFrame())

            def vle_opportunity(start_day: float, stop_day: float, activity_types=None) -> int:
                if schedule.empty:
                    return 0
                start_week = start_day / 7.0
                stop_week = stop_day / 7.0
                query = schedule[
                    (schedule["week_to"].fillna(schedule["week_from"]) >= start_week)
                    & (schedule["week_from"].fillna(schedule["week_to"]) <= stop_week)
                ]
                if activity_types is not None:
                    query = query[query["activity_type"].isin(activity_types)]
                return int(len(query))

            scheduled_assessments = assessment_schedule.get(
                (module, presentation), np.empty((0, 2))
            )
            if len(scheduled_assessments):
                assessment_future_opp = int(
                    (
                        (scheduled_assessments[:, 0] > cutoff_day)
                        & (scheduled_assessments[:, 0] <= target_day)
                    ).sum()
                )
                assessment_current_opp = int(
                    (
                        (scheduled_assessments[:, 0] > history_start_day)
                        & (scheduled_assessments[:, 0] <= cutoff_day)
                    ).sum()
                )
            else:
                assessment_future_opp = assessment_current_opp = 0

            future_opportunities = {
                "ASSESSMENT_COMPLETION": assessment_future_opp,
                "STUDY_REGULARITY": max(1, int(target_day - cutoff_day)),
                "VLE_ENGAGEMENT": vle_opportunity(cutoff_day, target_day),
                "QUIZ_OR_RETRIEVAL_PRACTICE": vle_opportunity(
                    cutoff_day, target_day, VLE_TYPES["QUIZ_OR_RETRIEVAL_PRACTICE"]
                ),
                "CONTENT_REVIEW": vle_opportunity(
                    cutoff_day, target_day, VLE_TYPES["CONTENT_REVIEW"]
                ),
            }
            current_opportunities = {
                "ASSESSMENT_COMPLETION": assessment_current_opp,
                "STUDY_REGULARITY": max(1, int(cutoff_day - history_start_day)),
                "VLE_ENGAGEMENT": vle_opportunity(history_start_day, cutoff_day),
                "QUIZ_OR_RETRIEVAL_PRACTICE": vle_opportunity(
                    history_start_day,
                    cutoff_day,
                    VLE_TYPES["QUIZ_OR_RETRIEVAL_PRACTICE"],
                ),
                "CONTENT_REVIEW": vle_opportunity(
                    history_start_day, cutoff_day, VLE_TYPES["CONTENT_REVIEW"]
                ),
            }

            active_ratio = _sum(current, history_start, current_length, "active_days") / max(
                1, current_length - history_start
            )
            current_click_rate = _sum(current, history_start, current_length, "total_clicks") / max(
                1, current_opportunities["VLE_ENGAGEMENT"]
            )
            current_quiz_rate = _sum(current, history_start, current_length, "quiz_clicks") / max(
                1, current_opportunities["QUIZ_OR_RETRIEVAL_PRACTICE"]
            )
            current_content_rate = _sum(current, history_start, current_length, "content_clicks") / max(
                1, current_opportunities["CONTENT_REVIEW"]
            )
            inactivity = float(current[current_length - 1, INDEX["days_since_last_vle_activity"]])
            recent_delta = float(
                current[current_length - 1, INDEX["total_clicks"]]
                - current[max(0, current_length - 2), INDEX["total_clicks"]]
            )
            assessment_progress = float(
                current[current_length - 1, INDEX["cumulative_weighted_score"]]
            )

            needed = {
                "ASSESSMENT_COMPLETION": int(
                    assessment_future_opp > 0 and assessment_progress < 0.80
                ),
                "STUDY_REGULARITY": int(active_ratio < 0.50 or inactivity >= 2),
                "VLE_ENGAGEMENT": int(current_click_rate < 10 or recent_delta < 0),
                "QUIZ_OR_RETRIEVAL_PRACTICE": int(current_quiz_rate < 1),
                "CONTENT_REVIEW": int(current_content_rate < 5),
            }
            eligible = [
                family
                for family in FAMILIES
                if future_opportunities[family] > 0 and needed[family] == 1
            ]
            rankable = len(eligible) >= 2
            groups.append(
                {
                    "group_id": group_id,
                    "base_record_id": learner,
                    "stage": stage,
                    "outer_fold": int(record.outer_fold),
                    "course": module,
                    "presentation": presentation,
                    "rankable": rankable,
                    "eligible_action_count": len(eligible),
                }
            )
            if not rankable:
                continue

            stage_key = {
                "EARLY_20": "EARLY_20",
                "EARLY_35": "EARLY_35",
                "MIDDLE_50": "MIDDLE_50",
            }[stage]
            risk = float(risk_map.get((learner, source_stage), np.nan))
            seed_stage = source_stage.replace("M1_MIDDLE_FROZEN", "M1_MIDDLE_50PCT")
            uncertainty = float(uncertainty_map.get((learner, seed_stage), 0.0))

            for family in eligible:
                if family == "ASSESSMENT_COMPLETION":
                    current_signal = _sum(
                        current, history_start, current_length, "submitted_assessment_count"
                    ) / max(1, assessment_current_opp)
                    future_signal = _sum(
                        future, current_length, future_length, "submitted_assessment_count"
                    ) / max(1, assessment_future_opp)
                    proximal_available = int(
                        _sum(future, current_length, future_length, "available_score_count") > 0
                    )
                    silver_positive = _improved(current_signal, future_signal) and bool(
                        proximal_available
                    )
                    deficit = max(0.0, 1.0 - assessment_progress)
                elif family == "STUDY_REGULARITY":
                    current_signal = active_ratio
                    future_signal = _sum(
                        future, current_length, future_length, "active_days"
                    ) / max(1, future_length - current_length)
                    future_inactivity = float(
                        future[future_length - 1, INDEX["days_since_last_vle_activity"]]
                    )
                    silver_positive = _improved(current_signal, future_signal) and (
                        future_inactivity < inactivity
                    )
                    deficit = max(0.0, 0.50 - active_ratio) + max(0.0, inactivity - 1.0) / 10.0
                elif family == "VLE_ENGAGEMENT":
                    current_signal = current_click_rate
                    future_signal = _sum(
                        future, current_length, future_length, "total_clicks"
                    ) / max(1, future_opportunities[family])
                    silver_positive = _improved(current_signal, future_signal)
                    deficit = max(0.0, 10.0 - current_click_rate) / 10.0
                elif family == "QUIZ_OR_RETRIEVAL_PRACTICE":
                    current_signal = current_quiz_rate
                    future_signal = _sum(
                        future, current_length, future_length, "quiz_clicks"
                    ) / max(1, future_opportunities[family])
                    silver_positive = _improved(current_signal, future_signal)
                    deficit = max(0.0, 1.0 - current_quiz_rate)
                else:
                    current_signal = current_content_rate
                    future_signal = _sum(
                        future, current_length, future_length, "content_clicks"
                    ) / max(1, future_opportunities[family])
                    silver_positive = _improved(current_signal, future_signal)
                    deficit = max(0.0, 5.0 - current_content_rate) / 5.0

                opportunity = int(future_opportunities[family])
                evidence = float(min(1.0, opportunity / 10.0))
                runtime_action = FAMILY_TO_RUNTIME_ACTION[family]
                rows.append(
                    {
                        "group_id": group_id,
                        "base_record_id": learner,
                        "stage": stage,
                        "outer_fold": int(record.outer_fold),
                        "course": module,
                        "presentation": presentation,
                        "action_family": family,
                        "runtime_action_id": runtime_action,
                        "risk_probability": risk,
                        "risk_uncertainty": uncertainty,
                        "risk_reduction": float(
                            counterfactual_map.get(
                                (learner, stage_key, runtime_action), np.nan
                            )
                        ),
                        "evidence_strength": evidence,
                        "deficit_score": float(deficit),
                        "workload_minutes": int(WORKLOAD[family]),
                        "action_available": 1,
                        "prerequisite_status": 1,
                        "current_behavior_signal": float(current_signal),
                        "future_behavior_signal": float(future_signal),
                        "silver_positive": int(silver_positive),
                    }
                )

    group_frame = pd.DataFrame(groups)
    candidate_frame = pd.DataFrame(rows)
    group_frame.to_parquet(dataset_out / "learner_stage_groups.parquet", index=False)
    candidate_frame.to_parquet(dataset_out / "candidate_rows.parquet", index=False)

    positive_by_group = candidate_frame.groupby("group_id")["silver_positive"].max()
    flow = {
        "transition_groups": int(len(group_frame)),
        "rankable_groups": int(group_frame["rankable"].sum()),
        "candidate_rows": int(len(candidate_frame)),
        "groups_with_positive_action": int((positive_by_group > 0).sum()),
        "groups_without_positive_action": int((positive_by_group == 0).sum()),
        "action_distribution": candidate_frame["action_family"].value_counts().to_dict(),
        "labels_used_for_scoring": False,
        "additional_learned_model_used": False,
    }
    _write_json(dataset_out / "cohort_flow.json", flow)
    _write_json(
        dataset_out / "schema.json",
        {
            "runtime_features": [
                "risk_probability",
                "risk_uncertainty",
                "risk_reduction",
                "evidence_strength",
                "deficit_score",
                "workload_minutes",
                "action_available",
                "prerequisite_status",
            ],
            "evaluation_only": [
                "current_behavior_signal",
                "future_behavior_signal",
                "silver_positive",
            ],
            "protected_features": [],
            "future_features_in_scoring": False,
        },
    )
    checksum_paths = [path for path in dataset_out.iterdir() if path.is_file()]
    _write_json(
        dataset_out / "CHECKSUMS.json",
        {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in checksum_paths
        },
    )
    print(json.dumps(flow, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
