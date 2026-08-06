"""Evaluate Recommendation V2 on the full OULAD learner-stage population."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.final.actions import canonical_action_id  # noqa: E402
from src.recommend_hybrid.v2.eligibility import (  # noqa: E402
    EligibilityDecision,
    apply_eligibility_policy,
    normalized_binary_entropy,
    select_eligibility_policy,
)
from src.recommend_hybrid.v2.evaluation import eligibility_metrics, ranking_metrics  # noqa: E402
from src.recommend_hybrid.v2.ranking import (  # noqa: E402
    MinMaxNormalizer,
    ranking_baselines,
    select_ranking_weights,
    utility_scores,
)
from src.recommend_hybrid.v2.taxonomy import LEARNED_ACTIONS  # noqa: E402

DEFAULT_LANDMARK = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
DEFAULT_OOF = ROOT / "artifacts/recommend_hybrid/final_stage_aware_v2/oof_predictions.parquet"
DEFAULT_LABELS = ROOT / "artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"
DEFAULT_SIMULATION = ROOT / "artifacts/recommend_hybrid/v2/simulation_rows.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/v2/FULL_POPULATION_EVIDENCE.json"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/v2/FULL_POPULATION_RECOMMENDATION_RESULTS.md"
STAGES = ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")
WORKLOAD = {
    "ASSESSMENT_COMPLETION": 150.0 / 180.0,
    "STUDY_REGULARITY": 30.0 / 180.0,
    "VLE_ENGAGEMENT": 90.0 / 180.0,
    "QUIZ_OR_RETRIEVAL_PRACTICE": 75.0 / 180.0,
    "CONTENT_REVIEW": 90.0 / 180.0,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    result[~positive] = exp / (1.0 + exp)
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_confidence(labels_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(labels_path)
    required = {
        "dataset",
        "student_key",
        "stage",
        "action_id",
        "silver_status",
        "silver_confidence",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"silver labels are missing columns: {missing}")
    frame = frame.loc[
        frame["dataset"].eq("oulad")
        & frame["silver_status"].eq("RETAINED")
        & frame["stage"].isin(STAGES)
    ].copy()
    canonical: list[str | None] = []
    for value in frame["action_id"]:
        try:
            canonical.append(canonical_action_id(value))
        except ValueError:
            canonical.append(None)
    frame["action_id"] = canonical
    frame = frame.loc[frame["action_id"].isin(LEARNED_ACTIONS)].copy()
    frame["record_id"] = frame["student_key"].astype(str)
    frame["silver_confidence"] = pd.to_numeric(
        frame["silver_confidence"],
        errors="coerce",
    ).fillna(0.0)
    return frame.sort_values(
        ["record_id", "stage", "action_id", "silver_confidence"],
        ascending=[True, True, True, False],
        kind="stable",
    ).drop_duplicates(["record_id", "stage", "action_id"], keep="first")


def _prepare_arrays(
    landmark_path: Path,
    oof_path: Path,
    labels_path: Path,
    simulation_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    landmark = pd.read_parquet(landmark_path)
    required_landmark = {
        "record_id",
        "student_id",
        "stage",
        "action_id",
        "protocol_split",
        "prediction_target",
        "baseline_measure",
        "feature__risk_probability",
    }
    missing = sorted(required_landmark.difference(landmark.columns))
    if missing:
        raise KeyError(f"landmark is missing columns: {missing}")
    landmark = landmark.loc[
        landmark["stage"].isin(STAGES)
        & landmark["action_id"].isin(LEARNED_ACTIONS)
    ].copy()
    landmark["record_id"] = landmark["record_id"].astype(str)
    landmark["student_id"] = landmark["student_id"].astype(str)
    if landmark.duplicated(["record_id", "stage", "action_id"]).any():
        raise ValueError("landmark record-stage-action identity is not unique")

    group = landmark.drop_duplicates(["record_id", "stage"]).loc[
        :,
        [
            "record_id",
            "student_id",
            "stage",
            "protocol_split",
            "prediction_target",
            "feature__risk_probability",
        ],
    ]
    baseline = landmark.pivot(
        index=["record_id", "stage"],
        columns="action_id",
        values="baseline_measure",
    ).reindex(columns=LEARNED_ACTIONS)
    availability = baseline.notna()
    baseline = baseline.fillna(1.0)
    confidence = _canonical_confidence(labels_path).pivot(
        index=["record_id", "stage"],
        columns="action_id",
        values="silver_confidence",
    ).reindex(columns=LEARNED_ACTIONS).fillna(0.0)

    oof = pd.read_parquet(oof_path)
    required_oof = {"record_id", "stage"}
    for action in LEARNED_ACTIONS:
        required_oof.update(
            {
                f"action_logit__{action}",
                f"action_target__{action}",
                f"action_mask__{action}",
            }
        )
    missing_oof = sorted(required_oof.difference(oof.columns))
    if missing_oof:
        raise KeyError(f"OOF action evidence is missing columns: {missing_oof}")
    oof["record_id"] = oof["record_id"].astype(str)
    oof_columns = ["record_id", "stage"] + sorted(required_oof - {"record_id", "stage"})
    oof = oof.loc[:, oof_columns].drop_duplicates(["record_id", "stage"])
    group = group.merge(oof, on=["record_id", "stage"], how="inner", validate="one_to_one")
    group = group.set_index(["record_id", "stage"]).sort_index()
    baseline = baseline.reindex(group.index)
    availability = availability.reindex(group.index).fillna(False)
    confidence = confidence.reindex(group.index).fillna(0.0)
    if baseline.isna().any().any():
        raise ValueError("baseline action matrix did not align with OOF groups")

    action_probability = np.column_stack(
        [
            _sigmoid(group[f"action_logit__{action}"].to_numpy(dtype=float))
            for action in LEARNED_ACTIONS
        ]
    )
    action_target = np.column_stack(
        [
            group[f"action_target__{action}"].to_numpy(dtype=np.int8)
            for action in LEARNED_ACTIONS
        ]
    )
    action_mask = np.column_stack(
        [
            group[f"action_mask__{action}"].to_numpy(dtype=bool)
            for action in LEARNED_ACTIONS
        ]
    ) & availability.to_numpy(dtype=bool)
    need = np.clip(1.0 - baseline.to_numpy(dtype=float), 0.0, 1.0)
    risk = group["feature__risk_probability"].to_numpy(dtype=float)
    entropy = normalized_binary_entropy(risk)
    uncertainty = np.repeat(entropy[:, None], len(LEARNED_ACTIONS), axis=1)
    workload = np.repeat(
        np.asarray([WORKLOAD[action] for action in LEARNED_ACTIONS], dtype=float)[None, :],
        len(group),
        axis=0,
    )

    risk_reduction = np.zeros_like(action_probability)
    simulation_status = "NOT_AVAILABLE"
    if simulation_path.is_file():
        simulation = pd.read_parquet(simulation_path)
        required_simulation = {"record_id", "stage", "action_id", "strength", "risk_delta"}
        missing_simulation = sorted(required_simulation.difference(simulation.columns))
        if missing_simulation:
            raise KeyError(f"simulation rows are missing columns: {missing_simulation}")
        simulation["record_id"] = simulation["record_id"].astype(str)
        simulation = simulation.loc[
            simulation["strength"].eq("moderate")
            & simulation["action_id"].isin(LEARNED_ACTIONS)
        ].copy()
        pivot = simulation.pivot_table(
            index=["record_id", "stage"],
            columns="action_id",
            values="risk_delta",
            aggfunc="mean",
        ).reindex(index=group.index, columns=LEARNED_ACTIONS).fillna(0.0)
        risk_reduction = pivot.to_numpy(dtype=float)
        simulation_status = "AVAILABLE"

    arrays = {
        "risk": risk,
        "support_target": group["prediction_target"].to_numpy(dtype=np.int8),
        "need": need,
        "maximum_need": np.where(action_mask, need, 0.0).max(axis=1),
        "entropy": entropy,
        "action_probability": action_probability,
        "action_target": action_target,
        "action_mask": action_mask,
        "evidence_confidence": confidence.to_numpy(dtype=float),
        "workload": workload,
        "uncertainty": uncertainty,
        "raw_risk_reduction": risk_reduction,
    }
    return group.reset_index(), arrays, simulation_status


def _per_stage_eligibility(
    frame: pd.DataFrame,
    target: np.ndarray,
    risk: np.ndarray,
    decisions: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for stage in STAGES:
        selected = frame["stage"].eq(stage).to_numpy()
        if selected.any():
            result[stage] = eligibility_metrics(
                target=target[selected],
                risk_probability=risk[selected],
                decisions=decisions[selected],
            )
    return result


def run(
    *,
    landmark_path: Path,
    oof_path: Path,
    labels_path: Path,
    simulation_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    frame, arrays, simulation_status = _prepare_arrays(
        landmark_path,
        oof_path,
        labels_path,
        simulation_path,
    )
    validation = frame["protocol_split"].eq("validation").to_numpy()
    test = frame["protocol_split"].eq("test").to_numpy()
    if not validation.any() or not test.any():
        raise ValueError("protocol_split must contain validation and test groups")
    validation_students = set(frame.loc[validation, "student_id"].astype(str))
    test_students = set(frame.loc[test, "student_id"].astype(str))
    if validation_students.intersection(test_students):
        raise RuntimeError("student leakage detected between validation and test")

    policy, validation_eligibility = select_eligibility_policy(
        validation_target=arrays["support_target"][validation],
        validation_risk_probability=arrays["risk"][validation],
        validation_need_score=arrays["maximum_need"][validation],
        validation_entropy=arrays["entropy"][validation],
        validation_seed_disagreement=np.zeros(int(validation.sum())),
    )
    validation_decisions = apply_eligibility_policy(
        risk_probability=arrays["risk"][validation],
        need_score=arrays["maximum_need"][validation],
        predictive_entropy=arrays["entropy"][validation],
        seed_disagreement=np.zeros(int(validation.sum())),
        policy=policy,
    )
    test_decisions = apply_eligibility_policy(
        risk_probability=arrays["risk"][test],
        need_score=arrays["maximum_need"][test],
        predictive_entropy=arrays["entropy"][test],
        seed_disagreement=np.zeros(int(test.sum())),
        policy=policy,
    )
    test_eligibility = eligibility_metrics(
        target=arrays["support_target"][test],
        risk_probability=arrays["risk"][test],
        decisions=test_decisions,
    )
    test_frame = frame.loc[test].reset_index(drop=True)

    reduction_normalizer = MinMaxNormalizer.fit(
        arrays["raw_risk_reduction"][validation],
        arrays["action_mask"][validation],
    )
    normalized_reduction = reduction_normalizer.transform(arrays["raw_risk_reduction"])
    validation_issued = validation_decisions == EligibilityDecision.BEHAVIOURAL_ACTION.value
    test_issued = test_decisions == EligibilityDecision.BEHAVIOURAL_ACTION.value
    validation_indices = np.flatnonzero(validation)[validation_issued]
    test_indices = np.flatnonzero(test)[test_issued]
    if not len(validation_indices) or not len(test_indices):
        raise RuntimeError("eligibility policy must issue behavioural actions in both splits")

    weights, validation_ranking = select_ranking_weights(
        action_probability=arrays["action_probability"][validation_indices],
        need_severity=arrays["need"][validation_indices],
        simulated_risk_reduction=normalized_reduction[validation_indices],
        evidence_confidence=arrays["evidence_confidence"][validation_indices],
        workload=arrays["workload"][validation_indices],
        uncertainty=arrays["uncertainty"][validation_indices],
        mask=arrays["action_mask"][validation_indices],
        target=arrays["action_target"][validation_indices],
    )
    test_scores = utility_scores(
        action_probability=arrays["action_probability"][test_indices],
        need_severity=arrays["need"][test_indices],
        simulated_risk_reduction=normalized_reduction[test_indices],
        evidence_confidence=arrays["evidence_confidence"][test_indices],
        workload=arrays["workload"][test_indices],
        uncertainty=arrays["uncertainty"][test_indices],
        mask=arrays["action_mask"][test_indices],
        weights=weights,
    )
    test_ranking = ranking_metrics(
        test_scores,
        arrays["action_target"][test_indices],
        arrays["action_mask"][test_indices],
    )
    all_positive_test = (
        arrays["action_target"][test] * arrays["action_mask"][test]
    ).sum(axis=1) > 0
    issued_positive = all_positive_test & test_issued
    test_ranking["positive_group_coverage"] = float(
        issued_positive.sum() / max(all_positive_test.sum(), 1)
    )
    test_ranking["issued_groups"] = int(test_issued.sum())

    validation_prevalence = (
        arrays["action_target"][validation_indices].sum(axis=0)
        / np.maximum(arrays["action_mask"][validation_indices].sum(axis=0), 1)
    )
    baseline_scores = ranking_baselines(
        action_probability=arrays["action_probability"][test_indices],
        need_severity=arrays["need"][test_indices],
        simulated_risk_reduction=normalized_reduction[test_indices],
        evidence_confidence=arrays["evidence_confidence"][test_indices],
        workload=arrays["workload"][test_indices],
        mask=arrays["action_mask"][test_indices],
        prevalence=validation_prevalence,
    )
    baselines = {
        name: ranking_metrics(
            score,
            arrays["action_target"][test_indices],
            arrays["action_mask"][test_indices],
        )
        for name, score in baseline_scores.items()
    }

    payload = {
        "status": "COMPLETE",
        "problem_definition": {
            "population": "ALL_CUTOFF_ELIGIBLE_OULAD_LEARNER_COURSE_STAGE_GROUPS",
            "eligibility_target": "HELD_OUT_FINAL_AT_RISK_OUTCOME_OFFLINE_ONLY",
            "ranking_target": "TRAIN_ONLY_SCIENTIFIC_SILVER_ACTION_LABELS",
            "simulation": "FROZEN_HYBRID_MODEL_BASED_SENSITIVITY",
        },
        "population": {
            "groups": int(len(frame)),
            "students": int(frame["student_id"].nunique()),
            "validation_groups": int(validation.sum()),
            "test_groups": int(test.sum()),
            "student_leakage_count": 0,
        },
        "eligibility": {
            "selected_policy": policy.to_dict(),
            "validation": validation_eligibility,
            "test": test_eligibility,
            "per_stage_test": _per_stage_eligibility(
                test_frame,
                arrays["support_target"][test],
                arrays["risk"][test],
                test_decisions,
            ),
        },
        "ranking": {
            "selected_weights": weights.to_dict(),
            "validation": validation_ranking,
            "test": test_ranking,
            "baselines": baselines,
            "simulation_component_status": simulation_status,
        },
        "action_order": list(LEARNED_ACTIONS),
        "governance_routes_are_outside_ranker": True,
        "test_used_for_policy_or_weight_selection": False,
        "runtime_authorized": False,
        "claim_boundary": "OFFLINE_EVALUATION_AND_MODEL_SENSITIVITY_NOT_CAUSAL_EFFECT",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    eligibility_test = payload["eligibility"]["test"]
    ranking_test = payload["ranking"]["test"]
    lines = [
        "# Recommendation V2 Full-Population Results",
        "",
        "## Problem definition",
        "",
        "1. Decide whether to issue support on every cutoff-eligible learner-stage group.",
        "2. Rank one of five observable behavioural actions only when behavioural support is issued.",
        "3. Use frozen-Hybrid risk simulation as sensitivity evidence, not causal evidence.",
        "",
        "## Eligibility on held-out test population",
        "",
        f"- Precision: **{eligibility_test['precision']:.4f}**",
        f"- Recall: **{eligibility_test['recall']:.4f}**",
        f"- F1: **{eligibility_test['f1']:.4f}**",
        f"- Balanced accuracy: **{eligibility_test['balanced_accuracy']:.4f}**",
        f"- PR-AUC: **{eligibility_test['pr_auc']:.4f}**",
        f"- Brier: **{eligibility_test['brier_score']:.4f}**",
        f"- ECE: **{eligibility_test['ece']:.4f}**",
        f"- Intervention rate: **{eligibility_test['intervention_rate']:.4f}**",
        f"- False-issue rate: **{eligibility_test['false_issue_rate']:.4f}**",
        f"- Missed-support rate: **{eligibility_test['missed_support_rate']:.4f}**",
        f"- Defer rate: **{eligibility_test['defer_rate']:.4f}**",
        "",
        "## Action ranking on issued test groups",
        "",
        f"- Precision@1: **{ranking_test['precision_at_1']:.4f}**",
        f"- Recall@1: **{ranking_test['recall_at_1']:.4f}**",
        f"- Recall@3: **{ranking_test['recall_at_3']:.4f}**",
        f"- NDCG@3: **{ranking_test['ndcg_at_3']:.4f}**",
        f"- MRR: **{ranking_test['mrr']:.4f}**",
        f"- Pairwise accuracy: **{ranking_test['pairwise_accuracy']:.4f}**",
        f"- Positive-group coverage: **{ranking_test['positive_group_coverage']:.4f}**",
        "",
        "## Interpretation",
        "",
        "Eligibility metrics use the full held-out population. Ranking metrics remain silver-label consistency metrics. Simulation deltas, when available, describe frozen-model response only and do not prove intervention effectiveness.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark", type=Path, default=DEFAULT_LANDMARK)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--simulation", type=Path, default=DEFAULT_SIMULATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(
        landmark_path=args.landmark,
        oof_path=args.oof,
        labels_path=args.labels,
        simulation_path=args.simulation,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
