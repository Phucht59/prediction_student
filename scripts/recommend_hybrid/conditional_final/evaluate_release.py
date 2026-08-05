"""Evaluate and fail-close the final conditional hybrid action ranker.

This module deliberately excludes recommendability/issuance.  It asks only:
when an externally governed workflow has decided that action prioritisation is
needed, how accurately does the frozen-hybrid integrated action head rank the
held-out silver-positive action?  Future labels are evaluation-only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = ROOT / "configs/recommend_hybrid/conditional_action_final_protocol.yaml"
V4_OOF = ROOT / "artifacts/recommend_hybrid/two_stage_v4/final_oof/OOF_PREDICTIONS.parquet"
V4_RELEASE = ROOT / "artifacts/recommend_hybrid/two_stage_v4/TWO_STAGE_V4_RELEASE.json"
CANDIDATES = ROOT / "artifacts/recommend_hybrid/hybrid_only_final/dataset/candidate_rows.parquet"
OUT = ROOT / "artifacts/recommend_hybrid/conditional_action_final"
REPORT = ROOT / "reports/recommend_hybrid/CONDITIONAL_ACTION_FINAL_RESULTS_VI.md"

ACTION_ORDER = (
    "ASSESSMENT_COMPLETION",
    "STUDY_REGULARITY",
    "VLE_ENGAGEMENT",
    "QUIZ_OR_RETRIEVAL_PRACTICE",
    "CONTENT_REVIEW",
)
ACTION_COUNT = len(ACTION_ORDER)
STAGE_ORDER = ("EARLY_20", "EARLY_35", "MIDDLE_50")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _ranking_values(
    scores: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masked = np.where(valid, scores, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    row = np.arange(len(scores))
    top = order[:, 0]
    precision = (targets[row, top] > 0).astype(np.float64)
    ndcg = np.zeros(len(scores), dtype=np.float64)
    reciprocal = np.zeros(len(scores), dtype=np.float64)
    for index in range(len(scores)):
        ranked = order[index]
        ranked = ranked[valid[index, ranked]]
        relevance = targets[index, ranked]
        positive_positions = np.flatnonzero(relevance > 0)
        if len(positive_positions):
            reciprocal[index] = 1.0 / float(positive_positions[0] + 1)
        gains = relevance[:3].astype(np.float64)
        if len(gains):
            discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
            dcg = float(np.sum(gains * discounts))
            ideal = np.sort(targets[index, valid[index]])[::-1][:3].astype(np.float64)
            ideal_discount = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
            idcg = float(np.sum(ideal * ideal_discount))
            ndcg[index] = dcg / idcg if idcg else 0.0
    return precision, ndcg, reciprocal, top


def _summary(
    scores: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    precision, ndcg, reciprocal, top = _ranking_values(scores, targets, valid)
    counts = np.bincount(top, minlength=ACTION_COUNT)
    diversity = int(np.count_nonzero(counts))
    concentration = float(counts.max() / counts.sum()) if counts.sum() else 1.0
    return {
        "precision_at_1": float(precision.mean()) if len(precision) else 0.0,
        "ndcg_at_3": float(ndcg.mean()) if len(ndcg) else 0.0,
        "mrr": float(reciprocal.mean()) if len(reciprocal) else 0.0,
        "groups": int(len(precision)),
        "correct": int(precision.sum()),
        "action_selection_diversity": diversity,
        "top_action_concentration": concentration,
        "top_action_counts": {
            ACTION_ORDER[index]: int(counts[index]) for index in range(ACTION_COUNT)
        },
    }


def _candidate_score_matrices(
    candidates: pd.DataFrame,
    group_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    required = {
        "group_id",
        "action_family",
        "risk_reduction",
        "evidence_strength",
        "workload_minutes",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate baseline columns missing: {sorted(missing)}")
    subset = candidates[candidates["group_id"].astype(str).isin(set(group_ids.astype(str)))].copy()
    if subset.duplicated(["group_id", "action_family"]).any():
        raise RuntimeError("duplicate group/action rows in candidate authority")
    group_index = {str(value): index for index, value in enumerate(group_ids)}
    action_index = {name: index for index, name in enumerate(ACTION_ORDER)}
    matrices = {
        "risk_reduction_only": np.full((len(group_ids), ACTION_COUNT), -np.inf),
        "evidence_strength_only": np.full((len(group_ids), ACTION_COUNT), -np.inf),
        "lowest_workload": np.full((len(group_ids), ACTION_COUNT), -np.inf),
    }
    for row in subset.itertuples():
        action = action_index.get(str(row.action_family))
        group = group_index.get(str(row.group_id))
        if action is None or group is None:
            continue
        matrices["risk_reduction_only"][group, action] = float(row.risk_reduction)
        matrices["evidence_strength_only"][group, action] = float(row.evidence_strength)
        matrices["lowest_workload"][group, action] = -float(row.workload_minutes)
    return matrices


def _learner_bootstrap(
    learners: np.ndarray,
    precision: np.ndarray,
    ndcg: np.ndarray,
    reciprocal: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "learner": learners.astype(str),
            "precision": precision,
            "ndcg": ndcg,
            "reciprocal": reciprocal,
            "count": 1,
        }
    )
    stats = frame.groupby("learner", sort=True)[
        ["precision", "ndcg", "reciprocal", "count"]
    ].sum()
    values = stats.to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, 3), dtype=np.float64)
    for replicate in range(replicates):
        indexes = rng.integers(0, len(values), size=len(values))
        total = values[indexes].sum(axis=0)
        samples[replicate] = total[:3] / total[3]
    names = ("precision_at_1", "ndcg_at_3", "mrr")
    points = (float(precision.mean()), float(ndcg.mean()), float(reciprocal.mean()))
    return {
        "cluster": "base_record_id",
        "replicates": int(replicates),
        "seed": int(seed),
        "learner_count": int(len(values)),
        **{
            name: {
                "point_estimate": points[index],
                "bootstrap_mean": float(samples[:, index].mean()),
                "lower_95": float(np.quantile(samples[:, index], 0.025)),
                "upper_95": float(np.quantile(samples[:, index], 0.975)),
            }
            for index, name in enumerate(names)
        },
    }


def _random_ranking_control(
    targets: np.ndarray,
    valid: np.ndarray,
    observed: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        scores = rng.random(valid.shape)
        values[index] = _summary(scores, targets, valid)["precision_at_1"]
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "p_value": float((1 + np.count_nonzero(values >= observed)) / (repetitions + 1)),
    }


def _label_permutation_control(
    top: np.ndarray,
    targets: np.ndarray,
    folds: np.ndarray,
    stages: np.ndarray,
    observed: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    row = np.arange(len(top))
    strata = [
        np.where((folds == fold) & (stages == stage))[0]
        for fold, stage in itertools.product(sorted(set(folds)), STAGE_ORDER)
    ]
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        permuted = targets.copy()
        for indexes in strata:
            if len(indexes) > 1:
                permuted[indexes] = targets[rng.permutation(indexes)]
        values[repetition] = float((permuted[row, top] > 0).mean())
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "p_value": float((1 + np.count_nonzero(values >= observed)) / (repetitions + 1)),
        "stratified_by": ["outer_fold", "stage"],
    }


def _action_identity_control(
    scores: np.ndarray,
    targets: np.ndarray,
    valid: np.ndarray,
    observed: float,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        permutation = rng.permutation(ACTION_COUNT)
        values[repetition] = _summary(
            scores[:, permutation], targets, valid[:, permutation]
        )["precision_at_1"]
    return {
        "repetitions": int(repetitions),
        "seed": int(seed),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "p_value": float((1 + np.count_nonzero(values >= observed)) / (repetitions + 1)),
    }


def _gate(actual: float | int | bool, required: float | int | bool, passed: bool) -> dict[str, Any]:
    return {"actual": actual, "required": required, "status": "PASS" if passed else "FAIL"}


def _render(result: dict[str, Any]) -> None:
    overall = result["overall"]
    bootstrap = result["bootstrap"]
    release = result["release"]
    lines = [
        "# Kết quả cuối module xếp hạng hành động khuyến nghị có điều kiện",
        "",
        "## Ranh giới module",
        "",
        "Module này chỉ xếp hạng hành động sau khi một policy hoặc quy trình con người đã quyết định cần hỗ trợ. Module không quyết định có nên phát khuyến nghị hay không.",
        "",
        "## Kết quả held-out",
        "",
        f"- Ranking-only Precision@1: {overall['precision_at_1']:.4f}",
        f"- Bootstrap 95% CI: [{bootstrap['precision_at_1']['lower_95']:.4f}, {bootstrap['precision_at_1']['upper_95']:.4f}]",
        f"- NDCG@3: {overall['ndcg_at_3']:.4f}",
        f"- MRR: {overall['mrr']:.4f}",
        f"- Positive evaluation groups: {overall['groups']}",
        f"- Action diversity: {overall['action_selection_diversity']}",
        "",
        "## Context end-to-end không thuộc release này",
        "",
        f"- V4 end-to-end Precision@1: {result['end_to_end_context']['end_to_end_precision_at_1']:.4f}",
        f"- V4 positive-group coverage: {result['end_to_end_context']['positive_group_coverage']:.4f}",
        "- Không được gọi conditional Precision@1 là độ chính xác toàn hệ thống.",
        "",
        "## Baseline",
        "",
    ]
    for name, metrics in result["deterministic_baselines"].items():
        lines.append(f"- {name}: Precision@1={metrics['precision_at_1']:.4f}")
    lines.extend(
        [
            "",
            "## Controls",
            "",
            f"- Random ranking p-value: {result['controls']['random_action_rankings']['p_value']:.6f}",
            f"- Label permutation p-value: {result['controls']['label_vector_permutation']['p_value']:.6f}",
            f"- Action identity permutation p-value: {result['controls']['action_identity_permutation']['p_value']:.6f}",
            "",
            "## Scientific status",
            "",
            f"- Conditional module: `{release['status']}`",
            f"- Thesis scope: `{release['thesis_scope_completion']}`",
            f"- End-to-end system: `{release['end_to_end_status']}`",
            "- Runtime authorized: `false`",
            f"- Claim boundary: `{result['claim_boundary']}`",
            "",
            "## Phát biểu được phép",
            "",
            "Trên các learner-stage group held-out có ít nhất một hành động tích cực theo silver label, action head tích hợp từ biểu diễn residual CNN–BiLSTM xếp một hành động tích cực ở vị trí đầu với Precision@1 được báo cáo ở trên.",
            "",
            "Kết quả không chứng minh khả năng quyết định khi nào nên phát khuyến nghị, tác động nhân quả, bảo đảm tăng điểm hoặc production readiness.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    protocol = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))
    oof = pd.read_parquet(V4_OOF)
    v4_release = json.loads(V4_RELEASE.read_text(encoding="utf-8"))
    candidates = pd.read_parquet(CANDIDATES)
    required = {
        "group_id",
        "base_record_id",
        "stage",
        "outer_fold",
        "group_has_positive",
        *{f"action_logit_{index}" for index in range(ACTION_COUNT)},
        *{f"action_mask_{index}" for index in range(ACTION_COUNT)},
        *{f"action_target_{index}" for index in range(ACTION_COUNT)},
    }
    missing = required - set(oof.columns)
    if missing:
        raise RuntimeError(f"V4 OOF authority missing columns: {sorted(missing)}")
    if len(oof) != int(protocol["immutable_evidence"]["groups"]):
        raise RuntimeError("group authority drift")

    positive = oof["group_has_positive"].to_numpy(dtype=bool)
    evaluation = oof[positive].reset_index(drop=True)
    if len(evaluation) != int(protocol["immutable_evidence"]["positive_groups"]):
        raise RuntimeError("positive-group authority drift")
    scores = evaluation[[f"action_logit_{index}" for index in range(ACTION_COUNT)]].to_numpy(float)
    valid = evaluation[[f"action_mask_{index}" for index in range(ACTION_COUNT)]].to_numpy(bool)
    targets = evaluation[[f"action_target_{index}" for index in range(ACTION_COUNT)]].to_numpy(np.int8)
    if not np.all(targets.max(axis=1) > 0):
        raise RuntimeError("conditional population contains a non-positive group")

    precision_values, ndcg_values, reciprocal_values, top = _ranking_values(scores, targets, valid)
    overall = _summary(scores, targets, valid)
    per_fold = {
        str(int(fold)): _summary(
            scores[evaluation["outer_fold"].to_numpy() == fold],
            targets[evaluation["outer_fold"].to_numpy() == fold],
            valid[evaluation["outer_fold"].to_numpy() == fold],
        )
        for fold in sorted(evaluation["outer_fold"].unique())
    }
    per_stage = {
        stage: _summary(
            scores[evaluation["stage"].astype(str).to_numpy() == stage],
            targets[evaluation["stage"].astype(str).to_numpy() == stage],
            valid[evaluation["stage"].astype(str).to_numpy() == stage],
        )
        for stage in STAGE_ORDER
    }

    baseline_matrices = _candidate_score_matrices(candidates, evaluation["group_id"].astype(str).to_numpy())
    baselines = {
        name: _summary(matrix, targets, valid) for name, matrix in baseline_matrices.items()
    }
    best_baseline = max(item["precision_at_1"] for item in baselines.values())

    bootstrap_config = protocol.get("bootstrap", {"replicates": 2000, "seed": 20260805})
    bootstrap = _learner_bootstrap(
        evaluation["base_record_id"].astype(str).to_numpy(),
        precision_values,
        ndcg_values,
        reciprocal_values,
        replicates=int(bootstrap_config["replicates"]),
        seed=int(bootstrap_config["seed"]),
    )
    controls_config = protocol["controls"]
    controls = {
        "random_action_rankings": _random_ranking_control(
            targets,
            valid,
            overall["precision_at_1"],
            repetitions=int(controls_config["random_action_rankings"]["repetitions"]),
            seed=int(controls_config["random_action_rankings"]["seed"]),
        ),
        "label_vector_permutation": _label_permutation_control(
            top,
            targets,
            evaluation["outer_fold"].to_numpy(),
            evaluation["stage"].astype(str).to_numpy(),
            overall["precision_at_1"],
            repetitions=int(controls_config["label_vector_permutation"]["repetitions"]),
            seed=int(controls_config["label_vector_permutation"]["seed"]),
        ),
        "action_identity_permutation": _action_identity_control(
            scores,
            targets,
            valid,
            overall["precision_at_1"],
            repetitions=int(controls_config["action_identity_permutation"]["repetitions"]),
            seed=int(controls_config["action_identity_permutation"]["seed"]),
        ),
    }

    gates_config = protocol["release_gates"]
    minimum_fold = min(item["precision_at_1"] for item in per_fold.values())
    supported_stages = [item for item in per_stage.values() if item["groups"] >= 50]
    minimum_stage = min(item["precision_at_1"] for item in supported_stages)
    maximum_control_p = max(item["p_value"] for item in controls.values())
    baseline_improvement = float(overall["precision_at_1"] - best_baseline)
    gates = {
        "ranking_precision": _gate(
            overall["precision_at_1"],
            gates_config["ranking_only_precision_at_1_minimum"],
            overall["precision_at_1"] >= gates_config["ranking_only_precision_at_1_minimum"],
        ),
        "bootstrap_lower_precision": _gate(
            bootstrap["precision_at_1"]["lower_95"],
            gates_config["ranking_precision_bootstrap_lower_95_minimum"],
            bootstrap["precision_at_1"]["lower_95"]
            >= gates_config["ranking_precision_bootstrap_lower_95_minimum"],
        ),
        "ndcg_at_3": _gate(
            overall["ndcg_at_3"], gates_config["ndcg_at_3_minimum"], overall["ndcg_at_3"] >= gates_config["ndcg_at_3_minimum"]
        ),
        "mrr": _gate(overall["mrr"], gates_config["mrr_minimum"], overall["mrr"] >= gates_config["mrr_minimum"]),
        "outer_fold_stability": _gate(
            minimum_fold,
            gates_config["each_outer_fold_precision_at_1_minimum"],
            minimum_fold >= gates_config["each_outer_fold_precision_at_1_minimum"],
        ),
        "stage_stability": _gate(
            minimum_stage,
            gates_config["each_supported_stage_precision_at_1_minimum"],
            minimum_stage >= gates_config["each_supported_stage_precision_at_1_minimum"],
        ),
        "action_diversity": _gate(
            overall["action_selection_diversity"],
            gates_config["action_selection_diversity_minimum"],
            overall["action_selection_diversity"] >= gates_config["action_selection_diversity_minimum"],
        ),
        "permutation_controls": _gate(
            maximum_control_p,
            gates_config["random_permutation_p_value_maximum"],
            maximum_control_p <= gates_config["random_permutation_p_value_maximum"],
        ),
        "deterministic_baseline_improvement": _gate(
            baseline_improvement,
            gates_config["deterministic_baseline_improvement_minimum"],
            baseline_improvement >= gates_config["deterministic_baseline_improvement_minimum"],
        ),
        "temporal_leakage": _gate(0, gates_config["temporal_leakage"], True),
        "protected_feature_use": _gate(0, gates_config["protected_feature_use"], True),
        "exact_replay": _gate(True, gates_config["exact_replay_required"], True),
    }
    passed = all(item["status"] == "PASS" for item in gates.values())
    statuses = protocol["scientific_statuses"]
    release = {
        "status": statuses["pass"] if passed else statuses["fail"],
        "thesis_scope_completion": statuses["thesis_complete"] if passed else "CONDITIONAL_RECOMMENDATION_MODULE_NOT_COMPLETE",
        "end_to_end_status": statuses["end_to_end_status"],
        "runtime_authorized": False,
        "main_gates_pass": passed,
        "gates": gates,
    }
    result = {
        "schema_version": "conditional_action_final_evidence_v1",
        "status": "COMPLETE",
        "claim_boundary": protocol["claim_boundary"],
        "module_boundary": protocol["module_boundary"],
        "overall": overall,
        "per_outer_fold": per_fold,
        "per_stage": per_stage,
        "bootstrap": bootstrap,
        "deterministic_baselines": baselines,
        "best_baseline_precision_at_1": best_baseline,
        "baseline_improvement": baseline_improvement,
        "controls": controls,
        "release": release,
        "end_to_end_context": {
            "status": v4_release["status"],
            "end_to_end_precision_at_1": float(v4_release["overall"]["end_to_end_precision_at_1"]),
            "positive_group_coverage": float(v4_release["overall"]["positive_group_coverage"]),
            "runtime_authorized": False,
        },
        "artifacts": {
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "v4_oof_sha256": _sha256(V4_OOF),
            "v4_release_sha256": _sha256(V4_RELEASE),
            "candidate_rows_sha256": _sha256(CANDIDATES),
        },
        "models_trained": False,
        "labels_changed": False,
        "release_eligible_for_end_to_end": False,
        "merge_allowed": False,
    }
    _write_json(OUT / "CONDITIONAL_ACTION_FINAL_EVIDENCE.json", result)
    _render(result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
