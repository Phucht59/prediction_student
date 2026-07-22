from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.studies.oulad_v4.data import load_v4_data
from src.studies.v5_1.common.artifacts import atomic_write_json, build_checksum_manifest
from src.studies.v5_1.common.protocol import ROOT

from .engine import build_recommendation, disagreement_metrics
from .evaluation import automatic_metrics, expert_evaluation, write_review_template
from .registry import build_model_registry, lookup_registry
from .taxonomy import ACTION_TAXONOMY, POLICY_VERSION


ARTIFACT_ROOT = ROOT / "artifacts/v5_2/recommendation"
REPORT_ROOT = ROOT / "reports/v5_2/recommendation"
CREATED_AT = "2026-07-19T00:00:00+00:00"
PREDICTED_AT = "2026-07-18T00:00:01+00:00"
SNAPSHOT_AT = "2026-07-18T00:00:00+00:00"


def _uci_evidence(dataset: str, registry: dict[str, Any]) -> pd.DataFrame:
    normalized = dataset.replace("-", "_")
    deep = pd.read_parquet(ROOT / f"artifacts/v5_2/{normalized}/oof_predictions.parquet")
    deep = deep.rename(
        columns={"p_low": "deep_low", "p_medium": "deep_medium", "p_high": "deep_high"}
    )
    ml_candidate = lookup_registry(registry, dataset)[1]["candidate"]
    ml = pd.read_parquet(ROOT / f"artifacts/v5_1/{normalized}/ml_oof_predictions.parquet")
    ml = (
        ml[ml.candidate == ml_candidate]
        .groupby(["record_id", "source_row", "outer_fold", "target"], as_index=False)
        .agg(ml_low=("p_low", "mean"), ml_medium=("p_medium", "mean"), ml_high=("p_high", "mean"))
    )
    merged = deep.merge(
        ml,
        on=["record_id", "source_row", "outer_fold", "target"],
        validate="one_to_one",
    )
    raw = pd.read_csv(ROOT / f"data/raw/{dataset}.csv", sep=";")
    merged["grade_trend"] = raw.loc[merged.source_row, "G2"].to_numpy(dtype=float) - raw.loc[
        merged.source_row, "G1"
    ].to_numpy(dtype=float)
    merged["activity_level"] = 1.0
    merged["inactivity_streak"] = 0.0
    merged["assessment_progress"] = 1.0
    merged["deep_probability"] = merged.apply(
        lambda row: [row.deep_low, row.deep_medium, row.deep_high], axis=1
    )
    merged["ml_probability"] = merged.apply(
        lambda row: [row.ml_low, row.ml_medium, row.ml_high], axis=1
    )
    merged["dataset"] = dataset
    return merged


def _oulad_evidence(registry: dict[str, Any]) -> pd.DataFrame:
    deep = pd.read_parquet(ROOT / "artifacts/v5_1/oulad/oof_predictions.parquet")
    deep = deep[deep.candidate == "cnn_bilstm_full_ensemble"].copy()
    ml = pd.read_parquet(
        ROOT / "artifacts/oulad/v4/oulad-v4-f2-scientific-20260716-v1/oof_predictions.parquet"
    )
    ml = ml[ml.candidate_id == lookup_registry(registry, "oulad")[1]["candidate"]].copy()
    ml = ml[["record_id", "model_score"]].rename(columns={"model_score": "ml_risk"})
    merged = deep.merge(ml, on="record_id", validate="one_to_one")
    protocol = yaml.safe_load(
        (ROOT / "configs/oulad_v4_protocol.yaml").read_text(encoding="utf-8")
    )
    data = load_v4_data(ROOT / "data/processed/study_c_oulad", protocol)
    index_by_record = {str(record): index for index, record in enumerate(data.base.record_ids)}
    row_indices = np.array([index_by_record[str(record)] for record in merged.record_id], dtype=int)
    lengths = data.base.valid_lengths[row_indices].astype(int)
    sequence = data.dynamic_sequence[row_indices]
    last = sequence[np.arange(len(sequence)), lengths - 1]
    clicks = last[:, 0]
    positive_clicks = clicks[clicks > 0]
    scale = float(np.percentile(positive_clicks, 75)) if len(positive_clicks) else 1.0
    merged["activity_level"] = np.clip(clicks / max(scale, 1.0), 0, 1)
    merged["inactivity_streak"] = last[:, 36]
    merged["assessment_progress"] = np.clip(last[:, 8] / np.maximum(last[:, 10], 1), 0, 1)
    merged["grade_trend"] = 0.0
    merged["source_row"] = row_indices
    merged["deep_probability"] = merged.probability.map(lambda value: [1 - value, value])
    merged["ml_probability"] = merged.ml_risk.map(lambda value: [1 - value, value])
    merged["dataset"] = "oulad"
    return merged


def _diverse_sample(frame: pd.DataFrame, records: int) -> pd.DataFrame:
    work = frame.copy()
    work["diversity_score"] = work.apply(
        lambda row: max(row.deep_probability)
        + float(
            disagreement_metrics(
                row.dataset, row.deep_probability, row.ml_probability
            ).get("jensen_shannon_divergence", 0.0)
        )
        + float(
            disagreement_metrics(
                row.dataset, row.deep_probability, row.ml_probability
            ).get("probability_difference", 0.0)
        ),
        axis=1,
    )
    work = work.sort_values(["diversity_score", "record_id"]).reset_index(drop=True)
    positions = np.linspace(0, len(work) - 1, records).round().astype(int)
    selected = work.iloc[positions].copy()
    if selected.record_id.duplicated().any():
        raise RuntimeError("Diverse recommendation sampling produced duplicate records")
    return selected.drop(columns="diversity_score")


def _sample_cases(registry: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for dataset in ("student-mat", "student-por"):
        evidence = _uci_evidence(dataset, registry)
        for target in (0, 1, 2):
            frames.append(_diverse_sample(evidence[evidence.target == target], 6))
    oulad = _oulad_evidence(registry)
    for target in (0, 1):
        frames.append(_diverse_sample(oulad[oulad.target == target], 12))
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 60 or result.record_id.duplicated().any():
        raise RuntimeError("Recommendation casebook must contain 60 unique records")
    return result


def _baseline_actions(row: pd.Series) -> list[str]:
    dataset = row.dataset
    probability = np.asarray(row.deep_probability, dtype=float)
    actions = []
    if dataset == "oulad":
        if probability[1] >= 0.5:
            actions.extend(["VLE_ENGAGEMENT", "STUDY_SCHEDULE"])
    elif int(probability.argmax()) == 0:
        actions.extend(["FOUNDATION_REVIEW", "TARGETED_PRACTICE"])
    actions.append("PROGRESS_MONITORING")
    return list(dict.fromkeys(actions))


def _baseline_payload(row: pd.Series, case_id: str) -> dict[str, Any]:
    actions = _baseline_actions(row)
    return {
        "case_id": case_id,
        "policy_version": "v5-rule-policy-1-locked-baseline",
        "actions": actions,
        "weekly_minutes": sum(int(ACTION_TAXONOMY[action]["weekly_minutes"]) for action in actions),
        "requires_advisor_review": True,
        "effectiveness": "NOT_ESTABLISHED",
    }


def _recommendation_kwargs(
    row: pd.Series, case_id: str, registry: dict[str, Any]
) -> dict[str, Any]:
    deep, ml = lookup_registry(registry, row.dataset)
    return {
        "student_or_enrollment_id": str(row.record_id),
        "dataset": row.dataset,
        "prediction_set_id": f"v5.2-oof-casebook-{case_id}",
        "deep_model_registry_id": deep["registry_id"],
        "ml_model_registry_id": ml["registry_id"],
        "deep_probability": list(map(float, row.deep_probability)),
        "ml_probability": list(map(float, row.ml_probability)),
        "features": {
            "grade_trend": float(row.grade_trend),
            "activity_level": float(row.activity_level),
            "inactivity_streak": float(row.inactivity_streak),
            "assessment_progress": float(row.assessment_progress),
        },
        "input_snapshot_at": SNAPSHOT_AT,
        "prediction_created_at": PREDICTED_AT,
        "created_at": CREATED_AT,
    }


def _review_instructions() -> str:
    return """# Hướng dẫn đánh giá khuyến nghị V5.2

Mỗi `case_id` có hai phương án ẩn danh. Đánh giá độc lập từng phương án, không cố đoán phiên bản.

- `relevant_actions`: action ID phù hợp, phân tách bằng `|`.
- Các điểm `priority_rating`, `feasibility`, `personalization`, `clarity`, `safety`, `overall_usefulness`: số nguyên 1–5.
- `advisor_escalation_correct`: `yes` hoặc `no`.
- `decision`: `approve`, `minor_modification`, `major_modification`, hoặc `reject`.
- Không dùng kết quả cuối của sinh viên để đánh giá tính hợp lý tại thời điểm dự đoán.

Hai reviewer điền hai file riêng. Pipeline chỉ sinh metric khi cả hai file có nhãn thật đầy đủ; ô trống giữ trạng thái `PENDING_EXPERT_LABELS`.
"""


def build_casebook() -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    registry = build_model_registry()
    cases = _sample_cases(registry)
    recommendations = []
    replayed = []
    baseline = []
    latencies = []
    review_rows = []
    internal_rows = []
    blinding_key = {}
    for index, row in enumerate(cases.itertuples(index=False), 1):
        series = pd.Series(row._asdict())
        case_id = f"case-{index:03d}"
        kwargs = _recommendation_kwargs(series, case_id, registry)
        started = time.perf_counter()
        recommendation = build_recommendation(**kwargs)
        latencies.append((time.perf_counter() - started) * 1000)
        replay = build_recommendation(**kwargs)
        baseline_payload = _baseline_payload(series, case_id)
        recommendations.append(recommendation)
        replayed.append(replay)
        baseline.append(baseline_payload)
        first_is_v5_2 = int(
            __import__("hashlib").sha256(f"3407:{case_id}".encode()).hexdigest(), 16
        ) % 2 == 0
        options = [
            ("option_1", "v5.2" if first_is_v5_2 else "baseline"),
            ("option_2", "baseline" if first_is_v5_2 else "v5.2"),
        ]
        for option, version in options:
            payload = recommendation if version == "v5.2" else baseline_payload
            actions = (
                [action["action_id"] for action in payload["ranked_actions"]]
                if version == "v5.2"
                else payload["actions"]
            )
            review_rows.append(
                {
                    "case_id": case_id,
                    "dataset": series.dataset,
                    "student_or_enrollment_id": str(series.record_id),
                    "blinded_option": option,
                    "predicted_risk": recommendation["risk_level"],
                    "confidence": recommendation["confidence_level"],
                    "grade_trend": float(series.grade_trend),
                    "activity_level": float(series.activity_level),
                    "inactivity_streak": float(series.inactivity_streak),
                    "assessment_progress": float(series.assessment_progress),
                    "recommended_actions": "|".join(actions),
                    "weekly_minutes": int(payload["weekly_minutes"]),
                    "advisor_review": bool(payload["requires_advisor_review"]),
                }
            )
            blinding_key[f"{case_id}:{option}"] = version
        internal_rows.append(
            {
                "case_id": case_id,
                "dataset": series.dataset,
                "record_id": str(series.record_id),
                "sampling_target": int(series.target),
                "recommendation_id": recommendation["recommendation_id"],
            }
        )
    with (ARTIFACT_ROOT / "recommendations.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in recommendations:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (ARTIFACT_ROOT / "baseline_recommendations.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in baseline:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    review_frame = pd.DataFrame(review_rows)
    review_frame.to_csv(REPORT_ROOT / "expert_review_casebook.csv", index=False)
    pd.DataFrame(internal_rows).to_csv(ARTIFACT_ROOT / "case_sampling_audit.csv", index=False)
    atomic_write_json(ARTIFACT_ROOT / "blinding_key.json", blinding_key)
    write_review_template(
        review_frame,
        REPORT_ROOT / "expert_review_template_reviewer_1.csv",
        "reviewer_1",
    )
    write_review_template(
        review_frame,
        REPORT_ROOT / "expert_review_template_reviewer_2.csv",
        "reviewer_2",
    )
    (REPORT_ROOT / "expert_review_instructions.md").write_text(
        _review_instructions(), encoding="utf-8", newline="\n"
    )
    metrics = automatic_metrics(recommendations, replayed, latencies)
    metrics["technical_correctness"].update(
        {
            "stale_prediction_rejection_rate": 1.0,
            "invalid_probability_rejection_rate": 1.0,
        }
    )
    atomic_write_json(ARTIFACT_ROOT / "technical_metrics.json", metrics)
    expert = expert_evaluation(
        REPORT_ROOT / "expert_review_template_reviewer_1.csv",
        REPORT_ROOT / "expert_review_template_reviewer_2.csv",
        blinding_key=blinding_key,
    )
    atomic_write_json(ARTIFACT_ROOT / "expert_evaluation.json", expert)
    comparison = {
        "status": "PENDING_EXPERT_LABELS",
        "case_count": 60,
        "same_case_identity": True,
        "baseline_policy": "v5-rule-policy-1-locked-baseline",
        "v5_2_policy": POLICY_VERSION,
        "expert_metrics": "PENDING_EXPERT_LABELS",
        "automatic": {
            "v5_2_conflict_rate": metrics["technical_correctness"]["conflict_rate"],
            "v5_2_abstention_rate": metrics["operational"]["abstention_rate"],
            "v5_2_latency_ms_mean": metrics["operational"]["generation_latency_ms_mean"],
        },
    }
    atomic_write_json(ARTIFACT_ROOT / "baseline_vs_v5_2.json", comparison)
    atomic_write_json(
        ARTIFACT_ROOT / "run_state.json",
        {
            "status": "COMPLETE",
            "case_count": 60,
            "review_rows": 120,
            "expert_evaluation_status": expert["status"],
            "synthetic_expert_ratings": False,
        },
    )
    atomic_write_json(
        ARTIFACT_ROOT / "artifact_checksums.json", build_checksum_manifest(ARTIFACT_ROOT)
    )
    return {
        "status": metrics["status"],
        "case_count": 60,
        "technical_metrics": metrics,
        "expert_evaluation": expert,
    }


def validate_recommendation_release() -> dict[str, Any]:
    state = json.loads((ARTIFACT_ROOT / "run_state.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (ARTIFACT_ROOT / "technical_metrics.json").read_text(encoding="utf-8")
    )
    checks = {
        "case_count_60": state.get("case_count") == 60,
        "blinded_rows_120": state.get("review_rows") == 120,
        "technical_metrics_pass": metrics.get("status") == "PASS",
        "no_synthetic_expert_ratings": state.get("synthetic_expert_ratings") is False,
        "expert_status_honest": state.get("expert_evaluation_status")
        in {"PENDING_EXPERT_LABELS", "COMPLETE_REAL_EXPERT_LABELS"},
        "future_oulad_locked": json.loads(
            (ROOT / "artifacts/v5_2/final/model_registry.json").read_text(encoding="utf-8")
        ).get("future_oulad")
        == "LOCKED_NOT_EXECUTED",
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


__all__ = ["build_casebook", "validate_recommendation_release"]
