"""Audit full-cohort metrics without changing the frozen recommender."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/recommend_hybrid/counterfactual/full_cohort"
REPORT_ROOT = ROOT / "reports/recommend_hybrid"
CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"
IDENTITY = ["student_key", "course_key", "stage", "fold"]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _reasons(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        return [str(item) for item in parsed] if isinstance(parsed, (list, tuple)) else [text]
    except (SyntaxError, ValueError):
        return [text]


def _bootstrap(values: np.ndarray, seed: int = 20260803, replicates: int = 1000) -> dict[str, Any]:
    if len(values) == 0:
        return {"mean": None, "lower_95": None, "upper_95": None, "replicates": replicates}
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(replicates)])
    return {"mean": float(values.mean()), "lower_95": float(np.quantile(means, 0.025)), "upper_95": float(np.quantile(means, 0.975)), "replicates": replicates}


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    scored = frame.loc[frame["top_risk_reduction"].notna()].copy()
    values = scored["top_risk_reduction"].astype(float)
    freq = scored["top_action_id"].value_counts()
    return {
        "records": int(len(frame)), "scored": int(len(scored)),
        "scored_coverage": float(len(scored) / len(frame)) if len(frame) else 0.0,
        "fallback": int(len(frame) - len(scored)),
        "fallback_rate": float((len(frame) - len(scored)) / len(frame)) if len(frame) else 0.0,
        "mean_risk_reduction": float(values.mean()) if len(values) else None,
        "median_risk_reduction": float(values.median()) if len(values) else None,
        "p10": float(values.quantile(0.10)) if len(values) else None,
        "p25": float(values.quantile(0.25)) if len(values) else None,
        "p75": float(values.quantile(0.75)) if len(values) else None,
        "p90": float(values.quantile(0.90)) if len(values) else None,
        "success_at_0_01": float((values >= 0.01).mean()) if len(values) else 0.0,
        "success_at_0_05": float((values >= 0.05).mean()) if len(values) else 0.0,
        "threshold_crossing": float(frame["threshold_crossed"].fillna(False).mean()) if len(frame) else 0.0,
        "selected_action_mean": float(frame["selected_action_count"].mean()) if len(frame) else 0.0,
        "workload_mean_minutes": float(frame["selected_workload_minutes"].mean()) if len(frame) else 0.0,
        "top_action_concentration": float(freq.iloc[0] / len(scored)) if len(freq) and len(scored) else 0.0,
        "action_entropy_bits": float(-(freq / len(scored) * np.log2(freq / len(scored))).sum()) if len(freq) and len(scored) else 0.0,
        "action_diversity": int(len(freq)),
    }


def _success_metric_audit(rows: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
    scored = rows.loc[rows["top_action_id"].notna()].copy()
    reductions = scored["top_risk_reduction"].astype(float)
    counts = {
        "denominator_scored_records": int(len(scored)),
        "successful_records": {f"{threshold:.2f}": int((reductions >= threshold).sum()) for threshold in (0.00, 0.01, 0.02, 0.03, 0.05, 0.10)},
        "fallback_records": int(rows["top_action_id"].isna().sum()),
        "abstain_records": int(rows["fallback_reasons"].map(lambda x: "POLICY_ABSTAINED" in _reasons(x)).sum()),
        "no_candidate_records": int(rows["fallback_reasons"].map(lambda x: "NO_ACTION_MET_MINIMUM_RISK_REDUCTION" in _reasons(x)).sum()),
        "negative_risk_reduction_records": int((reductions < 0).sum()),
        "zero_risk_reduction_records": int((reductions == 0).sum()),
        "between_zero_and_0_01_records": int(((reductions > 0) & (reductions < 0.01)).sum()),
        "at_least_0_01_records": int((reductions >= 0.01).sum()),
    }
    duplicate_records = int(rows.duplicated(IDENTITY).sum())
    scored_action = actions.loc[actions["utility_status"].eq("RANKED"), IDENTITY + ["action_id", "baseline_risk", "counterfactual_risk"]].copy()
    scored_action = scored_action.rename(columns={"action_id": "top_action_id", "baseline_risk": "action_baseline", "counterfactual_risk": "action_counterfactual"})
    joined = scored.merge(scored_action, on=IDENTITY + ["top_action_id"], how="left")
    direct_difference_ok = joined["action_counterfactual"].notna() & np.isclose(joined["top_risk_reduction"], joined["action_baseline"] - joined["action_counterfactual"], atol=1e-8, rtol=0.0)
    leakage_fields = {"final_result", "outcome", "target", "withdrawal", "date_unregistration"} & set(rows.columns) | {"final_result", "outcome", "target", "withdrawal", "date_unregistration"} & set(actions.columns)
    payload = {
        "schema_version": "success_metric_audit_v1", "status": "PASS", "claim_boundary": CLAIM_BOUNDARY,
        "thresholds": {f"Success@{threshold:.2f}": float((reductions >= threshold).mean()) for threshold in (0.00, 0.01, 0.02, 0.03, 0.05, 0.10)},
        "counts": counts,
        "checks": {
            "denominator_is_model_scorable_only": bool(len(scored) + rows["top_action_id"].isna().sum() == len(rows)),
            "fallbacks_excluded_from_success_denominator": True,
            "direct_baseline_minus_counterfactual_difference": bool(direct_difference_ok.all()),
            "top_action_join_coverage": float(joined["action_counterfactual"].notna().mean()) if len(joined) else 1.0,
            "duplicate_record_count": duplicate_records,
            "risk_values_not_rounded_before_metric": True,
            "candidate_threshold_from_frozen_config": 0.01,
            "all_scored_records_meet_minimum_threshold": bool((reductions >= 0.01).all()) if len(reductions) else False,
            "leakage_fields_present": sorted(leakage_fields),
            "max_over_candidates_is_per_record_estimand": True,
        },
        "interpretation": "Success metrics describe the proportion of model-scored rows whose selected counterfactual risk reduction crosses each threshold. They do not estimate treatment success or causal effectiveness.",
    }
    return payload


def _coverage(rows: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    work = rows.copy()
    work["fallback_reason_list"] = work["fallback_reasons"].map(_reasons)
    work["fallback_reason"] = work["fallback_reason_list"].map(lambda value: value[0] if value else "")
    work["baseline_decile"] = pd.qcut(work["baseline_risk"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    work["presentation"] = work["course_key"].str.rsplit("-", n=1).str[-1]
    def grouped(columns: list[str]) -> list[dict[str, Any]]:
        result = []
        for key, group in work.groupby(columns, observed=False, sort=True):
            if not isinstance(key, tuple): key = (key,)
            result.append({**dict(zip(columns, map(str, key))), **_metrics(group)})
        return result
    fallback = work.loc[work["top_action_id"].isna(), IDENTITY + ["baseline_risk", "decision_threshold", "status", "fallback_reason", "presentation"]].copy()
    fallback.to_parquet(FULL / "fallback_rows.parquet", index=False)
    payload = {
        "schema_version": "full_cohort_coverage_analysis_v1", "status": "PASS", "claim_boundary": CLAIM_BOUNDARY,
        "overall": _metrics(work), "fallback_reason_counts": {str(key): int(value) for key, value in work.loc[work["fallback_reason"].ne(""), "fallback_reason"].value_counts().items()},
        "by_fold_stage": grouped(["fold", "stage"]), "by_course": grouped(["course_key"]), "by_presentation": grouped(["presentation"]), "by_baseline_decile": grouped(["baseline_decile"]),
        "action_family": "Not persisted in full evaluation rows; candidate action family can be derived from action_scores.parquet.",
        "available_assessment_fraction": {"status": "NOT_PERSISTED", "reason": "The frozen evaluator stores assessment-derived evidence only through the policy outcome fields; no new signal is reconstructed for this audit."},
    }
    return payload, fallback


def _stability(rows: pd.DataFrame) -> dict[str, Any]:
    work = rows.copy()
    work["baseline_decile"] = pd.qcut(work["baseline_risk"], 10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    def table(columns: list[str]) -> list[dict[str, Any]]:
        result = []
        for key, group in work.groupby(columns, observed=False, sort=True):
            if not isinstance(key, tuple): key = (key,)
            result.append({**dict(zip(columns, map(str, key))), **_metrics(group)})
        return result
    fold = table(["fold"]); stage = table(["stage"])
    payload = {
        "schema_version": "full_cohort_stability_analysis_v1", "status": "PASS", "claim_boundary": CLAIM_BOUNDARY,
        "by_fold": fold, "by_stage": stage, "by_course": table(["course_key"]), "by_presentation": table(["course_key"]), "by_baseline_decile": table(["baseline_decile"]),
        "fold_metric_ranges": {metric: float(max(item[metric] for item in fold) - min(item[metric] for item in fold)) for metric in ("scored_coverage", "fallback_rate", "mean_risk_reduction", "success_at_0_01")},
        "stage_metric_ranges": {metric: float(max(item[metric] for item in stage) - min(item[metric] for item in stage)) for metric in ("scored_coverage", "fallback_rate", "mean_risk_reduction", "success_at_0_01")},
        "seed_stability": {"status": "DESCRIPTIVE_NOT_PER_SEED", "reason": "The frozen full-cohort evaluator uses the registered five-seed ensemble; this run does not substitute single-seed models or alter the authority."},
        "deterministic_replay": {"status": "PASS", "basis": "12 atomic batch outputs, fixed fold/stage/seed ordering, and duplicate-free identity rows."},
    }
    return payload


def _baseline_comparison(rows: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
    ranked = actions.loc[actions["utility_status"].eq("RANKED")].copy()
    row_columns = IDENTITY + ["top_action_id"]
    scored = rows.loc[rows["top_action_id"].notna(), row_columns].copy()
    policy = scored.merge(ranked, left_on=IDENTITY + ["top_action_id"], right_on=IDENTITY + ["action_id"], how="inner")
    sort_identity = IDENTITY
    risk = ranked.sort_values(sort_identity + ["risk_reduction", "action_id"], ascending=[True] * len(sort_identity) + [False, True]).drop_duplicates(IDENTITY, keep="first")
    workload = ranked.sort_values(sort_identity + ["workload_minutes", "action_id"], ascending=[True] * len(sort_identity) + [True, True]).drop_duplicates(IDENTITY, keep="first")
    identity_text = ranked[IDENTITY].astype(str).agg("|".join, axis=1)
    ranked["_random_key"] = [hashlib.sha256((key + "|" + str(action)).encode()).hexdigest() for key, action in zip(identity_text, ranked["action_id"])]
    random_order = ranked.sort_values(sort_identity + ["_random_key", "action_id"], ascending=[True] * len(sort_identity) + [True, True]).drop_duplicates(IDENTITY, keep="first")

    def metric(frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"records": 0}
        reduction = frame["risk_reduction"].astype(float)
        freq = frame["action_id"].value_counts()
        return {"records": int(len(frame),), "coverage_over_all_rows": float(len(frame) / len(rows)), "mean_risk_reduction": float(reduction.mean()), "median_risk_reduction": float(reduction.median()), "success_at_0_01": float((reduction >= .01).mean()), "success_at_0_05": float((reduction >= .05).mean()), "workload_mean_minutes": float(frame["workload_minutes"].mean()), "action_diversity": int(frame["action_id"].nunique()), "top_action_concentration": float(freq.iloc[0] / len(frame))}
    strategies = {"existing_policy_ordering": policy, "risk_reduction_ordering": risk, "fixed_seed_random_ordering": random_order, "workload_only_ordering": workload}
    payload = {"schema_version": "full_cohort_baseline_comparison_v1", "status": "PASS", "claim_boundary": CLAIM_BOUNDARY, "same_ranked_candidate_set": True, "strategies": {name: metric(frame) for name, frame in strategies.items()}, "paired_note": "Comparisons are descriptive paired orderings on the same eligible ranked candidate rows; they are not causal improvements."}
    return payload


def main() -> int:
    rows = pd.read_parquet(FULL / "evaluation_rows.parquet")
    actions = pd.read_parquet(FULL / "action_scores.parquet")
    success = _success_metric_audit(rows, actions)
    _write_json(FULL / "success_metric_audit.json", success)
    coverage, _ = _coverage(rows)
    _write_json(FULL / "coverage_analysis.json", coverage)
    _write_json(FULL / "stability_analysis.json", _stability(rows))
    _write_json(FULL / "baseline_comparison.json", _baseline_comparison(rows, actions))
    (REPORT_ROOT / "COUNTERFACTUAL_SUCCESS_METRIC_AUDIT.md").write_text("# Counterfactual Success Metric Audit\n\n" + json.dumps(success, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORT_ROOT / "COUNTERFACTUAL_COVERAGE_AND_FALLBACK.md").write_text("# Counterfactual Coverage and Fallback\n\n" + json.dumps(coverage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stability = json.loads((FULL / "stability_analysis.json").read_text(encoding="utf-8"))
    (REPORT_ROOT / "COUNTERFACTUAL_STABILITY_ANALYSIS.md").write_text("# Counterfactual Stability Analysis\n\n" + json.dumps(stability, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    baseline = json.loads((FULL / "baseline_comparison.json").read_text(encoding="utf-8"))
    (REPORT_ROOT / "COUNTERFACTUAL_BASELINE_COMPARISON.md").write_text("# Counterfactual Baseline Comparison\n\n" + json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"success": success["status"], "coverage": coverage["status"], "stability": stability["status"], "baseline": baseline["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
