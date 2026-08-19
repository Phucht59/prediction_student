"""One-shot Panel C evaluation against frozen Five-EBM-C0 / B0 / B1."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from panel_c_common import ROOT, V3, build_case_payload, features_from_row, sha256_file
from src.recommend_hybrid.v3.contracts import CanonicalAction, RiskThresholds, RouteStatus, SafetyThresholds
from src.recommend_hybrid.v3.feasibility import evaluate_action
from src.recommend_hybrid.v3.metrics import (
    evaluate_runtime_equivalent_ranking,
    grouped_bootstrap_difference_runtime_equivalent,
)
from src.recommend_hybrid.v3.pipeline import RecommendationV3Pipeline
from src.recommend_hybrid.v3.ranker import ActionStagePriorRanker, FiveEBMC0Ranker, RuleScoreRanker

PANEL = V3 / "panel_c"
REPORTS = ROOT / "reports" / "recommend_hybrid" / "v3"


def _router_thresholds() -> tuple[RiskThresholds, SafetyThresholds]:
    config = json.loads((V3 / "router" / "ROUTER_CONFIG.json").read_text(encoding="utf-8"))
    risk = RiskThresholds(
        maximum_automatic_uncertainty=float(config["risk"]["maximum_automatic_uncertainty"]),
        minimum_risk_margin=float(config["risk"]["minimum_risk_margin"]),
    )
    safety = SafetyThresholds(
        minimum_top1_score=float(config["safety"]["minimum_top1_score"]),
        minimum_top1_margin=float(config["safety"]["minimum_top1_margin"]),
        maximum_uncertainty=float(config["safety"]["maximum_uncertainty"]),
    )
    return risk, safety


def _load_reviews() -> pd.DataFrame:
    path = PANEL / "PANEL_C_REVIEWS_FROZEN.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(records)


def _load_prior() -> dict[tuple[str, str], float]:
    raw = json.loads((V3 / "ranker" / "B0_ACTION_STAGE_PRIOR.json").read_text(encoding="utf-8"))
    prior = {}
    for key, value in raw.items():
        stage, action = key.split("::", 1)
        prior[(stage, action)] = float(value)
    return prior


def exact_best_agreement(frame: pd.DataFrame) -> float | None:
    hits = []
    for _, query in frame.groupby("query_id", sort=False):
        eligible = query.loc[query.eligible.astype(bool)]
        if eligible.empty:
            continue
        rel = eligible["relevance"]
        if rel.isna().any():
            continue
        max_rel = rel.max()
        best = set(eligible.loc[rel.eq(max_rel), "action_id"].astype(str))
        if len(best) != 1:
            continue
        top = eligible.sort_values(["score", "action_id"], ascending=[False, True]).iloc[0]
        hits.append(str(top.action_id) in best)
    if not hits:
        return None
    return float(sum(hits) / len(hits))


def main() -> None:
    protocol = json.loads((PANEL / "PANEL_C_PROTOCOL.json").read_text(encoding="utf-8"))
    provider = json.loads((PANEL / "PANEL_C_PROVIDER_MANIFEST.json").read_text(encoding="utf-8"))
    if provider.get("status") != "COMPLETE":
        payload = {
            "status": "NOT_EVALUATED",
            "reason": f"Panel C provider status is {provider.get('status')}",
            "PANEL_C_AUTHENTIC_PROVENANCE": "FAIL",
        }
        (PANEL / "PANEL_C_FINAL_RESULTS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("PANEL_C_EVALUATION_BLOCKED")

    reviews = _load_reviews()
    features = pd.read_parquet(V3 / "data" / "learner_stage_features.parquet")
    cases = pd.read_parquet(PANEL / "PANEL_C_SAMPLED_CASES.parquet")
    rows = cases.merge(features, on=["query_id", "student_key", "course_key", "stage", "cutoff_day"], how="left")
    ranker = FiveEBMC0Ranker.from_artifacts(V3 / "ranker" / "final_models")
    prior = _load_prior()
    b0_ranker = ActionStagePriorRanker(prior)
    b1_ranker = RuleScoreRanker()
    risk_t, safety_t = _router_thresholds()
    pipe = RecommendationV3Pipeline(ranker, risk_t, safety_t, review_k=3)

    score_rows = []
    route_counts = {status.value: 0 for status in RouteStatus}
    invalid_emitted = 0
    issued = 0
    top1_actions = []
    for _, row in rows.iterrows():
        feats = features_from_row(row)
        case_id, payload, evaluations = build_case_payload(row)
        eligible_actions = tuple(
            CanonicalAction(item["action_id"]) for item in payload["candidate_actions"]
        )
        ebm = ranker.score(feats, eligible_actions) if eligible_actions else ()
        b0 = b0_ranker.score(feats, eligible_actions) if eligible_actions else ()
        b1 = b1_ranker.score(feats, eligible_actions) if eligible_actions else ()
        ebm_map = {item.action.value: item.score for item in ebm}
        b0_map = {item.action.value: item.score for item in b0}
        b1_map = {item.action.value: item.score for item in b1}
        decision = pipe.recommend(feats)
        route_counts[decision.route.value] = route_counts.get(decision.route.value, 0) + 1
        issued += 1
        emitted = [item.action.value for item in decision.ranked_actions]
        eligible_set = {action.value for action in eligible_actions}
        invalid_emitted += sum(action not in eligible_set for action in emitted)
        if decision.ranked_actions:
            top1_actions.append(decision.ranked_actions[0].action.value)
        for ev in evaluations:
            action = ev["action_id"]
            match = reviews.loc[reviews.case_id.eq(case_id) & reviews.action_id.eq(action)]
            if match.empty:
                relevance = None
                abstain = True if ev["eligible"] else None
            else:
                rec = match.iloc[0]
                abstain = bool(rec.abstain)
                relevance = None if abstain else rec.relevance_score
            score_rows.append(
                {
                    "query_id": str(row["query_id"]),
                    "case_id": case_id,
                    "student_key": str(row["student_key"]),
                    "stage": str(row["stage"]),
                    "action_id": action,
                    "eligible": bool(ev["eligible"]),
                    "score": ebm_map.get(action),
                    "b0_score": b0_map.get(action),
                    "b1_score": b1_map.get(action),
                    "relevance": relevance,
                    "abstain": abstain,
                    "route": decision.route.value,
                    "risk_route": decision.risk_route.value,
                }
            )

    scores = pd.DataFrame(score_rows)
    scores.to_parquet(PANEL / "PANEL_C_FINAL_SCORES.parquet", index=False)

    reviewable = scores.loc[scores.eligible.astype(bool)].copy()
    complete_case_ids = []
    for case_id, group in reviewable.groupby("case_id"):
        if group.abstain.fillna(True).any() or group.relevance.isna().any():
            continue
        complete_case_ids.append(case_id)
    primary = reviewable.loc[reviewable.case_id.isin(complete_case_ids)].copy()
    primary = primary.loc[primary.score.notna()].copy()

    ebm_eval = primary.copy()
    b0_eval = primary.copy()
    b0_eval["score"] = b0_eval["b0_score"]
    b1_eval = primary.copy()
    b1_eval["score"] = b1_eval["b1_score"]

    ebm_metrics = evaluate_runtime_equivalent_ranking(ebm_eval)
    b0_metrics = evaluate_runtime_equivalent_ranking(b0_eval)
    b1_metrics = evaluate_runtime_equivalent_ranking(b1_eval)
    bootstrap_b0 = grouped_bootstrap_difference_runtime_equivalent(
        ebm_eval, b0_eval, iterations=int(protocol["bootstrap"]["iterations"]), seed=int(protocol["bootstrap"]["seed"])
    )
    bootstrap_b1 = grouped_bootstrap_difference_runtime_equivalent(
        ebm_eval, b1_eval, iterations=int(protocol["bootstrap"]["iterations"]), seed=int(protocol["bootstrap"]["seed"])
    )
    best_baseline = max(b0_metrics.ndcg_at_3, b1_metrics.ndcg_at_3)
    if b0_metrics.ndcg_at_3 >= b1_metrics.ndcg_at_3:
        v3_minus_best = bootstrap_b0
        best_name = "B0"
    else:
        v3_minus_best = bootstrap_b1
        best_name = "B1"

    n_cases = int(rows.query_id.nunique())
    results = {
        "status": "EVALUATED",
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_authority": "Phase4 Hybrid C0",
        "recommendation_authority": "Five-EBM-C0",
        "n_students": int(rows.student_key.nunique()),
        "n_cases": n_cases,
        "n_review_records": int(len(reviews)),
        "n_primary_metric_cases": int(primary.query_id.nunique()),
        "n_excluded_incomplete_or_abstain": int(reviewable.case_id.nunique() - primary.case_id.nunique()),
        "five_ebm_c0": ebm_metrics.to_dict(),
        "baseline_b0": b0_metrics.to_dict(),
        "baseline_b1": b1_metrics.to_dict(),
        "exact_best_top1_agreement": exact_best_agreement(ebm_eval),
        "pipeline_system": {
            "n_cases": n_cases,
            "invalid_action_rate": (invalid_emitted / issued) if issued else 0.0,
            "recommendation_coverage": route_counts.get("RECOMMEND", 0) / n_cases if n_cases else 0.0,
            "HUMAN_REVIEW_rate": route_counts.get("HUMAN_REVIEW", 0) / n_cases if n_cases else 0.0,
            "INSUFFICIENT_EVIDENCE_rate": route_counts.get("INSUFFICIENT_EVIDENCE", 0) / n_cases if n_cases else 0.0,
            "NO_FEASIBLE_ACTION_rate": route_counts.get("NO_FEASIBLE_ACTION", 0) / n_cases if n_cases else 0.0,
            "route_counts": route_counts,
            "top1_action_distribution": {k: int(v) for k, v in pd.Series(top1_actions).value_counts().items()} if top1_actions else {},
            "unique_top1_actions": int(len(set(top1_actions))),
        },
        "bootstrap": {
            "v3_minus_b0": bootstrap_b0,
            "v3_minus_b1": bootstrap_b1,
            "best_baseline": best_name,
            "v3_minus_best_baseline": v3_minus_best,
        },
        "panel_b_used_for_tuning": False,
        "panel_c_used_for_tuning": False,
        "post_freeze_tuning": False,
        "invalid_action_rate_required_zero": ebm_metrics.invalid_action_rate == 0.0
        and (invalid_emitted / issued if issued else 0.0) == 0.0,
    }
    bootstrap_path = PANEL / "PANEL_C_BOOTSTRAP.json"
    bootstrap_path.write_text(json.dumps(results["bootstrap"], indent=2) + "\n", encoding="utf-8")
    (PANEL / "PANEL_C_FINAL_RESULTS.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    hashes = {
        "PANEL_C_PROTOCOL.json": sha256_file(PANEL / "PANEL_C_PROTOCOL.json"),
        "PANEL_C_CASE_MANIFEST.json": sha256_file(PANEL / "PANEL_C_CASE_MANIFEST.json"),
        "PANEL_C_PROVIDER_MANIFEST.json": sha256_file(PANEL / "PANEL_C_PROVIDER_MANIFEST.json"),
        "PANEL_C_REVIEWS_FROZEN.jsonl": sha256_file(PANEL / "PANEL_C_REVIEWS_FROZEN.jsonl"),
        "PANEL_C_FINAL_SCORES.parquet": sha256_file(PANEL / "PANEL_C_FINAL_SCORES.parquet"),
        "PANEL_C_FINAL_RESULTS.json": sha256_file(PANEL / "PANEL_C_FINAL_RESULTS.json"),
        "PANEL_C_BOOTSTRAP.json": sha256_file(bootstrap_path),
    }
    (PANEL / "checksums.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in hashes.items()) + "\n",
        encoding="utf-8",
    )

    ebm = results["five_ebm_c0"]
    report = f"""# 10 — Panel C final held-out results

**STATUS: EVALUATED**

This is the only official V3 held-out claim. Development weak-label metrics are not held-out.

| Model | cases | NDCG@3 | P@1 | MRR | R@3 | pairwise | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| Five-EBM-C0 | {ebm['query_count']} | {ebm['ndcg_at_3']:.5f} | {ebm['precision_at_1']:.5f} | {ebm['mrr']:.5f} | {ebm['recall_at_3']:.5f} | {ebm['pairwise_accuracy']:.5f} | {ebm['invalid_action_rate']} |
| B0 action+stage | {b0_metrics.query_count} | {b0_metrics.ndcg_at_3:.5f} | {b0_metrics.precision_at_1:.5f} | {b0_metrics.mrr:.5f} | {b0_metrics.recall_at_3:.5f} | {b0_metrics.pairwise_accuracy:.5f} | {b0_metrics.invalid_action_rate} |
| B1 rule score | {b1_metrics.query_count} | {b1_metrics.ndcg_at_3:.5f} | {b1_metrics.precision_at_1:.5f} | {b1_metrics.mrr:.5f} | {b1_metrics.recall_at_3:.5f} | {b1_metrics.pairwise_accuracy:.5f} | {b1_metrics.invalid_action_rate} |

Exact-best Top-1 agreement: {results['exact_best_top1_agreement']}

Bootstrap Five-EBM-C0 minus {best_name} NDCG@3: mean={v3_minus_best['mean_difference']:.5f}, 95% CI [{v3_minus_best['ci_low_95']:.5f}, {v3_minus_best['ci_high_95']:.5f}], P(diff>0)={v3_minus_best['probability_difference_positive']:.4f}, iterations={v3_minus_best['iterations']}, seed={v3_minus_best['seed']}.

Pipeline system rates: {json.dumps(results['pipeline_system'], indent=2)}

Historical Panel B is Recommendation V2 held-out evidence and was not used for V3 tuning or this evaluation.
"""
    (REPORTS / "10_PANEL_C_FINAL_RESULTS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "EVALUATED", "ndcg": ebm["ndcg_at_3"], "invalid": ebm["invalid_action_rate"]}, indent=2))


if __name__ == "__main__":
    main()
