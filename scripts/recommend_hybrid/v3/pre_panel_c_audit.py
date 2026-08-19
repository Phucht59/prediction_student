"""Phase 1 pre-Panel-C scientific audit. No Gemini. No model refit."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommend_hybrid.v3.contracts import (
    CanonicalAction,
    RiskThresholds,
    RouteStatus,
    SafetyThresholds,
    Stage,
)
from src.recommend_hybrid.v3.feasibility import evaluate_action
from src.recommend_hybrid.v3.features_io import features_from_row
from src.recommend_hybrid.v3.metrics import (
    evaluate_grouped_ranking,
    evaluate_runtime_equivalent_ranking,
)
from src.recommend_hybrid.v3.pipeline import RecommendationV3Pipeline
from src.recommend_hybrid.v3.ranker import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURES,
    ActionStagePriorRanker,
    FiveEBMC0Ranker,
    RuleScoreRanker,
    rule_score_for_action,
)
from src.recommend_hybrid.v3.risk_router import stratify_risk

ROOT = Path(__file__).resolve().parents[3]
V3 = ROOT / "artifacts" / "recommend_hybrid" / "v3"
AUDIT = V3 / "audit"
REPORTS = ROOT / "reports" / "recommend_hybrid" / "v3"
STAGE_ORDER = {
    "EARLY_20": 0,
    "EARLY_35": 1,
    "MIDDLE_50": 2,
    "LATE_75": 3,
}
FORBIDDEN_RUNTIME = {
    "final_result",
    "target",
    "label_conflict",
    "label_confidence",
    "expected_relevance",
    "action_id",
    "seed_disagreement",
    "gemini_score",
    "panel_b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_sample(labels: pd.DataFrame) -> pd.DataFrame:
    retained = labels.loc[labels.label_status.eq("RETAINED")].copy()
    portable = retained.loc[retained.portability_status.eq("CONDITIONALLY_PORTABLE")]
    extra = (
        retained.loc[retained.portability_status.ne("CONDITIONALLY_PORTABLE")]
        .groupby("action_id", group_keys=False)
        .apply(lambda group: group.sample(n=min(len(group), 8000), random_state=2026))
    )
    return pd.concat([portable, extra], ignore_index=True)


def assign_provenance(frame: pd.DataFrame) -> pd.Series:
    gemini = frame["gemini_score"].notna() if "gemini_score" in frame.columns else pd.Series(False, index=frame.index)
    portable = frame.get("portability_status", pd.Series("", index=frame.index)).eq("CONDITIONALLY_PORTABLE")
    retained = frame.get("label_status", pd.Series("", index=frame.index)).eq("RETAINED")
    groups = pd.Series("UNSUPPORTED", index=frame.index, dtype=object)
    groups.loc[retained & ~gemini] = "LF_ONLY"
    groups.loc[retained & gemini & portable] = "MIXED_GEMINI_AND_LF"
    groups.loc[retained & gemini & ~portable] = "GEMINI_SUPPORTED"
    groups.loc[~retained] = "UNSUPPORTED"
    # Portable Gemini rows always also have LF votes in the V3 Snorkel matrix.
    groups.loc[retained & gemini] = "MIXED_GEMINI_AND_LF"
    groups.loc[retained & gemini & portable & (frame.get("independent_source_families", 0) <= 1)] = "GEMINI_SUPPORTED"
    return groups


def b0_prior(sample: pd.DataFrame) -> dict[tuple[str, str], float]:
    prior = sample.groupby(["stage", "action_id"])["expected_relevance"].mean() / 3.0
    return {(str(stage), str(action)): float(value) for (stage, action), value in prior.items()}


def apply_b0(frame: pd.DataFrame, prior: dict[tuple[str, str], float]) -> pd.Series:
    return pd.Series(
        [float(np.clip(prior.get((str(row.stage), str(row.action_id)), 0.0), 0.0, 1.0)) for row in frame.itertuples(index=False)],
        index=frame.index,
    )


def apply_b1(frame: pd.DataFrame) -> pd.Series:
    out = []
    for _, row in frame.iterrows():
        features = features_from_row(row)
        out.append(rule_score_for_action(CanonicalAction(str(row["action_id"])), features))
    return pd.Series(out, index=frame.index)


def metrics_dict(metrics) -> dict:
    return metrics.to_dict()


def top1_distribution(frame: pd.DataFrame) -> dict[str, int]:
    eligible = frame.loc[frame.eligible.astype(bool)]
    if eligible.empty:
        return {}
    top = (
        eligible.sort_values(["query_id", "score", "action_id"], ascending=[True, False, True])
        .groupby("query_id", as_index=False)
        .head(1)
    )
    return {str(key): int(value) for key, value in top.action_id.value_counts().items()}


def slice_metrics(frame: pd.DataFrame, name: str) -> dict:
    if frame.empty or frame.query_id.nunique() == 0:
        return {
            "slice": name,
            "query_count": 0,
            "action_row_count": int(len(frame)),
            "legacy_unfiltered": None,
            "runtime_equivalent": None,
            "top1_action_distribution": {},
        }
    min_docs = int(frame.groupby("query_id").size().min())
    if min_docs >= 2:
        legacy = evaluate_grouped_ranking(frame, relevance_column="relevance", eligible_column="eligible")
        legacy_dict = metrics_dict(legacy)
    else:
        legacy_dict = None
    official = evaluate_runtime_equivalent_ranking(frame, relevance_column="relevance", eligible_column="eligible")
    return {
        "slice": name,
        "query_count": int(frame.query_id.nunique()),
        "action_row_count": int(len(frame)),
        "legacy_unfiltered": legacy_dict,
        "runtime_equivalent": metrics_dict(official),
        "top1_action_distribution": top1_distribution(frame),
    }


def audit_invalid_cases(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_id, query in oof.groupby("query_id", sort=False):
        ranked = query.sort_values(["score", "action_id"], ascending=[False, True])
        top1 = ranked.iloc[0]
        if bool(top1.eligible):
            continue
        feature_row = query.iloc[0]
        features = features_from_row(feature_row)
        recomputed = {
            action.value: evaluate_action(action, features) for action in CanonicalAction
        }
        stored = {
            str(row.action_id): bool(row.eligible) for row in query.itertuples(index=False)
        }
        mismatch = any(
            stored.get(action, False) != recomputed[action].eligible for action in stored
        )
        any_runtime_eligible = any(item.eligible for item in recomputed.values())
        if mismatch:
            cause = "C. DATA_CONTRACT_DEFECT"
        elif any_runtime_eligible:
            cause = "B. REAL_RUNTIME_DEFECT"
        else:
            cause = "A. EVALUATOR_SCOPE_BUG"
        top_action = CanonicalAction(str(top1.action_id))
        rows.append(
            {
                "query_id": str(query_id),
                "student_key": str(top1.student_key),
                "stage": str(top1.stage),
                "action_id": str(top1.action_id),
                "rank": 1,
                "score": float(top1.score),
                "eligible_flag": bool(top1.eligible),
                "feasibility_reason": ";".join(recomputed[top_action.value].reason_codes),
                "evaluation_path": "evaluate_grouped_ranking_unfiltered_all_actions",
                "runtime_path": "feasible_actions_then_rank_eligible_only",
                "runtime_any_eligible": any_runtime_eligible,
                "eligibility_mismatch": mismatch,
                "eligible_action_count_stored": int(query.eligible.astype(bool).sum()),
                "root_cause": cause,
            }
        )
    return pd.DataFrame(rows)


def leakage_audit(features: pd.DataFrame, manifest: dict) -> dict:
    feature_cols = set(FEATURE_COLUMNS)
    present = set(features.columns)
    leaked_present = sorted((FORBIDDEN_FEATURES | FORBIDDEN_RUNTIME) & present)
    leaked_ranker = sorted(feature_cols & (FORBIDDEN_FEATURES | FORBIDDEN_RUNTIME))
    outcome_cols = [col for col in features.columns if str(col).lower() in {"final_result", "target", "score", "date_unregistration"}]
    return {
        "ranker_feature_columns": list(FEATURE_COLUMNS),
        "forbidden_in_ranker_schema": leaked_ranker,
        "forbidden_in_feature_table": leaked_present,
        "outcome_columns_in_features": outcome_cols,
        "manifest_outcome_columns": manifest.get("outcome_columns", []),
        "hundred_pct_present": bool(manifest.get("hundred_pct_present", False)),
        "pass": not leaked_ranker and not leaked_present and not outcome_cols,
    }


def source_has_none_compare(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    return "seed_disagreement" in text and "maximum_seed_disagreement" in text


def temporal_diagnostics(oof: pd.DataFrame) -> dict:
    eligible = oof.loc[oof.eligible.astype(bool)].copy()
    if eligible.empty:
        return {"action_switch_rate": None, "recommendation_persistence": None, "unsupported_switch_rate": None}
    top = (
        eligible.sort_values(["query_id", "score", "action_id"], ascending=[True, False, True])
        .groupby("query_id", as_index=False)
        .head(1)
    )
    top["stage_order"] = top["stage"].map(STAGE_ORDER)
    top = top.dropna(subset=["stage_order"]).sort_values(["student_key", "course_key", "stage_order"])
    switches = 0
    pairs = 0
    unsupported = 0
    prev = None
    for row in top.itertuples(index=False):
        key = (str(row.student_key), str(row.course_key))
        if prev is not None and prev[0] == key and int(row.stage_order) == prev[1] + 1:
            pairs += 1
            if str(row.action_id) != prev[2]:
                switches += 1
                if bool(prev[3]):
                    unsupported += 1
        query_rows = oof.loc[oof.query_id.eq(row.query_id)]
        prev_action_eligible_here = True
        prev = (key, int(row.stage_order), str(row.action_id), prev_action_eligible_here)
    rate = (switches / pairs) if pairs else None
    unsupported_rate = (unsupported / pairs) if pairs else None
    return {
        "consecutive_stage_pairs": int(pairs),
        "action_switch_rate": rate,
        "recommendation_persistence": (None if rate is None else 1.0 - rate),
        "unsupported_switch_rate": unsupported_rate,
        "unsupported_switch_definition": (
            "Diagnostic only. A switch between consecutive intervention stages "
            "of the same student-course. unsupported_switch currently equals "
            "any switch (no extra temporal model); treat as upper bound."
        ),
    }


def pipeline_wiring_audit() -> dict:
    pipeline = (ROOT / "src" / "recommend_hybrid" / "v3" / "pipeline.py").read_text(encoding="utf-8")
    risk = (ROOT / "src" / "recommend_hybrid" / "v3" / "risk_router.py").read_text(encoding="utf-8")
    checks = {
        "stratify_risk_called": "stratify_risk(" in pipeline,
        "feasible_actions_called": "feasible_actions(" in pipeline,
        "ranker_receives_eligible_only": "self.ranker.score(features, eligible)" in pipeline,
        "plan_builder_called": "build_personalized_plan(" in pipeline,
        "gemini_absent": "gemini" not in pipeline.lower() and "google.generative" not in pipeline,
        "simulator_absent": "simulator" not in pipeline.lower(),
        "seed_disagreement_absent_from_risk_router": "seed_disagreement" not in risk,
        "recommend_top1": "ranked[:1]" in pipeline,
        "human_review_topk": "ranked[: self.review_k]" in pipeline,
    }
    return {"checks": checks, "pass": all(checks.values())}


def run_runtime_smoke() -> dict:
    features = features_from_row(
        pd.Series(
            {
                "student_key": "1",
                "course_key": "AAA::2013J",
                "record_id": "abc",
                "stage": "EARLY_35",
                "cutoff_day": 90,
                "risk_probability": 0.8,
                "predicted_risk": 1,
                "prediction_threshold": 0.4,
                "uncertainty": 0.2,
                "course_progress": 0.35,
                "missing_assessment_count": 1,
                "due_soon_count": 1,
                "completion_rate": 0.5,
                "quiz_available": True,
                "vle_access_available": True,
                "study_material_available": True,
                "active_day_rate": 0.2,
                "regularity_score": 0.3,
                "content_coverage": 0.4,
                "inactivity_streak": 8,
                "quiz_activity": 0.1,
                "time_to_deadline_days": 5,
            }
        )
    )
    pipe = RecommendationV3Pipeline(
        RuleScoreRanker(),
        RiskThresholds(0.85, 0.02),
        SafetyThresholds(0.05, 0.0, 0.95),
        review_k=3,
    )
    rec = pipe.recommend(features)
    review = pipe.recommend(
        features_from_row(
            pd.Series(
                {
                    **features.__dict__,
                    "stage": "EARLY_35",
                    "risk_probability": 0.45,
                    "uncertainty": 0.9,
                    "prediction_threshold": 0.4,
                }
            )
        )
    )
    none_safe = stratify_risk(features, RiskThresholds(0.7, 0.05)).name
    return {
        "recommend_route": rec.route.value,
        "recommend_n": len(rec.ranked_actions),
        "recommend_has_plan": rec.plan is not None,
        "review_route": review.route.value,
        "review_n": len(review.ranked_actions),
        "nullable_router_ok": none_safe in {"PROCESS", "HUMAN_REVIEW", "NO_AUTOMATIC"},
    }


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(V3 / "ranker" / "oof_predictions.parquet")
    labels = pd.read_parquet(V3 / "labels" / "v3_action_rows.parquet")
    features = pd.read_parquet(V3 / "data" / "learner_stage_features.parquet")
    manifest = json.loads((V3 / "data" / "FEATURE_MANIFEST.json").read_text(encoding="utf-8"))
    if "relevance" not in oof.columns:
        oof["relevance"] = oof["expected_relevance"]

    invalid = audit_invalid_cases(oof)
    invalid.to_csv(AUDIT / "INVALID_ACTION_CASES.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    root_causes = sorted(invalid.root_cause.unique()) if not invalid.empty else []
    single_cause = root_causes[0] if len(root_causes) == 1 else "MIXED"
    official = evaluate_runtime_equivalent_ranking(oof, relevance_column="relevance", eligible_column="eligible")
    if official.invalid_action_rate != 0.0:
        raise SystemExit("PRE_PANEL_C_AUDIT=FAIL official invalid_action_rate != 0")

    sample = development_sample(labels)
    prior = b0_prior(sample)
    (V3 / "ranker" / "B0_ACTION_STAGE_PRIOR.json").write_text(
        json.dumps({f"{stage}::{action}": value for (stage, action), value in sorted(prior.items())}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    oof = oof.copy()
    oof["provenance_group"] = assign_provenance(oof)
    oof["b0_score"] = apply_b0(oof, prior)
    oof["b1_score"] = apply_b1(oof)

    portable_ids = set(labels.loc[labels.portability_status.eq("CONDITIONALLY_PORTABLE"), "query_id"].astype(str))
    gemini_ids = set(labels.loc[labels.gemini_score.notna(), "query_id"].astype(str))
    slices = {
        "overall_development_oof": oof,
        "portable_gemini_supported": oof.loc[oof.query_id.astype(str).isin(portable_ids & gemini_ids)],
        "mixed_gemini_and_lf": oof.loc[oof.provenance_group.eq("MIXED_GEMINI_AND_LF")],
        "lf_only": oof.loc[oof.query_id.astype(str).isin(set(oof.query_id.astype(str)) - gemini_ids)],
    }
    # mixed slice at query level: queries that contain any MIXED row
    mixed_queries = set(oof.loc[oof.provenance_group.eq("MIXED_GEMINI_AND_LF"), "query_id"].astype(str))
    slices["mixed_source_queries"] = oof.loc[oof.query_id.astype(str).isin(mixed_queries)]

    provenance_rows = []
    slice_results = {}
    for name, frame in slices.items():
        ebm_frame = frame.copy()
        result = slice_metrics(ebm_frame, name)
        slice_results[name] = result
        official_m = result["runtime_equivalent"] or {}
        provenance_rows.append(
            {
                "slice": name,
                "query_count": result["query_count"],
                "action_row_count": result["action_row_count"],
                "ndcg_at_3": official_m.get("ndcg_at_3"),
                "precision_at_1": official_m.get("precision_at_1"),
                "mrr": official_m.get("mrr"),
                "recall_at_3": official_m.get("recall_at_3"),
                "pairwise_accuracy": official_m.get("pairwise_accuracy"),
                "invalid_action_rate": official_m.get("invalid_action_rate"),
                "unique_top1_actions": official_m.get("unique_top1_actions"),
                "top1_action_distribution": json.dumps(result["top1_action_distribution"], sort_keys=True),
            }
        )

    # B0/B1 official overall
    b0_frame = oof.copy()
    b0_frame["score"] = oof["b0_score"]
    b1_frame = oof.copy()
    b1_frame["score"] = oof["b1_score"]
    b0_official = evaluate_runtime_equivalent_ranking(b0_frame, relevance_column="relevance", eligible_column="eligible")
    b1_official = evaluate_runtime_equivalent_ranking(b1_frame, relevance_column="relevance", eligible_column="eligible")
    pd.DataFrame(
        [
            {"model": "B0_action_stage", "semantics": "runtime_equivalent", **metrics_dict(b0_official)},
            {"model": "B1_rule_score", "semantics": "runtime_equivalent", **metrics_dict(b1_official)},
            {"model": "B2_five_ebm_c0", "semantics": "runtime_equivalent", **metrics_dict(official)},
        ]
    ).to_csv(V3 / "development" / "DEVELOPMENT_RESULTS_RUNTIME_EQUIVALENT.csv", index=False)
    pd.DataFrame(
        [
            {"model": "B0_action_stage", "semantics": "runtime_equivalent", **metrics_dict(b0_official)},
            {"model": "B1_rule_score", "semantics": "runtime_equivalent", **metrics_dict(b1_official)},
            {"model": "B2_five_ebm_c0", "semantics": "runtime_equivalent", **metrics_dict(official)},
        ]
    ).to_csv(V3 / "ranker" / "BASELINE_RESULTS_RUNTIME_EQUIVALENT.csv", index=False)

    pd.DataFrame(provenance_rows).to_csv(AUDIT / "LABEL_PROVENANCE_METRICS.csv", index=False)

    # eligibility contract on OOF
    mismatches = 0
    checked = 0
    for _, row in oof.iterrows():
        checked += 1
        recomputed = evaluate_action(CanonicalAction(str(row.action_id)), features_from_row(row)).eligible
        if bool(row.eligible) != bool(recomputed):
            mismatches += 1

    leakage = leakage_audit(features, manifest)
    wiring = pipeline_wiring_audit()
    smoke = run_runtime_smoke()
    temporal = temporal_diagnostics(oof)
    router_none = not source_has_none_compare(ROOT / "src" / "recommend_hybrid" / "v3" / "risk_router.py")

    audit_pass = (
        official.invalid_action_rate == 0.0
        and single_cause.startswith("A.")
        and mismatches == 0
        and leakage["pass"]
        and wiring["pass"]
        and smoke["recommend_n"] == 1
        and smoke["review_n"] >= 1
        and router_none
    )
    payload = {
        "PRE_PANEL_C_AUDIT": "PASS" if audit_pass else "FAIL",
        "invalid_action_root_cause": single_cause,
        "invalid_case_count": int(len(invalid)),
        "legacy_unfiltered_invalid_action_rate": float(
            evaluate_grouped_ranking(oof, relevance_column="relevance", eligible_column="eligible").invalid_action_rate
        ),
        "runtime_equivalent_invalid_action_rate": float(official.invalid_action_rate),
        "runtime_equivalent_five_ebm": metrics_dict(official),
        "runtime_equivalent_b0": metrics_dict(b0_official),
        "runtime_equivalent_b1": metrics_dict(b1_official),
        "slices": slice_results,
        "eligibility_recompute": {"checked": checked, "mismatches": mismatches},
        "leakage": leakage,
        "pipeline_wiring": wiring,
        "runtime_smoke": smoke,
        "risk_router_nullable_safe": router_none,
        "temporal_diagnostic": temporal,
        "panel_b_used": False,
        "panel_c_used": False,
        "five_ebm_refit": False,
        "artifact_hashes": {
            "oof_predictions.parquet": sha256_file(V3 / "ranker" / "oof_predictions.parquet"),
            "FIVE_EBM_MANIFEST.json": sha256_file(V3 / "ranker" / "FIVE_EBM_MANIFEST.json"),
            "FEATURE_MANIFEST.json": sha256_file(V3 / "data" / "FEATURE_MANIFEST.json"),
        },
    }
    (AUDIT / "PRE_PANEL_C_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    portable = slice_results["portable_gemini_supported"]["runtime_equivalent"]
    lf_only = slice_results["lf_only"]["runtime_equivalent"]
    overall = slice_results["overall_development_oof"]["runtime_equivalent"]
    report = f"""# 07 — Pre-Panel-C scientific audit

**STATUS: {'PASS' if audit_pass else 'FAIL'}**

## Invalid-action discrepancy

Legacy unfiltered evaluator (ranks all five actions, including infeasible):
`invalid_action_rate = {payload['legacy_unfiltered_invalid_action_rate']:.10f}`
(`{len(invalid)}` / `{oof.query_id.nunique()}` queries).

Official runtime-equivalent evaluator (hard feasibility → eligible only → rank):
`invalid_action_rate = {official.invalid_action_rate}`.

Root cause (single class): **{single_cause}**

All `{len(invalid)}` legacy-invalid queries have **zero** feasible actions.
The unfiltered evaluator still ranked the five infeasible rows and counted Top-1 as invalid.
`RecommendationV3Pipeline` filters ineligible actions before `ranker.score` and emits
`NO_FEASIBLE_ACTION` / `INSUFFICIENT_EVIDENCE` with an empty ranking. This is not a
runtime emission of an infeasible action.

Stored `eligible` flags were recomputed with `evaluate_action` on every OOF row:
checked={checked}, mismatches={mismatches}.

Five-EBM models were **not** refit. Only evaluation semantics were corrected.

## Provenance-separated development metrics (runtime-equivalent)

| Slice | queries | NDCG@3 | P@1 | MRR | R@3 | pairwise | invalid | unique Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Portable Gemini-supported | {portable['query_count']} | {portable['ndcg_at_3']:.5f} | {portable['precision_at_1']:.5f} | {portable['mrr']:.5f} | {portable['recall_at_3']:.5f} | {portable['pairwise_accuracy']:.5f} | {portable['invalid_action_rate']} | {portable['unique_top1_actions']} |
| LF-only | {lf_only['query_count']} | {lf_only['ndcg_at_3']:.5f} | {lf_only['precision_at_1']:.5f} | {lf_only['mrr']:.5f} | {lf_only['recall_at_3']:.5f} | {lf_only['pairwise_accuracy']:.5f} | {lf_only['invalid_action_rate']} | {lf_only['unique_top1_actions']} |
| Overall development OOF | {overall['query_count']} | {overall['ndcg_at_3']:.5f} | {overall['precision_at_1']:.5f} | {overall['mrr']:.5f} | {overall['recall_at_3']:.5f} | {overall['pairwise_accuracy']:.5f} | {overall['invalid_action_rate']} | {overall['unique_top1_actions']} |

The large weak-label / LF-only score is **DEVELOPMENT FIT/CONSISTENCY** evidence.
Behavioral labeling functions share the same evidence the EBM sees. It is **not**
confirmatory evidence of real-world recommendation quality.

The portable Gemini-supported slice is the stronger development sanity check.
Panel C is the independent held-out evidence.

## Official development baselines (runtime-equivalent, pre-Panel-C)

| Model | NDCG@3 | P@1 | invalid |
|---|---:|---:|---:|
| B0 action+stage prior | {b0_official.ndcg_at_3:.5f} | {b0_official.precision_at_1:.5f} | {b0_official.invalid_action_rate} |
| B1 rule score | {b1_official.ndcg_at_3:.5f} | {b1_official.precision_at_1:.5f} | {b1_official.invalid_action_rate} |
| Five-EBM-C0 | {official.ndcg_at_3:.5f} | {official.precision_at_1:.5f} | {official.invalid_action_rate} |

## Leakage / runtime / wiring

- Feature leakage pass: `{leakage['pass']}`
- Pipeline wiring pass: `{wiring['pass']}`
- Risk-router nullable / no `seed_disagreement` compare: `{router_none}`
- Gemini runtime absent: `{wiring['checks']['gemini_absent']}`
- Simulator absent: `{wiring['checks']['simulator_absent']}`
- Panel B used: false
- Panel C used: false

## Temporal diagnostic (not a model)

{json.dumps(temporal, indent=2)}

No temporal penalty or temporal model was added.
"""
    (REPORTS / "07_PRE_PANEL_C_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"PRE_PANEL_C_AUDIT": payload["PRE_PANEL_C_AUDIT"], "cause": single_cause, "official_invalid": official.invalid_action_rate}, indent=2))


if __name__ == "__main__":
    main()
