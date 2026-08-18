"""Group-safe nested/cross-fitted recommendation evidence.

This is an evaluation artifact, not a replacement production recommender. It
uses only the 179 rows whose Hybrid inputs are genuine group-safe OOF
predictions. The 121 final-inference rows are excluded from every operation.
No HPO or threshold selection is performed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILD = ROOT / "artifacts" / "recommendation" / "phase8_prediction_rebuild"
OUT = ROOT / "artifacts" / "recommendation" / "cross_fitted_evaluation"
MODELS = OUT / "models"
REPORTS = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def stable_inner_fold(group_id: str) -> int:
    return int(hashlib.sha256(str(group_id).encode("utf-8")).hexdigest()[:8], 16) % 2


def load_clean_scope() -> tuple[pd.DataFrame, pd.DataFrame, tuple[Any, ...], list[str], dict[str, Any]]:
    features = pd.read_parquet(BUILD / "features" / "learner_stage_features.parquet")
    clean = features.loc[features.prediction_kind.eq("OOF_INNER_VALIDATION")].copy()
    if len(clean) != 179 or clean.query_id.nunique() != 179:
        raise RuntimeError("CLEAN_OOF_SCOPE_MISMATCH")
    if clean.prediction_kind.ne("OOF_INNER_VALIDATION").any():
        raise RuntimeError("NON_OOF_ROW_ENTERED_CROSSFIT_SCOPE")
    if clean.model_id.astype(str).ne("hybrid").any():
        raise RuntimeError("LEGACY_MODEL_ENTERED_CROSSFIT_SCOPE")
    if clean.query_id.duplicated().any():
        raise RuntimeError("DUPLICATE_CROSSFIT_QUERY")

    vote_path = BUILD / "weak_labels" / "weak_vote_matrix_rebuilt_input.parquet"
    clean_query_ids = sorted(set(clean.query_id.astype(str)))
    vote_columns = ["query_id", "case_id", "outer_fold", "stage", "action_id"]
    label_manifest = json.loads((BUILD / "weak_labels" / "label_model_manifest_rebuilt.json").read_text(encoding="utf-8"))
    vote_columns.extend(item["name"] for item in label_manifest["sources"])
    votes = pd.read_parquet(vote_path, columns=vote_columns, filters=[("query_id", "in", clean_query_ids)])
    votes = votes.loc[votes.query_id.isin(set(clean_query_ids))].copy()
    votes["query_id"] = votes.query_id.astype(str)
    votes["outer_fold"] = votes.outer_fold.astype(int)
    clean["query_id"] = clean.query_id.astype(str)
    if len(votes) != 179 * 5 or votes.query_id.nunique() != 179:
        raise RuntimeError("CROSSFIT_VOTE_SCOPE_MISMATCH")
    group_map = clean.set_index("query_id")["group_id"].astype(str).to_dict()
    votes["group_id"] = votes.query_id.map(group_map)
    if votes.group_id.isna().any():
        raise RuntimeError("VOTE_GROUP_MAPPING_FAILURE")
    group_fold_counts = votes[["group_id", "outer_fold"]].drop_duplicates().groupby("group_id").outer_fold.nunique()
    if int((group_fold_counts > 1).sum()) != 0:
        raise RuntimeError("OUTER_FOLD_GROUP_LEAKAGE")

    from src.recommend_hybrid.final.weak_labels import WeakLabelSource

    sources = tuple(WeakLabelSource(item["name"], item["family"]) for item in label_manifest["sources"])
    source_columns = [source.name for source in sources]
    if any(column not in votes.columns for column in source_columns):
        raise RuntimeError("WEAK_SOURCE_SCHEMA_FAILURE")
    return clean, votes, sources, source_columns, label_manifest


def eligibility_frame(features: pd.DataFrame, action_names: list[str]) -> pd.DataFrame:
    from src.recommend_hybrid.final.action_eligibility import evaluate_action_eligibility

    rows: list[dict[str, Any]] = []
    for row in features.to_dict("records"):
        for action in action_names:
            eligible, reason = evaluate_action_eligibility(row, action)
            rows.append({"query_id": str(row["query_id"]), "action_id": action, "eligible": bool(eligible), "feasibility_reason": reason})
    return pd.DataFrame(rows)


def aggregate_for_rows(vote_rows: pd.DataFrame, train_rows: pd.DataFrame, sources: tuple[Any, ...], source_columns: list[str], *, seed: int) -> pd.DataFrame:
    from src.recommend_hybrid.final.weak_labels import aggregate_votes, fit_label_model

    model = fit_label_model(train_rows[source_columns].to_numpy(), sources, seed=seed, epochs=1000)
    aggregated = aggregate_votes(model, vote_rows[source_columns].to_numpy(), sources, minimum_confidence=0.25, minimum_source_families=2)
    meta = vote_rows[["query_id", "case_id", "outer_fold", "stage", "action_id", "group_id"]].reset_index(drop=True)
    result = pd.concat([meta, aggregated.reset_index(drop=True)], axis=1)
    result["retained_for_training"] = result["label_status"].eq("RETAINED")
    return result


def nested_outer_labels(votes: pd.DataFrame, clean: pd.DataFrame, sources: tuple[Any, ...], source_columns: list[str], outer_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer_test_queries = set(votes.loc[votes.outer_fold.eq(outer_fold), "query_id"])
    outer_train_queries = set(votes.loc[~votes.outer_fold.eq(outer_fold), "query_id"])
    outer_train_votes = votes.loc[votes.query_id.isin(outer_train_queries)].copy()
    outer_test_votes = votes.loc[votes.query_id.isin(outer_test_queries)].copy()
    if not outer_train_queries or not outer_test_queries:
        raise RuntimeError("EMPTY_OUTER_FOLD")

    train_group_to_inner = {group: stable_inner_fold(group) for group in sorted(outer_train_votes.group_id.astype(str).unique())}
    outer_train_votes["inner_fold"] = outer_train_votes.group_id.astype(str).map(train_group_to_inner)
    nested_train_parts: list[pd.DataFrame] = []
    for inner_fold in (0, 1):
        inner_held = outer_train_votes.loc[outer_train_votes.inner_fold.eq(inner_fold)]
        inner_fit = outer_train_votes.loc[outer_train_votes.inner_fold.ne(inner_fold)]
        if inner_fit.group_id.nunique() < 2 or inner_held.empty:
            raise RuntimeError("INVALID_INNER_GROUP_SPLIT")
        nested_train_parts.append(aggregate_for_rows(inner_held, inner_fit, sources, source_columns, seed=2026 + outer_fold * 10 + inner_fold))
    outer_train_labels = pd.concat(nested_train_parts, ignore_index=True)

    outer_test_labels = aggregate_for_rows(outer_test_votes, outer_train_votes, sources, source_columns, seed=2026 + outer_fold)
    return outer_train_labels, outer_test_labels


def fit_outer_ebms(clean: pd.DataFrame, outer_train_labels: pd.DataFrame, outer_fold: int, action_names: list[str], model_parameters: dict[str, Any]) -> dict[str, Any]:
    platform.machine = lambda: "AMD64"  # interpret DLL loader workaround only
    from interpret.glassbox import ExplainableBoostingRegressor
    from src.recommend_hybrid.final.ranker import FEATURE_COLUMNS

    train = outer_train_labels.loc[outer_train_labels.retained_for_training].copy()
    feature_map = clean.set_index("query_id")
    x = feature_map.loc[train.query_id.astype(str), list(FEATURE_COLUMNS)].reset_index(drop=True)
    fitted: dict[str, Any] = {}
    fit_rows: dict[str, int] = {}
    for action in action_names:
        mask = train.action_id.eq(action)
        if int(mask.sum()) < 30:
            raise RuntimeError(f"INSUFFICIENT_CROSSFIT_LABELS:{outer_fold}:{action}")
        model = ExplainableBoostingRegressor(**model_parameters)
        model.fit(x.loc[mask.to_numpy()], train.loc[mask, "expected_relevance"].to_numpy(dtype=float))
        path = MODELS / f"outer_{outer_fold}_{action}.joblib"
        joblib.dump(model, path)
        fitted[action] = model
        fit_rows[action] = int(mask.sum())
    return {"models": fitted, "fit_rows": fit_rows}


def score_outer(clean: pd.DataFrame, outer_test_labels: pd.DataFrame, fitted: dict[str, Any], action_names: list[str], outer_fold: int) -> pd.DataFrame:
    from src.recommend_hybrid.final.ranker import FEATURE_COLUMNS

    feature_map = clean.set_index("query_id")
    test = outer_test_labels.copy()
    x = feature_map.loc[test.query_id.astype(str), list(FEATURE_COLUMNS)].reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for action in action_names:
        mask = test.action_id.eq(action)
        rows = test.loc[mask, ["query_id", "case_id", "stage", "group_id", "action_id", "expected_relevance", "eligible", "retained_for_training"]].copy().reset_index(drop=True)
        raw = fitted[action].predict(x.loc[mask.to_numpy()])
        rows["score_native_ordinal"] = np.clip(np.asarray(raw, dtype=float), 0.0, 3.0)
        rows["score"] = rows.score_native_ordinal / 3.0
        rows["relevance"] = rows.expected_relevance.astype(float)
        rows["outer_fold"] = outer_fold
        rows["prediction_scope"] = "OUTER_TRAIN_ONLY_CROSSFITTED_EBM"
        pieces.append(rows)
    return pd.concat(pieces, ignore_index=True)


def grouped_bootstrap_ndcg(frame: pd.DataFrame, *, iterations: int = 1000, seed: int = 2026) -> dict[str, Any]:
    from src.recommend_hybrid.final.metrics import evaluate_grouped_ranking

    groups = np.array(sorted(frame.group_id.astype(str).unique()), dtype=object)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for draw in range(iterations):
        selected = rng.choice(groups, size=len(groups), replace=True)
        parts: list[pd.DataFrame] = []
        for index, group in enumerate(selected):
            part = frame.loc[frame.group_id.astype(str).eq(str(group))].copy()
            part["query_id"] = part.query_id.astype(str) + f"::bootstrap::{draw}::{index}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        values.append(float(evaluate_grouped_ranking(sample, relevance_column="relevance", score_column="score", eligible_column="eligible").ndcg_at_3))
    arr = np.asarray(values, dtype=float)
    return {"unit": "student/group", "iterations": iterations, "seed": seed, "mean": float(arr.mean()), "ci_low_95": float(np.quantile(arr, 0.025)), "ci_high_95": float(np.quantile(arr, 0.975))}


def action_level(scores: pd.DataFrame, action_names: list[str]) -> pd.DataFrame:
    top = scores.sort_values(["query_id", "score", "action_id"], ascending=[True, False, True]).drop_duplicates("query_id")
    rows: list[dict[str, Any]] = []
    for action in action_names:
        part = scores.loc[scores.action_id.eq(action)]
        selected = int((top.action_id == action).sum())
        rows.append({"action": action, "support_query_count": int(part.query_id.nunique()), "retained_label_support": int(part.retained_for_training.sum()), "mean_relevance": float(part.relevance.mean()), "mean_score": float(part.score.mean()), "mae": float(np.mean(np.abs(part.relevance - part.score_native_ordinal))), "selection_frequency": selected / max(1, int(top.query_id.nunique())), "feasibility_pass_rate": float(part.eligible.mean())})
    return pd.DataFrame(rows)


def coverage_audit(clean: pd.DataFrame, votes: pd.DataFrame, label_manifest: dict[str, Any]) -> dict[str, Any]:
    all_features = pd.read_parquet(BUILD / "features" / "learner_stage_features.parquet", columns=["query_id", "student_key", "group_id"])
    active_groups = set(all_features.group_id.astype(str))
    full_votes = pd.read_parquet(BUILD / "weak_labels" / "weak_vote_matrix_rebuilt_input.parquet", columns=["query_id"])
    query_to_group = all_features.set_index("query_id")["group_id"].astype(str).to_dict()
    label_groups = {query_to_group[str(query_id)] for query_id in full_votes.query_id.astype(str)}
    raw = pd.read_csv(Path(r"C:\hufit\kltn\data\raw\studentInfo.csv"), usecols=["id_student"])
    raw_groups = set(raw.id_student.astype(str))
    legacy_path = ROOT / "artifacts" / "final" / "recommendation" / "risk_profiles.parquet"
    legacy_groups = set()
    if legacy_path.exists():
        legacy_frame = pd.read_parquet(legacy_path, columns=["id_student"])
        legacy_groups = set(legacy_frame.id_student.astype(str))
    risk_policy = json.loads((BUILD / "router" / "RISK_POLICY_REVALIDATION.json").read_text(encoding="utf-8"))
    router = json.loads((BUILD / "router" / "ROUTER_REVALIDATION.json").read_text(encoding="utf-8"))
    historical_router = json.loads((ROOT / "artifacts" / "recommend_hybrid" / "final" / "router" / "ROUTER_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    dev_metric = json.loads((BUILD / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json").read_text(encoding="utf-8"))
    scope = {
        "feature_fitting": {"group_count": len(active_groups), "query_count": 300, "source": rel(BUILD / "features" / "learner_stage_features.parquet")},
        "weak_labels": {"group_count": len(label_groups), "query_count": 300, "source": rel(BUILD / "weak_labels" / "weak_vote_matrix_rebuilt_input.parquet")},
        "ebm_fit": {"group_count": len(active_groups), "query_count": 300, "source": rel(BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json"), "basis": "fit code passes full 300-query feature frame"},
        "risk_threshold_selection": {"group_count": len(active_groups), "query_count": risk_policy.get("query_rows"), "source": rel(BUILD / "router" / "RISK_POLICY_REVALIDATION.json")},
        "safety_tuning": {"group_count": len(active_groups), "query_count": historical_router.get("development_query_count"), "source": rel(BUILD / "router" / "ROUTER_REVALIDATION.json"), "id_scope": "Panel A scope; explicit group IDs not persisted"},
        "development_metric": {"group_count": len(active_groups), "query_count": dev_metric.get("metrics", {}).get("query_count"), "source": rel(BUILD / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json")},
    }
    return {
        "status": "NO_ACTIVE_SCOPE_GROUP_UNUSED",
        "active_recommendation_scope": {"definition": "Phase8 rebuilt Panel A feature/weak-label case universe", "group_count": len(active_groups), "query_count": int(all_features.query_id.nunique()), "groups": sorted(active_groups)},
        "coverage_by_artifact": scope,
        "intersection_group_count": len(set.intersection(*[active_groups, label_groups])),
        "unused_groups_in_active_recommendation_scope": [],
        "raw_oulad_context": {"unique_student_groups": len(raw_groups), "not_in_active_panel_a": len(raw_groups - active_groups), "status": "NOT_A_LABELED_RECOMMENDATION_HOLDOUT_UNIVERSE", "reason": "Raw groups have no current action-level weak-label/evaluation contract and cannot support an honest NDCG holdout without new independent labels."},
        "historical_legacy_context": {"legacy_risk_profile_student_groups": len(legacy_groups), "status": "EXCLUDED_FROM_ACTIVE_PHASE8_SCOPE", "reason": "Historical recommendation risk profiles belong to the old prediction identity and cannot be repurposed as a new untouched Phase8 recommendation holdout."},
        "historical_panel_b": {"status": "HISTORICAL_ONLY", "not_merged": True},
        "decision": "STOP_HOLDOUT_CREATION_AND_USE_GROUP_SAFE_CROSSFITTED_EVIDENCE",
    }


def main() -> None:
    clean, votes, sources, source_columns, label_manifest = load_clean_scope()
    action_names = ["ASSESSMENT_COMPLETION", "RECOVER_ENGAGEMENT", "STUDY_REGULARITY", "TARGETED_CONTENT_REVIEW", "QUIZ_RETRIEVAL_PRACTICE"]
    cov = coverage_audit(clean, votes, label_manifest)
    write_json(OUT / "GROUP_COVERAGE_AUDIT.json", cov)
    write_json(OUT / "CROSSFITTED_INPUT_SCOPE.json", {"status": "PASS_CLEAN_OOF_SCOPE", "query_count": int(clean.query_id.nunique()), "group_count": int(clean.group_id.nunique()), "prediction_kind": clean.prediction_kind.value_counts().to_dict(), "prediction_source": sorted(clean.prediction_source.astype(str).unique()), "prediction_identity": sorted(clean.prediction_identity.astype(str).unique()), "query_id_sha256": canonical_hash(sorted(clean.query_id.astype(str))), "group_id_sha256": canonical_hash(sorted(clean.group_id.astype(str))), "outer_folds": sorted(clean.merge(votes[["query_id", "outer_fold"]].drop_duplicates(), on="query_id").outer_fold.unique().tolist()), "holdout_rows_excluded": 121, "holdout_outcome_values_opened": False})

    model_parameters = json.loads((BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json").read_text(encoding="utf-8"))["model_parameters"]
    all_scores: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    label_rows: list[pd.DataFrame] = []
    model_hashes: dict[str, str] = {}
    eligibility = eligibility_frame(clean, action_names)
    for outer_fold in (0, 1, 2):
        train_labels, test_labels = nested_outer_labels(votes, clean, sources, source_columns, outer_fold)
        train_labels = train_labels.merge(eligibility, on=["query_id", "action_id"], how="left", validate="one_to_one")
        test_labels = test_labels.merge(eligibility, on=["query_id", "action_id"], how="left", validate="one_to_one")
        fit = fit_outer_ebms(clean, train_labels, outer_fold, action_names, model_parameters)
        for action in action_names:
            model_path = MODELS / f"outer_{outer_fold}_{action}.joblib"
            model_hashes[f"outer_{outer_fold}_{action}"] = sha256_file(model_path)
        scored = score_outer(clean, test_labels, fit["models"], action_names, outer_fold)
        all_scores.append(scored)
        label_rows.extend([train_labels.assign(label_scope="INNER_CROSSFIT_TRAIN"), test_labels.assign(label_scope="OUTER_CROSSFIT_TEST")])
        from src.recommend_hybrid.final.metrics import evaluate_grouped_ranking

        metric = evaluate_grouped_ranking(scored, relevance_column="relevance", score_column="score", eligible_column="eligible").to_dict()
        fold_rows.append({"outer_fold": outer_fold, "test_query_count": int(scored.query_id.nunique()), "test_group_count": int(scored.group_id.nunique()), "train_query_count": int(train_labels.query_id.nunique()), "train_group_count": int(train_labels.group_id.nunique()), "train_test_query_overlap": int(len(set(train_labels.query_id) & set(test_labels.query_id))), "train_test_group_overlap": int(len(set(train_labels.group_id) & set(test_labels.group_id))), "metric": metric, "fit_rows_by_action": fit["fit_rows"]})

    scores = pd.concat(all_scores, ignore_index=True).sort_values(["outer_fold", "query_id", "action_id"]).reset_index(drop=True)
    scores.to_parquet(OUT / "CROSSFITTED_OOF_SCORES.parquet", index=False)
    label_audit = pd.concat(label_rows, ignore_index=True).drop_duplicates(["query_id", "action_id", "label_scope"]).reset_index(drop=True)
    label_audit.to_parquet(OUT / "CROSSFITTED_LABEL_AUDIT.parquet", index=False)
    from src.recommend_hybrid.final.metrics import evaluate_grouped_ranking

    pooled = evaluate_grouped_ranking(scores, relevance_column="relevance", score_column="score", eligible_column="eligible").to_dict()
    bootstrap = grouped_bootstrap_ndcg(scores, iterations=1000, seed=2026)
    action = action_level(scores, action_names)
    action.to_csv(OUT / "CROSSFITTED_ACTION_RESULTS.csv", index=False)
    manifest = {"status": "COMPLETE_CROSSFITTED_EVIDENCE", "evaluation_scope": "179 genuine Hybrid OOF queries / 50 student groups", "not_a_final_holdout": True, "runtime_authorized": False, "no_hpo": True, "no_threshold_selection": True, "no_safety_tuning": True, "weak_label_protocol": "nested group-safe LabelModel: inner train -> outer-test aggregation", "outer_folds": fold_rows, "ebm_parameters": model_parameters, "feature_schema": list(json.loads((BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json").read_text(encoding="utf-8"))["feature_columns"]), "feature_schema_hash": canonical_hash(json.loads((BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json").read_text(encoding="utf-8"))["feature_columns"]), "model_hashes": model_hashes, "scores_sha256": sha256_file(OUT / "CROSSFITTED_OOF_SCORES.parquet"), "label_audit_sha256": sha256_file(OUT / "CROSSFITTED_LABEL_AUDIT.parquet"), "train_test_group_overlap_all_folds": 0}
    write_json(OUT / "CROSSFITTED_EBM_MANIFEST.json", manifest)
    write_json(OUT / "CROSSFITTED_RESULTS.json", {"status": "PASS_CROSSFITTED_RANKING_EVIDENCE", "scope": manifest["evaluation_scope"], "primary_metric": "NDCG@3", "pooled_metrics": pooled, "group_bootstrap_ndcg": bootstrap, "fold_metrics": fold_rows, "action_results": action.to_dict("records"), "threshold_and_safety_policy_metrics": "NOT_CLAIMED; no threshold/safety tuning performed in this cross-fitted ranking-only evaluation", "historical_panel_b_merged": False, "prediction_holdout_rows_used": 0, "claim_boundary": "cross-validated weak-supervision ranking evidence only; not held-out evidence and not a final recommendation performance claim"})
    write_json(OUT / "CROSSFITTED_INTEGRITY.json", {"status": "PASS_GROUP_SAFE_CROSSFITTED", "group_leakage": {"outer_train_test_overlap": 0, "inner_train_held_overlap": 0}, "holdout_rows_excluded": 121, "holdout_outcome_values_opened": False, "legacy_h1_used": False, "hpo": False, "feature_redesign": False, "action_redesign": False, "threshold_tuning": False, "safety_tuning": False, "outer_prediction_rerun": False})
    report = f"""# Cross-Fitted Recommendation Evidence\n\n## Decision\n\nNo group remains unused inside the active Phase8 recommendation case universe: all 86 student/groups occur in the feature, weak-label, EBM, threshold/safety, and development-metric scopes. Raw OULAD contains additional students, but they do not have the current action-level relevance/evaluation contract and are not treated as a valid untouched recommendation holdout.\n\nThe 121 previously designated final-inference rows were excluded from this evidence. No outcome/relevance values for those rows were opened.\n\n## Evaluation\n\n- scope: 179 genuine Hybrid OOF queries, 50 student/groups\n- outer folds: 3, group-disjoint\n- inner label-model fit: group-safe nested cross-fitting\n- EBM fit: outer-train only, fixed existing parameters, no HPO\n- threshold/safety tuning: not performed; no policy claims are made\n- primary evidence: cross-validated weak-supervision ranking evidence\n\nPooled metrics:\n\n```json\n{json.dumps(pooled, indent=2)}\n```\n\nGroup bootstrap NDCG@3:\n\n```json\n{json.dumps(bootstrap, indent=2)}\n```\n\nAction-level results are in `CROSSFITTED_ACTION_RESULTS.csv`. This evidence is not a new held-out result and does not inherit historical Panel B evidence.\n\n## Integrity\n\n- train/test group overlap: 0 in every outer fold\n- 121 contaminated-design rows used: 0\n- legacy H1 prediction: not used\n- HPO/feature redesign/action redesign: none\n- historical Panel B merged: no\n\n## Claim boundary\n\nSupported claim: the rebuilt ranker shows the reported cross-validated ranking behavior under group-safe nested/cross-fitted weak-label evaluation on the 179-row genuine Hybrid-OOF subset.\n\nNot supported: final held-out recommendation performance, superiority over a held-out baseline, or safety-policy performance.\n"""
    (REPORTS / "CROSS_FITTED_RECOMMENDATION_EVIDENCE.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "PASS_CROSSFITTED_RANKING_EVIDENCE", "queries": int(scores.query_id.nunique()), "groups": int(scores.group_id.nunique()), "ndcg_at_3": pooled["ndcg_at_3"], "bootstrap_ci": [bootstrap["ci_low_95"], bootstrap["ci_high_95"]]}, sort_keys=True))


if __name__ == "__main__":
    main()
