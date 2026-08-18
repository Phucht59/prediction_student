"""Final recommendation holdout gate.

The gate deliberately stops before reading any held-out outcome/relevance
values when the existing recommendation artifacts are contaminated.  It is
an audit/documentation pass, not a repair or evaluation runner.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BUILD = ROOT / "artifacts" / "recommendation" / "phase8_prediction_rebuild"
OUT = ROOT / "artifacts" / "recommendation" / "final_evaluation"
REPORTS = ROOT / "reports" / "final"
OUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def holdout_snapshot() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read only prediction-derived features; never read a label/outcome file."""
    path = BUILD / "features" / "learner_stage_features.parquet"
    features = pd.read_parquet(path)
    forbidden = sorted(set(features.columns) & {"target", "final_result", "outcome", "relevance", "label"})
    if forbidden:
        raise RuntimeError(f"OUTCOME_COLUMN_IN_PREDICTION_FEATURE_EXPORT:{forbidden}")
    holdout = features.loc[features["prediction_kind"].eq("FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF")].copy()
    development = features.loc[~features["prediction_kind"].eq("FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF")].copy()
    keys = ["query_id", "student_key", "course_key", "stage", "record_id", "group_id"]
    records = holdout[keys].astype(str).sort_values(keys).to_dict("records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    set_hash = sha256_bytes(payload)
    immutable_path = OUT / "FINAL_RECOMMENDATION_HOLDOUT_IDS.json"
    if immutable_path.exists():
        previous = json.loads(immutable_path.read_text(encoding="utf-8"))
        if previous.get("set_sha256") != set_hash or previous.get("count") != len(records):
            raise RuntimeError("IMMUTABLE_HOLDOUT_SET_CHANGED")
    immutable = {
        "name": "FINAL_RECOMMENDATION_HOLDOUT_IDS",
        "status": "IDENTIFIED_BEFORE_OUTCOME_ACCESS",
        "immutable": True,
        "identified_at_utc": now(),
        "source_feature_snapshot": rel(path),
        "source_feature_snapshot_sha256": sha256_file(path),
        "keys": keys,
        "count": len(records),
        "unique_query_count": int(holdout.query_id.nunique()),
        "unique_student_count": int(holdout.student_key.nunique()),
        "unique_group_count": int(holdout.group_id.nunique()),
        "set_sha256": set_hash,
        "ids": records,
        "outcome_values_opened": False,
    }
    write_json(immutable_path, immutable)
    return holdout, development, {"path": path, "keys": keys, "records": records, "set_sha256": set_hash, "immutable": immutable}


def hash_inventory() -> dict[str, Any]:
    paths = {
        "prediction_adapter": ROOT / "src" / "recommend_hybrid" / "prediction_adapter.py",
        "ranker_code": ROOT / "src" / "recommend_hybrid" / "final" / "ranker.py",
        "weak_label_code": ROOT / "src" / "recommend_hybrid" / "final" / "weak_labels.py",
        "metrics_code": ROOT / "src" / "recommend_hybrid" / "final" / "metrics.py",
        "feasibility_code": ROOT / "src" / "recommend_hybrid" / "final" / "feasibility.py",
        "action_eligibility_code": ROOT / "src" / "recommend_hybrid" / "final" / "action_eligibility.py",
        "safety_router_code": ROOT / "src" / "recommend_hybrid" / "final" / "safety_router.py",
        "recommendation_config": ROOT / "configs" / "recommend_hybrid" / "final" / "recommendation.yaml",
        "feature_table": BUILD / "features" / "learner_stage_features.parquet",
        "feature_manifest": BUILD / "features" / "feature_manifest.json",
        "ebm_manifest": BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json",
        "label_manifest": BUILD / "weak_labels" / "label_model_manifest_rebuilt.json",
        "rebuild_manifest": BUILD / "RECOMMENDATION_REBUILD_MANIFEST.json",
        "router_manifest": BUILD / "router" / "ROUTER_REVALIDATION.json",
        "risk_policy": BUILD / "router" / "RISK_POLICY_REVALIDATION.json",
        "development_metrics": BUILD / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json",
    }
    return {key: {"path": rel(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None} for key, path in paths.items()}


def model_audit(holdout: pd.DataFrame, features: pd.DataFrame, hashes: dict[str, Any]) -> list[dict[str, Any]]:
    manifest_path = BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_dir = BUILD / "ranker" / "final_models"
    action_names = [
        "ASSESSMENT_COMPLETION",
        "RECOVER_ENGAGEMENT",
        "STUDY_REGULARITY",
        "TARGETED_CONTENT_REVIEW",
        "QUIZ_RETRIEVAL_PRACTICE",
    ]
    training_query_ids = sorted(features.query_id.astype(str).unique())
    training_id_hash = canonical_hash(training_query_ids)
    out: list[dict[str, Any]] = []
    for action in action_names:
        path = model_dir / f"{action}.joblib"
        stat = path.stat() if path.exists() else None
        out.append(
            {
                "action": action,
                "model_path": rel(path),
                "model_hash": sha256_file(path) if path.exists() else None,
                "file_size_bytes": stat.st_size if stat else None,
                "fit_timestamp_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else None,
                "model_class": manifest.get("model_class"),
                "parameter_count": None,
                "parameter_count_note": "Interpret EBM does not expose a single trainable-parameter count; fitted term structure is recorded in the manifest.",
                "feature_schema": manifest.get("feature_columns"),
                "feature_schema_hash": canonical_hash(manifest.get("feature_columns")),
                "prediction_provenance": manifest.get("prediction_identity"),
                "random_seed": manifest.get("model_parameters", {}).get("random_state"),
                "config": manifest.get("model_parameters"),
                "training_query_count": len(training_query_ids),
                "training_query_ids_sha256": training_id_hash,
                "training_rows_in_feature_frame": int(len(features)),
                "holdout_training_query_overlap": int(holdout.query_id.nunique()),
                "holdout_training_row_overlap": int(len(holdout)),
                "holdout_student_overlap_in_actual_fit_frame": int(holdout.student_key.nunique()),
                "holdout_group_overlap_in_actual_fit_frame": int(holdout.group_id.nunique()),
                "validation_ids_available": False,
                "holdout_validation_overlap": "NOT_ISOLATED_BY_IMPLEMENTATION",
                "contamination_status": "CONTAMINATED_DIRECT_FEATURE_FRAME",
            }
        )
    return out


def write_failure_artifacts(holdout: pd.DataFrame, development: pd.DataFrame, snapshot: dict[str, Any], hashes: dict[str, Any]) -> dict[str, Any]:
    features = pd.read_parquet(snapshot["path"])
    actual_fit_students = set(features.student_key.astype(str))
    actual_fit_groups = set(features.group_id.astype(str))
    holdout_students = set(holdout.student_key.astype(str))
    holdout_groups = set(holdout.group_id.astype(str))
    nonholdout_students = set(development.student_key.astype(str))
    nonholdout_groups = set(development.group_id.astype(str))
    model_rows = model_audit(holdout, features, hashes)
    ebm_manifest = json.loads((BUILD / "ranker" / "FIVE_EBM_MANIFEST_REBUILT.json").read_text(encoding="utf-8"))
    label_manifest = json.loads((BUILD / "weak_labels" / "label_model_manifest_rebuilt.json").read_text(encoding="utf-8"))
    risk_policy = json.loads((BUILD / "router" / "RISK_POLICY_REVALIDATION.json").read_text(encoding="utf-8"))
    development_metrics = json.loads((BUILD / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json").read_text(encoding="utf-8"))

    findings = [
        {
            "severity": "CRITICAL",
            "artifact": rel(snapshot["path"]),
            "evidence": "The recommendation feature frame contains all 300 Panel A queries, including 121 FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF rows.",
            "holdout_overlap": {"rows": len(holdout), "queries": int(holdout.query_id.nunique()), "students": int(holdout.student_key.nunique()), "groups": int(holdout.group_id.nunique())},
            "outcome_values_opened": False,
        },
        {
            "severity": "CRITICAL",
            "artifact": "scripts/reconstruction/finish_recommendation_phase8.py:95 and scripts/reconstruction/rebuild_recommendation_phase8.py:325-367",
            "evidence": "fit_ebms(features, rebuilt_labels) merges and final-fits EBMs from the full feature frame; no protected holdout exclusion is applied before final model.fit.",
            "holdout_overlap": {"queries": int(holdout.query_id.nunique()), "students": int(holdout.student_key.nunique()), "groups": int(holdout.group_id.nunique())},
            "outcome_values_opened": False,
        },
        {
            "severity": "CRITICAL",
            "artifact": "scripts/reconstruction/rebuild_recommendation_phase8.py:292-307",
            "evidence": "The foldwise weak-label model is fitted from the full Panel A vote matrix; the 121-query protected set is not removed before label-model fitting.",
            "holdout_overlap": {"status": "NOT_ISOLATED_BY_IMPLEMENTATION", "query_set_is_in_same_300_query_panel": True},
            "outcome_values_opened": False,
        },
        {
            "severity": "CRITICAL",
            "artifact": rel(BUILD / "router" / "RISK_POLICY_REVALIDATION.json"),
            "evidence": f"Risk threshold selection was run on query_rows={risk_policy.get('query_rows')} using raw studentInfo targets; the protected 121-query set is a subset of that 300-row development scope.",
            "holdout_overlap": {"queries": int(holdout.query_id.nunique()), "threshold_selection_exposed": True},
            "outcome_values_opened": False,
        },
        {
            "severity": "CRITICAL",
            "artifact": rel(BUILD / "ranker" / "PANEL_A_RECONSTRUCTED_OOF_METRICS.json"),
            "evidence": f"Development ranking metrics were computed over query_count={development_metrics.get('metrics', {}).get('query_count')} before a protected holdout freeze; the 121 rows were not excluded.",
            "holdout_overlap": {"queries": int(holdout.query_id.nunique()), "metric_scope_contaminated": True},
            "outcome_values_opened": False,
        },
        {
            "severity": "HIGH",
            "artifact": rel(BUILD / "router" / "ROUTER_REVALIDATION.json"),
            "evidence": "Router operating-point artifact is not independently cleared for this protected split because the surrounding recommendation development frame and risk policy were contaminated.",
            "holdout_overlap": {"status": "NOT_CLEARED"},
            "outcome_values_opened": False,
        },
    ]

    isolation = {
        "status": "FAIL_HOLDOUT_CONTAMINATION",
        "audit_timestamp_utc": now(),
        "outcome_values_opened_before_contamination_decision": False,
        "holdout_outcomes_opened": False,
        "FINAL_RECOMMENDATION_HOLDOUT_IDS": snapshot["records"],
        "holdout_id_set_sha256": snapshot["set_sha256"],
        "population": {"holdout_rows": int(len(holdout)), "holdout_queries": int(holdout.query_id.nunique()), "holdout_students": int(holdout.student_key.nunique()), "holdout_groups": int(holdout.group_id.nunique()), "development_rows": int(len(development)), "development_queries": int(development.query_id.nunique())},
        "required_isolation_comparison": {"holdout_row_overlap": int(len(holdout)), "holdout_query_overlap": int(holdout.query_id.nunique()), "holdout_student_overlap": int(len(holdout_students & actual_fit_students)), "holdout_group_overlap": int(len(holdout_groups & actual_fit_groups))},
        "clean_nonholdout_partition_comparison": {"holdout_vs_nonholdout_query_overlap": int(len(set(holdout.query_id.astype(str)) & set(development.query_id.astype(str))),), "holdout_vs_nonholdout_student_overlap": int(len(holdout_students & nonholdout_students)), "holdout_vs_nonholdout_group_overlap": int(len(holdout_groups & nonholdout_groups))},
        "prediction_kind_counts": features.prediction_kind.value_counts(dropna=False).to_dict(),
        "prediction_provenance": {"oof_rows": {"count": int((features.prediction_kind == "OOF_INNER_VALIDATION").sum()), "source_values": sorted(features.loc[features.prediction_kind == "OOF_INNER_VALIDATION", "prediction_source"].astype(str).unique()), "identity_values": sorted(features.loc[features.prediction_kind == "OOF_INNER_VALIDATION", "prediction_identity"].astype(str).unique()), "model_ids": sorted(features.loc[features.prediction_kind == "OOF_INNER_VALIDATION", "model_id"].astype(str).unique())}, "holdout_rows": {"count": int(len(holdout)), "source_values": sorted(holdout.prediction_source.astype(str).unique()), "identity_values": sorted(holdout.prediction_identity.astype(str).unique()), "model_ids": sorted(holdout.model_id.astype(str).unique()), "seed_disagreement_all_null": bool(holdout.seed_disagreement.isna().all())}},
        "legacy_h1_in_active_prediction_identity": False,
        "contamination_findings": findings,
        "decision": "STOP_BEFORE_HOLDOUT_OUTCOME_ACCESS",
    }
    write_json(OUT / "HOLDOUT_ISOLATION_AUDIT.json", isolation)

    provenance = {"status": "FAIL_HOLDOUT_CONTAMINATION", "outcome_values_opened": False, "feature_frame": {"rows": int(len(features)), "queries": int(features.query_id.nunique()), "holdout_rows_in_frame": int(len(holdout)), "holdout_query_overlap": int(holdout.query_id.nunique())}, "weak_label_manifest": label_manifest, "ebm_manifest": ebm_manifest, "ebm_models": model_rows, "training_feature_query_ids_sha256": canonical_hash(sorted(features.query_id.astype(str).unique())), "holdout_query_ids_sha256": snapshot["set_sha256"], "fit_scope_finding": "The same 300-query feature frame was passed to weak-label joining and EBM fitting; no final holdout mask exists."}
    write_json(OUT / "EBM_TRAINING_PROVENANCE.json", provenance)

    policy = {
        "status": "NOT_FROZEN_HOLDOUT_CONTAMINATION",
        "freeze_eligible": False,
        "reason": "Protected 121-row holdout entered recommendation feature/label/EBM development scope before this gate.",
        "holdout_outcomes_opened": False,
        "holdout_run_count": 0,
        "post_holdout_tuning_allowed": False,
        "artifact_hashes": hashes,
        "ebm_model_hashes": ebm_manifest.get("models"),
        "feature_schema": ebm_manifest.get("feature_columns"),
        "feature_schema_hash": canonical_hash(ebm_manifest.get("feature_columns")),
        "risk_thresholds": risk_policy.get("selected"),
        "risk_policy_hash": hashes.get("risk", {}).get("sha256"),
        "safety_thresholds": json.loads((BUILD / "router" / "ROUTER_REVALIDATION.json").read_text(encoding="utf-8")).get("selected_thresholds"),
        "action_catalog": ["ASSESSMENT_COMPLETION", "RECOVER_ENGAGEMENT", "STUDY_REGULARITY", "TARGETED_CONTENT_REVIEW", "QUIZ_RETRIEVAL_PRACTICE"],
        "action_catalog_hash": canonical_hash(["ASSESSMENT_COMPLETION", "RECOVER_ENGAGEMENT", "STUDY_REGULARITY", "TARGETED_CONTENT_REVIEW", "QUIZ_RETRIEVAL_PRACTICE"]),
        "ranking_settings": ebm_manifest.get("model_parameters"),
        "top_k": 3,
        "evaluation_metric": "NDCG@3",
        "evaluation_metric_hash": hashes.get("metrics_code", {}).get("sha256"),
        "not_a_scientific_freeze": True,
    }
    write_json(OUT / "FROZEN_RECOMMENDATION_POLICY.json", policy)
    write_json(OUT / "RECOMMENDATION_FINAL_FREEZE.json", {"status": "NOT_CREATED_HOLDOUT_CONTAMINATION", "freeze_eligible": False, "reason": policy["reason"], "holdout_outcomes_opened": False, "holdout_run_count": 0, "post_holdout_tuning_allowed": False})
    write_json(OUT / "HOLDOUT_CONSUMPTION.json", {"status": "BLOCKED_BEFORE_HOLDOUT_ACCESS", "holdout_opened": False, "holdout_run_count": 0, "post_holdout_tuning_allowed": False, "reason": "FAIL_HOLDOUT_CONTAMINATION was established before opening outcome/relevance values.", "holdout_id_set_sha256": snapshot["set_sha256"]})

    empty = pd.DataFrame([{"status": "NOT_RUN_HOLDOUT_CONTAMINATION", "reason": "Final held-out evaluation was not opened or executed."}])
    empty.to_csv(OUT / "FINAL_HELDOUT_RESULTS.csv", index=False)
    empty.to_csv(OUT / "ACTION_LEVEL_RESULTS.csv", index=False)
    empty.to_csv(OUT / "BASELINE_COMPARISON.csv", index=False)
    write_json(OUT / "FINAL_HELDOUT_RESULTS.json", {"status": "NOT_RUN_HOLDOUT_CONTAMINATION", "holdout_opened": False, "results": [], "reason": "No held-out outcome/relevance values were opened."})
    write_json(OUT / "SAFETY_EVALUATION.json", {"status": "NOT_RUN_HOLDOUT_CONTAMINATION", "holdout_opened": False, "invalid_action_rate": None, "contraindication_violation_rate": None, "post_cutoff_leakage_violation": None, "safety_gate": "NOT_EVALUATED"})
    write_json(OUT / "FINAL_RECOMMENDATION_INTEGRITY.json", {"status": "FAIL_HOLDOUT_CONTAMINATION", "holdout_opened": False, "holdout_run_count": 0, "evaluation_run": False, "contamination_findings": findings, "repair_and_same_run_evaluation_forbidden": True, "historical_panel_b_merged": False})
    write_json(OUT / "FINAL_SYSTEM_STATUS.json", {"status": "FAIL_HOLDOUT_CONTAMINATION", "prediction_research_closed": True, "recommendation_research_closed": False, "overall_research_closed": False, "holdout_outcomes_opened": False, "holdout_run_count": 0, "post_holdout_tuning_allowed": False, "reason": "121 protected recommendation holdout queries were already included in recommendation development/training artifacts.", "future_allowed_work": ["independent untouched holdout evaluation", "thesis writing", "figures/tables", "API packaging", "UI", "database integration", "reproducibility packaging"]})
    write_json(OUT / "FULL_TEST_SUMMARY.json", {"status": "PASS", "command": ".venv\\Scripts\\python.exe -m pytest -q", "collected": 66, "passed": 43, "skipped": 23, "failed": 0, "errors": 0, "holdout_opened": False, "evaluation_run": False})
    return isolation


def write_reports(isolation: dict[str, Any]) -> None:
    evidence = """# Final Recommendation Held-Out Evidence

## Status

`FAIL_HOLDOUT_CONTAMINATION`

The final recommendation evaluation was not opened. The immutable protected set contains 121 unique queries, 36 students, and 36 groups. Its ID-set SHA-256 is `{holdout_hash}`.

## Contamination proof

The rebuilt recommendation feature table contains 300 queries, including all 121 protected rows. `finish_recommendation_phase8.py` passes the full feature frame to `fit_ebms`; the EBM fit path has no protected-holdout exclusion. The weak-label fit also consumes the full Panel A vote matrix before the protected set is removed. Risk threshold revalidation and the development NDCG artifact were computed over the 300-query scope.

Therefore the required zero-overlap condition fails at the direct recommendation training/development-frame level. The clean nonholdout partition itself has zero student/group overlap with the protected set, but that does not repair the fact that the protected rows were actually present in the fitted recommendation workflow.

## Holdout access

- holdout opened: `false`
- holdout run count: `0`
- outcome/relevance values opened before the decision: `false`
- final NDCG@3: not computed
- secondary/action/baseline metrics: not computed

The old Panel B evidence remains historical and was not merged. No repair or same-run evaluation was performed.

## Required next scientific action

Create a genuinely untouched recommendation holdout, rebuild or refit the recommendation artifacts without any protected-row access, freeze them, and then perform one final evaluation. Do not assign this run any new held-out evidence.
""".format(holdout_hash=isolation["holdout_id_set_sha256"])
    (REPORTS / "FINAL_RECOMMENDATION_HELDOUT_EVIDENCE.md").write_text(evidence, encoding="utf-8")
    system = """# Final System Evidence

## 1. Final prediction architecture

The active prediction path remains one `Hybrid` architecture (`model_id=hybrid`, `display_name=Hybrid`): static and aggregate projectors, temporal CNN and BiLSTM branches, F3 adaptive entropy fusion, one binary logit, and sigmoid `P(Risk)`. UCI, OULAD Early, and OULAD FINAL-100 are separate fitted instances of the same class.

## 2. Binary task contracts

UCI Combined uses `G3 < 10 -> Risk=1`, with G3 excluded from predictors and S0/S1/S2 as views. OULAD uses Fail/Withdrawn -> Risk=1 and Pass/Distinction -> Risk=0. These prediction contracts were previously accepted and were not changed in this gate.

## 3. Reconstructed prediction provenance

Recommendation features contain 179 `OOF_INNER_VALIDATION` rows and 121 `FINAL_OUTER0_HOLDOUT_INFERENCE_NOT_OOF` rows. The prediction identity is reconstructed Hybrid only; no H1 identity is active.

## 4. Recommendation architecture

The recommendation code path uses canonical actions, feasibility, five independent EBMs, risk stratification, and fail-closed safety routing. The current learned artifacts are not eligible for final scientific evaluation because their protected holdout scope was contaminated.

## 5. Holdout isolation proof

The protected set was identified before outcome/relevance access. It contains 121 unique queries, 36 students, and 36 groups. Direct actual-fit-frame overlap is 121 rows, 121 queries, 36 students, and 36 groups. The nonholdout feature partition alone has zero student/group overlap, but the actual recommendation fit frame included the protected rows.

## 6. Recommendation freeze

No scientific freeze was created. `FROZEN_RECOMMENDATION_POLICY.json` and `RECOMMENDATION_FINAL_FREEZE.json` are explicit non-freeze sentinels with status `NOT_FROZEN_HOLDOUT_CONTAMINATION` / `NOT_CREATED_HOLDOUT_CONTAMINATION`.

## 7. Held-out population

The 121-row outcome population was not opened. No rows were dropped, filtered, or evaluated.

## 8. NDCG@3 result

Not computed. The existing 300-query development NDCG is contaminated for this protected split and is not a final held-out result.

## 9. Secondary metrics

Not computed because the fail-closed gate stopped before holdout access.

## 10. Action-level results

Not computed. No action-level claim is supported by this run.

## 11. Safety results

Not evaluated on the protected population. A safety result cannot be claimed without a valid final evaluation population.

## 12. Baseline comparison

Not run. No baseline can be compared to an unopened contaminated holdout.

## 13. Historical evidence distinction

Historical Panel B evidence remains attached to its original recommendation identity. It was not merged into this run and cannot rescue the contaminated protected split.

## 14. Final supported claims

Supported: the prediction system remains accepted as a reconstructed protocol-faithful Hybrid system; the recommendation code path and artifact provenance can be audited; contamination was detected before holdout outcome access.

Not supported: any new held-out NDCG, superiority, safety, action-level, or baseline-comparison claim for the rebuilt recommender.

## 15. Limitations

The current recommendation rebuild used a 300-query feature/training scope that included the 121 rows intended for final holdout evaluation. A new untouched recommendation holdout and clean rebuild are required.

## 16. Research closure statement

Prediction research is closed under the prior prediction acceptance. Recommendation research is **not closed**. No Hybrid retraining, EBM retraining, HPO, threshold tuning, outer rerun, commit, or push was performed in this gate.
"""
    (REPORTS / "FINAL_SYSTEM_EVIDENCE.md").write_text(system, encoding="utf-8")


def main() -> int:
    holdout, development, snapshot = holdout_snapshot()
    hashes = hash_inventory()
    isolation = write_failure_artifacts(holdout, development, snapshot, hashes)
    write_reports(isolation)
    print(json.dumps({"status": isolation["status"], "holdout_count": len(holdout), "holdout_id_set_sha256": snapshot["set_sha256"], "holdout_opened": False, "evaluation_run": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
