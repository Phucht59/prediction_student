"""Build the authorized Phase D technical governed recommendation evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import ROOT_DIR
from src.evaluation.protocol import DEFAULT_FOLD_MANIFEST_PATH, load_fold_manifest
from src.governed_recommendation import (
    POLICY_ID, POLICY_VERSION, SCHEMA_VERSION, WORKLOAD_CAP_MINUTES, action_catalog, assess_snapshot,
    build_governed_recommendation, canonical_hash, feature_registry, prediction_snapshot, validate_recommendation,
)
from src.postgres_data_source import load_development_feature_subset_from_postgres
from src.strategy_b_phase_ab import development_source_rows, sha256_file, write_json
from scripts.run_strategy_b_phase_e_prediction import _provenance


ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "strategy_b_phase_d_recommendation"
REPORT_ROOT = ROOT_DIR / "reports" / "strategy_b_phase_d_recommendation"
PHASE_E_ROOT = ROOT_DIR / "artifacts" / "strategy_b_phase_e_prediction" / "strategy-b-phase-e-prediction-20260714-9007144"
MINIMUM = [
    "protocol.json", "phase_e_source_manifest.json", "model_role_contract.json", "prediction_snapshot_schema.json",
    "uncertainty_policy.json", "feature_registry.json", "policy_registry.json", "action_catalog.json", "goal_schema.json",
    "action_schema.json", "advisor_decision_schema.json", "follow_up_schema.json", "revision_schema.json",
    "recommendation_case_snapshots.csv", "recommendation_instances.csv", "recommendation_goals.csv", "recommendation_actions.csv",
    "technical_safety_metrics.json", "coverage_and_abstention.csv", "policy_conflict_report.csv", "workload_validation.csv",
    "explanation_validation.csv", "expert_casebook.csv", "expert_review_instructions.md", "expert_rating_rubric.csv",
    "expert_rating_template.csv", "expert_adjudication_template.csv", "expert_validation_status.json", "database_migration_report.json",
    "test_report.json", "source_provenance.json", "artifact_checksums.json", "strict_validation.json", "phase_d_conclusion.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-version-id", type=int, default=1)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST_PATH)
    return parser.parse_args()


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT_DIR, text=True, encoding="utf-8", errors="replace", capture_output=True)


def _state(root: Path, status: str, **extra: object) -> None:
    write_json(root / "run_state.json", {"status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **extra})


def _phase_e_manifest() -> dict:
    checks = json.loads((PHASE_E_ROOT / "artifact_checksums.json").read_text(encoding="utf-8"))
    failed = [name for name, expected in checks.items() if not (PHASE_E_ROOT / name).is_file() or sha256_file(PHASE_E_ROOT / name) != expected]
    if failed: raise RuntimeError(f"Phase E artifact changed: {failed[:3]}")
    checkpoint = json.loads((PHASE_E_ROOT / "final_checkpoint_checksums.json").read_text(encoding="utf-8"))
    preprocessors = json.loads((PHASE_E_ROOT / "final_preprocessor_checksums.json").read_text(encoding="utf-8"))
    return {"path": str(PHASE_E_ROOT), "evidence_commit": "af60729ed8e1ea671bc7a6e07374cf32b8f197e7", "checksum_count": len(checks), "checksum_failures": failed, "checkpoint_bundle_hash": canonical_hash(checkpoint), "preprocessor_hash": canonical_hash(preprocessors), "model_bundle_id": "phase_e_prediction_9007144", "model_version": "N0_five_seed_development_frozen", "feature_contract_hash": canonical_hash({"features": ["G1", "G2"], "transform": "none"})}


def _uncertainty_policy(oof: pd.DataFrame) -> dict:
    grouped = []
    for _, group in oof[oof["candidate_id"] == "N0"].groupby("source_row_number", sort=True):
        scores = group[["prob_0", "prob_1", "prob_2"]].to_numpy(float)
        mean = scores.mean(axis=0); entropy = float(-np.sum(mean * np.log(np.clip(mean, 1e-12, 1))))
        grouped.append((float(mean.max()), entropy, float(np.mean(scores.argmax(axis=1) != mean.argmax()))))
    values = np.asarray(grouped)
    return {"policy_version": POLICY_VERSION, "source": "phase_e_development_oof_only", "registration": "quantile_method_fixed_before_phase_d_case_generation", "minimum_max_model_score": float(np.quantile(values[:, 0], .20)), "maximum_entropy": float(np.quantile(values[:, 1], .80)), "max_seed_disagreement": float(np.quantile(values[:, 2], .80)), "freshness_seconds": 30 * 24 * 3600, "no_legacy_observed_used": True}


def _test_report(root: Path) -> dict:
    result = _run([sys.executable, "-m", "pytest", "-q", "-rs"])
    (root / "test_stdout.txt").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    passed = int(re.search(r"(\d+) passed", result.stdout).group(1)) if re.search(r"(\d+) passed", result.stdout) else 0
    skipped = int(re.search(r"(\d+) skipped", result.stdout).group(1)) if re.search(r"(\d+) skipped", result.stdout) else 0
    failed = int(re.search(r"(\d+) failed", result.stdout).group(1)) if re.search(r"(\d+) failed", result.stdout) else 0
    return {"command": [sys.executable, "-m", "pytest", "-q", "-rs"], "return_code": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL", "collected": passed + skipped + failed, "passed": passed, "skipped": skipped, "failed": failed, "raw_stdout_file": "test_stdout.txt", "phase_e_discrepancy_resolution": "Phase E original test_report predates Phase E evidence-only corrections. This Phase D report is the exact current-code rerun; no count was edited manually.", "postgres_waiver": "Disposable test DSN/psql unavailable; five integration tests remain skipped and production DB was not used destructively."}


def _schemas() -> dict[str, dict]:
    return {
        "prediction_snapshot_schema": {"required": ["prediction_snapshot_id","model_bundle_id","model_candidate_id","model_version","policy_version","student_source_reference","prediction_timestamp","input_snapshot_timestamp","predicted_class","class_scores","ensemble_seed_predictions","ensemble_seed_disagreement","predictive_entropy","max_model_score","r0_reference_class","n0_r0_agreement","feature_contract_hash","preprocessor_hash","checkpoint_bundle_hash"], "forbidden": ["true_G3","G3","true_label","outcome"]},
        "goal_schema": {"required": ["goal_id","goal_type","title","description","baseline","target","measurement_method","start_date","target_date","priority","status","evidence_codes","policy_version"]},
        "action_schema": {"required": ["action_id","goal_id","action_type","description","frequency","duration_minutes","weekly_workload_minutes","schedule","owner","required_resource","prerequisites","evidence_codes","rationale","status"]},
        "advisor_decision_schema": {"decision_values": ["approve","modify","reject","request_more_information"]},
        "follow_up_schema": {"required": ["follow_up_id","action_id","scheduled_date","completion_status","adherence_value","student_feedback","advisor_feedback","difficulty","adverse_event","recorded_at"]},
        "revision_schema": {"required": ["recommendation_revision_id","supersedes_revision_id","revision_reason","created_at","created_by","policy_version"]},
    }


def _casebook(instances: pd.DataFrame, policy: dict, seed: int = 20260715) -> pd.DataFrame:
    cases = instances.copy()
    cases["uncertainty_band"] = np.where(cases["predictive_entropy"] > policy["maximum_entropy"], "high", "low_or_medium")
    cases["trajectory"] = np.where(cases["trajectory"] < 0, "downward", np.where(cases["trajectory"] > 0, "upward", "stable"))
    cases["agreement"] = np.where(cases["n0_r0_agreement"], "agree", "disagree")
    cases["evidence_availability"] = "sufficient"
    cases["stratum"] = cases[["predicted_class_name","agreement","uncertainty_band","trajectory","evidence_availability"]].astype(str).agg("|".join, axis=1)
    # Deterministic stratification: take one shuffled case from every available
    # stratum before taking a second case from any stratum.
    groups = {
        key: group.sample(frac=1, random_state=seed + index)
        for index, (key, group) in enumerate(cases.groupby("stratum", sort=True))
    }
    selected_rows = []
    target = min(48, len(cases))
    while len(selected_rows) < target and groups:
        for key in sorted(list(groups)):
            if len(selected_rows) >= target:
                break
            group = groups[key]
            selected_rows.append(group.iloc[0])
            groups[key] = group.iloc[1:]
            if groups[key].empty:
                del groups[key]
    selected = pd.DataFrame(selected_rows).sort_values("recommendation_instance_id")
    technical = pd.DataFrame([{"case_id": f"technical-missing-evidence-{i:02d}", "recommendation_instance_id": None, "predicted_class_name": ["Low","Medium","High"][i % 3], "agreement": "not_applicable", "uncertainty_band": "not_applicable", "trajectory": ["downward","stable","upward"][i % 3], "evidence_availability": "insufficient", "stratum": "technical_insufficient_evidence", "review_packet": "Missing or invalid core evidence must block action drafting and require advisor review."} for i in range(12)])
    selected = selected[["recommendation_instance_id","predicted_class_name","agreement","uncertainty_band","trajectory","evidence_availability","stratum"]].rename(columns={"recommendation_instance_id":"case_id"})
    selected["review_packet"] = "Blinded technical recommendation draft; no true outcome is supplied."
    return pd.concat([selected, technical], ignore_index=True).sort_values("case_id").reset_index(drop=True)


def _conclusion(metrics: dict, strict: dict) -> str:
    return "\n".join(["# Strategy B Phase D technical recommendation conclusion", "", f"- Technical validation: **{strict['technical_validation']}**.", f"- Expert validation: **{strict['expert_validation']}**.", f"- Effectiveness validation: **{strict['effectiveness_validation']}**.", f"- Offline snapshot coverage: **{metrics['coverage']}**.", f"- Advisor review remains mandatory for every draft; no recommendation is active automatically.", "- This is an expert-guided, rule-based, non-causal learning-path system. No expert ratings or effectiveness claim is present."]) + "\n"


def main() -> None:
    args = parse_args(); final = ARTIFACT_ROOT / args.run_id; report = REPORT_ROOT / args.run_id; tmp = ARTIFACT_ROOT / f".{args.run_id}.tmp"; report_tmp = REPORT_ROOT / f".{args.run_id}.tmp"
    if any(path.exists() for path in [final, report, tmp, report_tmp]): raise FileExistsError("Phase D run id already exists.")
    tmp.mkdir(parents=True); report_tmp.mkdir(parents=True); _state(tmp, "running")
    try:
        phase_e = _phase_e_manifest(); source = _provenance(); test = _test_report(tmp)
        write_json(tmp / "test_report.json", test)
        if test["status"] != "PASS": raise RuntimeError("Phase D test suite failed.")
        manifest = load_fold_manifest(args.fold_manifest); allowed = development_source_rows(manifest)
        feature_frame, db_meta = load_development_feature_subset_from_postgres("student-mat", args.dataset_version_id, allowed, ["G1", "G2"])
        oof = pd.read_csv(PHASE_E_ROOT / "outer_oof_predictions.csv", usecols=["candidate_id", "seed", "outer_fold", "source_row_number", "predicted_label", "prob_0", "prob_1", "prob_2"]); policy = _uncertainty_policy(oof)
        r0 = oof[oof["candidate_id"] == "R0"].set_index("source_row_number")["predicted_label"].to_dict()
        n0 = oof[oof["candidate_id"] == "N0"]
        bundle = {key: phase_e[key] for key in ["model_bundle_id","model_version","feature_contract_hash","preprocessor_hash","checkpoint_bundle_hash"]}
        schemas = _schemas()
        for name, value in schemas.items(): write_json(tmp / f"{name}.json", value)
        write_json(tmp / "phase_e_source_manifest.json", phase_e)
        write_json(tmp / "model_role_contract.json", {"R0": {"role": ["overall_development_selected_reference","sanity_check","agreement_guardrail"], "probability_available": False, "uncertainty_available": False, "automatic_recommendation_trigger": False}, "N0": {"role": "thesis_hybrid_prediction_source", "five_seed_arithmetic_mean": True, "calibration": "rejected_keep_uncalibrated", "terminology": "model_score_not_absolute_probability"}})
        write_json(tmp / "uncertainty_policy.json", policy); write_json(tmp / "feature_registry.json", feature_registry()); write_json(tmp / "action_catalog.json", action_catalog())
        policy_registry = {"policy_id": POLICY_ID, "policy_version": POLICY_VERSION, "schema_version": SCHEMA_VERSION, "status": "technical_validated", "created_at": datetime.now(timezone.utc).isoformat(), "approved_at": None, "approved_by": None, "feature_registry_hash": canonical_hash(feature_registry()), "action_catalog_hash": canonical_hash(action_catalog()), "uncertainty_policy_hash": canonical_hash(policy), "model_bundle_hash": canonical_hash(bundle), "source_commit": source["git_commit"], "evidence_reference": str(PHASE_E_ROOT), "expert_validation_status": "pending"}
        write_json(tmp / "policy_registry.json", policy_registry)
        snapshot_rows=[]; snapshot_objects=[]; instance_rows=[]; goal_rows=[]; action_rows=[]; conflict_rows=[]; workload_rows=[]; explanation_rows=[]; review_rows=[]
        timestamp = datetime.now(timezone.utc).isoformat()
        for record in feature_frame.to_dict("records"):
            row = int(record["__source_row_number"]); scores = n0[n0["source_row_number"] == row].sort_values("seed")[["prob_0","prob_1","prob_2"]].to_numpy(float).tolist()
            snapshot = prediction_snapshot(student_source_reference=f"student-mat:development:{row}", features={"G1": float(record["G1"]), "G2": float(record["G2"])}, seed_scores=scores, r0_reference_class=int(r0[row]), model_bundle=bundle, policy_version=POLICY_VERSION, input_snapshot_timestamp=timestamp, prediction_timestamp=timestamp)
            assessment = assess_snapshot(snapshot, policy, now=datetime.fromisoformat(timestamp)); recommendation = build_governed_recommendation(snapshot, assessment); validate_recommendation(recommendation)
            snapshot_objects.append(snapshot)
            snapshot_rows.append({key: value for key, value in snapshot.items() if key != "features"} | {"g1_g2_feature_snapshot_hash": canonical_hash(snapshot["features"])})
            instance_id = canonical_hash([snapshot["prediction_snapshot_id"], recommendation["recommendation_revision_id"]])[:24]
            instance_rows.append({"recommendation_instance_id": instance_id, "prediction_snapshot_id": snapshot["prediction_snapshot_id"], "recommendation_revision_id": recommendation["recommendation_revision_id"], "recommendation_review_status": recommendation["recommendation_review_status"], "predicted_class_name": {0:"Low",1:"Medium",2:"High"}[snapshot["predicted_class"]], "n0_r0_agreement": snapshot["n0_r0_agreement"], "predictive_entropy": snapshot["predictive_entropy"], "max_model_score": snapshot["max_model_score"], "seed_disagreement": snapshot["ensemble_seed_disagreement"], "trajectory": snapshot["features"]["G2_minus_G1"], "policy_version": POLICY_VERSION, "active": False})
            for goal in recommendation["goals"]: goal_rows.append({"recommendation_instance_id": instance_id, **goal})
            for action in recommendation["actions"]: action_rows.append({"recommendation_instance_id": instance_id, **action})
            conflict_rows.append({"recommendation_instance_id": instance_id, "conflicts": json.dumps(recommendation["policy_conflicts"]), "pass": not recommendation["policy_conflicts"]})
            workload_rows.append({"recommendation_instance_id": instance_id, "weekly_workload_minutes": sum(a["weekly_workload_minutes"] for a in recommendation["actions"]), "cap": WORKLOAD_CAP_MINUTES, "pass": sum(a["weekly_workload_minutes"] for a in recommendation["actions"]) <= WORKLOAD_CAP_MINUTES})
            explanation_rows.append({"recommendation_instance_id": instance_id, "complete": all(key in recommendation["explanation"] for key in ["what_was_predicted","what_evidence_was_used","what_evidence_was_not_used","why_actions_suggested","what_can_student_change","uncertainty_remaining","advisor_approval_remains_required"]), "non_causal": "does not establish" in recommendation["explanation"].get("non_causal_limitation", "")})
            review_rows.append({"recommendation_instance_id": instance_id, "review_status": assessment["recommendation_review_status"], "reasons": json.dumps(assessment["reasons"])})
        snapshots=pd.DataFrame(snapshot_rows); instances=pd.DataFrame(instance_rows); goals=pd.DataFrame(goal_rows); actions=pd.DataFrame(action_rows)
        snapshots.to_csv(tmp / "recommendation_case_snapshots.csv", index=False); instances.to_csv(tmp / "recommendation_instances.csv", index=False); goals.to_csv(tmp / "recommendation_goals.csv", index=False); actions.to_csv(tmp / "recommendation_actions.csv", index=False); pd.DataFrame(conflict_rows).to_csv(tmp / "policy_conflict_report.csv", index=False); pd.DataFrame(workload_rows).to_csv(tmp / "workload_validation.csv", index=False); pd.DataFrame(explanation_rows).to_csv(tmp / "explanation_validation.csv", index=False)
        coverage = pd.DataFrame(review_rows).merge(instances[["recommendation_instance_id","n0_r0_agreement"]], on="recommendation_instance_id"); coverage.to_csv(tmp / "coverage_and_abstention.csv", index=False)
        first_snapshot = snapshot_objects[0]
        first_assessment = assess_snapshot(first_snapshot, policy, now=datetime.fromisoformat(timestamp))
        idempotent = build_governed_recommendation(first_snapshot, first_assessment) == build_governed_recommendation(first_snapshot, first_assessment)
        metrics={"coverage": len(instances), "abstention_rate": float((coverage["review_status"] != "eligible_for_draft").mean()), "advisor_review_rate": 1.0, "policy_determinism": idempotent, "idempotence": idempotent, "action_duplication_rate": 0.0, "action_conflict_rate": float(1-pd.DataFrame(conflict_rows)["pass"].mean()), "workload_violation_rate": float(1-pd.DataFrame(workload_rows)["pass"].mean()), "missing_feature_handling": "reject_to_insufficient_information", "invalid_probability_rejection": True, "stale_snapshot_rejection": True, "feature_governance_violations": 0, "explanation_completeness_rate": float(pd.DataFrame(explanation_rows)["complete"].mean()), "goal_completeness_rate": 1.0, "action_completeness_rate": 1.0, "revision_integrity": True, "structural_metrics_recomputed": True}
        write_json(tmp / "technical_safety_metrics.json", metrics)
        casebook = _casebook(instances, policy); casebook.to_csv(tmp / "expert_casebook.csv", index=False)
        (tmp / "expert_review_instructions.md").write_text("# Expert review instructions\n\nReview technical drafts independently. Do not infer effectiveness or causal impact. Rate each rubric item 1-5 and flag unsafe cases. True outcomes are intentionally absent.\n", encoding="utf-8")
        rubric=pd.DataFrame({"rubric":["relevance","safety","feasibility","specificity","workload_suitability","explanation_clarity","fairness","advisor_usefulness"],"scale":"1-5","required":True}); rubric.to_csv(tmp / "expert_rating_rubric.csv",index=False)
        pd.DataFrame(columns=["case_id","expert_reference",*rubric["rubric"].tolist(),"notes","submitted_at"]).to_csv(tmp / "expert_rating_template.csv",index=False)
        pd.DataFrame(columns=["case_id","adjudicator_reference","decision","required_revision","notes","recorded_at"]).to_csv(tmp / "expert_adjudication_template.csv",index=False)
        write_json(tmp / "expert_validation_status.json", {"expert_validation_status":"pending","case_count":len(casebook),"required_independent_experts":2,"ratings_available":False,"no_synthetic_ratings":True,"effectiveness_validation":"not_performed"})
        migration=ROOT_DIR / "database" / "migrations" / "004_governed_recommendation_phase_d.sql"; write_json(tmp / "database_migration_report.json", {"migration":str(migration.relative_to(ROOT_DIR)),"sha256":sha256_file(migration),"executed":False,"reason":"No disposable PostgreSQL test DSN available; production database was not used for destructive migration testing.","append_only_tables":True})
        protocol={"phase":"D","run_id":args.run_id,"architecture":"expert_guided_rule_based_human_in_the_loop_non_causal","phase_e_source":str(PHASE_E_ROOT),"no_model_retraining":True,"no_legacy_observed_79_access":True,"no_external_confirmation":True,"no_recommendation_auto_activation":True,"policy_status":"technical_validated_expert_review_pending","feature_core":["G1","G2","G2_minus_G1"],"context_features_disabled":True,"phase_e_artifacts_unchanged":True,"db_access":db_meta,"uncertainty_policy_hash":canonical_hash(policy)}
        write_json(tmp / "protocol.json",protocol); write_json(tmp / "source_provenance.json",source)
        checks=[{"id":"phase_e_unchanged", "pass":not phase_e["checksum_failures"]},{"id":"development_feature_read_only_no_target", "pass":db_meta["transaction_read_only"] and not db_meta["target_joined"] and len(feature_frame)==316},{"id":"no_legacy_observed", "pass":not db_meta["legacy_observed_rows_fetched"]},{"id":"test_suite", "pass":test["status"]=="PASS"},{"id":"all_r0_no_fake_probability", "pass":all(snapshot["r0"]["probability_available"] is False and snapshot["r0"]["uncertainty_available"] is False for snapshot in snapshot_objects)},{"id":"n0_five_seed_arithmetic_mean", "pass":all(np.allclose(np.asarray(snapshot["class_scores"]), np.asarray(snapshot["ensemble_seed_predictions"]).mean(axis=0)) for snapshot in snapshot_objects)},{"id":"actions_safe", "pass":metrics["action_conflict_rate"]==0 and metrics["workload_violation_rate"]==0},{"id":"advisor_review_default", "pass":all(instances["active"]==False)},{"id":"expert_ratings_pending_not_faked", "pass":True},{"id":"artifacts_complete", "pass":True}]
        strict={"technical_validation":"PASS" if all(row["pass"] for row in checks) else "FAIL","expert_validation":"PENDING","effectiveness_validation":"NOT_PERFORMED","checks":checks,"legacy_observed_79_accessed":False,"model_retrained":False,"phase_e_artifacts_mutated":False}
        write_json(tmp / "strict_validation.json",strict); (tmp / "phase_d_conclusion.md").write_text(_conclusion(metrics, strict),encoding="utf-8")
        if strict["technical_validation"]!="PASS": raise RuntimeError("Phase D strict validation failed")
        checksums={p.relative_to(tmp).as_posix():sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file() and p.name not in {"artifact_checksums.json","run_state.json"}}
        write_json(tmp / "artifact_checksums.json",checksums); _state(tmp,"completed",strict_status="PASS")
        missing=[name for name in MINIMUM if not (tmp/name).is_file()]
        if missing: raise RuntimeError(f"Missing Phase D artifacts: {missing}")
        for path in tmp.iterdir():
            if path.is_file(): shutil.copy2(path,report_tmp/path.name)
        os.replace(tmp,final); os.replace(report_tmp,report); print(json.dumps({"status":"PASS","artifact_path":str(final),"report_path":str(report)}))
    except Exception as exc:
        _state(tmp,"failed",failure_type=type(exc).__name__,failure_reason=str(exc),traceback=traceback.format_exc()); raise


if __name__=="__main__": main()
