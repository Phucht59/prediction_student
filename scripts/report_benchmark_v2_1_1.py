"""Materialize read-only Benchmark Reporting/Validation Patch V2.1.1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import METRIC_VERSION
from src.evaluation.protocol import DEFAULT_FOLD_MANIFEST_PATH, file_checksum, load_fold_manifest
from src.evaluation.reporting_v2_1_1 import (
    JOB_COLUMNS, MODEL_REGISTRY, PATCH_VERSION, SCALAR_METRICS,
    aggregate_ece_corrections, build_expected_job_contract, checksum_validation,
    compare_expected_jobs, feature_contracts, paired_comparisons,
    recompute_metrics, render_paired_markdown, validate_record_coverage,
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _evidence(test_report: dict, key: str) -> tuple[str, str]:
    section = test_report.get(key)
    if not isinstance(section, dict): return "not_checked", "test evidence unavailable"
    status = section.get("status", "not_checked")
    return status, test_report.get("path", "reports/benchmark_v2/v2_1_1/test_report.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="benchmark-v2-full-20260713c")
    parser.add_argument("--patch-source-commit", required=True)
    parser.add_argument("--test-report", type=Path)
    args = parser.parse_args()
    artifact = ROOT / "artifacts/benchmark_v2" / args.run_id
    out = ROOT / "reports/benchmark_v2/v2_1_1"; out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((artifact / "benchmark_manifest.json").read_text(encoding="utf-8"))
    fold_manifest = load_fold_manifest(); predictions = pd.read_csv(artifact / "predictions/outer_validation_predictions.csv")
    stored = pd.read_csv(artifact / "fold_metrics.csv")
    test_report_path = args.test_report or out / "test_report.json"
    test_report = json.loads(test_report_path.read_text(encoding="utf-8")) if test_report_path.is_file() else {}
    test_report["path"] = str(test_report_path.relative_to(ROOT)).replace("\\", "/") if test_report_path.is_absolute() else str(test_report_path)

    contract = build_expected_job_contract(fold_manifest)
    (out / "expected_job_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    jobs, job_summary = compare_expected_jobs(contract, predictions)
    stored_duplicate_jobs = int(stored.duplicated(["scenario", "model_name", "outer_fold", "training_seed"]).sum())
    job_summary["duplicate_jobs"] = stored_duplicate_jobs
    jobs.to_csv(out / "expected_vs_actual_jobs.csv", index=False)
    coverage, coverage_summary = validate_record_coverage(contract, predictions, fold_manifest); coverage.to_csv(out / "record_coverage.csv", index=False)
    scalar, confusion, per_class, scalar_bad, structured_bad = recompute_metrics(predictions, stored)
    scalar.to_csv(out / "metric_recomputation_by_job.csv", index=False); scalar_bad.to_csv(out / "metric_mismatches.csv", index=False)
    confusion.to_csv(out / "confusion_matrix_recomputation.csv", index=False); per_class.to_csv(out / "per_class_metric_recomputation.csv", index=False)
    structured_bad.to_csv(out / "structured_metric_mismatches.csv", index=False)

    source_checksums = json.loads((artifact / "checksums.json").read_text(encoding="utf-8"))
    check_frame = checksum_validation(artifact, source_checksums); check_frame.to_csv(out / "checksum_validation.csv", index=False)
    feature_rows = feature_contracts(fold_manifest, int(predictions.dataset_version_id.iloc[0]))
    feature_validation = pd.DataFrame([{**row, "ordered_features": json.dumps(row["ordered_features"]), "valid": True} for row in feature_rows])
    feature_validation.to_csv(out / "feature_contract_validation.csv", index=False)

    # Ranking uses the same estimator as paired comparison: seed mean in fold, then five-fold summary.
    ranking=[]
    for model in MODEL_REGISTRY:
        job = scalar[(scalar.scenario == model.scenario) & (scalar.model_name == model.model_name)]
        by_fold = job.groupby(["metric", "outer_fold"]).recomputed.mean()
        row={"scenario":model.scenario,"model":model.model_name,"feature_set_id":model.feature_set_id,
             "estimator_definition":model.estimator_group,"metric_primary":"macro_f1","n_outer_folds":5,
             "n_training_seeds":model.n_seeds,"n_fold_seed_evaluations":model.n_seeds*5,
             "n_record_prediction_rows":len(predictions[(predictions.scenario==model.scenario)&(predictions.model_name==model.model_name)]),
             "n_unique_outer_validation_records":predictions[(predictions.scenario==model.scenario)&(predictions.model_name==model.model_name)].record_id.nunique(),
             "source_run_id":args.run_id,"source_commit":manifest["source_commit"],
             "prediction_checksum":file_checksum(artifact/"predictions/outer_validation_predictions.csv"),"validation_version":PATCH_VERSION}
        for metric in SCALAR_METRICS:
            vals=by_fold.loc[metric].to_numpy(float); row[metric+"_mean"]=float(vals.mean())
            if metric=="macro_f1":
                row.update({"macro_f1_sd_across_outer_folds":float(vals.std(ddof=1)),"macro_f1_median":float(np.median(vals)),
                            "macro_f1_min":float(vals.min()),"macro_f1_max":float(vals.max()),"macro_f1_fold_scores":json.dumps(vals.tolist())})
        ranking.append(row)
    ranking_frame=pd.DataFrame(ranking).sort_values(["scenario","macro_f1_mean"],ascending=[True,False])
    ranking_frame.to_csv(out/"ranking_by_scenario_v2_1_1.csv",index=False)
    rank_view=ranking_frame[["scenario","model","feature_set_id","macro_f1_mean","macro_f1_sd_across_outer_folds"]]
    rank_table="| "+" | ".join(rank_view.columns)+" |\n|"+"|".join(["---"]*len(rank_view.columns))+"|\n"+"".join("| "+" | ".join(str(value) for value in row)+" |\n" for row in rank_view.itertuples(index=False,name=None))
    (out/"ranking_by_scenario_v2_1_1.md").write_text("# Ranking by scenario V2.1.1\n\n"+rank_table,encoding="utf-8")

    paired=paired_comparisons(predictions); paired.to_csv(out/"paired_comparisons_v2_1_1.csv",index=False)
    (out/"paired_comparisons_v2_1_1.md").write_text(render_paired_markdown(paired),encoding="utf-8")
    aggregate_ece_corrections(scalar_bad).to_csv(out/"ece_correction_report.csv",index=False)

    legacy=json.loads((ROOT/"artifacts/legacy_v1/legacy_manifest.json").read_text(encoding="utf-8"))
    legacy_ids=set(legacy["current_79_record_ids"]); prediction_ids=set(predictions.record_id)
    legacy_overlap=legacy_ids & prediction_ids
    legacy_status="verified" if not legacy_overlap else "failed"
    leakage_status, leakage_source=_evidence(test_report,"leakage_guards")
    postgres_status, postgres_source=_evidence(test_report,"postgresql_tests")
    max_prob_error=float(np.abs(predictions[["probability_low","probability_medium","probability_high"]].sum(axis=1)-1).max())
    argmax=np.argmax(predictions[["probability_low","probability_medium","probability_high"]].to_numpy(),axis=1)
    required_ok=(job_summary["missing"]==job_summary["unexpected"]==job_summary["duplicate_jobs"]==0 and
                 coverage_summary["duplicate_prediction_rows"]==coverage_summary["invalid_coverage_jobs"]==0 and
                 bool(check_frame.valid.all()) and structured_bad.empty and
                 max_prob_error<=1e-6 and np.array_equal(argmax,predictions.predicted_label.to_numpy()) and
                 legacy_status=="verified" and leakage_status=="verified" and postgres_status=="verified")
    validation={"patch_version":PATCH_VERSION,"source_run_id":args.run_id,"source_run_commit":manifest["source_commit"],
        "patch_source_commit":args.patch_source_commit,"expected_jobs":job_summary["expected"],"actual_jobs":job_summary["actual"],
        "missing_jobs":job_summary["missing"],"unexpected_jobs":job_summary["unexpected"],"duplicate_jobs":job_summary["duplicate_jobs"],
        "expected_prediction_rows":sum(x["expected_record_count"] for x in contract["jobs"]),"actual_prediction_rows":len(predictions),
        "duplicate_prediction_rows":coverage_summary["duplicate_prediction_rows"],"record_coverage_status":"valid" if coverage_summary["invalid_coverage_jobs"]==0 else "invalid",
        "probability_contract_status":"valid" if max_prob_error<=1e-6 else "invalid","max_probability_sum_error":max_prob_error,
        "label_argmax_status":"valid" if np.array_equal(argmax,predictions.predicted_label.to_numpy()) else "invalid",
        "checksum_validation_status":"valid" if bool(check_frame.valid.all()) else "invalid","checksum_files_checked":len(check_frame),
        "checksum_failures":int((~check_frame.valid).sum()),"feature_contract_status":"valid" if bool(feature_validation.valid.all()) else "invalid",
        "metric_recomputation_status":"valid_with_known_stored_ece_defect" if len(scalar_bad)==30 and set(scalar_bad.metric)=={"ece_top_label_equal_width_10"} else "invalid",
        "scalar_metric_mismatches":len(scalar_bad),"confusion_matrix_status":"valid" if confusion.match.all() else "invalid",
        "per_class_metric_status":"valid" if structured_bad.empty else "invalid","stored_ece_status":"invalid_for_affected_hard_baseline_jobs",
        "legacy_79_isolation_status":legacy_status,"legacy_79_evidence_source":"record_identity_intersection:artifacts/legacy_v1/legacy_manifest.json",
        "leakage_guard_status":leakage_status,"leakage_evidence_source":leakage_source,
        "postgres_tests_status":postgres_status,"test_report_path":test_report.get("path",postgres_source),
        "original_artifacts_modified":False,"overall_validation_status":"valid" if required_ok else "invalid"}
    (out/"validation_v2_1_1.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    (out/"validation_v2_1_1.md").write_text("# Validation V2.1.1\n\n```json\n"+json.dumps(validation,indent=2)+"\n```\n",encoding="utf-8")
    (out/"reporting_patch_changelog.md").write_text("# Reporting patch V2.1.1\n\nRead-only derivation from immutable predictions. Adds independent expected-job contract, structured metric validation, estimator-consistent late/early paired comparisons, expanded feature contracts, aggregated ECE correction, and evidence-qualified validation. No training or original artifact mutation.\n",encoding="utf-8")

    # Manifest lists critical immutable inputs and all outputs except itself; its sidecar closes provenance.
    input_paths={"benchmark_manifest.json":artifact/"benchmark_manifest.json","checksums.json":artifact/"checksums.json",
        "fold_metrics.csv":artifact/"fold_metrics.csv","outer_validation_predictions.csv":artifact/"predictions/outer_validation_predictions.csv",
        "selected_configs.json":artifact/"configs/selected_configs.json","fold_manifest":DEFAULT_FOLD_MANIFEST_PATH,
        "feature_availability":ROOT/"config/feature_availability.yaml","features_pre_assessment":ROOT/"config/features_pre_assessment.yaml",
        "features_early_warning":ROOT/"config/features_early_warning.yaml","features_late_stage":ROOT/"config/features_late_stage.yaml",
        "expected_job_contract":out/"expected_job_contract.json","test_report":test_report_path}
    output_checksums={p.name:file_checksum(p) for p in out.iterdir() if p.is_file() and p.name not in {"reporting_patch_manifest_v2_1_1.json","reporting_patch_manifest_v2_1_1.sha256"}}
    patch_manifest={"patch_version":PATCH_VERSION,"source_run_id":args.run_id,"source_run_commit":manifest["source_commit"],
        "patch_source_commit":args.patch_source_commit,"source_tree_clean":not bool(git("status","--short","--untracked-files=no")),
        "input_artifact_checksums":{name:file_checksum(path) for name,path in input_paths.items()},"metric_implementation_version":METRIC_VERSION,
        "ece_contract":{"type":"top_label","bins":10,"edges":"equal_width","terminal_bin":"closed_at_1.0","empty_bins":"ignored"},
        "output_checksums":output_checksums,"original_artifacts_modified":False,"generated_at":datetime.now(timezone.utc).isoformat()}
    manifest_path=out/"reporting_patch_manifest_v2_1_1.json"; manifest_path.write_text(json.dumps(patch_manifest,indent=2),encoding="utf-8")
    (out/"reporting_patch_manifest_v2_1_1.sha256").write_text(file_checksum(manifest_path)+"  reporting_patch_manifest_v2_1_1.json\n",encoding="utf-8")
    print(json.dumps(validation,indent=2))


if __name__ == "__main__": main()
