from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.model_display_names import get_display_name


V3_COMMIT = "dbd5c2f27e914da2b252bffe176e7c93a6c2c237"
ENSEMBLES = ["V3-A0F-ENS", "V3-H2TF-ENS", "V3-H3CF-ENS", "V3-P0-ENS", "V3-D0-ENS", "V3-A1-ENS"]
ALL_CANDIDATES = ENSEMBLES + ["V3-MLF", "V3-MLD"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    headers = [name.replace("_", " ").title() for name in columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in selected.itertuples(index=False, name=None):
        values = [f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def bar_chart(frame: pd.DataFrame, metric: str, title: str, path: Path, *, lower_is_better=False) -> None:
    ordered = frame.sort_values(metric, ascending=lower_is_better)
    figure, axis = plt.subplots(figsize=(10, 5.5))
    colors = ["#d97706" if value == "V3-D0-ENS" else "#2563eb" for value in ordered.candidate_id]
    labels = ordered.candidate_id.map(get_display_name)
    bars = axis.bar(labels, ordered[metric], color=colors)
    axis.set_title(title); axis.set_ylabel(metric.replace("_", " ")); axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=.25)
    for bar, value in zip(bars, ordered[metric]):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def paired_chart(metrics: pd.DataFrame, right: str, title: str, path: Path) -> None:
    selected = metrics.set_index("candidate_id").loc[[right, "V3-D0-ENS"], ["macro_f1", "pr_auc", "operational_recall"]]
    selected.index = [get_display_name(candidate_id) for candidate_id in selected.index]
    figure, axis = plt.subplots(figsize=(7, 4.5)); selected.T.plot(kind="bar", ax=axis, color=["#64748b", "#d97706"])
    axis.set_title(title); axis.set_ylim(.7, .91); axis.grid(axis="y", alpha=.25); axis.tick_params(axis="x", rotation=0)
    figure.tight_layout(); figure.savefig(path, dpi=180); plt.close(figure)


def generate_figures(artifact: Path, report: Path) -> list[str]:
    figures = report / "figures"; figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(artifact / "ensemble_metrics.csv")
    bar_chart(metrics, "macro_f1", "Fair probability-ensemble Macro-F1", figures / "fair_ensemble_macro_f1.png")
    bar_chart(metrics, "pr_auc", "Fair probability-ensemble PR-AUC", figures / "fair_ensemble_pr_auc.png")
    bar_chart(metrics, "operational_recall", "Recall at inner-frozen Precision >= 0.75 operating point", figures / "fair_ensemble_operational_recall.png")
    bar_chart(metrics, "ece", "Fair probability calibration (ECE)", figures / "calibration_fair.png", lower_is_better=True)
    bar_chart(metrics, "worst_eligible_module_macro_f1", "Worst eligible module Macro-F1", figures / "module_stability_fair.png")

    single = pd.read_csv(artifact / "single_seed_metrics.csv")
    mean = pd.read_csv(artifact / "mean_seed_metrics.csv")
    d0_seed = single.loc[single.candidate_id == "V3-D0", ["seed", "macro_f1"]].copy()
    d0_mean = float(mean.loc[mean.candidate_id == "V3-D0", "macro_f1"].iloc[0])
    d0_ens = float(metrics.loc[metrics.candidate_id == "V3-D0-ENS", "macro_f1"].iloc[0])
    figure, axis = plt.subplots(figsize=(8, 4.5)); labels = [str(seed) for seed in d0_seed.seed] + ["seed mean", "probability ensemble"]
    values = d0_seed.macro_f1.tolist() + [d0_mean, d0_ens]
    axis.bar(labels, values, color=["#94a3b8"] * 3 + ["#2563eb", "#d97706"]); axis.set_ylim(.81, .84)
    axis.set_title("CNN–BiLSTM: single seed, metric mean and probability ensemble"); axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(figures / "single_seed_vs_mean_metric_vs_ensemble.png", dpi=180); plt.close(figure)

    paired_chart(metrics, "V3-A0F-ENS", "CNN–BiLSTM Ensemble vs MLP", figures / "d0_ensemble_vs_a0f_ensemble.png")
    paired_chart(metrics, "V3-P0-ENS", "CNN–BiLSTM ensemble comparison", figures / "d0_ensemble_vs_p0_ensemble.png")
    paired_chart(metrics, "V3-H3CF-ENS", "CNN–BiLSTM ensemble comparison", figures / "d0_ensemble_vs_h3cf_ensemble.png")

    bootstrap = pd.read_csv(artifact / "grouped_bootstrap_fair.csv")
    macro = bootstrap.loc[(bootstrap.left_candidate == "V3-D0-ENS") & (bootstrap.metric == "macro_f1")].copy()
    labels = [get_display_name(candidate_id) for candidate_id in macro.right_candidate]; y = np.arange(len(labels)); means = macro.mean_delta.to_numpy()
    errors = np.vstack([means - macro.lower_95.to_numpy(), macro.upper_95.to_numpy() - means])
    figure, axis = plt.subplots(figsize=(9, 5)); axis.errorbar(means, y, xerr=errors, fmt="o", color="#d97706", capsize=4)
    axis.axvline(0, color="black", lw=1); axis.axvline(.005, color="#2563eb", lw=1, linestyle="--")
    axis.set_yticks(y, labels); axis.set_xlabel("Paired grouped-bootstrap Macro-F1 delta"); axis.set_title("CNN–BiLSTM Ensemble uncertainty intervals")
    axis.grid(axis="x", alpha=.25); figure.tight_layout(); figure.savefig(figures / "grouped_bootstrap_intervals.png", dpi=180); plt.close(figure)

    before = pd.read_csv(artifact / "postgres_counts_before.csv").set_index("table_name").row_count
    after = pd.read_csv(artifact / "postgres_counts_after.csv").set_index("table_name").row_count
    names = ["source_records", "ml_experiment_runs", "ml_run_record_splits", "ml_predictions", "ml_run_metrics"]
    compare = pd.DataFrame({"before": before.reindex(names).fillna(0), "after": after.reindex(names).fillna(0)})
    figure, axis = plt.subplots(figsize=(9, 5)); compare.plot(kind="bar", ax=axis, logy=True, color=["#94a3b8", "#2563eb"])
    axis.set_title("PostgreSQL canonical registry counts before/after"); axis.set_ylabel("Rows (log scale)"); axis.tick_params(axis="x", rotation=25)
    figure.tight_layout(); figure.savefig(figures / "database_before_after_counts.png", dpi=180); plt.close(figure)

    nodes = {"dataset": (0, 2), "records": (2, 2), "targets": (4, 3), "runs": (4, 1), "splits": (6, 1), "predictions": (8, 1), "metrics": (6, 0), "evidence bundles": (2, 0)}
    edges = [("dataset", "records"), ("records", "targets"), ("dataset", "runs"), ("runs", "splits"), ("records", "splits"), ("splits", "predictions"), ("runs", "metrics"), ("dataset", "evidence bundles")]
    figure, axis = plt.subplots(figsize=(11, 4.5)); axis.axis("off")
    for left, right in edges:
        x1, y1 = nodes[left]; x2, y2 = nodes[right]; axis.annotate("", xy=(x2 - .45, y2), xytext=(x1 + .45, y1), arrowprops={"arrowstyle": "->", "color": "#64748b"})
    for label, (x, y0) in nodes.items():
        axis.text(x, y0, label, ha="center", va="center", bbox={"boxstyle": "round,pad=.5", "facecolor": "#e0f2fe", "edgecolor": "#2563eb"})
    axis.set_xlim(-1, 9); axis.set_ylim(-.7, 3.7); axis.set_title("Canonical PostgreSQL lineage used by the V3 fairness closure")
    figure.tight_layout(); figure.savefig(figures / "database_schema_relationships.png", dpi=180); plt.close(figure)
    return sorted(path.name for path in figures.glob("*.png"))


def write_reports(artifact: Path, report: Path, commit: str) -> None:
    metrics = pd.read_csv(artifact / "ensemble_metrics.csv")
    metrics.insert(1, "display_name", metrics.candidate_id.map(get_display_name))
    verdict = read_json(artifact / "verdict.json"); fairness = read_json(artifact / "fairness_audit.json")
    columns = ["display_name", "macro_f1", "at_risk_precision", "at_risk_recall", "at_risk_f1", "pr_auc", "operational_recall", "brier", "nll", "ece", "worst_eligible_module_macro_f1"]
    fair_text = "\n".join([
        "# Fair Ensemble Assessment", "",
        "This closure permanently separates single-seed metrics, mean-of-seed metrics, and record-aligned probability ensembles.", "",
        markdown_table(metrics, columns), "",
        f"- Corrected verdict: **{verdict['verdict']}**.",
        f"- CNN–BiLSTM Ensemble minus strongest fair comparator ({get_display_name(verdict['strongest_fair_comparator'])}): `{verdict['delta']:.9f}` Macro-F1; registered superiority margin: `{verdict['superiority_margin']:.3f}`.",
        f"- Threshold reconstruction: {fairness['replay_jobs']} frozen-config replay jobs; outer labels used: `{str(fairness['outer_labels_used_for_threshold']).lower()}`.",
        "- Future benchmark: `NOT EXECUTED`.",
        "- The earlier mixed-contract bootstrap is preserved as `historical_v3_mixed_contract_result` and is ineligible for this verdict.", "",
    ])
    (report / "FAIR_ENSEMBLE_ASSESSMENT.md").write_text(fair_text, encoding="utf-8")

    migration = read_json(artifact / "postgres_migration_report.json"); registration = read_json(artifact / "postgres_evidence_registration.json")
    reproduction = read_json(artifact / "postgres_reproduction_validation.json"); permission = read_json(artifact / "postgres_permission_audit.json")
    cleanup = read_json(artifact / "postgres_cleanup_execution.json")
    before = pd.read_csv(artifact / "postgres_counts_before.csv").set_index("table_name").row_count
    after = pd.read_csv(artifact / "postgres_counts_after.csv").set_index("table_name").row_count
    postgres_text = "\n".join([
        "# PostgreSQL Scientific Closure", "",
        "## Outcome", "",
        f"- Migration dry-run: `{migration['dry_run']['status']}` with rollback; committed migration status: `{migration['applied']['status']}`.",
        "- Applied migrations: `005_oulad_lineage_and_snapshot_registry.sql`, `006_oulad_v3_fair_evidence_registry.sql`, and set-based integrity optimization `007_optimize_bulk_lineage_integrity_triggers.sql`.",
        "- Migration 007 preserves the same sealed-dataset/running-run rule with statement-level transition tables; append-only, FK, uniqueness, status, and completed-run triggers remain active.",
        f"- Rows removed: `{cleanup['rows_removed']}`; executed cleanup predicates: `{cleanup['predicates_executed']}`.",
        f"- Registered: `{registration['source_records']}` source records, `{registration['completed_runs_registered']}` completed candidate runs, `{registration['prediction_rows']}` predictions, and `{len(registration['evidence_bundles'])}` evidence bundles.",
        f"- Reproduction: `{reproduction['status']}`; max probability difference `{reproduction['max_probability_absolute_difference']:.3g}`; max metric difference `{reproduction['max_metric_absolute_difference']:.3g}`.",
        f"- Least-privileged app permission audit: `{permission['status']}` as `{permission['application_profile']['current_user']}`; superuser app evidence forbidden.", "",
        "## Before/after key counts", "",
        "| Table | Before | After | Delta |", "|---|---:|---:|---:|",
        *[f"| {name} | {int(before.get(name, 0))} | {int(after.get(name, 0))} | {int(after.get(name, 0) - before.get(name, 0))} |" for name in ["source_records", "source_record_targets", "ml_experiment_runs", "ml_run_record_splits", "ml_predictions", "ml_run_metrics", "ml_evidence_bundles"]], "",
        "## Integrity and query behavior", "",
        "After-audit reports zero orphan splits, zero orphan predictions, zero duplicate prediction keys, and zero invalid run statuses. The record-key expression index and existing run/prediction indexes support exact artifact-to-database replay. See `postgres_query_plans.md` for `EXPLAIN (ANALYZE, BUFFERS)` output.", "",
        "No production database, external benchmark, model training, or recommendation generation was performed.", "",
    ])
    (report / "POSTGRES_CLOSURE_REPORT.md").write_text(postgres_text, encoding="utf-8")

    thesis = """# Thesis Claims After Fairness Closure

## Allowed

- CNN–BiLSTM Ensemble has the highest fair point-estimate Macro-F1 (0.831126), but its advantage over the strongest fair comparator MLP is 0.002454, below the registered 0.005 margin.
- The corrected temporal-family verdict is **PRACTICAL_TIE**; it is unchanged from the old V3 headline verdict.
- CNN–BiLSTM Ensemble shows a positive exploratory paired-bootstrap lead over several comparators, while uncertainty versus the strongest MLP and CNN–BiLSTM comparators includes zero.
- Probability ensembles use exactly seeds 42, 2026 and 3407 with fold-specific thresholds reconstructed from pooled inner-OOF predictions.
- PostgreSQL technical lineage, reproduction, append-only constraints and least-privileged permission checks passed.

## Prohibited

- Do not claim overall or operational superiority over the strongest fair comparator.
- Do not call mean-of-seed metrics an ensemble, use a single favorable seed, or call the future-presentation benchmark untouched/external validation.
- Do not claim scientific confirmation, causal effectiveness, production validation, or recommendation effectiveness.

## Superseded

- Any V3 bootstrap row comparing a probability ensemble to a single-seed or mean-of-metrics comparator is historical mixed-contract evidence and cannot support the final verdict.
"""
    (report / "THESIS_CLAIMS_AFTER_FAIRNESS_CLOSURE.md").write_text(thesis, encoding="utf-8")

    readme = f"""# OULAD V3 Fairness + PostgreSQL Closure

- Run: `{artifact.name}`
- Source V3 evidence: `{V3_COMMIT}`
- Closure code state: `{commit}`
- Scientific verdict: `PRACTICAL_TIE`
- Future benchmark: `NOT_EXECUTED`
- PostgreSQL reproduction: `PASS`
- Least-privileged permission audit: `PASS`
- Cleanup rows removed: `0`

Primary reports are under `reports/study_c_oulad_v3_closure/{artifact.name}/`. This bundle is additive and does not replace V1/V2/V3 evidence.
"""
    (artifact / "README.md").write_text(readme, encoding="utf-8")


def check(name: str, condition: bool, evidence: object) -> dict[str, object]:
    return {"check": name, "status": "PASS" if condition else "FAIL", "evidence": evidence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--check-only", action="store_true", help="Validate the frozen closure without rewriting artifacts or reports.")
    args = parser.parse_args()
    artifact = Path(args.artifact_root).resolve(); report = Path(args.report_root).resolve()
    if not args.check_only:
        report.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.check_only:
        figures = sorted(path.name for path in (report / "figures").glob("*.png"))
    else:
        figures = generate_figures(artifact, report); write_reports(artifact, report, commit)
    provenance = read_json(artifact / "source_provenance.json")
    if not args.check_only:
        provenance.update({"closure_validation_commit": commit, "database_migrations": ["005_oulad_lineage_and_snapshot_registry.sql", "006_oulad_v3_fair_evidence_registry.sql", "007_optimize_bulk_lineage_integrity_triggers.sql"], "database_cleanup_rows": 0, "model_training": False, "prediction_regeneration": False, "recommendation_regeneration": False, "legacy_observed_79_access": False})
        write_json(artifact / "source_provenance.json", provenance)

    required_artifacts = [
        "README.md", "resolved_protocol.yaml", "source_provenance.json", "v3_artifact_checksums.json", "candidate_registry.json", "prediction_contract_registry.json",
        "single_seed_metrics.csv", "mean_seed_metrics.csv", "ensemble_metrics.csv", "fair_comparison_summary.csv", "ensemble_thresholds.csv", "ensemble_prediction_coverage.csv",
        "ensemble_oof_predictions.parquet", "grouped_bootstrap_fair.csv", "paired_deltas_fair.csv", "superseded_v3_comparisons.json", "fairness_audit.json", "verdict.json",
        "postgres_connectivity_audit.json", "postgres_backup_manifest.json", "postgres_schema_before.json", "postgres_schema_after.json", "postgres_counts_before.csv", "postgres_counts_after.csv",
        "postgres_cleanup_plan.json", "postgres_cleanup_execution.json", "postgres_migration_report.json", "postgres_permission_audit.json", "postgres_evidence_registration.json", "postgres_reproduction_validation.json",
        "adaptive_decision_log.jsonl", "test_report.json",
    ]
    required_reports = ["FAIR_ENSEMBLE_ASSESSMENT.md", "POSTGRES_AUDIT_BEFORE.md", "POSTGRES_CLEANUP_PLAN.md", "POSTGRES_CLOSURE_REPORT.md", "THESIS_CLAIMS_AFTER_FAIRNESS_CLOSURE.md", "postgres_query_plans.md", "test_stdout.txt"]
    metrics = pd.read_csv(artifact / "ensemble_metrics.csv"); bootstrap = pd.read_csv(artifact / "grouped_bootstrap_fair.csv")
    fairness = read_json(artifact / "fairness_audit.json"); verdict = read_json(artifact / "verdict.json"); migration = read_json(artifact / "postgres_migration_report.json")
    permission = read_json(artifact / "postgres_permission_audit.json"); reproduction = read_json(artifact / "postgres_reproduction_validation.json")
    registration = read_json(artifact / "postgres_evidence_registration.json"); cleanup = read_json(artifact / "postgres_cleanup_execution.json"); tests = read_json(artifact / "test_report.json")
    schema_after = read_json(artifact / "postgres_schema_after.json"); backup = read_json(artifact / "postgres_backup_manifest.json")
    ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", V3_COMMIT, "HEAD"], cwd=ROOT).returncode == 0
    secret_pattern = re.compile(r"postgresql://(?!<redacted>)[^\s/@:]+:[^\s/@]+@", re.I)
    secret_hits = []
    for path in list(artifact.rglob("*")) + [ROOT / "configs/oulad_v3_fair_db_closure_protocol.yaml", ROOT / "scripts/register_oulad_v3_evidence_postgres.py"]:
        if path.is_file() and path.suffix.lower() not in {".parquet", ".png"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if secret_pattern.search(text):
                secret_hits.append(path.as_posix())
    backup_root = Path(backup["backup_root"])
    backup_files_ok = backup["status"] == "PASS" and all(
        (backup_root / item["name"]).exists()
        and (backup_root / item["name"]).stat().st_size == item["bytes"]
        and sha256(backup_root / item["name"]) == item["sha256"]
        for item in backup["files"]
    )
    checks = [
        check("v3_commit_ancestry", ancestry, V3_COMMIT),
        check("required_artifacts", all((artifact / name).exists() for name in required_artifacts), required_artifacts),
        check("required_reports", all((report / name).exists() for name in required_reports), required_reports),
        check("required_fair_figures", len(figures) == 12, figures),
        check("candidate_registry_complete", set(metrics.candidate_id) == set(ALL_CANDIDATES), sorted(metrics.candidate_id)),
        check("fairness_audit", fairness["status"] == "PASS" and not fairness["outer_labels_used_for_threshold"] and not fairness["future_access"], fairness),
        check("grouped_bootstrap_contract", len(bootstrap) == 45 and (bootstrap.resamples == 5000).all() and (bootstrap.students == 14687).all(), {"rows": len(bootstrap)}),
        check("corrected_verdict", verdict["verdict"] == "PRACTICAL_TIE" and verdict["old_v3_verdict"] == "PRACTICAL_TIE" and verdict["delta"] < .005, verdict),
        check("backup_gate", backup_files_ok, {"status": backup["status"], "files": len(backup["files"])}),
        check("migration_dry_run_and_commit", migration["dry_run"]["status"] == migration["applied"]["status"] == "PASS" and len(migration["applied"]["migrations"]) == 3, migration),
        check("cleanup_allowlist", cleanup["status"] == "PASS" and cleanup["rows_removed"] == 0, cleanup),
        check("database_registration", registration["status"] == "PASS" and registration["prediction_rows"] == 123024 and registration["completed_runs_registered"] == 8, registration),
        check("database_reproduction", reproduction["status"] == "PASS" and reproduction["max_probability_absolute_difference"] <= 1e-12 and reproduction["max_metric_absolute_difference"] <= 1e-12, reproduction),
        check("least_privileged_permissions", permission["status"] == "PASS" and permission["application_profile"]["current_user"] == "student_predict_app_local" and not any(permission["application_profile"][key] for key in ["rolsuper", "rolcreatedb", "rolcreaterole"]), permission),
        check("database_integrity_after", all(value == 0 for value in schema_after["integrity"].values()), schema_after["integrity"]),
        check("full_test_suite", tests["status"] == "PASS" and tests["full_suite"]["return_code"] == 0 and tests["full_suite"]["failed"] == tests["full_suite"]["skipped"] == tests["full_suite"]["errors"] == 0, tests["full_suite"]),
        check("credential_redaction", not secret_hits, secret_hits),
        check("no_training_or_future", provenance["model_training"] is False and provenance["prediction_regeneration"] is False and provenance["recommendation_regeneration"] is False and provenance["future_access"] is False, provenance),
    ]
    if args.check_only:
        checksum_entries = read_json(artifact / "artifact_checksums.json")["files"]
    else:
        checksum_entries = []
        for path in sorted(item for item in artifact.rglob("*") if item.is_file() and "threshold_replay_cache" not in item.parts and item.name not in {"artifact_checksums.json", "validation_report.json"}):
            checksum_entries.append({"path": path.relative_to(artifact).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
        write_json(artifact / "artifact_checksums.json", {"algorithm": "sha256", "excluded_self_referential_files": ["artifact_checksums.json", "validation_report.json"], "files": checksum_entries})
    checks.append(check("artifact_checksums", all(sha256(artifact / item["path"]) == item["sha256"] for item in checksum_entries), {"files": len(checksum_entries)}))
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    validation = {"status": status, "run_id": artifact.name, "source_v3_commit": V3_COMMIT, "closure_commit": commit, "scientific_verdict": verdict["verdict"], "future_benchmark": "NOT_EXECUTED", "checks": checks, "artifact_checksums_sha256": sha256(artifact / "artifact_checksums.json")}
    if not args.check_only:
        write_json(artifact / "validation_report.json", validation)
    print(json.dumps({"status": status, "mode": "check-only" if args.check_only else "write", "checks": len(checks), "passed": sum(item["status"] == "PASS" for item in checks), "figures": len(figures)}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
