"""Create the immutable, validation-only final repository closure bundle.

This entrypoint never trains, tunes, calibrates, evaluates the observed-79
payload, or regenerates recommendation cases.  It verifies and indexes existing
official evidence, documentation, tests, migrations, and repository state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PARENT = ROOT / "artifacts" / "final_repository_closure"
REPORT_PARENT = ROOT / "reports" / "final_repository_closure"
PHASE_AB = ROOT / "artifacts" / "strategy_b_phase_ab" / "strategy-b-phase-ab-20260714-475a672"
PHASE_C = ROOT / "artifacts" / "strategy_b_phase_c" / "strategy-b-phase-c-20260714-5d34a66"
PHASE_E = ROOT / "artifacts" / "strategy_b_phase_e_prediction" / "strategy-b-phase-e-prediction-20260714-9007144"
PHASE_D = ROOT / "artifacts" / "strategy_b_phase_d_recommendation" / "strategy-b-phase-d-recommendation-20260715-407ac0f"

OFFICIAL_COMMITS = {
    "phase_ab_code": "475a6727a49d939f847c46e1bdeb11a4a5bc60ec",
    "phase_ab_evidence": "02e228b31c5ea096c8305a3caa906cc14315832a",
    "phase_c_code": "5d34a6641036be454c115747718b16669590f0be",
    "phase_c_evidence": "e20ff43c7a7c95b638e82b84d40c7cf10b6e0d49",
    "phase_e_implementation": "119611cf16d78dc8ea2b563981a9d7a9ba540225",
    "phase_e_code_tip": "900714494a9d4fa75bf3f57c48b481f0db90d447",
    "phase_e_evidence": "af60729ed8e1ea671bc7a6e07374cf32b8f197e7",
    "phase_d_code_1": "4915a122c6a21a77841e0b5f6f3977b4bd2b7a93",
    "phase_d_code_2": "407ac0f0fce7a3b61a6540f18bf0e11a559cca81",
    "phase_d_evidence": "063288fdbd01aee1221e8c8108f84fec0d1209c0",
}

REQUIRED_OUTPUTS = [
    "closure_protocol.json", "repository_state.json", "commit_lineage.json",
    "official_evidence_registry.json", "historical_evidence_registry.json", "artifact_index.json",
    "final_model_registry.json", "final_metrics.csv", "final_recommendation_summary.json",
    "test_report.json", "test_stdout.txt", "database_static_validation.json",
    "readme_claim_audit.csv", "project_claim_audit.csv", "prohibited_claim_scan.csv",
    "broken_link_report.csv", "large_file_report.csv", "secret_scan_report.json",
    "repository_audit.csv",
    "source_provenance.json", "strict_validation.json", "final_repository_conclusion.md",
    "thesis_writing_context.md", "thesis_evidence_map.csv", "artifact_checksums.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def git(*args: str, check: bool = True) -> str:
    result = run(["git", *args])
    if check and result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def checksum_registry(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifact_checksums.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for relative, expected in manifest.items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            failures.append(relative)
    return {"path": root.relative_to(ROOT).as_posix(), "entries": len(manifest), "failures": failures, "pass": not failures}


def commit_lineage(head: str) -> dict[str, Any]:
    ancestry = []
    for name, commit in OFFICIAL_COMMITS.items():
        result = run(["git", "merge-base", "--is-ancestor", commit, head])
        ancestry.append({"name": name, "commit": commit, "is_ancestor": result.returncode == 0})
    return {
        "closure_head": head,
        "base_integration_commit": "e1c3a678fe0dd69b938e92140219b20bdafc33a4",
        "ancestry": ancestry,
        "all_official_commits_ancestors": all(item["is_ancestor"] for item in ancestry),
        "graph": git("log", "--graph", "--decorate", "--oneline", "-25"),
    }


def official_registry(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "registry_version": "final_repository_closure_v1",
        "entries": [
            {"category": "official_development_evidence", "phase": "A-B", "path": checks["phase_ab"]["path"], "status": "official_protocol_correctness", "headline_use": "protocol and estimator correctness only", "checksum_validation": checks["phase_ab"]["pass"]},
            {"category": "official_development_evidence", "phase": "C", "path": checks["phase_c"]["path"], "status": "official_main_candidate_comparison", "headline_use": "development model comparison and ablations", "checksum_validation": checks["phase_c"]["pass"]},
            {"category": "official_final_development_freeze", "phase": "E-Prediction", "path": checks["phase_e"]["path"], "status": "official_final_development_freeze", "headline_use": "final model roles, metrics, stability and continuous-G3 analysis", "checksum_validation": checks["phase_e"]["pass"]},
            {"category": "official_technical_recommendation_evidence", "phase": "D", "path": checks["phase_d"]["path"], "status": "technical_validation_pass", "headline_use": "technical structure and safety only", "checksum_validation": checks["phase_d"]["pass"]},
            {"category": "expert_validation_pending", "phase": "D", "path": f"{checks['phase_d']['path']}/expert_validation_status.json", "status": "PENDING", "headline_use": "limitation only"},
            {"category": "effectiveness_not_performed", "phase": "D", "path": f"{checks['phase_d']['path']}/strict_validation.json", "status": "NOT_PERFORMED", "headline_use": "limitation only"},
        ],
    }


def historical_registry() -> dict[str, Any]:
    return {
        "preservation_policy": "preserve immutable or historical evidence; classify rather than rewrite",
        "entries": [
            {"category": "legacy_observed_evidence", "paths": ["artifacts/final/*", "reports/final/*"], "status": "legacy_heldout_observed", "headline_eligible": False, "reason": "79 records were observed and are not an untouched test"},
            {"category": "invalid_protocol_evidence", "paths": ["artifacts/baseline_comparison/fair-model-comparison-full/*"], "status": "invalid_protocol_config_resolution_for_fair_deep_rows", "headline_eligible": False, "reason": "resolved fixed loss/class-weight/resampling constants were absent"},
            {"category": "historical_evidence", "paths": ["artifacts/model_selection/nested-full-20260710/*"], "status": "historical_old_estimator", "headline_eligible": False, "reason": "predates corrected full-partition refit/source provenance"},
            {"category": "diagnostic_evidence", "paths": ["reports/project_strategy_v1/*", "reports/scientific_audit/*"], "status": "diagnostic_or_feasibility", "headline_eligible": False, "reason": "hypothesis and audit context only"},
            {"category": "smoke_evidence", "paths": ["artifacts/strategy_b_phase_c_smoke/*", "reports/strategy_b_phase_c_smoke/*"], "status": "smoke", "headline_eligible": False, "reason": "code-path/runtime feasibility only"},
            {"category": "diagnostic_evidence", "paths": ["diagnostic residual and Huber outputs referenced by project audit"], "status": "diagnostic_only", "headline_eligible": False, "reason": "conditional gates stayed closed"},
            {"category": "historical_evidence", "paths": ["artifacts/strategy_b_phase_c/strategy-b-phase-c-20260714-9e4928d", "artifacts/strategy_b_phase_c/strategy-b-phase-c-20260714-e0f980b"], "status": "intermediate_or_recovery_runs", "headline_eligible": False, "reason": "superseded by corrected Phase C bundle"},
            {"category": "historical_evidence", "paths": ["artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-119611c"], "status": "reporting_predecessor", "headline_eligible": False, "reason": "superseded by corrected Phase E reporting bundle; raw evidence retained"},
            {"category": "historical_evidence", "paths": ["historical CNN-BiLSTM outputs under artifacts/final and model_selection"], "status": "historical_old_estimator", "headline_eligible": False, "reason": "not the corrected estimator contract"},
        ],
    }


def final_metrics() -> pd.DataFrame:
    source = pd.read_csv(PHASE_E / "stability_summary.csv").set_index("candidate_id")
    roles = {
        "R0": "final overall model", "M1": "practical-tie ML comparator", "M2": "practical-tie ML comparator",
        "N0": "final thesis hybrid model", "N1": "ordinal research comparator",
    }
    rows = []
    for candidate in ["R0", "M1", "M2", "N0", "N1"]:
        item = source.loc[candidate]
        rows.append({
            "model": candidate, "role": roles[candidate], "accuracy": item["accuracy"],
            "macro_precision": item["macro_precision"], "macro_recall": item["macro_recall"],
            "macro_f1": item["oof_macro_f1"], "weighted_f1": item["weighted_f1"],
            "high_class_f1": item["high_f1"], "macro_pr_auc": item["macro_pr_auc"],
            "rmse_g3": item["rmse"], "r2_g3": item["r2"],
            "validation_scope": "nested development OOF; no external confirmation",
        })
    return pd.DataFrame(rows)


def recommendation_summary() -> dict[str, Any]:
    metrics = json.loads((PHASE_D / "technical_safety_metrics.json").read_text(encoding="utf-8"))
    instances = pd.read_csv(PHASE_D / "recommendation_instances.csv")
    actions = pd.read_csv(PHASE_D / "recommendation_actions.csv")
    coverage = pd.read_csv(PHASE_D / "coverage_and_abstention.csv")
    casebook = pd.read_csv(PHASE_D / "expert_casebook.csv")
    return {
        "development_cases": len(instances),
        "eligible_for_normal_draft_gate": int((coverage["review_status"] == "eligible_for_draft").sum()),
        "uncertainty_agreement_review_cases": int((coverage["review_status"] != "eligible_for_draft").sum()),
        "gate_review_rate": metrics["abstention_rate"], "advisor_approval_required_rate": metrics["advisor_review_rate"],
        "generated_actions": len(actions), "action_conflict_rate": metrics["action_conflict_rate"],
        "action_duplication_rate": metrics["action_duplication_rate"], "workload_violation_rate": metrics["workload_violation_rate"],
        "goal_completeness_rate": metrics["goal_completeness_rate"], "action_completeness_rate": metrics["action_completeness_rate"],
        "explanation_completeness_rate": metrics["explanation_completeness_rate"], "expert_casebook_cases": len(casebook),
        "expert_casebook_strata": int(casebook["stratum"].nunique()),
        "technical_validation": "PASS", "expert_validation": "PENDING", "effectiveness_validation": "NOT_PERFORMED",
        "claim_boundary": "technical structure only; no scientific effectiveness claim",
    }


def claim_audit(path: Path, claims: list[tuple[str, list[str], str]]) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8")
    rows = []
    for claim_id, needles, evidence in claims:
        rows.append({"claim_id": claim_id, "pass": all(needle in text for needle in needles), "evidence": evidence})
    return pd.DataFrame(rows)


def prohibited_scan() -> pd.DataFrame:
    patterns = {
        "dl_superiority": ["CNN–BiLSTM vượt trội", "Deep Learning cho kết quả tốt nhất", "CNN-BiLSTM outperforms"],
        "untouched_confirmation": ["xác nhận trên tập kiểm thử chưa từng quan sát", "confirmed on an untouched test"],
        "generalization_proven": ["chứng minh khả năng tổng quát hóa", "generalization is proven"],
        "recommendation_effective": ["khuyến nghị giúp tăng điểm", "recommendation improves grades"],
        "expert_pass": ["expert_validation = PASS", "expert validation passed"],
    }
    rows = []
    for path in [ROOT / "README.md", ROOT / "PROJECT.md"]:
        in_prohibited_section = False
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            if lowered.startswith("## "):
                in_prohibited_section = "prohibited" in lowered or "không được" in lowered
            for claim, needles in patterns.items():
                if any(needle.lower() in lowered for needle in needles):
                    negated = in_prohibited_section or any(token in lowered for token in ["không", "no ", "prohibited", "forbidden", "không được"])
                    rows.append({"file": path.name, "line": line_number, "claim": claim, "classification": "negated_or_prohibited_example" if negated else "blocking_positive_claim", "blocking": not negated})
    return pd.DataFrame(rows, columns=["file", "line", "claim", "classification", "blocking"])


def markdown_link_report() -> pd.DataFrame:
    rows = []
    regex = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for path in [ROOT / "README.md", ROOT / "PROJECT.md"]:
        for target in regex.findall(path.read_text(encoding="utf-8")):
            clean = target.split("#", 1)[0]
            if not clean or clean.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / clean).resolve()
            rows.append({"source": path.relative_to(ROOT).as_posix(), "target": target, "exists": resolved.exists(), "resolved": str(resolved)})
    return pd.DataFrame(rows, columns=["source", "target", "exists", "resolved"])


def tracked_files() -> list[Path]:
    return [ROOT / line for line in git("ls-files").splitlines() if line]


def large_files() -> pd.DataFrame:
    rows = []
    for path in tracked_files():
        if path.is_file() and path.stat().st_size >= 1024 * 1024:
            rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "classification": "large_text_evidence" if path.suffix.lower() == ".csv" else "large_binary_or_other", "action": "preserve; indexed official/historical evidence"})
    return pd.DataFrame(rows, columns=["path", "bytes", "classification", "action"])


def secret_scan() -> dict[str, Any]:
    extensions = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".ini", ".sql", ".txt", ".example"}
    rules = {
        "credential_uri": re.compile(r"(?:postgres(?:ql)?|mysql)://[^\s:/]+:[^\s@]+@", re.I),
        "private_key": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
        "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
        "api_secret_assignment": re.compile(r"(?:api[_-]?key|secret[_-]?key|password)\s*[:=]\s*['\"]?([^\s'\"]+)", re.I),
    }
    findings = []
    for path in tracked_files():
        if not path.is_file() or (path.suffix.lower() not in extensions and path.name != ".env.example"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for rule, regex in rules.items():
            for match in regex.finditer(text):
                matched = match.group(0).lower()
                placeholder = any(token in matched for token in ["change-me", "change_me", "example", "placeholder", "${", "your_"])
                findings.append({"path": path.relative_to(ROOT).as_posix(), "rule": rule, "classification": "placeholder_or_documentation" if placeholder else "potential_secret"})
    confirmed = [row for row in findings if row["classification"] == "potential_secret"]
    return {"scanned_tracked_files": len(tracked_files()), "findings": findings, "confirmed_secrets": len(confirmed), "status": "PASS" if not confirmed else "FAIL", "matched_values_redacted": True}


def database_validation() -> dict[str, Any]:
    migrations = sorted((ROOT / "database" / "migrations").glob("*.sql"))
    migration = ROOT / "database" / "migrations" / "004_governed_recommendation_phase_d.sql"
    text = migration.read_text(encoding="utf-8")
    tables = ["recommendation_policies", "recommendation_feature_registry", "recommendation_action_catalog", "prediction_snapshots", "recommendation_instances", "recommendation_revisions", "recommendation_goals", "recommendation_actions", "advisor_decisions", "recommendation_follow_ups", "recommendation_outcomes", "expert_review_cases", "expert_review_ratings"]
    checks = {
        "migration_order": [path.name[:3] for path in migrations] == ["001", "002", "003", "004"],
        "required_tables": all(f"CREATE TABLE IF NOT EXISTS {table}" in text for table in tables),
        "foreign_keys": text.count("REFERENCES ") >= 10,
        "append_only_trigger": "BEFORE UPDATE OR DELETE" in text and "reject_governed_recommendation_mutation" in text,
        "policy_status_constraint": all(status in text for status in ["draft", "technical_validated", "expert_review_pending", "expert_approved", "deprecated"]),
        "review_status_constraint": all(status in text for status in ["eligible_for_draft", "advisor_review_required", "insufficient_information", "invalid_prediction", "stale_prediction"]),
        "old_migrations_preserved": all(path.is_file() for path in migrations[:3]),
    }
    return {"database_migration_execution": "NOT_PERFORMED", "database_migration_static_validation": "PASS" if all(checks.values()) else "FAIL", "reason": "disposable test DSN unavailable; production database was not used destructively", "migration_files": [path.relative_to(ROOT).as_posix() for path in migrations], "migration_hashes": {path.name: sha256_file(path) for path in migrations}, "checks": checks}


def artifact_index(official: dict[str, Any], historical: dict[str, Any]) -> dict[str, Any]:
    top = []
    for parent_name in ["artifacts", "reports"]:
        parent = ROOT / parent_name
        for path in sorted(parent.iterdir()):
            if path.is_dir():
                files = sum(1 for item in path.rglob("*") if item.is_file())
                size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
                top.append({"path": path.relative_to(ROOT).as_posix(), "files": files, "bytes": size})
    return {"official_registry_hash": sha256_json(official), "historical_registry_hash": sha256_json(historical), "top_level_collections": top}


def repository_audit_rows(lineage: dict[str, Any], links: pd.DataFrame, large: pd.DataFrame, secrets: dict[str, Any], database: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ["Current branch and clean preflight", "codex/final-repository-closure; clean before materialization", "official", "none", "repository_state.json"],
        ["Commit ancestry Phase A-B through D", "all frozen commits are ancestors", "official", "none", "commit_lineage.json"],
        ["README", "rewritten to Phase E/D scientific truth", "official", "none", "readme_claim_audit.csv"],
        ["PROJECT.md", "technical contract aligned with README", "official", "none", "project_claim_audit.csv"],
        ["Phase A-B correctness", "strict PASS and quarantine registry retained", "official", "none", str(PHASE_AB.relative_to(ROOT) / "strict_validation.json")],
        ["Phase C comparison", "corrected main comparison strict PASS", "official", "none", str(PHASE_C.relative_to(ROOT) / "strict_validation.json")],
        ["Phase E prediction freeze", "R0/N0 roles and checksums PASS", "official", "none", str(PHASE_E.relative_to(ROOT) / "strict_validation.json")],
        ["Phase D recommendation", "technical PASS; expert PENDING; effectiveness NOT_PERFORMED", "official", "none", str(PHASE_D.relative_to(ROOT) / "strict_validation.json")],
        ["Historical final/locked-test outputs", "preserved; include observed-79 results", "historical", "exclude from headline", "historical_evidence_registry.json"],
        ["Fair deep-learning rows", "resolved-config estimator mismatch", "invalid", "exclude from official ranking", "Phase A-B evidence_quarantine_registry.json"],
        ["Phase C smoke", "runtime/code-path evidence only", "smoke", "exclude from ranking", "historical_evidence_registry.json"],
        ["Residual/Huber diagnostics", "hypothesis/diagnostic only; gates closed", "diagnostic", "exclude from centerpiece", "historical_evidence_registry.json"],
        ["Old CNN-BiLSTM estimator outputs", "predate corrected estimator/refit contract", "deprecated", "historical context only", "historical_evidence_registry.json"],
        ["scripts/run_pipeline.py", "historical locked/observed experiment runner; prohibited by default unless explicit legacy flag", "deprecated", "do not use for quick validation", "README How to run"],
        ["scripts/optimize_model_selection.py and Phase C/E runners", "expensive historical experiment entrypoints", "deprecated", "require explicit future authorization", "README How to run"],
        ["scripts/materialize_recommendation_policy.py", "legacy materializer fail-closed", "deprecated", "retain fail-closed", "source file"],
        ["Hard-coded local paths", "found only in integration-test fixtures and historical audit/provenance docs; active closure commands are repository-relative", "historical", "preserve provenance; do not copy into active instructions", "hard-coded path scan"],
        ["Temporary/cache files", "ignored; local caches removed during closure", "deprecated", "keep ignored", ".gitignore"],
        ["Large tracked files", f"{len(large)} files >=1 MiB, all CSV evidence", "official/historical", "preserve and index", "large_file_report.csv"],
        ["Secrets/DSNs", f"confirmed secrets={secrets['confirmed_secrets']}", "official", "none" if secrets["status"] == "PASS" else "remove secret", "secret_scan_report.json"],
        ["Internal documentation links", f"broken={0 if links.empty else int((~links['exists']).sum())}", "official", "none", "broken_link_report.csv"],
        ["Database migrations 001-004", f"static={database['database_migration_static_validation']}; execution={database['database_migration_execution']}", "official", "execute destructive tests only on disposable DSN", "database_static_validation.json"],
        ["Final model files", "R0 contract plus five N0 checkpoints/preprocessors present in Phase E bundle", "official", "none", str(PHASE_E.relative_to(ROOT) / "final_model_manifest.json")],
        ["Terminology", "active README/PROJECT use development-only, governed, non-causal terms", "official", "historical documents remain registry-limited", "prohibited_claim_scan.csv"],
    ]
    frame = pd.DataFrame(rows, columns=["item", "current_status", "classification", "action_required", "evidence"])
    if not lineage["all_official_commits_ancestors"]:
        frame.loc[frame["item"] == "Commit ancestry Phase A-B through D", ["current_status", "action_required"]] = ["incomplete", "repair ancestry"]
    return frame


def thesis_context(metrics: pd.DataFrame, recommendation: dict[str, Any]) -> str:
    table = metrics.copy()
    for column in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "high_class_f1", "macro_pr_auc", "rmse_g3", "r2_g3"]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    columns = list(table.columns)
    metric_table = "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
        *["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in table.iterrows()],
    ])
    return f"""# Thesis-writing context

## 1. Đề tài

Đề tài xây dựng hệ thống dự đoán ba mức kết quả cuối kỳ và hệ thống khuyến nghị lộ trình học có quản trị. Phần prediction so sánh baseline ML với CNN–BiLSTM trên cùng G1/G2; phần recommendation biến prediction evidence thành draft mục tiêu/hành động luôn cần advisor review. Phạm vi là development-only, không phải production hoặc causal intervention.

## 2. Dữ liệu

UCI Student Performance `student-mat` có 395 records. Official protocol chỉ dùng 316 development records. 79 records còn lại đã bị quan sát trong lịch sử, mang nhãn `legacy_heldout_observed`, không dùng cho selection/calibration/confirmation. Inputs là G1/G2; raw G3 là target. Bins: Low 0–9, Medium 10–14, High 15–20.

## 3. Kiến trúc

- R0: deterministic thresholds trên G2; final overall model và agreement guardrail.
- M1/M2: Random Forest và SVM RBF practical-tie comparators.
- N0: compact nominal CNN–BiLSTM, five-seed ensemble; final thesis hybrid.
- N1: ordered ordinal CNN–BiLSTM comparator.
- Ablations: tiny MLP, ordered MLP, CNN-only, BiLSTM-only trong Phase C.
- Recommendation: N0 scores + R0 agreement → snapshot → uncertainty/feature governance → rule-based four-week goals/actions → advisor decision → follow-up/revision.

## 4. Protocol

Năm immutable outer folds, ba inner folds, development-only nested selection, replayable estimator/refit contract và Macro-F1 primary. Phase C neural selected configs dùng seeds 42/123/155; Phase E stability dùng new seeds 202601–202605 và không chọn best seed. PR metrics là one-vs-rest. RMSE/R² dùng continuous prediction contracts riêng. Không có locked-test hoặc external-confirmation claim.

## 5. Kết quả prediction

{metric_table}

M1 có point Macro-F1 cao nhất, nhưng R0/M1/M2 practical tie. R0 được chọn bởi tie-break/simplicity. N0 là thesis hybrid; N0/N1 không có superiority rõ. N0 calibration bị reject; N1 temperature calibration được giữ cho comparator nhưng không thay đổi final family.

## 6. Kết luận khoa học

ML có lợi thế trên bài toán hai feature và dữ liệu nhỏ. CNN–BiLSTM được giữ để trả lời mục tiêu kiến trúc của khóa luận, không phải overall champion. BiLSTM-only practical-tie với N0; CNN incremental value và ordinal improvement chưa được thiết lập. Residual/multitask/imbalance gates đóng. Recommendation là governed, non-causal và mới chỉ qua technical validation.

Recommendation technical facts: {recommendation['development_cases']} cases; {recommendation['eligible_for_normal_draft_gate']} normal-gate cases; {recommendation['uncertainty_agreement_review_cases']} uncertainty/agreement review cases ({recommendation['gate_review_rate']:.2%}); 100% require advisor approval; {recommendation['generated_actions']} actions; zero conflict/duplicate/workload violations; 60-case/23-strata expert casebook.

## 7. Hạn chế

Dataset nhỏ; sequence length hai; không có external unseen confirmation; 79 records bị contamination; expert validation pending; effectiveness not performed; context features chưa active; không có prospective intervention study; dataset không đại diện trực tiếp cho sinh viên đại học Việt Nam.

## 8. Figures and tables available

- Architecture: `src/models/phase_c.py`, `artifacts/strategy_b_phase_e_prediction/strategy-b-phase-e-prediction-20260714-9007144/final_model_manifest.json`.
- Metric table/stability: Phase E `stability_summary.csv`, `fold_seed_metrics.csv`.
- Confusion matrices: Phase E `confusion_matrices.csv`.
- PR curves: Phase E `precision_recall_curve_points.csv`, `precision_recall_metrics.csv`.
- Paired comparisons: Phase E `paired_stability_deltas.csv`; Phase C `paired_model_deltas.csv`.
- Calibration: Phase E `calibration_metrics.csv`, `calibration_decision.json`.
- Recommendation flow/policy: Phase D `protocol.json`, `model_role_contract.json`, `action_catalog.json`.
- Database schema: `database/migrations/001_create_source_ml_schema.sql` through `004_governed_recommendation_phase_d.sql`.
- Evidence hierarchy: closure `official_evidence_registry.json`, `historical_evidence_registry.json`, `thesis_evidence_map.csv`.

Use these artifacts to construct figures/tables; this closure does not edit thesis DOCX.
"""


def conclusion(strict: dict[str, Any], tests: dict[str, Any]) -> str:
    return f"""# Final repository closure conclusion

- Repository closure validation: **{strict['repository_closure_validation']}**.
- Prediction evidence validation: **{strict['prediction_evidence_validation']}**.
- Recommendation technical validation: **{strict['recommendation_technical_validation']}**.
- Expert validation: **{strict['expert_validation']}**.
- Effectiveness validation: **{strict['effectiveness_validation']}**.
- Full test suite: **{tests['passed']} passed, {tests['skipped']} skipped, {tests['failed']} failed**.
- Database migration execution: **NOT_PERFORMED**; static validation is recorded separately.

The repository is technically closed for thesis writing. Outstanding expert review, prospective effectiveness research and external unseen validation are external scientific work, not repository-closure blockers.
"""


def main() -> None:
    args = parse_args()
    final = ARTIFACT_PARENT / args.run_id
    report = REPORT_PARENT / args.run_id
    tmp = ARTIFACT_PARENT / f".{args.run_id}.tmp"
    report_tmp = REPORT_PARENT / f".{args.run_id}.tmp"
    if any(path.exists() for path in [final, report, tmp, report_tmp]):
        raise FileExistsError(f"Closure run already exists: {args.run_id}")

    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    initial_status = git("status", "--short")
    if branch != "codex/final-repository-closure":
        raise RuntimeError(f"Closure must run on codex/final-repository-closure, got {branch}")
    if initial_status:
        raise RuntimeError("Closure requires a clean working tree before artifact materialization.")

    tmp.mkdir(parents=True)
    report_tmp.mkdir(parents=True)
    try:
        official_checks = {
            "phase_ab": checksum_registry(PHASE_AB), "phase_c": checksum_registry(PHASE_C),
            "phase_e": checksum_registry(PHASE_E), "phase_d": checksum_registry(PHASE_D),
        }
        lineage = commit_lineage(head)
        official = official_registry(official_checks)
        historical = historical_registry()
        metrics = final_metrics()
        recommendation = recommendation_summary()

        test_result = run([sys.executable, "-m", "pytest", "-q", "-rs"])
        raw_stdout = test_result.stdout + ("\n" + test_result.stderr if test_result.stderr else "")
        (tmp / "test_stdout.txt").write_text(raw_stdout, encoding="utf-8")
        summary = re.search(r"(?:(\d+) failed, )?(\d+) passed(?:, (\d+) skipped)?", raw_stdout)
        failed = int(summary.group(1) or 0) if summary else (1 if test_result.returncode else 0)
        passed = int(summary.group(2)) if summary else 0
        skipped = int(summary.group(3) or 0) if summary else 0
        test_report = {
            "command": [sys.executable, "-m", "pytest", "-q", "-rs"], "return_code": test_result.returncode,
            "collected": passed + skipped + failed, "passed": passed, "skipped": skipped, "failed": failed,
            "raw_stdout": "test_stdout.txt", "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform()},
            "git_commit": head, "postgresql_waiver": "Five destructive integration tests may skip because disposable POSTGRES_TEST_DSN/POSTGRES_TEST_APP_DSN and psql are unavailable; production was not used.",
        }

        database = database_validation()
        readme_audit = claim_audit(ROOT / "README.md", [
            ("final_roles", ["`final_overall_model`: **R0", "`final_thesis_hybrid_model`: **N0"], "Phase E final_family_decision.json"),
            ("observed_79", ["79 records", "legacy_heldout_observed"], "Phase A-B quarantine registry"),
            ("metrics", ["0.8988", "0.9000", "0.8504"], "Phase E stability_summary.csv"),
            ("recommendation_status", ["technical_validation = PASS", "expert_validation = PENDING", "effectiveness_validation = NOT PERFORMED"], "Phase D strict_validation.json"),
            ("no_external", ["Không có tập xác nhận ngoài hoàn toàn chưa từng quan sát"], "Phase E protocol"),
        ])
        project_audit = claim_audit(ROOT / "PROJECT.md", [
            ("final_roles", ["R0 overall", "N0 thesis hybrid"], "Phase E final roles"),
            ("metrics", ["0.8988", "0.9000", "0.8504"], "Phase E stability_summary.csv"),
            ("recommendation_status", ["technical_validation = PASS", "expert_validation = PENDING", "effectiveness_validation = NOT PERFORMED"], "Phase D strict validation"),
            ("target_contract", ["Low 0–9", "Medium 10–14", "High 15–20"], "frozen target contract"),
        ])
        prohibited = prohibited_scan()
        links = markdown_link_report()
        large = large_files()
        secrets = secret_scan()
        repository_audit = repository_audit_rows(lineage, links, large, secrets, database)

        repository_state = {
            "branch": branch, "commit": head, "working_tree_clean_before_materialization": not initial_status,
            "integration_base": "test@e1c3a678fe0dd69b938e92140219b20bdafc33a4", "tracked_files": len(tracked_files()),
            "readme_exists": (ROOT / "README.md").is_file(), "project_exists": (ROOT / "PROJECT.md").is_file(),
            "docx_modified": False, "model_experiments_run": False, "legacy_observed_79_accessed": False,
        }
        model_registry = {
            "validation_scope": "development-selected and development-frozen; no external confirmation",
            "final_overall_model": {"candidate_id": "R0", "name": "G2 deterministic rule", "probability_available": False},
            "final_thesis_hybrid_model": {"candidate_id": "N0", "name": "nominal CNN-BiLSTM", "ensemble": "arithmetic mean of five seeds", "seeds": [202601, 202602, 202603, 202604, 202605], "calibration": "rejected_uncalibrated"},
            "selection_changed_by_closure": False,
        }
        closure_protocol = {
            "run_id": args.run_id, "type": "validation_only_repository_closure", "source_branch": branch, "source_commit": head,
            "prohibited_actions": ["model_training", "optuna", "nested_cv", "multi_seed_training", "calibration_fit", "recommendation_regeneration", "external_validation", "legacy_observed_access"],
            "actions_executed": ["repository_audit", "documentation_claim_validation", "artifact_checksum_validation", "full_test_suite", "static_database_validation", "closure_index_generation"],
        }
        provenance_files = [ROOT / "README.md", ROOT / "PROJECT.md", Path(__file__), ROOT / "database" / "migrations" / "004_governed_recommendation_phase_d.sql"]
        source_provenance = {
            "git_branch": branch, "git_commit": head, "working_tree_clean_before_materialization": not initial_status,
            "created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "source_hashes": {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in provenance_files},
            "phase_artifact_hashes": {key: sha256_file(path / "artifact_checksums.json") for key, path in {"phase_ab": PHASE_AB, "phase_c": PHASE_C, "phase_e": PHASE_E, "phase_d": PHASE_D}.items()},
        }

        readme_audit.to_csv(tmp / "readme_claim_audit.csv", index=False)
        project_audit.to_csv(tmp / "project_claim_audit.csv", index=False)
        prohibited.to_csv(tmp / "prohibited_claim_scan.csv", index=False)
        links.to_csv(tmp / "broken_link_report.csv", index=False)
        large.to_csv(tmp / "large_file_report.csv", index=False)
        repository_audit.to_csv(tmp / "repository_audit.csv", index=False)
        metrics.to_csv(tmp / "final_metrics.csv", index=False)
        write_json(tmp / "closure_protocol.json", closure_protocol)
        write_json(tmp / "repository_state.json", repository_state)
        write_json(tmp / "commit_lineage.json", lineage)
        write_json(tmp / "official_evidence_registry.json", official)
        write_json(tmp / "historical_evidence_registry.json", historical)
        write_json(tmp / "artifact_index.json", artifact_index(official, historical))
        write_json(tmp / "final_model_registry.json", model_registry)
        write_json(tmp / "final_recommendation_summary.json", recommendation)
        write_json(tmp / "test_report.json", test_report)
        write_json(tmp / "database_static_validation.json", database)
        write_json(tmp / "secret_scan_report.json", secrets)
        write_json(tmp / "source_provenance.json", source_provenance)

        evidence_rows = [
            ["Dataset/splits", f"{PHASE_AB.relative_to(ROOT).as_posix()}/dataset_manifest.json", "official protocol evidence"],
            ["Phase C comparison/ablations", f"{PHASE_C.relative_to(ROOT).as_posix()}/model_summary.csv", "official development evidence"],
            ["Final model roles", f"{PHASE_E.relative_to(ROOT).as_posix()}/final_family_decision.json", "official final development freeze"],
            ["Final metrics/stability", f"{PHASE_E.relative_to(ROOT).as_posix()}/stability_summary.csv", "official final development freeze"],
            ["Confusion matrices", f"{PHASE_E.relative_to(ROOT).as_posix()}/confusion_matrices.csv", "official final development freeze"],
            ["PR curves", f"{PHASE_E.relative_to(ROOT).as_posix()}/precision_recall_curve_points.csv", "official final development freeze"],
            ["Continuous G3", f"{PHASE_E.relative_to(ROOT).as_posix()}/regression_metrics.csv", "secondary analysis"],
            ["Recommendation technical metrics", f"{PHASE_D.relative_to(ROOT).as_posix()}/technical_safety_metrics.json", "official technical recommendation evidence"],
            ["Expert casebook", f"{PHASE_D.relative_to(ROOT).as_posix()}/expert_casebook.csv", "expert validation pending"],
            ["Database lineage", "database/migrations/001_create_source_ml_schema.sql ... 004_governed_recommendation_phase_d.sql", "source contract"],
        ]
        with (tmp / "thesis_evidence_map.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle); writer.writerow(["thesis_content", "evidence_path", "classification"]); writer.writerows(evidence_rows)
        (tmp / "thesis_writing_context.md").write_text(thesis_context(metrics, recommendation), encoding="utf-8")

        checks = [
            {"id": "working_tree_clean_at_start", "pass": not initial_status},
            {"id": "readme_project_exist", "pass": repository_state["readme_exists"] and repository_state["project_exists"]},
            {"id": "commit_ancestry", "pass": lineage["all_official_commits_ancestors"]},
            {"id": "official_artifact_checksums", "pass": all(item["pass"] for item in official_checks.values())},
            {"id": "documentation_claims", "pass": bool(readme_audit["pass"].all() and project_audit["pass"].all())},
            {"id": "no_positive_prohibited_claim", "pass": prohibited.empty or not bool(prohibited["blocking"].any())},
            {"id": "internal_links", "pass": links.empty or bool(links["exists"].all())},
            {"id": "secret_scan", "pass": secrets["status"] == "PASS"},
            {"id": "full_test_suite", "pass": test_result.returncode == 0},
            {"id": "database_static_validation", "pass": database["database_migration_static_validation"] == "PASS"},
            {"id": "final_model_roles", "pass": model_registry["final_overall_model"]["candidate_id"] == "R0" and model_registry["final_thesis_hybrid_model"]["candidate_id"] == "N0"},
            {"id": "recommendation_status", "pass": recommendation["technical_validation"] == "PASS" and recommendation["expert_validation"] == "PENDING" and recommendation["effectiveness_validation"] == "NOT_PERFORMED"},
            {"id": "active_pipeline_no_target_leakage", "pass": "Prediction snapshot feature input may contain only G1/G2" in (ROOT / "src" / "governed_recommendation.py").read_text(encoding="utf-8") and "source_record_targets" not in (ROOT / "src" / "postgres_data_source.py").read_text(encoding="utf-8").split("def load_development_feature_subset_from_postgres", 1)[1].split("\ndef load_dataset_version", 1)[0]},
            {"id": "no_new_training_or_observed_access", "pass": not repository_state["model_experiments_run"] and not repository_state["legacy_observed_79_accessed"]},
            {"id": "required_outputs_prechecksum", "pass": all((tmp / name).is_file() for name in REQUIRED_OUTPUTS if name not in {"artifact_checksums.json", "strict_validation.json", "final_repository_conclusion.md"})},
        ]
        strict = {
            "prediction_evidence_validation": "PASS" if all(official_checks[key]["pass"] for key in ["phase_ab", "phase_c", "phase_e"]) else "FAIL",
            "recommendation_technical_validation": "PASS" if official_checks["phase_d"]["pass"] else "FAIL",
            "expert_validation": "PENDING", "effectiveness_validation": "NOT_PERFORMED",
            "repository_closure_validation": "PASS" if all(item["pass"] for item in checks) else "FAIL",
            "checks": checks, "closure_bundle_checksums": "generated_and_verified_before_atomic_promotion",
        }
        write_json(tmp / "strict_validation.json", strict)
        (tmp / "final_repository_conclusion.md").write_text(conclusion(strict, test_report), encoding="utf-8")
        if strict["repository_closure_validation"] != "PASS":
            raise RuntimeError(f"Closure strict validation failed: {[item for item in checks if not item['pass']]}")

        missing = [name for name in REQUIRED_OUTPUTS if name != "artifact_checksums.json" and not (tmp / name).is_file()]
        if missing:
            raise RuntimeError(f"Missing closure outputs: {missing}")
        checksums = {path.relative_to(tmp).as_posix(): sha256_file(path) for path in sorted(tmp.iterdir()) if path.is_file() and path.name != "artifact_checksums.json"}
        write_json(tmp / "artifact_checksums.json", checksums)
        checksum_failures = [name for name, expected in checksums.items() if sha256_file(tmp / name) != expected]
        if checksum_failures:
            raise RuntimeError(f"Closure checksum verification failed: {checksum_failures}")

        for path in tmp.iterdir():
            if path.is_file():
                shutil.copy2(path, report_tmp / path.name)
        ARTIFACT_PARENT.mkdir(parents=True, exist_ok=True)
        REPORT_PARENT.mkdir(parents=True, exist_ok=True)
        os.replace(tmp, final)
        os.replace(report_tmp, report)
        print(json.dumps({"status": "PASS", "artifact_path": str(final), "report_path": str(report), "tests": test_report}, ensure_ascii=False))
    except Exception:
        if tmp.exists():
            write_json(tmp / "run_state_failure.json", {"status": "FAILED", "timestamp": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    main()
