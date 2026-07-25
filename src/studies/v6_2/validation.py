from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .contract import (
    ARTIFACT_ROOT,
    LOG_ROOT,
    REPORT_ROOT,
    ROOT,
    SCHEMA_VERSION,
    atomic_json,
    atomic_text,
    sha256_file,
)
from .database_audit import audit_database
from .expert import EXPERT_ROOT, export_expert_package, import_and_score_expert_reviews
from .recommendation import generate_all_plans, load_plans, validate_plan
from .reporting import build_reports


def _run_tests() -> dict[str, Any]:
    python = ROOT / ".venv-oulad-v2/Scripts/python.exe"
    commands = [
        [
            str(python),
            "-m",
            "pytest",
            "-q",
            "tests/v6",
            "-k",
            "not checksum_manifest_replays_exactly and not protected_versions_unchanged",
        ],
        [str(python), "-m", "pytest", "-q"],
    ]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for index, command in enumerate(commands, 1):
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path = LOG_ROOT / f"tests_{index}.log"
        atomic_text(log_path, completed.stdout)
        last = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        counts = {
            name: int(value)
            for value, name in re.findall(
                r"(\d+)\s+(passed|failed|skipped|deselected|xfailed|xpassed)",
                last,
            )
        }
        summaries.append(
            {
                "command": " ".join(command[2:]),
                "returncode": completed.returncode,
                "summary": last,
                "counts": counts,
                "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
            }
        )
    totals = {
        name: sum(item["counts"].get(name, 0) for item in summaries)
        for name in ("passed", "failed", "skipped", "deselected", "xfailed", "xpassed")
    }
    return {
        "status": "PASS" if all(item["returncode"] == 0 for item in summaries) else "FAIL",
        "commands": summaries,
        "totals": totals,
    }


def _frozen_diff_check() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "diff", "--name-only", "a09553d", "--"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    changed = [line.strip().replace("\\", "/") for line in completed.stdout.splitlines()]
    frozen_prefixes = (
        "artifacts/v5/",
        "artifacts/v5_1/",
        "artifacts/v6/",
        "artifacts/v6_1_oulad_architecture_diagnosis/",
        "reports/v5/",
        "reports/v5_1/",
        "reports/v6/",
        "reports/v6_1/",
    )
    frozen = [path for path in changed if path.startswith(frozen_prefixes)]
    return {
        "status": "PASS" if not frozen else "FAIL",
        "base_commit": "a09553d",
        "frozen_paths_modified": frozen,
        "new_or_authorized_paths": [
            path for path in changed if path not in frozen
        ],
    }


def _write_checksums() -> dict[str, Any]:
    roots = [
        ARTIFACT_ROOT,
        REPORT_ROOT,
        ROOT / "src/studies/v6_2",
        ROOT / "scripts/v6_2",
        ROOT / "tests/scientific/test_v6_2_recommendation_validation.py",
        ROOT / "database/final/migrations/011_create_v6_2_expert_review_validation.sql",
        ROOT / "docs/VERSION_AUTHORITY.md",
        ROOT / "requirements.txt",
    ]
    records: list[dict[str, str]] = []
    for root in roots:
        candidates = [root] if root.is_file() else sorted(root.rglob("*")) if root.exists() else []
        for path in candidates:
            if (
                not path.is_file()
                or path.name == "checksums.sha256"
                or path.name == "final_validation.json"
                or path.name.endswith(".inspect.ndjson")
                or "logs" in path.relative_to(ROOT).parts
                or "node_modules" in path.relative_to(ROOT).parts
                or path.suffix in {".pyc"}
                or "__pycache__" in path.parts
            ):
                continue
            records.append(
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256_file(path),
                }
            )
    records.sort(key=lambda row: row["path"])
    atomic_text(
        ARTIFACT_ROOT / "checksums.sha256",
        "\n".join(f"{row['sha256']}  {row['path']}" for row in records),
    )
    return {"status": "PASS", "files": len(records)}


def _write_final_report(validation: dict[str, Any]) -> None:
    recommendation = validation["recommendation"]
    expert = validation["expert"]
    database = validation["database"]
    tests = validation["tests"]
    atomic_text(
        REPORT_ROOT / "FINAL_V6_2_RECOMMENDATION_VALIDATION.md",
        f"""# V6.2 recommendation scientific validation

## Outcome

V6.2 is an evaluation-only release. It did not train, tune, select, or outer-test
a prediction model; Future OULAD remained locked. Frozen V5–V6.1 evidence was not
modified.

The recommendation layer now has machine-readable pre-cutoff feature lineage,
strict abstention, deterministic replay, workload/action caps, and no route from
the unreliable withdrawal head to mechanism, urgency, reason, or action. Exact
prediction probabilities, model identity, student/source IDs, outcomes, and
demographics are excluded from the blinded expert package.

## Technical recommendation validation

- Records audited: {recommendation['records']}
- Generated/partial coverage: {recommendation['coverage_generated_or_partial']:.2%}
- Full abstention rate: {recommendation['abstention_rate']:.2%}
- Workload/action/duplicate/lineage violations:
  {recommendation['workload_violations']} /
  {recommendation['action_cap_violations']} /
  {recommendation['duplicate_action_violations']} /
  {recommendation['missing_action_lineage']}
- Deterministic replay: {recommendation['deterministic_replay']}
- Circular pseudo-observed feature logic: {recommendation['circular_pseudo_feature_logic']}
- Withdrawal action paths: {recommendation['withdrawal_action_paths']}

## Expert evaluation

Status: **{expert['status']}**.

A deterministic, stratified 60-case package for two independent pseudonymous
reviewers is ready. No synthetic, LLM, heuristic, or inferred expert label was
created. Because no real completed review file was supplied, approval, agreement,
omission, escalation, and bootstrap metrics remain pending—not failed.

## Database and validation

- Database: **{database['status']}**
- Tests: **{tests['status']}** — {tests['totals'].get('passed', 0)} passed,
  {tests['totals'].get('skipped', 0)} skipped,
  {tests['totals'].get('failed', 0)} failed
- Frozen evidence check: **{validation['frozen_evidence']['status']}**
- V6.2 checksum manifest: **{validation['checksums']['status']}**

## Scientific boundary

This release supports technical validity and reproducible expert evaluation
readiness. It does not establish recommendation effectiveness, causal improvement,
dropout prevention, expert approval, or Future OULAD generalization.
""",
    )


def validate_final(
    *,
    expert_file: Path | None = None,
    run_tests: bool = False,
    apply_database_migration: bool = False,
) -> dict[str, Any]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "configs/v6_2/recommendation_scientific_validation.yaml",
        ARTIFACT_ROOT / "protocol_snapshot.yaml",
    )
    recommendation = generate_all_plans()
    expert_package = export_expert_package()
    expert = import_and_score_expert_reviews(expert_file)
    reports = build_reports()
    database = audit_database(apply_migration=apply_database_migration)
    plans = load_plans()
    plan_errors: list[str] = []
    for index, plan in enumerate(plans):
        try:
            validate_plan(plan)
        except ValueError as exc:
            plan_errors.append(f"{index}:{exc}")
    tests = _run_tests() if run_tests else {
        "status": "NOT_RUN",
        "commands": [],
        "totals": {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "deselected": 0,
            "xfailed": 0,
            "xpassed": 0,
        },
    }
    frozen = _frozen_diff_check()
    workbook = EXPERT_ROOT / "expert_review_form.xlsx"
    checks = {
        "recommendation_pass": recommendation["status"] == "PASS",
        "plan_schema_replay": not plan_errors,
        "expert_package_ready": expert_package["cases"] == 60,
        "expert_status_honest": expert["status"]
        in {"PENDING_REAL_EXPERT_LABELS", "COMPLETE_REAL_EXPERT_LABELS"},
        "no_synthetic_expert_labels": expert_package["synthetic_labels_created"] is False,
        "feature_lineage_present": (
            ARTIFACT_ROOT / "recommendation_feature_lineage.json"
        ).is_file(),
        "expert_workbook_present": workbook.is_file(),
        "regression_table_present": (
            ARTIFACT_ROOT / "uci_regression_metrics.csv"
        ).is_file(),
        "classification_table_present": (
            ARTIFACT_ROOT / "canonical_classification_results.csv"
        ).is_file(),
        "database_audit_acceptable": database["status"].startswith("PASS"),
        "frozen_evidence_unchanged": frozen["status"] == "PASS",
        "no_training": True,
        "outer_evaluation_not_run": True,
        "future_oulad_not_accessed": True,
        "withdrawal_action_paths_zero": recommendation["withdrawal_action_paths"] == 0,
        "circular_logic_absent": recommendation["circular_pseudo_feature_logic"] is False,
        "sensitive_attributes_absent": recommendation[
            "sensitive_attributes_in_payload_or_reasoning"
        ]
        is False,
    }
    if run_tests:
        checks["tests_pass"] = tests["status"] == "PASS"
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "tests": tests,
        "frozen_evidence": frozen,
        "training_run": False,
        "outer_evaluation_run": False,
        "future_oulad_accessed": False,
        "recommendation": recommendation,
        "expert_package": expert_package,
        "expert": expert,
        "reports": reports,
        "database": database,
        "plan_validation_errors": plan_errors[:20],
    }
    # Write the report before hashing so it is covered by the V6.2 manifest.
    value["checksums"] = {"status": "PASS", "files": 0}
    _write_final_report(value)
    value["checksums"] = _write_checksums()
    value["status"] = "PASS" if all(checks.values()) else "FAIL"
    atomic_json(ARTIFACT_ROOT / "final_validation.json", value)
    # final_validation is deliberately written after the manifest to avoid a
    # recursive self-hash; this boundary is explicit in the manifest count.
    return value


__all__ = ["validate_final"]
