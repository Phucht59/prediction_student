"""Phase 10 final, read-only scientific/security audit.

This verifier never calls a provider and never recomputes held-out metrics.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/recommend_hybrid/explainable_v2"
PANEL_B = BASE / "final_heldout/panel_b_v1"
OUT = BASE / "audit/final_release_v1/PHASE10_FINAL_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}
    dev_path = BASE / "frozen/development_v2/DEVELOPMENT_FREEZE_MANIFEST.json"
    dev = load(dev_path)
    final_manifest_path = PANEL_B / "PANEL_B_FINAL_HELDOUT_MANIFEST.json"
    final_manifest = load(final_manifest_path)
    metrics_path = PANEL_B / "PANEL_B_FINAL_HELDOUT_METRICS.json"
    metrics = load(metrics_path)

    checks["development_freeze_pass"] = dev.get("status") == "PASS"
    checks["development_freeze_panel_b_untouched"] = dev.get("panel_b_touched") is False
    checks["panel_b_manifest_pass"] = final_manifest.get("status") == "PASS"
    checks["panel_b_one_shot_complete"] = (
        final_manifest.get("panel_b_case_count") == 150
        and final_manifest.get("real_external_review_record_count") == 557
        and final_manifest.get("failed_provider_calls") == 0
        and final_manifest.get("post_panel_b_tuning_permitted") is False
    )
    checks["panel_b_metrics_scope"] = metrics.get("scope") == "PANEL_B_FINAL_HELDOUT"
    checks["panel_b_metrics_hash"] = sha256(metrics_path) == final_manifest["metrics_sha256"]
    checks["panel_b_reviews_hash"] = sha256(
        PANEL_B / "panel_b_real_external_reviews_frozen.jsonl"
    ) == final_manifest["frozen_reviews_sha256"]
    checks["panel_b_scores_hash"] = sha256(
        PANEL_B / "panel_b_final_heldout_scores.parquet"
    ) == final_manifest["scores_sha256"]
    checks["evaluator_unchanged"] = sha256(
        ROOT / "scripts/recommend_hybrid/explainable_v2/evaluate_panel_b_final_heldout_v1.py"
    ) == final_manifest["evaluator_sha256"]

    checksum_ok = True
    checksum_count = 0
    for line in (PANEL_B / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        checksum_count += 1
        checksum_ok &= sha256(PANEL_B / name.strip()) == expected
    checks["panel_b_checksum_inventory"] = checksum_ok and checksum_count == 9
    details["panel_b_checksum_file_count"] = checksum_count

    panel_a_freeze = BASE / "annotations/frozen/panel_a_v1/PANEL_A_FREEZE_MANIFEST.json"
    panel_a_reviews = BASE / "annotations/frozen/panel_a_v1/panel_a_external_reviews_frozen.jsonl"
    checks["panel_a_freeze_unchanged"] = (
        sha256(panel_a_freeze) == dev["panel_a_review_lineage"]["freeze_manifest_sha256"]
        and sha256(panel_a_reviews) == "4a4871426880bdcd1257dc15c29a36c23de34481f07be68d8e5095dc20efefb9"
    )

    release = load(BASE / "release_gates/panel_a_v1/PANEL_A_RELEASE_GATES.json")
    checks["panel_a_release_gates_pass"] = release.get("status") == "PASS"
    checks["no_invalid_action"] = release.get("invalid_action_rate") == 0
    checks["minimum_source_support"] = dev["labels"]["minimum_independent_source_families"] == 2
    checks["final_router_contract"] = dev["router"]["public_statuses"] == [
        "RECOMMEND", "INSUFFICIENT_EVIDENCE", "HUMAN_REVIEW", "NO_FEASIBLE_ACTION"
    ]
    checks["seed_disagreement_not_imputed"] = (
        dev["router"]["selected_thresholds"]["maximum_seed_disagreement"] is None
    )
    checks["selected_config_locked"] = dev["ranker"]["selected_config_id"] == "a70599afad40"
    checks["five_models_frozen"] = len(dev["ranker"]["five_model_sha256"]) == 5

    case_manifest = load(BASE / "annotations/exports/case_manifest.json")
    checks["zero_student_overlap"] = case_manifest.get("panel_student_overlap_count") == 0
    checks["zero_query_overlap"] = case_manifest.get("panel_query_overlap_count") == 0
    checks["post_cutoff_audit"] = dev["pre_freeze_checks"]["post_cutoff_violations"] == 0

    forbidden_feature_names = {"final_result", "outcome", "withdrawn", "passed", "failed"}
    checks["feature_leakage_audit"] = not (forbidden_feature_names & set(dev["feature_schema"]))

    token_patterns = [
        re.compile(rb"AIza[A-Za-z0-9_-]{30,}"),
        re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
        re.compile(rb"AQ\.[A-Za-z0-9_-]{20,}"),
    ]
    secret_paths: list[str] = []
    sensitive_paths: list[str] = []
    scanned = 0
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        lower = relative.lower()
        if any(part in lower for part in ("private_case_mapping", "case_export_salt")):
            sensitive_paths.append(relative)
        if Path(relative).name.lower().startswith(".env") and Path(relative).name != ".env.example":
            sensitive_paths.append(relative)
        if not path.is_file() or path.stat().st_size > 10_000_000:
            continue
        if path.suffix.lower() in {".joblib", ".parquet", ".png", ".jpg", ".pdf", ".pkl", ".pyc"}:
            continue
        data = path.read_bytes()
        scanned += 1
        if any(pattern.search(data) for pattern in token_patterns):
            secret_paths.append(relative)
    checks["secret_scan"] = not secret_paths
    checks["private_mapping_salt_env_scan"] = not sensitive_paths
    details["secret_scan_file_count"] = scanned
    details["secret_finding_paths"] = secret_paths
    details["sensitive_path_findings"] = sensitive_paths

    phase9 = load(BASE / "audit/final_integration_v1/PHASE9_END_TO_END_INTEGRATION_AUDIT.json")
    checks["phase9_end_to_end_pass"] = phase9.get("status") == "PASS"
    checks["provider_not_called_in_phase9"] = phase9.get("provider_called") is False
    checks["panel_b_not_recomputed_in_phase9"] = phase9.get("panel_b_recomputed") is False

    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema_version": "recommendation_v2_phase10_final_audit_v1",
        "scope": "FINAL_SCIENTIFIC_ENGINEERING_AUDIT",
        "status": status,
        "provider_called": False,
        "panel_b_recomputed": False,
        "runtime_authorized": False,
        "final_metrics_claimed": True,
        "checks": checks,
        "details": details,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PHASE10_STATUS={status}")
    print(f"CHECKS_PASSED={sum(checks.values())}/{len(checks)}")
    print(f"AUDIT_SHA256={sha256(OUT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
