"""Strict validation for the canonical final release."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from src.final_release.build import (
    FINAL_ROOT,
    REPORT_ROOT,
    ROOT,
    build_payload,
    sha256,
    write_text_lf,
)
from src.final_release.catalog import COMPARISON_MODELS, OFFICIAL_MODELS

LAB_PATTERN = re.compile(r"\bV(?:4|5|6)(?:[._-]\d+)?\b", re.IGNORECASE)


def verify_final_checkpoints() -> dict[str, Any]:
    manifest_path = FINAL_ROOT / "checksums" / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "checkpoint manifest missing"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    errors = []
    for entry in manifest.get("checkpoints", []):
        path = ROOT / entry["path"]
        if not path.is_file() or sha256(path) != entry["sha256"]:
            errors.append(entry["path"])
    return {
        "status": "PASS" if not errors else "FAIL",
        "checkpoint_count": len(manifest.get("checkpoints", [])),
        "errors": errors,
    }


def write_checksum_manifest() -> dict[str, Any]:
    paths = [
        FINAL_ROOT / "final_results.json",
        FINAL_ROOT / "final_results.csv",
        FINAL_ROOT / "model_registry.json",
        *sorted((ROOT / "configs" / "final").glob("*.yaml")),
        *sorted(REPORT_ROOT.glob("*.md")),
    ]
    manifest = {
        "schema_version": "final_checksum_manifest_v1",
        "files": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in paths
            if path.is_file()
        },
    }
    write_text_lf(
        FINAL_ROOT / "checksum_manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest


def _assert(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate() -> list[str]:
    errors: list[str] = []
    result_path = FINAL_ROOT / "final_results.json"
    _assert(result_path.is_file(), "final_results.json is missing", errors)
    if errors:
        return errors
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    expected = build_payload()
    _assert(
        payload == expected,
        "canonical JSON does not match recomputation from validated evidence",
        errors,
    )
    catalog = [model_id for model_id, _ in COMPARISON_MODELS]
    for dataset_id, dataset in payload["datasets"].items():
        _assert(
            [row["model_id"] for row in dataset["models"]] == catalog,
            f"{dataset_id}: model catalog/order mismatch",
            errors,
        )
        for row in dataset["models"]:
            for name, metric in row["metrics"].items():
                if metric.get("value") is None:
                    errors.append(
                        f"{dataset_id}/{row['model_id']}/{name}: applicable metric is missing"
                    )
                else:
                    source = ROOT / metric.get("source_artifact", "")
                    _assert(
                        source.is_file(),
                        f"{dataset_id}/{row['model_id']}/{name}: source missing",
                        errors,
                    )
                    if source.is_file():
                        _assert(
                            metric.get("source_checksum") == sha256(source),
                            f"{dataset_id}/{row['model_id']}/{name}: checksum mismatch",
                            errors,
                        )
            available = [
                item for item in row["per_class"] if item["f1"].get("value") is not None
            ]
            if available:
                mean_f1 = sum(item["f1"]["value"] for item in available) / len(
                    available
                )
                _assert(
                    math.isclose(
                        mean_f1, row["metrics"]["macro_f1"]["value"], abs_tol=1e-9
                    ),
                    f"{dataset_id}/{row['model_id']}: Macro-F1 != per-class mean",
                    errors,
                )
                supports = sum(item["support"]["value"] for item in available)
                cm = row["confusion_matrix"].get("value")
                if cm is not None:
                    _assert(
                        supports == sum(sum(line) for line in cm),
                        f"{dataset_id}/{row['model_id']}: support mismatch",
                        errors,
                    )
            for top in row["top_k"]:
                for name in ("precision", "recall", "f1", "ndcg"):
                    metric = top[name]
                    if metric.get("value") is not None:
                        _assert(
                            "source_artifact" in metric
                            and metric.get("calculation_method")
                            == "recomputed_from_record_aligned_ensemble_probability",
                            f"{dataset_id}/{row['model_id']}: top-k lacks probability source",
                            errors,
                        )
            _assert(
                "macro_f1" not in {
                    key for item in row["per_class"] for key in item
                },
                f"{dataset_id}/{row['model_id']}: per-class schema repeats Macro-F1",
                errors,
            )
            _assert(
                bool(row.get("evidence_origin"))
                and bool(row.get("protocol_id"))
                and bool(row.get("source_artifacts"))
                and bool(row.get("source_checksums")),
                f"{dataset_id}/{row['model_id']}: incomplete model provenance",
                errors,
            )
    _assert(
        payload.get("schema_version") == "final_results_v2"
        and payload.get("generated_from_validated_evidence") is True
        and payload.get("comparator_completion_performed") is True
        and payload.get("official_deep_models_retrained") is False
        and payload.get("future_oulad_executed") is False,
        "final_results_v2 completion flags are invalid",
        errors,
    )
    guard = verify_final_checkpoints()
    _assert(guard["status"] == "PASS", f"no-change guard failed: {guard}", errors)
    _assert(
        guard.get("checkpoint_count") == 65,
        "final checkpoint set must contain 65 ensemble checkpoints",
        errors,
    )
    completion_root = FINAL_ROOT / "comparator_completion"
    completion_validation_path = completion_root / "validation_report.json"
    _assert(
        completion_validation_path.is_file(),
        "comparator completion validation report is missing",
        errors,
    )
    if completion_validation_path.is_file():
        completion_validation = json.loads(
            completion_validation_path.read_text(encoding="utf-8")
        )
        _assert(
            completion_validation.get("status") == "PASS"
            and completion_validation.get("nine_models_each_dataset") is True
            and completion_validation.get("no_applicable_na") is True
            and completion_validation.get("future_oulad_executed") is False,
            "comparator completion validation did not pass",
            errors,
        )
    teacher_validation_path = (
        FINAL_ROOT / "teacher_feedback_validation" / "validation_report.json"
    )
    _assert(
        teacher_validation_path.is_file(),
        "teacher-feedback validation report is missing",
        errors,
    )
    if teacher_validation_path.is_file():
        teacher_validation = json.loads(
            teacher_validation_path.read_text(encoding="utf-8")
        )
        _assert(
            teacher_validation.get("status") == "PASS"
            and teacher_validation.get("future_oulad")
            == "LOCKED_NOT_EXECUTED"
            and teacher_validation.get("expert_status")
            == "PENDING_EXPERT_LABELS"
            and teacher_validation.get("xapi_in_final") is False,
            "teacher-feedback evidence validation did not pass",
            errors,
        )
    completion_manifest_path = completion_root / "checksum_manifest.json"
    _assert(
        completion_manifest_path.is_file(),
        "comparator completion checksum manifest is missing",
        errors,
    )
    if completion_manifest_path.is_file():
        completion_manifest = json.loads(
            completion_manifest_path.read_text(encoding="utf-8")
        )
        for name, digest in completion_manifest.get("files", {}).items():
            path = ROOT / name
            _assert(
                path.is_file() and sha256(path) == digest,
                f"comparator completion checksum mismatch: {name}",
                errors,
            )
    _assert(
        payload.get("future_oulad") == "LOCKED_NOT_EXECUTED",
        "Future OULAD is not locked",
        errors,
    )
    _assert(
        payload["recommendation"]["expert_status"]["value"] == "PENDING_EXPERT_LABELS",
        "expert status is fabricated",
        errors,
    )
    _assert(
        payload["recommendation"]["causal_effectiveness_claimed"]["value"] is False,
        "causal effectiveness was claimed",
        errors,
    )
    recommendation = payload["recommendation"]["metrics"]
    _assert(
        recommendation["records"]["value"] == 15378
        and recommendation["generated"]["value"] == 10953
        and recommendation["partial_evidence"]["value"] == 1209
        and recommendation["abstained"]["value"] == 3216
        and recommendation["deterministic_replay"]["value"] is True,
        "corrected recommendation counts/replay changed",
        errors,
    )
    for filename in ("README.md", "PROJECT.md"):
        text = (ROOT / filename).read_text(encoding="utf-8")
        _assert(
            not LAB_PATTERN.search(text), f"{filename} contains a lab version", errors
        )
        for metadata in OFFICIAL_MODELS.values():
            _assert(
                metadata["official_name"] in text,
                f"{filename} omits {metadata['official_name']}",
                errors,
            )
    ignored = subprocess.run(
        ["git", "check-ignore", "test_lab/"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked = subprocess.run(
        ["git", "ls-files", "test_lab"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _assert(ignored.returncode == 0, "test_lab is not gitignored", errors)
    _assert(not tracked.stdout.strip(), "test_lab contains tracked files", errors)
    csv_rows = list(
        csv.DictReader((FINAL_ROOT / "final_results.csv").open(encoding="utf-8"))
    )
    _assert(len(csv_rows) == 30, "final_results.csv must contain 30 model rows", errors)
    json_order = [
        (dataset_id, row["model_id"])
        for dataset_id, dataset in payload["datasets"].items()
        for row in dataset["models"]
    ]
    csv_order = [(row["dataset"], row["model_id"]) for row in csv_rows]
    _assert(json_order == csv_order, "CSV and JSON model order differ", errors)
    required_reports = {
        "FINAL_MODEL_RESULTS.md",
        "STUDENT_MAT_RESULTS.md",
        "STUDENT_POR_RESULTS.md",
        "OULAD_RESULTS.md",
        "IMBALANCE_RESULTS.md",
        "RECOMMENDATION_RESULTS.md",
        "CLAIM_BOUNDARIES.md",
        "FINAL_PROJECT_REVIEW.md",
        "COMPARATOR_COMPLETION_REPORT.md",
        "UNIFIED_STAGE_AWARE_RESULTS.md",
        "HYBRID_VS_ML_STAGE_MATRIX.md",
        "UNIFIED_MODEL_SELECTION_REPORT.md",
        "MLP_COMPARATOR_REPORT.md",
        "TEACHER_FEEDBACK_COMPLETION.md",
        "UCI_BASELINE_REVALIDATION_REPORT.md",
        "DATABASE_30_MODEL_CUTOVER_GUIDE.md",
        "MERGE_READINESS.md",
    }
    _assert(
        required_reports <= {path.name for path in REPORT_ROOT.glob("*.md")},
        "one or more final reports are missing",
        errors,
    )
    manifest_path = FINAL_ROOT / "checksum_manifest.json"
    unified_root = FINAL_ROOT / "unified_stage_aware_uci"
    _assert(manifest_path.is_file(), "checksum manifest is missing", errors)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_manifest_path = (
            ROOT
            / "artifacts"
            / "history"
            / "legacy_uci_separate_stage_v1"
            / "archive_manifest.json"
        )
        legacy_rows = {}
        if legacy_manifest_path.is_file():
            legacy_rows = {
                row["original_path"]: row
                for row in json.loads(
                    legacy_manifest_path.read_text(encoding="utf-8")
                )["rows"]
            }
        for name, digest in manifest.get("files", {}).items():
            if (
                name == "reports/final/PROJECT_LOCK_REPORT.md"
                and unified_root.is_dir()
            ):
                # The unified release owns this generated report and records
                # its checksum in unified_stage_evidence_manifest.json.
                continue
            path = ROOT / name
            if not path.is_file() and name in legacy_rows:
                path = ROOT / legacy_rows[name]["archived_path"]
            _assert(
                path.is_file() and sha256(path) == digest,
                f"release checksum mismatch: {name}",
                errors,
            )
    unified_validation = unified_root / "validation.json"
    _assert(
        unified_validation.is_file(),
        "unified stage-aware validation is missing",
        errors,
    )
    if unified_validation.is_file():
        unified = json.loads(unified_validation.read_text(encoding="utf-8"))
        _assert(unified.get("status") == "PASS", "unified validation failed", errors)
        _assert(
            unified.get("uci_model_identities") == 20
            and unified.get("uci_stage_rows") == 60
            and unified.get("one_estimator_all_stages") is True,
            "unified model/stage identity contract failed",
            errors,
        )
    stage_authority = FINAL_ROOT / "final_stage_results.csv"
    overall_authority = FINAL_ROOT / "final_overall_results.csv"
    _assert(
        stage_authority.is_file() and overall_authority.is_file(),
        "unified final authorities are missing",
        errors,
    )
    if stage_authority.is_file() and overall_authority.is_file():
        stage_rows = list(csv.DictReader(stage_authority.open(encoding="utf-8")))
        overall_rows = list(csv.DictReader(overall_authority.open(encoding="utf-8")))
        _assert(
            sum(row["dataset"] in {"student_mat", "student_por"} for row in stage_rows)
            == 60,
            "unified stage authority must contain 60 UCI rows",
            errors,
        )
        _assert(
            len(overall_rows) == 30,
            "unified overall authority must contain 30 model-dataset rows",
            errors,
        )
    return errors


def main() -> int:
    errors = validate()
    status = (
        "FINAL_COMPARATOR_COMPLETION_FAIL"
        if errors
        else "FINAL_COMPARATOR_COMPLETION_PASS"
    )
    print(
        json.dumps({"status": status, "errors": errors}, indent=2, ensure_ascii=False)
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
