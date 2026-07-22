"""Strict validation for the canonical final release."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from src.final_release.build import FINAL_ROOT, REPORT_ROOT, ROOT, build_payload, sha256
from src.final_release.catalog import COMPARISON_MODELS, OFFICIAL_MODELS

LAB_PATTERN = re.compile(r"\bV(?:4|5|6)(?:[._-]\d+)?\b", re.IGNORECASE)


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
    (FINAL_ROOT / "checksum_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
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
        "canonical JSON does not match recomputation from frozen evidence",
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
                    _assert(
                        metric.get("status") == "N/A" and bool(metric.get("reason")),
                        f"{dataset_id}/{row['model_id']}/{name}: missing metric is not explicit N/A",
                        errors,
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
                            and metric.get("calculation")
                            == "recomputed_from_frozen_predictions",
                            f"{dataset_id}/{row['model_id']}: top-k lacks probability source",
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
    _assert(len(csv_rows) == 27, "final_results.csv must contain 27 model rows", errors)
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
    }
    _assert(
        required_reports <= {path.name for path in REPORT_ROOT.glob("*.md")},
        "one or more final reports are missing",
        errors,
    )
    manifest_path = FINAL_ROOT / "checksum_manifest.json"
    _assert(manifest_path.is_file(), "checksum manifest is missing", errors)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, digest in manifest.get("files", {}).items():
            path = ROOT / name
            _assert(
                path.is_file() and sha256(path) == digest,
                f"release checksum mismatch: {name}",
                errors,
            )
    return errors


def main() -> int:
    errors = validate()
    status = "FINAL_PROJECT_CLEANUP_FAIL" if errors else "FINAL_PROJECT_CLEANUP_PASS"
    print(
        json.dumps({"status": status, "errors": errors}, indent=2, ensure_ascii=False)
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
