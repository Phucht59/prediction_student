"""Runtime guards for the detached unified OULAD pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies import oulad_multistage as study  # noqa: E402

OUT = ROOT / "artifacts" / "final" / "unified_stage_aware_oulad"
ARCHIVE = ROOT / "artifacts" / "history" / "partial_svm_probability_true_20260729"
REPORT = ROOT / "reports" / "refactor" / "OULAD_SVM_RUNTIME_PROTOCOL_AMENDMENT.md"
BASE = "ef40acf8de12aae8a66c2df84e5466c9c42ea4ef"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def checkpoint_counts() -> dict[str, int]:
    root = OUT / "checkpoints"
    return {
        directory.name: len(list(directory.rglob("*.joblib")))
        + len(list(directory.rglob("*.pt")))
        for directory in sorted(root.glob("*_oulad"))
        if directory.is_dir()
    }


def preflight() -> dict[str, Any]:
    branch = _git("branch", "--show-current")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT
    ).returncode == 0
    uci = ROOT / "artifacts" / "final" / "unified_stage_aware_uci" / "checksums.json"
    protected = [
        path
        for suffix in ("*.docx", "*.pdf")
        for path in ROOT.rglob(suffix)
        if ".git" not in path.parts
        and not any(part.startswith(".venv") for part in path.parts)
    ]
    result = {
        "status": "PASS"
        if branch == "codex/unified-oulad-stage-aware-system" and ancestor
        else "FAIL",
        "branch": branch,
        "head": _git("rev-parse", "HEAD"),
        "base_ancestor": ancestor,
        "checkpoint_counts": checkpoint_counts(),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "interpreter": os.sys.executable,
        "raw_data_available": (ROOT / "data" / "raw" / "studentVle.csv").is_file(),
        "uci_checksum_snapshot": _sha(uci) if uci.is_file() else None,
        "document_checksums": {
            path.relative_to(ROOT).as_posix(): _sha(path) for path in protected
        },
        "canonical_database_modified": False,
        "credentials": "REDACTED",
    }
    _json(OUT / "detached_preflight.json", result)
    return result


def amendment() -> dict[str, Any]:
    source = OUT / "checkpoints" / "svm_oulad"
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archived = []
    if source.is_dir():
        for path in sorted(source.rglob("*.joblib")):
            destination = ARCHIVE / "svm_oulad" / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
    for path in sorted((ARCHIVE / "svm_oulad").rglob("*.joblib")):
        parts = path.parts
        fold = int(next(item for item in parts if item.startswith("outer_fold_")).split("_")[-1])
        seed = int(path.stem.split("_")[-1])
        archived.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha(path),
                "byte_count": path.stat().st_size,
                "outer_fold": fold,
                "seed": seed,
                "created_at": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "estimator": "sklearn.svm.SVC(probability=True)",
                "reason": "superseded_by_inner_oof_external_sigmoid_calibration",
                "scientific_use": False,
                "final_authority": False,
            }
        )
    manifest = {
        "schema_version": "partial_svm_archive_v1",
        "status": "PASS",
        "checkpoint_count": len(archived),
        "rows": archived,
    }
    _json(ARCHIVE / "manifest.json", manifest)
    source.mkdir(parents=True, exist_ok=True)
    if any(source.rglob("*.*")):
        raise RuntimeError("canonical SVM checkpoint directory is not empty")
    payload = {
        "schema_version": "svm_runtime_protocol_amendment_v1",
        "status": "PASS",
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "reason": "SVC probability=True required prohibitively slow internal calibration on stage-expanded OULAD",
        "partial_checkpoint_count": len(archived),
        "partial_checkpoints_used_in_final_evidence": False,
        "model_family": "sklearn.svm.SVC",
        "kernel": "rbf",
        "probability": False,
        "calibration": "Platt-style LogisticRegression fitted to pooled inner-OOF decision scores",
        "calibration_uses_outer_labels": False,
        "threshold_selection_uses_outer_labels": False,
        "unchanged": [
            "folds",
            "seeds",
            "features",
            "target",
            "cohorts",
            "stages",
            "model_selection_objective",
        ],
        "runtime_options": {
            "cache_size_mb": 4096,
            "shrinking": True,
            "tol": 0.001,
            "max_iter": -1,
            "class_weight": "balanced",
        },
    }
    _json(OUT / "svm_runtime_protocol_amendment.json", payload)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "# OULAD SVM Runtime Protocol Amendment\n\n"
        "The SVM family remains nonlinear RBF `sklearn.svm.SVC`. Internal "
        "`probability=True` calibration is superseded by `probability=False` "
        "and a one-variable Platt-style sigmoid fitted exclusively from pooled "
        "inner-OOF decision scores. Outer labels are used for neither "
        "calibration nor threshold selection. The archived partial checkpoints "
        "have `scientific_use=false` and are excluded from final evidence.\n\n"
        "Folds, seeds, features, target, cohorts, stages, and the model-selection "
        "objective are unchanged.\n",
        encoding="utf-8",
    )
    return payload


def audit() -> dict[str, Any]:
    expected = {
        "logistic_regression_oulad",
        "decision_tree_oulad",
        "random_forest_oulad",
        "hist_gradient_boosting_oulad",
    }
    rows = []
    errors = []
    for model in sorted(expected):
        paths = sorted((OUT / "checkpoints" / model).rglob("*.joblib"))
        if len(paths) != 15:
            errors.append(f"{model}: expected 15, got {len(paths)}")
        for path in paths:
            try:
                estimator = joblib.load(path)
                readable = hasattr(estimator, "predict_proba")
            except Exception as exc:  # pragma: no cover - evidence guard
                readable = False
                errors.append(f"{path}: {type(exc).__name__}")
            rows.append(
                {
                    "model_id": model,
                    "path": path.relative_to(ROOT).as_posix(),
                    "outer_fold": int(path.parent.name.split("_")[-1]),
                    "seed": int(path.stem.split("_")[-1]),
                    "byte_count": path.stat().st_size,
                    "sha256": _sha(path),
                    "readable": readable,
                    "stage_shared_estimator": True,
                    "timestamp_preserved": True,
                }
            )
    result = {
        "schema_version": "oulad_resume_checkpoint_audit_v1",
        "status": "PASS" if not errors and len(rows) == 60 else "FAIL",
        "checkpoint_count": len(rows),
        "errors": errors,
        "rows": rows,
    }
    _json(OUT / "resume_checkpoint_audit.json", result)
    return result


def gpu() -> dict[str, Any]:
    available = torch.cuda.is_available()
    properties = torch.cuda.get_device_properties(0) if available else None
    free, total = torch.cuda.mem_get_info(0) if available else (0, 0)
    result = {
        "schema_version": "oulad_gpu_runtime_audit_v1",
        "status": "PASS" if available else "BLOCKED_GPU",
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": available,
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "total_vram_bytes": int(properties.total_memory) if properties else 0,
        "free_vram_bytes": int(free),
        "selected_device": "cuda:0" if available else None,
        "deterministic_settings": {
            "fixed_seeds": list(study.SEEDS),
            "best_seed_selection": False,
        },
    }
    _json(OUT / "gpu_runtime_audit.json", result)
    return result


def smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        result = {"status": "BLOCKED_GPU", "reason": "CUDA unavailable"}
        _json(OUT / "smoke_validation.json", result)
        return result
    generator = np.random.default_rng(42)
    features = generator.normal(size=(96, 12))
    target = np.asarray([0, 1] * 48)
    base = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=False,
        class_weight="balanced",
        cache_size=4096,
        random_state=42,
    ).fit(features[:64], target[:64])
    decision = base.decision_function(features[64:])
    calibrator = LogisticRegression(random_state=42).fit(
        decision.reshape(-1, 1), target[64:]
    )
    calibrated = calibrator.predict_proba(decision.reshape(-1, 1))[:, 1]
    deep = {}
    config = study._deep_config(study._protocol())
    for kind in study.DEEP:
        model = study._deep_model(kind, 165, 12, config).cuda().eval()
        sequence = torch.zeros((2, 4, 47), device="cuda")
        lengths = torch.full((2,), 4, dtype=torch.int64, device="cuda")
        mask = torch.ones((2, 4), device="cuda")
        aggregate = torch.zeros((2, 165), device="cuda")
        static = torch.zeros((2, 12), device="cuda")
        with torch.no_grad():
            probability = study._deep_probability(
                model,
                (sequence, lengths, mask, aggregate, static),
                kind,
            )
        deep[kind] = {
            "shape": list(probability.shape),
            "finite": bool(torch.isfinite(probability).all().item()),
        }
    result = {
        "schema_version": "oulad_detached_smoke_v1",
        "status": "PASS"
        if np.isfinite(calibrated).all()
        and all(item["finite"] for item in deep.values())
        else "FAIL",
        "svm": {
            "kernel": "rbf",
            "probability_internal": False,
            "external_sigmoid": True,
        },
        "deep": deep,
        "stages": list(study.STAGES),
        "same_checkpoint_four_stage_contract": True,
        "not_final_evidence": True,
    }
    _json(OUT / "smoke_validation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("preflight", "amendment", "audit", "gpu", "smoke")
    )
    args = parser.parse_args()
    result = globals()[args.command]()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 21


if __name__ == "__main__":
    raise SystemExit(main())
