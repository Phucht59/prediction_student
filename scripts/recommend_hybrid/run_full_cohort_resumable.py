"""Run the frozen counterfactual evaluator in resume-safe fold/stage batches."""

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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recommend_hybrid.evaluate_counterfactual_recommender import (  # noqa: E402
    STAGES,
    _bootstrap_mean_ci,
    _write_report,
)
from src.recommend_hybrid.counterfactual.evaluation import (  # noqa: E402
    aggregate_counterfactual_metrics,
    grouped_counterfactual_metrics,
)

FULL_ROOT = ROOT / "artifacts/recommend_hybrid/counterfactual/full_cohort"
INPUT_REGISTRY = ROOT / "artifacts/recommend_hybrid/counterfactual/full_evaluation_input_registry.json"
CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"
FOLDS = (0, 1, 2)
SEEDS = (42, 1201, 2026, 3407, 7319)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _authority_paths() -> dict[str, Path]:
    return {
        "raw_manifest": ROOT / "data/manifests/extension_raw_manifest.json",
        "split_manifest": ROOT / "data/processed/study_c_oulad/manifests/split_manifest.csv",
        "release_manifest": ROOT / "artifacts/recommend_hybrid/RESIDUAL_CHECKPOINT_RELEASE_MANIFEST.json",
        "recommendation_manifest": ROOT / "artifacts/recommend_hybrid/RECOMMEND_HYBRID_CHECKPOINT_MANIFEST.json",
        "model_authority": ROOT / "configs/recommend_hybrid/model_authority.yaml",
        "counterfactual_config": ROOT / "configs/recommend_hybrid/counterfactual_oulad.yaml",
        "tensor_config": ROOT / "configs/recommend_hybrid/counterfactual_oulad_tensor.yaml",
    }


def build_input_registry(*, bootstrap_replicates: int) -> dict[str, Any]:
    if _git("branch", "--show-current") != "codex/constrained-counterfactual-recommender":
        raise RuntimeError("full cohort requires the constrained counterfactual branch")
    if _git("status", "--porcelain"):
        raise RuntimeError("full cohort requires a clean working tree")
    paths = _authority_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing full-cohort authority files: {missing}")
    release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    if release.get("status") != "RELEASE_FROZEN" or int(release.get("checkpoint_count", -1)) != 30:
        raise RuntimeError("residual release manifest is not the frozen 30-checkpoint authority")
    authority_validation = json.loads(
        (ROOT / "artifacts/recommend_hybrid/counterfactual/checkpoint_authority_validation.json").read_text(encoding="utf-8")
    )
    if authority_validation.get("status") != "PASS":
        raise RuntimeError("checkpoint authority validation is not PASS")
    code_paths = [
        path for path in (ROOT / "src/recommend_hybrid").rglob("*.py")
    ] + [
        ROOT / "scripts/recommend_hybrid/evaluate_counterfactual_recommender.py",
        ROOT / "scripts/recommend_hybrid/run_full_cohort_resumable.py",
    ]
    policy_paths = [
        ROOT / "configs/recommend_hybrid/actions.yaml",
        ROOT / "configs/recommend_hybrid/policy_common.yaml",
        ROOT / "configs/recommend_hybrid/policy_oulad.yaml",
        ROOT / "configs/recommend_hybrid/counterfactual_oulad.yaml",
        ROOT / "src/recommend_hybrid/oulad/action_catalog.py",
        ROOT / "src/recommend_hybrid/oulad/policy.py",
        ROOT / "src/recommend_hybrid/counterfactual/plan_builder.py",
        ROOT / "src/recommend_hybrid/counterfactual/ranker.py",
        ROOT / "src/recommend_hybrid/counterfactual/simulator.py",
    ]
    preprocessor_hashes = sorted({
        str(row["preprocessor_fingerprint"])
        for row in authority_validation.get("checkpoints", [])
        if row.get("preprocessor_fingerprint")
    })
    return {
        "schema_version": "full_evaluation_input_registry_v1",
        "created_at_utc": _now(),
        "source_commit": _git("rev-parse", "HEAD"),
        "code_tree_hash": _hash_files(code_paths),
        "raw_manifest": {"path": str(paths["raw_manifest"].relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(paths["raw_manifest"])},
        "split_manifest": {"path": str(paths["split_manifest"].relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(paths["split_manifest"])},
        "checkpoint_release_manifest": {"path": str(paths["release_manifest"].relative_to(ROOT)).replace("\\", "/"), "sha256": _sha256(paths["release_manifest"])},
        "recommendation_manifest_sha256": _sha256(paths["recommendation_manifest"]),
        "config_hash": _hash_files([paths["model_authority"], paths["counterfactual_config"], paths["tensor_config"]]),
        "action_policy_hash": _hash_files(policy_paths),
        "preprocessor_hashes": preprocessor_hashes,
        "authority_architecture_hash": authority_validation.get("authority_architecture_hash"),
        "authority_parameter_count": authority_validation.get("authority_parameter_count"),
        "evaluation_command": {
            "script": "scripts/recommend_hybrid/evaluate_counterfactual_recommender.py",
            "folds": list(FOLDS),
            "stages": list(STAGES),
            "seeds": list(SEEDS),
            "max_records_per_fold_stage": None,
            "bootstrap_replicates": bootstrap_replicates,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _batch_id(fold: int, stage: str) -> str:
    return f"fold_{fold}__{stage}"


def _required_batch_files(directory: Path) -> list[Path]:
    return [directory / name for name in ("evaluation.json", "evaluation_rows.csv", "action_scores.csv")]


def _batch_checksum(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): _sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "batch_checksums.json"
    }


def _valid_completed_batch(directory: Path, fold: int, stage: str) -> bool:
    if not all(path.is_file() for path in _required_batch_files(directory)):
        return False
    payload = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
    config = payload.get("configuration", {})
    return (
        payload.get("status") == "PASS"
        and config.get("max_records_per_fold_stage") is None
        and config.get("folds") == [fold]
        and config.get("bundle_stages") == [stage]
    )


def _write_parquets(directory: Path) -> None:
    pd.read_csv(directory / "evaluation_rows.csv").to_parquet(directory / "evaluation_rows.parquet", index=False)
    pd.read_csv(directory / "action_scores.csv").to_parquet(directory / "action_scores.parquet", index=False)


def _aggregate(full_root: Path, batches: list[dict[str, Any]], bootstrap_replicates: int) -> dict[str, Any]:
    row_frames = [pd.read_csv(full_root / item["batch_id"] / "evaluation_rows.csv") for item in batches]
    action_frames = [pd.read_csv(full_root / item["batch_id"] / "action_scores.csv") for item in batches]
    rows = pd.concat(row_frames, ignore_index=True)
    actions = pd.concat(action_frames, ignore_index=True)
    rows.to_parquet(full_root / "evaluation_rows.parquet", index=False)
    actions.to_parquet(full_root / "action_scores.parquet", index=False)
    from src.recommend_hybrid.counterfactual.evaluation import CounterfactualEvaluationRow

    typed_rows = [
        CounterfactualEvaluationRow(
            student_key=str(row.student_key), course_key=str(row.course_key), stage=str(row.stage), fold=int(row.fold),
            baseline_risk=float(row.baseline_risk), decision_threshold=float(row.decision_threshold), status=str(row.status),
            top_action_id=None if pd.isna(row.top_action_id) else str(row.top_action_id),
            top_counterfactual_risk=None if pd.isna(row.top_counterfactual_risk) else float(row.top_counterfactual_risk),
            top_risk_reduction=None if pd.isna(row.top_risk_reduction) else float(row.top_risk_reduction),
            top_utility_score=None if pd.isna(row.top_utility_score) else float(row.top_utility_score),
            selected_action_count=int(row.selected_action_count), selected_workload_minutes=int(row.selected_workload_minutes),
            reference_profile_id=None if pd.isna(row.reference_profile_id) else str(row.reference_profile_id),
            fallback_reasons=tuple(json.loads(str(row.fallback_reasons).replace("'", '"'))) if str(row.fallback_reasons).startswith("[") else tuple(),
        )
        for row in rows.itertuples(index=False)
    ]
    metrics = aggregate_counterfactual_metrics(typed_rows)
    reductions = [float(row.top_risk_reduction) for row in typed_rows if row.top_risk_reduction is not None]
    metrics["mean_risk_reduction_bootstrap_95"] = _bootstrap_mean_ci(reductions, seed=20260803, replicates=bootstrap_replicates)
    payload = {
        "schema_version": "counterfactual_oulad_full_cohort_v1",
        "generated_at": _now(), "claim_boundary": CLAIM_BOUNDARY,
        "configuration": {"folds": list(FOLDS), "bundle_stages": list(STAGES), "seeds": list(SEEDS), "max_records_per_fold_stage": None, "bootstrap_replicates": bootstrap_replicates, "sampling": "deterministic_course_round_robin", "reference_scope": "TRAINING_FOLD_COURSE_STAGE_ONLY"},
        "overall": metrics, "grouped": grouped_counterfactual_metrics(typed_rows), "batch_count": len(batches),
        "scientific_guards": {"target_used_for_ranking": False, "final_result_used_for_ranking": False, "date_unregistration_used_for_ranking": False, "outer_validation_labels_used_for_tuning": False, "expert_labels_required": False, "silver_labels_used": False, "causal_effect_claimed": False},
        "status": "PASS",
    }
    _write_json(full_root / "evaluation.json", payload)
    _write_json(full_root / "bootstrap.json", payload["overall"]["mean_risk_reduction_bootstrap_95"])
    _write_report(payload, report_path=full_root / "evaluation.md")
    return payload


def run(*, bootstrap_replicates: int, resume: bool) -> int:
    registry = build_input_registry(bootstrap_replicates=bootstrap_replicates)
    FULL_ROOT.mkdir(parents=True, exist_ok=True)
    if INPUT_REGISTRY.is_file():
        existing = json.loads(INPUT_REGISTRY.read_text(encoding="utf-8"))
        immutable = ("source_commit", "code_tree_hash", "raw_manifest", "split_manifest", "checkpoint_release_manifest", "config_hash", "action_policy_hash", "preprocessor_hashes")
        if any(existing.get(key) != registry.get(key) for key in immutable):
            raise RuntimeError("full-evaluation input authority changed; refusing resume")
    else:
        _write_json(INPUT_REGISTRY, registry)
    progress_path = FULL_ROOT / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"schema_version": "full_cohort_progress_v1", "status": "RUNNING", "started_at": _now(), "completed_batches": [], "errors": []}
    progress["status"] = "RUNNING"
    _write_json(progress_path, progress)
    batch_registry: dict[str, Any] = {"schema_version": "full_cohort_batch_registry_v1", "input_registry_sha256": _sha256(INPUT_REGISTRY), "batches": {}}
    batch_registry_path = FULL_ROOT / "batch_registry.json"
    if batch_registry_path.is_file():
        batch_registry = json.loads(batch_registry_path.read_text(encoding="utf-8"))
    for stage in STAGES:
        for fold in FOLDS:
            batch_id = _batch_id(fold, stage)
            final_dir = FULL_ROOT / batch_id
            if resume and _valid_completed_batch(final_dir, fold, stage):
                batch_registry["batches"][batch_id] = {"batch_id": batch_id, "fold": fold, "stage": stage, "status": "PASS", "resumed": True, "checksums": _batch_checksum(final_dir)}
                if batch_id not in progress["completed_batches"]:
                    progress["completed_batches"].append(batch_id)
                _write_json(batch_registry_path, batch_registry); _write_json(progress_path, progress)
                continue
            temporary = FULL_ROOT / f".{batch_id}.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            command = [sys.executable, "scripts/recommend_hybrid/evaluate_counterfactual_recommender.py", "--folds", str(fold), "--stages", stage, "--seeds", ",".join(map(str, SEEDS)), "--max-records-per-fold-stage", "0", "--bootstrap-replicates", str(bootstrap_replicates), "--output-dir", str(temporary)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            (temporary / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (temporary / "stderr.log").write_text(result.stderr, encoding="utf-8")
            if result.returncode != 0 or not _valid_completed_batch(temporary, fold, stage):
                error = {"batch_id": batch_id, "fold": fold, "stage": stage, "status": "FAIL", "return_code": result.returncode, "stderr_tail": result.stderr[-4000:]}
                progress.setdefault("errors", []).append(error)
                batch_registry["batches"][batch_id] = error
                _write_json(batch_registry_path, batch_registry); _write_json(progress_path, progress)
                return 1
            _write_parquets(temporary)
            checksums = _batch_checksum(temporary)
            _write_json(temporary / "batch_checksums.json", checksums)
            if final_dir.exists():
                raise RuntimeError(f"incomplete batch exists and will not be overwritten: {final_dir}")
            temporary.replace(final_dir)
            entry = {"batch_id": batch_id, "fold": fold, "stage": stage, "status": "PASS", "resumed": False, "checksums": _batch_checksum(final_dir)}
            batch_registry["batches"][batch_id] = entry
            progress["completed_batches"].append(batch_id)
            _write_json(batch_registry_path, batch_registry); _write_json(progress_path, progress)
    batches = [batch_registry["batches"][_batch_id(fold, stage)] for stage in STAGES for fold in FOLDS]
    payload = _aggregate(FULL_ROOT, batches, bootstrap_replicates)
    checksums = {str(path.relative_to(FULL_ROOT)).replace("\\", "/"): _sha256(path) for path in sorted(FULL_ROOT.rglob("*")) if path.is_file() and path.name not in {"CHECKSUMS.json", "progress.json", "batch_registry.json"}}
    _write_json(FULL_ROOT / "CHECKSUMS.json", checksums)
    progress.update({"status": "PASS", "completed_at": _now(), "completed_batch_count": len(batches), "full_cohort_record_count": payload["overall"]["record_count"]})
    _write_json(progress_path, progress)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    return run(bootstrap_replicates=args.bootstrap_replicates, resume=not args.no_resume)


if __name__ == "__main__":
    raise SystemExit(main())
