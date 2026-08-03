"""Verify deterministic full-cohort batch replay without rerunning frozen models."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_counterfactual_recommender import STAGES
from src.recommend_hybrid.counterfactual.evaluation import (
    CounterfactualEvaluationRow,
    aggregate_counterfactual_metrics,
)

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "artifacts/recommend_hybrid/counterfactual/full_cohort"
REGISTRY = ROOT / "artifacts/recommend_hybrid/counterfactual/full_evaluation_input_registry.json"
OUT = FULL / "deterministic_replay.json"
CLAIM_BOUNDARY = "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"
FOLDS = (0, 1, 2)
IDENTITY = ["student_key", "course_key", "stage", "fold"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _row_from_series(row: Any) -> CounterfactualEvaluationRow:
    fallback = str(row.fallback_reasons)
    reasons = tuple(json.loads(fallback.replace("'", '"'))) if fallback.startswith("[") else tuple()
    return CounterfactualEvaluationRow(
        student_key=str(row.student_key), course_key=str(row.course_key), stage=str(row.stage), fold=int(row.fold),
        baseline_risk=float(row.baseline_risk), decision_threshold=float(row.decision_threshold), status=str(row.status),
        top_action_id=None if pd.isna(row.top_action_id) else str(row.top_action_id),
        top_counterfactual_risk=None if pd.isna(row.top_counterfactual_risk) else float(row.top_counterfactual_risk),
        top_risk_reduction=None if pd.isna(row.top_risk_reduction) else float(row.top_risk_reduction),
        top_utility_score=None if pd.isna(row.top_utility_score) else float(row.top_utility_score),
        selected_action_count=int(row.selected_action_count), selected_workload_minutes=int(row.selected_workload_minutes),
        reference_profile_id=None if pd.isna(row.reference_profile_id) else str(row.reference_profile_id), fallback_reasons=reasons,
    )


def main() -> int:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected_batches = [f"fold_{fold}__{stage}" for stage in STAGES for fold in FOLDS]
    batch_results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    for batch_id in expected_batches:
        directory = FULL / batch_id
        checksum_path = directory / "batch_checksums.json"
        if not checksum_path.is_file():
            failures.append(f"missing checksum: {batch_id}")
            continue
        expected = json.loads(checksum_path.read_text(encoding="utf-8"))
        actual = {str(path.relative_to(directory)).replace("\\", "/"): _sha256(path) for path in sorted(directory.iterdir()) if path.is_file() and path.name != "batch_checksums.json"}
        checksum_pass = actual == expected
        rows = pd.read_csv(directory / "evaluation_rows.csv") if (directory / "evaluation_rows.csv").is_file() else pd.DataFrame()
        evaluation = json.loads((directory / "evaluation.json").read_text(encoding="utf-8")) if (directory / "evaluation.json").is_file() else {}
        config = evaluation.get("configuration", {})
        batch_pass = checksum_pass and evaluation.get("status") == "PASS" and config.get("max_records_per_fold_stage") is None and len(rows) > 0 and not rows.duplicated(IDENTITY).any()
        if not batch_pass:
            failures.append(f"invalid batch: {batch_id}")
        frames.append(rows)
        batch_results.append({"batch_id": batch_id, "status": "PASS" if batch_pass else "FAIL", "row_count": int(len(rows)), "checksum_match": checksum_pass})
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    duplicate_count = int(rows.duplicated(IDENTITY).sum()) if len(rows) else 0
    stored = json.loads((FULL / "evaluation.json").read_text(encoding="utf-8"))
    typed = [_row_from_series(row) for row in rows.itertuples(index=False)]
    recomputed = aggregate_counterfactual_metrics(typed)
    stored_metrics = stored["overall"]
    metric_checks = {
        key: abs(float(recomputed[key]) - float(stored_metrics[key])) < 1e-12
        for key in ("record_count", "scored_count", "scored_coverage", "fallback_count", "fallback_rate", "mean_top_risk_reduction", "median_top_risk_reduction", "success_at_0_01", "success_at_0_05", "threshold_crossing_rate")
    }
    progress = json.loads((FULL / "progress.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "full_cohort_deterministic_replay_v1",
        "status": "PASS" if not failures and duplicate_count == 0 and all(metric_checks.values()) and progress.get("status") == "PASS" else "FAIL",
        "claim_boundary": CLAIM_BOUNDARY,
        "verification_mode": "CHECKSUM_AND_AGGREGATE_REPLAY",
        "model_rerun": False,
        "reason_model_rerun": "Existing atomic batches are immutable; authority guard prevents unsafe rerun/resume after source commit changes.",
        "registry_source_commit": registry.get("source_commit"),
        "original_source_commit": registry.get("original_source_commit"),
        "current_commit": _git("rev-parse", "HEAD"),
        "batch_count": len(batch_results),
        "passed_batch_count": sum(item["status"] == "PASS" for item in batch_results),
        "batch_results": batch_results,
        "global_record_count": int(len(rows)),
        "duplicate_identity_count": duplicate_count,
        "aggregate_metric_checks": metric_checks,
        "progress_status": progress.get("status"),
        "failures": failures,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "batch_count": len(batch_results), "record_count": len(rows), "duplicate_identity_count": duplicate_count}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
