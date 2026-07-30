from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "final" / "h1_final"
FREEZE = ROOT / "artifacts" / "final_candidate_freeze"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase6_complete_freeze_and_gate_integrity() -> None:
    original = _json(FREEZE / "FINAL_H1_FREEZE_MANIFEST.json")
    copied = _json(OUT / "freeze_manifest.json")
    gate = _json(OUT / "phase6_gate.json")
    integrity = _json(OUT / "integrity_report.json")
    status = _json(OUT / "runtime" / "phase6_status.json")
    assert original == copied
    assert gate["status"] == "PASS"
    assert integrity["status"] == "PASS"
    assert status["state"] == "COMPLETE"
    assert status["completed_runs"] == 45
    assert status["failed_runs"] == 0
    assert gate["postrun_validation"]["optuna_trials"] == 0


def test_freeze_commit_precedes_outer_execution() -> None:
    integrity = _json(OUT / "integrity_report.json")
    status = _json(OUT / "runtime" / "phase6_status.json")
    freeze_commit = integrity["freeze_commit"]
    committed_at = subprocess.check_output(
        ["git", "show", "-s", "--format=%cI", freeze_commit],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    assert datetime.fromisoformat(committed_at) < datetime.fromisoformat(
        status["started_at"]
    )
    assert subprocess.run(
        [
            "git",
            "cat-file",
            "-e",
            f"{freeze_commit}:artifacts/final_candidate_freeze/"
            "FINAL_H1_FREEZE_MANIFEST.json",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
    ).returncode == 0


def test_exact_runs_seeds_hashes_and_outer_firewall() -> None:
    manifest = _json(OUT / "run_manifest.json")
    runs = manifest["runs"]
    assert len(runs) == 45
    assert {row["candidate"] for row in runs} == {
        "H1_TABULAR_RESIDUAL_EXPERT",
        "H0_CURRENT_HYBRID",
        "M0_MLP",
    }
    assert {row["outer_fold"] for row in runs} == {0, 1, 2}
    assert {row["seed"] for row in runs} == {42, 1201, 2026, 3407, 7319}
    assert not any(row["outer_labels_used_for_training"] for row in runs)
    assert not any(
        row["outer_labels_used_for_threshold_selection"] for row in runs
    )
    h1 = [row for row in runs if row["candidate"] == "H1_TABULAR_RESIDUAL_EXPERT"]
    assert len(h1) == 15
    assert len({row["candidate_hash"] for row in h1}) == 1
    assert len({row["architecture_hash"] for row in h1}) == 1
    assert {row["parameter_count"] for row in h1} == {160_492}
    assert len({row["feature_schema_hash"] for row in h1}) == 1


def test_same_checkpoint_serves_all_four_stages() -> None:
    manifest = _json(OUT / "run_manifest.json")
    mapping = pd.DataFrame(manifest["stage_mapping"])
    assert len(mapping) == 45 * 4
    expected = {
        "E1_EARLY_20PCT",
        "E2_EARLY_35PCT",
        "M1_MIDDLE_FROZEN",
        "L1_LATE_75PCT",
    }
    for _, group in mapping.groupby("run_id"):
        assert set(group.prediction_stage) == expected
        assert group.checkpoint.nunique() == 1
        assert group.checkpoint_sha256.nunique() == 1


def test_predictions_metrics_and_threshold_manifest_consistency() -> None:
    predictions = pd.read_parquet(OUT / "predictions.parquet")
    thresholds = pd.read_csv(OUT / "threshold_summary.csv")
    metrics = pd.read_csv(OUT / "fold_seed_metrics.csv")
    assert predictions.probability.between(0, 1).all()
    assert predictions.candidate.nunique() == 3
    assert predictions.prediction_stage.nunique() == 4
    assert predictions.outer_fold.nunique() == 3
    assert len(thresholds) == 3 * 3 * 4
    assert not thresholds.outer_labels_used.astype(bool).any()
    assert set(thresholds.source) == {"PHASE5_POOLED_INNER_OOF_SEED42"}
    assert len(metrics) == 3 * 3 * 5 * 4


def test_old_official_evidence_remains_byte_identical() -> None:
    integrity = _json(OUT / "integrity_report.json")
    for relative, expected in integrity["old_official_checksums"].items():
        assert _sha(ROOT / relative) == expected
