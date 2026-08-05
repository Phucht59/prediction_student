"""Run the complete four-stage recommendation and causal evidence workflow."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
LANDMARK = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
LANDMARK_MANIFEST = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows_manifest.json"
SILVER = ROOT / "artifacts/recommend_hybrid/scientific_labeling/silver_labels.parquet"
RANKER_EVIDENCE = ROOT / "artifacts/recommend_hybrid/final_stage_aware_v2/FOUR_STAGE_ACTION_HEAD_EVIDENCE.json"
CAUSAL_VALIDATION = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json"
COMPLETE_VALIDATION = ROOT / "reports/recommend_hybrid/STAGE_AWARE_COMPLETE_VALIDATION.json"
WORKFLOW_MANIFEST = ROOT / "artifacts/recommend_hybrid/STAGE_AWARE_COMPLETE_MANIFEST.json"


def _run(arguments: Sequence[str]) -> None:
    process = subprocess.run([str(PYTHON), *arguments], cwd=ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"workflow command failed with exit code {process.returncode}: "
            + " ".join(arguments)
        )


def run(
    *,
    rebuild_landmark: bool,
    rebuild_labels: bool,
    chunksize: int,
    embedding_batch_size: int,
    training_batch_size: int,
    epochs: int,
    patience: int,
    device: str,
    bootstrap: int,
) -> dict[str, object]:
    if rebuild_landmark or not LANDMARK.is_file():
        command = [
            "scripts/recommend_hybrid/causal/build_oulad_landmark_rows_memory_safe.py",
            "--output",
            str(LANDMARK),
            "--manifest",
            str(LANDMARK_MANIFEST),
            "--chunksize",
            str(chunksize),
            "--batch-size",
            str(embedding_batch_size),
        ]
        if rebuild_landmark:
            command.append("--force-bundle")
        _run(command)

    if rebuild_labels or not SILVER.is_file():
        _run(["scripts/recommend_hybrid/build_scientific_candidates.py"])
        _run(["scripts/recommend_hybrid/fit_scientific_label_model.py"])
        _run(["scripts/recommend_hybrid/generate_scientific_silver_labels.py"])

    _run(
        [
            "scripts/recommend_hybrid/final/train_four_stage_action_head.py",
            "--landmark",
            str(LANDMARK),
            "--silver-labels",
            str(SILVER),
            "--epochs",
            str(epochs),
            "--patience",
            str(patience),
            "--batch-size",
            str(training_batch_size),
            "--device",
            device,
        ]
    )
    _run(
        [
            "scripts/recommend_hybrid/causal/run_all_stage_aware_causal.py",
            "--landmark",
            str(LANDMARK),
            "--landmark-manifest",
            str(LANDMARK_MANIFEST),
            "--bootstrap",
            str(bootstrap),
            "--batch-size",
            str(embedding_batch_size),
        ]
    )
    _run(
        [
            "scripts/recommend_hybrid/validate_complete_stage_aware_release.py",
            "--ranker",
            str(RANKER_EVIDENCE),
            "--causal",
            str(CAUSAL_VALIDATION),
            "--output",
            str(COMPLETE_VALIDATION),
        ]
    )
    validation = json.loads(COMPLETE_VALIDATION.read_text(encoding="utf-8"))
    manifest = {
        "status": "COMPLETE",
        "branch": "codex/stage-aware-causal-recommendation",
        "landmark": str(LANDMARK.relative_to(ROOT)),
        "landmark_manifest": str(LANDMARK_MANIFEST.relative_to(ROOT)),
        "silver_labels": str(SILVER.relative_to(ROOT)),
        "ranker_evidence": str(RANKER_EVIDENCE.relative_to(ROOT)),
        "causal_validation": str(CAUSAL_VALIDATION.relative_to(ROOT)),
        "complete_validation": str(COMPLETE_VALIDATION.relative_to(ROOT)),
        "validation_status": validation["status"],
        "scientific_status": validation["scientific_status"],
        "runtime_authorized": False,
    }
    WORKFLOW_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    WORKFLOW_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-landmark", action="store_true")
    parser.add_argument("--rebuild-labels", action="store_true")
    parser.add_argument("--chunksize", type=int, default=750_000)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    payload = run(
        rebuild_landmark=args.rebuild_landmark,
        rebuild_labels=args.rebuild_labels,
        chunksize=args.chunksize,
        embedding_batch_size=args.embedding_batch_size,
        training_batch_size=args.training_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        bootstrap=args.bootstrap,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
