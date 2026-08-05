"""Run the complete stage-aware causal recommendation evidence workflow.

This orchestrator does not build the landmark source table. The source table
must first be generated from frozen OULAD checkpoints and cutoff-safe raw data
according to the local execution protocol.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(sys.executable)
DEFAULT_LANDMARK = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
DEFAULT_INPUT_DIR = ROOT / "artifacts/recommend_hybrid/causal/input"
DEFAULT_STAGE_EMBEDDINGS = DEFAULT_INPUT_DIR / "embeddings_by_stage"
DEFAULT_IMBALANCE_DIR = ROOT / "artifacts/recommend_hybrid/causal/imbalance"
DEFAULT_TRIAL_DIR = ROOT / "artifacts/recommend_hybrid/causal/target_trials"
DEFAULT_VALIDATION = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_VALIDATION.json"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/causal/STAGE_AWARE_CAUSAL_RESULTS.md"
STAGES = ("EARLY_20", "EARLY_35", "MIDDLE_50", "LATE_75")


def _run(arguments: Sequence[str]) -> None:
    process = subprocess.run(
        [str(PYTHON), *arguments],
        cwd=ROOT,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"workflow command failed with exit code {process.returncode}: "
            + " ".join(arguments)
        )


def run(
    *,
    landmark_path: Path,
    input_dir: Path,
    stage_embedding_dir: Path,
    imbalance_dir: Path,
    trial_dir: Path,
    validation_path: Path,
    report_path: Path,
    splits: int,
    bootstrap: int,
    seed: int,
) -> dict[str, object]:
    if not landmark_path.is_file():
        raise FileNotFoundError(
            f"cutoff-safe landmark table is required before orchestration: {landmark_path}"
        )

    _run(
        [
            "scripts/recommend_hybrid/causal/prepare_causal_inputs.py",
            "--input",
            str(landmark_path),
            "--output-dir",
            str(input_dir),
        ]
    )
    _run(
        [
            "scripts/recommend_hybrid/causal/prepare_stage_embedding_archives.py",
            "--input",
            str(landmark_path),
            "--output-dir",
            str(stage_embedding_dir),
        ]
    )

    imbalance_dir.mkdir(parents=True, exist_ok=True)
    overall_imbalance = imbalance_dir / "metrics.json"
    _run(
        [
            "scripts/recommend_hybrid/causal/run_imbalance_evidence.py",
            "--input",
            str(input_dir / "frozen_embeddings.npz"),
            "--output",
            str(overall_imbalance),
            "--seed",
            str(seed),
        ]
    )
    stage_outputs: dict[str, str] = {}
    for stage in STAGES:
        stage_input = stage_embedding_dir / f"frozen_embeddings_{stage.lower()}.npz"
        stage_output = imbalance_dir / f"metrics_{stage.lower()}.json"
        _run(
            [
                "scripts/recommend_hybrid/causal/run_imbalance_evidence.py",
                "--input",
                str(stage_input),
                "--output",
                str(stage_output),
                "--seed",
                str(seed),
            ]
        )
        stage_outputs[stage] = str(stage_output.relative_to(ROOT))

    _run(
        [
            "scripts/recommend_hybrid/causal/run_stage_aware_target_trials.py",
            "--input",
            str(input_dir / "target_trials.npz"),
            "--output-dir",
            str(trial_dir),
            "--splits",
            str(splits),
            "--bootstrap",
            str(bootstrap),
            "--seed",
            str(seed),
        ]
    )
    _run(
        [
            "scripts/recommend_hybrid/causal/validate_stage_aware_causal_release.py",
            "--causal",
            str(trial_dir / "stage_action_effects.json"),
            "--individual",
            str(trial_dir / "individual_effects.csv"),
            "--imbalance",
            str(overall_imbalance),
            "--output",
            str(validation_path),
        ]
    )
    _run(
        [
            "scripts/recommend_hybrid/causal/build_stage_aware_causal_report.py",
            "--causal",
            str(trial_dir / "stage_action_effects.json"),
            "--imbalance",
            str(overall_imbalance),
            "--validation",
            str(validation_path),
            "--output",
            str(report_path),
        ]
    )

    manifest = {
        "status": "COMPLETE",
        "landmark_source": str(landmark_path.relative_to(ROOT)),
        "seed": seed,
        "cross_fit_splits": splits,
        "bootstrap_iterations": bootstrap,
        "overall_imbalance": str(overall_imbalance.relative_to(ROOT)),
        "stage_imbalance": stage_outputs,
        "target_trial_summary": str(
            (trial_dir / "stage_action_effects.json").relative_to(ROOT)
        ),
        "validation": str(validation_path.relative_to(ROOT)),
        "report": str(report_path.relative_to(ROOT)),
    }
    manifest_path = ROOT / "artifacts/recommend_hybrid/causal/WORKFLOW_MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark", type=Path, default=DEFAULT_LANDMARK)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--stage-embedding-dir", type=Path, default=DEFAULT_STAGE_EMBEDDINGS
    )
    parser.add_argument("--imbalance-dir", type=Path, default=DEFAULT_IMBALANCE_DIR)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    payload = run(
        landmark_path=args.landmark,
        input_dir=args.input_dir,
        stage_embedding_dir=args.stage_embedding_dir,
        imbalance_dir=args.imbalance_dir,
        trial_dir=args.trial_dir,
        validation_path=args.validation,
        report_path=args.report,
        splits=args.splits,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
