"""Run the complete Recommendation V2 audit, simulation and evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()
    python = sys.executable

    _run([python, "scripts/recommend_hybrid/v2/audit_action_taxonomy.py"])
    _run(
        [
            python,
            "scripts/recommend_hybrid/v2/audit_assessment_timeliness_candidate.py",
        ]
    )
    if not args.skip_simulation:
        _run(
            [
                python,
                "scripts/recommend_hybrid/v2/run_intervention_simulation.py",
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
            ]
        )
    _run([python, "scripts/recommend_hybrid/v2/evaluate_full_population.py"])
    _run([python, "scripts/recommend_hybrid/v2/validate_recommendation_v2.py"])
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "device": args.device,
                "simulation_skipped": bool(args.skip_simulation),
                "validation": "reports/recommend_hybrid/v2/RECOMMENDATION_V2_VALIDATION.json",
            }
        )
    )


if __name__ == "__main__":
    main()
