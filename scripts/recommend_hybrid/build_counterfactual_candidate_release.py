"""Build a reproducible counterfactual recommender candidate release locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/recommend_hybrid/counterfactual"
REGISTRY = OUT / "CANDIDATE_RELEASE_REGISTRY.json"
CHECKSUMS = OUT / "CHECKSUMS.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run(name: str, command: list[str]) -> dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "return_code": process.returncode,
        "status": "PASS" if process.returncode == 0 else "FAIL",
    }


def _artifact_registry(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _write_checksums(paths: list[Path], *, status: str) -> None:
    rows = []
    missing = []
    for path in [*paths, REGISTRY]:
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        else:
            missing.append(str(path.relative_to(ROOT)))
    _write_json(
        CHECKSUMS,
        {
            "schema_version": "counterfactual_candidate_checksums_v1",
            "status": status,
            "claim_boundary": "TECHNICAL_ARTIFACT_INTEGRITY_ONLY_NOT_CAUSAL_EFFECT",
            "files": sorted(rows, key=lambda row: row["path"]),
            "missing_required_files": sorted(missing),
        },
    )


def build(
    *,
    max_records_per_fold_stage: int,
    verify_hashes: bool,
    skip_historical: bool,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    python = sys.executable
    preflight = [
        python,
        "scripts/recommend_hybrid/preflight_counterfactual_evaluation.py",
    ]
    if verify_hashes:
        preflight.append("--verify-hashes")
    steps = [
        _run(
            "checkpoint_authority",
            [python, "scripts/recommend_hybrid/validate_checkpoint_authority.py"],
        ),
        _run("preflight", preflight),
    ]
    if all(step["status"] == "PASS" for step in steps):
        steps.append(
            _run(
                "technical_validation",
                [python, "scripts/recommend_hybrid/validate_counterfactual.py"],
            )
        )
    if all(step["status"] == "PASS" for step in steps):
        steps.append(
            _run(
                "real_checkpoint_smoke",
                [python, "scripts/recommend_hybrid/smoke_real_checkpoint_counterfactual.py"],
            )
        )
    if all(step["status"] == "PASS" for step in steps):
        steps.append(
            _run(
                "outer_fold_evaluation",
                [
                    python,
                    "scripts/recommend_hybrid/evaluate_counterfactual_recommender.py",
                    "--folds",
                    "0,1,2",
                    "--stages",
                    (
                        "E1_EARLY_20PCT,E2_EARLY_35PCT,"
                        "M1_MIDDLE_FROZEN,L1_LATE_75PCT"
                    ),
                    "--seeds",
                    "42,1201,2026,3407,7319",
                    "--max-records-per-fold-stage",
                    str(max_records_per_fold_stage),
                    "--bootstrap-replicates",
                    str(bootstrap_replicates),
                ],
            )
        )
    if (
        not skip_historical
        and all(step["status"] == "PASS" for step in steps)
    ):
        steps.append(
            _run(
                "historical_trajectory_validation",
                [
                    python,
                    "scripts/recommend_hybrid/evaluate_historical_trajectories.py",
                ],
            )
        )

    expected = [
        OUT / "preflight.json",
        OUT / "checkpoint_authority_validation.json",
        OUT / "real_checkpoint_smoke.json",
        OUT / "validation.json",
        OUT / "evaluation.json",
        OUT / "evaluation_rows.csv",
        OUT / "action_scores.csv",
        ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_VALIDATION.md",
        ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_EVALUATION.md",
        ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_CANDIDATE_RELEASE.md",
    ]
    if not skip_historical:
        expected.extend(
            [
                OUT / "historical_trajectory.json",
                OUT / "historical_trajectory_rows.csv",
                ROOT
                / "reports/recommend_hybrid/HISTORICAL_TRAJECTORY_VALIDATION.md",
            ]
        )
    candidate_report = ROOT / "reports/recommend_hybrid/COUNTERFACTUAL_CANDIDATE_RELEASE.md"
    report_lines = [
        "# Counterfactual Candidate Release",
        "",
        f"- Status: **{'PASS' if all(step['status'] == 'PASS' for step in steps) else 'FAIL'}**",
        "- Candidate state: `CANDIDATE_VALIDATED_PENDING_EXPERT_REVIEW` only when every release gate passes.",
        "- Claim boundary: `MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT`.",
        "",
        "## Steps",
        "",
    ]
    report_lines.extend(
        f"- `{step['name']}`: **{step['status']}** (return code `{step['return_code']}`)"
        for step in steps
    )
    if any(step["status"] != "PASS" for step in steps):
        report_lines.extend(
            [
                "",
                "## Remaining blockers",
                "",
                "- Do not merge or promote this candidate until the failed authority gate is repaired and rerun with the real release checkpoint.",
                "- No causal effectiveness, grade improvement, or expert-validation claim is made.",
            ]
        )
    candidate_report.parent.mkdir(parents=True, exist_ok=True)
    candidate_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    status = "PASS" if all(step["status"] == "PASS" for step in steps) else "FAIL"
    registry = {
        "schema_version": "counterfactual_candidate_release_v1",
        "generated_at": _utc_now(),
        "release_status": (
            "CANDIDATE_VALIDATED_PENDING_EXPERT_REVIEW"
            if status == "PASS"
            else "CANDIDATE_FAILED_GATE"
        ),
        "status": status,
        "system_id": "hybrid_cnn_bilstm_counterfactual_recommender",
        "recommendation_component": (
            "CONSTRAINED_COUNTERFACTUAL_RISK_REDUCTION_RECOMMENDER"
        ),
        "prediction_backbone": "FROZEN_HYBRID_CNN_BILSTM",
        "claim_boundary": (
            "MODEL_ESTIMATED_RISK_REDUCTION_NOT_CAUSAL_EFFECT"
        ),
        "configuration": {
            "max_records_per_fold_stage": max_records_per_fold_stage,
            "verify_hashes": verify_hashes,
            "skip_historical": skip_historical,
            "bootstrap_replicates": bootstrap_replicates,
            "folds": [0, 1, 2],
            "stages": [
                "E1_EARLY_20PCT",
                "E2_EARLY_35PCT",
                "M1_MIDDLE_FROZEN",
                "L1_LATE_75PCT",
            ],
            "seeds": [42, 1201, 2026, 3407, 7319],
        },
        "steps": steps,
        "artifacts": _artifact_registry(expected),
        "scientific_guards": {
            "expert_labels_required": False,
            "silver_labels_used": False,
            "outcome_labels_used_for_ranking": False,
            "causal_effect_claimed": False,
            "final_release_promoted": False,
        },
    }
    _write_json(REGISTRY, registry)
    _write_checksums(expected, status=status)
    return registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-records-per-fold-stage",
        type=int,
        default=100,
        help="Use 0 to evaluate every outer-validation row.",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--skip-historical", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.max_records_per_fold_stage < 0:
        raise ValueError("max records cannot be negative")
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    registry = build(
        max_records_per_fold_stage=args.max_records_per_fold_stage,
        verify_hashes=args.verify_hashes,
        skip_historical=args.skip_historical,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    print(json.dumps(registry, indent=2, sort_keys=True))
    return 0 if registry["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
