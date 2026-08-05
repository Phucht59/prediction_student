"""Run stage-action target trials from a leakage-audited local archive."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend_hybrid.causal.aipw import AIPWConfig  # noqa: E402
from src.recommend_hybrid.causal.pipeline import (  # noqa: E402
    StageActionTrialData,
    StageAwareCausalEvaluator,
)
from src.recommend_hybrid.causal.protocol import STAGE_ORDER  # noqa: E402
from src.recommend_hybrid.final.actions import ACTION_ORDER  # noqa: E402

DEFAULT_INPUT = ROOT / "artifacts/recommend_hybrid/causal/input/target_trials.npz"
DEFAULT_OUTPUT = ROOT / "artifacts/recommend_hybrid/causal/target_trials"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _required_archive_arrays(data: np.lib.npyio.NpzFile) -> None:
    required = {
        "features",
        "treatment",
        "outcome",
        "groups",
        "student_ids",
        "stages",
        "action_ids",
        "baseline_progress",
        "treatment_start_progress",
        "treatment_end_progress",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"target-trial archive is missing keys: {missing}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    input_path: Path,
    output_dir: Path,
    *,
    n_splits: int,
    bootstrap_iterations: int,
    seed: int,
) -> dict[str, object]:
    with np.load(input_path, allow_pickle=False) as data:
        _required_archive_arrays(data)
        features = np.asarray(data["features"], dtype=np.float64)
        treatment = np.asarray(data["treatment"], dtype=np.int8).reshape(-1)
        outcome = np.asarray(data["outcome"], dtype=np.int8).reshape(-1)
        groups = np.asarray(data["groups"]).astype(str).reshape(-1)
        student_ids = np.asarray(data["student_ids"]).astype(str).reshape(-1)
        stages = np.asarray(data["stages"]).astype(str).reshape(-1)
        action_ids = np.asarray(data["action_ids"]).astype(str).reshape(-1)
        baseline_progress = np.asarray(data["baseline_progress"], dtype=np.float64).reshape(-1)
        treatment_start = np.asarray(
            data["treatment_start_progress"], dtype=np.float64
        ).reshape(-1)
        treatment_end = np.asarray(
            data["treatment_end_progress"], dtype=np.float64
        ).reshape(-1)

    row_count = len(treatment)
    arrays = (
        outcome,
        groups,
        student_ids,
        stages,
        action_ids,
        baseline_progress,
        treatment_start,
        treatment_end,
    )
    if features.ndim != 2 or len(features) != row_count or any(
        len(value) != row_count for value in arrays
    ):
        raise ValueError("all target-trial arrays must align")
    unknown_stage = sorted(set(stages).difference(STAGE_ORDER))
    unknown_action = sorted(set(action_ids).difference(ACTION_ORDER))
    if unknown_stage or unknown_action:
        raise ValueError(
            f"archive contains unsupported stages/actions: {unknown_stage}, {unknown_action}"
        )

    evaluator = StageAwareCausalEvaluator(
        aipw_config=AIPWConfig(n_splits=n_splits, random_state=seed),
        bootstrap_iterations=bootstrap_iterations,
        random_state=seed,
    )
    summaries: list[dict[str, object]] = []
    individual_rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        for action_id in ACTION_ORDER:
            selected = (stages == stage) & (action_ids == action_id)
            if not selected.any():
                summaries.append(
                    {
                        "status": "TRIAL_DATA_NOT_AVAILABLE",
                        "protocol": {"stage": stage, "action_id": action_id},
                        "claim_boundary": "NO_CAUSAL_EFFECT_CLAIM",
                    }
                )
                continue
            trial = StageActionTrialData(
                stage=stage,
                action_id=action_id,
                features=features[selected],
                treatment=treatment[selected],
                outcome=outcome[selected],
                groups=groups[selected],
                student_ids=student_ids[selected],
                maximum_baseline_progress=float(np.max(baseline_progress[selected])),
                minimum_treatment_progress=float(np.min(treatment_start[selected])),
                maximum_treatment_progress=float(np.max(treatment_end[selected])),
            )
            try:
                evaluation = evaluator.evaluate(trial)
            except (ValueError, RuntimeError) as exc:
                summaries.append(
                    {
                        "status": "TRIAL_EXECUTION_FAILED",
                        "protocol": {"stage": stage, "action_id": action_id},
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "claim_boundary": "NO_CAUSAL_EFFECT_CLAIM",
                    }
                )
                continue
            summaries.append(evaluation.summary())
            individual_rows.extend(
                evaluation.individual_effect_records(student_ids[selected])
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stage_action_effects.json"
    individual_path = output_dir / "individual_effects.csv"
    payload = {
        "status": "COMPLETE",
        "input": str(input_path.relative_to(ROOT)),
        "stage_order": list(STAGE_ORDER),
        "action_order": list(ACTION_ORDER),
        "cross_fit_splits": n_splits,
        "bootstrap_iterations": bootstrap_iterations,
        "random_state": seed,
        "trial_count": len(summaries),
        "trials": summaries,
    }
    summary_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(individual_path, individual_rows)
    manifest = {
        "status": "COMPLETE",
        "summary": str(summary_path.relative_to(ROOT)),
        "individual_effects": str(individual_path.relative_to(ROOT)),
        "causal_claim": (
            "OBSERVATIONAL_EFFECT_UNDER_ASSUMPTIONS_ONLY; "
            "NOT DEPLOYMENT OR RANDOMIZED-TRIAL EFFECTIVENESS"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    payload = run(
        args.input,
        args.output_dir,
        n_splits=args.splits,
        bootstrap_iterations=args.bootstrap,
        seed=args.seed,
    )
    print(json.dumps({"status": payload["status"], "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
