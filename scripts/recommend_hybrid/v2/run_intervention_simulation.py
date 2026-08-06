"""Run bounded action simulations through the frozen Hybrid OULAD ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_v3.oulad_data import build_canonical_bundle  # noqa: E402
from src.pipelines import oulad  # noqa: E402
from src.recommend_hybrid.contracts import Stage  # noqa: E402
from src.recommend_hybrid.prediction_adapter import HybridPredictionAdapter  # noqa: E402
from src.recommend_hybrid.v2.simulation import (  # noqa: E402
    SimulationStrength,
    simulate_action_inputs,
)
from src.recommend_hybrid.v2.taxonomy import LEARNED_ACTIONS  # noqa: E402

DEFAULT_LANDMARK = ROOT / "artifacts/recommend_hybrid/causal/input/landmark_rows.parquet"
DEFAULT_ROWS = ROOT / "artifacts/recommend_hybrid/v2/simulation_rows.parquet"
DEFAULT_SUMMARY = ROOT / "artifacts/recommend_hybrid/v2/simulation_summary.json"
DEFAULT_REPORT = ROOT / "reports/recommend_hybrid/v2/HYBRID_INTERVENTION_SENSITIVITY.md"
STAGE_SOURCE = {
    "EARLY_20": "E1_EARLY_20PCT",
    "EARLY_35": "E2_EARLY_35PCT",
    "MIDDLE_50": "M1_MIDDLE_50PCT",
    "LATE_75": "L1_LATE_75PCT",
}
STAGE_ENUM = {
    "EARLY_20": Stage.EARLY_20,
    "EARLY_35": Stage.EARLY_35,
    "MIDDLE_50": Stage.MIDDLE_50,
    "LATE_75": Stage.LATE_75,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _restore_preprocessor(state: dict[str, Any]) -> Any:
    preprocessor = oulad._DeepPreprocessor()
    for key, value in state.items():
        setattr(preprocessor, key, value)
    return preprocessor


def _preprocessor_for_adapter(adapter: HybridPredictionAdapter) -> Any:
    reference = adapter.checkpoint_references[0]
    payload = torch.load(ROOT / reference.path, map_location="cpu", weights_only=False)
    if "preprocessor" not in payload:
        raise KeyError("frozen Hybrid checkpoint has no train-fitted preprocessor")
    return _restore_preprocessor(payload["preprocessor"])


def _predict_batches(
    adapter: HybridPredictionAdapter,
    *,
    frame: pd.DataFrame,
    sequence: np.ndarray,
    lengths: np.ndarray,
    mask: np.ndarray,
    raw_aggregate: np.ndarray,
    preprocessor: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    aggregate, static = preprocessor.transform(frame, raw_aggregate)
    output: list[np.ndarray] = []
    for start in range(0, len(frame), batch_size):
        stop = min(len(frame), start + batch_size)
        inputs = {
            "sequence": torch.from_numpy(sequence[start:stop].astype(np.float32)).to(device),
            "lengths": torch.from_numpy(lengths[start:stop].astype(np.int64)).to(device),
            "mask": torch.from_numpy(mask[start:stop].astype(np.float32)).to(device),
            "aggregate": torch.from_numpy(aggregate[start:stop].astype(np.float32)).to(device),
            "static": torch.from_numpy(static[start:stop].astype(np.float32)).to(device),
        }
        prediction = adapter.predict(inputs)
        output.append(prediction.probabilities[:, 1].detach().cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def _summary(rows: pd.DataFrame) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for (stage, action, strength), frame in rows.groupby(
        ["stage", "action_id", "strength"],
        sort=True,
    ):
        delta = frame["risk_delta"].to_numpy(dtype=float)
        groups.append(
            {
                "stage": str(stage),
                "action_id": str(action),
                "strength": str(strength),
                "rows": int(len(frame)),
                "mean_baseline_risk": float(frame["baseline_risk"].mean()),
                "mean_simulated_risk": float(frame["simulated_risk"].mean()),
                "mean_risk_delta": float(np.mean(delta)),
                "median_risk_delta": float(np.median(delta)),
                "positive_reduction_fraction": float(np.mean(delta > 0.0)),
                "threshold_crossing_fraction": float(frame["threshold_crossed"].mean()),
                "constraint_violation_count": int(frame["constraint_violation_count"].sum()),
            }
        )
    monotonic_rows: list[dict[str, object]] = []
    pivot = rows.pivot_table(
        index=["record_id", "student_id", "stage", "outer_fold", "action_id"],
        columns="strength",
        values="simulated_risk",
        aggfunc="first",
    )
    required = [strength.value for strength in SimulationStrength]
    if set(required).issubset(pivot.columns):
        monotonic = (
            (pivot[required[1]] <= pivot[required[0]] + 1.0e-8)
            & (pivot[required[2]] <= pivot[required[1]] + 1.0e-8)
        )
        table = monotonic.rename("monotonic").reset_index()
        for (stage, action), frame in table.groupby(["stage", "action_id"]):
            monotonic_rows.append(
                {
                    "stage": str(stage),
                    "action_id": str(action),
                    "monotonic_strength_fraction": float(frame["monotonic"].mean()),
                }
            )
    return {
        "status": "COMPLETE",
        "claim_boundary": "MODEL_BASED_SENSITIVITY_NOT_CAUSAL_EFFECT",
        "rows": int(len(rows)),
        "groups": groups,
        "monotonicity": monotonic_rows,
        "constraint_violation_count": int(rows["constraint_violation_count"].sum()),
    }


def run(
    *,
    landmark_path: Path,
    rows_path: Path,
    summary_path: Path,
    report_path: Path,
    batch_size: int,
    device_name: str,
) -> dict[str, object]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    landmark = pd.read_parquet(landmark_path)
    required = {"record_id", "stage", "action_id"}
    missing = sorted(required.difference(landmark.columns))
    if missing:
        raise KeyError(f"landmark table is missing columns: {missing}")
    available = {
        (str(row.record_id), str(row.stage), str(row.action_id))
        for row in landmark.loc[landmark["action_id"].isin(LEARNED_ACTIONS)].itertuples()
    }
    bundle = build_canonical_bundle()
    output_parts: list[pd.DataFrame] = []
    checkpoint_hashes: set[str] = set()

    for stage, source in STAGE_SOURCE.items():
        data = bundle.stages[source]
        for fold in sorted(data.frame["outer_fold"].astype(int).unique()):
            selected = data.frame["outer_fold"].astype(int).eq(int(fold)).to_numpy()
            indices = np.flatnonzero(selected)
            if not len(indices):
                continue
            frame = data.frame.iloc[indices].reset_index(drop=True)
            sequence = data.sequence[indices]
            lengths = data.lengths[indices]
            mask = data.mask[indices]
            raw_aggregate = data.aggregate[indices]
            context = raw_aggregate[:, -len(oulad.CONTEXT_COLUMNS) :]
            adapter = HybridPredictionAdapter.from_manifest(
                ROOT,
                stage=STAGE_ENUM[stage],
                fold=int(fold),
            )
            for model in adapter.models:
                model.to(device).eval()
            checkpoint_hashes.update(ref.sha256 for ref in adapter.checkpoint_references)
            preprocessor = _preprocessor_for_adapter(adapter)
            baseline = _predict_batches(
                adapter,
                frame=frame,
                sequence=sequence,
                lengths=lengths,
                mask=mask,
                raw_aggregate=raw_aggregate,
                preprocessor=preprocessor,
                batch_size=batch_size,
                device=device,
            )
            record_ids = frame["base_record_id"].astype(str).to_numpy()
            student_ids = frame["id_student"].astype(str).to_numpy()
            for action_id in LEARNED_ACTIONS:
                applicable = np.asarray(
                    [(record_id, stage, action_id) in available for record_id in record_ids],
                    dtype=bool,
                )
                if not applicable.any():
                    continue
                for strength in SimulationStrength:
                    simulated = simulate_action_inputs(
                        full_sequence=sequence,
                        lengths=lengths,
                        stage_context=context,
                        action_id=action_id,
                        strength=strength,
                        applicable=applicable,
                    )
                    risk = _predict_batches(
                        adapter,
                        frame=frame,
                        sequence=simulated.full_sequence,
                        lengths=lengths,
                        mask=mask,
                        raw_aggregate=simulated.raw_aggregate,
                        preprocessor=preprocessor,
                        batch_size=batch_size,
                        device=device,
                    )
                    keep = applicable
                    delta = baseline[keep] - risk[keep]
                    output_parts.append(
                        pd.DataFrame(
                            {
                                "record_id": record_ids[keep],
                                "student_id": student_ids[keep],
                                "stage": stage,
                                "outer_fold": int(fold),
                                "action_id": action_id,
                                "strength": strength.value,
                                "baseline_risk": baseline[keep],
                                "simulated_risk": risk[keep],
                                "risk_delta": delta,
                                "decision_threshold": adapter.decision_threshold,
                                "threshold_crossed": (
                                    (baseline[keep] >= adapter.decision_threshold)
                                    & (risk[keep] < adapter.decision_threshold)
                                ),
                                "constraint_violation_count": len(
                                    simulated.constraint_violations
                                ),
                            }
                        )
                    )
            for model in adapter.models:
                model.cpu()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not output_parts:
        raise RuntimeError("simulation produced no rows")
    rows = pd.concat(output_parts, ignore_index=True)
    payload = _summary(rows)
    payload.update(
        {
            "device": str(device),
            "landmark": str(landmark_path.relative_to(ROOT)),
            "landmark_sha256": _sha256(landmark_path),
            "checkpoint_hashes": sorted(checkpoint_hashes),
            "strength_order": [strength.value for strength in SimulationStrength],
            "learned_actions": list(LEARNED_ACTIONS),
            "frozen_hybrid_modified": False,
        }
    )
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(rows_path, index=False)
    payload["row_artifact"] = str(rows_path.relative_to(ROOT))
    payload["row_artifact_sha256"] = _sha256(rows_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Frozen Hybrid Behaviour-Intervention Sensitivity",
        "",
        "This is a constrained model-response analysis, not causal evidence.",
        "",
        "| Stage | Action | Strength | Rows | Mean risk delta | Positive reduction | Threshold crossing |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["groups"]:
        lines.append(
            "| {stage} | {action_id} | {strength} | {rows} | {mean_risk_delta:.6f} | {positive_reduction_fraction:.4f} | {threshold_crossing_fraction:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"Constraint violations: **{payload['constraint_violation_count']}**.",
            "",
            "A positive risk delta means the frozen Hybrid model assigned lower risk after a bounded, internally consistent behaviour edit. It does not prove that displaying a recommendation causes the behaviour or changes the final grade.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--landmark", type=Path, default=DEFAULT_LANDMARK)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    payload = run(
        landmark_path=args.landmark,
        rows_path=args.rows,
        summary_path=args.summary,
        report_path=args.report,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    print(json.dumps({"status": payload["status"], "rows": payload["rows"]}))


if __name__ == "__main__":
    main()
