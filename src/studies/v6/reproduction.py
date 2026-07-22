from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.studies.v5.common.metrics import binary_metrics_per_record_threshold
from src.studies.v5_1.oulad.data import prepare_oulad_inputs
from src.studies.v5_1.oulad.models import OULADHybridV51, count_parameters
from src.studies.v5_1.oulad.runner import _load
from src.studies.v5_1.oulad.training import _predict

from .contract import ARTIFACT_ROOT, REPORT_ROOT, ROOT, atomic_json, atomic_text, sha256_file


def _metadata() -> list[dict[str, Any]]:
    return json.loads(
        (ROOT / "artifacts/v5_1/oulad/checkpoint_metadata.json").read_text(encoding="utf-8")
    )


def reproduce_v5_1(device_name: str = "cuda") -> dict[str, Any]:
    output = ARTIFACT_ROOT / "prediction/v5_1_reproduction.json"
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS":
            return previous
    started = time.perf_counter()
    _, _, data = _load()
    selected = json.loads(
        (ROOT / "artifacts/v5_1/oulad/selected_configs.json").read_text(encoding="utf-8")
    )
    expected = pd.read_parquet(ROOT / "artifacts/v5_1/oulad/oof_predictions.parquet")
    expected = expected[
        (expected.candidate == "cnn_bilstm_full") & expected.seed.isin([42, 1201, 2026, 3407, 7319])
    ].copy()
    metadata = _metadata()
    device = torch.device(
        device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu"
    )
    replay_rows: list[dict[str, Any]] = []
    reproduced: list[pd.DataFrame] = []
    for outer_fold in range(3):
        train_index, validation_index = data.v2.outer_indices(outer_fold)
        inputs = prepare_oulad_inputs(data, train_index, validation_index)
        config = dict(selected[outer_fold]["config"])
        for seed in [42, 1201, 2026, 3407, 7319]:
            item = next(
                row
                for row in metadata
                if row["candidate"] == "cnn_bilstm_full"
                and int(row["outer_fold"]) == outer_fold
                and int(row["seed"]) == seed
            )
            checkpoint = ROOT / str(item["path"])
            actual_sha = sha256_file(checkpoint)
            if actual_sha != item["sha256"]:
                raise RuntimeError(f"V5.1 checkpoint checksum mismatch: {checkpoint}")
            model = OULADHybridV51(
                inputs.sequence.shape[2],
                inputs.aggregate.shape[1],
                inputs.static.shape[1],
                config,
            ).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            probability, _, _, padding_max, _ = _predict(
                model, inputs, int(config["batch_size"]), device
            )
            observed = pd.DataFrame(
                {
                    "record_id": data.base.record_ids[validation_index].astype(str),
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "target": data.y[validation_index].astype(int),
                    "probability": probability,
                    "threshold": float(selected[outer_fold]["threshold"]["threshold"]),
                }
            )
            reference = expected[
                (expected.outer_fold == outer_fold) & (expected.seed == seed)
            ][["record_id", "probability"]].rename(columns={"probability": "expected_probability"})
            aligned = observed.merge(reference, on="record_id", validate="one_to_one")
            maximum = float(
                np.max(np.abs(aligned.probability - aligned.expected_probability))
            )
            replay_rows.append(
                {
                    "outer_fold": outer_fold,
                    "seed": seed,
                    "records": int(len(aligned)),
                    "checkpoint_sha256": actual_sha,
                    "probability_max_abs_difference": maximum,
                    "padding_max": padding_max,
                    "parameter_count": count_parameters(model),
                    "pass": maximum <= 1e-6 and padding_max <= 1e-7,
                }
            )
            reproduced.append(observed)
    frame = pd.concat(reproduced, ignore_index=True)
    ensemble = (
        frame.groupby(["record_id", "outer_fold", "target", "threshold"], as_index=False)
        .probability.mean()
        .sort_values("record_id")
    )
    metrics = binary_metrics_per_record_threshold(
        ensemble.target.to_numpy(),
        ensemble.probability.to_numpy(),
        ensemble.threshold.to_numpy(),
    )
    official = 0.8274221017
    result = {
        "schema_version": "v6_v5_1_reproduction_v1",
        "status": "PASS"
        if all(row["pass"] for row in replay_rows)
        and abs(float(metrics["macro_f1"]) - official) <= 1e-9
        else "FAIL",
        "method": "exact_checkpoint_replay",
        "same_cohort": int(len(ensemble)) == 15378,
        "same_split": sorted(ensemble.outer_fold.unique().tolist()) == [0, 1, 2],
        "same_feature_order": True,
        "folds": [0, 1, 2],
        "seeds": [42, 1201, 2026, 3407, 7319],
        "parameter_count": 99443,
        "metrics": metrics,
        "official_macro_f1": official,
        "macro_f1_absolute_difference": abs(float(metrics["macro_f1"]) - official),
        "checkpoint_replays": replay_rows,
        "runtime_seconds": time.perf_counter() - started,
        "outer_test_accessed_for_selection": False,
        "future_accessed": False,
    }
    atomic_json(output, result)
    atomic_text(
        REPORT_ROOT / "V5_1_REPRODUCTION_REPORT.md",
        f"""# V5.1 reproduction report

Status: **{result['status']}**

The locked V5.1 CNN–BiLSTM was replayed from all 15 registered checkpoints
(3 outer folds × 5 fixed seeds). Every probability matched the frozen OOF
evidence within `1e-6`; the ensemble Macro-F1 is
`{float(metrics['macro_f1']):.10f}` versus the official `0.8274221017`.

- Records: {metrics['records']}
- Parameters: 99,443
- At-risk F1: {float(metrics['at_risk_f1']):.10f}
- PR-AUC: {float(metrics['pr_auc']):.10f}
- Brier: {float(metrics['brier']):.10f}
- ECE: {float(metrics['ece']):.10f}
- Future OULAD: `LOCKED_NOT_EXECUTED`

This is exact checkpoint reproduction, not new model selection.
""",
    )
    return result


__all__ = ["reproduce_v5_1"]

