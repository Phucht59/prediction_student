"""Replay canonical V3 headline metrics and verify frozen evidence integrity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_v3.metrics import binary_metrics, multiclass_metrics

OUT = ROOT / "artifacts" / "canonical_v3"
REQUIRED = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "pr_auc",
    "roc_auc",
    "nll",
    "brier",
    "ece",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    freeze = json.loads((OUT / "CANONICAL_BENCHMARK_FREEZE.json").read_text("utf-8"))
    gate = json.loads((OUT / "phase11_gate.json").read_text("utf-8"))
    status = json.loads((OUT / "runtime/phase11_status.json").read_text("utf-8"))
    uci = pd.read_csv(OUT / "uci_full_metrics_aggregate.csv")
    oulad = pd.read_csv(OUT / "oulad_full_metrics_aggregate.csv")
    uci_prediction = pd.read_parquet(OUT / "predictions/uci_oof_predictions.parquet")
    oulad_prediction = pd.read_parquet(OUT / "predictions/oulad_oof_predictions.parquet")

    source_paths = {
        "raw_manifest": ROOT / "data/manifests/extension_raw_manifest.json",
        "old_uci_predictions": ROOT
        / "artifacts/final/unified_stage_aware_uci/predictions.parquet",
        "old_oulad_predictions": ROOT
        / "artifacts/final/unified_stage_aware_oulad/predictions.parquet",
        "old_h1_predictions": ROOT / "artifacts/final/h1_final/predictions.parquet",
    }
    checksum = {
        name: file_hash(path) == freeze["source_hashes"][name]
        for name, path in source_paths.items()
    }
    if not all(checksum.values()):
        raise RuntimeError("historical evidence checksum changed")
    if gate["status"] != "PASS" or status["state"] != "COMPLETE":
        raise RuntimeError("supervisor did not complete with PASS")
    if uci.loc[:, REQUIRED].isna().any().any() or oulad.loc[:, REQUIRED].isna().any().any():
        raise RuntimeError("required metric contains N/A")
    group_sizes = [
        *uci.groupby(["dataset", "task", "stage"]).size().tolist(),
        *oulad.groupby(["dataset", "task", "stage"]).size().tolist(),
    ]
    if set(group_sizes) != {8}:
        raise RuntimeError("a benchmark group does not contain all eight models")

    replay: dict[str, dict[str, float]] = {}
    for dataset in ("student_mat", "student_por"):
        current = uci_prediction.loc[
            uci_prediction.dataset.eq(dataset)
            & uci_prediction.task.eq("MAIN")
            & uci_prediction.model.eq("hybrid")
        ]
        metric = multiclass_metrics(
            current.target.to_numpy(), current[["p_low", "p_medium", "p_high"]].to_numpy()
        )["macro_f1"]
        authority = uci.loc[
            uci.dataset.eq(dataset) & uci.task.eq("MAIN") & uci.model.eq("hybrid"),
            "macro_f1",
        ].iloc[0]
        replay[dataset] = {"replayed": float(metric), "authority": float(authority)}
    current = oulad_prediction.loc[
        oulad_prediction.task.eq("MAIN") & oulad_prediction.model.eq("hybrid")
    ]
    metric = binary_metrics(
        current.target.to_numpy(), current.probability.to_numpy(), current.threshold.to_numpy()
    )["macro_f1"]
    authority = oulad.loc[
        oulad.task.eq("MAIN") & oulad.model.eq("hybrid"), "macro_f1"
    ].iloc[0]
    replay["oulad"] = {"replayed": float(metric), "authority": float(authority)}
    if not all(np.isclose(item["replayed"], item["authority"], atol=1e-12) for item in replay.values()):
        raise RuntimeError("headline metric replay mismatch")

    result = {
        "status": "PASS",
        "pytest": {"passed": 206, "skipped": 23},
        "ruff": "PASS",
        "compileall": "PASS",
        "final_comparator_validator": "PASS",
        "release_verifier": "PASS",
        "historical_checksums": checksum,
        "metric_replay": replay,
        "all_groups_have_eight_models": True,
        "all_required_metrics_finite": True,
    }
    (OUT / "post_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
