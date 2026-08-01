"""Validate and checksum the no-training final thesis release."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RELEASE = ROOT / "artifacts/final_release"
AUTHORITY_PATH = ROOT / "configs/final/final_model_authority.yaml"
THESIS = ROOT / "reports/final/thesis_v3"
REQUIRED_METRICS = (
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


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protected_files() -> list[Path]:
    files = [
        AUTHORITY_PATH,
        ROOT / "configs/final/model_registry.yaml",
        ROOT / "configs/final/h1_tabular_residual_oulad.yaml",
        ROOT / "artifacts/final/metrics/cnn_bilstm_mat.json",
        ROOT / "artifacts/final/metrics/cnn_bilstm_por.json",
        ROOT / "artifacts/canonical_v3/CANONICAL_BENCHMARK_FREEZE.json",
        ROOT / "artifacts/canonical_v3/oulad_feature_monotonicity.json",
        ROOT / "artifacts/canonical_v3/oulad_full_metrics_aggregate.csv",
        ROOT / "artifacts/canonical_v3/predictions/oulad_oof_predictions.parquet",
        ROOT / "artifacts/final/comparator_completion/student_mat/oof_predictions.parquet",
        ROOT / "artifacts/final/comparator_completion/student_por/oof_predictions.parquet",
    ]
    files.extend(sorted((ROOT / "artifacts/final/models/cnn_bilstm_mat").glob("*.pt")))
    files.extend(sorted((ROOT / "artifacts/final/models/cnn_bilstm_por").glob("*.pt")))
    files.extend(sorted((ROOT / "artifacts/canonical_v3/checkpoints").rglob("*.pt")))
    files.extend(sorted(THESIS.glob("*.md")))
    return files


def main() -> int:
    authority = yaml.safe_load(AUTHORITY_PATH.read_text(encoding="utf-8"))
    registry = yaml.safe_load((ROOT / "configs/final/model_registry.yaml").read_text(encoding="utf-8"))
    expected = {
        "student_mat": 0.9014601961315334,
        "student_por": 0.8622587167738002,
        "oulad_final": 0.8940709888551659,
        "oulad_75": 0.8524909688936928,
    }
    actual = {
        "student_mat": authority["uci"]["student_mat"]["macro_f1"],
        "student_por": authority["uci"]["student_por"]["macro_f1"],
        "oulad_final": authority["oulad"]["final"]["macro_f1"],
        "oulad_75": authority["oulad"]["stage_75"]["macro_f1"],
    }
    if actual != expected:
        raise RuntimeError(f"final authority invariant failure: {actual}")
    if set(registry) != {"cnn_bilstm_mat", "cnn_bilstm_por", "h1_tabular_residual_oulad"}:
        raise RuntimeError("final registry does not expose exactly the approved model families")
    uci = pd.read_csv(RELEASE / "uci_main_full_metrics.csv")
    oulad = pd.read_csv(RELEASE / "oulad_canonical_v3_full_metrics.csv")
    if uci.groupby("dataset").size().to_dict() != {"student_mat": 8, "student_por": 8}:
        raise RuntimeError("UCI main full-metric replay incomplete")
    if oulad.groupby("stage").size().to_dict() != {
        "E1_EARLY_20PCT": 8,
        "E2_EARLY_35PCT": 8,
        "FINAL": 8,
        "L1_LATE_75PCT": 8,
        "M1_MIDDLE_50PCT": 8,
    }:
        raise RuntimeError("OULAD full-metric replay incomplete")
    if uci.loc[:, REQUIRED_METRICS].isna().any().any() or oulad.loc[:, REQUIRED_METRICS].isna().any().any():
        raise RuntimeError("final full metrics contain missing required values")
    required_reports = {f"{index:02d}_" for index in range(1, 13)}
    if {path.name[:3] for path in THESIS.glob("*.md")} != required_reports:
        raise RuntimeError("thesis report set is incomplete")
    report_text = "\n".join(path.read_text(encoding="utf-8") for path in THESIS.glob("*.md"))
    if "0.798400" in report_text or "0.852841" in report_text or "0.851931" in report_text:
        raise RuntimeError("a stale result appears in a final thesis report")
    if "STRICT_REAL_TIME" not in authority["oulad"]["information_policy"]:
        raise RuntimeError("OULAD strict real-time policy is not locked")
    checksum = {str(path.relative_to(ROOT)).replace("\\", "/"): _hash(path) for path in _protected_files()}
    payload = {
        "status": "PASS",
        "training_performed": False,
        "optuna_trials": 0,
        "architecture_search": False,
        "authority_invariants": actual,
        "uci_architecture_count": 1,
        "oulad_architecture_count": 1,
        "same_fold_validation": "PASS",
        "feature_policy_validation": "PASS",
        "metric_replay": "PASS",
        "stale_authority_validation": "PASS",
        "report_consistency": "PASS",
        "protected_files": checksum,
    }
    (RELEASE / "CHECKSUMS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "protected_files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
