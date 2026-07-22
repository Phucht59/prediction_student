from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.studies.v5_1.common.artifacts import verify_checksum_manifest  # noqa: E402
from src.studies.v5_1.common.protocol import sha256_file  # noqa: E402

SEEDS = {42, 1201, 2026, 3407, 7319}


def main() -> int:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: object = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": detail})

    for dataset in ["student_mat", "student_por", "oulad"]:
        root = ROOT / "artifacts/v5_1" / dataset
        state_path = root / "run_state.json"
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
        else:
            final_value = json.loads((root / "final_metrics.json").read_text(encoding="utf-8"))
            state = {
                "status": final_value.get("status", "COMPLETE")
                if isinstance(final_value, dict)
                else "COMPLETE",
                "future_accessed": False,
            }
        add(f"{dataset}_complete", state.get("status") == "COMPLETE")
        add(f"{dataset}_future_locked", state.get("future_accessed") is False)
        registry = json.loads((root / "model_registry.json").read_text(encoding="utf-8"))
        add(f"{dataset}_seed_registry", set(registry["fixed_seeds"]) == SEEDS)
        add(f"{dataset}_checkpoint_replay", float(registry["max_replay_difference"]) <= 1e-8)
        manifest = json.loads((root / "artifact_checksums.json").read_text(encoding="utf-8"))
        add(f"{dataset}_checksums", verify_checksum_manifest(root, manifest))
        predictions = pd.read_parquet(root / "oof_predictions.parquet")
        add(f"{dataset}_outer_fold_coverage", set(predictions.outer_fold.unique()) == set(range(5 if dataset != "oulad" else 3)))
        if dataset == "oulad":
            full = predictions.loc[predictions.candidate == "cnn_bilstm_full"]
            add(f"{dataset}_five_seed_coverage", set(full.seed.unique()) == SEEDS)
            ensemble = predictions.loc[predictions.candidate == "cnn_bilstm_full_ensemble"]
            add(f"{dataset}_ensemble_unique", not ensemble.record_id.duplicated().any() and len(ensemble) == 15378)
        else:
            candidate = "cnn_bilstm_v5_1_transfer_selected" if dataset == "student_mat" else "cnn_bilstm_v5_1"
            full = predictions.loc[predictions.candidate == candidate]
            add(f"{dataset}_five_seed_coverage", set(full.seed.unique()) == SEEDS)
            add(f"{dataset}_record_seed_coverage", full.groupby("record_id").seed.nunique().eq(5).all())

    project = json.loads((ROOT / "artifacts/v5_1/final/summary.json").read_text(encoding="utf-8"))
    add("consolidated_complete", project.get("status") == "COMPLETE")
    add("future_oulad_locked", project.get("future_oulad") == "LOCKED_NOT_EXECUTED")
    bootstrap = json.loads((ROOT / "reports/v5_1/final/paired_bootstrap.json").read_text(encoding="utf-8"))
    add("paired_bootstrap_complete", bootstrap.get("status") == "COMPLETE" and bootstrap.get("replicates") == 5000)
    add("primary_comparisons_present", len(bootstrap.get("comparisons", [])) >= 3)

    expected_v5_manifest = "4a3fbef64ca760995365e2cda789a8f1491859d8d1b897bd25b803fc853f48c6"
    expected_v5_report = "20659eaaa0742b05766d4035c87565b7df579bef490573bea0d4e7a972ab0d79"
    add("v5_manifest_immutable", sha256_file(ROOT / "artifacts/v5/final/artifact_checksums.json") == expected_v5_manifest)
    add("v5_report_immutable", sha256_file(ROOT / "reports/v5/final/validation_report.json") == expected_v5_report)

    status = "PASS" if all(row["pass"] for row in checks) else "FAIL"
    result = {
        "status": status,
        "checks": checks,
        "directional_targets_are_not_correctness_gates": True,
        "future_oulad": "LOCKED_NOT_EXECUTED",
    }
    output = ROOT / "reports/v5_1/final/validation_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
