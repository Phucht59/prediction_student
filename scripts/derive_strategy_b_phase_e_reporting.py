"""Create an immutable reporting-only correction of a completed Phase E bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from scripts.run_strategy_b_phase_e_prediction import (
    ARTIFACT_ROOT, MINIMUM_OUTPUTS, REPORT_ROOT, _conclusion, _provenance,
)
from src.strategy_b_phase_ab import sha256_file, write_json
from src.strategy_b_phase_e_prediction import OVERALL_FINALISTS, HYBRID_FINALISTS, choose_final, seed_stability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _validate_checksums(root: Path) -> None:
    checks = json.loads((root / "artifact_checksums.json").read_text(encoding="utf-8"))
    failures = [name for name, expected in checks.items() if not (root / name).is_file() or sha256_file(root / name) != expected]
    if failures:
        raise RuntimeError(f"Source artifact checksums failed: {failures[:5]}")


def main() -> None:
    args = parse_args()
    source = ARTIFACT_ROOT / args.source_run_id
    target, report = ARTIFACT_ROOT / args.run_id, REPORT_ROOT / args.run_id
    target_tmp, report_tmp = ARTIFACT_ROOT / f".{args.run_id}.tmp", REPORT_ROOT / f".{args.run_id}.tmp"
    if not source.is_dir() or any(path.exists() for path in [target, report, target_tmp, report_tmp]):
        raise FileExistsError("Source must exist and target paths must be new.")
    _validate_checksums(source)
    shutil.copytree(source, target_tmp)
    try:
        oof = pd.read_csv(target_tmp / "outer_oof_predictions.csv")
        classification = pd.read_csv(target_tmp / "classification_metrics.csv")
        updated_stability = seed_stability(oof, classification)
        updated_stability.to_csv(target_tmp / "seed_disagreement.csv", index=False)
        summary = pd.read_csv(target_tmp / "stability_summary.csv")
        for _, stable in updated_stability.iterrows():
            mask = summary["candidate_id"] == stable["candidate_id"]
            for field in ["seed_sd", "seed_sd_not_applicable", "worst_seed"]:
                summary.loc[mask, field] = stable[field]
        summary.to_csv(target_tmp / "stability_summary.csv", index=False)
        paired = pd.read_csv(target_tmp / "paired_stability_deltas.csv")
        overall, overall_reason = choose_final(summary, OVERALL_FINALISTS, paired)
        hybrid, hybrid_reason = choose_final(summary, HYBRID_FINALISTS, paired)
        decision = json.loads((target_tmp / "final_family_decision.json").read_text(encoding="utf-8"))
        decision.update({"final_overall_model": overall, "overall_reason": overall_reason, "final_thesis_hybrid_model": hybrid, "hybrid_reason": hybrid_reason})
        write_json(target_tmp / "final_family_decision.json", decision)
        strict = json.loads((target_tmp / "strict_validation.json").read_text(encoding="utf-8"))
        strict["reporting_correction"] = {
            "source_run_id": args.source_run_id,
            "reason": "seed_stability_uses_full_seed_oof_macro_f1_not_mean_fold_macro_f1",
            "raw_predictions_training_final_models_or_calibration_changed": False,
            "final_selections_changed": False,
            "corrected_seed_stability_recomputed": True,
        }
        strict["run_id"] = args.run_id
        strict["status"] = "PASS"
        write_json(target_tmp / "strict_validation.json", strict)
        (target_tmp / "phase_e_prediction_conclusion.md").write_text(_conclusion(summary, decision, strict), encoding="utf-8")
        provenance = json.loads((target_tmp / "source_provenance.json").read_text(encoding="utf-8"))
        provenance["reporting_correction"] = {**strict["reporting_correction"], "correction_source_git_commit": _provenance()["git_commit"]}
        write_json(target_tmp / "source_provenance.json", provenance)
        protocol = json.loads((target_tmp / "protocol.json").read_text(encoding="utf-8"))
        protocol["run_id"] = args.run_id
        protocol["reporting_correction"] = strict["reporting_correction"]
        write_json(target_tmp / "protocol.json", protocol)
        checks = {path.relative_to(target_tmp).as_posix(): sha256_file(path) for path in sorted(target_tmp.rglob("*")) if path.is_file() and path.name not in {"artifact_checksums.json", "run_state.json"}}
        write_json(target_tmp / "artifact_checksums.json", checks)
        state = json.loads((target_tmp / "run_state.json").read_text(encoding="utf-8"))
        state["reporting_correction"] = strict["reporting_correction"]
        write_json(target_tmp / "run_state.json", state)
        missing = [name for name in MINIMUM_OUTPUTS if not (target_tmp / name).is_file()]
        if missing:
            raise RuntimeError(f"Derived artifact missing required files: {missing}")
        report_tmp.mkdir(parents=True)
        for path in target_tmp.iterdir():
            if path.is_file():
                shutil.copy2(path, report_tmp / path.name)
        os.replace(target_tmp, target)
        os.replace(report_tmp, report)
        print(json.dumps({"status": "PASS", "artifact_path": str(target), "report_path": str(report), "final_overall": overall, "final_hybrid": hybrid}))
    except Exception:
        shutil.rmtree(target_tmp, ignore_errors=True)
        shutil.rmtree(report_tmp, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
