"""Create an immutable corrected-report bundle without changing Phase C evidence.

This command is intentionally limited to the pre-identified runtime aggregation
correction.  It verifies every source checksum and all completeness gates, then
copies the evidence to a new run id and recomputes only reporting/provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.run_strategy_b_phase_c import (
    FINAL_REPORT_ROOT,
    FINAL_ROOT,
    MINIMUM_OUTPUTS,
    _conclusion,
    _run,
    _source_provenance,
)
from src.strategy_b_phase_ab import sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def _verify_source(source: Path) -> dict:
    checksums = json.loads((source / "artifact_checksums.json").read_text(encoding="utf-8"))
    failures = [
        relative for relative, expected in checksums.items()
        if not (source / relative).is_file() or sha256_file(source / relative) != expected
    ]
    strict = json.loads((source / "strict_validation.json").read_text(encoding="utf-8"))
    jobs = pd.read_csv(source / "job_ledger.csv")
    trials = pd.read_csv(source / "trial_history.csv")
    oof = pd.read_csv(source / "outer_oof_predictions.csv")
    checkpoints = json.loads((source / "checkpoint_checksums.json").read_text(encoding="utf-8"))
    checks = {
        "source_checksum_entries": len(checksums),
        "source_checksum_failures": failures,
        "strict_pass": strict.get("status") == "PASS",
        "jobs_complete": len(jobs) == 2805 and bool((jobs["status"] == "completed").all()),
        "trials_complete": len(trials) == 900 and bool((trials["state"] == "COMPLETE").all()),
        "oof_complete": len(oof) == 9 * 3 * 316 and all(
            len(frame) == 316 for _, frame in oof.groupby(["candidate_id", "seed"])
        ),
        "checkpoints_complete": checkpoints.get("checkpoint_count") == 100 and checkpoints.get("all_reproduced") is True,
        "metrics_exact": float(strict.get("metric_recomputation_max_abs_difference", 1.0)) == 0.0,
    }
    if failures or not all(value for key, value in checks.items() if key not in {"source_checksum_entries", "source_checksum_failures"}):
        raise RuntimeError(f"Source evidence cannot be derived safely: {checks}")
    return checks


def main() -> None:
    args = parse_args()
    source = FINAL_ROOT / args.source_run_id
    destination = FINAL_ROOT / args.run_id
    report_destination = FINAL_REPORT_ROOT / args.run_id
    temporary = FINAL_ROOT / f".{args.run_id}.tmp"
    report_temporary = FINAL_REPORT_ROOT / f".{args.run_id}.tmp"
    if not source.is_dir() or any(path.exists() for path in [destination, report_destination, temporary, report_temporary]):
        raise FileExistsError("Source must exist and corrected destination paths must be new.")
    source_checks = _verify_source(source)
    shutil.copytree(source, temporary)
    report_temporary.mkdir(parents=True)
    try:
        trial_history = pd.read_csv(temporary / "trial_history.csv")
        jobs = pd.read_csv(temporary / "job_ledger.csv")
        summary = pd.read_csv(temporary / "model_summary.csv")
        search = trial_history.groupby("candidate_id", as_index=False)["runtime_seconds"].sum().rename(
            columns={"runtime_seconds": "search_runtime_seconds"}
        )
        outer = jobs[jobs["stage"] == "outer_evaluation"].groupby("candidate_id", as_index=False)["runtime_seconds"].sum().rename(
            columns={"runtime_seconds": "outer_runtime_seconds"}
        )
        trial_counts = trial_history.groupby(["candidate_id", "state"]).size().unstack(fill_value=0).reset_index().rename(
            columns={"COMPLETE": "actual_completed_trials", "PRUNED": "actual_pruned_trials", "FAIL": "actual_failed_trials"}
        )
        for column in ["actual_completed_trials", "actual_pruned_trials", "actual_failed_trials"]:
            if column not in trial_counts:
                trial_counts[column] = 0
        fit_stages = trial_history.groupby("candidate_id", as_index=False)["actual_fit_stages"].sum().rename(
            columns={"actual_fit_stages": "actual_inner_fit_stages"}
        )
        runtime = pd.DataFrame({"candidate_id": summary["candidate_id"]}).merge(search, on="candidate_id", how="left").merge(
            outer, on="candidate_id", how="left"
        ).merge(trial_counts, on="candidate_id", how="left").merge(fit_stages, on="candidate_id", how="left").fillna(0)
        runtime["runtime_seconds"] = runtime["search_runtime_seconds"] + runtime["outer_runtime_seconds"]
        runtime["actual_total_fit_stages"] = runtime["candidate_id"].map(
            jobs.groupby("candidate_id")["fit_stages_completed"].sum()
        ).fillna(0)
        runtime.to_csv(temporary / "runtime_summary.csv", index=False)
        runtime_lookup = runtime.set_index("candidate_id")["runtime_seconds"]
        summary["runtime_seconds"] = summary["candidate_id"].map(runtime_lookup)
        summary.to_csv(temporary / "model_summary.csv", index=False)

        original_test = json.loads((temporary / "test_report.json").read_text(encoding="utf-8"))
        write_json(temporary / "training_test_report.json", original_test)
        started = time.perf_counter()
        completed = _run([sys.executable, "-m", "pytest", "-q", "-rs"])
        corrected_test = {
            "official": True, "status": "PASS" if completed.returncode == 0 else "FAIL",
            "return_code": completed.returncode, "duration_seconds": time.perf_counter() - started,
            "stdout": completed.stdout, "stderr": completed.stderr,
            "training_pre_run_report_preserved_as": "training_test_report.json",
            "postgres_integration_waiver": original_test.get("postgres_integration_waiver"),
        }
        write_json(temporary / "test_report.json", corrected_test)
        if completed.returncode != 0:
            raise RuntimeError("Corrected reporting source failed the full test suite.")

        strict_path = temporary / "strict_validation.json"
        strict = json.loads(strict_path.read_text(encoding="utf-8"))
        strict["run_id"] = args.run_id
        strict["reporting_correction"] = {
            "derived_from_run_id": args.source_run_id,
            "scope": "runtime aggregation includes inner search plus outer evaluation",
            "predictions_metrics_selections_changed": False,
            "source_validation": source_checks,
        }
        write_json(strict_path, strict)
        protocol_path = temporary / "protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        protocol["run_id"] = args.run_id
        protocol["reporting_correction"] = strict["reporting_correction"]
        write_json(protocol_path, protocol)
        provenance_path = temporary / "source_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        correction_provenance = _source_provenance()
        provenance["reporting_correction"] = {
            "derived_from_run_id": args.source_run_id,
            "git_commit": correction_provenance["git_commit"],
            "source_tree_hash": correction_provenance["source_tree_hash"],
            "predictions_metrics_selections_changed": False,
        }
        write_json(provenance_path, provenance)
        gates = json.loads((temporary / "conditional_gate_assessment.json").read_text(encoding="utf-8"))
        paired = pd.read_csv(temporary / "paired_model_deltas.csv")
        (temporary / "phase_c_conclusion.md").write_text(
            _conclusion(summary, paired, strict, gates, "full"), encoding="utf-8"
        )
        write_json(temporary / "run_state.json", {
            "status": "completed", "strict_status": "PASS", "stage": "full",
            "run_id": args.run_id, "derived_from_run_id": args.source_run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "reporting_correction_only": True,
        })
        checksums = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path.name not in {"artifact_checksums.json", "run_state.json"}
        }
        write_json(temporary / "artifact_checksums.json", checksums)
        missing = [name for name in MINIMUM_OUTPUTS if not (temporary / name).is_file()]
        if missing:
            raise RuntimeError(f"Corrected bundle misses required artifacts: {missing}")
        for path in temporary.iterdir():
            if path.is_file():
                shutil.copy2(path, report_temporary / path.name)
        os.replace(temporary, destination)
        os.replace(report_temporary, report_destination)
        print(json.dumps({"artifact_path": str(destination), "report_path": str(report_destination), "status": "PASS"}))
    except Exception:
        write_json(temporary / "run_state.json", {
            "status": "failed", "stage": "reporting_correction", "run_id": args.run_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise


if __name__ == "__main__":
    main()
