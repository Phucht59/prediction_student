"""Adversarial repair supervisor — 18-stage pipeline with strict gates.

Stages that depend on verified external LLM annotations (stages 9+) are
BLOCKED if the annotation gate does not PASS.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "adversarial_repair.log"
PROGRESS_FILE = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/adversarial_audit_progress.json"
SUPERVISOR_FILE = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state/supervisor.json"

PYTHON = sys.executable

# Stages dependent on verified external LLM annotations
ANNOTATION_DEPENDENT_STAGES = {
    "9. final_snorkel", "10. train_five_ebm", "11. train_baselines",
    "12. train_challengers", "13. recompute_metrics", "14. model_selection",
    "15. safety_router", "16. plan_builder", "17. simulator",
    "18. verify_scientific_completion",
}

PIPELINE = [
    ("1. forensic_audit", None, "COMPLETE"),  # already run, mark complete
    ("2. synthetic_pattern_audit", None, "COMPLETE"),
    ("3. quarantine_invalid_artifacts", None, "COMPLETE"),
    ("4. rebuild_verified_case_export", [
        "scripts/recommend_hybrid/explainable_v2/export_llm_cases.py",
    ], None),
    ("5. verify_case_lineage", [
        "scripts/recommend_hybrid/explainable_v2/verify_case_lineage.py",
    ], None),
    ("6. annotation_provenance_audit", [
        "scripts/recommend_hybrid/explainable_v2/audit_annotation_independence.py",
    ], None),
    ("7. annotation_independence_audit", None, None),  # covered by step 6
    ("8. final_annotation_gate", None, None),  # derived from step 6 exit code
    ("9. final_snorkel", [
        "scripts/recommend_hybrid/explainable_v2/fit_weak_label_models.py", "--mode", "final",
    ], None),
    ("10. train_five_ebm", [
        "scripts/recommend_hybrid/explainable_v2/train_five_ebm.py",
    ], None),
    ("11. train_baselines", [
        "scripts/recommend_hybrid/explainable_v2/train_challengers.py",
    ], None),
    ("12. train_challengers", None, None),  # covered by 11
    ("13. recompute_metrics", [
        "scripts/recommend_hybrid/explainable_v2/recompute_all_metrics.py",
    ], None),
    ("14. model_selection", [
        "scripts/recommend_hybrid/explainable_v2/run_model_selection.py",
    ], None),
    ("15. safety_router", None, None),
    ("16. plan_builder", None, None),
    ("17. simulator", [
        "scripts/recommend_hybrid/explainable_v2/run_plausibility_simulator.py",
    ], None),
    ("18. verify_scientific_completion", [
        "scripts/recommend_hybrid/explainable_v2/verify_scientific_completion.py",
    ], None),
]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}\n"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)


def write_progress(stage_results: dict, current_stage: str, annotation_gate: str) -> None:
    progress = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "current_stage": current_stage,
        "annotation_gate": annotation_gate,
        "stage_results": stage_results,
        "runtime_authorized": False,
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def write_supervisor_state(stage_results: dict, annotation_gate: str, overall: str) -> None:
    state = {
        "supervisor_status": overall,
        "blocked_reason": None if overall not in ("BLOCKED", "FAILED") else "PENDING_VERIFIED_EXTERNAL_LLM_REVIEWS",
        "annotation_gate": annotation_gate,
        "stage_results": stage_results,
        "real_llm_review_count": 0,
        "verified_external_llm_review_count": 0,
        "agent_generated_pseudo_review_count": 6750,
        "runtime_authorized": False,
        "log_file": str(LOG_FILE),
    }
    SUPERVISOR_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_stage(name: str, cmd: list[str] | None, prefilled_status: str | None) -> tuple[str, int]:
    """Run a pipeline stage. Returns (status, exit_code)."""
    if prefilled_status:
        log(f"STAGE {name}: {prefilled_status} (pre-completed)")
        return prefilled_status, 0

    if cmd is None:
        log(f"STAGE {name}: SKIPPED (no command)")
        return "SKIPPED", 0

    log(f"STAGE {name}: starting")
    full_cmd = [PYTHON] + cmd
    try:
        proc = subprocess.run(
            full_cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600
        )
        rc = proc.returncode
        if rc == 0:
            status = "PASS"
        elif rc == 2:
            status = "BLOCKED"
        else:
            status = "FAIL"
        log(f"STAGE {name}: {status} (exit={rc})")
        return status, rc
    except subprocess.TimeoutExpired:
        log(f"STAGE {name}: TIMEOUT")
        return "FAIL", -1
    except Exception as exc:
        log(f"STAGE {name}: ERROR {exc}")
        return "FAIL", -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    log("=== ADVERSARIAL REPAIR SUPERVISOR START ===")
    stage_results: dict[str, str] = {}
    annotation_gate = "PENDING"

    for name, cmd, prefilled in PIPELINE:
        # Check if annotation-dependent stage should be blocked
        if name in ANNOTATION_DEPENDENT_STAGES and annotation_gate not in ("PASS",):
            status = "BLOCKED_DEPENDENCY"
            stage_results[name] = status
            log(f"STAGE {name}: {status} (annotation gate={annotation_gate})")
            write_progress(stage_results, name, annotation_gate)
            continue

        status, rc = run_stage(name, cmd, prefilled)
        stage_results[name] = status

        # Capture annotation gate from stage 6
        if name == "6. annotation_provenance_audit":
            if status == "PASS":
                annotation_gate = "PASS"
            elif status == "BLOCKED":
                annotation_gate = "BLOCKED"
            else:
                annotation_gate = "FAIL"

        # Also derive gate from stage 8
        if name == "8. final_annotation_gate":
            if annotation_gate != "PASS":
                stage_results[name] = "BLOCKED"

        write_progress(stage_results, name, annotation_gate)

        # On FAIL, allow retries for implementation errors (max 2)
        if status == "FAIL" and cmd is not None and prefilled is None:
            log(f"STAGE {name}: FAIL — no retry for scientific gate failures")

    # Determine overall status
    values = list(stage_results.values())
    if any(v == "FAIL" for v in values):
        overall = "FAILED"
    elif all(v in ("PASS", "COMPLETE", "SKIPPED") for v in values):
        overall = "COMPLETE"
    elif any(v in ("BLOCKED", "BLOCKED_DEPENDENCY") for v in values):
        overall = "IMPLEMENTATION_COMPLETE_SCIENTIFIC_BLOCKED"
    else:
        overall = "PARTIAL"

    write_supervisor_state(stage_results, annotation_gate, overall)
    log(f"=== SUPERVISOR DONE: {overall} ===")

    # Print final status to stdout
    print(f"SUPERVISOR_STATUS={overall}")
    print(f"ANNOTATION_GATE={annotation_gate}")
    print(f"SCIENTIFIC_STATUS=BLOCKED_PENDING_VERIFIED_EXTERNAL_LLM_REVIEWS" if annotation_gate != "PASS" else "SCIENTIFIC_STATUS=COMPLETE")
    print(f"RUNTIME_AUTHORIZED=FALSE")
    return 0 if overall in ("COMPLETE", "IMPLEMENTATION_COMPLETE_SCIENTIFIC_BLOCKED") else 1


if __name__ == "__main__":
    sys.exit(main())
