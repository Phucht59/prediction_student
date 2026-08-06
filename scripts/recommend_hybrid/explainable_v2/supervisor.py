"""Single-process, resume-safe supervisor for Scientific Recommendation V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_command_step(cmd_args: list[str]) -> tuple[int, str]:
    python_exe = sys.executable
    full_cmd = [python_exe] + cmd_args
    proc = subprocess.run(
        full_cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8"
    )
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def compute_file_hash(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_pipeline(force_rerun: bool = False) -> dict:
    run_state_dir = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state"
    logs_dir = run_state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "supervisor.log"
    progress_file = run_state_dir / "progress.json"
    supervisor_json_file = run_state_dir / "supervisor.json"
    state_file = run_state_dir / "supervisor_state.json"

    state = {}
    if state_file.exists() and not force_rerun:
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    log_handle = log_file.open("a", encoding="utf-8")

    def log(msg: str):
        log_handle.write(msg + "\n")
        log_handle.flush()

    log("=== STARTING RECOMMENDATION PIPELINE V2 SUPERVISOR ===")

    stages = [
        ("1. repository_audit", ["scripts/recommend_hybrid/explainable_v2/audit_legacy_pipeline.py"]),
        ("2. build_action_candidates", ["scripts/recommend_hybrid/explainable_v2/build_feature_table.py"]),
        ("3. case_export_v2", ["scripts/recommend_hybrid/explainable_v2/export_llm_cases.py"]),
        ("4. annotation_import_audit", ["scripts/recommend_hybrid/explainable_v2/import_llm_annotations.py"]),
        ("5. preliminary_weak_labeling", ["scripts/recommend_hybrid/explainable_v2/fit_weak_label_models.py", "--mode", "preliminary"]),
        ("6. train_five_ebm", ["scripts/recommend_hybrid/explainable_v2/train_five_ebm.py"]),
        ("7. train_challengers", ["scripts/recommend_hybrid/explainable_v2/train_challengers.py"]),
        ("8. run_model_selection", ["scripts/recommend_hybrid/explainable_v2/run_model_selection.py"]),
        ("9. run_plausibility_simulator", ["scripts/recommend_hybrid/explainable_v2/run_plausibility_simulator.py"]),
    ]

    stage_results = {}
    blocked_reason = None
    overall_status = "COMPLETE"

    for stage_index, (stage_name, cmd) in enumerate(stages, start=1):
        if state.get(stage_name) == "PASS" and not force_rerun:
            log(f"Stage [{stage_name}] ... SKIPPED (Already PASS)")
            stage_results[stage_name] = "PASS"
            continue

        log(f"Stage [{stage_name}] ... RUNNING")

        # Update progress.json
        progress_payload = {
            "current_stage": stage_name,
            "completed_stages": [k for k, v in stage_results.items() if v == "PASS"],
            "total_stages": len(stages),
            "stage_index": stage_index,
            "status": "IN_PROGRESS",
        }
        progress_file.write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")

        ret, output = run_command_step(cmd)
        log_handle.write(output + "\n")
        log_handle.flush()

        if ret == 0:
            stage_results[stage_name] = "PASS"
            state[stage_name] = "PASS"
            log(f"Stage [{stage_name}] ... PASS")
        elif ret == 2 or "BLOCKED" in output:
            stage_results[stage_name] = "BLOCKED"
            state[stage_name] = "BLOCKED"
            overall_status = "BLOCKED"
            blocked_reason = "PENDING_REAL_LLM_ANNOTATION_RESPONSES"
            log(f"Stage [{stage_name}] ... BLOCKED ({blocked_reason})")
        else:
            stage_results[stage_name] = "FAILED"
            state[stage_name] = "FAILED"
            overall_status = "FAILED"
            log(f"Stage [{stage_name}] ... FAILED")
            break

    # Save final states
    state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    import_manifest_path = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/imports/import_manifest.json"
    )
    real_llm_count = 0
    real_human_count = 0
    if import_manifest_path.exists():
        im_data = json.loads(import_manifest_path.read_text(encoding="utf-8"))
        real_llm_count = im_data.get("real_llm_review_count", 0)
        real_human_count = im_data.get("real_human_review_count", 0)

    case_manifest_path = (
        ROOT / "artifacts/recommend_hybrid/explainable_v2/annotations/exports/case_manifest.json"
    )
    pa_count = 0
    pb_count = 0
    if case_manifest_path.exists():
        cm_data = json.loads(case_manifest_path.read_text(encoding="utf-8"))
        pa_count = cm_data.get("panel_a_count", 0)
        pb_count = cm_data.get("panel_b_count", 0)

    summary = {
        "supervisor_status": overall_status,
        "blocked_reason": blocked_reason,
        "stage_results": stage_results,
        "real_llm_review_count": real_llm_count,
        "real_human_review_count": real_human_count,
        "panel_a_cases": pa_count,
        "panel_b_cases": pb_count,
        "runtime_authorized": False,
        "log_file": str(log_file),
    }

    run_state_dir.mkdir(parents=True, exist_ok=True)
    supervisor_json_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    progress_payload = {
        "current_stage": "COMPLETE" if overall_status == "COMPLETE" else "FINISHED_WITH_STATUS",
        "completed_stages": [k for k, v in stage_results.items() if v == "PASS"],
        "total_stages": len(stages),
        "overall_status": overall_status,
    }
    progress_file.write_text(json.dumps(progress_payload, indent=2), encoding="utf-8")

    _write_final_implementation_status(summary)
    log(f"=== SUPERVISOR FINISHED STATUS: {overall_status} ===")
    log_handle.close()
    return summary


def _write_final_implementation_status(summary: dict) -> None:
    report_path = (
        ROOT / "reports/recommend_hybrid_v2/FINAL_IMPLEMENTATION_STATUS.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Final Implementation Status Report

## Pipeline Status Summary
- **Supervisor Status**: `{summary['supervisor_status']}`
- **Blocked Reason**: `{summary['blocked_reason'] or 'NONE'}`
- **Real Human Review Count**: `{summary['real_human_review_count']}`
- **Real LLM Review Count**: `{summary['real_llm_review_count']}`
- **Panel A Cases Exported**: `{summary['panel_a_cases']}`
- **Panel B Cases Exported**: `{summary['panel_b_cases']}`
- **Runtime Authorized**: `{summary['runtime_authorized']}`

## Stage Execution Matrix
| Stage | Status |
| --- | --- |
| 1. Repository Audit | `{summary['stage_results'].get('1. repository_audit', 'NOT_RUN')}` |
| 2. Build Action Candidates | `{summary['stage_results'].get('2. build_action_candidates', 'NOT_RUN')}` |
| 3. Case Export V2 | `{summary['stage_results'].get('3. case_export_v2', 'NOT_RUN')}` |
| 4. Annotation Import Audit | `{summary['stage_results'].get('4. annotation_import_audit', 'NOT_RUN')}` |
| 5. Preliminary Weak Labeling | `{summary['stage_results'].get('5. preliminary_weak_labeling', 'NOT_RUN')}` |
| 6. Train Five-EBM | `{summary['stage_results'].get('6. train_five_ebm', 'NOT_RUN')}` |
| 7. Train Challengers | `{summary['stage_results'].get('7. train_challengers', 'NOT_RUN')}` |
| 8. Run Model Selection | `{summary['stage_results'].get('8. run_model_selection', 'NOT_RUN')}` |
| 9. Run Plausibility Simulator | `{summary['stage_results'].get('9. run_plausibility_simulator', 'NOT_RUN')}` |

## Next Steps for User
To transition from `PRELIMINARY_WEAK_LABELS` to `FINAL_MODEL_SELECTION`:
1. Collect real LLM annotations using prompt batches in `artifacts/recommend_hybrid/explainable_v2/annotations/prompts/`.
2. Place response files in `artifacts/recommend_hybrid/explainable_v2/annotations/imports/raw/`.
3. Re-run the supervisor:
   ```bash
   python scripts/recommend_hybrid/explainable_v2/supervisor.py
   ```
"""
    report_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.force_rerun)
