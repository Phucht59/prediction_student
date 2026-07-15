from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def is_pass(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
    except (json.JSONDecodeError, OSError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume-aware Study B + Study C orchestrator")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--max-wall-clock-hours", type=float, default=6.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--study-b-run", default="study-b-student-por-20260715-v1")
    parser.add_argument("--study-c-run", default="study-c-oulad-20260715-v1")
    parser.add_argument("--execution-run", default="study-bc-extension-20260715-v1")
    args = parser.parse_args()
    started = time.monotonic()
    stop_new = max(0.0, args.max_wall_clock_hours - 0.75) * 3600
    python = sys.executable
    b = ROOT / "artifacts" / "study_b_student_por" / args.study_b_run
    c = ROOT / "artifacts" / "study_c_oulad" / args.study_c_run
    processed = ROOT / "data" / "processed" / "study_c_oulad"
    ledger: list[dict[str, object]] = []

    stages = [
        ("raw_audit", [python, "scripts/audit_extension_raw_data.py"], ROOT / "data/manifests/extension_raw_manifest.json"),
        ("oulad_audit", [python, "scripts/audit_oulad_release.py"], ROOT / "data/manifests/oulad_release_audit.json"),
        ("study_b", [python, "scripts/run_study_b_student_por.py", "--run-id", args.study_b_run], b / "validation_report.json"),
        ("oulad_materialize", [python, "scripts/materialize_oulad_forecasts.py", "--protocol", str(args.protocol), "--resume"], processed / "manifests/F3_LATE.json"),
        ("oulad_splits", [python, "scripts/materialize_oulad_splits.py", "--protocol", str(args.protocol)], processed / "manifests/split_complete.json"),
        ("study_c", [python, "scripts/run_study_c_oulad.py", "--protocol", str(args.protocol), "--run-id", args.study_c_run, "--max-wall-clock-hours", str(args.max_wall_clock_hours), "--resume"], c / "validation_report.json"),
        ("study_b_seed_stability", [python, "scripts/run_study_b_seed_stability.py", "--run-id", args.study_b_run], b / "seed_stability.csv"),
        ("study_c_seed_stability", [python, "scripts/run_study_c_seed_stability.py", "--run-id", args.study_c_run], c / "seed_stability.csv"),
        ("build_evidence", [python, "scripts/build_extension_evidence.py", "--study-b-run", args.study_b_run, "--study-c-run", args.study_c_run], c / "artifact_checksums.json"),
    ]
    for name, command, marker in stages:
        already_done = marker.is_file() and (name not in {"study_b", "study_c"} or is_pass(marker))
        if args.resume and already_done:
            ledger.append({"stage": name, "status": "SKIPPED_ALREADY_PASS", "marker": str(marker.relative_to(ROOT))})
            continue
        if time.monotonic() - started >= stop_new:
            ledger.append({"stage": name, "status": "PENDING_RESUME", "reason": "wall_clock_stop_new_jobs"})
            break
        if args.dry_run:
            ledger.append({"stage": name, "status": "DRY_RUN", "command": subprocess.list2cmdline(command)})
            continue
        completed = subprocess.run(command, cwd=ROOT, text=True)
        ledger.append({"stage": name, "status": "PASS" if completed.returncode == 0 else "FAIL", "return_code": completed.returncode})
        if completed.returncode != 0:
            break
    execution = ROOT / "reports" / "extension_execution" / args.execution_run
    execution.mkdir(parents=True, exist_ok=True)
    (execution / "job_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "DRY_RUN" if args.dry_run else ledger[-1]["status"], "ledger": ledger}, indent=2))
    return 0 if all(row["status"] not in {"FAIL"} for row in ledger) else 1


if __name__ == "__main__":
    raise SystemExit(main())
