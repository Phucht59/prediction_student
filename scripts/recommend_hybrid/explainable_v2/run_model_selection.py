"""Single, resume-safe supervisor for the protocol-locked V2 pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "artifacts/recommend_hybrid/explainable_v2/run_state"
LOGS = STATE / "logs"
SUPERVISOR = STATE / "supervisor.json"
PROGRESS = STATE / "progress.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    existing = None
    if SUPERVISOR.exists():
        try:
            existing = json.loads(SUPERVISOR.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
    if existing and existing.get("status") == "running" and alive(int(existing.get("pid", -1))):
        return 0

    manifest = {
        "schema_version": "recommend_hybrid_explainable_v2_supervisor",
        "pid": os.getpid(), "started_at": now(), "updated_at": now(),
        "status": "running", "resume_safe": True, "runtime_authorized": False,
        "duplicate_process": False,
    }
    write(SUPERVISOR, manifest)
    stages = [
        ("static_validation", [sys.executable, str(ROOT / "scripts/recommend_hybrid/validate_explainable_v2_static.py")]),
        ("unit_tests", [sys.executable, "-m", "pytest", "-q", "tests/recommend_hybrid/explainable_v2"]),
        ("hybrid_authority_audit", [sys.executable, str(ROOT / "scripts/recommend_hybrid/explainable_v2/verify_hybrid_oof_authority.py")]),
        ("feature_table", ROOT / "scripts/recommend_hybrid/explainable_v2/build_feature_table.py"),
        ("risk_policy", ROOT / "scripts/recommend_hybrid/explainable_v2/select_risk_policy.py"),
        ("action_candidates", ROOT / "scripts/recommend_hybrid/explainable_v2/build_action_candidates.py"),
        ("weak_labels", ROOT / "scripts/recommend_hybrid/explainable_v2/fit_weak_label_models.py"),
        ("model_selection", None),
    ]
    progress = {"schema_version": "recommend_hybrid_explainable_v2_progress", "runtime_authorized": False, "stages": {}}
    for name, command in stages:
        entry = {"status": "pending", "started_at": now()}
        progress["stages"][name] = entry
        write(PROGRESS, progress)
        log = LOGS / f"{name}.log"
        if isinstance(command, Path):
            command = [sys.executable, str(command)] if command.exists() else None
        if command is None:
            entry.update(status="blocked", reason="required_protocol_stage_not_implemented_or_missing")
            log.write_text("BLOCKED: required stage is not available; no substitute or fabricated data was used.\n", encoding="utf-8")
        else:
            with log.open("a", encoding="utf-8") as handle:
                result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
            entry.update(status="completed" if result.returncode == 0 else "failed", returncode=result.returncode)
        entry["finished_at"] = now()
        write(PROGRESS, progress)
        if entry["status"] == "failed":
            manifest.update(status="failed", error_stage=name, updated_at=now())
            write(SUPERVISOR, manifest)
            return 1
    manifest.update(status="blocked_pending_required_stages", updated_at=now(), finished_at=now())
    write(SUPERVISOR, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
