"""Resume-safe supervisor for the isolated 50-job UCI NONE reproduction."""
from __future__ import annotations

import traceback
from datetime import datetime, timezone
from pathlib import Path

from adapter_common import ROOT, RUNTIME, SEEDS, assert_equivalence, atomic_json
from run_none_reproduction import run


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid(run_id: str) -> bool:
    folder = ROOT / "runs" / run_id
    manifest = folder / "run_manifest.json"
    return manifest.is_file() and all((folder / name).is_file() for name in ("checkpoint.pt", "predictions.npz", "metrics.json")) and '"COMPLETE"' in manifest.read_text(encoding="utf-8")


def status(**updates) -> None:
    path = RUNTIME / "NONE_STATUS.json"
    base = {"status": "RUNNING", "total_expected": 50, "completed": 0, "failed": 0, "started_at": now(), "last_update_at": now()}
    if path.is_file():
        import json
        base.update(json.loads(path.read_text(encoding="utf-8")))
    base.update(updates, last_update_at=now())
    atomic_json(path, base)


def main() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    assert_equivalence("student-mat"); assert_equivalence("student-por")
    (RUNTIME / "NONE_TRAINING_FAILED").unlink(missing_ok=True)
    (RUNTIME / "NONE_TRAINING_COMPLETE").unlink(missing_ok=True)
    status(status="RUNNING", error=None, current_run=None)
    (RUNTIME / "NONE_TRAINING_RUNNING").write_text("RUNNING\n", encoding="utf-8")
    jobs = [(dataset, fold, seed) for dataset in ("student-mat", "student-por") for fold in range(5) for seed in SEEDS]
    completed = 0
    try:
        for dataset, fold, seed in jobs:
            run_id = f"{dataset.replace('-', '_')}__none__fold{fold}__seed{seed}"
            if valid(run_id):
                completed += 1
                continue
            status(completed=completed, current_run=run_id)
            run(dataset, fold, seed)
            if not valid(run_id):
                raise RuntimeError(f"Invalid completed job: {run_id}")
            completed += 1
            status(completed=completed, last_completed_run=run_id, current_run=None)
        status(status="COMPLETE", completed=completed, current_run=None)
        (RUNTIME / "NONE_TRAINING_COMPLETE").write_text("COMPLETE\n", encoding="utf-8")
    except Exception:
        status(status="FAILED", completed=completed, error=traceback.format_exc())
        (RUNTIME / "NONE_TRAINING_FAILED").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        (RUNTIME / "NONE_TRAINING_RUNNING").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
