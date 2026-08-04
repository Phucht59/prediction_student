"""Archive and reopen only failed full-grid trial checkpoints.

Successful inner-fold checkpoints are preserved. Failed inner-fold JSON and the
trial summary are copied into an append-only attempt archive before deletion, so
rerunning the resumable search never erases diagnostic evidence.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from run_resumable_full_registered_search import WORK_SELECTION


def next_attempt_directory(trial_directory: Path) -> Path:
    archive = trial_directory / "failed_attempts"
    archive.mkdir(parents=True, exist_ok=True)
    number = 1
    while (archive / f"attempt_{number:03d}").exists():
        number += 1
    destination = archive / f"attempt_{number:03d}"
    destination.mkdir()
    return destination


def reopen_trial(trial_directory: Path) -> bool:
    summary = trial_directory / "trial.json"
    if not summary.exists():
        return False
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if payload.get("status") == "COMPLETE":
        return False

    destination = next_attempt_directory(trial_directory)
    shutil.copy2(summary, destination / "trial.json")
    summary.unlink()
    for inner_path in sorted(trial_directory.glob("inner_*.json")):
        inner = json.loads(inner_path.read_text(encoding="utf-8"))
        if inner.get("status") == "COMPLETE":
            continue
        shutil.copy2(inner_path, destination / inner_path.name)
        inner_path.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", choices=["all", "0", "1", "2"], default="all")
    args = parser.parse_args()
    folds = [0, 1, 2] if args.outer_fold == "all" else [int(args.outer_fold)]
    reopened = []
    for fold in folds:
        root = WORK_SELECTION / f"fold_{fold}" / "trials"
        if not root.exists():
            continue
        for trial_directory in sorted(root.glob("trial_*")):
            if reopen_trial(trial_directory):
                reopened.append(str(trial_directory))
    print(json.dumps({"reopened_trials": reopened, "count": len(reopened)}, indent=2))


if __name__ == "__main__":
    main()
