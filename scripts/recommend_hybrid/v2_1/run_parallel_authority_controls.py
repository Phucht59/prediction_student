"""Run authority-bound retrained negative controls in resumable parallel batches.

This dispatcher does not reduce the registered 200 replicates or the selected
model capacity.  It changes only execution parallelism: each XGBoost/LightGBM
fit uses one CPU thread while independent replicate batches run in separate
processes.  Every batch is written atomically and remains resume-safe.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import corrected_negative_controls as controls
import run_exact_negative_controls as exact
from postsearch_authority import atomic_json, current_model_authority, prepare_namespace

_RAW: pd.DataFrame | None = None


def _worker_init(data_path: str) -> None:
    """Load the cohort once per worker and install the exact-capacity evaluator."""
    global _RAW
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    _RAW = pd.read_parquet(data_path)

    def exact_single_thread(
        raw_train: pd.DataFrame,
        raw_test: pd.DataFrame,
        family: str,
        parameters: dict[str, Any],
        control: str,
        seed: int,
    ) -> float:
        execution_parameters = dict(parameters)
        # Thread count changes execution only; tree count/depth/features remain
        # exactly the selected full-grid model authority.
        execution_parameters["n_jobs"] = 1
        return exact.exact_fit_and_evaluate_control(
            raw_train,
            raw_test,
            family,
            execution_parameters,
            control,
            seed,
        )

    controls.fit_and_evaluate_control = exact_single_thread


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _run_batch_task(task: tuple[str, int, int]) -> dict[str, Any]:
    if _RAW is None:
        raise RuntimeError("Control worker cohort was not initialized")
    control, start, stop = task
    path = controls.batch_path(control, start, stop)
    if path.exists():
        return {
            "control": control,
            "start": start,
            "stop": stop,
            "status": "RESUMED_EXISTING",
        }
    frame = controls.run_batch(_RAW, control, start, stop)
    _atomic_csv(path, frame)
    return {
        "control": control,
        "start": start,
        "stop": stop,
        "status": "COMPLETE",
    }


def _pending_tasks(
    requested_controls: list[str],
    replicates: int,
    batch_size: int,
) -> list[tuple[str, int, int]]:
    tasks: list[tuple[str, int, int]] = []
    for control in requested_controls:
        for start in range(0, replicates, batch_size):
            stop = min(start + batch_size, replicates)
            if not controls.batch_path(control, start, stop).exists():
                tasks.append((control, start, stop))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--control",
        choices=controls.CONTROLS + ["all"],
        default="all",
    )
    args = parser.parse_args()
    if args.replicates != 200:
        raise ValueError("The frozen V2.1 protocol requires exactly 200 replicates")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be between 1 and 4")

    authority = current_model_authority()
    prepare_namespace(
        exact.CONTROL_OUT,
        exact.MARKER,
        "negative_controls_stale_model_archive",
        authority,
    )
    controls.CONTROL_OUT.joinpath("batches").mkdir(parents=True, exist_ok=True)
    requested = controls.CONTROLS if args.control == "all" else [args.control]
    tasks = _pending_tasks(requested, args.replicates, args.batch_size)

    completed: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(controls.DATA / "candidate_rows.parquet"),),
        ) as executor:
            future_map = {executor.submit(_run_batch_task, task): task for task in tasks}
            for future in as_completed(future_map):
                result = future.result()
                completed.append(result)
                print(
                    f"{result['control']} {result['start']:04d}:{result['stop']:04d} "
                    f"{result['status']}",
                    flush=True,
                )

    summary = controls.summarize(args.replicates)
    controls.update_progress(summary)
    exact.finalize_marker()
    marker = json.loads(exact.MARKER.read_text(encoding="utf-8"))
    marker.update(
        {
            **authority,
            "execution": "PROCESS_PARALLEL_BATCH_RESUMABLE",
            "workers": args.workers,
            "model_threads_per_worker": 1,
            "batch_size": args.batch_size,
            "newly_completed_batches": len(completed),
        }
    )
    atomic_json(exact.MARKER, marker)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
