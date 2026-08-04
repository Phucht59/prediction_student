"""Run authority-bound retrained negative controls in resumable parallel batches.

The dispatcher preserves the frozen 200-replicate protocol and selected model
capacity. It changes only execution scheduling: each ranker fit uses one CPU
thread, independent batches run in separate processes, and round-robin task
ordering exposes evidence across all controls instead of finishing NC1 first.
Every batch and progress snapshot is written atomically and remains resume-safe.
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
        # Thread count changes execution only. Tree count, depth, features,
        # folds and labels remain identical to the selected model authority.
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
    schedule: str = "round_robin",
) -> list[tuple[str, int, int]]:
    """Return missing batches in deterministic order.

    ``round_robin`` schedules the same replicate window for every requested
    control before advancing to the next window. This prevents a time-limited
    run from producing NC1-only evidence. ``control_major`` retains the old
    behavior for targeted completion of one control.
    """
    starts = list(range(0, replicates, batch_size))
    tasks: list[tuple[str, int, int]] = []
    if schedule == "round_robin":
        candidates = (
            (control, start, min(start + batch_size, replicates))
            for start in starts
            for control in requested_controls
        )
    elif schedule == "control_major":
        candidates = (
            (control, start, min(start + batch_size, replicates))
            for control in requested_controls
            for start in starts
        )
    else:
        raise ValueError(f"Unknown control schedule: {schedule}")

    for control, start, stop in candidates:
        if not controls.batch_path(control, start, stop).exists():
            tasks.append((control, start, stop))
    return tasks


def _write_dispatch_snapshot(
    authority: dict[str, Any],
    args: argparse.Namespace,
    completed_batches: int,
    total_submitted: int,
    failures: list[dict[str, Any]],
) -> pd.DataFrame:
    """Refresh scientific summary and durable dispatcher progress."""
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
            "schedule": args.schedule,
            "max_batches_this_run": args.max_batches,
            "completed_batches_this_run": completed_batches,
            "submitted_batches_this_run": total_submitted,
            "failed_batches_this_run": failures,
        }
    )
    atomic_json(exact.MARKER, marker)
    atomic_json(
        controls.CONTROL_OUT / "DISPATCH_PROGRESS.json",
        {
            **authority,
            "registered_replicates": args.replicates,
            "batch_size": args.batch_size,
            "schedule": args.schedule,
            "completed_batches_this_run": completed_batches,
            "submitted_batches_this_run": total_submitted,
            "failed_batches_this_run": failures,
            "controls": summary.to_dict(orient="records"),
        },
    )
    return summary


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
    parser.add_argument(
        "--schedule",
        choices=["round_robin", "control_major"],
        default="round_robin",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Run at most this many pending batches; 0 means all pending batches.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Refresh SUMMARY/marker after this many newly completed batches.",
    )
    args = parser.parse_args()
    if args.replicates != 200:
        raise ValueError("The frozen V2.1 protocol requires exactly 200 replicates")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be between 1 and 4")
    if args.max_batches < 0:
        raise ValueError("max-batches must be non-negative")
    if args.checkpoint_every <= 0:
        raise ValueError("checkpoint-every must be positive")

    authority = current_model_authority()
    prepare_namespace(
        exact.CONTROL_OUT,
        exact.MARKER,
        "negative_controls_stale_model_archive",
        authority,
    )
    controls.CONTROL_OUT.joinpath("batches").mkdir(parents=True, exist_ok=True)
    requested = controls.CONTROLS if args.control == "all" else [args.control]
    tasks = _pending_tasks(
        requested,
        args.replicates,
        args.batch_size,
        schedule=args.schedule,
    )
    if args.max_batches:
        tasks = tasks[: args.max_batches]

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(str(controls.DATA / "candidate_rows.parquet"),),
        ) as executor:
            future_map = {executor.submit(_run_batch_task, task): task for task in tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:  # other batches remain resumable
                    failure = {
                        "control": task[0],
                        "start": task[1],
                        "stop": task[2],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    print(
                        f"{task[0]} {task[1]:04d}:{task[2]:04d} ERROR: {exc}",
                        flush=True,
                    )
                    continue

                completed.append(result)
                print(
                    f"{result['control']} {result['start']:04d}:{result['stop']:04d} "
                    f"{result['status']}",
                    flush=True,
                )
                if len(completed) % args.checkpoint_every == 0:
                    _write_dispatch_snapshot(
                        authority,
                        args,
                        len(completed),
                        len(tasks),
                        failures,
                    )

    summary = _write_dispatch_snapshot(
        authority,
        args,
        len(completed),
        len(tasks),
        failures,
    )
    print(summary.to_string(index=False))
    if failures:
        raise RuntimeError(
            f"{len(failures)} control batches failed; completed batches remain resumable"
        )


if __name__ == "__main__":
    main()
