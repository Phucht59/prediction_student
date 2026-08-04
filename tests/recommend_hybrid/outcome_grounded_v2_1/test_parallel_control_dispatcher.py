from __future__ import annotations

import importlib
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "recommend_hybrid" / "v2_1"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_parallel_control_pending_tasks_are_batch_resumable(tmp_path, monkeypatch):
    dispatcher = importlib.import_module("run_parallel_authority_controls")

    def fake_batch_path(control: str, start: int, stop: int) -> Path:
        return tmp_path / f"{control}__{start:04d}_{stop:04d}.csv"

    monkeypatch.setattr(dispatcher.controls, "batch_path", fake_batch_path)
    tasks = dispatcher._pending_tasks(["NC1_LABEL_SHUFFLE_RETRAIN"], 10, 5)
    assert tasks == [
        ("NC1_LABEL_SHUFFLE_RETRAIN", 0, 5),
        ("NC1_LABEL_SHUFFLE_RETRAIN", 5, 10),
    ]

    fake_batch_path("NC1_LABEL_SHUFFLE_RETRAIN", 0, 5).write_text(
        "control,replicate,ndcg_at_3\n",
        encoding="utf-8",
    )
    resumed = dispatcher._pending_tasks(["NC1_LABEL_SHUFFLE_RETRAIN"], 10, 5)
    assert resumed == [("NC1_LABEL_SHUFFLE_RETRAIN", 5, 10)]


def test_parallel_control_round_robin_exposes_all_controls_early(tmp_path, monkeypatch):
    dispatcher = importlib.import_module("run_parallel_authority_controls")

    def fake_batch_path(control: str, start: int, stop: int) -> Path:
        return tmp_path / f"{control}__{start:04d}_{stop:04d}.csv"

    monkeypatch.setattr(dispatcher.controls, "batch_path", fake_batch_path)
    controls = ["NC1_LABEL_SHUFFLE_RETRAIN", "NC2A_TRAIN_STATE_SHUFFLE"]
    tasks = dispatcher._pending_tasks(controls, 10, 5, schedule="round_robin")
    assert tasks == [
        ("NC1_LABEL_SHUFFLE_RETRAIN", 0, 5),
        ("NC2A_TRAIN_STATE_SHUFFLE", 0, 5),
        ("NC1_LABEL_SHUFFLE_RETRAIN", 5, 10),
        ("NC2A_TRAIN_STATE_SHUFFLE", 5, 10),
    ]


def test_parallel_control_major_schedule_remains_available(tmp_path, monkeypatch):
    dispatcher = importlib.import_module("run_parallel_authority_controls")

    def fake_batch_path(control: str, start: int, stop: int) -> Path:
        return tmp_path / f"{control}__{start:04d}_{stop:04d}.csv"

    monkeypatch.setattr(dispatcher.controls, "batch_path", fake_batch_path)
    controls = ["NC1_LABEL_SHUFFLE_RETRAIN", "NC2A_TRAIN_STATE_SHUFFLE"]
    tasks = dispatcher._pending_tasks(controls, 10, 5, schedule="control_major")
    assert tasks == [
        ("NC1_LABEL_SHUFFLE_RETRAIN", 0, 5),
        ("NC1_LABEL_SHUFFLE_RETRAIN", 5, 10),
        ("NC2A_TRAIN_STATE_SHUFFLE", 0, 5),
        ("NC2A_TRAIN_STATE_SHUFFLE", 5, 10),
    ]


def test_parallel_control_dispatcher_preserves_registered_protocol():
    dispatcher = importlib.import_module("run_parallel_authority_controls")
    source = Path(dispatcher.__file__).read_text(encoding="utf-8")
    assert 'args.replicates != 200' in source
    assert 'execution_parameters["n_jobs"] = 1' in source
    assert 'execution_parameters["n_estimators"]' not in source
    assert 'model_threads_per_worker' in source
    assert 'default="round_robin"' in source
    assert '"--max-batches"' in source
    assert '"--checkpoint-every"' in source
