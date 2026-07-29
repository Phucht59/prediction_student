from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "run_oulad_multistage_detached.ps1"
LAUNCHER = ROOT / "scripts" / "launch_oulad_multistage_detached.ps1"
RUNTIME_PATH = ROOT / "scripts" / "oulad_multistage_runtime.py"


def _runtime_module():
    spec = importlib.util.spec_from_file_location("oulad_runtime_test", RUNTIME_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_external_command_capture_is_nonempty_and_typed() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "$process.WaitForExit()" in text
    assert "$exitCode = [int]$process.ExitCode" in text
    assert "exit_code=$exitCode; detail=$detail" in text
    assert "last_non_empty_stderr_line" in text
    assert "last_non_empty_stdout_line" in text


def test_stale_lock_recovery_requires_dead_wrapper() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "$wrapperAlive" in text
    assert "An active detached OULAD wrapper already exists" in text
    assert "REMOVED_STALE_LOCK" in text
    assert "old_status_sha256" in text


def test_runtime_files_are_not_used_as_dirty_source_guard() -> None:
    runtime = _runtime_module()
    assert runtime._tracked_source_changes.__doc__
    text = RUNTIME_PATH.read_text(encoding="utf-8")
    assert "--untracked-files=no" in text
    assert "logs/" not in runtime._tracked_source_changes.__doc__


def test_wrong_branch_is_explicit_preflight_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime_module()
    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "main"
        if args[:2] == ("status", "--porcelain=v1"):
            return ""
        if args == ("rev-parse", "HEAD"):
            return runtime.DETACHED_FIX_BASE
        raise AssertionError(args)

    monkeypatch.setattr(runtime, "_git", fake_git)
    monkeypatch.setattr(runtime.subprocess, "run", lambda *args, **kwargs: type("R", (), {"returncode": 0})())
    monkeypatch.setattr(runtime, "_json", lambda *args, **kwargs: None)
    result = runtime.preflight()
    assert result["status"] == "FAIL"
    assert "wrong branch: main" in result["errors"]


def test_missing_interpreter_has_explicit_wrapper_error() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    assert "executable not found: $Executable" in text
    assert "exit_code=127" in text
    assert ".venv-oulad-v2\\Scripts\\python.exe" in text


def test_cuda_false_maps_to_blocked_gpu_exit_20(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime_module()
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(runtime, "_json", lambda *args, **kwargs: None)
    result = runtime.gpu()
    assert result["status"] == "BLOCKED_GPU"
    text = RUNTIME_PATH.read_text(encoding="utf-8")
    assert 'result["status"] == "BLOCKED_GPU"' in text
    assert "return 20" in text


def test_preflight_only_exits_before_training_and_amendment() -> None:
    text = WRAPPER.read_text(encoding="utf-8")
    gate = text.index("if ($PreflightOnly)")
    amendment = text.index('"SVM_AMENDMENT"')
    training = text.index('"FULL_TRAIN_RESUME"')
    assert gate < amendment < training
    assert 'Write-Status -State "PREFLIGHT_PASS"' in text


def test_existing_sixty_checkpoints_and_archive_are_preserved() -> None:
    checkpoint_root = (
        ROOT / "artifacts" / "final" / "unified_stage_aware_oulad" / "checkpoints"
    )
    expected = {
        "logistic_regression_oulad": 15,
        "decision_tree_oulad": 15,
        "random_forest_oulad": 15,
        "hist_gradient_boosting_oulad": 15,
    }
    actual = {
        name: len(list((checkpoint_root / name).rglob("*.joblib")))
        for name in expected
    }
    assert actual == expected
    assert not list((checkpoint_root / "svm_oulad").rglob("*.joblib"))
    archive = ROOT / "artifacts" / "history" / "partial_svm_probability_true_20260729"
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["checkpoint_count"] == 8
    assert len(list(archive.rglob("*.joblib"))) == 8
