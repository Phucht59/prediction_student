"""Hardware capture, CUDA fail-fast, and performance microbenchmarks."""
from __future__ import annotations

import os
import platform
import shutil
import time
from typing import Any

import numpy as np

from .io_utils import env_names_present, git_branch, git_commit, git_dirty, load_dotenv, utc_now, write_json
from .paths import MANIFEST_DIR, ensure_dirs


def require_cuda() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_HYBRID_TRAINING")
    props = torch.cuda.get_device_properties(0)
    info = {
        "cuda_available": True,
        "device": "cuda:0",
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "cudnn": int(torch.backends.cudnn.version() or 0),
        "vram_gb": round(props.total_memory / 1024**3, 3),
        "capability": list(torch.cuda.get_device_capability(0)),
        "fail_fast_if_cpu": True,
        "silent_cpu_fallback": False,
        "tf32_supported": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False) and props.major >= 8),
        "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()),
    }
    return info


def cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None
    if psutil is not None:
        vm = psutil.virtual_memory()
        info["ram_gb"] = round(vm.total / 1024**3, 3)
        info["ram_available_gb"] = round(vm.available / 1024**3, 3)
    else:
        info["ram_gb"] = None
    usage = shutil.disk_usage(os.path.abspath("."))
    info["disk_free_gb"] = round(usage.free / 1024**3, 3)
    info["disk_total_gb"] = round(usage.total / 1024**3, 3)
    return info


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "sklearn", "optuna", "xgboost", "catboost", "numpy", "pandas", "psycopg2", "imblearn"):
        try:
            mod = __import__(name if name != "sklearn" else "sklearn")
            versions[name] = getattr(mod, "__version__", "present")
        except Exception as exc:
            versions[name] = f"MISSING:{type(exc).__name__}"
    return versions


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bench_matmul(dtype, device, size: int = 2048, repeats: int = 8) -> dict[str, float]:
    import torch

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(3):
        torch.mm(a, b)
    _sync()
    started = time.perf_counter()
    for _ in range(repeats):
        torch.mm(a, b)
    _sync()
    elapsed = (time.perf_counter() - started) / repeats
    flops = 2 * size * size * size
    return {"seconds": elapsed, "tflops": flops / elapsed / 1e12}


def microbenchmark(full: bool = False) -> dict[str, Any]:
    import torch

    cuda = require_cuda()
    device = torch.device("cuda")
    result: dict[str, Any] = {"cuda": cuda}
    result["fp32"] = _bench_matmul(torch.float32, device)
    with torch.autocast("cuda", dtype=torch.float16):
        result["amp_fp16"] = _bench_matmul(torch.float16, device)
    if cuda["bf16_supported"]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result["amp_bf16"] = _bench_matmul(torch.bfloat16, device)
    chosen = "fp16"
    result["selected_amp"] = chosen
    result["reason"] = "Turing RTX 2060: FP16 AMP with GradScaler; TF32 is Ampere+ so left off"
    if full:
        from .model import SuperiorityConfig, SuperiorityHybrid, count_parameters

        cfg = SuperiorityConfig(candidate="C3-G", static_dim=64, temporal_dim=13, aggregate_dim=13)
        model = SuperiorityHybrid(cfg).cuda()
        result["c3g_params"] = count_parameters(model)
        batch = 64
        t = 40
        static = torch.randn(batch, 64, device=device)
        temporal = torch.randn(batch, t, 13, device=device)
        mask = torch.ones(batch, t, dtype=torch.bool, device=device)
        lengths = torch.full((batch,), t, device=device)
        aggregate = torch.randn(batch, 13, device=device)
        agg_avail = torch.ones(batch, dtype=torch.bool, device=device)
        progress = torch.rand(batch, device=device)
        scaler = torch.amp.GradScaler("cuda")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # warmup
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(static, temporal, mask, lengths, aggregate, agg_avail, progress)
                loss = logits.float().pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        _sync()
        n = 10
        started = time.perf_counter()
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(static, temporal, mask, lengths, aggregate, agg_avail, progress)
                loss = logits.float().pow(2).mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        _sync()
        result["train_step_seconds"] = (time.perf_counter() - started) / n
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        # compile probe
        compile_ok = False
        compile_speedup = None
        try:
            compiled = torch.compile(model)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                compiled(static, temporal, mask, lengths, aggregate, agg_avail, progress)
            _sync()
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(8):
                    compiled(static, temporal, mask, lengths, aggregate, agg_avail, progress)
            _sync()
            compiled_t = (time.perf_counter() - started) / 8
            started = time.perf_counter()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(8):
                    model(static, temporal, mask, lengths, aggregate, agg_avail, progress)
            _sync()
            eager_t = (time.perf_counter() - started) / 8
            compile_speedup = eager_t / compiled_t if compiled_t > 0 else 0
            compile_ok = compile_speedup >= 1.05
        except Exception as exc:
            result["compile_error"] = type(exc).__name__
        result["torch_compile"] = {"ok": compile_ok, "speedup": compile_speedup, "keep": bool(compile_ok)}
    return result


def capture_hardware(*, full_bench: bool = False) -> dict[str, Any]:
    ensure_dirs()
    load_dotenv()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    payload = {
        "timestamp": utc_now(),
        "git_commit": git_commit(),
        "git_branch": git_branch(),
        "git_dirty": git_dirty(),
        "cpu": cpu_info(),
        "packages": package_versions(),
        "env_names_present": env_names_present(),
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
        "benchmark": microbenchmark(full=full_bench),
    }
    write_json(MANIFEST_DIR / "hardware_manifest.json", payload)
    return payload


def probe_batch_size(model, make_batch, start: int = 256, min_size: int = 16) -> int:
    import torch

    require_cuda()
    size = start
    while size >= min_size:
        try:
            torch.cuda.empty_cache()
            batch = make_batch(size)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**batch)
                loss = logits.float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            del batch, logits, loss
            torch.cuda.empty_cache()
            return size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            size //= 2
    raise RuntimeError("BATCH_PROBE_OOM")
