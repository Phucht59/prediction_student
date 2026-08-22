"""Software GPU thermal cap at 80 C. No overclock."""
from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from .paths import RUN_DIR, ensure_dirs

TEMP_HARD_C = 80
TEMP_PAUSE_UNTIL_C = 74
TEMP_SOFT_C = 76
POLL_SEC = 2.0
HEALTH_PATH = RUN_DIR / "gpu_health.json"


def snapshot() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        ).strip()
        parts = [p.strip() for p in raw.split(",")]
        info = {
            "temp_c": float(parts[0]),
            "util": float(parts[1]),
            "mem_used_mb": float(parts[2]),
            "mem_total_mb": float(parts[3]),
            "power_w": float(parts[4]),
            "ok": True,
            "ts": time.time(),
        }
    except Exception as exc:
        info = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "temp_c": None, "ts": time.time()}
    ensure_dirs()
    HEALTH_PATH.write_text(json.dumps(info), encoding="utf-8")
    return info


def wait_if_hot(*, hard_c: float = TEMP_HARD_C, resume_c: float = TEMP_PAUSE_UNTIL_C, soft_c: float = TEMP_SOFT_C) -> dict[str, Any]:
    info = snapshot()
    temp = info.get("temp_c")
    paused = 0.0
    if temp is None:
        return info
    while float(temp) >= hard_c:
        time.sleep(POLL_SEC)
        paused += POLL_SEC
        info = snapshot()
        temp = info.get("temp_c")
        if temp is None or paused > 180:
            break
    if temp is not None and float(temp) >= soft_c:
        time.sleep(0.3)
        info = snapshot()
    info["paused_sec"] = paused
    return info
