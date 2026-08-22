"""Keep GPU <= 80 C while training. No overclock. nvidia-smi -pl needs admin so software throttle is primary."""
from __future__ import annotations

import json
import subprocess
import time
from typing import Any

from .paths import HEALTH_PATH, ensure_dirs

TEMP_HARD_C = 80
TEMP_PAUSE_UNTIL_C = 74
TEMP_SOFT_C = 76
POLL_SEC = 2.0


def _query() -> dict[str, Any]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        ).strip()
        parts = [p.strip() for p in raw.split(",")]
        return {
            "temp_c": float(parts[0]),
            "util": float(parts[1]),
            "mem_used_mb": float(parts[2]),
            "mem_total_mb": float(parts[3]),
            "power_w": float(parts[4]),
            "power_limit_w": float(parts[5]),
            "sm_clock": float(parts[6]) if parts[6] not in {"[N/A]", "N/A"} else None,
            "ok": True,
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "temp_c": None}


def snapshot() -> dict[str, Any]:
    info = _query()
    info["ts"] = time.time()
    ensure_dirs()
    HEALTH_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def gpu_temp() -> float | None:
    info = _query()
    return info.get("temp_c")


def wait_if_hot(*, hard_c: float = TEMP_HARD_C, resume_c: float = TEMP_PAUSE_UNTIL_C, soft_c: float = TEMP_SOFT_C) -> dict[str, Any]:
    """Block while GPU is at/above hard_c. Soft-throttle with a short sleep near the cap."""
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
        if temp is None:
            break
        if paused > 180:
            break
    if temp is not None and float(temp) >= soft_c:
        time.sleep(0.4)
        info = snapshot()
        info["soft_throttle"] = True
    info["paused_sec"] = paused
    return info


def try_set_power_limit(watts: int) -> bool:
    try:
        subprocess.check_call(["nvidia-smi", "-pl", str(int(watts))], timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
