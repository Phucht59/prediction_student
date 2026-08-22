"""OVERNIGHT_STATUS writer for C4-STAR v2.1."""
from __future__ import annotations

import json
from typing import Any

from experiments.hybrid_superiority_v2.io_utils import git_branch, git_commit, utc_now

from .paths import HEALTH_PATH, REPORT_ROOT, STATE_PATH, ensure_dirs
from .protocol import protocol_hash
from .thermal import snapshot


def load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"phase": "boot", "completed": [], "failed": [], "best": {}}


def save_state(state: dict[str, Any]) -> None:
    ensure_dirs()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def write_status(
    *,
    phase: str,
    completed: list[str] | None = None,
    evidence: list[str] | None = None,
    decision: str = "",
    next_step: str = "",
    blockers: list[str] | None = None,
    extra: str = "",
    best: dict | None = None,
    eta: str = "",
    study: str = "",
) -> None:
    ensure_dirs()
    hw = snapshot()
    temp = hw.get("temp_c")
    body = f"""# OVERNIGHT_STATUS — C4-STAR v2.1

- Updated: `{utc_now()}`
- Branch: `{git_branch()}`
- Commit: `{git_commit()}`
- Protocol hash: `{protocol_hash()}`
- Phase: **{phase}**
- Study: `{study or "—"}`
- GPU temp: `{temp}` C (hard cap 80)
- GPU util/power: `{hw.get("util")}%` / `{hw.get("power_w")}W`
- VRAM: `{hw.get("mem_used_mb")} / {hw.get("mem_total_mb")} MiB`
- Best J / deltas: `{json.dumps(best or {}, default=str)}`
- ETA: `{eta or "throughput-based after first completed trial"}`

## Completed
{chr(10).join(f"- {x}" for x in (completed or [])) or "- (none yet)"}

## Evidence
{chr(10).join(f"- {x}" for x in (evidence or [])) or "- (none yet)"}

## Decision
{decision}

## Next
{next_step}

## Blockers
{chr(10).join(f"- {x}" for x in (blockers or [])) if blockers else "- none"}

{extra}
"""
    (REPORT_ROOT / "OVERNIGHT_STATUS.md").write_text(body, encoding="utf-8")
