"""OVERNIGHT_STATUS.md writer."""
from __future__ import annotations

from typing import Any

from .io_utils import git_branch, git_commit, utc_now
from .paths import PROJECT_ROOT, REPORT_ROOT, ensure_dirs
from .protocol import protocol_hash

STATUS_PATH = PROJECT_ROOT / "OVERNIGHT_STATUS.md"


def write_status(*, phase: str, completed: list[str], evidence: list[str], decision: str, next_step: str, blockers: list[str] | None = None, extra: str = "") -> None:
    ensure_dirs()
    blockers = blockers or []
    body = f"""# OVERNIGHT_STATUS — hybrid_superiority_v2

- Updated: `{utc_now()}`
- Branch: `{git_branch()}`
- Commit: `{git_commit()}`
- Protocol hash: `{protocol_hash()}`
- Phase: **{phase}**

## Completed
{chr(10).join(f'- {item}' for item in completed) or '- (none yet)'}

## Evidence
{chr(10).join(f'- {item}' for item in evidence) or '- (none yet)'}

## Decision
{decision}

## Next
{next_step}

## Blockers
{chr(10).join(f'- {item}' for item in blockers) if blockers else '- none'}

{extra}
"""
    STATUS_PATH.write_text(body, encoding="utf-8")
    (REPORT_ROOT / "OVERNIGHT_STATUS.md").write_text(body, encoding="utf-8")
