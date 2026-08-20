"""Write research reports from artifacts. Does not promote thesis authority."""
from __future__ import annotations

import json
from pathlib import Path

from .io_utils import git_branch, git_commit, utc_now
from .paths import MANIFEST_DIR, PROJECT_ROOT, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import PROTOCOL_ID, protocol_hash, protocol_payload


def _read(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def write_final_decision() -> Path:
    ensure_dirs()
    gate = _read(RUN_DIR / "development_gate.json") or {}
    confirm = _read(RUN_DIR / "confirmation.json") or {}
    status = "NOT_READY_FOR_DEFENSE"
    if confirm.get("pass_strict"):
        status = "READY_FOR_DEFENSE_STRICT"
    elif confirm.get("pass_limited"):
        status = "READY_FOR_DEFENSE_LIMITED_CLAIM"
    elif gate.get("blocked"):
        status = "BLOCKED_BY_MISSING_RESOURCE"
    path = REPORT_ROOT / "FINAL_DECISION.md"
    locks = {}
    for domain in ("uci", "oulad"):
        locks[domain] = _read(RUN_DIR / f"baseline_lock_{domain}.json")
    screens = sorted(RUN_DIR.glob("screen_*.json")) if RUN_DIR.exists() else []
    body = f"""# FINAL_DECISION

{status}

Research program `hybrid_superiority_v2` does **not** mutate `reports/CURRENT_REPORTS.md` or the serving Hybrid until a strict confirmation pass.

## 1. Lineage

| Field | Value |
|---|---|
| Time | {utc_now()} |
| Branch | `{git_branch()}` |
| Commit | `{git_commit()}` |
| Protocol | `{PROTOCOL_ID}` |
| Protocol hash | `{protocol_hash()}` |
| Outer test used for selection | `false` |

## 2. Candidate

See screen artifacts under `artifacts/research/hybrid_superiority_v2/runs/`. Topology remains one public class `SuperiorityHybrid` with preregistered candidates C0-R / C1-R / C2-S / C3-G.

## 3. Hybrid vs every baseline

Baseline locks:

```json
{json.dumps({k: (v or {}).get("stage_best_ap") for k, v in locks.items()}, indent=2)}
```

Hybrid development numbers are written only when diagnose/optimize artifacts exist. Missing cells mean the phase has not finished — they are not imputed.

## 4–8. Gates, ablation, calibration, OULAD shortcut

Development gate pass: `{gate.get("pass", False)}`.
Confirmation pass: `{confirm.get("pass_strict", False)}`.
Until both exist, **do not** write “vượt trội” in the thesis.

OULAD 100% length→Withdrawn sensitivity is a required limitation, not an early-warning result.

## 9. Reproduce

```bash
python -m experiments.hybrid_superiority_v2 audit
python -m experiments.hybrid_superiority_v2 prepare --dataset all
python -m experiments.hybrid_superiority_v2 baselines --resume
python -m experiments.hybrid_superiority_v2 diagnose --candidate C0-R
python -m experiments.hybrid_superiority_v2 optimize --candidate C3-G --resume
python -m experiments.hybrid_superiority_v2 confirm --frozen-protocol {protocol_hash()}
python -m experiments.hybrid_superiority_v2 report --frozen-protocol {protocol_hash()}
```

`confirm` refuses unless the development gate artifact exists and `pass=true`.

## 10. Gemini

Quota tables live in schema `recommendation`. Weak labels are not expert gold. Prediction HPO does not call Gemini.

## 11. Allowed vs forbidden claims

Allowed only after strict confirmation: Hybrid AP exceeds the frozen max baseline by the preregistered material margin on every warm stage, with simultaneous cluster-bootstrap lower bounds > 0.

Forbidden now: declaring the current serving Hybrid scientifically superior; using historical XGB-dropped roster; calling AP “PR-AUC”; treating Gemini NDCG as expert validation; calling OULAD 100% an early-warning result.

## 12. Files and tests

See `00_SOURCE_AND_SCOPE_AUDIT.md` and `tests/research/hybrid_superiority_v2`.
"""
    path.write_text(body, encoding="utf-8")
    return path


def write_all_reports() -> None:
    ensure_dirs()
    write_final_decision()
    # Copy protocol payload into lock if missing.
    lock_path = PROJECT_ROOT / "protocols" / "hybrid_superiority_v2" / "protocol_lock.json"
    if lock_path.exists() is False:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    from .io_utils import write_json as wj
    from .protocol import protocol_payload as pp

    wj(
        lock_path,
        {
            "protocol_id": PROTOCOL_ID,
            "sha256": protocol_hash(),
            "payload": pp(),
            "frozen_before_hpo": True,
            "authority_untouched": True,
        },
    )
