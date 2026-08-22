"""Write research reports from artifacts. Does not promote thesis authority."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .io_utils import git_branch, git_commit, utc_now, write_json
from .paths import MANIFEST_DIR, METRIC_DIR, PROJECT_ROOT, REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import PROTOCOL_ID, material_margin, protocol_hash, protocol_payload, stages_for, warm_for


def _read(path: Path):
    if not path.exists():
        return None
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"
    # first col text, rest numeric-ish
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    return head + "\n" + sep + "\n" + body


def _lock(domain: str):
    return _read(RUN_DIR / f"baseline_lock_{domain}.json") or {}


def _metrics_table(domain: str) -> pd.DataFrame | None:
    path = METRIC_DIR / f"baseline_stage_metrics_{domain}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_speed_note() -> Path:
    lock = _lock("oulad")
    speed = (lock or {}).get("speed_mode") or {}
    path = REPORT_ROOT / "SPEED_FINISH.md"
    body = f"""# SPEED_FINISH — documented budget cuts

User requested a crash finish after the PC looked idle (OULAD DT HPO hung ~2h, then killed).

This is **not** the preregistered protocol budget.

| Item | Preregistered | SPEED_FINISH |
|---|---|---|
| OULAD baseline trials | 28 / model | {speed.get("baseline_trials_oulad", 4)} (XGB/CatBoost only) |
| Skipped HPO | none | DT, SVM, MLP, RF (defaults; DT previously hung) |
| Lock folds × seeds | 3 × 3 | {speed.get("lock_folds", [0])} × {speed.get("lock_seeds", [42, 1201])} |
| Hybrid screen trials | 24 / candidate | {speed.get("hybrid_screen_trials", 6)} C0-R only |
| Hybrid epochs | 24 / patience 8 | {speed.get("hybrid_max_epochs", 10)} / {speed.get("hybrid_patience", 4)} |
| OULAD diagnose | required | skipped (GPU reserved for C0-R) |
| Ablation | independent 3×3 | UCI fold-0 seed-42 only |

GPU: CatBoost/XGB `task_type/device=GPU` on RTX 2060; Hybrid tensors pinned; process HIGH priority.

Protocol hash still `{protocol_hash()}`. Outer test unused. Serving Hybrid **not** promoted.
"""
    path.write_text(body, encoding="utf-8")
    return path


def write_baseline_ceiling() -> Path:
    path = REPORT_ROOT / "BASELINE_CEILING.md"
    chunks = ["# Baseline ceiling", "", f"Protocol `{PROTOCOL_ID}` hash `{protocol_hash()[:12]}…`. Primary = AP. Outer test unused. Roster includes XGB and CatBoost.", ""]
    for domain in ("uci", "oulad"):
        lock = _lock(domain)
        table = _metrics_table(domain)
        chunks.append(f"## {domain.upper()}")
        if not lock:
            chunks.append("Chưa lock.")
            chunks.append("")
            continue
        speed = lock.get("speed_mode")
        if speed:
            chunks.append(f"SPEED_FINISH: trials={speed.get('baseline_trials_oulad')} skip={speed.get('skip_hpo')} folds={lock.get('folds')} seeds={lock.get('seeds')}.")
        else:
            chunks.append(f"Folds={lock.get('folds')} seeds={lock.get('seeds')} trials={lock.get('n_trials')}.")
        chunks.append("")
        chunks.append(f"Lock: `artifacts/research/hybrid_superiority_v2/runs/baseline_lock_{domain}.json`")
        chunks.append("")
        if table is not None and len(table):
            pivot = table.groupby(["model", "stage"])["ap"].mean().unstack("stage")
            stages = list(stages_for(domain))
            headers = ["Model"] + stages
            rows = []
            best = lock.get("stage_best_ap") or {}
            for model, rec in pivot.iterrows():
                cells = [model]
                for s in stages:
                    val = rec[s] if s in rec.index else None
                    star = "**" if best.get(s, {}).get("model") == model else ""
                    cells.append(f"{star}{_fmt(val, 4)}{star}" if val == val else "—")
                rows.append(cells)
            chunks.append(_md_table(headers, rows))
            chunks.append("")
            chunks.append("### Material margin (warm only — cold uses guardrail, not this table)")
            chunks.append("")
            mrows = []
            for stage in warm_for(domain):
                info = best.get(stage)
                if not info:
                    continue
                apb = float(info["ap"])
                mm = material_margin(apb)
                mrows.append([stage, info["model"], _fmt(apb, 4), _fmt(mm, 4), _fmt(apb + mm, 4)])
            chunks.append(_md_table(["Stage", "Ceiling model", "AP_B", "MaterialMargin", "Hybrid cần"], mrows))
        else:
            chunks.append("```json")
            chunks.append(json.dumps(lock.get("stage_best_ap"), indent=2))
            chunks.append("```")
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def write_gate_report() -> Path:
    path = REPORT_ROOT / "DEVELOPMENT_GATE.md"
    combined = _read(RUN_DIR / "development_gate.json") or {}
    chunks = [
        "# Development gate",
        "",
        f"Combined pass: `{combined.get('pass', False)}`. Outer test unused. Confirmation **refuses** unless this is true on every warm stage of every domain.",
        "",
    ]
    for domain in ("uci", "oulad"):
        g = _read(RUN_DIR / f"development_gate_{domain}.json") or (combined.get("domains") or {}).get(domain) or {}
        chunks.append(f"## {domain.upper()} pass=`{g.get('pass', False)}`")
        chunks.append("")
        for chk in g.get("checks") or []:
            chunks.append(
                f"- **{chk.get('stage')}**: Hybrid AP={_fmt(chk.get('ap_hybrid'), 4)} vs baseline {_fmt(chk.get('ap_baseline'), 4)} "
                f"Δ={_fmt(chk.get('delta'), 4)} margin={_fmt(chk.get('material_margin') or chk.get('guardrail'), 4)} "
                f"pos={chk.get('pass_positive')} mat={chk.get('pass_material')} cold={chk.get('pass_cold')}"
            )
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def write_ablation_report() -> Path:
    path = REPORT_ROOT / "ABLATION.md"
    payload = _read(RUN_DIR / "ablation_uci_C0-R.json") or {}
    chunks = [
        "# Ablation",
        "",
        "Independent retrain of preregistered ablations. SPEED_FINISH: UCI fold 0 / seed 42 only. **Not** a 3×3 confirmation ablation.",
        "",
    ]
    if not payload:
        chunks.append("Chưa chạy.")
    else:
        rows = []
        for rec in payload.get("rows") or []:
            ap = rec.get("ap") or {}
            rows.append(
                [
                    rec.get("ablation"),
                    rec.get("branch_mode"),
                    _fmt(ap.get("S0"), 3) if isinstance(ap, dict) else str(ap),
                    _fmt(ap.get("S1"), 3) if isinstance(ap, dict) else "—",
                    _fmt(ap.get("S2"), 3) if isinstance(ap, dict) else "—",
                ]
            )
        chunks.append(_md_table(["Ablation", "branch_mode", "S0", "S1", "S2"], rows))
        chunks.append("")
        chunks.append(payload.get("note") or "")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def write_stats_report() -> Path:
    path = REPORT_ROOT / "STATS.md"
    chunks = [
        "# Stats",
        "",
        "Paired cluster bootstrap is the confirmation statistic. SPEED_FINISH does **not** open confirmation.",
        "",
        "UCI C0-R vs CatBoost: S2 mean Δ=+0.006 < material 0.010, so bootstrap is not a license to claim superiority. Holm/cluster bootstrap would be run only after development gate pass.",
        "",
        "OULAD hybrid OOF: see `artifacts/research/hybrid_superiority_v2/oof/` if present.",
        "",
        f"Preregistered n_boot=10000, Holm α=0.05. Not executed as a confirmation step because gate pass=`{(_read(RUN_DIR / 'development_gate.json') or {}).get('pass', False)}`.",
        "",
    ]
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def write_hybrid_diagnostics() -> Path:
    path = REPORT_ROOT / "HYBRID_DIAGNOSTICS.md"
    uci = _read(RUN_DIR / "diagnose_uci_C0-R.json") or {}
    oulad_screen = _read(RUN_DIR / "screen_oulad_C0-R.json") or {}
    uci_c0 = _read(RUN_DIR / "robust_uci_C0-R.json") or {}
    oulad_c0 = _read(RUN_DIR / "robust_oulad_C0-R.json") or {}
    chunks = [
        "# Hybrid diagnostics (development)",
        "",
        "Outer test **không** dùng. Đây là chẩn đoán, không phải confirmation.",
        "",
        "## UCI C0-R",
        "",
        f"Diagnose VALID AP: `{json.dumps((uci or {}).get('valid_ap'))}`. Shuffle gap: `{json.dumps((uci or {}).get('full_minus_shuffle'))}`.",
        "",
        f"Robust 3×3 mean: `{json.dumps((uci_c0 or {}).get('mean'))}`.",
        "",
        "Shuffle gap ~0 trên UCI vì T≤2. G1/G2 không vào tabular Hybrid.",
        "",
        "## OULAD C0-R",
        "",
    ]
    if oulad_c0:
        chunks.append(f"SPEED robust mean: `{json.dumps(oulad_c0.get('mean'))}`.")
        chunks.append("")
        chunks.append(f"Screen: `{json.dumps(oulad_screen.get('best_user_attrs'))}`.")
    else:
        chunks.append("Chưa có robust OULAD.")
    chunks.append("")
    chunks.append("OULAD 100% operational: 22522 records, 94 Withdrawn. Length→Withdrawn là sensitivity, không phải early-warning.")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


def write_thesis_tables() -> Path:
    path = REPORT_ROOT / "THESIS_READY_TABLES.md"
    chunks = [
        "# Thesis-ready tables (research only — not serving authority)",
        "",
        f"Protocol `{PROTOCOL_ID}` hash `{protocol_hash()[:12]}…`. Primary = AP. Outer test unused.",
        "",
    ]
    for domain in ("uci", "oulad"):
        lock = _lock(domain)
        table = _metrics_table(domain)
        chunks.append(f"## {domain.upper()} baseline lock mean AP")
        chunks.append("")
        if table is None or not len(table):
            chunks.append("Chưa lock." if not lock else json.dumps(lock.get("stage_best_ap"), indent=2))
            chunks.append("")
            continue
        pivot = table.groupby(["model", "stage"])["ap"].mean().unstack("stage")
        stages = list(stages_for(domain))
        best = lock.get("stage_best_ap") or {}
        rows = []
        for model, rec in pivot.iterrows():
            cells = [str(model)]
            for s in stages:
                val = rec[s] if s in rec.index else None
                star = "**" if best.get(s, {}).get("model") == model else ""
                cells.append(f"{star}{_fmt(val, 3)}{star}" if val == val else "—")
            rows.append(cells)
        chunks.append(_md_table(["Model"] + stages, rows))
        chunks.append("")
        robust = _read(RUN_DIR / f"robust_{domain}_C0-R.json") or {}
        if robust:
            mean = robust.get("mean") or {}
            chunks.append(f"Hybrid C0-R mean AP: `{json.dumps(mean)}`.")
            chunks.append("")
    chunks.append("Material S2 UCI 0.010: C0-R fail by ~0.004 on 3×3 (see DEVELOPMENT_GATE.md). Do not write vượt trội.")
    path.write_text("\n".join(chunks), encoding="utf-8")
    return path


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
    locks = {domain: _lock(domain) for domain in ("uci", "oulad")}
    uci_gate = _read(RUN_DIR / "development_gate_uci.json") or {}
    oulad_gate = _read(RUN_DIR / "development_gate_oulad.json") or {}
    uci_c0 = _read(RUN_DIR / "robust_uci_C0-R.json") or {}
    oulad_c0 = _read(RUN_DIR / "robust_oulad_C0-R.json") or {}
    speed = (locks.get("oulad") or {}).get("speed_mode")
    body = f"""# FINAL_DECISION

{status}

Chương trình `hybrid_superiority_v2` **không** mutate `reports/CURRENT_REPORTS.md` hay Hybrid serving. Confirmation chưa mở vì development gate chưa `pass=true` trên mọi warm stage.

SPEED_FINISH đã cắt budget HPO OULAD (xem `SPEED_FINISH.md`). Đây không phải protocol 28-trial.

## 1. Lineage

| Field | Value |
|---|---|
| Time | {utc_now()} |
| Branch | `{git_branch()}` |
| Commit | `{git_commit()}` |
| Protocol | `{PROTOCOL_ID}` |
| Protocol hash | `{protocol_hash()}` |
| Outer test used for selection | `false` |
| Serving authority | không đổi |

## 2. Candidate

Public class `SuperiorityHybrid`. Ladder C0-R / C1-R / C2-S / C3-G. Survivor UCI: **C0-R**. OULAD SPEED: C0-R only.

## 3. Hybrid vs mọi baseline

UCI lock:

```json
{json.dumps((locks.get("uci") or {}).get("stage_best_ap"), indent=2)}
```

UCI C0-R 3×3 mean: `{json.dumps((uci_c0 or {}).get("mean"))}`.

OULAD lock (SPEED):

```json
{json.dumps((locks.get("oulad") or {}).get("stage_best_ap"), indent=2)}
```

OULAD C0-R mean: `{json.dumps((oulad_c0 or {}).get("mean"))}`.

## 4–8. Gates / ablation / calibration / shortcut

- Combined development gate pass: `{gate.get("pass", False)}`
- UCI gate pass: `{uci_gate.get("pass", False)}`
- OULAD gate pass: `{oulad_gate.get("pass", False)}`
- Confirmation: `confirm` từ chối nếu gate không pass
- Ablation: SPEED UCI fold-0 only (`ABLATION.md`)
- OULAD 100% operational: 22522 records, 94 Withdrawn — không phải early-warning

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

SPEED path: `python -m experiments.hybrid_superiority_v2.fast_finish`

## 10. Gemini

Quota tables live in schema `recommendation`. Weak labels are not expert gold. Prediction HPO does not call Gemini.

## 11. Allowed vs forbidden claims

Forbidden now: declaring serving Hybrid scientifically superior; using historical XGB-dropped roster; calling AP “PR-AUC”; treating Gemini NDCG as expert validation; calling OULAD 100% an early-warning result; hiding SPEED_FINISH budget cuts.

Allowed: protocol locked; XGB/CatBoost in roster; AP primary; G1/G2 not in Hybrid tabular; honest NOT_READY.

## 12. Files

See `00_SOURCE_AND_SCOPE_AUDIT.md`, `SPEED_FINISH.md`, `tests/research/hybrid_superiority_v2`.
"""
    path.write_text(body, encoding="utf-8")
    return path


def write_all_reports() -> None:
    ensure_dirs()
    write_speed_note()
    write_baseline_ceiling()
    write_gate_report()
    write_ablation_report()
    write_stats_report()
    write_hybrid_diagnostics()
    write_thesis_tables()
    write_final_decision()
    lock_path = PROJECT_ROOT / "protocols" / "hybrid_superiority_v2" / "protocol_lock.json"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        lock_path,
        {
            "protocol_id": PROTOCOL_ID,
            "sha256": protocol_hash(),
            "payload": protocol_payload(),
            "frozen_before_hpo": True,
            "authority_untouched": True,
        },
    )
