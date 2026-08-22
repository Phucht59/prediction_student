"""Write remaining v2.1 reports from artifacts. Does not promote serving Hybrid."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.hybrid_superiority_v2.io_utils import git_commit, utc_now
from experiments.hybrid_superiority_v2.protocol import material_margin

from .paths import REPORT_ROOT, RUN_DIR, ensure_dirs
from .protocol import PROTOCOL_ID, protocol_hash, stages_for, warm_for, cold_for


def _load(name: str):
    path = RUN_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _status_token() -> str:
    uci = (_load("robust_uci.json") or {}).get("gate") or _load("development_gate_uci.json") or {}
    oulad = (_load("robust_oulad.json") or {}).get("gate") or _load("development_gate_oulad.json") or {}
    hpo_o = _load("hpo_oulad.json") or {}
    sel = (hpo_o.get("best_attrs") or {}).get("selection") or {}
    if uci.get("pass") and oulad.get("pass"):
        return "DEVELOPMENT_GOLD_MATERIAL_SUPERIORITY"
    # no-loss: all warm point estimates positive
    oulad_no_loss = False
    if oulad.get("checks"):
        oulad_no_loss = all(c.get("pass_positive") for c in oulad["checks"] if "pass_positive" in c) and oulad.get("cold_ok", False)
    elif sel.get("n_warm_loss") == 0:
        oulad_no_loss = True
    uci_fail = True
    if uci.get("checks"):
        uci_fail = any(not c.get("pass_positive") for c in uci["checks"] if "pass_positive" in c)
    if oulad_no_loss and not uci.get("pass"):
        # joint protocol requires both domains
        return "DEVELOPMENT_GATE_FAILED"
    if uci.get("pass") is False or oulad.get("pass") is False:
        return "DEVELOPMENT_GATE_FAILED"
    return "DEVELOPMENT_GATE_FAILED"


def write_all_v21_reports() -> None:
    ensure_dirs()
    uci_lock = _load("baseline_lock_uci.json") or {}
    oulad_lock = _load("baseline_lock_oulad.json") or {}
    screen = _load("joint_screen_summary.json") or {}
    hpo_u = _load("hpo_uci.json") or {}
    hpo_o = _load("hpo_oulad.json") or {}
    robust_u = _load("robust_uci.json") or {}
    robust_o = _load("robust_oulad.json") or {}
    ladder = _load("c4_ladder.json") or {}
    token = _status_token()

    # 03 already exists; refresh from json
    lines = ["# 03 Joint screen", "", "Fold 0, seeds 42/1201/2026, inner VALID. Constrained J (warm losses heavily penalized).", ""]
    lines.append("| Domain | Candidate | mean J | mean warm losses | mean min r |")
    lines.append("|---|---|---:|---:|---:|")
    for row in screen.get("summary") or []:
        lines.append(f"| {row.get('domain')} | {row.get('candidate')} | {_fmt(row.get('mean_J'),3)} | {_fmt(row.get('mean_warm_loss'),2)} | {_fmt(row.get('mean_min_r'),3)} |")
    lines += ["", "UCI: all four backbones lose both warm stages vs CatBoost 3×3 ceiling under this screen budget.", "OULAD: **C0-R** is the least-bad backbone (fewest warm losses).", ""]
    (REPORT_ROOT / "03_EXISTING_CANDIDATE_JOINT_SCREEN.md").write_text("\n".join(lines), encoding="utf-8")

    diag_lines = [
        "# 04 Temporal diagnostics",
        "",
        "Inner VALID. UCI shuffle gap ~0 is expected (T≤2).",
        "OULAD C0-R seed 42 (identity vs shuffle/reverse):",
        "",
        "| Stage | identity AP | shuffle gap | reverse gap |",
        "|---|---:|---:|---:|",
    ]
    d0 = _load("diagnose_oulad_C0-R_s42.json") or {}
    for s in stages_for("oulad"):
        ident = (d0.get("identity") or {}).get(s)
        sg = (d0.get("shuffle_gap") or {}).get(s)
        rg = (d0.get("reverse_gap") or {}).get(s)
        diag_lines.append(f"| {s} | {_fmt(ident)} | {_fmt(sg)} | {_fmt(rg)} |")
    diag_lines += [
        "",
        "OULAD warm shuffle gaps are **positive** (~0.008–0.013). Reverse gaps larger (~0.013–0.024).",
        "Order/dynamics exist on OULAD; they are not large enough by themselves to clear the material margin vs XGB/LR.",
        "See `runs/diagnose_*.json`.",
        "",
    ]
    (REPORT_ROOT / "04_TEMPORAL_SIGNAL_DIAGNOSTICS.md").write_text("\n".join(diag_lines), encoding="utf-8")

    lad = ["# 05 C4-STAR ladder", "", "Fold 0 seed 42 after AMP-safe KD fix. Same topology family.", ""]
    lad += ["| Domain | Mech | J | notes |", "|---|---|---:|---|"]
    for rec in ladder.get("rows") or []:
        if rec.get("error"):
            lad.append(f"| {rec.get('domain')} | {rec.get('mechanism')} | — | `{rec['error'][:80]}` |")
        else:
            lad.append(f"| {rec.get('domain')} | {rec.get('mechanism')} | {_fmt(rec.get('J'),3)} | epoch {rec.get('best_epoch')} |")
    lad.append("")
    (REPORT_ROOT / "05_C4_STAR_LADDER.md").write_text("\n".join(lad), encoding="utf-8")

    hpo_md = [
        "# 06 HPO and convergence",
        "",
        "Optuna constrained J, fold 0 seed 42, inner VALID. Study names `c4_v21_hpo_{domain}_ce758268ce0c`.",
        "",
        f"UCI complete trials (last snapshot): **{hpo_u.get('n_complete')}**. Best J `{_fmt(hpo_u.get('best'),3)}`.",
        f"Params: `{json.dumps(hpo_u.get('best_params'))}`",
        f"AP: `{json.dumps((hpo_u.get('best_attrs') or {}).get('ap'))}`",
        "",
        "UCI still **loses both warm stages** vs CatBoost (S1 0.769 / S2 0.907).",
        "",
        f"OULAD complete trials (last snapshot): **{hpo_o.get('n_complete')}**. Best J `{_fmt(hpo_o.get('best'),3)}`.",
        f"Params: `{json.dumps(hpo_o.get('best_params'))}`",
        f"AP: `{json.dumps((hpo_o.get('best_attrs') or {}).get('ap'))}`",
        "",
        "OULAD HPO winner (fold 0 / seed 42) has **n_warm_loss=0** vs the v2.1 3×3 ceiling, but normalized margins r_s ≪ 1 (not Gold).",
        "That single-fold result is **not** confirmation. See 07 for 3×3.",
        "",
    ]
    (REPORT_ROOT / "06_HPO_AND_CONVERGENCE.md").write_text("\n".join(hpo_md), encoding="utf-8")

    def _robust_table(domain, robust, lock):
        lines = [f"## {domain.upper()}", ""]
        ceil = (lock or {}).get("stage_best_ap") or (robust or {}).get("ceiling") or {}
        mean = (robust or {}).get("mean") or {}
        if not mean:
            lines.append("3×3 not written yet.")
            lines.append("")
            return lines
        lines.append("| Stage | Ceiling | C4 mean | Δ | material | pos | material pass |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        gate = (robust or {}).get("gate") or {}
        by = {c.get("stage"): c for c in gate.get("checks") or []}
        for s in stages_for(domain):
            b = ceil.get(s, {})
            bap = b.get("ap") if isinstance(b, dict) else b
            h = mean.get(s)
            chk = by.get(s) or {}
            lines.append(
                f"| {s} | {_fmt(bap)} ({b.get('model') if isinstance(b, dict) else ''}) | {_fmt(h)} | {_fmt(chk.get('delta'))} | {_fmt(chk.get('material_margin') or chk.get('guardrail'))} | {chk.get('pass_positive') or chk.get('pass_cold')} | {chk.get('pass_material')} |"
            )
        lines.append("")
        lines.append(f"Gate pass=`{gate.get('pass')}`. Runs=`{(robust or {}).get('n_runs')}`. Mechanism=`{(robust or {}).get('mechanism')}`.")
        lines.append("")
        return lines

    r7 = ["# 07 Robust 3×3", "", "Inner 3 folds × 3 seeds. Outer unused.", ""]
    r7 += _robust_table("uci", robust_u, uci_lock)
    r7 += _robust_table("oulad", robust_o, oulad_lock)
    (REPORT_ROOT / "07_ROBUST_3X3_RESULTS.md").write_text("\n".join(r7), encoding="utf-8")

    (REPORT_ROOT / "08_ABLATION_AND_SYNERGY.md").write_text(
        "\n".join(
            [
                "# 08 Ablation and synergy",
                "",
                "Ladder M0–M7 is the mechanism ablation (fold 0 seed 42).",
                "UCI: GroupDRO (M4+) slightly better J than M0 but still far below CatBoost.",
                "OULAD: KD (M2+) reduces warm-loss penalty vs M0/M1; HPO then found M4 with n_warm_loss=0 on fold 0.",
                "CNN-only / BiLSTM-only sequence-only roster was not a separate 3×3 (C4 branch_mode ablations remain fold-0).",
                "Do not claim CNN–BiLSTM synergy on UCI: T≤2 and CatBoost still dominates S2.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (REPORT_ROOT / "09_CALIBRATION_AND_OPERATIONAL_UTILITY.md").write_text(
        "\n".join(
            [
                "# 09 Calibration and operational utility",
                "",
                "Primary metric remains AP. ECE/Brier were not the HPO primary.",
                "OULAD 100% operational risk-set still has 94 Withdrawn — not an early-warning panel.",
                "Recall@20 is stored on robust rows when 3×3 finished.",
                "No Gemini labels used.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    thesis = [
        "# Thesis-ready tables v2.1 (research only)",
        "",
        f"Protocol `{PROTOCOL_ID}` hash `{protocol_hash()[:12]}…`. Primary=AP. Outer unused.",
        "",
        "## OULAD ceiling 3×3 (v2.1 lock, not SPEED_FINISH)",
        "",
    ]
    sb = (oulad_lock or {}).get("stage_best_ap") or {}
    thesis.append("| Stage | Model | AP |")
    thesis.append("|---|---|---:|")
    for s, v in sb.items():
        thesis.append(f"| {s} | {v.get('model')} | {_fmt(v.get('ap'))} |")
    thesis += ["", "## C4-STAR HPO fold0 (not 3×3)", ""]
    thesis.append(f"OULAD best AP `{json.dumps((hpo_o.get('best_attrs') or {}).get('ap'))}` J={_fmt(hpo_o.get('best'),3)} n_warm_loss={(hpo_o.get('best_attrs') or {}).get('selection', {}).get('n_warm_loss')}")
    thesis.append("")
    thesis.append(f"UCI best AP `{json.dumps((hpo_u.get('best_attrs') or {}).get('ap'))}` — still below CatBoost.")
    thesis.append("")
    (REPORT_ROOT / "THESIS_READY_TABLES_V2_1.md").write_text("\n".join(thesis), encoding="utf-8")

    body = f"""{token}

# FINAL_DECISION_V2_1

Outer test **not opened**. Serving Hybrid **not** promoted.

| Field | Value |
|---|---|
| Time | {utc_now()} |
| Commit | `{git_commit()}` |
| Protocol | `{PROTOCOL_ID}` |
| Hash | `{protocol_hash()}` |

## Verified ceilings

- UCI CatBoost 3×3: S0 0.5010 / S1 0.7694 / S2 0.9067
- OULAD v2.1 3×3: 20% LR 0.7678; 35–100% XGB 0.8077 / 0.8545 / 0.8969 / 0.9245

## C4-STAR vs ceiling (official = robust 3×3, not HPO fold 0)

UCI M4 9-run mean: S0 0.493 / S1 0.775 / S2 0.856 vs CatBoost 0.501 / 0.769 / 0.907. S1 slightly positive but **not material**; S2 loses ~0.051. **UCI gate fail.**

OULAD M4 9-run mean loses every warm stage by ~0.002–0.005 vs XGB (35% 0.803 vs 0.808; 100% 0.921 vs 0.924). Cold 20% within guardrail. Fold-0 HPO had n_warm_loss=0; **3×3 does not replicate that.** **OULAD gate fail.**

Joint development gate requires both domains. Combined: **FAIL**. See `07_ROBUST_3X3_RESULTS.md`.

## Temporal

OULAD shuffle/reverse gaps are positive. Sequence order is real, not sufficient for material AP.

## Claims

Forbidden: vượt trội; OULAD 100% early-warning; SPEED_FINISH as confirmation.

Allowed: protocol v2.1 frozen; OULAD ceiling rebuilt 3×3; C0-R still best existing backbone on OULAD screen; C4-STAR M4 can match/slightly exceed the OULAD envelope on one fold without material margin; UCI Hybrid/C4 does not beat CatBoost at S2.
"""
    (REPORT_ROOT / "FINAL_DECISION_V2_1.md").write_text(body, encoding="utf-8")
    (REPORT_ROOT / "NEXT_ACTIONS.md").write_text(
        "# NEXT_ACTIONS\n\n"
        "If 3×3 OULAD still no-loss-only and UCI fails: do **not** open outer test.\n"
        "Resume: `py -3.10 -u -m experiments.c4_star overnight`\n",
        encoding="utf-8",
    )
