"""Assemble HYBRID_VNEXT_PHASE4_FINAL_REPORT.md."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P4 = ROOT / "artifacts" / "hybrid_vnext" / "phase4"


def _j(name, default=None):
    path = P4 / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v, d=4):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    gate = _j("INNER_SUPERIORITY_GATE.json") or {}
    status = gate.get("status") or (_j("phase4_status.json") or {}).get("status") or "NOT_READY_FOR_FINAL_EVAL"
    if _j("FINAL_ACCEPTANCE.json", {}).get("accepted"):
        status = "FINAL_HYBRID_ACCEPTED"
    elif _j("FINAL_ACCEPTANCE.json") and _j("FINAL_ACCEPTANCE.json").get("accepted") is False:
        status = "FINAL_HYBRID_NOT_ACCEPTED"
    contract = _j("ONE_MODEL_CONTRACT.json") or {}
    leak = _j("LEAKAGE_AUDIT.json") or {}
    over = _j("OVERFIT_AUDIT.json") or {}
    ceil = _j("BASELINE_CEILING.json") or {}
    xgb = _j("XGBOOST_REMOVAL_MANIFEST.json") or {}
    svm = _j("SVM_CONFIG.json") or {}
    screen = _j("GATE1_SCREEN.json") or {}
    robust = _j("robust_summary.json") or {}
    shift = _j("DATASET_SHIFT_ROBUSTNESS.json") or {}
    selected = _j("SELECTED_STRATEGY.json") or {}
    lines = [
        "# HYBRID VNEXT PHASE 4 — SUPERIORITY FINAL REPORT",
        "",
        f"**Status:** `{status}`",
        "",
        "One Hybrid C0, shared structural tuple 128/64/128, UCI evaluated at S0→S1→S2, OULAD evaluated at 20→35→50→75→100.",
        "Active baselines: LR / DT / RF / SVM / MLP. XGBoost is not an active comparator. Outer unused unless both inner gates pass.",
        "",
        "## A. Executive conclusion",
        "",
        f"`{status}`",
        "",
        f"- UCI Hybrid `{_fmt((gate.get('uci') or {}).get('hybrid'))}` vs `{(gate.get('uci') or {}).get('best_baseline_name')}` `{_fmt((gate.get('uci') or {}).get('best_baseline'))}` "
        f"(Δ `{_fmt((gate.get('uci') or {}).get('delta'))}`, positive stages `{(gate.get('uci') or {}).get('positive_stages')}`, ok=`{(gate.get('uci') or {}).get('ok')}`)",
        f"- OULAD 5-stage Hybrid `{_fmt((gate.get('oulad') or {}).get('hybrid_5stage'))}` vs `{(gate.get('oulad') or {}).get('best_baseline_name')}` `{_fmt((gate.get('oulad') or {}).get('best_baseline'))}` "
        f"(Δ `{_fmt((gate.get('oulad') or {}).get('delta_5stage'))}`, early Δ `{_fmt((gate.get('oulad') or {}).get('delta_early'))}`, positive `{(gate.get('oulad') or {}).get('positive_stages')}/5`, ok=`{(gate.get('oulad') or {}).get('ok')}`)",
        "- A tie is not a win. Best seed was not selected. Authority was not updated unless FINAL_HYBRID_ACCEPTED.",
        "",
        "## B. One-model contract",
        "",
        "```json",
        json.dumps(contract, indent=2),
        "```",
        "",
        "## C. C0 topology integrity",
        "",
        f"- topology_hash `{contract.get('topology_hash')}`",
        "- parallel CNN ∥ BiLSTM, 3-way masked softmax, availability [1, temporal, temporal]",
        "- Structural HPO was not reopened.",
        "",
        "## D. Leakage audit",
        "",
        f"- pass=`{leak.get('pass')}`",
        f"- UCI: `{json.dumps(leak.get('uci'))}`",
        f"- OULAD: `{json.dumps({k: leak.get('oulad', {}).get(k) for k in ('has_100pct', 'forbidden_in_100pct', 'n_100pct', 'n_20pct')})}`",
        "",
        "## E. Overfitting audit",
        "",
        "```json",
        json.dumps(over, indent=2)[:4000],
        "```",
        "",
        "## F. Active baseline ceiling",
        "",
        "```json",
        json.dumps({d: {k: (ceil.get(d) or {}).get(k) for k in ("macro", "strongest")} for d in ("uci", "oulad")}, indent=2),
        "```",
        "",
        "## G. XGBoost removal",
        "",
        f"- Active roster: `{xgb.get('active_roster')}`",
        f"- Historical provenance preserved: `{xgb.get('historical_provenance_preserved')}`",
        f"- Active surface hits: `{xgb.get('active_surface_hits')}`",
        "",
        "## H. SVM integration",
        "",
        "```json",
        json.dumps(svm, indent=2),
        "```",
        "",
        "## I. Training superiority ladder",
        "",
        f"- Selected strategy: `{json.dumps(selected)}`",
        f"- Screen: `{json.dumps({k: {d: (v.get(d) or {}).get('macro') for d in ('uci', 'oulad')} for k, v in screen.items()}, indent=2)[:3000]}`",
        "",
        "## J. UCI S0",
        f"`{(robust.get('uci') or {}).get('stage_means', {}).get('S0')}`",
        "",
        "## K. UCI S1",
        f"`{(robust.get('uci') or {}).get('stage_means', {}).get('S1')}`",
        "",
        "## L. UCI S2",
        f"`{(robust.get('uci') or {}).get('stage_means', {}).get('S2')}`",
        "",
        "## M. UCI information-growth curve",
        "",
        "See `INFORMATION_GROWTH_ANALYSIS.csv`.",
        "",
        "## N. OULAD 20",
        f"`{(robust.get('oulad') or {}).get('stage_means', {}).get('20pct')}`",
        "",
        "## O. OULAD 35",
        f"`{(robust.get('oulad') or {}).get('stage_means', {}).get('35pct')}`",
        "",
        "## P. OULAD 50",
        f"`{(robust.get('oulad') or {}).get('stage_means', {}).get('50pct')}`",
        "",
        "## Q. OULAD 75",
        f"`{(robust.get('oulad') or {}).get('stage_means', {}).get('75pct')}`",
        "",
        "## R. OULAD 100",
        f"`{(robust.get('oulad') or {}).get('stage_means', {}).get('100pct')}`",
        "",
        "## S. OULAD information-growth curve",
        "",
        "See `INFORMATION_GROWTH_ANALYSIS.csv`. 100% remains one state of the same checkpoint. Length≈Withdrawn shortcut is diagnosed, not used as a feature.",
        "",
        "## T. Dataset-shift robustness",
        "",
        "```json",
        json.dumps(shift, indent=2),
        "```",
        "",
        "## U. Gate/branch diagnostics",
        "",
        "See `GATE_DIAGNOSTICS.csv` when written from robust payloads. UCI S0 must keep tabular_mass=1.",
        "",
        "## V. Robust inner superiority",
        "",
        "```json",
        json.dumps(gate, indent=2),
        "```",
        "",
        "## W. Final outer results if allowed",
        "",
        "Outer opens only after both UCI and OULAD strict gates pass.",
        "",
        "## X. Paired bootstrap if allowed",
        "",
        "Not computed unless outer ran.",
        "",
        "## Y. Authority decision",
        "",
        "`src/prediction` is updated only if FINAL_HYBRID_ACCEPTED.",
        "",
        "## Z. Limitations",
        "",
        "- UCI S0 has no temporal signal; trees remain strong on static tabular features.",
        "- OULAD 100% length≈Withdrawn shortcut exists and is reported, not exploited.",
        "- Numeric HPO is applied only after a winning training family is identified.",
        "- Tie ≠ win.",
        "",
        "## Required scientific answers",
        "",
        f"- Q1 UCI beat strongest active? `{(gate.get('uci') or {}).get('ok')}`",
        f"- Q2 OULAD beat strongest active? `{(gate.get('oulad') or {}).get('ok')}`",
        f"- Q3 same Hybrid across dataset nature? `true` (one C0 / one strategy family)",
        "- Q4 information growth: see section M/S",
        "- Q5 temporal gate: see diagnostics",
        "- Q6 overfit: see section E",
        f"- Q7 leakage-safe? `{leak.get('pass')}`",
        f"- Q8 mechanism: `{selected.get('name')}`",
        f"- Q9 3×3 stability: UCI std `{_fmt((robust.get('uci') or {}).get('std'))}` OULAD std `{_fmt((robust.get('oulad') or {}).get('std'))}`",
        f"- Q10 defensible superiority? `{status == 'FINAL_HYBRID_ACCEPTED'}`",
        "",
    ]
    text = "\n".join(lines) + "\n"
    dest = ROOT / "reports" / "hybrid_vnext" / "phase4"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "HYBRID_VNEXT_PHASE4_FINAL_REPORT.md").write_text(text, encoding="utf-8")
    print("WROTE_PHASE4_REPORT", status)


if __name__ == "__main__":
    main()
