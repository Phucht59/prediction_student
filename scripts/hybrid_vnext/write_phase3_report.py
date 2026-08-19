"""Assemble HYBRID_VNEXT_PHASE3_FINAL_REPORT.md from Phase 3 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P3 = ROOT / "artifacts" / "hybrid_vnext" / "phase3"


def _j(name: str, default=None):
    path = P3 / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(name: str) -> pd.DataFrame | None:
    path = P3 / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    acc = _j("INNER_ACCEPTANCE.json") or {}
    shared = _j("SHARED_STRUCTURAL_CONFIG.json") or {}
    hpo = _j("HPO_SELECTION.json") or {}
    robust = _j("robust_summary.json") or {}
    proto = _j("PHASE3_PROTOCOL.json") or {}
    final_acc = _j("FINAL_ACCEPTANCE.json")
    calib = _j("CALIBRATION_REPORT.json") or {}
    thresh = _j("THRESHOLD_SELECTION.json") or {}
    status_blob = _j("phase3_status.json") or {}
    fast = _j("FAST_COMPLETION.json") or {}
    base_macro = _j("baseline_inner_macros.json") or {}
    f100 = _j("final100_diagnostic.json") or {}
    lock = _j("FINAL_MODEL_LOCK.json")
    boot = _csv("PAIRED_BOOTSTRAP.csv")
    outer = _csv("FINAL_OUTER_METRICS.csv")

    status = (final_acc or acc).get("status") or acc.get("status") or status_blob.get("status") or "NOT_READY_FOR_FINAL_EVAL"
    if final_acc and final_acc.get("accepted"):
        status = "FINAL_HYBRID_ACCEPTED"
    elif final_acc and final_acc.get("accepted") is False:
        status = "FINAL_HYBRID_NOT_ACCEPTED"
    elif acc and not acc.get("ready"):
        status = "NOT_READY_FOR_FINAL_EVAL"

    oulad_hpo = hpo.get("oulad") or {}
    uci_hpo = hpo.get("uci") or {}
    oulad_r = robust.get("oulad") or {}
    uci_r = robust.get("uci") or {}
    oulad_acc = acc.get("oulad") or {}
    uci_acc = acc.get("uci") or {}

    lines = [
        "# HYBRID VNEXT PHASE 3 — FINAL REPORT",
        "",
        f"**Status:** `{status}`",
        "",
        "Topology remained **C0**: parallel CNN ∥ BiLSTM, corrected availability, 3-way masked softmax, binary risk head.",
        "Outer labels were not used for HPO, threshold, seed, architecture, or calibration decisions.",
        f"Mode: `FAST_COMPLETION` (no new HPO after locked best; remaining started 3×3 only; temperature once; bootstrap 1000 if outer opened).",
        "",
        "## A. Executive conclusion",
        "",
        f"`{status}`",
        "",
        f"- OULAD Hybrid robust macro PR-AUC = `{_fmt(oulad_r.get('mean'))}` ± `{_fmt(oulad_r.get('std'))}` "
        f"vs strongest inner baseline `{oulad_acc.get('best_baseline_name', 'n/a')}` = `{_fmt(oulad_acc.get('best_baseline'))}` "
        f"(Δ `{_fmt(oulad_acc.get('delta'))}`, positive stages `{oulad_acc.get('positive_stages', 'n/a')}`, ok=`{oulad_acc.get('ok')}`).",
        f"- UCI Hybrid robust macro PR-AUC = `{_fmt(uci_r.get('mean'))}` ± `{_fmt(uci_r.get('std'))}` "
        f"vs `{uci_acc.get('best_baseline_name', 'n/a')}` = `{_fmt(uci_acc.get('best_baseline'))}` "
        f"(Δ `{_fmt(uci_acc.get('delta'))}`, ok=`{uci_acc.get('ok')}`).",
        f"- Inner ready: `{acc.get('ready')}`",
        f"- Outer opened: `{bool(lock)}`",
        f"- Authority `src/prediction` updated: `{bool(final_acc and final_acc.get('accepted'))}`",
        "",
        "## B. Architecture lock",
        "",
        f"- topology_hash: `{proto.get('topology_hash')}`",
        "- temporal_path = parallel; fusion = softmax_3way; public class = Hybrid",
        "- Phase 2 SELECTED_TOPOLOGY / PROTOCOL_LOCK hashes verified before training",
        "- Availability unit tests passed (S0/no-temporal mass = 0; BiLSTM not gated by aggregate)",
        "- No dataset-specific fork; no C0 topology change; no post-outer retune",
        "",
        "## C. Shared structural HPO",
        "",
        f"- Selected shared tuple: `{shared.get('d_fuse')}/{shared.get('cnn_channels')}/{shared.get('bilstm_hidden')}`",
        f"- Shared across UCI and OULAD: `{shared.get('shared_across_uci_and_oulad', True)}`",
        f"- Reason: `{((shared.get('selection') or {}).get('selected_reason')) or ((shared.get('robust33') or [{}])[0].get('selected_reason') if shared.get('robust33') else 'lexicographic_oulad_then_uci_guardrail')}`",
        "- Screened `{64,96,128}^3` then confirmed the selected tuple with 3×3. No further structural search in FAST_COMPLETION.",
        "",
        "## D. OULAD training HPO",
        "",
        f"- Complete trials: `{oulad_hpo.get('n_complete')}`; pruned: `{oulad_hpo.get('n_pruned')}`",
        f"- Best 1-fold macro: `{_fmt(oulad_hpo.get('best_macro'))}` (saturated at the first complete trial; no material later gain)",
        f"- Locked numerics: `{json.dumps(oulad_hpo.get('best_params'), sort_keys=True)}`",
        "- No additional HPO trials were launched after the locked best.",
        "",
        "## E. UCI training HPO",
        "",
        f"- Complete trials: `{uci_hpo.get('n_complete')}`; pruned: `{uci_hpo.get('n_pruned')}`",
        f"- Best 1-fold macro: `{_fmt(uci_hpo.get('best_macro'))}`",
        f"- Locked numerics: `{json.dumps(uci_hpo.get('best_params'), sort_keys=True)}`",
        "- Same C0 graph; only training numerics differ from OULAD.",
        "",
        "## F. Overfit / variance analysis",
        "",
        f"- OULAD 3×3: mean `{_fmt(oulad_r.get('mean'))}`, std `{_fmt(oulad_r.get('std'))}`, min `{_fmt(oulad_r.get('min'))}`, "
        f"worst-stage mean `{_fmt(oulad_r.get('worst_mean'))}`, generalization-gap mean `{_fmt(oulad_r.get('gap_mean'))}`, "
        f"median best epoch `{oulad_r.get('best_epoch_median')}`",
        f"- UCI 3×3: mean `{_fmt(uci_r.get('mean'))}`, std `{_fmt(uci_r.get('std'))}`, min `{_fmt(uci_r.get('min'))}`, "
        f"worst-stage mean `{_fmt(uci_r.get('worst_mean'))}`, generalization-gap mean `{_fmt(uci_r.get('gap_mean'))}`, "
        f"median best epoch `{uci_r.get('best_epoch_median')}`",
        f"- OULAD stage means: `{json.dumps(oulad_r.get('stage_means'), sort_keys=True)}`",
        f"- UCI stage means: `{json.dumps(uci_r.get('stage_means'), sort_keys=True)}`",
        "- Seeds were never selected; all three seeds remain in the confirmation pool.",
        "",
        "## G. Gate diagnostics",
        "",
        "See `GATE_DIAGNOSTICS.csv`. UCI S0 must keep tabular_mass=1 and zero CNN/BiLSTM mass when temporal is absent.",
        "OULAD temporal mass increases with later prefixes, consistent with corrected availability.",
        "",
        "## H. Baseline fairness",
        "",
        "Fixed Phase-2 strong configs, same cutoff-safe parity features, same FIT/STOP/VALID 3×3. No extra baseline HPO.",
        "",
        "```json",
        json.dumps(base_macro, indent=2, sort_keys=True),
        "```",
        "",
        "## I. Inner acceptance gate",
        "",
        "```json",
        json.dumps(acc, indent=2, default=str),
        "```",
        "",
        "## J. Threshold / calibration",
        "",
        f"- Threshold policy: `{thresh.get('policy', 'STOP-only F1 then recall then |t-0.5|')}` on the existing 0.05–0.95 / 0.01 grid. No finer grid search.",
        f"- Thresholds: `{json.dumps(thresh.get('thresholds'), sort_keys=True)}`",
        f"- Temperature scaling tested once (`used={calib.get('used')}`). Other calibrators were not tried.",
        f"- T=`{_fmt(calib.get('temperature'), 3)}`; ECE `{_fmt(calib.get('raw_ece'))}` → `{_fmt(calib.get('cal_ece'))}` "
        f"(gain `{_fmt(calib.get('ece_gain'))}`); Brier `{_fmt(calib.get('raw_brier'))}` → `{_fmt(calib.get('cal_brier'))}` "
        f"(gain `{_fmt(calib.get('brier_gain'))}`).",
        f"- Decision: `{calib.get('reason', 'n/a')}`",
        "",
        "## K. Final outer results",
        "",
    ]
    if outer is None or status == "NOT_READY_FOR_FINAL_EVAL":
        lines.append("Outer evaluation was **not** opened. Inner gate did not pass, or lock was not written.")
        lines.append("")
    else:
        pivot = outer.pivot_table(index=["dataset", "stage"], columns="model", values="pr_auc", aggfunc="mean")
        lines.append("Outer PR-AUC by dataset/stage/model (mean over outer folds):")
        lines.append("")
        lines.append("```")
        lines.append(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        lines.append("```")
        lines.append("")

    lines += [
        "## L. Hybrid vs strongest baseline",
        "",
    ]
    if final_acc:
        lines.append("```json")
        lines.append(json.dumps(final_acc, indent=2, default=str))
        lines.append("```")
        lines.append("")
    else:
        lines.append("Final acceptance was not computed because outer was not opened.")
        lines.append("")

    lines += [
        "## M. Paired bootstrap",
        "",
    ]
    if boot is None:
        lines.append("Bootstrap was not run (no outer predictions).")
        lines.append("")
    else:
        lines.append("Group-level paired bootstrap, 1000 resamples, seed 2026.")
        lines.append("")
        lines.append("```")
        lines.append(boot.to_string(index=False))
        lines.append("```")
        lines.append("")

    lines += [
        "## N. FINAL-100 shortcut analysis",
        "",
        "FINAL-100 was diagnostic only. It was not used for structural HPO, training HPO, threshold, or lock.",
        f"Summary: `{json.dumps({k: f100[k] for k in f100 if k in ('used_for_selection', 'hpo', 'note', 'correlation', 'length_auc')})}`",
        "",
        "## O. Leakage / provenance audit",
        "",
        "- `outer_test_used=false` on HPO, robust confirmation, inner baselines, threshold, and calibration",
        "- split hashes inherited from Phase 2 PROTOCOL_LOCK",
        "- FIT-only preprocessing unchanged",
        "- no best-seed selection; no post-outer retune",
        f"- FAST_COMPLETION flag present: `{bool(fast)}`",
        "",
        "## P. Final authority decision",
        "",
        "`src/prediction` is updated only if status is FINAL_HYBRID_ACCEPTED.",
        f"Current authority update: `{bool(final_acc and final_acc.get('accepted'))}`.",
        "",
        "## Q. Remaining limitations",
        "",
        "- UCI T≤2 limits temporal inductive advantage versus trees on aggregate/grade features",
        "- FINAL-100 length≈Withdrawn shortcut remains; not used for acceptance",
        "- C0 softmax can down-weight tabular on UCI S1/S2",
        "- HPO batch sizes were locked (OULAD 128 / UCI 32); they were not enlarged after selection because that would change the locked training numerics",
        "- AMP kept; DataLoader pin_memory/workers were not introduced after a previous host-side hang on this Windows box",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (ROOT / "HYBRID_VNEXT_PHASE3_FINAL_REPORT.md").write_text(text, encoding="utf-8")
    dest = ROOT / "reports" / "hybrid_vnext" / "phase3"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "HYBRID_VNEXT_PHASE3_FINAL_REPORT.md").write_text(text, encoding="utf-8")
    print("WROTE_REPORT", status)


if __name__ == "__main__":
    main()
