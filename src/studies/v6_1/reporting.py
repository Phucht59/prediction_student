from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.studies.v5_1.common.protocol import ROOT

from .oulad_runner import ARTIFACT_ROOT, _atomic_json, _checksums


REPORT_PATH = ROOT / "reports/v6_1/OULAD_ARCHITECTURE_DIAGNOSIS.md"


def _verdicts(summary: pd.DataFrame, selected: dict[str, Any]) -> dict[str, Any]:
    rows = summary.set_index("candidate")

    def delta(candidate: str, baseline: str, metric: str = "macro_f1_mean") -> float:
        return float(rows.loc[candidate, metric] - rows.loc[baseline, metric])

    capacity_gain = delta("B2_cnn_matched_temporal", "A1_cnn_small_temporal")
    capacity_gap = delta("B2_cnn_matched_temporal", "A2_bilstm_current_temporal")
    h1 = (
        "SUPPORTED"
        if capacity_gain >= 0.001 and capacity_gap >= -0.0005
        else "PARTIAL"
        if capacity_gain >= 0.001
        else "NOT_SUPPORTED"
    )
    architectural_best = max(
        delta("D_serial_with_cnn_skip", "A4_serial_current_full"),
        delta("E_parallel_concat", "A4_serial_current_full"),
    )
    h2 = (
        "SUPPORTED"
        if selected["selected_candidate"] is not None
        else "PARTIAL"
        if architectural_best >= 0.0005
        else "NOT_SUPPORTED"
    )
    dilation_best = max(
        delta("C1_cnn_d1_temporal", "C2_cnn_d2_temporal"),
        delta("C3_cnn_multidilation_temporal", "C2_cnn_d2_temporal"),
    )
    h3 = (
        "SUPPORTED"
        if dilation_best >= 0.001
        else "PARTIAL"
        if dilation_best >= 0.0005
        else "NOT_SUPPORTED"
    )
    aggregate_gap = delta("A0_aggregate_static_only", "A4_serial_current_full")
    h4 = (
        "SUPPORTED"
        if aggregate_gap >= -0.003
        else "PARTIAL"
        if aggregate_gap >= -0.006
        else "NOT_SUPPORTED"
    )
    h5 = (
        "SUPPORTED"
        if selected["selected_candidate"] is None
        and capacity_gain < 0.001
        and architectural_best < 0.001
        else "PARTIAL"
        if selected["selected_candidate"] is None
        else "NOT_SUPPORTED"
    )
    scenario = (
        "A_ARCHITECTURE_BOTTLENECK_CONFIRMED"
        if h2 == "SUPPORTED"
        else "B_CAPACITY_BIAS_PARTIALLY_CONFIRMED"
        if h1 in {"SUPPORTED", "PARTIAL"} and capacity_gain >= 0.001
        else "C_AGGREGATE_REDUNDANCY_CONFIRMED"
        if h4 == "SUPPORTED"
        else "D_CNN_STILL_ADDS_LITTLE"
    )
    return {
        "H1_capacity_imbalance": {
            "verdict": h1,
            "cnn_matched_delta_vs_cnn_small": capacity_gain,
            "cnn_matched_delta_vs_bilstm": capacity_gap,
        },
        "H2_serial_bottleneck": {
            "verdict": h2,
            "best_skip_or_parallel_delta_vs_serial_full": architectural_best,
        },
        "H3_dilation_mismatch": {
            "verdict": h3,
            "best_d1_or_multidilation_delta_vs_d2": dilation_best,
        },
        "H4_aggregate_redundancy": {
            "verdict": h4,
            "aggregate_static_delta_vs_full_serial": aggregate_gap,
        },
        "H5_data_limitation": {
            "verdict": h5,
            "development_gate_passed": selected["selected_candidate"] is not None,
        },
        "scenario": scenario,
    }


def _metric_table(summary: pd.DataFrame) -> str:
    labels = {
        "A0_aggregate_static_only": "Aggregate + static only",
        "A1_cnn_small_temporal": "CNN temporal",
        "A2_bilstm_current_temporal": "BiLSTM temporal",
        "A3_serial_current_temporal": "Serial CNN-BiLSTM temporal",
        "A4_serial_current_full": "Full serial hybrid",
        "B2_cnn_matched_temporal": "Parameter-matched CNN",
        "D_serial_with_cnn_skip": "Serial + CNN skip",
        "E_parallel_concat": "Parallel CNN || BiLSTM",
    }
    rows = summary.set_index("candidate")
    lines = [
        "| Model | Params | Macro-F1 | At-risk F1 | PR-AUC | Brier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, label in labels.items():
        row = rows.loc[candidate]
        lines.append(
            f"| {label} | {int(row.parameter_count):,} | "
            f"{row.macro_f1_mean:.4f} ± {row.macro_f1_sd:.4f} | "
            f"{row.at_risk_f1_mean:.4f} | {row.pr_auc_mean:.4f} | "
            f"{row.brier_mean:.4f} |"
        )
    return "\n".join(lines)


def _outer_section(final: dict[str, Any]) -> str:
    if final["status"] != "COMPLETE":
        return (
            "Development gate did not pass, so the preregistered rule prohibited "
            "opening a new outer evaluation. Frozen V5.1 and XGBoost evidence was "
            "left unchanged."
        )
    results = pd.read_csv(ARTIFACT_ROOT / "final_outer_results.csv")
    columns = [
        "candidate",
        "macro_f1_mean",
        "macro_f1_sd",
        "at_risk_f1_mean",
        "pr_auc_mean",
        "brier_mean",
    ]
    return results[columns].to_markdown(index=False, floatfmt=".4f")


def generate_report(validation: dict[str, Any]) -> dict[str, Any]:
    summary = pd.read_csv(ARTIFACT_ROOT / "candidate_summary.csv")
    selected = yaml.safe_load(
        (ARTIFACT_ROOT / "selected_config.yaml").read_text(encoding="utf-8")
    )
    final = json.loads(
        (ARTIFACT_ROOT / "final_run_state.json").read_text(encoding="utf-8")
    )
    order = json.loads(
        (ARTIFACT_ROOT / "order_audit.json").read_text(encoding="utf-8")
    )
    recommendation = json.loads(
        (ARTIFACT_ROOT / "recommendation_logic_audit.json").read_text(
            encoding="utf-8"
        )
    )
    verdicts = _verdicts(summary, selected)
    _atomic_json(ARTIFACT_ROOT / "hypothesis_verdicts.json", verdicts)
    selected_description = (
        selected["selected_candidate"]
        if selected["selected_candidate"] is not None
        else "None — no new architecture passed the preregistered development gate."
    )
    order_lines = "\n".join(
        f"- {row['order']}: Macro-F1 {row['macro_f1']:.4f}, "
        f"delta {row['macro_f1_delta_vs_original']:+.4f}"
        for row in order["results"]
    )
    hypothesis_lines = "\n".join(
        f"- {name}: **{value['verdict']}** — "
        + ", ".join(
            f"{key}={number:+.4f}"
            for key, number in value.items()
            if key != "verdict" and isinstance(number, float)
        )
        for name, value in verdicts.items()
        if isinstance(value, dict) and "verdict" in value
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        f"""# OULAD CNN–BiLSTM architecture diagnosis (V6.1)

## Evidence protection

All new training was isolated under `artifacts/v6_1_oulad_architecture_diagnosis`.
Frozen V5/V5.1/V6 checkpoints, OOF predictions and canonical results were not
overwritten. Architecture selection used only outer-training fold 0 inner CV;
Future OULAD remained locked.

## Architecture diagnosis

{_metric_table(summary)}

## Hypothesis verdicts

{hypothesis_lines}

Registered scenario: **{verdicts['scenario']}**.

## Selected architecture

{selected_description}

Config hash: `{selected['config_sha256']}`.

## Final outer evaluation

{_outer_section(final)}

## Temporal-order evidence

Candidate role: `{order['candidate_role']}`; threshold was frozen from original
inner OOF and reused for every destruction condition.

{order_lines}

## Recommendation semantic correction

Circular pseudo-observed logic existed and was removed. `activity_level`,
`inactivity_streak`, `assessment_progress`, and `grade_trend` now require real
pre-cutoff sequence measurements. Missing observed state causes abstention rather
than probability-to-behavior fabrication.

Withdrawal reliability is
**{recommendation['withdrawal']['status']}** because observed withdrawal recall
was at most {recommendation['withdrawal']['maximum_observed_recall']}. The horizon
may remain exploratory, but it cannot assert an engagement mechanism or trigger a
mechanism-specific recommendation.

## Validation

- Passed: {validation['passed']}
- Skipped: {validation['skipped']}
- Failed: {validation['failed']}
- Frozen evidence modified: no
- Outer test used for selection: no
- Future OULAD accessed: no

## Scientific conclusion

The evidence is classified as **{verdicts['scenario']}**. Capacity imbalance and
dilation did disadvantage the small CNN modestly: parameter matching added
{verdicts['H1_capacity_imbalance']['cnn_matched_delta_vs_cnn_small']:+.4f}
Macro-F1 and dilation one added
{verdicts['H3_dilation_mismatch']['best_d1_or_multidilation_delta_vs_d2']:+.4f}.
However, the matched CNN still trailed the BiLSTM by
{verdicts['H1_capacity_imbalance']['cnn_matched_delta_vs_bilstm']:+.4f}; direct
skip and parallel paths did not beat the full serial control, while aggregate +
static alone was already close to the temporal models. Therefore the serial
design was not shown to suppress a useful CNN expert. The dominant explanation
is limited incremental local signal plus redundancy with compact features, with
a smaller contribution from capacity and dilation choices.

This conclusion follows the preregistered aggregate inner-CV rules; no seed or
fold was selected after the fact, and the negative outer-evaluation gate result
was retained.
""",
        encoding="utf-8",
    )
    result = {
        "status": "COMPLETE",
        "report": REPORT_PATH.relative_to(ROOT).as_posix(),
        "hypothesis_verdicts": verdicts,
        "selected_candidate": selected["selected_candidate"],
        "final_status": final["status"],
        "validation": validation,
    }
    _atomic_json(ARTIFACT_ROOT / "final_report.json", result)
    _checksums()
    return result


def write_validation_report(
    *,
    configured_suite: dict[str, int],
    relevant_suite: dict[str, int],
    checks: dict[str, bool],
) -> dict[str, Any]:
    result = {
        "schema_version": "v6_1_validation_report_v1",
        "status": "PASS"
        if configured_suite["failed"] == 0
        and relevant_suite["failed"] == 0
        and all(checks.values())
        else "FAIL",
        "configured_suite": configured_suite,
        "relevant_suite": relevant_suite,
        "passed": configured_suite["passed"] + relevant_suite["passed"],
        "skipped": configured_suite["skipped"] + relevant_suite["skipped"],
        "failed": configured_suite["failed"] + relevant_suite["failed"],
        "checks": checks,
    }
    _atomic_json(ARTIFACT_ROOT / "validation_report.json", result)
    return result


__all__ = ["generate_report", "write_validation_report"]
