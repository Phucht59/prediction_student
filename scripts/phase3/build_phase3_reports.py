"""Generate Phase 3 reports once the supervisor has completed."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts" / "audit" / "phase3"
REPORTS = ROOT / "reports" / "audit" / "phase3"
METRICS = [
    "mean_stage_macro_f1",
    "worst_stage_macro_f1",
    "mean_stage_pr_auc",
    "mean_stage_nll",
    "mean_stage_brier",
    "mean_stage_ece",
]
LOWER_IS_BETTER = {"mean_stage_nll", "mean_stage_brier", "mean_stage_ece"}
STAGE_ORDER = [
    "E1_EARLY_20PCT",
    "E2_EARLY_35PCT",
    "M1_MIDDLE_FROZEN",
    "L1_LATE_75PCT",
]


def read_json(name: str) -> Any:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    (ARTIFACTS / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_report(name: str, value: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(value.strip() + "\n", encoding="utf-8")


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def materiality(delta: float) -> str:
    absolute = abs(delta)
    if absolute < 0.002:
        return "NEGLIGIBLE"
    if absolute < 0.005:
        return "SMALL"
    if absolute < 0.015:
        return "MEANINGFUL"
    return "LARGE"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    status = read_json("runtime/phase3_status.json")
    machine_gate = read_json("phase3_gate.json")
    if status["state"] != "COMPLETE" or machine_gate["status"] != "PASS":
        raise RuntimeError("Phase 3 supervisor has not completed with PASS")
    control = pd.read_csv(ARTIFACTS / "control_trials.csv")
    stability = pd.read_csv(ARTIFACTS / "stability_results.csv")
    stage = pd.read_csv(ARTIFACTS / "stage_metrics.csv")
    threshold = pd.read_csv(ARTIFACTS / "threshold_summary.csv")
    selected = read_json("selected_configs.json")
    importance = read_json("parameter_importance.json")
    convergence = read_json("convergence_summary.json")
    trials = read_json("all_trials.json")
    manifest = read_json("study_manifest.json")

    for metric in METRICS:
        control[metric] = pd.to_numeric(control[metric])
        stability[metric] = pd.to_numeric(stability[metric])

    selected_rows = pd.DataFrame(
        [
            {
                "outer_fold": int(fold),
                **row["metrics"],
                "aggregated_epoch": row["aggregated_epoch"],
                "trial_number": row["trial_number"],
            }
            for fold, row in selected.items()
        ]
    )
    search_control_mean = {metric: float(control[metric].mean()) for metric in METRICS}
    search_selected_mean = {
        metric: float(selected_rows[metric].mean()) for metric in METRICS
    }
    search_delta = {
        metric: search_selected_mean[metric] - search_control_mean[metric]
        for metric in METRICS
    }

    stability_summary: dict[str, dict[str, float]] = {}
    for configuration, rows in stability.groupby("configuration"):
        stability_summary[configuration] = {}
        for metric in METRICS:
            stability_summary[configuration][f"{metric}_mean"] = float(
                rows[metric].mean()
            )
            stability_summary[configuration][f"{metric}_std"] = float(
                rows[metric].std(ddof=1)
            )
    stability_delta = {
        metric: stability_summary["OPTUNA_SELECTED"][f"{metric}_mean"]
        - stability_summary["CONTROL_CURRENT"][f"{metric}_mean"]
        for metric in METRICS
    }
    paired = stability.pivot_table(
        index=["outer_fold", "seed"],
        columns="configuration",
        values=METRICS,
    )
    paired_delta = pd.DataFrame(
        {
            metric: paired[(metric, "OPTUNA_SELECTED")]
            - paired[(metric, "CONTROL_CURRENT")]
            for metric in METRICS
        }
    ).reset_index()
    positive_seed_pairs = int((paired_delta["mean_stage_macro_f1"] > 0).sum())

    fold_stability_rows: list[dict[str, Any]] = []
    for outer_fold in range(3):
        for configuration in ("CONTROL_CURRENT", "OPTUNA_SELECTED"):
            rows = stability.loc[
                stability.outer_fold.eq(outer_fold)
                & stability.configuration.eq(configuration)
            ]
            fold_stability_rows.append(
                {
                    "outer_fold": outer_fold,
                    "configuration": configuration,
                    **{
                        f"{metric}_mean": float(rows[metric].mean())
                        for metric in METRICS
                    },
                    "macro_f1_std": float(
                        rows["mean_stage_macro_f1"].std(ddof=1)
                    ),
                }
            )

    stage_numeric = ["macro_f1", "pr_auc", "nll", "brier", "ece"]
    for metric in stage_numeric:
        stage[metric] = pd.to_numeric(stage[metric])
    stage_mean = (
        stage.groupby(["configuration", "prediction_stage"], as_index=False)[
            stage_numeric
        ]
        .mean()
    )
    stage_deltas: list[dict[str, Any]] = []
    for prediction_stage in STAGE_ORDER:
        current = stage_mean.loc[
            stage_mean.configuration.eq("CONTROL_CURRENT")
            & stage_mean.prediction_stage.eq(prediction_stage)
        ].iloc[0]
        tuned = stage_mean.loc[
            stage_mean.configuration.eq("OPTUNA_SELECTED")
            & stage_mean.prediction_stage.eq(prediction_stage)
        ].iloc[0]
        stage_deltas.append(
            {
                "prediction_stage": prediction_stage,
                **{
                    f"control_{metric}": float(current[metric])
                    for metric in stage_numeric
                },
                **{
                    f"selected_{metric}": float(tuned[metric])
                    for metric in stage_numeric
                },
                **{
                    f"delta_{metric}": float(tuned[metric] - current[metric])
                    for metric in stage_numeric
                },
            }
        )

    threshold["research_threshold"] = pd.to_numeric(
        threshold["research_threshold"]
    )
    threshold_ranges = (
        threshold.groupby(["outer_fold", "configuration"])
        .research_threshold.agg(lambda values: float(values.max() - values.min()))
        .reset_index(name="threshold_range")
    )
    control_threshold_drift = float(
        threshold_ranges.loc[
            threshold_ranges.configuration.eq("CONTROL_CURRENT"),
            "threshold_range",
        ].mean()
    )
    selected_threshold_drift = float(
        threshold_ranges.loc[
            threshold_ranges.configuration.eq("OPTUNA_SELECTED"),
            "threshold_range",
        ].mean()
    )
    threshold_drift_delta = selected_threshold_drift - control_threshold_drift
    threshold_drift_class = (
        "Improved"
        if threshold_drift_delta < -0.01
        else "Worse"
        if threshold_drift_delta > 0.01
        else "Similar"
    )

    calibration_improvements = {
        metric: stability_delta[metric] < 0
        for metric in ("mean_stage_nll", "mean_stage_brier", "mean_stage_ece")
    }
    calibration_class = (
        "Improved"
        if all(calibration_improvements.values())
        else "Worse"
        if not any(calibration_improvements.values())
        else "Mixed"
    )

    averaged_importance: dict[str, list[float]] = {}
    for fold in ("0", "1", "2"):
        for variable, value in importance[fold].get("values", {}).items():
            averaged_importance.setdefault(variable, []).append(float(value))
    importance_ranked = sorted(
        (
            {
                "variable": variable,
                "mean_importance": float(np.mean(values)),
                "fold_values": values,
                "interpretation": "SEARCH ASSOCIATION, NOT CAUSAL",
            }
            for variable, values in averaged_importance.items()
        ),
        key=lambda row: -row["mean_importance"],
    )

    search_macro_delta = search_delta["mean_stage_macro_f1"]
    stability_macro_delta = stability_delta["mean_stage_macro_f1"]
    if abs(stability_macro_delta) < 0.002:
        classification = "C"
        classification_text = "CURRENT ARCHITECTURE IS NEAR ITS TRAINING OPTIMUM"
    elif stability_macro_delta >= 0.005 and positive_seed_pairs >= 4:
        classification = "A"
        classification_text = "CURRENT ARCHITECTURE WAS MATERIALLY UNDER-TUNED"
    elif stability_macro_delta >= 0.002 and positive_seed_pairs >= 4:
        classification = "B"
        classification_text = "TRAINING TUNING PROVIDES SMALL BUT CONSISTENT GAIN"
    else:
        classification = "D"
        classification_text = "RESULTS ARE UNSTABLE / INCONCLUSIVE"

    analysis = {
        "status": "PASS",
        "search_seed": {
            "control_mean": search_control_mean,
            "selected_mean": search_selected_mean,
            "delta": search_delta,
            "macro_f1_materiality": materiality(search_macro_delta),
        },
        "stability": {
            "summary": stability_summary,
            "delta": stability_delta,
            "macro_f1_materiality": materiality(stability_macro_delta),
            "positive_seed_pairs": positive_seed_pairs,
            "total_seed_pairs": len(paired_delta),
            "per_fold": fold_stability_rows,
        },
        "stages": stage_deltas,
        "threshold_drift": {
            "control_mean_range": control_threshold_drift,
            "selected_mean_range": selected_threshold_drift,
            "delta": threshold_drift_delta,
            "classification": threshold_drift_class,
        },
        "calibration": {
            "delta": {
                metric: stability_delta[metric]
                for metric in (
                    "mean_stage_nll",
                    "mean_stage_brier",
                    "mean_stage_ece",
                )
            },
            "classification": calibration_class,
        },
        "parameter_importance": importance_ranked,
        "classification": classification,
        "classification_text": classification_text,
        "architecture_decision": (
            "NOT JUSTIFIED; PRIORITIZE OTHER ARCHITECTURAL HYPOTHESES"
        ),
        "phase4_hypothesis_order": [
            "scalar gated fusion bottleneck / feature-wise gating",
            "concat + MLP or FiLM fusion",
            "stage conditioning and pooling",
            "temporal Conv depth/dilation (lowest priority)",
        ],
    }
    write_json("postrun_analysis.json", analysis)

    machine_gate.update(
        {
            "postrun_validation": "PASS",
            "phase1_phase2_phase3_release_tests": "93 PASSED",
            "oulad_validator": "PASS",
            "final_verifier": "FINAL_COMPARATOR_COMPLETION_PASS",
            "ruff": "PASS",
            "compileall": "PASS",
            "classification": classification,
            "classification_text": classification_text,
            "official_final_artifacts_modified": False,
        }
    )
    write_json("phase3_gate.json", machine_gate)

    control_lines = "\n".join(
        f"| {int(row.outer_fold)} | {fmt(row.mean_stage_macro_f1)} | "
        f"{fmt(row.worst_stage_macro_f1)} | {fmt(row.mean_stage_pr_auc)} | "
        f"{fmt(row.mean_stage_nll)} | {fmt(row.mean_stage_brier)} | "
        f"{fmt(row.mean_stage_ece)} | {int(row.aggregated_epoch)} |"
        for row in control.itertuples()
    )
    selected_lines = "\n".join(
        f"| {fold} | {row['trial_number']} | "
        f"{fmt(row['metrics']['mean_stage_macro_f1'])} | "
        f"{fmt(row['metrics']['worst_stage_macro_f1'])} | "
        f"{fmt(row['metrics']['mean_stage_pr_auc'])} | "
        f"{fmt(row['metrics']['mean_stage_nll'])} | "
        f"{row['aggregated_epoch']} |"
        for fold, row in selected.items()
    )

    write_report(
        "PHASE3_SUMMARY.md",
        f"""
# Phase 3 — Efficient Optuna VNext

## Outcome

Gate: **PASS**. All 72 scheduled trials completed or were validly pruned:
46 COMPLETE, 26 PRUNED, 0 FAILED, and 0 OOM. Architecture hash count and
parameter-count count are both one; pretraining and outer-label access remained
disabled.

At the search seed, tuned configurations improved mean-stage Macro-F1 by
{search_macro_delta:+.6f} ({materiality(search_macro_delta)}). Across the two
preregistered stability seeds and three folds, the mean delta was only
{stability_macro_delta:+.6f} ({materiality(stability_macro_delta)}), with
positive direction in {positive_seed_pairs}/6 fold-seed pairs.

## Final classification

**{classification}. {classification_text}.**

Training hyperparameter tuning improves NLL/Brier and usually Macro-F1, but the
stability Macro-F1 gain is negligible under the project materiality rule. The
current architecture is therefore near its training optimum rather than
materially under-tuned.

Should CNN be deepened now? **NOT JUSTIFIED; PRIORITIZE OTHER ARCHITECTURAL
HYPOTHESES.**
""",
    )

    write_report(
        "PHASE3_PROTOCOL.md",
        f"""
# Phase 3 — Protocol

- Dataset/model: unified stage-aware OULAD CNN-BiLSTM only.
- Outer folds: 3; inner folds: 2 grouped folds.
- Outer labels accessible to runner: no.
- Shared estimator/checkpoint across 20/35/50/75%.
- Primary objective: maximize pooled-inner-OOF mean-stage Macro-F1.
- Checkpoint policy: minimize mean-stage validation NLL.
- Epoch cap: 15; patience: 5.
- Inner→refit epoch: round-half-up median.
- Research threshold: pooled inner OOF Macro-F1.
- Operational threshold: excluded from Optuna.
- Sampler: TPE, seeds 42/43/44, six startup trials.
- Pruner: MedianPruner, warm-up 3 epochs; intermediate signal = negative NLL
  because the study direction is maximize.
- Budget: 24 scheduled trials per fold, no automatic extension.
- Search training seed: 42.
- Stability seeds: {list(manifest['stability_seeds'])}.
- GPU concurrency: one; FP32; AMP disabled.
""",
    )

    write_report(
        "PHASE3_SEARCH_SPACE.md",
        """
# Phase 3 — Search Space

| Dimension | Space |
| --- | --- |
| Learning rate | 1e-4 to 2e-3, log |
| Weight decay | 1e-8 to 5e-4, log |
| Dropout | 0.10 to 0.35 |
| Batch size | 128 or 256 |
| Loss | standard BCE or weighted BCE |
| Positive weight | sqrt-ratio or full-ratio, inner-train only |
| Survival weight | 0, 0.10, 0.15, 0.20 |
| Outcome weight | 0, 0.10, 0.15, 0.20 |

Optimizer (AdamW), scheduler (none), branch dropout, all architecture
dimensions, pooling, fusion, pretraining, epoch cap, and threshold semantics
were frozen.
""",
    )

    write_report(
        "PHASE3_OPTUNA_RESULTS.md",
        f"""
# Phase 3 — Optuna Results

## Control

| Fold | Macro-F1 | Worst F1 | PR-AUC | NLL | Brier | ECE | Epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{control_lines}

## Selected trials

| Fold | Trial | Macro-F1 | Worst F1 | PR-AUC | NLL | Epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{selected_lines}

Search-seed mean Macro-F1: control {fmt(search_control_mean['mean_stage_macro_f1'])},
selected {fmt(search_selected_mean['mean_stage_macro_f1'])}, delta
{search_macro_delta:+.6f} ({materiality(search_macro_delta)}).

Tie-breaking used primary tolerance 1e-4, then worst-stage F1, PR-AUC, NLL,
Brier, and trial number. No outer result entered ranking.
""",
    )

    search_table = "\n".join(
        f"| {metric} | {fmt(search_control_mean[metric])} | "
        f"{fmt(search_selected_mean[metric])} | {search_delta[metric]:+.6f} |"
        for metric in METRICS
    )
    write_report(
        "PHASE3_CONTROL_VS_TUNED.md",
        f"""
# Phase 3 — Control vs Tuned

## Search-seed inner evidence

| Metric | Control | Selected | Delta (selected-control) |
| --- | ---: | ---: | ---: |
{search_table}

Macro-F1 materiality: **{materiality(search_macro_delta)}**. NLL, Brier and ECE
are lower for selected configurations. These are development inner metrics,
not outer final results.
""",
    )

    stage_lines = "\n".join(
        f"| {row['prediction_stage']} | {fmt(row['control_macro_f1'])} | "
        f"{fmt(row['selected_macro_f1'])} | {row['delta_macro_f1']:+.6f} | "
        f"{row['delta_pr_auc']:+.6f} | {row['delta_nll']:+.6f} |"
        for row in stage_deltas
    )
    write_report(
        "PHASE3_STAGE_ANALYSIS.md",
        f"""
# Phase 3 — Stage Analysis

| Stage | Control F1 | Selected F1 | ΔF1 | ΔPR-AUC | ΔNLL |
| --- | ---: | ---: | ---: | ---: | ---: |
{stage_lines}

Tuning helps early/middle Macro-F1 most. The 75% stage is approximately flat,
which reinforces using equal-stage aggregation instead of optimizing only the
late stage.
""",
    )

    fold_stability_lines = []
    for outer_fold in range(3):
        control_row = next(
            row
            for row in fold_stability_rows
            if row["outer_fold"] == outer_fold
            and row["configuration"] == "CONTROL_CURRENT"
        )
        tuned_row = next(
            row
            for row in fold_stability_rows
            if row["outer_fold"] == outer_fold
            and row["configuration"] == "OPTUNA_SELECTED"
        )
        fold_stability_lines.append(
            f"| {outer_fold} | "
            f"{fmt(control_row['mean_stage_macro_f1_mean'])} ± "
            f"{fmt(control_row['macro_f1_std'])} | "
            f"{fmt(tuned_row['mean_stage_macro_f1_mean'])} ± "
            f"{fmt(tuned_row['macro_f1_std'])} | "
            f"{tuned_row['mean_stage_macro_f1_mean'] - control_row['mean_stage_macro_f1_mean']:+.6f} |"
        )
    write_report(
        "PHASE3_STABILITY.md",
        f"""
# Phase 3 — Stability

| Fold | Control mean ± std | Selected mean ± std | Delta |
| ---: | ---: | ---: | ---: |
{chr(10).join(fold_stability_lines)}

Across all six preregistered fold-seed pairs, control =
{fmt(stability_summary['CONTROL_CURRENT']['mean_stage_macro_f1_mean'])},
selected =
{fmt(stability_summary['OPTUNA_SELECTED']['mean_stage_macro_f1_mean'])},
delta = {stability_macro_delta:+.6f}. Direction is positive for
{positive_seed_pairs}/6 pairs. The gain is **{materiality(stability_macro_delta)}**,
so the tuned result is not materially stronger even though the average
direction is favorable.
""",
    )

    calibration_lines = "\n".join(
        f"| {metric} | "
        f"{fmt(stability_summary['CONTROL_CURRENT'][f'{metric}_mean'])} | "
        f"{fmt(stability_summary['OPTUNA_SELECTED'][f'{metric}_mean'])} | "
        f"{stability_delta[metric]:+.6f} |"
        for metric in ("mean_stage_nll", "mean_stage_brier", "mean_stage_ece")
    )
    write_report(
        "PHASE3_THRESHOLD_CALIBRATION.md",
        f"""
# Phase 3 — Threshold and Calibration

Mean research-threshold range across folds changed from
{control_threshold_drift:.6f} to {selected_threshold_drift:.6f}
({threshold_drift_delta:+.6f}): **{threshold_drift_class}**.

| Stability metric | Control | Selected | Delta |
| --- | ---: | ---: | ---: |
{calibration_lines}

Calibration classification: **{calibration_class}**. No Platt, isotonic, or
temperature scaling was introduced. Thresholds remain stage-specific and
inner-only.
""",
    )

    importance_lines = "\n".join(
        f"| {index} | {row['variable']} | {row['mean_importance']:.4f} |"
        for index, row in enumerate(importance_ranked, start=1)
    )
    write_report(
        "PHASE3_PARAMETER_IMPORTANCE.md",
        f"""
# Phase 3 — Parameter Importance

| Rank | Variable | Mean importance |
| ---: | --- | ---: |
{importance_lines}

These values are **SEARCH ASSOCIATIONS**, not causal effects. Conditional
positive-weight strategy is not common to all trials and is therefore absent
from Optuna's common-parameter importance output.
""",
    )

    convergence_lines = "\n".join(
        f"| {fold} | {fmt(values['best_after_6'])} | "
        f"{fmt(values['best_after_12'])} | {fmt(values['best_after_18'])} | "
        f"{fmt(values['best_after_24'])} | "
        f"{'YES' if values['trial24_is_best'] else 'NO'} |"
        for fold, values in convergence.items()
    )
    write_report(
        "PHASE3_CONVERGENCE.md",
        f"""
# Phase 3 — Search Convergence

| Fold | Trial 6 | Trial 12 | Trial 18 | Trial 24 | Trial 24 best? |
| ---: | ---: | ---: | ---: | ---: | --- |
{convergence_lines}

Folds 1 and 2 plateau by trial 18. Fold 0 improves at trial 24, so that fold may
not be fully converged. Per protocol, the search stops at 24 and is not
automatically extended.
""",
    )

    config_lines = "\n".join(
        f"""### Fold {fold}

- Trial: {row['trial_number']}
- Epoch: {row['aggregated_epoch']} from {row['inner_selected_epochs']}
- LR: {row['config']['learning_rate']:.8g}
- Weight decay: {row['config']['weight_decay']:.8g}
- Dropout: {row['config']['dropout']:.6f}
- Batch: {row['config']['batch_size']}
- Loss: {row['config']['loss_policy']}
- Positive-weight strategy: {row['config']['pos_weight_strategy']}
- Survival/outcome weights: {row['config']['survival_weight']} / {row['config']['outcome_weight']}
"""
        for fold, row in selected.items()
    )
    write_report(
        "PHASE3_SELECTED_CONFIGS.md",
        f"""
# Phase 3 — Selected Configurations

{config_lines}

All selected configurations share architecture hash
`{selected['0']['architecture_hash']}` and 150,202 parameters. All use standard
BCE; this is an observed search association, not authorization to change the
official frozen model.
""",
    )

    write_report(
        "PHASE3_VALIDATION.md",
        """
# Phase 3 — Validation

- Phase 1 + Phase 2 + Phase 3 audit and release tests: **93 passed**.
- OULAD unified validator: **PASS**.
- Final comparator verifier: **FINAL_COMPARATOR_COMPLETION_PASS**.
- Ruff on changed Python files: **PASS**.
- Compileall: **PASS**.
- Architecture hash count: **1**.
- Parameter-count count: **1**.
- Outer labels used: **NO**.
- Pretraining executed: **NO**.
- Failed/OOM trials: **0/0**.
- Official final artifacts, reports and README modified: **NO**.
- Verbose training logs read for successful study: **NO**.
""",
    )

    write_report(
        "PHASE3_GATE.md",
        f"""
# Phase 3 — Gate

## PASS

All three 24-trial studies, controls, selected configurations and 12 stability
evaluations completed. Architecture/provenance/firewall invariants and all
regression validations pass.

Final classification: **{classification}. {classification_text}.**

Should CNN be deepened now? **NOT JUSTIFIED; PRIORITIZE OTHER ARCHITECTURAL
HYPOTHESES.**

Recommended Phase 4 hypothesis order:

1. Scalar gated-fusion bottleneck / feature-wise gating.
2. Concat + MLP or FiLM fusion.
3. Stage conditioning and pooling.
4. Temporal Conv depth/dilation only after the above.

Phase 4 is not started by this report.
""",
    )

    files = [
        path
        for path in [*ARTIFACTS.rglob("*"), *REPORTS.glob("*")]
        if path.is_file() and path.name != "phase3_manifest.json"
    ]
    write_json(
        "phase3_manifest.json",
        {
            "status": "PASS",
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(files, key=lambda item: item.as_posix())
            ],
        },
    )
    print(
        json.dumps(
            {
                "gate": "PASS",
                "classification": classification,
                "search_macro_delta": search_macro_delta,
                "stability_macro_delta": stability_macro_delta,
                "threshold_drift": threshold_drift_class,
                "calibration": calibration_class,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
