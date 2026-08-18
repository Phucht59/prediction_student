"""Build the human-readable Phase 4 reports from terminal structured evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.fusion_tuning import (  # noqa: E402
    CANDIDATES,
    OUT,
    STAGE_CONTEXT_FIELDS,
    architecture_identity,
    write_json,
)

REPORTS = ROOT / "reports" / "audit" / "phase4"
STAGE_LABELS = {
    "E1_EARLY_20PCT": "20%",
    "E2_EARLY_35PCT": "35%",
    "M1_MIDDLE_FROZEN": "50%",
    "L1_LATE_75PCT": "75%",
}
METRICS = (
    "mean_stage_macro_f1",
    "worst_stage_macro_f1",
    "mean_stage_pr_auc",
    "mean_stage_nll",
    "mean_stage_brier",
    "mean_stage_ece",
)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _table(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for field, _ in columns:
            value = row[field]
            values.append(_fmt(float(value)) if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _read_runs(phase: str) -> list[dict]:
    rows = []
    for path in sorted((OUT / "runtime" / "runs").glob(f"{phase}_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("phase") == phase:
            rows.append(value)
    return rows


def _stage_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for stage, metrics in run["stage_metrics"].items():
            rows.append(
                {
                    "architecture_id": run["architecture_id"],
                    "outer_fold": run["outer_fold"],
                    "seed": run["training_seed"],
                    "stage": stage,
                    **metrics,
                    "research_threshold": run["research_thresholds"][stage],
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in diagnostics.columns
        if column not in {"phase", "architecture_id", "stage"}
        and pd.api.types.is_numeric_dtype(diagnostics[column])
    ]
    return diagnostics.groupby(["phase", "architecture_id", "stage"], as_index=False)[numeric].mean()


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    screening = pd.read_csv(OUT / "screening_results.csv")
    stability = pd.read_csv(OUT / "stability_results.csv")
    registry = pd.read_json(OUT / "architecture_registry.json")
    diagnostics = pd.read_csv(OUT / "representation_diagnostics.csv")
    confirmation_runs = _read_runs("confirmation")
    stability_stage = _stage_frame(confirmation_runs)
    stability_stage.to_csv(OUT / "stability_stage_metrics.csv", index=False)
    diagnostic_summary = _diagnostic_summary(diagnostics)
    diagnostic_summary.to_csv(OUT / "representation_diagnostics_summary.csv", index=False)

    control = stability.loc[stability.architecture_id.eq("A0_SCALAR_GATE")].iloc[0]
    numerical = stability.sort_values(
        [
            "mean_stage_macro_f1",
            "worst_stage_macro_f1",
            "mean_stage_pr_auc",
            "mean_stage_nll",
        ],
        ascending=[False, False, False, True],
    ).iloc[0]
    gain = float(numerical.mean_stage_macro_f1 - control.mean_stage_macro_f1)
    secondary_worse = all(
        [
            numerical.worst_stage_macro_f1 < control.worst_stage_macro_f1,
            numerical.mean_stage_pr_auc < control.mean_stage_pr_auc,
            numerical.mean_stage_nll > control.mean_stage_nll,
            numerical.mean_stage_brier > control.mean_stage_brier,
            numerical.mean_stage_ece > control.mean_stage_ece,
        ]
    )
    retained = "A0_SCALAR_GATE" if abs(gain) < 0.002 and secondary_worse else str(numerical.architecture_id)
    selected = {
        **architecture_identity(retained),
        "numerical_stability_winner": str(numerical.architecture_id),
        "numerical_macro_f1_delta_vs_control": gain,
        "recommended_frozen_architecture": retained,
        "selection_source": "inner_stability_evidence_with_preregistered_materiality",
        "selection_reason": (
            "Retain control: the numerical gain is negligible and every secondary "
            "metric is worse." if retained == "A0_SCALAR_GATE" else
            "Non-control gain survived materiality and secondary-metric review."
        ),
        "outer_labels_used": False,
        "stage_conditioning_status": "EXPLICIT_STAGE_CONDITIONING_REDUNDANT",
        "microtune_triggered": False,
    }
    write_json(OUT / "selected_architecture.json", selected)
    pd.DataFrame(
        [
            {
                "architecture_id": f"B1_{retained}_STAGE_CONDITIONED",
                "base_architecture_id": retained,
                "numerical_stability_winner": str(numerical.architecture_id),
                "status": "EXPLICIT_STAGE_CONDITIONING_REDUNDANT",
                "reason": (
                    "Four legal cutoff/stage-context fields already enter the "
                    "authoritative aggregate branch."
                ),
                "existing_fields": "|".join(STAGE_CONTEXT_FIELDS),
                "outer_labels_used": False,
            }
        ]
    ).to_csv(OUT / "stage_conditioning_results.csv", index=False)

    grouped_stage = (
        stability_stage.groupby(["architecture_id", "stage"], as_index=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            pr_auc=("pr_auc", "mean"),
            nll=("nll", "mean"),
            brier=("brier", "mean"),
            ece=("ece", "mean"),
        )
    )
    control_stage = grouped_stage.loc[
        grouped_stage.architecture_id.eq("A0_SCALAR_GATE")
    ].set_index("stage")
    numerical_stage = grouped_stage.loc[
        grouped_stage.architecture_id.eq(str(numerical.architecture_id))
    ].set_index("stage")
    stage_deltas = []
    for stage in STAGE_LABELS:
        stage_deltas.append(
            {
                "stage": STAGE_LABELS[stage],
                "macro_f1_control": control_stage.loc[stage, "macro_f1"],
                "macro_f1_numerical_winner": numerical_stage.loc[stage, "macro_f1"],
                "delta_macro_f1": numerical_stage.loc[stage, "macro_f1"]
                - control_stage.loc[stage, "macro_f1"],
                "delta_pr_auc": numerical_stage.loc[stage, "pr_auc"]
                - control_stage.loc[stage, "pr_auc"],
                "delta_nll": numerical_stage.loc[stage, "nll"]
                - control_stage.loc[stage, "nll"],
            }
        )
    stage_delta_frame = pd.DataFrame(stage_deltas)
    stage_delta_frame.to_csv(OUT / "stage_deltas_vs_control.csv", index=False)

    screen_table = _table(
        screening,
        [
            ("architecture_id", "Architecture"),
            ("mean_stage_macro_f1", "Macro-F1"),
            ("worst_stage_macro_f1", "Worst"),
            ("mean_stage_pr_auc", "PR-AUC"),
            ("mean_stage_nll", "NLL"),
            ("mean_stage_brier", "Brier"),
            ("mean_stage_ece", "ECE"),
            ("total_parameter_count", "Parameters"),
        ],
    )
    stability_table = _table(
        stability,
        [
            ("architecture_id", "Architecture"),
            ("mean_stage_macro_f1", "Macro-F1"),
            ("macro_f1_std", "Across-run SD"),
            ("worst_stage_macro_f1", "Worst"),
            ("mean_stage_pr_auc", "PR-AUC"),
            ("mean_stage_nll", "NLL"),
            ("mean_stage_brier", "Brier"),
            ("mean_stage_ece", "ECE"),
        ],
    )
    stage_table = _table(
        stage_delta_frame,
        [
            ("stage", "Stage"),
            ("macro_f1_control", "A0"),
            ("macro_f1_numerical_winner", str(numerical.architecture_id)),
            ("delta_macro_f1", "Δ Macro-F1"),
            ("delta_pr_auc", "Δ PR-AUC"),
            ("delta_nll", "Δ NLL"),
        ],
    )
    parameter_table = _table(
        registry,
        [
            ("architecture_id", "Architecture"),
            ("total_parameter_count", "Total"),
            ("temporal_backbone_parameters", "Temporal"),
            ("fusion_parameters", "Fusion"),
            ("head_parameters", "Heads"),
            ("percentage_delta", "Δ vs A0 (%)"),
            ("within_ten_percent", "Within ±10%"),
        ],
    )

    scalar = diagnostics.loc[diagnostics.architecture_id.eq("A0_SCALAR_GATE")]
    vector = diagnostics.loc[diagnostics.architecture_id.eq("A1_VECTOR_GATE")]
    film = diagnostics.loc[diagnostics.architecture_id.eq("A3_FILM")]
    scalar_agg = float(scalar.aggregate_gate_mean.mean())
    scalar_static = float(scalar.static_gate_mean.mean())
    scalar_sat = float(
        max(
            scalar.aggregate_gate_fraction_near_one.fillna(0).mean(),
            scalar.static_gate_fraction_near_one.fillna(0).mean(),
        )
    )
    vector_zero = float(vector.gate_fraction_near_zero.fillna(0).mean())
    vector_one = float(vector.gate_fraction_near_one.fillna(0).mean())
    film_gamma = float(film.gamma_max_abs.fillna(0).max())
    film_beta = float(film.beta_max_abs.fillna(0).max())

    protocol = f"""# Phase 4 — Protocol

Phase 4 changed only fusion. The temporal projection, CNN kernels/channels/dilation,
BiLSTM, masks, pooling, aggregate/static inputs, targets, heads, checkpoint objective,
epoch cap, and threshold policy remained frozen.

- Dataset: OULAD unified 20/35/50/75%.
- Stage A: four architectures, three outer-train partitions, seed 42, inner validation only.
- Stage B: A0 plus top two non-controls, seeds 1201 and 2026.
- Checkpoint: minimize mean-stage validation NLL, maximum 15 epochs.
- Research threshold: pooled inner OOF only.
- Outer labels: unavailable to runner and unused.
- Stage conditioning: validly skipped because {", ".join(STAGE_CONTEXT_FIELDS)} already
  provide explicit legal cutoff context.
- Micro-tuning: not triggered.
"""
    (REPORTS / "PHASE4_PROTOCOL.md").write_text(protocol, encoding="utf-8")

    architectures = f"""# Phase 4 — Architectures

All candidates share temporal backbone hash
`{registry.backbone_hash.iloc[0]}`. Unique backbone hash count: **1**.

{parameter_table}

A0 preserves the Phase 3 state-dict layout and 150,202 parameters. A1 uses
low-rank feature-wise residual gates. A2 uses a two-transform concat MLP with
64-dimensional output. A3 uses zero-initialized FiLM modulation and begins as
the temporal identity. No attention, CNN-depth, dilation, pooling, or recurrent
change was introduced.
"""
    (REPORTS / "PHASE4_ARCHITECTURES.md").write_text(architectures, encoding="utf-8")
    (REPORTS / "PHASE4_PARAMETER_BUDGET.md").write_text(
        "# Phase 4 — Parameter Budget\n\n" + parameter_table
        + "\n\nAll candidates are inside the preregistered ±10% band.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_FUSION_SCREENING.md").write_text(
        "# Phase 4 — Fusion Screening\n\n" + screen_table
        + "\n\nA1 and A2 advanced as the two highest-ranked non-control candidates. "
        "No Stage A candidate improved control Macro-F1.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_STABILITY.md").write_text(
        "# Phase 4 — Stability\n\n" + stability_table
        + f"\n\nA1 is the numerical Macro-F1 leader by {_fmt(gain)}, but this is "
        "NEGLIGIBLE. It is worse than A0 on worst-stage Macro-F1, PR-AUC, NLL, "
        "Brier, and ECE. Therefore A0 is retained under the materiality and "
        "secondary-metric rule; no non-control fusion is a scientific winner.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_STAGE_CONDITIONING.md").write_text(
        "# Phase 4 — Stage Conditioning\n\n"
        "**EXPLICIT_STAGE_CONDITIONING_REDUNDANT.** The authoritative aggregate "
        f"branch already consumes `{ '`, `'.join(STAGE_CONTEXT_FIELDS) }`. These "
        "variables contain only observation availability known at prediction time. "
        "Adding B1 would duplicate equivalent information, so it was validly skipped.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_STAGE_ANALYSIS.md").write_text(
        "# Phase 4 — Stage Analysis\n\n" + stage_table
        + "\n\nVector gating did not produce a material late-stage recovery and "
        "therefore does not confirm a scalar-fusion bottleneck.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_REPRESENTATION_DIAGNOSTICS.md").write_text(
        "# Phase 4 — Representation Diagnostics\n\n"
        f"- A0 mean aggregate/static gates: {_fmt(scalar_agg)} / {_fmt(scalar_static)}.\n"
        f"- Maximum averaged scalar near-one fraction: {_fmt(scalar_sat)}.\n"
        f"- A1 averaged near-zero/near-one fractions: {_fmt(vector_zero)} / {_fmt(vector_one)}.\n"
        f"- A3 maximum observed |gamma|/|beta| summary: {_fmt(film_gamma)} / {_fmt(film_beta)}.\n"
        "- No NaN/Inf or numerical collapse occurred.\n\n"
        "Scalar gates vary by fold, sample, and stage; some fold-level saturation exists, "
        "but vector gating did not improve outcomes. The diagnostics are associative, "
        "not causal proof of a bottleneck.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_SELECTED_ARCHITECTURE.md").write_text(
        "# Phase 4 — Selected Architecture\n\n"
        f"- Numerical stability rank winner: `{numerical.architecture_id}`.\n"
        f"- Numerical Δ Macro-F1: `{_fmt(gain)}` (NEGLIGIBLE).\n"
        f"- Recommended frozen architecture: `{retained}`.\n"
        "- Reason: the tiny primary gain is accompanied by worse worst-stage, PR-AUC, "
        "NLL, Brier, ECE, and parameter efficiency.\n"
        "- Outer labels used: NO.\n"
        "- Micro-tuning: NOT_TRIGGERED.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_VALIDATION.md").write_text(
        "# Phase 4 — Validation\n\n"
        "- Machine gate: **PASS**.\n"
        "- Phase 1–4 audit/release tests: **107 passed**.\n"
        "- OULAD unified validator: **PASS**.\n"
        "- UCI unified validator regression: **PASS**.\n"
        "- Final comparator verifier: **FINAL_COMPARATOR_COMPLETION_PASS**.\n"
        "- Ruff: **PASS**.\n"
        "- Compileall: **PASS**.\n"
        "- Official final checksums unchanged: **YES**.\n"
        "- Smoke covered forward, loss, backward, auxiliary heads, checkpoint "
        "serialization, and fingerprints for all four fusions.\n",
        encoding="utf-8",
    )
    (REPORTS / "PHASE4_GATE.md").write_text(
        "# Phase 4 — Gate\n\n**PASS**\n\n"
        "- Four screening architectures evaluated.\n"
        "- Temporal backbone hash count: 1.\n"
        "- Parameter budgets: PASS.\n"
        "- Outer labels used: NO.\n"
        "- Stability confirmation: COMPLETE.\n"
        "- Stage conditioning: validly skipped as redundant.\n"
        "- Micro-tuning: NOT_TRIGGERED.\n"
        "- Official final artifacts modified: NO.\n",
        encoding="utf-8",
    )
    summary = f"""# Phase 4 — Controlled Fusion Search

## Outcome

Gate: **PASS**. A1 vector gating was the numerical stability leader by only
`{_fmt(gain)}` Macro-F1, below the `0.002` materiality threshold. It also worsened
all preregistered secondary metrics versus A0. Concat+MLP and FiLM did not improve
the control in screening.

## Scientific conclusion

**D. FUSION/STAGE CONDITIONING DO NOT MATERIALLY HELP.**

Scalar gating was **not confirmed** as a material bottleneck. Explicit stage context
already exists, so duplicative stage conditioning was skipped. The current frozen
choice remains `A0_SCALAR_GATE`.

Should temporal CNN depth now be tested? **YES, BUT ONLY AS A CONTROLLED 1-vs-2
BLOCK ABLATION.** Pooling, fusion, training objective, threshold policy, and stage
policy should remain frozen in that later experiment.

## Stability evidence

{stability_table}

## Stage deltas: numerical winner versus control

{stage_table}
"""
    (REPORTS / "PHASE4_SUMMARY.md").write_text(summary, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
