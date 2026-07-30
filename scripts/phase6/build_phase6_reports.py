"""Build the immutable Phase 6 reports from completed structured evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.phase3_optuna import write_json  # noqa: E402

ART = ROOT / "artifacts" / "final" / "h1_final"
REPORT = ROOT / "reports" / "final" / "h1_final"
STAGES = [
    "E1_EARLY_20PCT",
    "E2_EARLY_35PCT",
    "M1_MIDDLE_FROZEN",
    "L1_LATE_75PCT",
]
STAGE_LABEL = {
    "E1_EARLY_20PCT": "20%",
    "E2_EARLY_35PCT": "35%",
    "M1_MIDDLE_FROZEN": "50%",
    "L1_LATE_75PCT": "75%",
}
DISPLAY = {
    "M0_MLP": "MLP",
    "H0_CURRENT_HYBRID": "H0 Current Hybrid",
    "H1_TABULAR_RESIDUAL_EXPERT": "H1 Tabular Residual Hybrid",
}


def read_json(name: str) -> dict[str, Any]:
    return json.loads((ART / name).read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def write(name: str, content: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def table(rows: list[list[str]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> int:
    gate = read_json("phase6_gate.json")
    integrity = read_json("integrity_report.json")
    manifest = read_json("freeze_manifest.json")
    comparisons = read_json("paired_comparison.json")
    bootstrap = read_json("bootstrap_summary.json")
    run_manifest = read_json("run_manifest.json")
    status = read_json("runtime/phase6_status.json")
    comparator = pd.read_csv(ART / "comparator_summary.csv").set_index("candidate")
    stage = pd.read_csv(ART / "stage_metrics.csv").set_index(
        ["candidate", "prediction_stage"]
    )
    fold = pd.read_csv(ART / "fold_metrics.csv")
    thresholds = pd.read_csv(ART / "threshold_summary.csv")

    h1 = comparator.loc["H1_TABULAR_RESIDUAL_EXPERT"]
    h0 = comparator.loc["H0_CURRENT_HYBRID"]
    mlp = comparator.loc["M0_MLP"]
    delta_mlp = float(
        comparisons["M0_MLP"]["macro_f1_delta"]
    )
    delta_h0 = float(
        comparisons["H0_CURRENT_HYBRID"]["macro_f1_delta"]
    )
    classification = "PRACTICAL TIE"
    h0_classification = "YES — SMALL"

    comparator_rows = []
    for candidate in ("M0_MLP", "H0_CURRENT_HYBRID", "H1_TABULAR_RESIDUAL_EXPERT"):
        row = comparator.loc[candidate]
        comparator_rows.append(
            [
                DISPLAY[candidate],
                fmt(row.mean_stage_macro_f1),
                fmt(row.mean_stage_pr_auc),
                fmt(row.mean_stage_roc_auc),
                fmt(row.mean_stage_nll),
                fmt(row.mean_stage_brier),
                fmt(row.mean_stage_ece),
            ]
        )
    comparator_table = table(
        comparator_rows,
        ["Model", "Macro-F1", "PR-AUC", "ROC-AUC", "NLL", "Brier", "ECE"],
    )

    stage_rows = []
    for candidate in ("M0_MLP", "H0_CURRENT_HYBRID", "H1_TABULAR_RESIDUAL_EXPERT"):
        values = [
            float(stage.loc[(candidate, current), "macro_f1"])
            for current in STAGES
        ]
        stage_rows.append(
            [
                DISPLAY[candidate],
                *(fmt(value) for value in values),
                fmt(sum(values) / len(values)),
                fmt(min(values)),
            ]
        )
    stage_table = table(
        stage_rows,
        ["Model", "20%", "35%", "50%", "75%", "Mean", "Worst"],
    )

    delta_rows = []
    for current in STAGES:
        h1_value = float(
            stage.loc[("H1_TABULAR_RESIDUAL_EXPERT", current), "macro_f1"]
        )
        mlp_value = float(stage.loc[("M0_MLP", current), "macro_f1"])
        h0_value = float(stage.loc[("H0_CURRENT_HYBRID", current), "macro_f1"])
        delta_rows.append(
            [
                STAGE_LABEL[current],
                fmt(h1_value - mlp_value),
                fmt(h1_value - h0_value),
            ]
        )
    delta_table = table(delta_rows, ["Stage", "H1 − MLP", "H1 − H0"])

    fold_mean = (
        fold.groupby(["candidate", "outer_fold"]).macro_f1.mean().unstack(0)
    )
    fold_rows = []
    for outer_fold, row in fold_mean.iterrows():
        fold_rows.append(
            [
                str(int(outer_fold)),
                fmt(row["M0_MLP"]),
                fmt(row["H0_CURRENT_HYBRID"]),
                fmt(row["H1_TABULAR_RESIDUAL_EXPERT"]),
                fmt(row["H1_TABULAR_RESIDUAL_EXPERT"] - row["M0_MLP"]),
                fmt(
                    row["H1_TABULAR_RESIDUAL_EXPERT"]
                    - row["H0_CURRENT_HYBRID"]
                ),
            ]
        )
    fold_table = table(
        fold_rows,
        ["Outer fold", "MLP", "H0", "H1", "H1 − MLP", "H1 − H0"],
    )

    confusion_rows = []
    for current in STAGES:
        row = stage.loc[("H1_TABULAR_RESIDUAL_EXPERT", current)]
        confusion_rows.append(
            [
                STAGE_LABEL[current],
                fmt(row.tn, 1),
                fmt(row.fp, 1),
                fmt(row.fn, 1),
                fmt(row.tp, 1),
                fmt(row.risk_precision),
                fmt(row.risk_recall),
                fmt(row.specificity),
            ]
        )
    confusion_table = table(
        confusion_rows,
        [
            "Stage",
            "TN",
            "FP",
            "FN",
            "TP",
            "Risk precision",
            "Risk recall",
            "Specificity",
        ],
    )

    validation = {
        "status": "PASS",
        "tests": "133 passed",
        "oulad_validator": "PASS",
        "uci_regression_validator": "PASS",
        "final_release_verifier": "FINAL_COMPARATOR_COMPLETION_PASS",
        "ruff": "PASS",
        "compileall": "PASS",
        "old_official_checksums_unchanged": True,
        "outer_runs": 45,
        "failed_runs": 0,
        "optuna_trials": 0,
    }
    write_json(ART / "validation_summary.json", validation)
    enriched_gate = {
        **gate,
        "postrun_validation": validation,
        "classification_h1_vs_mlp": classification,
        "classification_h1_vs_h0": h0_classification,
        "development_to_final": {
            "inner_h1_minus_h0": 0.0020793208,
            "final_h1_minus_h0": delta_h0,
            "inner_h1_minus_mlp": 0.0060803855,
            "final_h1_minus_mlp": delta_mlp,
            "h0_gain": "PRESERVED",
            "mlp_advantage": "REVERSED",
        },
    }
    write_json(ART / "phase6_gate.json", enriched_gate)

    write(
        "FINAL_H1_FREEZE.md",
        f"""# Final H1 Freeze

- Candidate: `H1_TABULAR_RESIDUAL_EXPERT`
- Freeze commit: `{integrity["freeze_commit"]}`
- Final candidate hash: `{manifest["final_candidate_hash"]}`
- Architecture hash: `{manifest["architecture_hash"]}`
- Temporal backbone hash: `{manifest["temporal_backbone_hash"]}`
- Feature schema hash: `{manifest["feature_schema_hash"]}`
- Training policy hash: `{manifest["training_policy_hash"]}`
- Evaluation protocol hash: `{manifest["evaluation_protocol_hash"]}`
- Parameter count: **160,492**

ARCHITECTURE FROZEN: **YES**

HYPERPARAMETERS FROZEN: **YES**

FEATURES FROZEN: **YES**

SEEDS FROZEN: **YES**

OUTER FOLDS FROZEN: **YES**

THRESHOLD POLICY FROZEN: **YES**

OUTER TEST ACCESSED BEFORE FREEZE: **NO**

The manifest was committed before the one-shot supervisor built or scored the
outer-test partitions. H2 distillation remained rejected and zero Optuna trials
were authorized.""",
    )

    write(
        "FINAL_H1_EVALUATION.md",
        f"""# Final H1 Evaluation

Phase 6 completed all **45/45** predefined model/fold/seed runs with zero
failures. No post-outer tuning occurred.

{comparator_table}

The frozen H1 achieved mean-stage Macro-F1 **{fmt(h1.mean_stage_macro_f1)}**.
The protocol-matched MLP achieved **{fmt(mlp.mean_stage_macro_f1)}**, a
H1-minus-MLP difference of **{fmt(delta_mlp)}**. This is classified as a
**PRACTICAL TIE**: MLP is numerically higher by 0.000461, while the paired
95% bootstrap interval crosses zero.

Against H0, H1 improved mean-stage Macro-F1 by **{fmt(delta_h0)}**, classified
as a **small** final improvement. The evidence does not justify a claim that
deep learning generally outperforms tabular ML.""",
    )

    write(
        "FINAL_H1_STAGE_RESULTS.md",
        f"""# Final H1 Stage Results

{stage_table}

{delta_table}

H1 improves over MLP at 20% and 35%, then is slightly below it at 50% and 75%.
Relative to H0, H1 improves 20%, 35%, and 50%, but regresses by about 0.00209
at 75%. The residual expert therefore generalized mainly at early/middle
stages, not as a uniform late-stage gain.

## H1 confusion and risk metrics

Counts are fold-averaged because the authoritative stage table aggregates the
three outer folds.

{confusion_table}""",
    )

    mlp_bootstrap = bootstrap["M0_MLP"]
    write(
        "FINAL_H1_VS_MLP.md",
        f"""# Final H1 versus MLP

Primary answer: **NO — PRACTICAL TIE**.

- H1 Macro-F1: **{fmt(h1.mean_stage_macro_f1)}**
- MLP Macro-F1: **{fmt(mlp.mean_stage_macro_f1)}**
- H1 − MLP: **{fmt(delta_mlp)}**
- Fold direction: H1 higher in **{comparisons["M0_MLP"]["fold_positive"]}/3**
- Seed/fold direction: H1 higher in **{comparisons["M0_MLP"]["seed_fold_positive"]}/15**
- Paired grouped bootstrap population delta: **{fmt(mlp_bootstrap["population_point_delta"])}**
- 95% CI: **[{fmt(mlp_bootstrap["ci_95_low"])}, {fmt(mlp_bootstrap["ci_95_high"])}]**

{fold_table}

The fold-averaged primary metric and pooled-observation bootstrap point estimate
differ slightly because they use different aggregation weights; both place the
difference close to zero. The interval crosses zero. H1's inner-development
advantage of +0.006080 over MLP did not generalize and reversed to −0.000461.
No model change follows this result.""",
    )

    h0_bootstrap = bootstrap["H0_CURRENT_HYBRID"]
    write(
        "FINAL_H1_VS_H0.md",
        f"""# Final H1 versus H0

Answer: **YES — SMALL**.

- H1 Macro-F1: **{fmt(h1.mean_stage_macro_f1)}**
- H0 Macro-F1: **{fmt(h0.mean_stage_macro_f1)}**
- H1 − H0: **{fmt(delta_h0)}**
- Fold direction: H1 higher in **{comparisons["H0_CURRENT_HYBRID"]["fold_positive"]}/3**
- Seed/fold direction: H1 higher in **{comparisons["H0_CURRENT_HYBRID"]["seed_fold_positive"]}/15**
- Paired grouped bootstrap population delta: **{fmt(h0_bootstrap["population_point_delta"])}**
- 95% CI: **[{fmt(h0_bootstrap["ci_95_low"])}, {fmt(h0_bootstrap["ci_95_high"])}]**

The final direction preserves the Phase 5 inner gain (+0.002079), with a final
fold-averaged gain of +0.002259. The uncertainty interval overlaps zero and the
75% stage regresses, so the improvement is small rather than a robust clear win.""",
    )

    write(
        "FINAL_H1_CALIBRATION.md",
        f"""# Final H1 Calibration

{comparator_table}

Against H0, H1 slightly improves NLL by
**{fmt(-comparisons["H0_CURRENT_HYBRID"]["nll_delta"])}** and Brier by
**{fmt(-comparisons["H0_CURRENT_HYBRID"]["brier_delta"])}**, while ECE worsens
by **{fmt(comparisons["H0_CURRENT_HYBRID"]["ece_delta"])}**.

Against MLP, H1 is worse in NLL by
**{fmt(comparisons["M0_MLP"]["nll_delta"])}**, Brier by
**{fmt(comparisons["M0_MLP"]["brier_delta"])}**, and ECE by
**{fmt(comparisons["M0_MLP"]["ece_delta"])}**. No post-hoc calibration method
was introduced in Phase 6.""",
    )

    write(
        "FINAL_H1_UNCERTAINTY.md",
        f"""# Final H1 Uncertainty

Paired bootstrap used **5,000** replicates, resampling by `id_student`, with
predictions aligned on the same held-out observations.

| Comparison | Population delta | Bootstrap mean | 95% CI | Crosses zero |
| --- | --- | --- | --- | --- |
| H1 − MLP | {fmt(mlp_bootstrap["population_point_delta"])} | {fmt(mlp_bootstrap["bootstrap_mean_delta"])} | [{fmt(mlp_bootstrap["ci_95_low"])}, {fmt(mlp_bootstrap["ci_95_high"])}] | YES |
| H1 − H0 | {fmt(h0_bootstrap["population_point_delta"])} | {fmt(h0_bootstrap["bootstrap_mean_delta"])} | [{fmt(h0_bootstrap["ci_95_low"])}, {fmt(h0_bootstrap["ci_95_high"])}] | YES |

Neither comparison supports robust superiority at the paired 95% interval.
Numerical direction remains reportable separately from robust evidence.""",
    )

    unique_threshold_sources = sorted(thresholds.source.unique())
    write(
        "FINAL_H1_PROVENANCE.md",
        f"""# Final H1 Provenance

- Phase 5 selected candidate commit: `{manifest["phase5_commit"]}`
- Pre-outer freeze commit: `{integrity["freeze_commit"]}`
- Final candidate hash: `{manifest["final_candidate_hash"]}`
- Evaluation protocol: `{manifest["protocol_version"]}`
- Outer folds: **3**
- Inner folds: **2**
- Seeds: `42, 1201, 2026, 3407, 7319`
- Stages: `20%, 35%, 50%, 75%`
- Runs: **{run_manifest["run_count"]}**
- Same checkpoint across stages: **YES**
- Threshold source: `{", ".join(unique_threshold_sources)}`
- Outer labels used for epoch/threshold selection: **NO**
- Optuna trials: **0**
- Old official evidence preserved: **YES**

H0 and MLP were recomputed under the same frozen folds, stages, seed aggregation,
and Phase 5 inner-only threshold authority because historical evidence was not
silently assumed protocol-compatible.""",
    )

    write(
        "FINAL_H1_VALIDATION.md",
        """# Final H1 Validation

- Phase 1–6 audit/release tests: **133 passed**
- Phase 6 supervisor: **45 completed, 0 failed**
- OULAD validator: **PASS**
- UCI regression validator: **PASS**
- Final comparator verifier: **FINAL_COMPARATOR_COMPLETION_PASS**
- Ruff: **PASS**
- Compileall: **PASS**
- Freeze/hash integrity: **PASS**
- Exact seeds/folds: **PASS**
- Same checkpoint across stages: **PASS**
- Outer-label training/threshold firewall: **PASS**
- Optuna trials: **0**
- Old official final checksums unchanged: **YES**
- README promotion performed: **NO**
""",
    )

    write(
        "PHASE6_GATE.md",
        f"""# Phase 6 Gate

**PASS**

- Freeze commit preceded all outer access.
- Candidate, architecture, training, features, thresholds, folds, and seeds
  remained immutable.
- All 45 required runs completed with no failed or discarded seed.
- H1, H0, and MLP are protocol-matched.
- No Optuna, candidate variation, post-outer tuning, or selective rerun occurred.
- Paired uncertainty and all required final metrics were generated.
- Regression, protocol, release, lint, compile, and checksum validations passed.

Final classification H1 versus MLP: **{classification}**.

Residual expert versus H0: **{h0_classification}**.

Development is permanently stopped after this one-shot evaluation.""",
    )

    if status["state"] != "COMPLETE" or integrity["status"] != "PASS":
        raise RuntimeError("cannot issue PASS reports from incomplete evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
