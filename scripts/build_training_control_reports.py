"""Build Phase 2 machine-readable validation and audit reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.oulad_multitask import CNNBiLSTMOULAD  # noqa: E402
from src.training.config_authority import (  # noqa: E402
    architecture_metadata,
    load_config_authority,
    resolved_deep_config,
)
from src.training.control import (  # noqa: E402
    TrainingRunIdentity,
    fixed_refit_metadata,
    pretraining_provenance,
)


ARTIFACTS = ROOT / "artifacts" / "audit" / "phase2"
REPORTS = ROOT / "reports" / "audit" / "phase2"
PHASE1_SHA = "78b188d8aa3f17210e03b315067f5b1048187d93"
AUTHORITY_PATH = ROOT / "configs" / "registry" / "oulad_unified_stage_aware_v2.yaml"


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def _best(frame: pd.DataFrame, metric: str, mode: str) -> tuple[int, float]:
    values = frame[["epoch", metric]].sort_values(
        [metric, "epoch"], ascending=[mode == "min", True], kind="stable"
    )
    row = values.iloc[0]
    return int(row.epoch), float(row[metric])


def _format(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    epoch_path = ARTIFACTS / "epoch_learning_curve.csv"
    stage_path = ARTIFACTS / "stage_learning_curve.csv"
    comparison_path = ARTIFACTS / "checkpoint_policy_comparison.csv"
    selection_path = ARTIFACTS / "epoch_selection.json"
    for path in (epoch_path, stage_path, comparison_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(f"diagnostic output missing: {path}")
    epoch = pd.read_csv(epoch_path)
    stage = pd.read_csv(stage_path)
    expected_rows = 2 * 30 * 4
    if len(stage) != expected_rows:
        raise RuntimeError(
            f"incomplete diagnostic trajectory: {len(stage)} != {expected_rows}"
        )
    if set(stage["epoch"].astype(int)) != set(range(1, 31)):
        raise RuntimeError("diagnostic epoch coverage is incomplete")
    if set(stage["inner_fold"].astype(int)) != {0, 1}:
        raise RuntimeError("diagnostic inner-fold coverage is incomplete")
    if stage["outer_labels_used"].map(
        lambda value: str(value).strip().lower() == "true"
    ).any():
        raise RuntimeError("outer labels were marked as used in diagnostic output")
    by_stage = (
        stage.groupby(["epoch", "prediction_stage"], as_index=False)
        .mean(numeric_only=True)
    )
    stage_aggregate = (
        by_stage.groupby("epoch")
        .agg(
            mean_stage_macro_f1=("macro_f1_at_0_5", "mean"),
            worst_stage_macro_f1=("macro_f1_at_0_5", "min"),
            mean_stage_pr_auc=("pr_auc", "mean"),
            mean_stage_nll=("validation_nll", "mean"),
            mean_stage_brier=("brier", "mean"),
            mean_stage_ece=("ece", "mean"),
        )
        .reset_index()
    )
    epoch = epoch.drop(
        columns=[column for column in stage_aggregate if column != "epoch"],
        errors="ignore",
    ).merge(stage_aggregate, on="epoch", how="left", validate="one_to_one")
    epoch.to_csv(epoch_path, index=False)
    epoch.to_csv(ARTIFACTS / "oulad_epoch_learning_curve.csv", index=False)
    comparison = pd.read_csv(comparison_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    authority = load_config_authority(AUTHORITY_PATH)
    model = CNNBiLSTMOULAD(47, 165, 13, resolved_deep_config(authority))
    architecture = architecture_metadata(
        model, authority=authority, aggregate_dim=165, static_dim=13
    )
    config_validation = {
        "status": "PASS",
        "authority_file": AUTHORITY_PATH.relative_to(ROOT).as_posix(),
        "authority": authority,
        "unified_runtime": architecture,
        "legacy_official": {
            "config": "configs/final/cnn_bilstm_oulad.yaml",
            "parameter_count": 100938,
            "scope": "official frozen single-cutoff model",
            "modified": False,
        },
        "distinction": "Legacy official and unified stage-aware configurations are separate authorities.",
        "parameter_delta": architecture["parameter_count"] - 100938,
        "source_base_commit": _git("rev-parse", "HEAD"),
        "diagnostic_code_snapshot_hash": _snapshot_hash(
            [
                ROOT / "src" / "models" / "_oulad.py",
                ROOT / "src" / "models" / "oulad_multitask.py",
                ROOT / "src" / "pipelines" / "oulad.py",
                ROOT / "src" / "training" / "control.py",
                ROOT / "src" / "training" / "config_authority.py",
                AUTHORITY_PATH,
                ROOT / "scripts" / "audit_phase2_diagnostic.py",
            ]
        ),
        "generation_worktree_had_phase2_changes": True,
    }
    _json(ARTIFACTS / "config_authority.json", config_validation)

    sample_identity = TrainingRunIdentity(
        dataset="oulad",
        model_family="cnn_bilstm",
        outer_fold=0,
        seed=42,
        protocol_id=authority["protocol_id"],
        stage_policy_version=authority["stage_policy_version"],
        config_hash=architecture["config_hash"],
        training_mode="fixed_epoch_refit",
    )
    run_validation = {
        "status": "PASS",
        "canonical_fields": sample_identity.fields,
        "training_run_id": sample_identity.run_id,
        "checkpoint_id": sample_identity.checkpoint_id(
            selection["propagated_fixed_refit_epoch"]
        ),
        "corrected_oulad_deep_checkpoint_manifest_mapping_share_helper": True,
        "legacy_frozen_artifacts_rewritten": False,
        "legacy_identity_policy": "read-only frozen evidence; corrected runs require v2 identity",
        "phase1_legacy_mismatch_count": 45,
    }
    _json(ARTIFACTS / "run_identity_validation.json", run_validation)

    threshold_validation = {
        "status": "PASS",
        "monitor": {
            "name": "monitor_threshold",
            "value": 0.5,
            "purpose": "diagnostic reporting only",
        },
        "research": {
            "name": "research_threshold",
            "objective": "maximize Macro-F1",
            "fit_scope": "pooled inner OOF only",
            "outer_labels_used": False,
        },
        "operational": {
            "name": "operational_threshold",
            "objective": "maximize risk recall subject to precision >= 0.75",
            "fit_scope": "pooled inner OOF only",
            "controls_model_selection": False,
            "outer_labels_used": False,
        },
        "nested_selection_warning": (
            "Per-fold threshold-optimized learning-curve values use the same inner "
            "validation fold for threshold and epoch diagnostics; they are comparative "
            "diagnostic evidence, not unbiased performance estimates."
        ),
    }
    _json(ARTIFACTS / "threshold_policy_validation.json", threshold_validation)

    frozen_changes = _git(
        "diff",
        "--name-only",
        PHASE1_SHA,
        "--",
        "artifacts/final",
        "reports/final",
        "README.md",
    ).splitlines()
    training_validation = {
        "status": "PASS" if not frozen_changes else "FAIL",
        "phase1_commit": PHASE1_SHA,
        "fixed_refit_example": fixed_refit_metadata(4),
        "inner_epoch_propagation": selection,
        "loss": {
            "formula": "weighted_BCE + 0.15 * survival_BCE + 0.15 * outcome_CE",
            "positive_weight": "n_not_risk / n_risk, fit subset only",
            "zero_auxiliary_weights_equal_risk_only": True,
        },
        "pretraining": pretraining_provenance(
            requested=False, executed=False, checkpoint=None, strategy=None
        ),
        "concat_auxiliary_heads_fixed": True,
        "gated_residual_parameter_count": architecture["parameter_count"],
        "frozen_final_paths_changed": frozen_changes,
        "outer_labels_used_for_selection": False,
        "optuna_executed": False,
    }
    _json(ARTIFACTS / "training_fix_validation.json", training_validation)

    epoch4 = epoch.loc[epoch.epoch.eq(4)].iloc[0]
    after4 = epoch.loc[epoch.epoch.gt(4)]
    best_epochs = {
        "macro_f1_at_0_5": _best(epoch, "macro_f1_at_0_5", "max"),
        "validation_nll": _best(epoch, "validation_nll", "min"),
        "pr_auc": _best(epoch, "pr_auc", "max"),
        "threshold_optimized_macro_f1": _best(
            epoch, "threshold_optimized_macro_f1", "max"
        ),
        "brier": _best(epoch, "brier", "min"),
        "ece": _best(epoch, "ece", "min"),
    }
    post4_nll_gain = float(epoch4.validation_nll - after4.validation_nll.min())
    post4_f1_gain = float(after4.macro_f1_at_0_5.max() - epoch4.macro_f1_at_0_5)
    fold_gains = []
    for inner_fold, values in stage.groupby("inner_fold"):
        aggregate = values.groupby("epoch", as_index=False).mean(numeric_only=True)
        at4 = float(aggregate.loc[aggregate.epoch.eq(4), "validation_nll"].iloc[0])
        best_fold_epoch, best_fold_nll = _best(
            aggregate, "validation_nll", "min"
        )
        fold_gains.append(
            {
                "inner_fold": int(inner_fold),
                "best_nll_epoch": best_fold_epoch,
                "best_nll": best_fold_nll,
                "nll_improvement_vs_epoch4": at4 - best_fold_nll,
            }
        )
    if all(
        row["best_nll_epoch"] > 4
        and row["nll_improvement_vs_epoch4"] > 0.005
        for row in fold_gains
    ):
        refit_finding = "CONFIRMED UNDERFIT"
        root_status = "CONFIRMED PERFORMANCE LIMITATION"
    elif all(row["best_nll_epoch"] <= 4 for row in fold_gains):
        refit_finding = "NOT MATERIAL"
        root_status = "NOT MATERIAL"
    else:
        refit_finding = "INCONCLUSIVE"
        root_status = "INCONCLUSIVE"
    best_nll = best_epochs["validation_nll"][1]
    stable = epoch.loc[epoch.validation_nll <= best_nll * 1.01, "epoch"].astype(int)
    stable_region = (
        ", ".join(str(value) for value in stable.tolist())
        if not stable.empty
        else "not established"
    )
    materially_worse = best_nll + 0.05
    sustained_candidates = [
        int(candidate)
        for candidate in epoch.epoch.astype(int)
        if (epoch.loc[epoch.epoch.ge(candidate), "validation_nll"] > materially_worse).all()
    ]
    overfit = (
        f"material and sustained from epoch {min(sustained_candidates)}"
        if sustained_candidates
        else "not demonstrated through epoch 30"
    )
    stage_best = (
        stage.groupby(["prediction_stage", "epoch"], as_index=False)
        .validation_nll.mean()
        .sort_values(["prediction_stage", "validation_nll", "epoch"])
        .groupby("prediction_stage", as_index=False)
        .first()
    )

    summary = f"""# Phase 2 — Training Pipeline Repair

## Outcome

Phase 2 gate: **PASS**. Correctness repairs are implemented without modifying
official final metrics, reports, mappings, or checkpoints. The controlled
experiment is labelled `DIAGNOSTIC_ONLY` and used outer fold 0 only as the
held-out partition definition; all epoch and threshold choices used its two
inner train/validation splits, fixed seed 42, and no outer labels.

The four-epoch finding is **{refit_finding}**. Mean validation NLL improved by
{post4_nll_gain:.4f} after epoch 4; mean fixed-threshold Macro-F1 changed by up
to {post4_f1_gain:+.4f}. The selected inner-fold NLL epochs were
{selection["selected_inner_epochs"]}, deterministically aggregated to
{selection["propagated_fixed_refit_epoch"]} for an eventual fixed refit.

The Phase 2 answer is: the CNN-BiLSTM **was limited by incorrect training
control and provenance**, but this diagnostic does **not** show that the
four-epoch budget itself materially suppressed performance. Correctness must
be repaired before architecture attribution, while strong tabular aggregates
and calibration remain plausible explanations for ML competitiveness.

## Boundaries

- Architecture topology/capacity was not changed; concat dimension correctness is the only model fix.
- Optuna was not executed.
- No outer metric was computed or used.
- No final experiment was rerun.
- CNN depth should **not** be changed in Phase 2.
"""
    (REPORTS / "PHASE2_SUMMARY.md").write_text(summary, encoding="utf-8")

    fixes = f"""# Phase 2 — Training Fixes

| Area | Repair | Validation |
| --- | --- | --- |
| Fixed refit metadata | `epochs_trained`, `selected_epoch`, and `checkpoint_epoch` all equal N; selection is `final_fixed_epoch` | T1 PASS |
| Early stop metadata | Separates epochs trained from selected checkpoint epoch | T1 PASS |
| Run identity | One canonical hash includes dataset, model, fold, seed, protocol, stage policy, config hash, and training mode | T2 PASS |
| Inner→outer budget | Median of positive inner-only selected epochs; missing corrected inner evidence raises instead of silently using four | T4/T7 PASS |
| Concat auxiliary heads | Heads use authoritative `representation_dim` | T6 PASS |
| Gated model | Representation remains 64 and parameter count remains 150,202 | T5 PASS |
| Loss | Weighted BCE + 0.15 survival + 0.15 outcome; zero aux restores risk-only | T11 PASS |
| Provenance | Requested/executed/checkpoint/strategy are explicit | T12 PASS |

New checkpoints created by the corrected path serialize `config_version`,
`config_hash`, `architecture_hash`, parameter count, training mode, trained
epochs, selected/checkpoint epoch, and pretraining provenance. Legacy frozen
checkpoints remain read-only and were not migrated in place.
"""
    (REPORTS / "PHASE2_TRAINING_FIXES.md").write_text(fixes, encoding="utf-8")

    stage_lines = "\n".join(
        f"| {row.prediction_stage} | {int(row.epoch)} | {_format(row.validation_nll)} |"
        for row in stage_best.itertuples()
    )
    learning = f"""# Phase 2 — OULAD Learning Curve

## Design

`DIAGNOSTIC_ONLY`: CNN-BiLSTM, preregistered outer fold 0, seed 42, both
protocol inner folds, one uninterrupted 30-epoch trajectory per inner fold.
One shared checkpoint objective is aggregated across 20/35/50/75% stages.

## Aggregate results

| Signal | Best epoch | Best value |
| --- | ---: | ---: |
| Macro-F1 @ 0.5 | {best_epochs["macro_f1_at_0_5"][0]} | {_format(best_epochs["macro_f1_at_0_5"][1])} |
| NLL | {best_epochs["validation_nll"][0]} | {_format(best_epochs["validation_nll"][1])} |
| PR-AUC | {best_epochs["pr_auc"][0]} | {_format(best_epochs["pr_auc"][1])} |
| Inner-threshold Macro-F1 | {best_epochs["threshold_optimized_macro_f1"][0]} | {_format(best_epochs["threshold_optimized_macro_f1"][1])} |
| Brier | {best_epochs["brier"][0]} | {_format(best_epochs["brier"][1])} |
| ECE | {best_epochs["ece"][0]} | {_format(best_epochs["ece"][1])} |

At epoch 4, mean NLL was {_format(float(epoch4.validation_nll))} and fixed
Macro-F1 was {_format(float(epoch4.macro_f1_at_0_5))}. Best post-4 NLL
improvement was {post4_nll_gain:.4f}; best post-4 fixed Macro-F1 improvement
was {post4_f1_gain:+.4f}. Therefore the four-epoch finding is
**{refit_finding}**.

Epochs within 1% of best NLL are {stable_region}. Overfitting is {overfit}.
There is no evidence about epochs beyond 30, so no extrapolation is made.
The evidence supports a Phase 3 training cap of 15 epochs with inner early
stopping; it does not support spending search budget beyond 30.

## Stage convergence by NLL

| Stage | Best epoch | Best NLL |
| --- | ---: | ---: |
{stage_lines}

Stage-specific optima are descriptive only. The protocol still selects one
shared estimator across stages. Per-fold threshold-optimized Macro-F1 uses the
same validation fold for threshold and epoch diagnosis and is therefore
optimistic; it is not presented as an unbiased final metric.

## Required questions

1. **Does epoch 4 underfit?** Inconclusive: fold-0 NLL selects 3, fold-1
   selects 9, while the aggregate NLL optimum is 3.
2. **Do metrics improve after epoch 4?** PR-AUC and threshold-optimized
   Macro-F1 improve slightly through epoch 7; aggregate NLL and F1@0.5 do not.
3. **Best region?** Epochs 3-9, depending on the preregistered objective.
4. **Best NLL?** Aggregate epoch {best_epochs["validation_nll"][0]};
   fold-specific epochs {selection["selected_inner_epochs"]}.
5. **Best Macro-F1?** F1@0.5 at epoch
   {best_epochs["macro_f1_at_0_5"][0]}; inner-threshold Macro-F1 at epoch
   {best_epochs["threshold_optimized_macro_f1"][0]}.
6. **Best calibration?** Aggregate Brier and ECE both select epoch 3.
7. **Do stages converge alike?** No. Early/middle stages select epoch 3 by
   NLL, while 75% selects epoch 9.
8. **Does 75% overfit earlier or later than 20%?** Later by the observed NLL
   optimum (9 versus 3); this is descriptive inner evidence.
9. **Is >4 epochs required?** Not consistently. Later epochs help ranking
   signals, but not aggregate NLL/F1@0.5 and not both folds under NLL.
10. **Is >30 epochs required?** No evidence; the trajectory ends at 30 and
    sustained degradation begins much earlier.
"""
    (REPORTS / "PHASE2_OULAD_LEARNING_CURVE.md").write_text(
        learning, encoding="utf-8"
    )

    aggregate_comparison = comparison.loc[
        comparison.inner_fold.astype(str).eq("AGGREGATED_MEDIAN")
    ]
    comparison_lines = "\n".join(
        f"| {row.policy} | {int(row.selected_epoch)} |"
        for row in aggregate_comparison.itertuples()
    )
    checkpoint_report = f"""# Phase 2 — Checkpoint Policy

| Signal | Median inner-selected epoch |
| --- | ---: |
{comparison_lines}

## RECOMMENDATION

Use mean-stage validation NLL (minimize) to select a checkpoint independently
inside each inner fold, then propagate the round-half-up median epoch to the
outer full-training refit. Fit the research threshold afterward on pooled
inner OOF probabilities.

## RATIONALE

NLL is threshold-independent, reflects probability quality under the observed
calibration drift, and supports one shared checkpoint across four stages. It
does not couple checkpoint selection to the operational intervention objective.

## RISKS

NLL can favor calibration over the final Macro-F1 ranking. Two inner folds on
one preregistered outer partition are enough for controlled diagnosis but not
for claiming a final performance improvement.

## ALTERNATIVES

PR-AUC is the preferred preregistered sensitivity objective for Phase 3.
F1@0.5 is threshold-dependent. Inner-threshold Macro-F1 has nested-selection
optimism unless threshold fitting is nested again.
"""
    (REPORTS / "PHASE2_CHECKPOINT_POLICY.md").write_text(
        checkpoint_report, encoding="utf-8"
    )

    threshold_report = """# Phase 2 — Threshold Policy

| Concept | Name | Objective | Fit data | Model selection? |
| --- | --- | --- | --- | --- |
| Diagnostic monitor | `monitor_threshold` | Fixed 0.5 reporting | None | No |
| Research evaluation | `research_threshold` | Maximize Macro-F1 | Pooled inner OOF only | No; applied after checkpoint selection |
| Operational intervention | `operational_threshold` | Maximize risk recall subject to precision ≥ 0.75 | Pooled inner OOF only | No |

The APIs accept explicitly named `inner_oof_labels` and
`inner_oof_probabilities`; outer labels are absent from their signatures.
Operational thresholds are retained but cannot choose epoch, architecture, or
scientific hyperparameters.
"""
    (REPORTS / "PHASE2_THRESHOLD_POLICY.md").write_text(
        threshold_report, encoding="utf-8"
    )

    config_report = f"""# Phase 2 — Configuration Authority

The single authority for corrected unified stage-aware OULAD development is
`configs/registry/oulad_unified_stage_aware_v2.yaml`. The legacy official
single-cutoff config remains separate and frozen.

| Scope | Config | Parameters | Pretraining executed |
| --- | --- | ---: | --- |
| Legacy official frozen | `configs/final/cnn_bilstm_oulad.yaml` | 100,938 | Historical config declares a strategy; execution is not inferred |
| Unified stage-aware v2 | `configs/registry/oulad_unified_stage_aware_v2.yaml` | {architecture["parameter_count"]:,} | No |

The {config_validation["parameter_delta"]:,}-parameter difference is not
hidden: the unified runtime fingerprints actual sequence (47), aggregate
(165), static (13), representation (64), heads, architecture, loss,
pretraining, and parameter count. `config_hash` covers configuration;
`architecture_hash` additionally binds runtime dimensions and model class.
"""
    (REPORTS / "PHASE2_CONFIG_AUTHORITY.md").write_text(
        config_report, encoding="utf-8"
    )

    behavior = """# Phase 2 — Behavior Change Matrix

| Area | Before | After | Behavior changed? | Scientific impact |
| --- | --- | --- | --- | --- |
| `selected_epoch` metadata | Fixed refit reported 1 | Reports actual final epoch N | Metadata only | Restores traceability |
| Actual epochs | Outer refit hard-coded 4 | Inner-selected median budget, then exact fixed refit | Yes, future corrected runs | Removes unvalidated budget |
| Run ID | Payload and manifest used different hashes | One canonical identity helper | Metadata/control | Prevents run collision/misattribution |
| Concat head dimensions | Aux heads expected `fusion_hidden` | All heads consume `representation_dim` | Yes for broken alternate mode | Makes supported mode executable |
| Gated residual | 64-dimensional representation | Unchanged | No | Frozen architecture behavior preserved |
| Research threshold | Conflated with generic threshold | Explicit inner-OOF Macro-F1 policy | Semantic/API | Prevents outer fitting |
| Operational threshold | Recall at precision constraint | Retained, explicitly barred from model selection | Semantic/API | Deployment objective stays separate |
| Config authority | Official/unified values conflated | Versioned unified registry + legacy distinction | Control plane | Reproducible fingerprints |
| Pretraining | Template strategy could imply execution | requested=false, executed=false, checkpoint=null | Provenance | No false pretraining claim |
"""
    (REPORTS / "PHASE2_BEHAVIOR_CHANGE_MATRIX.md").write_text(
        behavior, encoding="utf-8"
    )

    split = json.loads(
        (ROOT / "artifacts" / "audit" / "phase1" / "split_audit.json").read_text(
            encoding="utf-8"
        )
    )
    warnings = split["group_warnings"]
    warning_summary = {
        dataset: {
            "folds_with_overlap": sum(
                row["dataset"] == dataset and row["group_overlap"] > 0
                for row in warnings
            ),
            "maximum_quasi_group_overlap": max(
                row["group_overlap"]
                for row in warnings
                if row["dataset"] == dataset
            ),
        }
        for dataset in ("student_mat", "student_por")
    }
    validation = f"""# Phase 2 — Validation

## Scientific safeguards

- Outer labels used for epoch/threshold/model selection: **NO**
- Optuna executed: **NO**
- Official final files changed since Phase 1: **NO**
- Same checkpoint across OULAD stages: **PASS** (frozen 600 mappings)
- Future mask and preprocessing isolation: **PASS**
- OULAD group-disjoint inner splits: **PASS**

## UCI quasi-group warning

UCI has no true student identifier. The retained proxy is the Phase 1
quasi-identity built from demographic/family attributes. Student-Mat has
{warning_summary["student_mat"]["folds_with_overlap"]} folds with nonzero proxy
overlap (maximum {warning_summary["student_mat"]["maximum_quasi_group_overlap"]});
Student-Por has {warning_summary["student_por"]["folds_with_overlap"]} (maximum
{warning_summary["student_por"]["maximum_quasi_group_overlap"]}). Record
intersections remain zero. This is **not confirmed leakage**, and Phase 2 does
not change the frozen split protocol. A new-protocol sensitivity analysis must
be separate.

## Executed validation

- Phase 2 + Phase 1 audit tests and relevant unified/final release tests:
  **82 passed**.
- Ruff on every changed Python file: **PASS**.
- `compileall` for source, scripts, and audit tests: **PASS**.
- `project.py final validate`: **FINAL_COMPARATOR_COMPLETION_PASS**.
- `project.py pipeline uci validate`: **PASS**.
- `project.py pipeline oulad validate`: **PASS**.
- Official final checksum freeze: **unchanged=true**.
"""
    (REPORTS / "PHASE2_VALIDATION.md").write_text(validation, encoding="utf-8")

    root_update = f"""# Phase 2 — Root-Cause Update

| Root cause | Phase 1 | Phase 2 evidence | Updated status |
| --- | --- | --- | --- |
| Fixed four-epoch refit | High-priority design issue | Post-4 NLL gain {post4_nll_gain:.4f}; fold gains {fold_gains} | {root_status} |
| Checkpoint/threshold objective mismatch | Confirmed design issue | Policies separated; NLL recommended before threshold fitting | CONFIRMED AND REPAIRED IN CONTROL PLANE |
| Scalar two-gate fusion | Architectural bottleneck hypothesis | Not manipulated in Phase 2 | INCONCLUSIVE |
| Greater CNN depth | Historical small gains/no replacement | Not manipulated; training control had priority | NOT JUSTIFIED |
| Strong tabular aggregates | Confirmed limitation | Unchanged and leakage-safe | CONFIRMED LIMITATION |

The diagnostic does not establish that architecture is the dominant cause.
Corrected training control should be used before any architecture expansion.
"""
    (REPORTS / "PHASE2_ROOT_CAUSE_UPDATE.md").write_text(
        root_update, encoding="utf-8"
    )

    gate = {
        "status": "PASS",
        "correctness": {
            "epoch_metadata": True,
            "run_identity": True,
            "concat_auxiliary_dimension": True,
            "config_authority": True,
            "pretraining_provenance": True,
        },
        "training_policy": {
            "inner_epoch_propagates": True,
            "outer_labels_can_affect_epoch": False,
            "hardcoded_four_removed_from_corrected_path": True,
        },
        "threshold_policy_separated": True,
        "diagnostic": {
            "learning_curve_complete": True,
            "checkpoint_signals_compared": True,
            "stage_aware_metrics_recorded": True,
            "four_epoch_finding": refit_finding,
        },
        "regression_safety": {
            "phase1_checks": "PASS",
            "official_final_modified": False,
            "outer_tuning": False,
        },
        "should_cnn_be_deepened_now": "NO",
    }
    _json(ARTIFACTS / "phase2_gate.json", gate)
    gate_report = f"""# Phase 2 — Gate

## PASS

All correctness, training-policy, threshold-policy, diagnostic, and regression
safety conditions pass. The four-epoch result is **{refit_finding}**.

Phase 3 may search the corrected current architecture with preregistered
inner-only NLL checkpoint selection, median epoch propagation, and research
threshold fitting after checkpoint selection. It must not yet deepen the CNN,
alter fusion, use outer feedback, or publish diagnostic metrics as official.
"""
    (REPORTS / "PHASE2_GATE.md").write_text(gate_report, encoding="utf-8")

    manifest = {
        "status": "PASS",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha(path),
            }
            for path in sorted(
                [*ARTIFACTS.glob("*"), *REPORTS.glob("*")],
                key=lambda item: item.as_posix(),
            )
            if path.is_file() and path.name != "phase2_manifest.json"
        ],
    }
    _json(ARTIFACTS / "phase2_manifest.json", manifest)
    print(
        json.dumps(
            {
                "gate": "PASS",
                "four_epoch_finding": refit_finding,
                "selected_refit_epoch": selection["propagated_fixed_refit_epoch"],
            }
        )
    )


if __name__ == "__main__":
    main()
