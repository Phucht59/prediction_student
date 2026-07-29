# Phase 1 — Forensic Audit

## 1. Executive Summary

The frozen CNN-BiLSTM is being evaluated without the audited forms of
record-level leakage, OULAD student leakage, future-timestep leakage,
stage-specific retraining, best-seed selection, or outer-label threshold
tuning. Preprocessing is fit on training partitions, masks are effective, and
all four OULAD stages share one checkpoint per fold/seed.

Three findings best explain why the unified hybrid does not clearly dominate:

1. Unified OULAD deep models use one untuned `frozen_default` and a four-epoch
   last-state refit. Inner best epochs are discarded.
2. Hybrid early-stage probabilities are much less calibrated than strong ML
   models, and the reported operational thresholds target constrained recall,
   not Macro-F1.
3. ML receives 161 strong, cutoff-safe aggregates that summarize the same
   sequence in a form highly suitable for boosted trees. This is a legitimate
   tabular inductive advantage, not leakage.

Confirmed bugs are metadata/provenance and an inactive fusion-mode dimension
bug. None proves that the current gated model's weights are corrupt:

- all 45 deep unified OULAD checkpoints say `selected_epoch=1` although the
  code executed four fixed epochs;
- checkpoint and manifest run IDs use different hash formulas;
- OULAD multitask auxiliary heads fail under concatenation fusion;
- the architecture audit parameter count assumes the wrong static dimension.

Phase 1 status is **PASS**. The epoch anomaly is fully explained without new
training. Phase 2 should repair training/checkpoint/objective/provenance
mechanics before any VNext Optuna or architecture expansion.

## 2. Frozen Baseline

Official final:

| Dataset | Protocol | Macro-F1 |
| --- | --- | ---: |
| Student-Mat | V5.1 official G1+G2 ensemble | 0.901460 |
| Student-Por | V5.1 official G1+G2 ensemble | 0.862259 |
| OULAD | Official F2 single-cutoff ensemble | 0.828084 |

OULAD `0.828084` is not stage 100%. Unified OULAD is a separate one-estimator,
four-view protocol:

| Stage | Macro-F1 | PR-AUC | Brier | NLL | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20% | 0.700323 | 0.762423 | 0.204659 | 0.592877 | 0.127438 |
| 35% | 0.743506 | 0.808627 | 0.173336 | 0.516599 | 0.097896 |
| 50% | 0.785192 | 0.859072 | 0.137354 | 0.424336 | 0.061745 |
| 75% | 0.806211 | 0.903516 | 0.100588 | 0.326819 | 0.047840 |

## 3. Data & Split Integrity

OULAD uses frozen 3-fold student grouping and 2-fold inner
`StratifiedGroupKFold`; automated intersections are zero. Unified stage views
inherit the base fold before expansion.

UCI record IDs are disjoint, but the frozen historical outer folds were made
with `StratifiedKFold`. The current quasi-identity proxy overlaps across outer
train/test partitions. This is a **POTENTIAL ISSUE**, not confirmed student
leakage, because UCI has no true ID and proxy collisions may be different
students. Inner UCI selection is group-aware.

Seeds are fixed. No best seed is selected; five outer seeds are averaged.

## 4. Preprocessing Integrity

### OULAD temporal

- 47 channels = 16 weekly state channels + 31 present/past dynamics.
- Events and submissions are filtered before weekly aggregation.
- Scores are unavailable and excluded; the missing-score mask is explicit.
- Future/padded values are zero.
- The encoder multiplies projected and convolved values by the mask and packs
  only valid lengths before BiLSTM.
- There is no train-fitted global temporal mean/std. Per-timestep LayerNorm is
  used inside the encoder; padded timesteps do not enter cross-record
  statistics.

### OULAD aggregate/static

`_DeepPreprocessor` fits aggregate mean/std, static numeric mean/std, and
categorical levels on fit rows only. ML sklearn preprocessors likewise fit only
inside `.fit()` on the training rows.

The 161 aggregates include, for each temporal channel, total, mean, standard
deviation, min, max, last, slope, recent-two-week mean, first-half mean, and
second-half mean, plus inactivity. Four stage-context values are appended.
This is a **STRONG TABULAR INDUCTIVE ADVANTAGE**.

### UCI

Context imputation, scaling, and one-hot categories are fit on the current
training indices. G3 is target-only; G1/G2 are removed from context and appear
only in the temporal view permitted by the stage.

## 5. UCI Pipeline Audit

The implemented temporal tensor is `[N, 2, 7]`.

- S0 mask `[0,0]`: grade tensor is zero; recurrent branch is bypassed and
  temporal embedding is exactly zero.
- S1 mask `[1,0]`: only G1-derived timestep is visible; packed length is one.
- S2 mask `[1,1]`: G1 and G2 are visible.
- G3 is never a predictor.

The backward LSTM cannot consume unavailable future padding because positive
length rows are packed. Mask-invariance tests pass.

With length 0/1/2, adding multiple deeper convolution blocks is not
architecturally well motivated: there is no long hierarchy to learn, kernel 2
already spans the entire maximum sequence, and S0 contains no temporal signal.
The context branch and ML models receive equivalent information. The correct
recommendation is **do not deepen now**.

Unified UCI deep search is **LIMITED DEEP SEARCH**:

- two candidates only;
- LR 0.001 or 0.0005;
- WD 0.0001 or 0.001;
- dropout 0.15 or 0.25;
- class weight balanced or none;
- batch 64, max 30, patience 5;
- one LSTM/context layer, GELU, gated fusion forced;
- selection by mean-stage Macro-F1 with worst-stage and calibration tie-breaks.

This is much narrower than the historical official V5.1 search.

## 6. OULAD Pipeline Audit

One training identity is `(model, outer_fold, seed, config_hash)`. Stage is not
part of training identity. Every run maps to 20/35/50/75% using the exact same
checkpoint path and SHA. There is no hidden per-stage refit.

Eligibility removes students not registered by cutoff and students whose
unregistration/outcome is already known before cutoff. Cohort size and positive
prevalence consequently decline with stage; this is not a fixed-cohort trend
unless the common-cohort artifact is used.

No final result, post-cutoff click, post-cutoff submission, score without a
release timestamp, or exact future outcome is used as a predictor.

## 7. Training & Early Stopping

Unified OULAD risk loss uses BCE with a fit-partition positive weight. The
hybrid adds survival and outcome losses, each weight 0.15. AdamW uses
LR 0.0005, WD 0.00001, batch 256, and gradient clipping at 1.0. There is no
scheduler or balanced sampler.

Inner early stopping maximizes mean-stage Macro-F1 at threshold 0.5 with
patience 2 and `min_delta=1e-8`. It evaluates once per epoch and restores the
best inner state. However, the best epoch is not carried to outer refit.

Outer training always executes four fixed epochs, skips validation, and saves
the final state. Thus the final checkpoint is not selected by early stopping.
Whether four epochs underfit is plausible but not measured by retained curves.

## 8. Checkpoint Selection

`selected_epoch=1` is a metadata bug:

- all 45 deep payloads and manifest rows show 1;
- all payload configs show max epochs 4;
- fixed mode executes epochs 1 through 4;
- fixed mode bypasses the only assignment that updates `best_epoch`;
- checkpoint hashes match the manifest.

Payload and manifest training run IDs mismatch 45/45, but checkpoint path and
SHA identity remain valid.

## 9. Threshold & Calibration

Fold-specific CNN-BiLSTM thresholds:

| Fold | 20% | 35% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.659473 | 0.570541 | 0.425408 | 0.286158 |
| 1 | 0.690714 | 0.599402 | 0.438914 | 0.311600 |
| 2 | 0.711064 | 0.648616 | 0.476069 | 0.334922 |

Thresholds are selected from pooled inner-OOF labels separately for
outer-fold/stage. The objective is maximum risk recall among thresholds whose
inner risk precision is at least 0.75; otherwise closest precision. It is not a
Macro-F1 optimum.

Hybrid probability means versus prevalence:

| Stage | Prevalence | Hybrid mean p | Bias | HGB mean p | HGB ECE | Hybrid ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20% | 0.4413 | 0.5679 | +0.1267 | 0.4356 | 0.0186 | 0.1274 |
| 35% | 0.4194 | 0.5169 | +0.0975 | 0.4212 | 0.0166 | 0.0979 |
| 50% | 0.3978 | 0.4596 | +0.0618 | 0.3950 | 0.0148 | 0.0617 |
| 75% | 0.3673 | 0.4139 | +0.0466 | 0.3665 | 0.0141 | 0.0478 |

The representation improves with stage—PR-AUC rises from 0.762 to 0.904 and
Brier/NLL fall—but the model is also heavily dependent on stage-specific
calibration. It is incorrect to attribute the Macro-F1 gain solely to better
representation.

At 75%, fixed 0.5 gives Macro-F1 0.8511 while the registered operational
threshold gives 0.8062 and much higher risk recall. This is an objective
tradeoff shown on frozen outer labels, not authorization to select 0.5 from
outer performance.

## 10. Architecture Inspection

### Unified OULAD CNN-BiLSTM

```text
47-channel sequence
  -> Linear 47→48 + LayerNorm + GELU
  -> parallel same-length Conv1D kernels 2/3/5, 32 channels each
  -> concatenate to 96 + projected residual + LayerNorm + GELU
  -> 1-layer BiLSTM, hidden 64/direction -> 128
  -> masked mean + masked max -> 256
  -> Linear 256→64 + LayerNorm + GELU
  -> temporal projection 64→64

165 aggregate -> Dense 165→64→64
13 static     -> Dense 13→32→64

[temporal, aggregate, static] -> Linear 192→2 + sigmoid
fused = temporal + scalar_gate_1*aggregate + scalar_gate_2*static
```

The gate is two **scalar gates per record**, not a feature-wise 64-dimensional
gate. This is an **ARCHITECTURAL BOTTLENECK HYPOTHESIS**, not a bug.

Parameter counts:

| Component | Parameters |
| --- | ---: |
| Temporal encoder | 122,272 |
| Temporal projection | 4,160 |
| Aggregate branch | 14,912 |
| Static branch | 2,624 |
| Two scalar gates | 386 |
| Risk head | 4,353 |
| Survival head | 1,300 |
| Outcome head | 195 |
| Total | 150,202 |

Dimensions before fusion are 64/64/64; gated output is 64. Concatenation output
would be 192.

### Multitask latent bug

The risk head correctly adapts its LayerNorm input to concatenation. The
survival and outcome heads are always constructed for 64 inputs, while
`representation()` returns 192 under concatenation. A forward test confirms a
matrix-shape failure. This is a **CONFIRMED LATENT BUG** in an alternative
configuration and does not affect the frozen gated-residual model.

All three supported UCI fusion modes—concatenation, gated, and film residual—
pass forward tests.

## 11. Optuna Lineage

The historical OULAD study ran 72 completed F2-only trials on an earlier family.
It tuned single-kernel H2T and aggregate/concatenation controls, not the current
multi-kernel gated-residual multitask model. Unified OULAD uses one
`frozen_default`.

Conclusion: **FINAL ARCHITECTURE IS NOT FULLY OPTUNA-TUNED**.

## 12. Configuration & Provenance

The official canonical YAML says pretraining
`P1_MASKED_AND_NEXT_WEEK`, parameter count 100,938, and official thresholds.
Unified config prohibits pretrained checkpoints, actual checkpoints contain
150,202 parameters, and thresholds are stage/fold inner-OOF. The unified
architecture is partly hardcoded in `_deep_config`.

`architecture_freeze_audit.json` reports 150,234 because it instantiates
static dimension 14 rather than the actual 13. A versioned unified
single-source config does not currently exist.

## 13. Historical Ablation Evidence

Controlled inner-development evidence shows:

- small CNN temporal: 0.820780 Macro-F1;
- capacity-matched CNN: 0.822486, gain +0.001706;
- BiLSTM temporal: 0.824847, still +0.002361 over matched CNN;
- full serial hybrid: 0.828214;
- aggregate/static only: 0.824696, only −0.003518;
- best dilation-1/multidilation gain over dilation-2: about +0.001112;
- best skip/parallel alternative remained −0.000518 below full serial;
- reversing/shuffling/bagging weeks reduced Macro-F1 only 0.0036–0.0066.

The development gate selected no replacement architecture. Deeper/wider CNN
capacity did not create a large gain.

## 14. Confirmed Bugs

1. Unified OULAD epoch metadata reports 1 for four-epoch refits.
2. Unified OULAD checkpoint and manifest run IDs use inconsistent formulas.
3. OULAD concatenation fusion is incompatible with multitask auxiliary heads.
4. Architecture freeze audit parameter count uses the wrong static dimension.

## 15. Confirmed Design Limitations

1. Four-epoch fixed OULAD refit; inner best epochs discarded.
2. Early checkpoint metric at 0.5 differs from operational threshold objective.
3. Operational threshold optimizes constrained recall, not headline Macro-F1.
4. Hybrid probability calibration is materially worse than HGB/XGBoost.
5. ML receives strong stage-safe aggregates.
6. Unified deep search is narrow; final OULAD architecture is not fully Optuna
   tuned.
7. UCI has at most two timesteps, limiting the value of deeper CNN hierarchy.

## 16. Unproven Hypotheses

- Four epochs underfit the hybrid: plausible, not proven without inner curves.
- Scalar gated residual suppresses aggregate/static information: plausible,
  not proven by gate diagnostics.
- UCI quasi-group collisions are the same students: unprovable from available
  IDs.
- Larger/deeper CNN will improve results: historical evidence makes this weak.
- A calibrated hybrid must beat ML: not assumed and not supported.

## 17. Root-Cause Ranking

Highest priority:

1. Fixed four-epoch OULAD training and lost inner epoch selection.
2. Early-stage calibration drift.
3. Threshold/headline-objective mismatch.
4. Strong tabular aggregate inductive advantage.
5. Metadata/run identity bugs.

Full structured ranking is in the dedicated report and JSON artifact.

## 18. Recommended Phase 2 Scope

Phase 2 should fix:

1. epoch metadata, run-ID identity, and complete unified config serialization;
2. inner-only training diagnostics and principled propagation of selected
   epochs/checkpoint objective;
3. explicit separation of operational constrained-recall and Macro-F1
   reporting, with inner-only calibration;
4. concatenation auxiliary-head dimensions before that mode enters any search;
5. UCI group-safety claim wording and a preregistered decision on whether a new
   group-aware protocol is needed without replacing frozen official evidence.

## 19. Things Phase 2 Must NOT Change Yet

- CNN depth;
- LSTM depth;
- fusion architecture;
- model capacity;
- final Optuna search;
- outer splits or official results;
- threshold based on outer labels.

## 20. PASS / PARTIAL PASS / FAIL

**PASS**.

All 15 required gate questions have evidence-backed answers. No controlled
training diagnostic was needed to explain epoch 1. A small inner-only curve run
is recommended in Phase 2 to measure, not assume, the impact of the four-epoch
budget.
