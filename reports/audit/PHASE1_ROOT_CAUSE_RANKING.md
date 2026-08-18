# Phase 1 — Root-Cause Ranking

The ranking is based on confirmed source control flow, checkpoint contents,
frozen predictions/metrics, and automated split tests. Expected metric impact
is not treated as measured unless frozen evidence directly supports it.

## RC-01

```text
ID: RC-01
TITLE: Unified OULAD deep training uses a four-epoch frozen default
CATEGORY: TRAINING
STATUS: CONFIRMED DESIGN ISSUE
SEVERITY: HIGH
CONFIDENCE: HIGH
AFFECTED DATASET: OULAD unified
AFFECTED STAGE: 20%, 35%, 50%, 75%
EVIDENCE: oulad_prediction.yaml; oulad.py:758-812,954-999
WHY IT MATTERS: Inner selected epochs are discarded and every outer model is a four-epoch last-state refit.
EXPECTED PERFORMANCE IMPACT: Potentially material, not quantified without inner-only diagnostic.
FIX PHASE: Phase 2
```

## RC-02

```text
ID: RC-02
TITLE: Early-stage hybrid probabilities are materially miscalibrated
CATEGORY: CALIBRATION
STATUS: CONFIRMED DESIGN ISSUE
SEVERITY: HIGH
CONFIDENCE: HIGH
AFFECTED DATASET: OULAD unified
AFFECTED STAGE: 20%, 35%
EVIDENCE: calibration_audit.csv; frozen stage_metrics.csv
WHY IT MATTERS: Hybrid ECE is 0.127/0.098 versus HGB 0.019/0.017, causing strong threshold dependence.
EXPECTED PERFORMANCE IMPACT: Large impact on thresholded metrics; ranking metrics remain competitive.
FIX PHASE: Phase 2
```

## RC-03

```text
ID: RC-03
TITLE: Operational threshold objective differs from headline Macro-F1
CATEGORY: OBJECTIVE
STATUS: CONFIRMED DESIGN ISSUE
SEVERITY: HIGH
CONFIDENCE: HIGH
AFFECTED DATASET: OULAD unified
AFFECTED STAGE: all
EVIDENCE: oulad.py:838-846,849-875,1050-1063
WHY IT MATTERS: Threshold maximizes recall subject to inner precision >=0.75, not Macro-F1.
EXPECTED PERFORMANCE IMPACT: At 75%, frozen Macro-F1 is 0.8062 operational versus 0.8511 at fixed 0.5.
FIX PHASE: Phase 2 protocol/objective clarification
```

## RC-04

```text
ID: RC-04
TITLE: ML receives a strongly aggregated stage-safe sequence representation
CATEGORY: DATA
STATUS: CONFIRMED LIMITATION
SEVERITY: HIGH
CONFIDENCE: HIGH
AFFECTED DATASET: OULAD unified
AFFECTED STAGE: all
EVIDENCE: oulad.py:356-385,524-558
WHY IT MATTERS: 161 totals/means/extrema/last/recent/slope/half-window features are an excellent tabular inductive prior.
EXPECTED PERFORMANCE IMPACT: Explains competitive HGB/XGBoost without leakage.
FIX PHASE: No bug fix; preserve and report
```

## RC-05 to RC-08

| ID | Status | Finding | Direct metric impact |
| --- | --- | --- | --- |
| RC-05 | CONFIRMED BUG | `selected_epoch=1` metadata although 4 epochs execute | None |
| RC-06 | CONFIRMED BUG | Payload/manifest run IDs mismatch 45/45 | None |
| RC-07 | CONFIRMED BUG | OULAD concatenation fusion breaks auxiliary head dimensions | None for current gated model |
| RC-08 | CONFIRMED DESIGN ISSUE | Official and unified config authorities are conflated | None; major reproducibility impact |

## RC-09 to RC-12

| ID | Status | Finding |
| --- | --- | --- |
| RC-09 | POTENTIAL ISSUE | Frozen UCI outer folds are record-disjoint but not quasi-group-safe; true student leakage is unprovable without IDs |
| RC-10 | CONFIRMED LIMITATION | UCI temporal length is 0/1/2 and unified deep search has only two candidates |
| RC-11 | CONFIRMED LIMITATION | Historical CNN capacity/structural variants produced only small inner gains and failed the replacement gate |
| RC-12 | NOT AN ISSUE | No record overlap, OULAD student overlap, future-feature leakage, threshold outer-label use, or stage-specific checkpoint retraining found |

The complete field-by-field ranking is in
`artifacts/audit/phase1/root_cause_ranking.json`.
