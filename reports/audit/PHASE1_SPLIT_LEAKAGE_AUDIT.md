# Phase 1 — Split and Leakage Audit

## Summary

| Dataset | Outer split | Inner split | Group-safe | Preprocess train-only | Threshold inner-only | Leakage status |
| --- | --- | --- | --- | --- | --- | --- |
| Student-Mat | Frozen 5-fold stratified OOF | 3-fold StratifiedGroupKFold | **No for quasi-ID proxy** | Yes | N/A, argmax | Record-disjoint; potential quasi-group overlap |
| Student-Por | Frozen 5-fold stratified OOF | 3-fold StratifiedGroupKFold | **No for quasi-ID proxy** | Yes | N/A, argmax | Record-disjoint; potential quasi-group overlap |
| OULAD | Frozen 3-fold student-group OOF | 2-fold StratifiedGroupKFold | Yes (`id_student`) | Yes | Yes | PASS |

## Automated intersections

All audited folds satisfy:

```text
train record ∩ validation record = ∅
train record ∩ test record = ∅
validation record ∩ test record = ∅
```

OULAD additionally has zero `id_student` overlap in all outer folds and zero
base-record overlap in the audited inner folds.

## UCI quasi-group finding

The frozen UCI outer assignments came from historical
`StratifiedKFold`, not `StratifiedGroupKFold`. The unified pipeline reconstructs
a conservative quasi-identity from school, sex, age, address, family structure,
parent education/jobs, reason, nursery, and internet status.

- Student-Mat: quasi-group overlap in outer folds 0–3, counts 1, 1, 4, 2;
  fold 4 has zero.
- Student-Por: overlap in all folds, counts 5, 5, 6, 3, 3.

This is a **POTENTIAL ISSUE**, not confirmed student leakage. UCI provides no
true student identifier, each dataset contains one row per observation, and a
quasi-identity collision may represent different students. The correct claim
is that the frozen outer split is record-disjoint but not group-safe under the
project's own proxy. Phase 1 does not replace the frozen split or recompute
official results.

## OULAD stage and feature safety

- VLE events are filtered with `date >= 0` and `date < cutoff_day` before
  aggregation.
- Submissions are filtered with `date_submitted < cutoff_day`.
- Raw score values are excluded because release timestamps are unavailable;
  `score_missing_mask=1`.
- Future padded weeks are zero and excluded by mask and packed lengths.
- Aggregate features are derived from each stage's already cutoff-filtered
  temporal view.
- `final_result` and `date_unregistration` are not predictors.
- `date_unregistration` is retained only for eligibility/survival targets.
- Stage expansion occurs after frozen base-record fold assignment.

## Preprocessing scope

OULAD tabular sklearn pipelines and `_DeepPreprocessor` are fit only on the
provided fit rows. OULAD sequence values use per-timestep LayerNorm inside the
model rather than a dataset-fitted temporal scaler; padding is masked before
convolution and recurrence. UCI context preprocessors are fit on the current
fit indices before transforming other rows.

## Threshold scope

OULAD thresholds are chosen from pooled inner-OOF predictions by outer fold and
stage. Outer labels are not passed to `_threshold`. Outer labels are used only
later for evaluation. The audit found no threshold-selection leakage.

Machine-readable fold intersections are in
`artifacts/audit/phase1/split_audit.json`.
