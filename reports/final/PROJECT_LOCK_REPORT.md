# Final Project Lock

## Git state

- Branch: `codex/unified-stage-aware-system`
- Base: `origin/main@8de5e7bb957afadb213a5e485179435ec014ddda`
- Main merged: NO
- Canonical database cutover: NO

## Unified UCI authority

- One estimator/checkpoint per dataset/model/fold/seed.
- 10 model families per UCI dataset.
- 20 UCI model identities.
- 60 UCI stage-result rows.
- 500 training runs/checkpoints.
- S0/S1/S2 share the same base records and frozen outer folds.
- Grade-band reference is diagnostic only and is not a model identity.

CNN-BiLSTM unified Macro-F1:

| Dataset | S0 | S1 | S2 | Overall pooled |
|---|---:|---:|---:|---:|
| Student-Mat | 0.413558 | 0.743811 | 0.846139 | 0.660556 |
| Student-Por | 0.508886 | 0.754180 | 0.851947 | 0.698301 |

These values are the unified stage-aware authority and are not presented as a
reproduction of the earlier separately trained stage models.

## Scientific freeze

The pre-refactor canonical artifacts remain byte-identical:

- CNN-BiLSTM MAT frozen Macro-F1: `0.9014601961315334`
- CNN-BiLSTM POR frozen Macro-F1: `0.8622587167738002`
- CNN-BiLSTM OULAD frozen Macro-F1: `0.8280835945631038`
- Future OULAD: `LOCKED_NOT_EXECUTED`
- OULAD retrained: NO
- Recommendation expert status: `PENDING_EXPERT_LABELS`
- xAPI: ABSENT

## Replacement database

- Status: `READY_FOR_DATABASE_CUTOVER`
- Replacement: `student_predict_unified_replacement_3`
- Canonical `student_predict` modified: NO
- Models/runs: 30/30
- Metrics: 2,715
- Predictions: 185,100
- Risk profiles/plans/actions: 15,378 / 15,378 / 27,355
- Reviews: 0
- Prediction natural key: `(run_id, record_pk, prediction_stage)`
- UCI stage rows: 10 per dataset/stage
- OULAD stage: `F2_MIDDLE`, 10 models, frozen predictions

## Validation

- Unified validator: PASS
- Final validator: PASS
- Project suite: 62 passed
- Replacement database suite: 51 passed
- Total: 113 passed
- Ruff: PASS
- Canonical checksum freeze: PASS
- Legacy relocation SHA-256 preservation: PASS
- DOCX/PDF modified: NO

## Status

`UNIFIED_STAGE_AWARE_SYSTEM_READY_FOR_REVIEW`

## OULAD unified multi-stage authority

- 10 OULAD model identities; 40 operational stage rows; 10 OULAD overall summaries.
- One estimator/checkpoint is reused across E1, E2, M1 and L1 for every `(model, outer_fold, seed)` run.
- M1 uses the exact historical F2 cutoff, but unified multi-stage results are a replacement authority and are not asserted to reproduce the frozen single-cutoff score.
- Canonical frozen F2 artifacts remain historical compatibility evidence.
