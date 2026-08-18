# Prediction / Student State Authority Reconciliation

## Final authority

| Check | Result | Evidence |
|---|---|---|
| Frozen Hybrid seeds | PASS | `[42, 1201, 2026]`; exactly 3 rows per record-stage after filtering |
| Probability column | PASS | `score` from `artifacts/prediction/final/predictions/predictions.parquet`; Student State uses its mean over seeds |
| Ensemble location | PASS | adapter aggregation in `src/recommendation/contracts/prediction.py`; no model inference or retraining |
| Prediction lineage | PASS | source artifact SHA-256 is pinned in `prediction_source_version` |
| Stage scope | PASS | early stages only: 20pct/35pct/50pct/75pct; FINAL-100 excluded |
| State grain | PASS | one row per existing `record_id × stage`; 100061 rows |
| Prediction/state coverage | PASS | missing=0, extra=0 |
| Outer-fold reconciliation | PASS | 0 prediction rows disagree with `oulad_outer.parquet` |
| OOF authority | PASS | safety/consumption manifests say outer-test authorized and consumed; exact outer-fold assignment matches; no in-sample column is consumed |
| Config discrepancy | RESOLVED | runtime consumption/safety and outer-OOF report are stronger than stale selection configs that say `outer_test_used=false` |
| Checkpoint status | REVIEW | standalone Phase8 final checkpoint is missing; existing frozen prediction artifact remains read-only authority |
| Identity contract | PASS | `record_id = sha256('oulad|code_module|code_presentation|id_student')[:24]`; `student_id=id_student`; module/presentation retained |
| PostgreSQL mapping | PASS WITH HANDOFF | `record_id` is the deterministic future `external_enrollment_id`; current import does not populate that column and no fake enrollment was created |

## Reconciled stage counts

| Stage | Prediction record-stage rows | Student State rows |
|---|---:|---:|
| 20pct | 26697 | 26697 |
| 35pct | 25606 | 25606 |
| 50pct | 24599 | 24599 |
| 75pct | 23159 | 23159 |

## Grain and identifier notes
- One Student State row is one OULAD student/module/presentation enrollment case at one recommendation stage.
- The 26,697 / 25,606 / 24,599 / 23,159 counts are the existing Hybrid early-warning prediction cohort after collapsing its three seed rows per record-stage.
- No FINAL-100 row, final outcome, assessment score, or future feature is used in the state.
- Current PostgreSQL `catalog.enrollment` uses UUID internally; the deterministic OULAD `record_id` is the only allowed external mapping key for a later ingest.

Prediction artifact SHA-256: `bd3ea11558fce882ae4371af1d6421b2a86f6adaf2ff57254c14b8ec54fca768`
