# Phase 1-2 Validation

| Gate | Result | Evidence |
|---|---|---|
| Frozen Hybrid adapter | PASS | 100061 unique record-stage rows after 3-seed mean |
| OULAD stages 20/35/50/75 | PASS | all four present |
| FINAL-100 exclusion | PASS | adapter scope excludes FINAL-100 |
| Identity join | PASS | record_id one-to-one with outer split and features |
| State contract | PASS | deterministic case_id, bounds, booleans, lineage |
| Leakage blacklist | PASS | forbidden fields absent from output; sources are cutoff-safe |
| Uncertainty | PASS WITH UNAVAILABLE | no persisted uncertainty; no invented metric |
| Prediction authority reconciliation | PASS | runtime safety/consumption + exact outer-fold match resolve stale selection-config flag |
| PostgreSQL compatibility | REVIEW | existing tables link prediction/recommendation through UUID enrollment_id; no student_state table or migration was added; external_enrollment_id is the future mapping point |
| API/Snorkel/EBM/retraining | PASS | no calls or training paths in this build |

## Counts

| Stage | Rows |
|---|---:|
| 20pct | 26697 |
| 35pct | 25606 |
| 50pct | 24599 |
| 75pct | 23159 |

State artifact SHA-256: `60d9f7d2e2bb4307271ebaf329e6062d1ad4c8058863625f863edff50f55b162`

No FINAL-100 rows are in the recommendation state artifact.
