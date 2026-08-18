# Unified Database Migration Report

- Status: `READY_FOR_DATABASE_CUTOVER`
- Source: `student_predict` (read-only)
- Replacement: `student_predict_unified_replacement_3`
- Canonical database modified: NO
- UCI prediction key: `(run_id, record_pk, prediction_stage)`
- Metrics support `STAGE` and `OVERALL` scopes.
- Prediction rows: 185,100
  - Unified UCI stage predictions: 31,320
  - Frozen OULAD predictions: 153,780
- Models: 30
- Risk profiles: 15,378
- Plans: 15,378
- Actions: 27,355
- Future OULAD: `LOCKED_NOT_EXECUTED`
- OULAD retrained: NO

The replacement database is retained for review. No production/canonical cutover was performed.
