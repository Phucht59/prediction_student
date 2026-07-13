# Benchmark V2 preflight audit

## Decision: benchmark is permitted to start

The shared manifest validates with 316 development records, five outer folds, exactly one outer-validation assignment per record, no within-fold overlap, and checksum `bf5e5cbd8d09679f5d34900486ba23cc5ac57c93b28aa38ac3f3ce2578307ce1`. Dataset version 1 loaded DB-first with the matching checksum `e47f9ee225e1ee6e69b7564e6dac7123e80b8486677fe111f351964cef5dec80`.

The legacy manifest marks all 79 identities as `legacy_heldout_observed`. The V2 fold loader requires exact development membership and `assert_no_legacy_records` rejects any observed identity. Existing tests also verify that a legacy identity cannot be supplied to the V2 outer-fold constructor.

`fit_fold_predict_proba` selects epochs with an internal split, then `train_fixed_epochs` refits on all outer-train rows. Outer-validation is used only for inference. Feature guards reject G3/G3-derived columns and scenario contracts admit only `[G1, G2]` for late-stage and `[G1]` for early-warning. Prediction rows must correspond exactly to outer-validation identities.

## Skipped tests

All five skips are in `tests/test_postgres_source_ml_integration.py` and share one explicit prerequisite: `POSTGRES_TEST_DSN`, `POSTGRES_TEST_APP_DSN`, and a `psql` executable.

1. `test_schema_and_append_only_lineage_contract` (line 202)
2. `test_source_rows_cannot_be_added_after_first_run` (line 312)
3. `test_run_completion_requires_complete_prediction_coverage` (line 403)
4. `test_app_role_is_insert_only_for_source_split_prediction_ledgers` (line 502)
5. `test_completed_run_allows_append_only_recommendation_policy_versions_only` (line 628)

They are PostgreSQL schema/role integration tests, not benchmark, fold, leakage, or metric tests. They can be rerun when a disposable PostgreSQL test database and both DSNs are supplied. Their skip is expected in this workstation configuration and is not a protocol blocker. The DB-first dataset loader itself was successfully read during preflight.

## Preflight result

No severe protocol defect was found. Benchmark execution may begin; results must remain nested-CV evidence on the 316-record development cohort and may not be ranked against Protocol V1 metrics.
