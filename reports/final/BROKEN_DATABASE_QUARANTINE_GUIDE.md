# Broken Database Quarantine Guide

## Quarantined database

The database affected by the failed in-place restore is retained as:

`student_predict_broken_restore_20260728t050600`

It does not accept new connections. It was not dropped, truncated, reused, or
selected as a recovery source.

The canonical application database is now the fully validated replacement
under the normal name:

`student_predict`

## Retention requirement

Do not remove the quarantined database until all of the following are true:

1. `main` has been pushed successfully.
2. `PROJECT_LOCKED_READY_FOR_THESIS_UPDATE` is recorded.
3. The verified 27-model recovery backup is retained:
   `student_predict_pre_30_model_cutover_20260728T043451Z.dump`.
4. The verified 30-model backup is retained:
   `student_predict_validated_30_model_20260728T050437Z.dump`.
5. The forensic partial-cutover backup is retained:
   `student_predict_failed_partial_cutover_20260728T044215Z.dump`.
6. Their SHA-256 values match the committed manifests.

## Manual removal procedure

Removal is intentionally not automated by the project. An administrator should
first inspect active sessions through `pg_stat_activity`, confirm the exact
quarantine name, and retain a final inventory if required by local policy.

Only then may the administrator remove
`student_predict_broken_restore_20260728t050600` from an administrative
connection. Never use a wildcard or derive the target from an unvalidated
environment variable.

The quarantine database contains no canonical recovery authority. The
validated backups and committed evidence are the recovery sources of record.
