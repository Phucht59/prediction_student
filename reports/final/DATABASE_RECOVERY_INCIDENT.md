# Database Recovery Incident

## Summary

An in-place restore of the verified 27-model backup was attempted after the
first database cutover exposed stale canonical metric rows. PostgreSQL stopped
the restore while processing `DROP SCHEMA recommendation`.

The restored archive predates migration 011. The partially cut-over database
already contained three empty migration-011 tables:

- `recommendation.expert_review_case`
- `recommendation.expert_plan_review`
- `recommendation.expert_action_review`

Those objects were not archive members, so `pg_restore --clean` did not issue
individual drop statements for them. They therefore blocked removal of their
parent schema. Because `--exit-on-error` was active, the operation stopped
without restoring the archive.

## Resulting state

A read-only inventory after the failure found only the three expert-review
tables above. The affected `student_predict` database is not usable and must
not receive another in-place restore.

No scientific artifact, prediction, model checkpoint, split manifest,
recommendation policy, DOCX, or PDF was modified by this database incident.

## Data safety

Two independently readable backups remain available:

| Purpose | File | SHA-256 |
|---|---|---|
| Verified pre-cutover recovery | `student_predict_pre_30_model_cutover_20260728T043451Z.dump` | `725d26e93493038f5f6f87812e29137287d6c43092c0617846bcefd70eee62b2` |
| Failed partial-cutover forensics | `student_predict_failed_partial_cutover_20260728T044215Z.dump` | `3b56938e9cefcf0c887b4058712f0ecfec40899c365ec4ab446776b531637758` |

The recovery backup had already passed an independent restore test and
reproduced 27 models, 27 runs, 891 metrics, 15,378 risk profiles, 15,378 plans,
27,355 actions, and zero reviews.

## Recovery decision

The approved recovery path is:

1. create an empty replacement database;
2. restore the verified archive without `--clean`;
3. validate the raw 27-model state;
4. apply canonical migration 011;
5. run and validate the repaired atomic cutover;
6. back up the validated 30-model replacement;
7. rename the broken database to a quarantine name;
8. rename the validated replacement to `student_predict`.

The broken database is retained for traceability and is not dropped.
