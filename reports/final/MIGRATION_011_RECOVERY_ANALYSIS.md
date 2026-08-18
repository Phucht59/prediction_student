# Migration 011 Recovery Analysis

## Backup state

The verified pre-cutover archive contains migration ledger entries 001 through
010. Its schema hash is
`ae06a0afce55148dbe2b5452a9fe4efbf4d37860c5c05209fdb73799f40bf57e`.
It correctly does not contain the three expert-evaluation tables introduced
after that backup.

There is no ledger contradiction: migration 011 is absent and its objects are
absent in the raw restored replacement.

## Canonical migration

The project migration runner applied:

- file: `011_create_v6_2_expert_review_validation.sql`
- version: 11
- SHA-256:
  `2f8d1f5d85e2fe0aff6cb55de23c04fc1d3e52102bc15e77e740e0d1b3db6b02`

No table or migration-ledger row was created manually.

After the migration:

- `recommendation.expert_review_case`: present, 0 rows
- `recommendation.expert_plan_review`: present, 0 rows
- `recommendation.expert_action_review`: present, 0 rows
- `recommendation.review`: 0 rows
- expert status: `PENDING_EXPERT_LABELS`

The replacement remained at 27 models, 27 runs, 891 metrics, 15,378 risk
profiles, 15,378 plans, and 27,355 actions.

## Idempotency

A second invocation of the canonical migration runner applied no migrations
and skipped migrations 001 through 011. The migration ledger filename,
version, and SHA-256 remained exact.

## Conclusion

Migration 011 recovery is consistent with both the archive and repository
history. The three tables are structural support for future real expert labels;
they remain empty and do not change the locked
`PENDING_EXPERT_LABELS` status.
