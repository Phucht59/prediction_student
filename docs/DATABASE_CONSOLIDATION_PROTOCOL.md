# Final Database Consolidation Protocol

Protocol ID: `final_database_consolidation_v1`

Decision time: 2026-07-23, before any database mutation.

Starting repository SHA:
`fd605e12289ad5e6cbfa4fa38ecd27f353938c34`.

Read-only audit commit:
`8c244e60b1f8c38d1ab04cb650ea9d4c24bf7f7f`.

## Locked sources

| Source | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/final/final_results.json` | 586203 | `07afefd254ad8950746c92cc064f3c6f0c8a2273cd2e14496d70ac826f49a5ba` |
| `artifacts/final/final_results.csv` | 8140 | `682c99357fae996624bdca65e2ca999559c4b859a8f9cd3b96d48f0b68be209a` |
| `artifacts/final/model_registry.json` | 2713 | `49ebd0f168ab42f96d4221613cc66796ff7eb1feaef7a101703e51497704788c` |
| `artifacts/final/checksum_manifest.json` | 2464 | `cf047d1e6ebcc5b5b51cc9b148bb71d64aeca176bd0fae0e3ae0bd7036be9139` |
| `artifacts/v6/prediction/risk_profiles.parquet` | 3523965 | `a0178477871e16b81eebc4ec50dd23567fa4df6ec5b9d75d9e75d14f7ebe5625` |
| `artifacts/v6/recommendation/plans.jsonl` | 22325477 | `d34e61d0fbbaaa9a8db7299dba174caeb2bb92308bf99981788a05fb5ba06cc3` |
| `artifacts/v6/registry/policy_registry.json` | 664 | `f7bb538881fb6d10fdba84244d4f6b295bee6cf4def913e139b2adb9d13b8532` |

Any mismatch is `STOP_MIGRATION_CONFLICT`. The loader never selects a
different value from a legacy table when a canonical value exists.

## Sequence and gates

1. Audit the source in a read-only transaction.
2. Commit audit and this protocol before mutation.
3. Create a custom-format, no-owner, no-ACL backup.
4. Restore the backup to `student_predict_restore_test`.
5. Restore again to `student_predict_final_dev`.
6. Apply immutable checksummed migrations on `final_dev`.
7. Load only the locked canonical artifacts.
8. Validate structure, values, checksums, permissions, expert pending state,
   and Future OULAD lock.
9. Exercise rollback on a separate disposable database.
10. Run repository tests.
11. Cut over the target only with explicit production confirmation and the
    validated backup manifest.
12. Roll back the transaction on any failure.

No target mutation is allowed before gates 3–10 pass.

## Legacy disposition

All 29 audited legacy tables have exact row count zero. They still receive an
explicit destination in `database/final/LEGACY_TO_FINAL_MAPPING.yaml`. They
are not removed merely because they are empty: removal additionally requires
successful backup and restore, absence of runtime references after refactor,
no dependency, a validated disposition, and
`--confirm-drop-empty-legacy`.

## Cutover and rollback

The cutover uses an advisory lock, bounded lock/statement timeouts, and
transactional DDL. Non-empty legacy tables are never dropped. Rollback is
validated both by transaction failure injection and by restoring the custom
dump into a disposable database.

Credentials are provided only through process environment variables and are
never printed or persisted.
