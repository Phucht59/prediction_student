# Database Architecture

PostgreSQL stores final product metadata, normalized results, OULAD risk
profiles, and recommendation plans. Large predictions, checkpoints,
probability arrays, split manifests, and feature snapshots remain files and
are referenced by path, SHA-256, byte count, and row count.

## Active schemas

| Schema | Responsibility | Tables |
|---|---|---:|
| `system` | Immutable migration ledger | 1 |
| `catalog` | Dataset identity, sealed versions, final cohort records | 3 |
| `ml` | Model identity, runs, artifacts, metrics | 4 |
| `recommendation` | Policy, risk profiles, plans, actions, reviews | 5 |

The runtime search path is `catalog, ml, recommendation`. `system` is
available to the migrator only. `public` has no application table. The
`legacy_202607` schema is excluded from all runtime search paths and contains
only archived orphan trigger functions retained for forensic reference.

## Data boundaries

- Canonical model results originate in `artifacts/final/final_results.json`.
- Exactly 27 model–dataset identities and final runs are normalized.
- Overall, per-class, Top-k, calibration, stability, multitask, and
  recommendation metrics share `ml.metric`.
- OULAD prediction rows used by the recommendation system are normalized into
  15,378 risk profiles.
- Exactly 15,378 source plans and 27,355 source actions are preserved.
- No expert row exists while status is `PENDING_EXPERT_LABELS`.
- Future OULAD is `LOCKED_NOT_EXECUTED`.

## Immutability and provenance

Two triggers reject mutation of sealed dataset versions and completed final
runs. Roles, append-only service behavior, per-row checksums, migration
checksums, and artifact checksums provide the remaining controls. Every
migration takes the same advisory lock and uses bounded lock and statement
timeouts.

See [the ERD](DATABASE_ERD.md), [data dictionary](DATABASE_DATA_DICTIONARY.md),
and [operations guide](DATABASE_OPERATIONS.md).
