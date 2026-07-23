# Final Database Schema Contract

Contract ID: `final_database_v1`

The active application database contains exactly four schemas, 13 core base
tables, and two ordinary views. Large predictions, probability matrices,
checkpoints, split members, feature snapshots, and bootstrap samples remain
file artifacts; PostgreSQL stores their path, checksum, size, row count, and
metadata.

## Core tables

| Schema | Table | Natural identity | Purpose |
|---|---|---|---|
| system | schema_migration | filename | Immutable migration ledger |
| catalog | dataset | slug | Three canonical datasets |
| catalog | dataset_version | dataset + version label | Sealed input contract |
| catalog | record | dataset version + source record ID | Final cohort record |
| ml | model | dataset + model key | 27 model–dataset identities |
| ml | run | model + dataset version + result scope | Final/evaluation run |
| ml | artifact | run + kind + path | One checksummed artifact registry |
| ml | metric | run + metric dimensions | Overall, class, top-k and calibration metrics |
| recommendation | policy | policy name + version | Versioned recommendation rules |
| recommendation | risk_profile | run + record | OULAD prediction/risk profile |
| recommendation | plan | risk profile + revision | Final recommendation plan |
| recommendation | action | plan + action code + week + priority | Ordered interventions |
| recommendation | review | plan/action + review identity | Advisor, expert, follow-up, validation review |

## Required counts and state

- `catalog.dataset`: 3.
- `ml.model`: 27, exactly nine models per dataset.
- `recommendation.risk_profile`: 15,378.
- `recommendation.plan`: 15,378.
- `recommendation.action`: exact count from the locked plan artifact.
- Expert evidence: no fabricated row; status remains
  `PENDING_EXPERT_LABELS` in validation metadata.
- Future OULAD: no row and locked.

## Views

- `ml.final_model_results`
- `recommendation.plan_summary`

## Immutability

Only two table triggers are permitted:

1. A sealed dataset version cannot be updated or deleted.
2. A completed final run cannot be updated or deleted.

All other immutability is enforced by role permissions, append-only service
contracts, source checksums, and validation.

## Active search path and legacy

The runtime search path is `catalog, ml, recommendation`. `public` contains no
application table after cutover. Any non-empty legacy table is moved to
`legacy_202607` and made read-only. An empty legacy table is removable only
after backup/restore validation and the explicit
`--confirm-drop-empty-legacy` gate.

## Prohibited behavior

- No model training, prediction changes, metric changes, or recommendation changes.
- No Future OULAD load.
- No fake expert labels or reviews.
- No `DROP DATABASE`, destructive `TRUNCATE`, or unguarded production cascade.
- No credential in code, migration, log, report, or committed artifact.
