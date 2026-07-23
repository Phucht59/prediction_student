# Database Migration Report

## Source

- PostgreSQL 18.4, database name redacted in connection logs.
- Exactly 29 application base tables in `public`.
- Exact total source rows across those tables: 0.
- Backup: custom format, no owner, no ACL, checksum verified.
- Restore test: structural signature PASS.

## Migration

- 10 immutable, checksummed migrations applied.
- 13 final core tables and two ordinary views created.
- 29 empty legacy tables received explicit final destinations and were removed
  only after the explicit empty-legacy gate.
- 10 orphan legacy trigger functions were moved to `legacy_202607`.
- No non-empty table was dropped.
- No model training or canonical result mutation occurred.

## Loaded and reconciled

| Entity | Source | Destination | Rows | Status |
|---|---|---|---:|---|
| Datasets | final release JSON | `catalog.dataset` | 3 | PASS |
| Dataset versions | checksummed OOF contracts | `catalog.dataset_version` | 3 | PASS |
| Cohort records | record-aligned OOF/risk artifacts | `catalog.record` | 16,422 | PASS |
| Model–dataset identities | final results | `ml.model` | 27 | PASS |
| Final runs | final results | `ml.run` | 27 | PASS |
| Artifacts | locked source registries | `ml.artifact` | 81 | PASS |
| Metrics | overall/class/Top-k/multitask/recommendation | `ml.metric` | 891 | PASS |
| Risk profiles | risk profile parquet | `recommendation.risk_profile` | 15,378 | PASS |
| Plans | final plans JSONL | `recommendation.plan` | 15,378 | PASS |
| Actions | embedded plan actions | `recommendation.action` | 27,355 | PASS |
| Expert reviews | no supplied expert labels | `recommendation.review` | 0 | PASS |

Raw numeric reconciliation tolerance was `1e-10`; every canonical metric
passed. Future OULAD remained locked.
