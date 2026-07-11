# PostgreSQL lineage

```mermaid
erDiagram
  source_dataset_versions ||--o{ source_records : contains
  source_records ||--o{ source_record_targets : labels
  source_dataset_versions ||--o{ ml_experiment_runs : used_by
  ml_experiment_runs ||--o{ ml_run_record_splits : records
  source_records ||--o{ ml_run_record_splits : membership
  ml_run_record_splits ||--o{ ml_predictions : permits
  ml_predictions ||--o{ ml_recommendations : explains
  ml_experiment_runs ||--o{ ml_run_metrics : measures
```

`student_predict` is the canonical source architecture. Dataset versions carry
content and ingestion-contract hashes; record, split, prediction, metric and
recommendation linkage supplies lineage. Migration 003 defines separate target
storage keyed by dataset version and record. Source code is complete, but live
migration, 395-row target backfill and credentialed integration tests are complete.
