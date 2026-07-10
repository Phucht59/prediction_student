# Final system architecture

```mermaid
flowchart LR
  CSV[CSV: ingestion boundary only] --> DV[PostgreSQL dataset version]
  DV --> SR[source_records]
  DV --> ST[source_record_targets: migration 003]
  SR --> DL[DB-native loader]
  ST --> DL
  DL --> PP[fold-local preprocessing]
  PP --> NS[nested Optuna selection]
  NS --> FC[frozen selected config]
  FC --> FM[single-seed CNN-BiLSTM]
  FM --> P[predictions and metrics]
  P --> R[rule-based advisory recommendation]
  P --> DB[PostgreSQL run ledger]
  R --> DB
  DB --> E[frozen evidence bundle]
```

CSV is only an ingestion source. Final/model-selection code uses the PostgreSQL
loader by dataset version and source-record lineage. Migration 003 implements
separate target storage but remains live-manual-pending; the diagram describes
the final source architecture, not a claim that live migration is complete.
