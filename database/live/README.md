# Live PostgreSQL (`student_db`)

This is the running application database on localhost.

## What is stored

| Layer | Tables | Content |
|---|---|---|
| Raw CSVs | `raw.uci_*`, `raw.oulad_*` | Full UCI + OULAD source files, including `studentVle` |
| Catalog | `catalog.student/course/enrollment` | Student–course identities |
| Features | `data.feature_snapshot`, `data.temporal_observation` | Cutoff-safe snapshots already loaded |
| Prediction | `prediction.prediction` | Hybrid C0 OOF probabilities (66,685 rows) |
| Recommendation V3 | `recommendation.recommendation` + `_item` | Five-EBM-C0 ranked actions |
| Recommendation V2 | `recommendation.plan/score` | Historical V2 plans (kept) |

## Commands

```powershell
python project.py db status
python project.py db migrate-raw
python project.py db load-raw
python project.py db load-predictions
python project.py db load-recommendations
python project.py db load-all
```

Connection comes from `.env` (`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`).
Raw CSVs are read from `RAW_DATA_DIR` or `C:\hufit\kltn\data\raw`, then copied into `data/raw/` (gitignored).
