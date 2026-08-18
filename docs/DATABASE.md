# PostgreSQL database layer

Schemas: `catalog`, `data`, `prediction`, `recommendation`, and `audit`. `.env` must provide `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD`.

```powershell
.venv\Scripts\python.exe -m scripts.database.test_connection
.venv\Scripts\python.exe -m scripts.database.check_schema
.venv\Scripts\python.exe -m scripts.database.import_uci
.venv\Scripts\python.exe -m scripts.database.import_oulad
.venv\Scripts\python.exe -m scripts.database.register_final_model
.venv\Scripts\python.exe scripts/database/leakage_audit.py
```

`src.database.repository` is the persistence API; `src.database.service` is the facade and `src.database.adapters.hybrid` reconstructs the existing `HybridDataView`. File/parquet loaders remain unchanged. OULAD temporal import requires the actual `studentVle.csv` payload, not a Git-LFS pointer.
