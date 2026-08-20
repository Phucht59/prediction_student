# Final PostgreSQL database

This directory is the only active database migration line.

- `migrations/`: immutable, checksummed SQL migrations.
- `rollback/`: schema cutback and custom-dump restore procedure.
- `FINAL_SCHEMA_CONTRACT.md`: 13-table contract.
- `LEGACY_TO_FINAL_MAPPING.yaml`: complete mapping for the 29 audited tables.

Live application database (`student_db`) loaders:

```powershell
python project.py db status
python project.py db load-all
```

See `database/live/README.md`.

Historical cutover CLI:

```powershell
python project.py db-final inventory
python project.py db-final backup
python project.py db-final plan
python project.py db-final migrate
python project.py db-final load-results
python project.py db-final validate
python project.py db-final cutover --confirm-production-cutover
python project.py db-final rollback
python project.py db-final status
```

Mutating `migrate`, `load-results`, and `rollback` commands reject database
names that do not contain a disposable marker. Target mutation is available
only through the explicitly confirmed cutover command.
