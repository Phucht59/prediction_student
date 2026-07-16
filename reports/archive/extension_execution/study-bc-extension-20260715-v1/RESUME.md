# Resume / verification

The scientific runs are complete. No training resume is pending. To revalidate compact evidence without retraining:

```powershell
py -3.10 scripts/validate_extension_evidence.py --study-b-run study-b-student-por-20260715-v1 --study-c-run study-c-oulad-20260715-v1 --execution-run study-bc-extension-20260715-v1
```

To inspect the end-to-end runner without starting expensive work:

```powershell
py -3.10 scripts/run_extension_end_to_end.py --protocol configs/extension_protocol_v1.yaml --max-wall-clock-hours 6.5 --resume --dry-run
```

PostgreSQL is reachable, but the configured application role lacks migration-owner DDL permission. After supplying an authorized migration-owner connection through the existing environment contract, register lineage with:

```powershell
py -3.10 scripts/apply_extension_migration.py --study-b-run study-b-student-por-20260715-v1 --study-c-run study-c-oulad-20260715-v1 --report reports/extension_execution/study-bc-extension-20260715-v1/database_registration.json
```
