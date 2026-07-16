# PostgreSQL Cleanup Plan

No blind cleanup is authorized.

| Table | Predicate | Rows | Dependencies | Authorized | Reason |
|---|---|---:|---:|---|---|
| ml_experiment_runs | `run_id = '6c9a1d19-a0b4-42cc-8d47-302d797cbe50'` | 1 | 395 | false | denylisted: not explicitly test/temp and/or has lineage dependencies |
| ml_experiment_runs | `run_id = 'd73f48fb-ff0a-4873-bc5f-b864371fad18'` | 1 | 395 | false | denylisted: not explicitly test/temp and/or has lineage dependencies |
| ml_experiment_runs | `run_id = 'a9b0bc7c-38f4-4382-be11-b2fdcc0d9c10'` | 1 | 395 | false | denylisted: not explicitly test/temp and/or has lineage dependencies |
| ml_experiment_runs | `run_id = '3d062f26-abec-44c5-97d6-3a46fe1f952d'` | 1 | 395 | false | denylisted: not explicitly test/temp and/or has lineage dependencies |
