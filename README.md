# Student performance prediction (CNN-BiLSTM)

This repository contains the frozen technical project for the UCI Student
Performance `student-mat` dataset (395 Portuguese secondary-school students).
The task is three-class prediction from `G3`: Low (`<=9`), Medium (`10-14`),
and High (`>=15`). G1 and G2 are two prior assessments, not a long time series.

## Scientific scope

- `late_stage` uses G1 and G2; `early_warning` excludes G2; `pre_assessment`
  excludes both G1 and G2. These scenarios are not directly comparable because
  they expose different information.
- The frozen CNN-BiLSTM does not beat the simple G2 rule. The nested-CV result
  is 0.8781 +/- 0.0448 Macro-F1; locked-test Macro-F1 is 0.9262 versus 0.9365
  for the G2 rule. HistGradientBoosting has locked-test Macro-F1 0.9463, but
  it was not selected using the locked test.
- Recommendations are a deterministic rule-based advisory policy
  (`student_mat_rule_policy_v3`), not a learned recommender. Expert review is
  still pending.

## PostgreSQL-first data flow

`student-mat.csv` is read only by the explicit ingestion command. All model
selection, training, final evaluation, inference, recommendation and evidence
queries use PostgreSQL:

```text
CSV (one-time ingestion)
  -> source_dataset_versions + source_records + source_record_targets
  -> DB-native loader(dataset_version_id)
  -> split ledger -> Optuna/training/evaluation -> DB predictions/metrics/recommendations
```

The canonical database is `student_predict`; credentials come from environment
variables. Targets are stored separately in `source_record_targets` and are
joined only for evaluation/training, never treated as model features.
Training/evaluation fail fast if migration 003 or target rows are missing; the
final path never falls back to G3 in `source_records.raw_payload`.

Install the audited dependency set with `requirements.txt`; use
`requirements-lock.txt` for the exact Python 3.10 versions verified in the
current workspace.

Apply migrations in order, including
`database/migrations/003_add_source_record_targets.sql`, then ingest:

```powershell
py -3.10 scripts/ingest_dataset_to_postgres.py --dataset student-mat
```

Run frozen evaluation (no CSV path and no Optuna):

```powershell
py -3.10 scripts/run_pipeline.py --dataset student-mat --target-mode 3class `
  --dataset-version-id 1 `
  --selection-config-json artifacts/model_selection/nested-full-20260710/selected_config.json
```

Full model selection uses PostgreSQL and a frozen protocol of 5 outer folds,
3 inner folds, 30 trials per inner search, seed 42:

```powershell
py -3.10 scripts/optimize_model_selection.py --dataset student-mat `
  --dataset-version-id 1 --n-trials 30 --outer-folds 5 --inner-folds 3 `
  --selection-seed 42 --selection-run-id nested-full-20260710
```

`--debug` is smoke-only and is never final evidence.

## Frozen evidence and verification

- Selection config: `artifacts/model_selection/nested-full-20260710/selected_config.json`
- Final scientific run: `a2945d79-9845-4979-b148-159f4853eca3`
- Live PostgreSQL DB-first verification run: `5a0b5041-5216-4a48-9e46-b0c16ab14866`
- `artifacts/final/LATEST_RUN.txt` identifies the active evidence bundle.

Verify checksums, predictions, metrics and DB counts:

```powershell
py -3.10 scripts/verify_final_evidence.py
```

The old report DOCX/PDF and generated report context are intentionally absent.
The thesis report will be written in the next stage from this frozen evidence;
no DOCX is edited by the project-cleanup pipeline.

## Live PostgreSQL status

The database is reachable and currently contains one dataset version and 395
source records. Migration 003 has been written but is not applied: the current
application role lacks `CREATE` privilege on schema `public`. The administrator
procedure is documented in [MANUAL_POSTGRESQL_MIGRATION.md](docs/MANUAL_POSTGRESQL_MIGRATION.md).
Migration 003 is applied on `student_predict`: 395 targets cover all 395 source
records (130/192/73). All five PostgreSQL integration tests pass. The DB-first
run reproduces all 79 predicted classes exactly; maximum probability drift is
`2.78e-08`, while the principal metrics are unchanged. Expert review remains pending.
